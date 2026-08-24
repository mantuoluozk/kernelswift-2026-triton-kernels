# Task01：GroupedTopk——从“照着算”到“证明后少算”

> 本章以海光 BW1000 的实现和实测数据为主线。其他芯片的参数、限制与结果统一放在对应平台迁移章节中。

代码：[reference.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task01_grouped_topk/reference.py) · [solution.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task01_grouped_topk/solution.py) · [benchmark.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task01_grouped_topk/benchmark.py)

核心 kernel：`_grouped_topk_kernel`。

## 1. 本题训练什么

这是理解“数学变换往往比低层微调更重要”的最佳入门题。你将学会：

- 一行对应一个 Triton program；
- 使用 `tl.max`、`tl.argmax` 实现小规模 Top-k；
- 利用 Softmax 单调性跳过无关计算；
- 利用二次归一化消去公共分母；
- 同时处理浮点权重和整数索引输出。

## 2. 输入、输出与语义

固定配置：

```text
tokens T = 83
experts E = 256
groups G = 8
experts_per_group = 32
topk_group = 4
topk = 8
```

真正参与计算的是 `gating_output[T,E]`。`hidden_states` 只用于检查 token 数一致，是接口契约的一部分，但不需要被 kernel 读取。

对每个 token：

1. 对 256 个 logits 做 Softmax；
2. 每组取 32 个专家中的最大 score；
3. 选择得分最高的 4 组；
4. 屏蔽其他组，从候选专家中选 Top-8；
5. 对 8 个权重重新归一化；
6. 输出 FP32 权重 `[T,8]` 和 INT32 专家编号 `[T,8]`。

## 3. 参考实现为什么慢

参考路径会依次产生：

- `[83,256]` Softmax 输出；
- `[83,8]` group score；
- `[83,4]` group index；
- `[83,8]` group mask；
- 扩展后的 `[83,256]` score mask；
- masked score；
- Top-k 权重与索引；
- 归一化结果。

每一步本身都不大，但会触发多次框架调度和多次显存往返，因此主要问题是 launch 与中间张量，而不是 256 个数的算术量。

## 4. 关键证明一：选择可以直接在 logits 上进行

对同一行 logits，Softmax 为：

```text
s_i = exp(l_i) / Σ_j exp(l_j)
```

`exp` 严格单调，分母对同一行所有专家相同，因此：

```text
l_a > l_b  ⇔  s_a > s_b
```

所以组内最大专家、Top-4 组和最终 Top-8 专家索引都可以直接在 logits 上求，不需要先计算 256 路 Softmax。

## 5. 关键证明二：完整 Softmax 分母会抵消

选中的 8 个专家再次归一化：

```text
w_i = s_i / Σ_selected s_j
    = (exp(l_i)/Z) / Σ_selected(exp(l_j)/Z)
    = exp(l_i) / Σ_selected exp(l_j)
```

因此只对选中的 8 个 logits 做稳定归一化即可。

代码选择第一个最大值 `v0` 作为平移量：

```python
e0 = tl.exp(v0 - v0)
e1 = tl.exp(v1 - v0)
...
denom = e0 + ... + e7
```

减去最大值避免指数溢出，`e0` 精确为 1。

## 6. program 映射

```python
row = tl.program_id(0)
expert = tl.arange(0, 256)
logits = tl.load(gating_ptr + row * 256 + expert)
```

grid 是 `(83,)`，每个 program 负责一个 token 的整行 256 个专家。这样组归约、Top-k 和最终归一化都能在同一 program 的片上状态中完成。

256 是 2 的幂且形状固定，不需要尾部 mask。如果做通用版本，应该使用 `BLOCK=next_power_of_2(E)` 并把越界 logits 设为 `-inf`。

## 7. 如何得到组分数

```python
group_logits = tl.reshape(logits, (8, 32))
group_scores = tl.max(group_logits, axis=1)
```

逻辑上把一维专家向量看成 `[group, expert_in_group]`，沿组内维度取 max，得到 8 个组分数。

随后通过重复“argmax → 把已选位置改成 `-inf`”选出 4 个组。因为 `G=8` 很小且固定，手工展开比实现通用排序网络更简单，编译器也能消除循环控制。

## 8. 如何得到专家 Top-8

先计算每个专家所属组：

```python
expert_group = expert // 32
selected_group = ((expert_group == g0) | ... | (expert_group == g3))
candidates = tl.where(selected_group, logits, -float("inf"))
```

再执行 8 轮 argmax。每轮把已选专家置为 `-inf`，防止重复。这里需要同时保存索引 `i0...i7` 和分值 `v0...v7`。

## 9. 为什么 `num_warps=1`

一行只有 256 个 FP32 元素，计算和归约状态不大。增加 program 内并行资源带来的调度/资源成本超过收益，实测 1 最快。这不是说归约永远用 1，而是当前行宽、grid 数和 BW1000 编译结果的平衡。

## 10. 正确性风险

### 10.1 Top-k 次序

权重正确但 ID 次序不同仍可能失败。benchmark 对 ID 使用 `torch.equal`，并在失败时打印集合差异与排列，区分“选错专家”和“只是不同行内排序”。

### 10.2 相同 logits

完全相同的分值涉及 tie-breaking。当前随机输入几乎不会产生精确相等值；通用库需要确认 `torch.topk` 与 `tl.argmax` 的同值规则是否一致。

### 10.3 配置专用化

`ModelNew.__init__` 明确断言 `(topk, G, topk_group)=(8,8,4)`。如果把形状常量写死却不检查，调用方会得到静默错误；明确拒绝未支持配置是正确做法。

## 11. 结果和收益来源

```text
PyTorch: 0.275516 ms
Triton : 0.088945 ms
Speedup: 3.098x
```

收益主要来自：跳过完整 Softmax、删除 mask 中间张量、把多个小操作融合为一次 kernel。

## 12. 迁移练习

1. 把 `E=256` 改为运行时参数，使用 mask 支持非 2 次幂；
2. 用 `tl.static_range` 写可配置的 Top-k，并比较与手工展开的性能；
3. 支持 sigmoid scoring。注意 sigmoid 也单调，但最终 renormalize 的公共因子推导与 Softmax 不同；
4. 构造包含相同 logits 的输入，研究 tie-breaking。
