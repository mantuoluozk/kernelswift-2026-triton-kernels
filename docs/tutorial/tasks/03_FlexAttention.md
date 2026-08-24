# Task03：FlexAttention——从完整注意力矩阵到在线 Softmax

> 本章以海光 BW1000 的实现和实测数据为主线。其他芯片的参数、限制与结果统一放在对应平台迁移章节中。

代码：[reference.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task03_flex_attention/reference.py) · [solution.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task03_flex_attention/solution.py) · [benchmark.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task03_flex_attention/benchmark.py)

核心 kernel：`_causal_attention_kernel`。

## 1. 本题训练什么

本题实现固定长度的因果多头 Attention，重点是 FlashAttention 类算法的核心：流式扫描 K/V、在线维护 Softmax、避免物化 `QK^T` 和 probability 矩阵。

## 2. 输入与目标公式

```text
Q,K,V: [T, heads, D] = [83,8,64]
output: [T, heads*D] = [83,512]
scale = 1/sqrt(64) = 1/8
```

每个 head：

```text
S = QK^T * scale
S[i,j] = -∞, 当 j > i
P = softmax(S, axis=j)
O = PV
```

因果 mask 表示第 i 个 token 只能看到 `[0,i]`。

## 3. 传统实现的中间状态

若直接实现，需要对每个 head 保存 `[83,83]` score，Softmax 后再保存同形状 probability。虽然本题矩阵不大，但还会发生 transpose/reshape 和多个 kernel 调度。

优化目标是：每个 query tile 只保留当前输出累加器，不保存完整行的 score。

## 4. program 映射

```python
block_m = tl.program_id(0)
head = tl.program_id(1)
m = block_m * 16 + tl.arange(0, 16)
d = tl.arange(0, 64)
```

grid：

```text
(ceil(83/16), 8) = (6,8)
```

每个 program 负责一个 head 的 16 个 query。`q_offsets = m*512 + head*64 + d` 直接读取原始 `[token,head,d]` 连续布局，无需生成 transpose 缓冲区。

## 5. K/V 为什么按 32 token 分块

```python
n = tl.arange(0, 32)
for start_n in range(0, 96, 32):
    cols = start_n + n
```

83 被向上覆盖到 96，共 3 块。无效 `cols>=83` 用 mask 处理。`BLOCK_N=32` 在 K/V 加载、score tile 大小和寄存器压力之间折中。

当前 score tile 是 `[16,32]`，输出 accumulator 是 `[16,64]`。如果把 M/N 都增大，`scores`、`probs` 和 `acc` 会同时变大，可能降低 occupancy。

## 6. 在线 Softmax 推导

假设已经处理若干 K 列：

- `row_max=m_old`；
- `row_sum=l_old=Σ exp(score-m_old)`；
- `acc_old=Σ exp(score-m_old) * V`。

新块 score 的最大值为 `m_blk`：

```text
m_new = max(m_old, m_blk)
alpha = exp(m_old - m_new)
```

旧状态换到新尺度：

```text
l_old'   = alpha * l_old
acc_old' = alpha * acc_old
```

新块概率分子：

```text
p_blk = exp(score_blk - m_new)
```

更新：

```text
l_new   = alpha*l_old + sum(p_blk)
acc_new = alpha*acc_old + p_blk @ V_blk
```

扫描完所有块后：

```text
output = acc / row_sum
```

这与一次性 Softmax 数学等价，并且每一步都减去当前最大值，数值稳定。

## 7. 因果 mask 如何进入公式

```python
valid = (
    (m[:, None] < 83)
    & (cols[None, :] < 83)
    & (cols[None, :] <= m[:, None])
)
scores = tl.where(valid, scores, -float("inf"))
```

`exp(-inf)=0`，无效位置对最大值和指数和都没有贡献。注意 Q 的尾部无效行也被屏蔽；store 时再次使用 `m<83`。

## 8. 两次 `tl.dot`

```python
scores = tl.dot(q, tl.trans(k)) * scale
acc = acc * alpha[:, None] + tl.dot(probs.to(tl.float16), v)
```

Q/K/V 为 FP16；row max、row sum、score 和 accumulator 为 FP32。第二次 dot 把概率转 FP16 以匹配高吞吐矩阵路径，最终误差在官方容差内。

这是性能与精度的明确交换：不能只因为更快就转换，必须用 reference 检验。

## 9. 为什么 `BLOCK_M=16, num_warps=1`

序列只有 83，因果 attention 的前部 query 有大量 mask。较小 M 减少无效计算和 accumulator 规模。实测 `num_warps=1` 最快，说明对当前小 tile，增加 program 内资源没有抵消开销。

对于长序列，常见配置会更大，并可能把因果区域分成完整块与对角 mask 块，以减少分支和无效计算。

## 10. 与 Task06 的区别

| 项目 | Task03 | Task06 |
| --- | --- | --- |
| batch | 1（隐式） | 2 |
| causal | 是 | 否 |
| Q tile | 16 | 64 |
| K/V tile | 32 | 32 |
| num_warps | 1 | 4 |

非因果版本每个 query 都扫描全部 K/V，矩形 tile 更规则，因此 Task06 可以使用更大的 M tile。

## 11. 结果与局限

```text
PyTorch: 0.143880 ms
Triton : 0.086850 ms
Speedup: 1.657x
```

参考 `scaled_dot_product_attention` 本身已经高度优化，所以提升小于那些多小算子案例。这里的价值是掌握在线 Softmax 和布局直读，而不是追求两位数加速。

当前实现固定 T=83、heads=8、D=64。通用版本需要传入 stride、长度、batch、heads 和因果开关，并根据形状 autotune。

## 12. 迁移练习

1. 手算两个 K block 的在线 Softmax，验证合并公式；
2. 改成非因果模式，并与 Task06 的 tile 比较；
3. 跳过完全位于因果边界右侧的 K block；
4. 支持不同 `num_kv_heads`，理解 GQA 中 K/V head 映射；
5. 记录 BLOCK_M/N 对 VGPR、occupancy 和耗时的影响。
