# Task02：FusedMoE——动态稀疏计算为何反而适合规则化

代码：[reference.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task02_fused_moe/reference.py) · [solution.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task02_fused_moe/solution.py) · [benchmark.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task02_fused_moe/benchmark.py)

核心 kernel：`_gate_up_kernel`、`_down_kernel`、`_route_kernel`。

## 1. 本题训练什么

本题把动态 MoE 路由转换成三个规则化 Triton 阶段，是 10 题中加速比最高的案例。核心能力：

- 识别 Python 循环、布尔索引和小 GEMM 的组合开销；
- 在“少算”与“规则地多算一点”之间做性能权衡；
- 使用 `tl.dot` 实现两个小型投影；
- 融合 SiLU、逐元素乘法和路由加权；
- 保持参数名称和 `state_dict` 完全兼容。

## 2. 数学结构

固定形状：

```text
T=83, E=8, top_k=2, H=128, I=64
w1: [E, 2I, H] = [8,128,128]
w2: [E, H, I]  = [8,128,64]
```

专家 `e` 的计算：

```text
[gate, up] = x @ w1[e]^T
act = SiLU(gate) ⊙ up
y_e = act @ w2[e]^T
```

路由为每个 token 选择两个专家 `e0,e1`：

```text
out = route_weight0 * y_e0 + route_weight1 * y_e1
```

## 3. 参考实现的瓶颈

参考实现逐专家循环：

```python
for e in range(num_experts):
    mask = flat_ids == e
    if not mask.any():
        continue
    x_e = x_rep[mask]
    ...
```

这会产生：

- 每个专家一次比较和 `mask.any()` 同步/判断；
- 动态 gather/scatter；
- 每个专家规模不同的极小 GEMM；
- `w1.to(dtype)`、`w2.to(dtype)` 的完整转换；
- 重复复制 token 到 `x_rep`；
- 最后再乘路由权重并归约。

算术上只计算选中的两个专家，但执行形状高度不规则，GPU/DCU 很难高效利用。

## 4. 关键决策：为全部 8 个专家计算

优化版对每个 token 计算全部 8 个专家，然后最后只加载 Top-2 输出。计算量从理论的 2 个专家扩大到 8 个，但得到完全规则的矩阵 tile：

```text
gate/up grid = (ceil(83/32), 8)
down grid    = (ceil(83/32), 8, 2)
```

为什么反而快：

- 没有 Python 专家循环；
- 没有动态 mask/gather/scatter；
- 每个 program 形状相同；
- `tl.dot` 可以走规则矩阵计算路径；
- 总尺寸很小，额外算术远低于动态调度成本。

这是重要经验：硬件时间不是 FLOP 数的简单函数。规则、高并行的多算，有时比稀疏但碎片化的少算更快。

## 5. 第一阶段：gate/up 与 SiLU 融合

一个 program 处理 32 个 token、一个 expert 和 64 个中间通道：

```python
m = block_m * 32 + tl.arange(0, 32)
n = tl.arange(0, 64)
k = tl.arange(0, 128)
```

加载：

```text
x tile      : [32,128]
gate weight : [128,64]
up weight   : [128,64]
```

计算：

```python
gate = tl.dot(x, gate_w)
up = tl.dot(x, up_w)
act = gate * tl.sigmoid(gate) * up
```

`gate * sigmoid(gate)` 就是 SiLU。激活不写出 gate/up 两个中间矩阵，只写融合后的 `act[E,T,I]`。

权重参数保持 FP32，与参考模型一致；kernel 在 tile 内 `.to(tl.float16)`，避免每次前向创建完整 FP16 权重副本。

## 6. 第二阶段：down projection

`act[E,T,I] @ w2[e]^T` 输出 `[E,T,H]`。H=128 被拆成两个 64 列 tile：

```text
grid = (ceil(83/32), 8 experts, 2 hidden tiles)
```

累加器由 `tl.dot` 管理，输入 FP16，输出临时 `expert_out` 为 FP16。这里没有把 down、路由强行融合，因为同时保存多个专家的两层 GEMM 状态会显著增加 program 复杂度和寄存器压力。

## 7. 第三阶段：Top-2 路由和加权

Softmax 后再对 Top-2 归一化时，完整 8 路分母同样会抵消：

```text
w0 = exp(l0) / (exp(l0)+exp(l1))
w1 = exp(l1) / (exp(l0)+exp(l1))
```

为稳定和少算，代码使用：

```python
ratio = tl.exp(l1 - l0)
w0 = 1 / (1 + ratio)
w1 = ratio / (1 + ratio)
```

一个 program 负责一个 token，同时加载两个 `[128]` 专家输出并加权写回。

## 8. 为什么配置是 32×64×128

- `M=32`：83 个 token 需要 3 个 tile，尾块浪费可控；
- `N=64`：正好覆盖 intermediate，down 的 H 只需 2 块；
- `K=128/64`：固定维度一次覆盖，无需 K 循环；
- `num_warps=4`：中型 `tl.dot` tile 需要足够并行度，实测优于更小或更大配置。

块大小不是只看整除。更大 M 会增大累加器和寄存器压力；更小 M 会增加 program 数和重复权重加载。

## 9. 正确性与参数兼容

benchmark 使用相同随机种子分别构造两个模型，再执行：

```python
optimized.load_state_dict(reference.state_dict())
```

这验证 `w1/w2` 名称、形状和初始化契约一致。若忘记同步参数，比较的是两个不同网络，结果没有意义。

FP32 权重转 FP16 参与 dot 会带来误差，但官方 `1e-2` 容差内通过。需要同时检查最大绝对误差，避免平均误差掩盖局部异常。

## 10. 结果和收益来源

```text
PyTorch: 2.602321 ms
Triton : 0.145260 ms
Speedup: 17.915x
```

最大收益不是某一条指令，而是把动态专家分发重构为规则的三阶段流水线。

## 11. 何时这种策略会失效

- 专家数从 8 增长到数百，计算全部专家不可接受；
- intermediate/hidden 很大，额外 FLOP 主导时间；
- token 很多且路由分布足够均匀，排序后的 grouped GEMM 更合适；
- 显存不足以保存 `[E,T,I]` 和 `[E,T,H]`。

通用 MoE 通常需要 token 排序、专家计数、prefix sum 和 grouped GEMM。本题策略是固定小 E、小 H、小 T 下的专用最优解。

## 12. 迁移练习

1. 比较“只算 Top-2”与“算全部专家”在不同 T/E 下的交叉点；
2. 把 gate/up 两次 `tl.dot` 改为一次输出 128 列的 dot，再切分，比较寄存器和性能；
3. 扫描 `M∈{16,32,64}`、`num_warps∈{2,4,8}`；
4. 设计大专家数版本：先按专家排序 token，再执行 grouped GEMM。
