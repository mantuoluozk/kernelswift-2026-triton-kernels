# Task04：SPLADESparsePooler——不要生成最终并不需要的大张量

代码：[reference.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task04_splade_sparse_pooler/reference.py) · [solution.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task04_splade_sparse_pooler/solution.py) · [benchmark.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task04_splade_sparse_pooler/benchmark.py)

## 1. 本题训练什么

本题是“跨算子看最终需求”的代表。参考模型先生成每个 token 的 30522 维词表 logits，做激活后才按 4 段 max pooling；但最终只需要每段的最大值。优化版让 decoder GEMM 的输出直接进入分段 max，不把 `[83,30522]` 写回显存。

## 2. 计算图与形状

```text
hidden: [83,768]
dense: [83,768] -> [83,768]
GELU: [83,768]
LayerNorm: [83,768]
decoder: [83,768] @ [768,30522] -> [83,30522]
activation: log(1 + max(logits,0))
segments: lengths [20,25,18,20]
output: 4 × [30522]
```

参考公式：

```text
h = LN(GELU(dense(x)))
z[t,v] = h[t,:] · W_decoder[v,:] + b[v]
f(z) = log(1 + ReLU(z))
out[s,v] = max_{t in segment s} f(z[t,v])
```

## 3. 第一层优化：GELU + LayerNorm

`dense` 保留为 PyTorch Linear，因为它已经是规则 GEMM。其输出进入 `_gelu_layer_norm_kernel`：

```python
row = tl.program_id(0)
columns = tl.arange(0, 1024)
mask = columns < 768
```

一个 program 处理一行 768 hidden，块向上补到 1024。

### 3.1 精确 GELU

参考 `nn.GELU()` 默认使用精确 erf 形式：

```text
GELU(x) = 0.5*x*(1 + erf(x/sqrt(2)))
```

不能未经验证替换成 tanh 近似。代码中的 `0.707106...` 是 `1/sqrt(2)`。

### 3.2 LayerNorm

```text
mean = Σx/H
variance = Σ(x-mean)^2/H
y = (x-mean)/sqrt(variance+eps) * weight + bias
```

GELU 输出保留在块状态中，直接参与 mean/variance 归约，不写中间张量。越界列要用 `tl.where(mask, ..., 0)` 排除，否则 1024-768 个填充值会污染均值和方差。

## 4. 第二层优化：利用激活函数单调性

`f(x)=log(1+ReLU(x))` 是单调不减函数，因此：

```text
max_t f(z_t) = f(max_t z_t)
```

于是对每个 segment/vocab，只需先求 logits 最大值，再执行一次 ReLU+log1p。原参考实现对 83×30522 个元素做激活，优化后只对 4×30522 个最大值做激活。

这里必须强调：该变换只对 max/min 与单调函数成立。若 pooling 是 sum，`Σf(z)` 不能改成 `f(Σz)`，所以 `ModelNew` 明确只支持 `pooling="max"`。

## 5. 第三层优化：decoder 与池化融合

核心 kernel `_decoder_splade_max_kernel` 的 grid：

```text
(4 segments, ceil(30522 / BLOCK_N))
BLOCK_M=32, BLOCK_N=64, BLOCK_K=64
```

一个 program 负责一个 segment 的最多 32 个 token，以及 64 个 vocab 输出。

### 5.1 K 维分块

768 hidden 被拆成 12 个 64 维 tile：

```python
accumulator = tl.zeros((32,64), tl.float32)
for k_start in tl.static_range(0, 768, 64):
    hidden_tile = ...  # [32,64]
    weight_tile = ...  # [64,64]
    accumulator += tl.dot(hidden_tile, weight_tile)
```

累加器保存 32×64 个 FP32 值。完成 decoder dot 和 bias 后，沿 32 token 维直接 `tl.max(axis=0)`，只写 64 个 pooled 值。

### 5.2 为什么按 segment 单独算 GEMM

另一个实验 kernel `_decoder_splade_all_segments_kernel` 一次计算 83 token，再分别做四段 max，理论上能复用 decoder 权重。但它需要 `[128,64]` FP32 accumulator，寄存器压力更大。最终 `direct_all_segments=False`，说明在 BW1000 当前形状下，较小 accumulator 的并发收益超过权重复用收益。

这正是“必须实测”的例子：减少重复加载不一定等于更快。

## 6. 备用 materialized 路径的意义

代码保留 `_splade_max_pool_kernel` 和 `--materialized`：先用 PyTorch decoder 生成 logits，再只融合激活与池化。这是一条很有价值的中间基线：

- 若它比 reference 快，说明激活+池化融合有效；
- 若 direct decoder+pool 更快，额外收益来自避免 logits 落盘；
- 若 direct 路径更慢，可能是自写 GEMM tile 不如库实现或寄存器压力过高。

优化应该逐层建立基线，而不是一次改完后猜收益来源。

## 7. 参数为什么是 64/64/4

- `BLOCK_N=64`：控制 vocab 并行和 accumulator 宽度；
- `BLOCK_K=64`：匹配 hidden 的整除分块和 `tl.dot`；
- `BLOCK_M=32`：覆盖最长 25 token segment，同时保持规则 tile；
- `num_warps=4`：中型矩阵 tile 的实测最优点。

`VOCAB_SIZE=30522` 不是 64 的整数倍，最后一块必须对权重加载和输出 store 使用 `vocab<VOCAB_SIZE` mask。

## 8. 正确性风险

- `dense/layer_norm/decoder` 的参数名称和形状必须与 reference 一致；
- GELU 必须使用精确公式；
- LayerNorm epsilon 是 `1e-12`，不能默认写成 `1e-5`；
- 四个 segment 的 start/length 固定为 `(0,20),(20,25),(45,18),(63,20)`；
- FP16 dot + FP32 accumulator 会改变最低位，必须逐个输出比较。

## 9. 结果与收益

```text
PyTorch: 0.903172 ms
Triton : 0.308420 ms
Speedup: 2.928x
max_abs_diff ≈ 5.4e-4
```

收益来自三层：GELU+LayerNorm 融合、激活移到 max 后、decoder+pool 融合避免巨大 logits。

### Ascend 910B 的融合边界

910B 上的 direct decoder+pool kernel 会触发 507014 AICore watchdog 超时。因此 910B 版本保留 GELU+LayerNorm Triton kernel，让 torch_npu 执行 decoder GEMM，再用 `BLOCK_N=2048` 的 Triton kernel 融合四段 max 与最终激活。正式结果为 `0.721766 → 0.659824 ms`，加速 `1.094×`。

这说明融合决策必须受单 kernel 执行时间和后端代码生成质量约束；在一个平台上有效的全融合，不一定能原样迁移到另一个平台。

## 10. 迁移练习

1. 用 profiler 比较 materialized 与 direct 两条路径的 kernel 数和显存流量；
2. 扫描 `BLOCK_N/K`，观察 accumulator 对寄存器的影响；
3. 支持动态 seq_lens，把硬编码 start/length 改为输入 prefix sum；
4. 思考 sum pooling 为什么不能使用同一单调性变换；
5. 若 vocab 更大，尝试让 program 调度顺序提高 decoder weight 的 L2 复用。

## S60 实测补充

S60 的二维 `grid.y=477` 无法启动，因此把 segment 和 vocab block 展平成一维。最终拆分厂商线性层与 Triton 池化，正式为 `0.940885 → 1.576604 ms`（0.597×）。
