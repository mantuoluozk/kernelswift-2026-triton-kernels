# Task08：hc_split_sinkhorn——让 4×4 矩阵在寄存器里迭代 20 轮

> 本章以海光 BW1000 的实现和实测数据为主线。其他芯片的参数、限制与结果统一放在对应平台迁移章节中。

代码：[reference.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task08_hc_split_sinkhorn/reference.py) · [solution.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task08_hc_split_sinkhorn/solution.py) · [benchmark.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task08_hc_split_sinkhorn/benchmark.py)

核心 kernel：`_hc_split_sinkhorn_kernel`。

## 1. 本题训练什么

这是最典型的小矩阵融合题。数据只有 16 行、每行一个 4×4 矩阵，但参考实现进行 20 轮行列归一化；如果每次 `sum/div` 都形成独立 kernel，调度成本远大于 16 个元素的算术。

你将学会：

- 一个 program 负责一个独立小问题；
- 使用二维块张量表达 4×4 矩阵；
- 在静态循环中保持状态片上驻留；
- 处理数值稳定的 exp 和交替归一化；
- 一个 kernel 写多个输出。

## 2. 输入拆分

```text
mixes: [B,S,24] = [2,8,24]
hc_scale: [3]
hc_base: [24]
```

每行 24 个数拆为：

```text
0:4   → pre gate
4:8   → post gate
8:24  → 4×4 comb matrix
```

输出：

```text
pre:  [2,8,4]
post: [2,8,4]
comb: [2,8,4,4]
```

## 3. Sinkhorn 在做什么

目标是把正矩阵交替做行归一化和列归一化，使其逐步接近双随机矩阵：每行和、每列和都接近 1。

初始化先做稳定指数化：

```text
A = raw*scale + base
A = exp(A - row_max)
```

减去每行最大值不改变归一化后的比例，但避免 `exp` 溢出。

随后：

```text
A = A / row_sum + eps
A = A / (col_sum + eps)
```

再执行 19 次行归一化+列归一化，共 20 轮。

## 4. 参考实现为什么慢

对 `[16,4,4]` 这样的极小张量，每一轮至少包含两次 sum 和两次除法，框架可能产生数十个小 kernel，并在轮次之间反复读写同一个 4×4 矩阵。

总算术量很小，主要耗时是 launch 和片外往返。

## 5. program 映射与二维索引

grid 为 `(B*S,)=(16,)`，每个 program 负责一行：

```python
row_id = tl.program_id(0)
hc = tl.arange(0,4)
matrix_offsets = hc[:,None]*4 + hc[None,:]
```

`matrix_offsets` 是：

```text
[[ 0, 1, 2, 3],
 [ 4, 5, 6, 7],
 [ 8, 9,10,11],
 [12,13,14,15]]
```

一次 load 得到 `[4,4]` 块张量。

## 6. 行归约与列归约

```python
row_max = tl.max(comb, axis=1)     # [4]
row_sum = tl.sum(comb, axis=1)     # [4]
comb = comb / row_sum[:,None]

col_sum = tl.sum(comb, axis=0)     # [4]
comb = comb / col_sum[None,:]
```

理解 `[:,None]` 与 `[None,:]` 是关键：前者把每个行统计量广播到该行，后者把每个列统计量广播到该列。

## 7. 为什么 20 轮可以留在片上

`comb` 只有 16 个 FP32 元素，row/col sum 各 4 个。`sinkhorn_iters` 是 `tl.constexpr`，循环可在编译期展开/优化：

```python
for _ in range(sinkhorn_iters - 1):
    ...
```

循环期间没有 `tl.store`，矩阵状态不会每轮写回全局内存。最后一次性写 `pre/post/comb`。

这正是寄存器驻留的理想场景：状态非常小、重复使用次数多。若矩阵是 128×128，完整驻留会造成严重寄存器压力，必须分块或多 kernel。

## 8. pre/post 也一起融合

```text
pre  = sigmoid(mix_pre*s0 + base_pre) + eps
post = 2*sigmoid(mix_post*s1 + base_post)
```

它们与 comb 共享同一行输入、scale 和 base，因此放进同一 kernel，避免额外两个小 kernel。注意 pre 有 `+eps`，post 没有；这些看似微小的语义不能丢。

## 9. 为什么 `num_warps=1`

每个 program 只有 24 个输入和一个 4×4 状态，块内并行量极小。增加 warps 只会增加资源和调度成本。大量独立行由 grid 提供并行度，program 内无需堆更多资源。

## 10. 正确性与数值稳定

- exp 前减 row max；
- 每次分母加 eps，顺序与 reference 一致；
- 20 轮不能少一轮或多一轮；
- reference 首轮在 row normalize 后先 `+eps` 再 column normalize，这个括号顺序必须保留；
- 输入在 benchmark 中从 CPU 搬到设备，solution 中 `.contiguous()` 保证布局。

记录最大绝对误差约 `1.19e-7`，说明融合后与参考非常接近。

## 11. 结果

```text
PyTorch: 1.399493 ms
Triton : 0.103106 ms
Speedup: 13.573x
```

高加速比几乎完全来自把数十个小 kernel 合成一个。

## 12. 迁移练习

1. 打印每轮行和、列和，观察收敛；
2. 扫描 sinkhorn_iters，观察 reference 时间近似线性增长而融合版增长方式不同；
3. 扩展到 8×8，测试寄存器压力和 num_warps；
4. 把 eps 放在不同位置，观察语义与误差变化；
5. 设计大矩阵分块 Sinkhorn，思考跨 program 归约为何需要多阶段。
