# Task07：mhc_post——把固定 4×4 混合展开成数据复用

代码：[reference.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task07_mhc_post/reference.py) · [solution.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task07_mhc_post/solution.py) · [benchmark.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task07_mhc_post/benchmark.py)

核心 kernel：`_mhc_post_kernel`。

## 1. 本题训练什么

本题展示固定小矩阵不一定适合调用通用 einsum/GEMM。4×4 系数极小，真正大的维度是 hidden=1280；把 4 路 residual 读取一次，在寄存器中展开四路输出，可以大幅减少中间张量和框架开销。

## 2. 形状与公式

```text
x:              [2,4096,1280] BF16
residual:       [2,4096,4,1280] BF16
post_layer_mix: [2,4096,4,1] FP32
comb_res_mix:   [2,4096,4,4] FP32
output:         [2,4096,4,1280] BF16
```

合并 `(batch,token)` 为 row，共 8192 行。对每个 hidden 位置 h：

```text
out[n,h] = x[h] * post[n] + Σ_{m=0..3} residual[m,h] * comb[m,n]
```

这是同一个 4×4 混合矩阵作用在每个 hidden 位置上。

## 3. 参考实现的代价

```python
term2 = einsum(..., residual.float())
return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()
```

问题包括：

- 大型 BF16 residual 转 FP32；
- einsum 生成 `[2,4096,4,1280]` FP32 中间量；
- x 转 FP32并广播乘法，再生成中间结果；
- 最终相加和 BF16 转换；
- 对 4×4 这种极小 K，通用 einsum 调度成本占比高。

## 4. program 映射

```python
row = tl.program_id(0)
hidden_offsets = tl.program_id(1)*BLOCK + tl.arange(0,BLOCK)
```

实际 grid 是 `(8192,1)`，`BLOCK=2048`，mask 覆盖 hidden<1280。也就是一个 program 处理一整行 hidden。

为什么 BLOCK 取 2048：Triton 块通常使用 2 的幂；1024 需要两个 program，重复加载 4×4 系数和 post；2048 一次覆盖整行，尾部 768 个位置被 mask。实测一次覆盖更快。

## 5. 数据只加载一次

每个有效 hidden 位置加载：

- 一个 `x`；
- 四个 residual `r0...r3`。

每个 row 只加载：

- 四个 post 标量；
- 十六个 comb 标量。

然后展开：

```python
out0 = x*p0 + r0*c00 + r1*c10 + r2*c20 + r3*c30
out1 = x*p1 + r0*c01 + r1*c11 + r2*c21 + r3*c31
...
```

四个 residual 向量被四路输出复用。如果为每个输出启动独立 kernel，residual 会被重复读取四次。

## 6. 为什么手工展开合理

4 是固定且很小的编译期常量。手工展开：

- 消除循环计数和动态索引；
- 让编译器看见所有复用关系；
- 系数保持标量/寄存器状态；
- 避免通用矩阵乘的打包和调度。

若 hc_mult 变成 16/32，手工展开会导致代码和寄存器爆炸，此时应改用 tiled dot 或循环。

## 7. BF16 与 FP32

参考显式 `.float()` 后计算，最后 `.bfloat16()`。Triton 加载 BF16 后参与与 FP32 系数的表达式，累加路径获得更高精度，store 时转 BF16。

加法顺序从通用 einsum 的归约树变为固定 `r0+r1+r2+r3`，因此结果可能有 BF16 舍入差异。记录的最大绝对误差约 `3.125e-2`，但在官方 `atol=rtol=1e-2` 的组合判据下通过。不要只比较最大绝对误差与 atol；`assert_close` 使用 `|a-b| <= atol + rtol*|b|`。

## 8. 访存布局

`residual[row,4,hidden]` 的 hidden 是连续维，四路 residual 的起始地址相差 `hidden_size`。同一 `tl.arange` 向量在每一路都访问连续地址，符合合并访存原则。

输出布局相同，四次 store 分别写四个连续 hidden 区间。

## 9. 结果

```text
PyTorch: 2.569192 ms
Triton : 0.223235 ms
Speedup: 11.509x
```

收益主要来自删除大型 FP32 中间张量、复用 residual 和固定小矩阵展开。

## 10. 迁移练习

1. 比较 `BLOCK=1024/2048`，解释重复系数加载与 mask 浪费；
2. 写 `hc_mult` 可配置的 `tl.static_range` 版本；
3. 为 hc_mult=8 比较手工展开、循环和 `tl.dot`；
4. 用 FP32 输出检查纯累加顺序误差，再加入 BF16 store；
5. 估算 reference 中间张量的字节数与优化版最小读写量。
