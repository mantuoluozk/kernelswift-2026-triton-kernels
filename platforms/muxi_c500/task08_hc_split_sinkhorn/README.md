# Task08：hc_split_sinkhorn

> C500 状态：已通过。原二维 `tl.arange` 广播和跨轴归约在当前 mcTriton 后端触发段错误，C500 版本将固定 4×4 矩阵显式标量化。

## 算子说明

对每个 4×4 组合矩阵执行 20 轮 Sinkhorn 行列归一化，同时生成 pre/post 门控结果。

## 优化方案

- 每个 Triton program 处理一个 `(batch, seq)` 行。
- 4×4 矩阵在完整的 20 轮 Sinkhorn 循环中始终保存在寄存器中。
- 将 sigmoid、指数、行归一化、列归一化和三个输出写回融合为一个 kernel。
- 避免每轮产生多个小型 PyTorch kernel 和中间张量。

## C500 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 1.585844 ms | 0.148244 ms | 10.697x |

本地最大绝对误差为 `1.19209290e-07`。

## 复现

```bash
cd /data/zk/kernelswift-2026-triton-kernels/platforms/muxi_c500/task08_hc_split_sinkhorn
CUDA_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task08 从零教程](../../../docs/tutorial/tasks/08_hc_split_sinkhorn.md)。

