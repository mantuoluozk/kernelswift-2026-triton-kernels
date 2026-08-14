# Task08：hc_split_sinkhorn

> Ascend 910B 状态：已通过。继续复用固定 4×4 矩阵的标量化实现，避免为极小矩阵引入复杂的二维归约状态。

## 算子说明

对每个 4×4 组合矩阵执行 20 轮 Sinkhorn 行列归一化，同时生成 pre/post 门控结果。

## 优化方案

- 每个 Triton program 处理一个 `(batch, seq)` 行。
- 4×4 矩阵在完整的 20 轮 Sinkhorn 循环中始终保存在寄存器中。
- 将 sigmoid、指数、行归一化、列归一化和三个输出写回融合为一个 kernel。
- 避免每轮产生多个小型 PyTorch kernel 和中间张量。

## Ascend 910B 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 3.032301 ms | 0.329475 ms | 9.203x |

本地最大绝对误差为 `1.19209290e-07`。

## 复现

```bash
cd /data/kernelswift-2026-triton-kernels/platforms/ascend910b/task08_hc_split_sinkhorn
source ../setup_env.sh
python benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task08 从零教程](../../../docs/tutorial/tasks/08_hc_split_sinkhorn.md)。
