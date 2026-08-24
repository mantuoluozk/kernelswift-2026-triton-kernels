# Task08：hc_split_sinkhorn

## 算子说明

对每个 4×4 组合矩阵执行 20 轮 Sinkhorn 行列归一化，同时生成 pre/post 门控结果。

## 优化方案

- 每个 Triton program 处理一个 `(batch, seq)` 行。
- 4×4 矩阵在完整的 20 轮 Sinkhorn 循环中始终保存在寄存器中。
- 将 sigmoid、指数、行归一化、列归一化和三个输出写回融合为一个 kernel。
- 避免每轮产生多个小型 PyTorch kernel 和中间张量。

## S60 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 2.015706 ms | 0.225345 ms | 8.945x |

本地最大绝对误差为 `1.19209290e-07`。

## 复现

```bash
cd /data/kernelswift/platforms/enflame_s60/task08_hc_split_sinkhorn
TOPS_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task08 从零教程](../../../docs/tutorial/tasks/08_hc_split_sinkhorn.md)。
