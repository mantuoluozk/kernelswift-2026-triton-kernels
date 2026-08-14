# Task01：GroupedTopk

## 算子说明

对 256 个专家进行分组选择，保留得分最高的 4 个专家组，再从候选专家中选出 Top-8 并归一化权重。

## 优化方案

- 利用 Softmax 的单调性，直接从 logits 选择专家组和专家。
- 将组内最大值、Top-4 组选择和 256 专家掩码融合为一个 Triton kernel。
- 原始 256 路 Softmax 的分母会在最终重新归一化时抵消，因此只对选中的 8 个 logits 做归一化。
- 最终实现用一个 kernel 完成选择、归一化以及两个输出的写回。

## BW1000 迁移基线

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 0.275516 ms | 0.088945 ms | 3.098x |

关键配置：`num_warps=1`。

## 复现

```bash
cd /data/zk/kernelswift-2026-triton-kernels/platforms/muxi_c500/task01_grouped_topk
CUDA_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task01 从零教程](../../../docs/tutorial/tasks/01_GroupedTopk.md)。

