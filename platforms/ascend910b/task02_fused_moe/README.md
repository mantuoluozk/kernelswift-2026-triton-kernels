# Task02：FusedMoE

## 算子说明

包含 8 个专家的 MoE 前向计算，每个 token 选择 Top-2 专家，并对两个专家输出加权求和。

## 优化方案

- 用三个规则化 Triton 阶段替代动态布尔分发、大量小 GEMM 和每次前向的完整权重转换。
- 第一阶段为所有专家计算 gate/up 投影，并融合 SiLU 与逐元素乘法。
- 第二阶段直接从紧凑的专家激活缓冲区计算 down 投影。
- 第三阶段直接从路由 logits 选择 Top-2，只归一化两个权重并融合专家输出。
- kernel 内按 tile 将 FP32 权重转换为 FP16，避免创建完整的权重转换副本。

## Ascend 910B 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 6.942960 ms | 0.606584 ms | 11.446x |

关键配置：32-token tile，`num_warps=4`。

## 复现

```bash
cd /data/kernelswift-2026-triton-kernels/platforms/ascend910b/task02_fused_moe
source ../setup_env.sh
python benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task02 从零教程](../../../docs/tutorial/tasks/02_FusedMoE.md)。
