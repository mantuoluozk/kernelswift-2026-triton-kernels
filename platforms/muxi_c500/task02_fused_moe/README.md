# Task02：FusedMoE

## 算子说明

包含 8 个专家的 MoE 前向计算，每个 token 选择 Top-2 专家，并对两个专家输出加权求和。

## 优化方案

- 用三个规则化 Triton 阶段替代动态布尔分发、大量小 GEMM 和每次前向的完整权重转换。
- 第一阶段为所有专家计算 gate/up 投影，并融合 SiLU 与逐元素乘法。
- 第二阶段直接从紧凑的专家激活缓冲区计算 down 投影。
- 第三阶段直接从路由 logits 选择 Top-2，只归一化两个权重并融合专家输出。
- kernel 内按 tile 将 FP32 权重转换为 FP16，避免创建完整的权重转换副本。

## BW1000 迁移基线

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 2.602321 ms | 0.145260 ms | 17.915x |

关键配置：32-token tile，`num_warps=4`。

## 复现

```bash
cd /data/zk/kernelswift-2026-triton-kernels/platforms/muxi_c500/task02_fused_moe
CUDA_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task02 从零教程](../../../docs/tutorial/tasks/02_FusedMoE.md)。

