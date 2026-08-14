# Task10：head_compute_mix_bwd

## 算子说明

计算 sigmoid 相关的输入梯度，以及 `mhc_scale` 和四个 `mhc_base` 参数的归约梯度。

## 优化方案

- 将 sigmoid backward、`grad_input_mix` 写回和局部参数梯度归约融合为一个 Triton kernel。
- 把固定输入 `2×1024×4`（8192 个 FP32 元素）拆成两个 4096 元素 tile，避免 UB 溢出。
- 每个 program 写回自己的输入梯度，并使用 FP32 atomic 汇总 `grad_scale` 和四路 `grad_base`。
- 减少计时区间内重复的形状、设备和连续性检查。

## Ascend 910B 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 0.359121 ms | 0.337515 ms | 1.064x |

关键配置：`BLOCK_SIZE=4096`、2 个 program、`num_warps=4`。8192 单块版本需要约 320 KiB UB，超过 910B 后端可用的 192 KiB。

## 复现

```bash
cd /data/kernelswift-2026-triton-kernels/platforms/ascend910b/task10_head_compute_mix_bwd
source ../setup_env.sh
python benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task10 从零教程](../../../docs/tutorial/tasks/10_head_compute_mix_bwd.md)。
