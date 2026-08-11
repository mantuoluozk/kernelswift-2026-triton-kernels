# Task10：head_compute_mix_bwd

## 算子说明

计算 sigmoid 相关的输入梯度，以及 `mhc_scale` 和四个 `mhc_base` 参数的归约梯度。

## 优化方案

- 将 sigmoid backward、`grad_input_mix` 写回和所有参数梯度归约融合为一个 Triton kernel。
- 针对固定输入 `2×1024×4`（8192 个 FP32 元素）静态专用化 block size。
- 减少计时区间内重复的形状、设备和连续性检查。

## BW1000 官方结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 0.176306 ms | 0.095530 ms | 1.846x |

关键配置：8 个 Triton warps。本地最大绝对误差为 `1.90734863e-06`。

## 复现

```bash
cd /data/zk/kernelswift-2026-triton-kernels/platforms/bw1000/task10_head_compute_mix_bwd
HIP_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task10 从零教程](../../../docs/tutorial/tasks/10_head_compute_mix_bwd.md)。
