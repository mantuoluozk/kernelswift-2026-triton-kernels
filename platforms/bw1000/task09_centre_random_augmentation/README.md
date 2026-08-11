# Task09：CentreRandomAugmentation

## 算子说明

对 256 个原子坐标进行带 mask 的中心化，并生成 4 组随机旋转和平移增强结果。

## 优化方案

- 保留参考实现的 3 次 `torch.rand` 和 1 次 `torch.randn` 调用边界及顺序，确保官方每次重置种子后随机值完全一致。
- 将带 mask 中心化、四元数转旋转矩阵、刚体变换、mask 和最终写回融合为一个 Triton kernel。
- 每个 program 处理一个增强样本，256 个原子的归约在寄存器中完成。

## BW1000 官方结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 0.934912 ms | 0.168371 ms | 5.553x |

关键配置：`num_warps=1`。本地最大绝对误差为 `2.38418579e-07`。

## 复现

```bash
cd /data/zk/kernelswift-2026-triton-kernels/platforms/bw1000/task09_centre_random_augmentation
HIP_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500 --num-warps 1
```

详细原理与代码拆解：[Task09 从零教程](../../../docs/tutorial/tasks/09_CentreRandomAugmentation.md)。
