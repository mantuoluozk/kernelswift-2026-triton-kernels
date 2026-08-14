# Task03：FlexAttention

## 算子说明

固定 head dimension 为 64 的因果注意力计算，输出布局为 `[83, 512]`。

## 优化方案

- 将 QK、因果掩码、在线 Softmax 和 PV 融合为一个 Triton kernel。
- 使用 `32×32` 的 query/key tile，在 910B 上减少 program 启动数量。
- 在寄存器中保存 Softmax 的运行最大值、分母和输出累加器。
- 不生成完整注意力矩阵，直接写入目标输出布局。

## Ascend 910B 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 0.372326 ms | 0.303849 ms | 1.225x |

关键配置：`BLOCK_M=32`、`BLOCK_N=32`、`num_warps=4`。

## 复现

```bash
cd /data/kernelswift-2026-triton-kernels/platforms/ascend910b/task03_flex_attention
source ../setup_env.sh
python benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task03 从零教程](../../../docs/tutorial/tasks/03_FlexAttention.md)。
