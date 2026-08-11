# Task06：MMEncoderAttention

## 算子说明

输入布局为 `[2, 83, 512]` 的非因果多头注意力计算。

## 优化方案

- 将 QK、在线 Softmax 和 PV 融合为一个非因果 Triton kernel。
- 每个 program 处理一个 batch/head 和 64-query tile。
- K/V 按 32-token tile 流式加载，Softmax 状态和输出累加器保存在寄存器中。
- 直接读取和写回原始连续布局，不创建 transpose 缓冲区或完整注意力矩阵。

## BW1000 官方结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 0.141271 ms | 0.085000 ms | 1.662x |

关键配置：`BLOCK_M=64`、`BLOCK_N=32`、`num_warps=4`。

## 复现

```bash
cd /data/zk/kernelswift-2026-triton-kernels/platforms/bw1000/task06_mm_encoder_attention
HIP_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task06 从零教程](../../../docs/tutorial/tasks/06_MMEncoderAttention.md)。
