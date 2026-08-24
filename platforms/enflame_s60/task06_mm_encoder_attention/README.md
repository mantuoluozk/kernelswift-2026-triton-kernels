# Task06：MMEncoderAttention

## 算子说明

输入布局为 `[2, 83, 512]` 的非因果多头注意力计算。

## 优化方案

- 将 QK、在线 Softmax 和 PV 融合为一个非因果 Triton kernel。
- 每个 program 处理一个 batch/head 和 64-query tile。
- K/V 按 32-token tile 流式加载，Softmax 状态和输出累加器保存在寄存器中。
- 直接读取和写回原始连续布局，不创建 transpose 缓冲区或完整注意力矩阵。

## S60 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 0.274237 ms | 0.335449 ms | 0.818x |

关键配置：`BLOCK_M=32`、`BLOCK_N=32`、`num_warps=1`；原海光配置在 S60 上约 9.88 ms。

## 复现

```bash
cd /data/kernelswift/platforms/enflame_s60/task06_mm_encoder_attention
TOPS_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task06 从零教程](../../../docs/tutorial/tasks/06_MMEncoderAttention.md)。
