# Task07：mhc_post

> C500 状态：已通过。mcPyTorch 默认会把 4D CPU 输入在搬到设备时转为 channels-last；设置 `PYTORCH_DEFAULT_NCHW=1` 后，Triton 原始指针可按标准连续布局读取。

## 算子说明

对四路 residual 做 4×4 混合，再与 post-mix 结果相加，最终输出 BF16 张量。

## 优化方案

- 每个 Triton program 处理一个 `(batch, token)` 行。
- 四个 residual 向量只加载一次，并复用于四路输出混合。
- 4×4 系数和中间累加结果保存在寄存器中。
- 直接写出 BF16，删除大型 FP32 einsum、post-mix 和加法中间张量。

## C500 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 4.076138 ms | 0.243420 ms | 16.745x |

关键配置：`PYTORCH_DEFAULT_NCHW=1`、hidden block size 2048、4 个 Triton warps。输出使用一维缓冲区写入后 reshape，避免 mcPyTorch 对新建 4D Tensor 采用特殊布局。

## 复现

```bash
cd /data/zk/kernelswift-2026-triton-kernels/platforms/muxi_c500/task07_mhc_post
CUDA_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task07 从零教程](../../../docs/tutorial/tasks/07_mhc_post.md)。

