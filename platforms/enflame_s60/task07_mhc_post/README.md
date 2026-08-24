# Task07：mhc_post

## 算子说明

对四路 residual 做 4×4 混合，再与 post-mix 结果相加，最终输出 BF16 张量。

## 优化方案

- 每个 Triton program 处理一个 `(batch, token)` 行。
- 四个 residual 向量只加载一次，并复用于四路输出混合。
- 4×4 系数和中间累加结果保存在寄存器中。
- 直接写出 BF16，删除大型 FP32 einsum、post-mix 和加法中间张量。

## S60 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 4.194152 ms | 29.462086 ms | 0.142x |

关键配置：hidden block size 2048，4 个 Triton warps。拆成 256 元素小块会进一步退化到约 36.9 ms，当前版本保留较快且全 Triton 的正确实现。

## 复现

```bash
cd /data/kernelswift/platforms/enflame_s60/task07_mhc_post
TOPS_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task07 从零教程](../../../docs/tutorial/tasks/07_mhc_post.md)。
