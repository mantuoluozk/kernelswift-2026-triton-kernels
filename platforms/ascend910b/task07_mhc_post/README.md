# Task07：mhc_post

> Ascend 910B 状态：已通过。torch_npu 输入保持本题 kernel 预期的连续布局，无需 C500 平台的 channels-last 兼容设置。

## 算子说明

对四路 residual 做 4×4 混合，再与 post-mix 结果相加，最终输出 BF16 张量。

## 优化方案

- 每个 Triton program 处理一个 `(batch, token)` 行。
- 四个 residual 向量只加载一次，并复用于四路输出混合。
- 4×4 系数和中间累加结果保存在寄存器中。
- 直接写出 BF16，删除大型 FP32 einsum、post-mix 和加法中间张量。

## Ascend 910B 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 1.980275 ms | 0.822972 ms | 2.406x |

关键配置：hidden block size 2048、4 个 Triton warps。输出使用一维缓冲区写入后 reshape，保持写回地址简单连续。

## 复现

```bash
cd /data/kernelswift-2026-triton-kernels/platforms/ascend910b/task07_mhc_post
source ../setup_env.sh
python benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task07 从零教程](../../../docs/tutorial/tasks/07_mhc_post.md)。
