# Task07：mhc_post

## 算子说明

对四路 residual 做 4×4 混合，再与 post-mix 结果相加，最终输出 BF16 张量。

## 优化方案

- 分段测量确认 torch_gcu 的 4×4 batched `einsum` 已高度优化，因此不再用 Triton 重写该 contraction。
- Triton epilogue 融合 `x * post_layer_mix + term2` 与 BF16 写回。
- 每个 Triton program 处理一个 `(batch, token)` 行，2048 元素 tile 一次覆盖 1280 hidden。
- S60 使用 `num_warps=1`；4 warps 会把混合版本从约 5.8 ms 拉高到约 15.6 ms。

## S60 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 4.213931 ms | 5.820457 ms | 0.724x |

关键配置：hidden block size 2048，1 个 Triton warp。旧全 Triton 版本为 29.462086 ms，新混合融合边界快约 5.06 倍。完整分段数据和失败实验见 [S60 教程](../../../docs/tutorial/06_S60环境与迁移.md)。

## 复现

```bash
cd /data/kernelswift/platforms/enflame_s60/task07_mhc_post
TOPS_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task07 从零教程](../../../docs/tutorial/tasks/07_mhc_post.md)。
