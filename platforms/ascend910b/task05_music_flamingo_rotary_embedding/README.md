# Task05：MusicFlamingoRotaryEmbedding

## 算子说明

根据 batch 位置和时间位置生成旋转位置编码所需的 cosine、sine 张量。

## 优化方案

- 从扁平索引直接推导每个输出坐标对应的 batch/time 频率。
- 将位置选择、相位计算、cosine 和 sine 融合为一个 Triton kernel。
- 不再生成 repeat、broadcast 和 concat 等中间张量。
- 保留原始 `inv_freq` 和 `position_angles` buffer，兼容官方权重加载。

## Ascend 910B 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 0.535493 ms | 0.286519 ms | 1.869x |

关键配置：block size 1024，8 个 Triton warps。本地 cosine、sine 输出逐位一致，`max_abs_diff=0`。

## 复现

```bash
cd /data/kernelswift-2026-triton-kernels/platforms/ascend910b/task05_music_flamingo_rotary_embedding
source ../setup_env.sh
python benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task05 从零教程](../../../docs/tutorial/tasks/05_RotaryEmbedding.md)。
