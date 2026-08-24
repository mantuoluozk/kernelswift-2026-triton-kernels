# Task10：head_compute_mix_bwd

## 算子说明

计算 sigmoid 相关的输入梯度，以及 `mhc_scale` 和四个 `mhc_base` 参数的归约梯度。

## 优化方案

- 将 sigmoid backward、`grad_input_mix` 写回和所有参数梯度归约融合为一个 Triton kernel。
- 针对固定输入 `2×1024×4`（8192 个 FP32 元素）静态专用化 block size。
- 减少计时区间内重复的形状、设备和连续性检查。

## S60 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 0.345480 ms | 0.288682 ms | 1.197x |

关键配置：`BLOCK_SIZE=256`、`num_warps=1`。索引必须用 `% 4`；等价的 `& 3` 在当前后端会段错误。Triton 融合点算子，torch_gcu 完成末级归约。

## 复现

```bash
cd /data/kernelswift/platforms/enflame_s60/task10_head_compute_mix_bwd
TOPS_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task10 从零教程](../../../docs/tutorial/tasks/10_head_compute_mix_bwd.md)。
