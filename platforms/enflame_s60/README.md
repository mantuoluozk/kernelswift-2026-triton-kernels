# 燧原 S60 平台

本目录保存 KernelSwift 赛道一在燧原 S60 上的 Triton 迁移与优化实现。10 项均在真实 S60 上按官方参数完成正确性和性能回归。

## 验证环境

- 设备：Enflame S60，42976 MiB 显存
- EFSMI：1.8.7；驱动：1.9.29；Boot Firmware：33.6.5
- Ubuntu 24.04.3，Python 3.12.3
- PyTorch：2.10.0+cpu
- torch_gcu：2.10.0+3.8.0.2
- Triton：3.6.0
- triton_gcu：3.6.0+1.0.20260722
- Triton target：`gcu300`，后端报告 `warp_size=12`

实例已预装完整软件栈，因此直接使用系统虚拟环境，没有覆盖厂商 PyTorch、驱动或固件。输入必须显式放到 `gcu`；`torch_gcu` 提供部分 CUDA 迁移兼容，但本项目评测器按真实后端使用 `torch.gcu.synchronize()`。

## 快速检查

```bash
efsmi
python - <<'PY'
import torch
import torch_gcu
print(torch.__version__, torch_gcu.__version__)
print(torch.gcu.is_available(), torch.gcu.device_count())
print(torch.gcu.get_device_name(0))
PY
python platforms/enflame_s60/smoke_test.py
```

选择设备使用 `TOPS_VISIBLE_DEVICES`，必须在 Python 进程启动前设置：

```bash
TOPS_VISIBLE_DEVICES=0 python your_program.py
```

## 运行评测

```bash
# 短回归
WARMUP=5 REPEAT=10 bash platforms/enflame_s60/run_all_benchmarks.sh

# 正式参数
WARMUP=200 REPEAT=500 bash platforms/enflame_s60/run_all_benchmarks.sh
```

单题示例：

```bash
TOPS_VISIBLE_DEVICES=0 python evaluator/auto_bench.py \
  --v0_file platforms/enflame_s60/task10_head_compute_mix_bwd/reference.py \
  --v1_file platforms/enflame_s60/task10_head_compute_mix_bwd/solution.py \
  --warmup 200 --repeat 500
```

## 正式回归结果

参数为 `warmup=200`、`repeat=500`，结果取中位数，精度阈值为 `atol=1e-2`、`rtol=1e-2`。

| 任务 | PyTorch（ms） | Triton（ms） | 加速比 | 正确性 | S60 迁移要点 |
| --- | ---: | ---: | ---: | --- | --- |
| Task01 · GroupedTopk | 0.457112 | 0.331359 | 1.380× | PASS | 选择与归一化 kernel 可复用 |
| Task02 · FusedMoE | 5.009264 | 0.385704 | 12.987× | PASS | 分阶段 GEMM 与路由融合可复用 |
| Task03 · FlexAttention | 0.231255 | 0.286620 | 0.807× | PASS | `num_warps=1` 显著优于 4 |
| Task04 · SPLADESparsePooler | 0.940885 | 1.576604 | 0.597× | PASS | 拆分厂商 GEMM/LayerNorm 与 Triton 池化 |
| Task05 · RotaryEmbedding | 0.435885 | 0.342935 | 1.271× | PASS | 逐元素融合可复用 |
| Task06 · MMEncoderAttention | 0.274237 | 0.335449 | 0.818× | PASS | query tile 调为 32，`num_warps=1` |
| Task07 · mhc_post | 4.213931 | 5.820457 | 0.724× | PASS | 厂商 einsum + Triton epilogue，1 warp |
| Task08 · hc_split_sinkhorn | 2.015706 | 0.225345 | 8.945× | PASS | 4×4 Sinkhorn 标量化可复用 |
| Task09 · CentreRandomAugmentation | 2.924597 | 1.422888 | 2.055× | PASS | 随机变换融合可复用 |
| Task10 · head_compute_mix_bwd | 0.345480 | 0.288682 | 1.197× | PASS | `% 4` 索引；Triton 点算子与 torch_gcu 归约混合 |
| **十项简单合计** | **16.848352** | **11.016043** | **1.529×** | **全部通过** | — |

合计只用于直观汇总，不代表比赛官方综合评分。Task07 通过保留厂商高效 einsum、使用 Triton 融合广播乘法/加法/BF16 写回，从 29.462086 ms 降至 5.820457 ms；它仍未反超 PyTorch，但证明跨后端需要重新选择融合边界。

环境命令、常见故障和从海光迁移的差异见 [S60 环境与迁移指南](S60_环境与迁移指南.md)。
