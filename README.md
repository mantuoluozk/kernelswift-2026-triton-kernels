# KernelSwift 2026 Triton 算子优化

[![在线教程](https://img.shields.io/badge/在线教程-GitHub_Pages-315efb)](https://mantuoluozk.github.io/kernelswift-2026-triton-kernels/)

**在线教程：** [从零学习 Triton 算子开发与优化——BW1000 与 C500 十题实战](https://mantuoluozk.github.io/kernelswift-2026-triton-kernels/)

本仓库包含 2026 KernelSwift 算子创新大赛赛道一的 Triton 优化实现。代码按芯片平台组织，当前已完成并验证海光 BW1000 与沐曦 C500 两个平台的 10 个任务。

## 仓库结构

```text
kernelswift-2026-triton-kernels/
├── docs/                         # 比赛规则、提交清单与说明
├── evaluator/                    # 官方 auto_bench.py 与上游许可证
├── platforms/
│   ├── README.md                 # 多平台目录约定
│   ├── bw1000/
│       ├── README.md             # BW1000 环境与成绩
│       ├── docker_create.sh      # 创建或进入测试容器
│       ├── run_all_benchmarks.sh # 一键运行 10 项官方评测
│       └── task01...task10/      # 参考实现、优化实现和任务说明
│   └── muxi_c500/                # C500 环境、迁移指南和 10 项实现
└── README.md
```

每个任务目录均包含：

- `solution.py`：Triton 优化实现，也是官方评测提交文件；
- `reference.py`：PyTorch 基准实现；
- `benchmark.py`：本地诊断、精度检查和性能测试脚本；
- `README.md`：任务原理、优化思路、成绩和复现方法。

## 平台进度

| 芯片平台 | 任务进度 | 正确性 | 状态 |
| --- | ---: | --- | --- |
| 海光 BW1000 | 10 / 10 | 全部通过 | 可提交 |
| 沐曦 C500（25% sGPU） | 10 / 10 | 全部通过 | 可复现 |

## BW1000 优化结果

测试环境为海光 BW1000（`gfx936`）、DTK 26.04、PyTorch 2.7.1、Triton 3.1.0。官方评测参数为 `warmup=200`、`repeat=500`，结果取中位数，精度阈值为 `atol=1e-2`、`rtol=1e-2`。

| 任务 | 算子 | PyTorch（ms） | Triton（ms） | 加速比 | 正确性 |
| --- | --- | ---: | ---: | ---: | --- |
| [Task01](platforms/bw1000/task01_grouped_topk/README.md) | GroupedTopk | 0.275516 | 0.088945 | 3.098x | PASS |
| [Task02](platforms/bw1000/task02_fused_moe/README.md) | FusedMoE | 2.602321 | 0.145260 | 17.915x | PASS |
| [Task03](platforms/bw1000/task03_flex_attention/README.md) | FlexAttention | 0.143880 | 0.086850 | 1.657x | PASS |
| [Task04](platforms/bw1000/task04_splade_sparse_pooler/README.md) | SPLADESparsePooler | 0.903172 | 0.308420 | 2.928x | PASS |
| [Task05](platforms/bw1000/task05_music_flamingo_rotary_embedding/README.md) | MusicFlamingoRotaryEmbedding | 0.254340 | 0.097860 | 2.599x | PASS |
| [Task06](platforms/bw1000/task06_mm_encoder_attention/README.md) | MMEncoderAttention | 0.141271 | 0.085000 | 1.662x | PASS |
| [Task07](platforms/bw1000/task07_mhc_post/README.md) | mhc_post | 2.569192 | 0.223235 | 11.509x | PASS |
| [Task08](platforms/bw1000/task08_hc_split_sinkhorn/README.md) | hc_split_sinkhorn | 1.399493 | 0.103106 | 13.573x | PASS |
| [Task09](platforms/bw1000/task09_centre_random_augmentation/README.md) | CentreRandomAugmentation | 0.934912 | 0.168371 | 5.553x | PASS |
| [Task10](platforms/bw1000/task10_head_compute_mix_bwd/README.md) | head_compute_mix_bwd | 0.176306 | 0.095530 | 1.846x | PASS |
| **十项简单合计** | — | **9.400403** | **1.402577** | **6.702x** | **全部通过** |

> 合计耗时仅用于直观汇总，不代表比赛官方综合评分。每道任务、每种芯片平台均以官方规则独立评测和排名。

## C500 优化结果

测试环境为沐曦 C500 的 25% sGPU 切片、MACA 3.0.0.8、mcPyTorch 2.4.0+metax3.0.0.3、mcTriton 3.0.0+metax3.0.0.3。评测参数同样为 `warmup=200`、`repeat=500`，结果取中位数，10 项正确性检查全部通过。

| 任务 | 算子 | Triton（ms） | 加速比 | 正确性 | 迁移说明 |
| --- | --- | ---: | ---: | --- | --- |
| [Task01](platforms/muxi_c500/task01_grouped_topk/README.md) | GroupedTopk | 0.097445 | 3.442x | PASS | BW1000 实现可复用 |
| [Task02](platforms/muxi_c500/task02_fused_moe/README.md) | FusedMoE | 0.153823 | 18.943x | PASS | BW1000 实现可复用 |
| [Task03](platforms/muxi_c500/task03_flex_attention/README.md) | FlexAttention | 0.102738 | 1.247x | PASS | 算法可复用，tile 仍可调优 |
| [Task04](platforms/muxi_c500/task04_splade_sparse_pooler/README.md) | SPLADESparsePooler | 0.411690 | 2.384x | PASS | 矩阵 tile 仍可调优 |
| [Task05](platforms/muxi_c500/task05_music_flamingo_rotary_embedding/README.md) | MusicFlamingoRotaryEmbedding | 0.094623 | 2.387x | PASS | BW1000 实现可复用 |
| [Task06](platforms/muxi_c500/task06_mm_encoder_attention/README.md) | MMEncoderAttention | 0.106350 | 1.285x | PASS | Attention tile 仍可调优 |
| [Task07](platforms/muxi_c500/task07_mhc_post/README.md) | mhc_post | 0.243420 | 16.745x | PASS | 设置 NCHW 后复用高性能 kernel |
| [Task08](platforms/muxi_c500/task08_hc_split_sinkhorn/README.md) | hc_split_sinkhorn | 0.148244 | 10.697x | PASS | 显式标量化 4×4 Sinkhorn |
| [Task09](platforms/muxi_c500/task09_centre_random_augmentation/README.md) | CentreRandomAugmentation | 0.166514 | 5.797x | PASS | BW1000 实现可复用 |
| [Task10](platforms/muxi_c500/task10_head_compute_mix_bwd/README.md) | head_compute_mix_bwd | 0.104272 | 1.636x | PASS | 归约参数仍可调优 |
| **十项简单合计** | — | **1.629119** | **7.071x** | **全部通过** | PyTorch 合计 **11.519283 ms** |

> C500 合计同样只用于直观汇总，不代表官方综合评分。完整环境配置、显卡监控、Tensor 布局和海光迁移差异参见 [C500 平台说明](platforms/muxi_c500/README.md)与 [C500 环境与迁移指南](platforms/muxi_c500/C500_环境与迁移指南.md)。

## 复现 BW1000 结果

先确认空闲设备，再进入平台目录：

```bash
hy-smi --showmeminfo vram --showuse
cd /data/zk/kernelswift-2026-triton-kernels/platforms/bw1000
```

创建或进入容器：

```bash
chmod +x docker_create.sh
./docker_create.sh
```

运行全部官方评测：

```bash
chmod +x run_all_benchmarks.sh
DEVICE_ID=0 ./run_all_benchmarks.sh
```

运行单个任务：

```bash
HIP_VISIBLE_DEVICES=0 python3 ../../evaluator/auto_bench.py \
  --v0_file task04_splade_sparse_pooler/reference.py \
  --v1_file task04_splade_sparse_pooler/solution.py \
  --warmup 200 --repeat 500
```

Docker 镜像、容器参数和完整结果参见 [BW1000 平台说明](platforms/bw1000/README.md)。比赛规则与交付要求参见 [比赛说明](docs/比赛说明.md) 和 [提交清单](docs/提交清单.md)。本机实际使用的 `.remote-dev.json` 已忽略，不会提交内网服务器地址。

面向 Triton 和算子优化初学者的完整学习资料参见：[《从零学习 Triton 算子开发与优化：BW1000 与 C500 十题实战》](docs/tutorial/README.md)。教程从编程模型、正确性和性能测量开始，并逐题解释 10 个优化实现的数学依据、kernel 映射、参数选择与跨平台迁移差异。

## 原创声明

本仓库中的 Triton 优化代码由参赛者基于官方参考实现独立开发和调优。提交内容未通过异常捕获、条件分支或 fallback 等方式绕过自定义算子执行；性能数据来自所声明的平台和官方评测脚本。
