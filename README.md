# KernelSwift 2026 Triton 算子优化

[![在线教程](https://img.shields.io/badge/在线教程-GitHub_Pages-315efb)](https://mantuoluozk.github.io/kernelswift-2026-triton-kernels/)

**在线教程：** [从零学习 Triton 算子开发与优化——四平台十题实战](https://mantuoluozk.github.io/kernelswift-2026-triton-kernels/)

本仓库包含 2026 KernelSwift 算子创新大赛赛道一的 Triton 优化实现。代码按芯片平台组织，当前已完成并验证海光 BW1000、沐曦 C500、昇腾 Ascend 910B 与燧原 S60 四个平台的 10 个任务。

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
│   ├── muxi_c500/                # C500 环境、迁移指南和 10 项实现
│   ├── ascend910b/               # 910B 环境、迁移指南和 10 项实现
│   └── enflame_s60/              # S60 环境、迁移指南和 10 项实现
└── README.md
```

每个任务目录均包含：

- `solution.py`：Triton 优化实现，也是官方评测提交文件；
- `reference.py`：PyTorch 基准实现；
- `benchmark.py`：本地诊断、精度检查和性能测试脚本；
- `README.md`：任务原理、优化思路、成绩和复现方法。

## 平台进度

| 芯片平台 | 任务进度 | PyTorch 合计 | Triton 合计 | 简单合计加速比 | 正确性 |
| --- | ---: | ---: | ---: | ---: | --- |
| 海光 BW1000 | 10 / 10 | 9.400403 ms | 1.402577 ms | 6.702× | 全部通过 |
| 沐曦 C500（25% sGPU） | 10 / 10 | 11.519283 ms | 1.629119 ms | 7.071× | 全部通过 |
| 昇腾 Ascend 910B1 | 10 / 10 | 17.297547 ms | 4.610002 ms | 3.752× | 全部通过 |
| 燧原 S60 | 10 / 10 | 16.828573 ms | 34.657672 ms | 0.486× | 全部通过 |

四平台均使用同一官方评测器和 `warmup=200`、`repeat=500` 参数。合计仅用于直观汇总，不代表官方跨任务或跨平台评分；每个任务的优化前后数据见下方对应平台完整表格。

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

| 任务 | 算子 | PyTorch（ms） | Triton（ms） | 加速比 | 正确性 | 迁移说明 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| [Task01](platforms/muxi_c500/task01_grouped_topk/README.md) | GroupedTopk | 0.335406 | 0.097445 | 3.442x | PASS | BW1000 实现可复用 |
| [Task02](platforms/muxi_c500/task02_fused_moe/README.md) | FusedMoE | 2.913869 | 0.153823 | 18.943x | PASS | BW1000 实现可复用 |
| [Task03](platforms/muxi_c500/task03_flex_attention/README.md) | FlexAttention | 0.128114 | 0.102738 | 1.247x | PASS | 算法可复用，tile 仍可调优 |
| [Task04](platforms/muxi_c500/task04_splade_sparse_pooler/README.md) | SPLADESparsePooler | 0.981469 | 0.411690 | 2.384x | PASS | 矩阵 tile 仍可调优 |
| [Task05](platforms/muxi_c500/task05_music_flamingo_rotary_embedding/README.md) | MusicFlamingoRotaryEmbedding | 0.225865 | 0.094623 | 2.387x | PASS | BW1000 实现可复用 |
| [Task06](platforms/muxi_c500/task06_mm_encoder_attention/README.md) | MMEncoderAttention | 0.136660 | 0.106350 | 1.285x | PASS | Attention tile 仍可调优 |
| [Task07](platforms/muxi_c500/task07_mhc_post/README.md) | mhc_post | 4.076068 | 0.243420 | 16.745x | PASS | 设置 NCHW 后复用高性能 kernel |
| [Task08](platforms/muxi_c500/task08_hc_split_sinkhorn/README.md) | hc_split_sinkhorn | 1.585766 | 0.148244 | 10.697x | PASS | 显式标量化 4×4 Sinkhorn |
| [Task09](platforms/muxi_c500/task09_centre_random_augmentation/README.md) | CentreRandomAugmentation | 0.965282 | 0.166514 | 5.797x | PASS | BW1000 实现可复用 |
| [Task10](platforms/muxi_c500/task10_head_compute_mix_bwd/README.md) | head_compute_mix_bwd | 0.170589 | 0.104272 | 1.636x | PASS | 归约参数仍可调优 |
| **十项简单合计** | — | **11.519283** | **1.629119** | **7.071x** | **全部通过** | — |

> C500 合计同样只用于直观汇总，不代表官方综合评分。完整环境配置、显卡监控、Tensor 布局和海光迁移差异参见 [C500 平台说明](platforms/muxi_c500/README.md)与 [C500 环境与迁移指南](platforms/muxi_c500/C500_环境与迁移指南.md)。

## 昇腾 Ascend 910B 优化结果

测试环境为 Ascend 910B1、CANN 9.0.0、PyTorch/torch_npu 2.10.0、Triton-Ascend 3.2.1、Python 3.11.15。评测参数为 `warmup=200`、`repeat=500`，10 项正确性检查全部通过。

| 任务 | 算子 | PyTorch（ms） | Triton（ms） | 加速比 | 正确性 | 迁移说明 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| [Task01](platforms/ascend910b/task01_grouped_topk/README.md) | GroupedTopk | 0.701666 | 0.283144 | 2.478x | PASS | 选择与归一化直接复用 |
| [Task02](platforms/ascend910b/task02_fused_moe/README.md) | FusedMoE | 6.942960 | 0.606584 | 11.446x | PASS | 路由与分阶段 GEMM 直接复用 |
| [Task03](platforms/ascend910b/task03_flex_attention/README.md) | FlexAttention | 0.372326 | 0.303849 | 1.225x | PASS | query tile 调为 32×32 |
| [Task04](platforms/ascend910b/task04_splade_sparse_pooler/README.md) | SPLADESparsePooler | 0.721766 | 0.659824 | 1.094x | PASS | 拆分 decoder 与池化，避开 watchdog |
| [Task05](platforms/ascend910b/task05_music_flamingo_rotary_embedding/README.md) | MusicFlamingoRotaryEmbedding | 0.535493 | 0.286519 | 1.869x | PASS | 逐元素融合直接复用 |
| [Task06](platforms/ascend910b/task06_mm_encoder_attention/README.md) | MMEncoderAttention | 0.349675 | 0.300920 | 1.162x | PASS | query tile 调为 32 |
| [Task07](platforms/ascend910b/task07_mhc_post/README.md) | mhc_post | 1.980275 | 0.822972 | 2.406x | PASS | 大张量融合直接复用 |
| [Task08](platforms/ascend910b/task08_hc_split_sinkhorn/README.md) | hc_split_sinkhorn | 3.032301 | 0.329475 | 9.203x | PASS | 4×4 标量化 Sinkhorn 直接复用 |
| [Task09](platforms/ascend910b/task09_centre_random_augmentation/README.md) | CentreRandomAugmentation | 2.301964 | 0.679200 | 3.389x | PASS | 随机变换融合直接复用 |
| [Task10](platforms/ascend910b/task10_head_compute_mix_bwd/README.md) | head_compute_mix_bwd | 0.359121 | 0.337515 | 1.064x | PASS | 两个 4096 tile，atomic 汇总 |
| **十项简单合计** | — | **17.297547** | **4.610002** | **3.752x** | **全部通过** | — |

> 910B 合计只用于直观汇总，不代表官方综合评分。完整环境、安装、UB 限制和 AICore watchdog 排错过程参见 [910B 平台说明](platforms/ascend910b/README.md)与 [910B 环境与迁移指南](platforms/ascend910b/Ascend910B_环境与迁移指南.md)。

## 燧原 S60 优化结果

测试环境为 Enflame S60、驱动 1.9.29、PyTorch 2.10.0、torch_gcu 2.10.0+3.8.0.2、Triton 3.6.0 与 triton_gcu 3.6.0+1.0.20260722。评测参数为 `warmup=200`、`repeat=500`，10 项正确性全部通过。

| 任务 | 算子 | PyTorch（ms） | Triton（ms） | 加速比 | 正确性 | 迁移说明 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| [Task01](platforms/enflame_s60/task01_grouped_topk/README.md) | GroupedTopk | 0.457112 | 0.331359 | 1.380x | PASS | 选择与归一化直接复用 |
| [Task02](platforms/enflame_s60/task02_fused_moe/README.md) | FusedMoE | 5.009264 | 0.385704 | 12.987x | PASS | 分阶段 GEMM 与路由融合 |
| [Task03](platforms/enflame_s60/task03_flex_attention/README.md) | FlexAttention | 0.231255 | 0.286620 | 0.807x | PASS | `num_warps=1` |
| [Task04](platforms/enflame_s60/task04_splade_sparse_pooler/README.md) | SPLADESparsePooler | 0.940885 | 1.576604 | 0.597x | PASS | 展平二维 grid，拆分池化 |
| [Task05](platforms/enflame_s60/task05_music_flamingo_rotary_embedding/README.md) | MusicFlamingoRotaryEmbedding | 0.435885 | 0.342935 | 1.271x | PASS | 逐元素融合直接复用 |
| [Task06](platforms/enflame_s60/task06_mm_encoder_attention/README.md) | MMEncoderAttention | 0.274237 | 0.335449 | 0.818x | PASS | 32 行 tile、1 warp |
| [Task07](platforms/enflame_s60/task07_mhc_post/README.md) | mhc_post | 4.194152 | 29.462086 | 0.142x | PASS | 全 Triton 正确，后端调度待优化 |
| [Task08](platforms/enflame_s60/task08_hc_split_sinkhorn/README.md) | hc_split_sinkhorn | 2.015706 | 0.225345 | 8.945x | PASS | 4×4 标量化 Sinkhorn |
| [Task09](platforms/enflame_s60/task09_centre_random_augmentation/README.md) | CentreRandomAugmentation | 2.924597 | 1.422888 | 2.055x | PASS | 随机变换融合直接复用 |
| [Task10](platforms/enflame_s60/task10_head_compute_mix_bwd/README.md) | head_compute_mix_bwd | 0.345480 | 0.288682 | 1.197x | PASS | `% 4` 索引与混合末级归约 |
| **十项简单合计** | — | **16.828573** | **34.657672** | **0.486x** | **全部通过** | — |

> S60 的简单合计受 Task07 明显影响，不代表逐题积分。完整环境、`efsmi`/`TOPS_VISIBLE_DEVICES` 命令、段错误排查与海光迁移差异参见 [S60 平台说明](platforms/enflame_s60/README.md)和 [S60 环境与迁移指南](platforms/enflame_s60/S60_环境与迁移指南.md)。

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

面向 Triton 和算子优化初学者的完整学习资料参见：[《从零学习 Triton 算子开发与优化：四平台十题实战》](docs/tutorial/README.md)。教程从编程模型、正确性和性能测量开始，并逐题解释 10 个优化实现的数学依据、kernel 映射、参数选择与跨平台迁移差异。

## 原创声明

本仓库中的 Triton 优化代码由参赛者基于官方参考实现独立开发和调优。提交内容未通过异常捕获、条件分支或 fallback 等方式绕过自定义算子执行；性能数据来自所声明的平台和官方评测脚本。
