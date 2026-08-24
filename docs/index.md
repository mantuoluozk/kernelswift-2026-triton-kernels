# Triton 算子开发与优化

本教程以 KernelSwift 2026 Triton 优化赛道的 10 个任务为例，介绍如何从 PyTorch 参考实现出发，完成语义分析、Triton kernel 编写、正确性验证和性能优化。

当前代码和性能数据来自海光 BW1000（`gfx936`）、沐曦 C500（25% sGPU）、昇腾 Ascend 910B1 和燧原 S60 四个平台。教程面向第一次接触 Triton 或 GPU/DCU/NPU/GCU 算子优化的读者，也可用于比较同一算法在不同 Triton 后端上的迁移表现。

!!! info "建议的阅读方式"
    初学者建议依次阅读基础章节，再进入十题实战。已经熟悉 Triton 编程模型的读者，可以直接从[算子优化方法论](tutorial/03_算子优化方法论.md)或具体任务开始。

## 学习路线

| 阶段 | 内容 | 目标 |
| --- | --- | --- |
| 1 | [整体认识与学习路线](tutorial/00_整体认识与学习路线.md) | 理解算子、kernel、program 和评测之间的关系 |
| 2 | [BW1000 环境与评测](tutorial/01_BW1000环境与评测.md) | 掌握设备选择、容器使用、计时和正确性验证 |
| 3 | [Triton 编程基础](tutorial/02_Triton编程基础.md) | 学会 grid、tile、mask、归约、`tl.dot` 和在线 Softmax |
| 4 | [算子优化方法论](tutorial/03_算子优化方法论.md) | 判断启动、访存和计算瓶颈，建立优化闭环 |
| 5 | [C500 环境与海光迁移](tutorial/04_C500环境与海光迁移.md) | 掌握 mcTriton 环境、显卡监控、布局差异和跨平台调优 |
| 6 | [Ascend 910B 环境与迁移](tutorial/05_Ascend910B环境与迁移.md) | 掌握 torch_npu、Triton-Ascend、UB 与 AICore 排错 |
| 7 | [燧原 S60 环境与迁移](tutorial/06_S60环境与迁移.md) | 掌握 torch_gcu、设备监控、grid/warp 与后端稳定性排错 |
| 8 | [十题实战](tutorial/README.md) | 阅读真实实现，理解每项优化为什么有效 |

## 十个任务

| 任务 | 算子 | 主要知识点 | BW1000 | C500 | 昇腾 910B | 燧原 S60 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| [Task 01](tutorial/tasks/01_GroupedTopk.md) | GroupedTopk | 单调变换、Top-k | 3.098× | 3.442× | 2.478× | 1.380× |
| [Task 02](tutorial/tasks/02_FusedMoE.md) | FusedMoE | 规则化 GEMM、路由融合 | 17.915× | 18.943× | 11.446× | 12.987× |
| [Task 03](tutorial/tasks/03_FlexAttention.md) | FlexAttention | 因果掩码、在线 Softmax | 1.657× | 1.247× | 1.225× | 0.807× |
| [Task 04](tutorial/tasks/04_SPLADESparsePooler.md) | SPLADESparsePooler | GEMM 与池化融合 | 2.928× | 2.384× | 1.094× | 0.597× |
| [Task 05](tutorial/tasks/05_RotaryEmbedding.md) | RotaryEmbedding | 广播消除、坐标映射 | 2.599× | 2.387× | 1.869× | 1.271× |
| [Task 06](tutorial/tasks/06_MMEncoderAttention.md) | MMEncoderAttention | 布局直读、流式 K/V | 1.662× | 1.285× | 1.162× | 0.818× |
| [Task 07](tutorial/tasks/07_mhc_post.md) | mhc_post | 小矩阵展开、BF16 | 11.509× | 16.745× | 2.406× | 0.142× |
| [Task 08](tutorial/tasks/08_hc_split_sinkhorn.md) | hc_split_sinkhorn | 片上迭代、小矩阵归一化 | 13.573× | 10.697× | 9.203× | 8.945× |
| [Task 09](tutorial/tasks/09_CentreRandomAugmentation.md) | CentreRandomAugmentation | 随机数语义、刚体变换 | 5.553× | 5.797× | 3.389× | 2.055× |
| [Task 10](tutorial/tasks/10_head_compute_mix_bwd.md) | head_compute_mix_bwd | 链式求导、多路归约 | 1.846× | 1.636× | 1.064× | 1.197× |
| **十项简单合计** | — | 仅作直观比较 | **6.702×** | **7.071×** | **3.752×** | **0.486×** |

上述数据使用同一官方评测器测得，参数为 `warmup=200`、`repeat=500`，10 项在四个平台上均通过正确性检查。S60 的 PyTorch/Triton 合计为 `16.828573/34.657672 ms`，其中 Task07 明显影响简单合计；简单合计只用于直观比较，不代表官方综合评分。完整环境与逐项数据见各平台 README。

## 项目资料

- [比赛说明](比赛说明.md)
- [提交清单](提交清单.md)
- [GitHub 源代码](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels)
