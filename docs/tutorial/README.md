# 从零学习 Triton 算子开发与优化：四平台十题实战

这套教程面向第一次接触 GPU/DCU 算子、Triton 和性能优化的读者。目标不是教你背诵 10 份答案，而是让你掌握一套可以迁移到新算子的工作方法：读懂参考计算图、找到真正的性能瓶颈、证明变换等价、把计算映射成 Triton program、验证精度并用数据选择参数。

教程中的代码、形状和性能数据来自海光 BW1000、沐曦 C500、Ascend 910B 与燧原 S60 上完成的 10 个任务。建议先用 BW1000 版本理解算法，再对照其他平台观察同一 kernel 在布局、tile、片上资源、grid 限制和编译器稳定性上的差异。

## 阅读顺序

1. [00：先建立算子优化的整体认识](00_整体认识与学习路线.md)
2. [01：BW1000 环境、正确性与性能测量](01_BW1000环境与评测.md)
3. [02：Triton 编程模型与常用语法](02_Triton编程基础.md)
4. [03：从 PyTorch 参考实现到高性能 kernel](03_算子优化方法论.md)
5. [04：沐曦 C500 环境与海光迁移](04_C500环境与海光迁移.md)
6. [05：Ascend 910B 环境与迁移](05_Ascend910B环境与迁移.md)
7. [06：燧原 S60 环境与迁移](06_S60环境与迁移.md)
8. 按下面的任务顺序学习。前几题偏融合与归约，中间进入 GEMM 和 Attention，后几题集中训练小矩阵、随机数和反向归约。

## 十题知识地图

| 任务 | 主要知识点 | 难度 | BW1000 | C500 | 昇腾 910B | 燧原 S60 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Task01 GroupedTopk](tasks/01_GroupedTopk.md) | 单调性、Top-k、行归约、消去 Softmax 分母 | 2/5 | 3.098× | 3.442× | 2.478× | 1.380× |
| [Task02 FusedMoE](tasks/02_FusedMoE.md) | 规则化 GEMM、SiLU 融合、路由、以算代分发 | 5/5 | 17.915× | 18.943× | 11.446× | 12.987× |
| [Task03 FlexAttention](tasks/03_FlexAttention.md) | 因果 Attention、在线 Softmax、流式 K/V | 5/5 | 1.657× | 1.247× | 1.225× | 0.807× |
| [Task04 SPLADE](tasks/04_SPLADESparsePooler.md) | GELU+LayerNorm、GEMM+池化、单调变换 | 5/5 | 2.928× | 2.384× | 1.094× | 0.597× |
| [Task05 RotaryEmbedding](tasks/05_RotaryEmbedding.md) | 广播消除、扁平索引、三角函数融合 | 2/5 | 2.599× | 2.387× | 1.869× | 1.271× |
| [Task06 MMEncoderAttention](tasks/06_MMEncoderAttention.md) | 非因果 Attention、布局直读、在线 Softmax | 4/5 | 1.662× | 1.285× | 1.162× | 0.818× |
| [Task07 mhc_post](tasks/07_mhc_post.md) | 固定小矩阵展开、数据复用、BF16 写回 | 3/5 | 11.509× | 16.745× | 2.406× | 0.142× |
| [Task08 Sinkhorn](tasks/08_hc_split_sinkhorn.md) | 4×4 小矩阵、迭代归一化、寄存器驻留 | 3/5 | 13.573× | 10.697× | 9.203× | 8.945× |
| [Task09 RandomAugmentation](tasks/09_CentreRandomAugmentation.md) | 随机数语义、中心化归约、刚体变换融合 | 4/5 | 5.553× | 5.797× | 3.389× | 2.055× |
| [Task10 backward](tasks/10_head_compute_mix_bwd.md) | 手写反向、多个梯度归约、静态专用化 | 3/5 | 1.846× | 1.636× | 1.064× | 1.197× |
| **十项简单合计** | 仅作直观比较 | — | **6.702×** | **7.071×** | **3.752×** | **0.486×** |

四列数据均使用同一官方评测器和 `warmup=200`、`repeat=500` 参数，10 项正确性全部通过。S60 的 PyTorch/Triton 十项耗时简单合计为 `16.828573/34.657672 ms`；Task07 是很有价值的反例，说明跨后端“能运行”和“能加速”是两件事。完整数据见 [GitHub 上的 S60 平台 README](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/enflame_s60/README.md)。简单合计不代表官方跨任务或跨平台评分。

## 学完后应该具备的能力

- 能把 PyTorch 表达式写成数学公式和明确的张量形状；
- 能区分 launch-bound、memory-bound、compute-bound 和同步开销；
- 能使用 `tl.program_id`、`tl.arange`、mask 和指针算术覆盖任意形状；
- 能实现逐元素、归约、小矩阵、GEMM 和在线 Softmax kernel；
- 能判断哪些中间张量可以融合，哪些运算可以交换或消去；
- 能设计公平 benchmark，理解 warmup、同步、中位数和容差；
- 能围绕 block size、`num_warps`、寄存器压力和访存布局进行有依据的调优。

## 重要边界

本仓库中的实现是比赛固定形状专用 kernel。固定形状专用化本身不是作弊，但它与通用算子开发的目标不同。阅读时应同时回答两个问题：当前固定形状为什么可以这样写；如果形状变化，需要把哪些常量改成运行时参数、哪些循环改成分块、哪些边界补上 mask。
