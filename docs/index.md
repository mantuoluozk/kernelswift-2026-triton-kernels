---
hide:
  - navigation
  - toc
---

<section class="ks-hero">
  <div class="ks-kicker">KERNELSWIFT · BW1000 FIELD NOTES</div>
  <h1>从 PyTorch 计算图<br>走到 Triton kernel</h1>
  <p>一套写给算子开发初学者的中文实战教程。用海光 BW1000 上的 10 个真实任务，讲清楚公式、数据布局、program 映射、融合、归约、精度与性能测量。</p>
  <div class="ks-actions">
    <a class="ks-button ks-button-primary" href="tutorial/00_整体认识与学习路线/">从第 0 章开始</a>
    <a class="ks-button" href="tutorial/">查看学习地图</a>
  </div>
</section>

<div class="ks-pipeline" aria-label="算子优化流程">
  <div><span>01</span><strong>读语义</strong><small>公式与契约</small></div>
  <div><span>02</span><strong>看布局</strong><small>shape 与 stride</small></div>
  <div><span>03</span><strong>找瓶颈</strong><small>launch / memory / compute</small></div>
  <div><span>04</span><strong>写 kernel</strong><small>grid、tile、mask</small></div>
  <div><span>05</span><strong>用数据验证</strong><small>精度、耗时、回归</small></div>
</div>

## 先把基础打牢

<div class="grid cards ks-learning-cards" markdown>

-   **00 · 整体认识**

    把算子拆成语义、布局、执行和性能四层视图，建立第一套优化闭环。

    [开始阅读 →](tutorial/00_整体认识与学习路线.md)

-   **01 · BW1000 评测**

    学会选空闲卡、同步计时、warmup、中位数、容差和 hipprof 分析。

    [进入环境篇 →](tutorial/01_BW1000环境与评测.md)

-   **02 · Triton 基础**

    从 `program_id`、`arange` 和 mask，走到归约、`tl.dot` 与在线 Softmax。

    [进入编程篇 →](tutorial/02_Triton编程基础.md)

-   **03 · 优化方法论**

    判断 launch、访存和计算瓶颈，用数学等价变换决定哪些工作根本不用做。

    [进入方法篇 →](tutorial/03_算子优化方法论.md)

</div>

## 十题不是十个答案，而是一张能力矩阵

<div class="ks-task-grid">
  <a href="tutorial/tasks/01_GroupedTopk/"><b>01</b><span>GroupedTopk</span><em>单调性 · Top-k</em><strong>3.098×</strong></a>
  <a href="tutorial/tasks/02_FusedMoE/"><b>02</b><span>FusedMoE</span><em>规则化 GEMM · 路由</em><strong>17.915×</strong></a>
  <a href="tutorial/tasks/03_FlexAttention/"><b>03</b><span>FlexAttention</span><em>因果 · 在线 Softmax</em><strong>1.657×</strong></a>
  <a href="tutorial/tasks/04_SPLADESparsePooler/"><b>04</b><span>SPLADE</span><em>GEMM · 池化融合</em><strong>2.928×</strong></a>
  <a href="tutorial/tasks/05_RotaryEmbedding/"><b>05</b><span>RotaryEmbedding</span><em>广播消除 · 坐标</em><strong>2.599×</strong></a>
  <a href="tutorial/tasks/06_MMEncoderAttention/"><b>06</b><span>MM Attention</span><em>布局直读 · 流式 K/V</em><strong>1.662×</strong></a>
  <a href="tutorial/tasks/07_mhc_post/"><b>07</b><span>mhc_post</span><em>4×4 展开 · BF16</em><strong>11.509×</strong></a>
  <a href="tutorial/tasks/08_hc_split_sinkhorn/"><b>08</b><span>Sinkhorn</span><em>片上迭代 · 小矩阵</em><strong>13.573×</strong></a>
  <a href="tutorial/tasks/09_CentreRandomAugmentation/"><b>09</b><span>RandomAug</span><em>RNG 语义 · 刚体变换</em><strong>5.553×</strong></a>
  <a href="tutorial/tasks/10_head_compute_mix_bwd/"><b>10</b><span>Backward</span><em>链式求导 · 多路归约</em><strong>1.846×</strong></a>
</div>

!!! note "教程与代码保持同源"
    网站直接从仓库 `docs/` 中的 Markdown 构建。修改教程并推送到 `main` 后，GitHub Actions 会重新构建并发布；无需复制内容到另一个网站后台。
