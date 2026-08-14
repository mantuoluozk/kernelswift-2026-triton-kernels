# Task04：SPLADESparsePooler

## 算子说明

执行 MLM head，并对四个不等长序列进行 `log1p(ReLU(x))` 激活和词表维度的最大池化。

## 优化方案

- 保留与参考模型一致的参数名称和形状，兼容官方 `state_dict` 注入。
- 将精确 GELU 和 LayerNorm 融合为一个 Triton kernel。
- decoder 投影交给 torch_npu 的矩阵乘实现，避免超大融合 kernel 触发 AICore watchdog。
- 使用 Triton 在词表维度分块，并融合四段 token 最大归约与最终激活。
- 利用 `log1p(ReLU(x))` 的单调性，先对 logits 求最大值，再对池化结果执行激活。
- 910B 版本保留完整 decoder logits，但消除了激活后的中间张量和四次独立池化。

## Ascend 910B 正式结果

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 0.721766 ms | 0.659824 ms | 1.094x |

关键配置：池化 `BLOCK_N=2048`、`num_warps=4`。尝试融合 decoder 的版本会触发 507014 AICore 超时，因此采用两级实现。

## 复现

```bash
cd /data/kernelswift-2026-triton-kernels/platforms/ascend910b/task04_splade_sparse_pooler
source ../setup_env.sh
python benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task04 从零教程](../../../docs/tutorial/tasks/04_SPLADESparsePooler.md)。
