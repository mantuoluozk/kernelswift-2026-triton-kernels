# Task04：SPLADESparsePooler

## 算子说明

执行 MLM head，并对四个不等长序列进行 `log1p(ReLU(x))` 激活和词表维度的最大池化。

## 优化方案

- 保留与参考模型一致的参数名称和形状，兼容官方 `state_dict` 注入。
- 将精确 GELU 和 LayerNorm 融合为一个 Triton kernel。
- 使用 FP16 tensor-core dot 直接计算 decoder 投影，并在寄存器中完成每段 token 的最大归约。
- 利用 `log1p(ReLU(x))` 的单调性，先对 logits 求最大值，再对池化结果执行激活。
- 避免生成和再次读取完整的 `[83, 30522]` 激活张量。

## BW1000 迁移基线

| 正确性 | 优化前 PyTorch | 优化后 Triton | 提升比例 |
| --- | ---: | ---: | ---: |
| 通过 | 0.903172 ms | 0.308420 ms | 2.928x |

关键配置：`BLOCK_N=64`、`BLOCK_K=64`、`num_warps=4`。本地最大绝对误差约 `5.4e-4`。

## 复现

```bash
cd /data/zk/kernelswift-2026-triton-kernels/platforms/muxi_c500/task04_splade_sparse_pooler
CUDA_VISIBLE_DEVICES=0 python3 benchmark.py --warmup 200 --repeat 500
```

详细原理与代码拆解：[Task04 从零教程](../../../docs/tutorial/tasks/04_SPLADESparsePooler.md)。

