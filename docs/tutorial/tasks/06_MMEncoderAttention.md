# Task06：MMEncoderAttention——非因果 Attention 的规则化在线计算

代码：[reference.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task06_mm_encoder_attention/reference.py) · [solution.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task06_mm_encoder_attention/solution.py) · [benchmark.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task06_mm_encoder_attention/benchmark.py)

核心 kernel：`_attention_kernel`。

## 1. 本题与 Task03 的关系

两题共享在线 Softmax 核心。Task06 是 batch=2 的非因果 Attention，所有 query 都访问完整的 83 个 K/V，因此工作量更规则，可以使用更大的 query tile。

先阅读 [Task03](03_FlexAttention.md) 的在线 Softmax推导，再看本题如何改变布局和 tile。

## 2. 输入布局

```text
Q,K,V: [B,T,H*D] = [2,83,512]
heads=8, D=64
output: [2,83,512]
```

参考实现 view 为 `[B,T,H,D]`，再 transpose 到 `[B,H,T,D]` 传给 SDPA，最后 transpose/reshape 回去。

优化 kernel 不创建 transpose 缓冲区，直接使用原始行主序地址：

```text
offset(b,t,h,d) = b*(83*512) + t*512 + h*64 + d
```

## 3. program 映射

```python
block_m = tl.program_id(0)
batch_head = tl.program_id(1)
batch = batch_head // 8
head = batch_head % 8
```

grid：

```text
(ceil(83/64), B*H) = (2,16)
```

每个 program 负责 64 个 query、一个 batch/head。总共只有 32 个 program，因此每个 program 需要足够工作量来摊薄调度成本。

## 4. Q/K/V tile

```text
Q tile: [64,64]
K tile: [32,64]
score : [64,32]
V tile: [32,64]
acc   : [64,64]
```

K/V 扫描三次，覆盖 `[0,32)、[32,64)、[64,96)`。最后 13 列用 mask 排除。

非因果模式没有 `cols<=m` 条件：

```python
valid = (m[:,None] < 83) & (cols[None,:] < 83)
```

## 5. 在线 Softmax 状态

每个 query 行维护：

```python
row_max: [64] FP32
row_sum: [64] FP32
acc: [64,64] FP32
```

对每个 K/V tile：

```python
scores = tl.dot(q, tl.trans(k)) * scale
block_max = tl.max(scores, axis=1)
new_max = tl.maximum(row_max, block_max)
alpha = tl.exp(row_max - new_max)
probs = tl.exp(scores - new_max[:,None])
row_sum = row_sum * alpha + tl.sum(probs, axis=1)
acc = acc * alpha[:,None] + tl.dot(probs.to(tl.float16), v)
row_max = new_max
```

最终 `acc/row_sum`。整个过程只读 Q 一次，K/V 各流式读一次，不写 score/probability。

## 6. 为什么 BLOCK_M 比 Task03 大

Task03 的因果 mask 会让靠前 query 的大量 score 无效，大 M 会加重浪费；Task06 每行都扫描全部 K/V，64×32 tile 更规则。更大的 M 还可以让同一个 program 加载的 K/V 被 64 个 query 复用。

代价是 `[64,64]` FP32 accumulator 很大，寄存器压力明显更高，所以 `BLOCK_M` 不能无限增大。`num_warps=4` 是实测平衡点。

## 7. 内存布局收益

原始最后一维 `[head,d]` 连续。对固定 head 的 64 个 d，地址连续；对不同 token，stride 为 512。Q/K/V tile 的 `d` 内层连续，有利于合并读取。

输出沿同一 `q_offsets` 写回，天然得到 `[B,T,512]`，不需要额外 transpose kernel。

## 8. 正确性风险

- `scale=1/sqrt(64)` 必须与 reference 相同；
- score mask 必须同时屏蔽尾部 query 和 key；
- 无效 score 填 `-inf`，而 Q/K/V load 越界填 0；
- 在线状态必须 FP32；
- `probs.to(float16)` 是精度折中，需通过容差验证；
- 当前 `num_kv_heads=num_heads=8`，不能直接用于 GQA。

## 9. 结果

```text
PyTorch: 0.141271 ms
Triton : 0.085000 ms
Speedup: 1.662x
```

参考 SDPA 已经很强，因此提升主要来自针对固定 T/D 的 tile 和省去通用布局处理。

## 10. 迁移练习

1. 将 B/T/H/D 和 stride 变为参数；
2. 测 `BLOCK_M={16,32,64}`、`BLOCK_N={16,32,64}`；
3. 用 hipprof 观察不同 M 下的 VGPR 和 occupancy；
4. 支持 `num_kv_heads<heads`，实现 `kv_head=head//group_size`；
5. 与官方 fused attention 教程对照，理解长序列下为何需要更复杂的 pipeline。

## S60 实测补充

海光的 64 行/4 warp 配置在 S60 上约 9.88 ms；调整为 `BLOCK_M=32`、`num_warps=1` 后正式为 0.335449 ms，接近 PyTorch 的 0.274237 ms。
