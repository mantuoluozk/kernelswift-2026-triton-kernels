# Task05：MusicFlamingoRotaryEmbedding——把广播图还原成坐标公式

> 本章以海光 BW1000 的实现和实测数据为主线。其他芯片的参数、限制与结果统一放在对应平台迁移章节中。

代码：[reference.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task05_music_flamingo_rotary_embedding/reference.py) · [solution.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task05_music_flamingo_rotary_embedding/solution.py) · [benchmark.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task05_music_flamingo_rotary_embedding/benchmark.py)

核心 kernel：`_rotary_embedding_kernel`。

## 1. 本题训练什么

这是一道很适合新手的逐元素融合题。重点不是复杂算法，而是把 `arange`、`repeat_interleave`、broadcast、concat、乘法、cos 和 sin 组成的框架图，还原为“每个输出元素如何由坐标直接计算”。

## 2. 输出形状从哪里来

固定配置：

```text
batch=4, seq=32, dim=64
batch frequency width=64
time frequency width=64
concat width=128
cos/sin output=[4,32,128]
```

`inv_freq` 只有 32 个值，因为每个频率会 repeat 两次。

## 3. 参考实现的中间张量

参考路径会创建/广播：

- batch positions `[4]`；
- batch freqs `[4,64]`；
- time freqs `[32,64]`；
- 两个广播视图 `[4,32,64]`；
- concat 后 freqs `[4,32,128]`；
- angle `[4,32]`；
- phase `[4,32,128]`；
- cos 和 sin 两个输出。

虽然元素数只有 16384，但包含多个小操作和中间张量，典型的 launch/memory-bound 链。

## 4. 从扁平 offset 反解三维坐标

输出最后一维 128 连续。对扁平索引 `offset`：

```python
feature = offset & 127       # 等价 offset % 128
token = offset // 128
time_index = token % 32
batch_index = token // 32
```

因为 128 是 2 的幂，`&127` 可表达取模；编译器对 `%128` 通常也会优化。教程中更重要的是理解：

```text
offset = (batch * seq + time) * 128 + feature
```

从这个等式就能逐层反解坐标。

## 5. repeat_interleave 如何消除

每个 `inv_freq[j]` 对应两个相邻 feature，所以：

```python
inv_index = (feature & 63) // 2
```

`feature&63` 把 concat 的左右两半都映射到 `[0,63]`，再除以 2 得到 `[0,31]` 的频率索引。不需要真正构造 repeat 后的 64 维张量。

## 6. 左右两半的频率公式

左半（feature<64）来自 batch position：

```text
batch_frequency = batch_index / max_seq_len * inv_freq
```

右半来自预计算 time position angles：

```text
time_frequency = time_index/max_seq_len * 2π * inv_freq
```

统一选择：

```python
frequency = tl.where(feature < 64, batch_frequency, time_frequency)
phase = frequency * (-timestamp * 2π)
```

然后同一 kernel 同时写 `cos(phase)` 和 `sin(phase)`。两个输出共享地址计算、频率加载和 phase 计算。

## 7. grid 与 block

总元素 `4*32*128=16384`，`BLOCK=1024`，grid 为 16。每个 program 处理连续 1024 个输出元素，因此 load/store 连续。

固定元素数正好整除 block，但代码仍保留 `mask=offset<n_elements`，这是好习惯，也便于未来改形状。

实测 `num_warps=8`。这里每 program 有 1024 个独立元素且三角函数计算较重，比 Task01 的小行归约更能利用较多并行资源。

## 8. 为什么保留 `position_angles` buffer

优化 kernel 实际可由 `time_index` 和 `inv_freq` 直接计算，不读取 `position_angles`。但 reference 模型注册了该 buffer，官方可能通过 `state_dict` 加载。保留同名 buffer 是接口兼容，而不是性能需要。

这是比赛模型优化的重要原则：不参与快速路径的参数/buffer 仍可能属于外部契约。

## 9. 正确性注意点

- `2π` 常量精度要足够；
- timestamps dtype 与输出 freqs dtype 的转换顺序要对应 reference；
- concat 左右半的顺序不能交换；
- `seq_len`、batch 和 dim 当前写死，构造函数必须明确限制或实现通用 grid；
- 三角函数对大 phase 的误差可能放大，不能只测随机小范围。

当前测试输出逐位一致，`max_abs_diff=0`。

## 10. 结果与收益

```text
PyTorch: 0.254340 ms
Triton : 0.097860 ms
Speedup: 2.599x
```

收益来自消除 repeat/broadcast/concat 中间图，并复用一次 phase 计算产生 cos/sin。

## 11. 迁移练习

1. 把 `batch/seq/dim` 改为运行时参数；
2. 用多维 grid 替代扁平 grid，比较地址计算与性能；
3. 只输出 cos 或 sin，测量共享计算对双输出的收益；
4. 测试非常大的 timestamp，研究三角函数精度；
5. 尝试读取预计算 `position_angles`，比较“多一次显存读取”和“现场计算”的取舍。
