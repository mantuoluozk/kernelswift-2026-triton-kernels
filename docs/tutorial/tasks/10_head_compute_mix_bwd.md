# Task10：head_compute_mix_bwd——从链式求导到融合反向 kernel

代码：[reference.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task10_head_compute_mix_bwd/reference.py) · [solution.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task10_head_compute_mix_bwd/solution.py) · [benchmark.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task10_head_compute_mix_bwd/benchmark.py)

核心 kernel：`_head_compute_mix_bwd_kernel`。

## 1. 本题训练什么

本题没有调用 autograd，而是给出明确的反向公式。优化目标是在一次遍历中计算逐元素输入梯度、标量 scale 梯度和四通道 base 梯度。

这是学习“逐元素计算 + 多路归约 + 多输出”的好案例。

## 2. 前向与导数

形状：

```text
input_mix: [2,1024,4] FP32，共 8192 元素
scale: [1]
base: [4]
grad_out: [2,1024,4]
```

假设前向：

```text
z = input * scale + base[channel]
y = sigmoid(z)
```

Sigmoid 导数：

```text
dy/dz = y(1-y)
grad_z = grad_out * y * (1-y)
```

链式法则：

```text
grad_input = grad_z * scale
grad_scale = Σ grad_z * input
grad_base[c] = Σ_{channel=c} grad_z
```

## 3. 参考实现的 kernel 链

参考代码依次计算：

- `input*scale + base`；
- sigmoid；
- `sigmoid*(1-sigmoid)`；
- 乘 grad_out；
- grad_input；
- grad_base reduce；
- grad_scale 的逐元素乘与 reduce。

每个阶段读取/写入中间张量。总元素只有 8192，launch 与显存流量占比较高。

## 4. 一维映射与通道计算

固定最后一维为 4，扁平 offset 对应通道：

```python
offsets = tl.arange(0,8192)
channels = offsets & 3
```

因为 4 是 2 的幂，`&3` 等价 `%4`。base 地址为：

```python
base = tl.load(base_ptr + channels)
```

同一个 4 元素 base 会被广播到所有 batch/token。

## 5. 为什么只有一个 program

```python
_kernel[(1,)](..., BLOCK_SIZE=8192, num_warps=8)
```

一个 program 覆盖所有元素，才能直接用 `tl.sum` 得到全局唯一的 scale/base 梯度，无需跨 program 原子加或第二阶段归约。

代价是块很大、寄存器压力高、并行 program 数为 1。当前 n=8192 且只有几个向量状态，实测仍然可行；`num_warps=8` 提供足够块内并行。

若 n 增长到百万，一个 program 不再合理，应采用：

1. 每个 program 计算局部 grad_scale/base；
2. 写入 partial buffer；
3. 第二个 kernel 归约 partial；

或在可接受非确定性时使用原子加。

## 6. 一次加载，多种用途

```python
input_mix = tl.load(...)
grad_out = tl.load(...)
scale = tl.load(scale_ptr)
base = tl.load(base_ptr + channels)

z = input_mix*scale + base
sigmoid = 1/(1+exp(-z))
grad_z = grad_out*sigmoid*(1-sigmoid)
```

`grad_z` 同时用于：

- 写 `grad_input`；
- 累加 `grad_scale`；
- 累加四个 `grad_base`。

不需要把 z、sigmoid 或 grad_z 写回显存。

## 7. 四通道归约

```python
grad_base0 = tl.sum(tl.where(mask & (channels==0), grad_z, 0.0))
...
```

对每个通道，把其他通道变为加法单位元 0，再归约。因为通道只有 4，手工展开清晰且编译期固定。

`grad_scale` 则对全部元素归约：

```python
grad_scale = tl.sum(grad_z * input_mix)
```

## 8. 数值与归约顺序

所有输入和输出都是 FP32，但并行归约树与 PyTorch 可能不同，因此最低位仍可能变化。记录最大绝对误差约 `1.91e-6`。

反向梯度对误差更敏感，建议除随机输入外增加：

- 大正/负 z，检查 sigmoid 饱和；
- scale 接近 0；
- grad_out 全 0/全 1；
- 与有限差分或 autograd 结果交叉验证。

## 9. 结果

```text
PyTorch: 0.176306 ms
Triton : 0.095530 ms
Speedup: 1.846x
```

提升来自融合，但单 program 大归约也限制了进一步加速，所以不如 Task08 的小矩阵融合夸张。

### Ascend 910B 的 UB 分块

8192 元素单 program 在 910B 上编译失败：需要约 320 KiB UB，而后端可用空间约 192 KiB。910B 版本改为两个 4096 元素 program，分别写回输入梯度，并通过 FP32 `tl.atomic_add` 汇总五个参数梯度。正式结果为 `0.359121 → 0.337515 ms`，加速 `1.064×`。

这是“硬件资源改变 program 映射”的典型案例：数学公式完全不变，但 tile 必须缩小，并引入跨 program 归约机制。

## 10. 迁移练习

1. 写两阶段归约版本，找到 n 增大后的交叉点；
2. 用 atomic add 写多 program 版本，比较速度、精度和确定性；
3. 把 channel 数改为 8/16，使用二维 reshape 后沿 batch/token 归约；
4. 使用 `torch.autograd.gradcheck` 的 FP64 小规模版本验证导数；
5. 扫描 `BLOCK_SIZE` 和 `num_warps`，结合 hipprof 查看寄存器压力。
