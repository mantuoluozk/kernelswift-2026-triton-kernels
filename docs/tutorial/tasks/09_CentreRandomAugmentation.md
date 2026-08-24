# Task09：CentreRandomAugmentation——优化随机算子首先要保持随机语义

代码：[reference.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task09_centre_random_augmentation/reference.py) · [solution.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task09_centre_random_augmentation/solution.py) · [benchmark.py](https://github.com/mantuoluozk/kernelswift-2026-triton-kernels/blob/main/platforms/bw1000/task09_centre_random_augmentation/benchmark.py)

核心 kernel：`_centre_random_augmentation_kernel`。

## 1. 本题训练什么

随机算子的正确性不仅由公式决定，还由随机数调用次数、形状、顺序和 seed 决定。本题先在 PyTorch 中保持与 reference 完全相同的 RNG 调用边界，再把中心化、四元数旋转、平移和 mask 融合到 Triton。

## 2. 输入输出

```text
coords: [256,3] FP32
mask: [256] FP32
n_sample=4
output: [4,256,3]
```

每个 sample 使用一组随机旋转矩阵 `R[3,3]` 和平移 `t[3]`：

```text
center = Σ(mask_i * p_i) / (Σmask_i + eps)
p'_i = p_i - center
out[s,i] = mask_i * (R_s p'_i + t_s)
```

## 3. 随机旋转的生成

参考实现用三组均匀随机数 `u1,u2,u3` 构造单位四元数：

```text
qx = sqrt(1-u1) * sin(2πu2)
qy = sqrt(1-u1) * cos(2πu2)
qz = sqrt(u1)   * sin(2πu3)
qw = sqrt(u1)   * cos(2πu3)
```

再把四元数展开为 3×3 旋转矩阵。

## 4. 为什么不在 Triton 内直接生成随机数

官方 harness 会在 reference 和 solution 调用前重置 seed。要得到相同随机结果，solution 必须消耗与 reference 相同的随机流。

因此保留：

```python
u1 = torch.rand(4)
u2 = torch.rand(4)
u3 = torch.rand(4)
translation = torch.randn(4,3)
```

调用边界、顺序、shape 和 dtype 都与 reference 一致。随后 Triton 只消费生成好的随机数。

如果把三次 rand 合并为一次 `[3,4]`，即使元素数相同，也不应未经验证地假定 RNG 流和布局完全一致。随机算子优化的第一原则是复现随机语义。

## 5. program 映射

grid 为 `(n_sample,)=(4,)`。每个 program 处理一个增强样本：

```python
sample = tl.program_id(0)
atom = tl.arange(0,256)
```

为什么不是一个 program 一个 atom？因为同一 sample 的 256 个 atom 共享：

- 同一个 center；
- 同一个旋转矩阵；
- 同一个 translation。

让一个 program 覆盖整个 sample，可以在归约后复用这些标量。

## 6. 中心化归约

坐标按 AoS 布局 `[x0,y0,z0,x1,y1,z1,...]`：

```python
px = tl.load(coords + atom*3)
py = tl.load(coords + atom*3 + 1)
pz = tl.load(coords + atom*3 + 2)
```

分别计算：

```python
denom = tl.sum(mask) + eps
cx = tl.sum(px*mask)/denom
cy = tl.sum(py*mask)/denom
cz = tl.sum(pz*mask)/denom
```

这三个归约共享 mask。由于输入固定 256，不需要尾部 mask。

AoS 的每个坐标分量访问 stride=3，并非最理想的连续访问；但输入小，且保持外部布局可避免转置。若处理大量点云，SoA `[3,N]` 可能更利于合并读取。

## 7. 四元数到旋转矩阵的展开

kernel 先计算 `xx,yy,zz,xy,xz,yz,wx,wy,wz`，再直接代入三个输出公式。没有显式创建 `[3,3]` 矩阵，也没有广播到 `[256,3,3]`。

例如：

```text
ox = (1-2(yy+zz))*px + 2(xy-wz)*py + 2(xz+wy)*pz + tx
```

`qx...qw` 和旋转系数对一个 sample 的所有 atom 复用。

## 8. 为什么仍然有四次 PyTorch RNG kernel

随机数生成没有被融合，所以最终不是单 kernel 全包。但这些调用属于不可随意改变的语义边界。优化集中在原参考实现后续的 stack、reshape、expand、中心化、矩阵构造、旋转和平移链。

如果评测只要求统计分布而不要求 seed 对齐，才可能换用 Triton RNG；本比赛按确定性输出比较，不能这样做。

## 9. mask 的双重作用

mask 同时用于：

1. 计算有效原子的中心；
2. 最终把无效原子的增强结果归零。

只在中心化时用 mask、忘记最后乘 mask，会在 padding 原子位置留下 translation，结果错误。

## 10. 参数与专用化

当前 kernel 明确要求：

```text
coords.shape == [256,3]
n_sample == 4
centre_only == False
mask is not None
```

`num_warps=1` 实测最好。program 数只有 4 看似很少，但每个 program 包含 256 元素归约和三路变换；增加 warps 没有改善当前后端的平衡。

## 11. 正确性测试

benchmark 在每次调用前：

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

然后分别执行 reference 和 solution。记录最大绝对误差约 `2.38e-7`。

性能循环也每轮重置 seed，因此比较包含相同的 RNG 行为。若只在正确性阶段重置、性能阶段不重置，虽然时间仍可测，但无法保证两边执行相同随机路径。

## 12. 结果

```text
PyTorch: 0.934912 ms
Triton : 0.168371 ms
Speedup: 5.553x
```

## 13. 迁移练习

1. 将 coords 改成 `[B,N,3]`，设计 sample 与 batch 的 grid；
2. 支持 `mask=None` 与 `centre_only=True`，避免运行时 Python fallback；
3. 比较 AoS 和 SoA 布局的带宽；
4. 故意合并三次 rand，观察固定 seed 下结果是否变化；
5. 测试 mask 全 0、部分 0 和 N 非 2 次幂的边界。

## S60 实测补充

中心化、随机旋转和平移融合正式为 `2.924597 → 1.422888 ms`（2.055×），同时通过固定 seed 的输出检查。
