# 02：Triton 编程模型与常用语法

## 1. 最小 kernel

下面是 Triton 官方向量加法教程的核心结构：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)

grid = (triton.cdiv(n, BLOCK),)
add_kernel[grid](x, y, out, n, BLOCK=256, num_warps=4)
```

必须理解五件事：

1. `@triton.jit` 把 Python 子集编译为设备 kernel；
2. `grid` 决定启动多少个 program 实例；
3. `tl.program_id(0)` 是当前 program 在 grid 第 0 维的编号；
4. `tl.arange` 构造一个块内索引张量；
5. mask 保护最后一个不完整块，避免越界。

官方入门示例：[Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)。

## 2. program、grid 和块张量

假设 `n=1000, BLOCK=256`，grid 是 4。四个 program 分别处理：

```text
pid 0 → [0, 256)
pid 1 → [256, 512)
pid 2 → [512, 768)
pid 3 → [768, 1024)，其中 [1000, 1024) 被 mask
```

二维 grid 同理。Task02 的 `_gate_up_kernel` 使用：

```python
grid = (ceil_div(83, 32), 8)
block_m = tl.program_id(0)
expert = tl.program_id(1)
```

第一维切 token，第二维枚举专家。

## 3. 指针算术和广播

Triton 不依赖高级索引语法，核心是计算地址。二维 tile 常写为：

```python
rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
ptrs = base + rows[:, None] * stride_m + cols[None, :] * stride_n
x = tl.load(ptrs, mask=(rows[:, None] < M) & (cols[None, :] < N))
```

`[:, None]` 和 `[None, :]` 触发块张量广播，得到 `[BLOCK_M, BLOCK_N]` 地址矩阵。先在纸上写出目标元素 `X[i,j]` 的线性地址，再翻译成 Triton，最不容易出错。

## 4. `tl.constexpr` 与专用化

标记为 `tl.constexpr` 的参数在编译时已知，可以用于：

- 块张量形状；
- `tl.static_range` 循环边界；
- 编译期分支；
- 循环展开和常量传播。

```python
def kernel(..., BLOCK: tl.constexpr, N: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
```

比赛固定形状适合专用化，但每组不同 constexpr 可能触发一个新编译版本。通用库不能无限枚举形状。

## 5. mask 和 `other`

加载越界数据时，`other` 的选择必须符合后续运算的单位元：

| 后续运算 | 越界填充值 |
| --- | --- |
| 求和、点积 | `0.0` |
| 求最大值 | `-inf` |
| 求最小值 | `+inf` |

Task03 的无效 Attention score 填 `-inf`，这样 `exp(-inf)=0`，不会影响 Softmax。Task04 的无效 token 同样填 `-inf`，不会污染 max pooling。

## 6. 逐元素计算与类型

常用操作：

```python
tl.exp(x)
tl.log(x)
tl.sqrt(x)
tl.rsqrt(x)
tl.sigmoid(x)
tl.sin(x)
tl.cos(x)
tl.where(mask, a, b)
x.to(tl.float16)
```

一般策略是输入/输出保持题目数据类型，累加和归一化状态尽量使用 FP32。Task03/06 的 Attention accumulator、row max 和 row sum 都是 FP32，矩阵点积输入则转 FP16 以利用矩阵计算路径。

## 7. 归约

```python
row_sum = tl.sum(x, axis=1)
row_max = tl.max(x, axis=1)
index = tl.argmax(x, axis=0)
```

归约会把一个维度压缩掉。例如 `[M,N]` 沿 `axis=1` 归约得到 `[M]`。DTK 最佳实践把归约描述为“分块求局部结果，再合并为最终结果”；Triton 编译器替你生成底层并行归约，但块大小、数据类型和寄存器压力仍由你的程序形状决定。

## 8. `tl.dot` 与分块 GEMM

矩阵乘：

```text
C[M,N] = A[M,K] @ B[K,N]
```

典型 Triton 结构：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
for k0 in tl.static_range(0, K, BLOCK_K):
    a = tl.load(a_ptrs_for_this_k_tile)
    b = tl.load(b_ptrs_for_this_k_tile)
    acc += tl.dot(a, b)
tl.store(c_ptrs, acc, mask=...)
```

Task02、Task03、Task04 和 Task06 都使用这一模式。官方矩阵乘教程详细解释了多维地址、K 分块和 program 排序：[Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。

## 9. Softmax 的稳定形式

直接计算 `exp(x)` 可能溢出，稳定形式是：

```text
m = max(x)
p_i = exp(x_i - m)
y_i = p_i / sum(p)
```

如果整行能放进片上状态，可以一次加载、归约并写回。官方 fused softmax 教程强调块大小通常取不小于列数的 2 的幂，并用 mask 填充：[Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。

当一整行放不下时，需要在线 Softmax，Task03/06 会详细推导。

## 10. `num_warps` 和资源平衡

`num_warps` 不是“越大越快”按钮。增加它可能提高块内并行度，也可能增加寄存器/调度开销或减少同时驻留的 program。DTK 手册指出寄存器和共享内存会限制 CU 上活跃 wave 数，因此参数必须实测。

本项目的结果很能说明问题：

| 模式 | 最优例子 |
| --- | --- |
| 小行归约/小矩阵 | Task01、08、09：`num_warps=1` |
| 中型 tile/GEMM | Task02、04、06、07：`num_warps=4` |
| 大一维块 | Task05、10：`num_warps=8` |

这只是当前形状和编译器的测量结果，不是通用规则。
