# Ascend 910B：从 GPU Triton 到 NPU Triton

本章介绍如何把已经在 BW1000、C500 上工作的 Triton kernel 迁移到 Ascend 910B。重点是环境版本、设备接口、UB 容量、AICore watchdog，以及如何用错误信息决定下一步。

## 1. 先认识软件栈

Ascend 上的对应关系是：

```text
PyTorch       → PyTorch + torch_npu
Triton        → Triton-Ascend
GPU runtime   → CANN
cuda:0        → npu:0
torch.cuda    → torch.npu
shared memory → UB（Unified Buffer）
```

Triton Python 层仍使用 `@triton.jit`、`tl.load`、`tl.store`、`tl.dot` 和 `tl.sum`。下层编译链则会经过 Triton IR、Ascend 后端、BiShengIR 和 CANN。

## 2. 为什么版本必须成套

本项目实测组合：

| 组件 | 版本 |
| --- | --- |
| 设备 | Ascend 910B1 |
| CANN | 9.0.0 |
| Python | 3.11.15 |
| PyTorch | 2.10.0 |
| torch_npu | 2.10.0 |
| Triton-Ascend | 3.2.1 |

系统已经提供 CANN 与 torch_npu。Triton-Ascend 安装在带 `--system-site-packages` 的独立虚拟环境中，从而复用厂商 torch_npu，又不覆盖系统包：

```bash
cd /data/kernelswift-2026-triton-kernels
bash platforms/ascend910b/install_env.sh
source platforms/ascend910b/setup_env.sh
```

不要只看 `import triton` 是否成功。必须实际编译 kernel：

```bash
python platforms/ascend910b/smoke_test.py
```

参考：[Triton-Ascend 官方仓库](https://github.com/triton-lang/triton-ascend)。

## 3. 容器内编号与物理编号

在线资源已经是预配置容器。`npu-smi` 看到的物理编号可能不是 0，但 torch_npu 在容器内只暴露一张逻辑设备：

```python
import torch
import torch_npu

print(torch.npu.device_count())
print(torch.npu.current_device())
print(torch.npu.get_device_name())
```

提交代码使用 `npu:0` 或输入 Tensor 自带的 `tensor.device`，不要硬编码宿主机物理编号。

## 4. 建立迁移基线

推荐顺序：

```bash
# 先用极短参数验证正确性和编译
WARMUP=1 REPEAT=3 bash platforms/ascend910b/run_all_benchmarks.sh

# 调参阶段使用更稳定的中位数
WARMUP=20 REPEAT=100 bash platforms/ascend910b/run_all_benchmarks.sh

# 最后使用官方参数
WARMUP=200 REPEAT=500 bash platforms/ascend910b/run_all_benchmarks.sh
```

只测某几题：

```bash
TASKS_ONLY="task03_flex_attention task10_head_compute_mix_bwd" \
WARMUP=20 REPEAT=100 \
bash platforms/ascend910b/run_all_benchmarks.sh
```

第一次运行包含编译，不能拿首次耗时评价 kernel 性能。

## 5. 读懂 UB overflow

Task10 原实现让一个 program 同时处理 8192 个 FP32 元素并做五路归约。Ascend 编译器报告：

```text
ub overflow, requires 2621440 bits while 1572864 bits available
```

换算后约为：

```text
需要 320 KiB
可用 192 KiB
```

解决办法不是只改 `num_warps`，而是减少每个 program 同时存活的数据：

```text
8192 元素单 program
        ↓
2 个 program × 4096 元素
        ↓
局部归约 + FP32 atomic 汇总
```

最终 Task10 正确性通过，正式 Triton 时间为 0.337515 ms。

## 6. 读懂 AICore watchdog

Task04 的超大融合 kernel 可以编译，但执行时触发：

```text
507014: The aicore execution times out
```

这说明单个 kernel 的执行路径过重或后端生成代码不适合当前融合方式。最终改成：

```text
Triton: GELU + LayerNorm
torch_npu: decoder GEMM
Triton: 四段 max + log1p(ReLU)
```

两级实现比“强行全融合”更稳定。这里仍真实执行两个自定义 Triton kernel，不是绕过自定义算子的 fallback。

## 7. Attention 为什么要重新选 tile

同一在线 Softmax 算法在不同后端上的最优参数不同：

| 任务 | 910B 选择 | 结果 |
| --- | --- | ---: |
| Task03 因果 Attention | `BLOCK_M=32`、`BLOCK_N=32`、`num_warps=4` | 0.303849 ms |
| Task06 非因果 Attention | `BLOCK_M=32`、`BLOCK_N=32`、`num_warps=1` | 0.300920 ms |

Task03 测过 16、32、64 行 tile；32 最快。Task06 的 32 行 tile 略优于 64。只保留有稳定测量依据的参数，不能因为某个平台的配置好就直接照搬。

## 8. 三类迁移结论

### 直接复用

Task01、02、05、07、08、09 的数学变换和主要 kernel 结构可以直接复用。

### 重新调参

Task03、06 的 Attention 算法不变，但 query tile 和 `num_warps` 需要重测。

### 改变融合边界或分块

Task04 需要拆开超大融合，Task10 需要因 UB 容量改为多 program + atomic。

## 9. 正式结果

| 任务 | PyTorch（ms） | Triton（ms） | 加速比 |
| --- | ---: | ---: | ---: |
| Task01 | 0.701666 | 0.283144 | 2.478× |
| Task02 | 6.942960 | 0.606584 | 11.446× |
| Task03 | 0.372326 | 0.303849 | 1.225× |
| Task04 | 0.721766 | 0.659824 | 1.094× |
| Task05 | 0.535493 | 0.286519 | 1.869× |
| Task06 | 0.349675 | 0.300920 | 1.162× |
| Task07 | 1.980275 | 0.822972 | 2.406× |
| Task08 | 3.032301 | 0.329475 | 9.203× |
| Task09 | 2.301964 | 0.679200 | 3.389× |
| Task10 | 0.359121 | 0.337515 | 1.064× |
| **简单合计** | **17.297547** | **4.610002** | **3.752×** |

10 项均通过官方评测器正确性检查。简单合计只用于直观比较，不代表比赛官方综合评分。

## 10. 排错清单

1. `npu-smi info` 确认设备健康、无其他计算进程。
2. 检查 CANN、torch_npu、Triton-Ascend 版本是否对应。
3. 先跑向量加法 smoke test，再跑比赛任务。
4. `device="cuda"`、`.is_cuda`、`torch.cuda.synchronize()` 都要改为后端无关写法或 NPU 接口。
5. 看到 UB overflow 时缩小 tile 或拆成两阶段归约。
6. 看到 AICore timeout 时缩小单 kernel 工作量或调整融合边界。
7. 正确性通过后再比较 20/100 中位数，最后跑 200/500。
