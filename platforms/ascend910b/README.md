# Ascend 910B 平台

本目录保存 KernelSwift 赛道一在 Ascend 910B 上的 Triton 移植与优化实现。10 项均已在 Ascend 910B1 上使用官方评测参数通过正确性检查。

## 验证环境

- 设备：Ascend 910B1，单卡 64 GiB HBM
- CANN：9.0.0
- `npu-smi`：25.5.1
- Python：3.11.15（aarch64）
- PyTorch：2.10.0
- torch_npu：2.10.0
- Triton-Ascend：3.2.1
- 独立虚拟环境：`/data/venvs/kernelswift-ascend910b`
- 项目目录：`/data/kernelswift-2026-triton-kernels`

Triton-Ascend 3.2.1 与本机 CANN 9.0.0 对应。环境脚本不会把密码、服务器地址或实例标识写入仓库。

## 安装与检查

这类在线实例本身已经是带 CANN、torch_npu 和 NPU 设备映射的容器，因此不再嵌套创建 Docker 容器。第一次使用时执行：

```bash
cd /data/kernelswift-2026-triton-kernels
bash platforms/ascend910b/install_env.sh
source platforms/ascend910b/setup_env.sh
python platforms/ascend910b/smoke_test.py
```

预期输出：

```text
PASS Triton-Ascend vector add
```

查看设备：

```bash
npu-smi info
python - <<'PY'
import torch
import torch_npu
print(torch.__version__, torch_npu.__version__)
print(torch.npu.is_available(), torch.npu.get_device_name())
PY
```

## 运行评测

短回归：

```bash
cd /data/kernelswift-2026-triton-kernels
WARMUP=5 REPEAT=10 bash platforms/ascend910b/run_all_benchmarks.sh
```

官方参数：

```bash
WARMUP=200 REPEAT=500 bash platforms/ascend910b/run_all_benchmarks.sh
```

只运行指定任务：

```bash
TASKS_ONLY="task04_splade_sparse_pooler task10_head_compute_mix_bwd" \
WARMUP=20 REPEAT=100 \
bash platforms/ascend910b/run_all_benchmarks.sh
```

## 正式回归结果

测试参数为 `warmup=200`、`repeat=500`，结果取中位数，精度阈值为 `atol=1e-2`、`rtol=1e-2`。

| 任务 | PyTorch（ms） | Triton（ms） | 加速比 | 正确性 | 910B 迁移要点 |
| --- | ---: | ---: | ---: | --- | --- |
| Task01 · GroupedTopk | 0.701666 | 0.283144 | 2.478× | PASS | 选择与归一化 kernel 直接复用 |
| Task02 · FusedMoE | 6.942960 | 0.606584 | 11.446× | PASS | 分阶段路由与 GEMM 融合直接复用 |
| Task03 · FlexAttention | 0.372326 | 0.303849 | 1.225× | PASS | query tile 调整为 32×32 |
| Task04 · SPLADESparsePooler | 0.721766 | 0.659824 | 1.094× | PASS | decoder 与 Triton 池化拆成两级，避开 watchdog |
| Task05 · RotaryEmbedding | 0.535493 | 0.286519 | 1.869× | PASS | 逐元素融合直接复用 |
| Task06 · MMEncoderAttention | 0.349675 | 0.300920 | 1.162× | PASS | query tile 调整为 32，`num_warps=1` |
| Task07 · mhc_post | 1.980275 | 0.822972 | 2.406× | PASS | 大张量逐元素融合直接复用 |
| Task08 · hc_split_sinkhorn | 3.032301 | 0.329475 | 9.203× | PASS | 4×4 标量化 Sinkhorn 直接复用 |
| Task09 · CentreRandomAugmentation | 2.301964 | 0.679200 | 3.389× | PASS | 随机变换与中心化融合直接复用 |
| Task10 · head_compute_mix_bwd | 0.359121 | 0.337515 | 1.064× | PASS | 8192 单块拆为两个 4096 tile，atomic 汇总 |
| **十项简单合计** | **17.297547** | **4.610002** | **3.752×** | **全部通过** | — |

合计耗时只用于直观汇总，不代表比赛官方综合评分。不同芯片平台应分别提交和排名。

## 关键平台差异

- Ascend 使用 `torch.npu`、`device="npu"` 和 `torch.npu.synchronize()`；评测器会把其他平台源码中的设备字面量自动改写到当前可用后端。
- 910B 的 UB 容量会直接限制单个 program 的 tile。Task10 的 8192 元素单块实现编译时需要约 320 KiB，超过后端报告的 192 KiB 可用空间。
- 超大融合并不总是更快。Task04 的 decoder+pool 单 kernel 会触发 AICore watchdog，拆成 torch_npu GEMM 与 Triton pool 后才稳定运行。
- 第一次执行会完成 Triton 编译和 CANN 代码生成，不能把首次编译时间算进 kernel 性能。

完整排错过程与平台对照见 [Ascend 910B 环境与迁移指南](Ascend910B_环境与迁移指南.md)。
