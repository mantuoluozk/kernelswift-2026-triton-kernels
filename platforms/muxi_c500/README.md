# 沐曦 C500 平台

本目录保存 KernelSwift 赛道一在沐曦 C500 上的 Triton 移植与优化实现。10 项均已使用官方评测参数在 C500 上通过正确性检查。

## 当前验证环境

- 平台：模力方舟容器实例
- 设备：MetaX C500 的 25% sGPU 切片
- 本实例显存配额：16000 MiB；PyTorch 可见约 15584 MiB
- 物理卡显存：65536 MiB；不要把物理卡容量当作当前实例可用容量
- Ubuntu：22.04.3 LTS
- Kernel Mode Driver：3.8.30
- MACA：3.0.0.8
- mcPyTorch：2.4.0+metax3.0.0.3
- mcTriton：3.0.0+metax3.0.0.3
- Python：`/opt/conda/bin/python` 3.10.10
- 工作目录：`/data/zk/kernelswift-2026-triton-kernels`

> mcPyTorch 和 mcTriton 是沐曦适配版本。不要用 PyPI 的普通 `torch` 或 `triton` 覆盖它们。

## 第一次登录

交互式 SSH 会由 `/root/.bashrc` 自动激活 Conda 和 MACA 环境，并进入 `/data`：

```bash
ssh root+实例ID@服务器地址 -p 端口
cd /data/zk/kernelswift-2026-triton-kernels
```

非交互 SSH、CI 或远程脚本不会执行 `.bashrc` 的交互部分，必须显式加载环境：

```bash
source /data/zk/kernelswift-2026-triton-kernels/platforms/muxi_c500/setup_env.sh
```

## 快速检查

```bash
mx-smi
python - <<'PY'
import torch, triton
print(torch.__version__, triton.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))
PY
```

沐曦版 PyTorch 沿用 `torch.cuda` 接口，因此代码中仍使用 `device="cuda"`、`torch.cuda.synchronize()` 和 `CUDA_VISIBLE_DEVICES`，这不代表设备是 NVIDIA。

## 运行评测

短基线：

```bash
cd /data/zk/kernelswift-2026-triton-kernels/platforms/muxi_c500
WARMUP=5 REPEAT=10 ./run_all_benchmarks.sh
```

正式参数：

```bash
WARMUP=200 REPEAT=500 ./run_all_benchmarks.sh
```

单题：

```bash
cd /data/zk/kernelswift-2026-triton-kernels
python evaluator/auto_bench.py \
  --v0_file platforms/muxi_c500/task08_hc_split_sinkhorn/reference.py \
  --v1_file platforms/muxi_c500/task08_hc_split_sinkhorn/solution.py \
  --warmup 200 --repeat 500
```

## C500 正式回归结果

测试参数为 `warmup=200`、`repeat=500`，结果取中位数。

| 任务 | 正确性 | Triton | 加速比 | 迁移情况 |
| --- | --- | ---: | ---: | --- |
| Task01 | PASS | 0.097445 ms | 3.442x | BW1000 实现可复用 |
| Task02 | PASS | 0.153823 ms | 18.943x | BW1000 实现可复用 |
| Task03 | PASS | 0.102738 ms | 1.247x | 算法可复用，tile 仍可调优 |
| Task04 | PASS | 0.411690 ms | 2.384x | 算法可复用，矩阵 tile 仍可调优 |
| Task05 | PASS | 0.094623 ms | 2.387x | BW1000 实现可复用 |
| Task06 | PASS | 0.106350 ms | 1.285x | 算法可复用，tile 仍可调优 |
| Task07 | PASS | 0.243420 ms | 16.745x | 设置 NCHW 后复用高性能 kernel |
| Task08 | PASS | 0.148244 ms | 10.697x | 显式标量 4×4 Sinkhorn |
| Task09 | PASS | 0.166514 ms | 5.797x | BW1000 实现可复用 |
| Task10 | PASS | 0.104272 ms | 1.636x | 算法可复用，归约参数仍可调优 |

十项简单合计：PyTorch `11.519283 ms`，Triton `1.629119 ms`，约 `7.071x`。合计仅用于直观汇总，不代表官方综合评分。

完整的环境、监控、排错和海光迁移对照参见 [C500 环境与迁移指南](C500_环境与迁移指南.md)。
