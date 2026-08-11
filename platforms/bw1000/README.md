# 海光 BW1000 平台

本目录保存 KernelSwift 赛道一在海光 BW1000 上完成的 10 个 Triton 优化任务。所有 `solution.py` 均已使用仓库中的官方评测器通过正确性检查。

## 验证环境

- 芯片：海光 BW1000（`gfx936`）
- DTK：26.04
- PyTorch：2.7.1
- Triton：3.1.0
- Docker 容器：`zk-triton-0810`
- Docker 镜像：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.10`
- 镜像 digest：`sha256:07c285c51837d76fbcc73b771b1a95ecf2d8b71203e5feadb3cb9cb12b5d3f4d`
- 官方评测：`warmup=200`、`repeat=500`，结果取中位数

## 使用方法

脚本默认仓库位于 `/data/zk/kernelswift-2026-triton-kernels`。如路径不同，通过 `PROJECT_DIR` 指定：

```bash
cd /data/zk/kernelswift-2026-triton-kernels/platforms/bw1000
PROJECT_DIR=/data/zk/kernelswift-2026-triton-kernels ./docker_create.sh
```

选择通过 `hy-smi` 确认的空闲卡，执行全部任务：

```bash
DEVICE_ID=0 ./run_all_benchmarks.sh
```

根目录 [README](../../README.md) 汇总了优化前后耗时和加速比，各任务目录的中文 README 记录了具体优化方案。

如果是第一次学习 Triton 或算子优化，建议从[从零学习 Triton 算子开发与优化教程](../../docs/tutorial/README.md)开始，教程包含基础原理和 10 个任务的逐题推导。
