# Ascend 910B 环境与迁移指南

本指南记录 KernelSwift 十个 Triton 算子从海光 BW1000、沐曦 C500 迁移到 Ascend 910B 的完整过程。重点不是重复 Triton 语法，而是说明在 Ascend 后端上哪些算法能够复用、哪些硬件约束会改变 kernel 设计。

## 软件栈对照

| 海光 BW1000 | 沐曦 C500 | Ascend 910B | 用途 |
| --- | --- | --- | --- |
| DTK / ROCm | MXMACA | CANN | 设备软件栈 |
| `hy-smi` | `mx-smi` | `npu-smi` | 设备监控 |
| ROCm PyTorch | mcPyTorch | PyTorch + torch_npu | 深度学习框架 |
| ROCm Triton | mcTriton | Triton-Ascend | Triton 编译后端 |
| `torch.cuda` | `torch.cuda` 兼容接口 | `torch.npu` | Python 设备接口 |
| LDS / shared memory | LDS / shared memory | UB（Unified Buffer） | 片上工作空间 |

Ascend 不是 CUDA 兼容接口。输入、模型和同步代码应使用 `npu` 设备；Triton kernel 的 `tl.load`、`tl.store`、`tl.dot` 等上层写法仍然相近，但会编译到 Ascend 的 BiShengIR/CANN 工具链。

## 版本选择

实测环境是 CANN 9.0.0、Python 3.11、torch_npu 2.10.0。根据 Triton-Ascend 的版本关系，本项目固定使用 3.2.1：

```bash
python3 -m venv --system-site-packages /data/venvs/kernelswift-ascend910b
/data/venvs/kernelswift-ascend910b/bin/python -m pip install \
  'triton-ascend==3.2.1' \
  --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

仓库已经封装为：

```bash
bash platforms/ascend910b/install_env.sh
source platforms/ascend910b/setup_env.sh
```

选择 `--system-site-packages` 是为了复用实例中与 CANN 配套的 torch_npu，而不是从 PyPI 覆盖厂商环境。Triton-Ascend 及其固定依赖安装在独立虚拟环境中，不污染系统 Python。

实际使用的 openEuler、aarch64、CANN 安装信息、Python 包来源、完整命令和故障排查已单独保存在 [环境安装记录](环境安装记录.md)，方便在同类新实例上逐项复现。

参考：[Triton-Ascend 官方仓库](https://github.com/triton-lang/triton-ascend)、[3.2.1 发布页](https://github.com/triton-lang/triton-ascend/releases/tag/v3.2.1)。

## 在线实例与设备编号

本次资源已经处于预配置容器中，CANN、驱动、torch_npu 和设备节点均已挂载，因此没有再创建嵌套 Docker 容器，也没有可记录的用户自建镜像名。

`npu-smi info` 显示物理设备编号可能不是 0，但容器内 PyTorch 只看到一张逻辑设备：

```python
import torch
import torch_npu

print(torch.npu.device_count())       # 1
print(torch.npu.current_device())     # 0
print(torch.npu.get_device_name())    # Ascend910B1
```

kernel 和输入应使用 `npu:0` 这一容器内逻辑编号，不要把宿主机 `npu-smi` 的物理编号硬编码到提交代码。

## 最小编译链验证

安装成功不等于 kernel 能编译。先运行：

```bash
python platforms/ascend910b/smoke_test.py
```

这个脚本会在 NPU 上编译并执行向量加法，再把结果搬回 CPU 做严格比较。只有它通过后才开始迁移比赛任务。

## 评测器的多后端处理

不同平台的参考文件可能把输入写成 `device="cuda"` 或 `device="npu"`。本项目评测器在加载源码 AST 时，只改写这些精确的设备字符串，不修改计算逻辑：

```text
cuda / npu / mlu / gcu  →  当前实际可用的加速后端
```

同时在可用时导入 `torch_npu`，确保 `torch.npu` 注册完成。正式提交文件仍应使用比赛要求的模型定义和 forward 参数，不依赖异常捕获或 PyTorch fallback 绕过自定义 kernel。

## Task04：超大融合触发 AICore watchdog

C500/BW1000 版本把 decoder GEMM、四段最大池化和激活融合成一个 kernel。相同代码能在 910B 上编译，但执行时长超过 AICore watchdog，报错：

```text
507014: The aicore execution times out
```

最终方案分为两级：

1. 自定义 Triton kernel 融合 GELU 与 LayerNorm；
2. torch_npu 执行 decoder GEMM；
3. 自定义 Triton kernel 按 `BLOCK_N=2048` 融合四段 max 与 `log1p(ReLU)`。

这个案例说明“融合越多越好”不成立。融合后单 program 工作量、program 总数、后端生成的流水和 watchdog 限制必须同时满足。

## Task10：UB 溢出与分块归约

原实现用一个 program 处理 8192 个 FP32 元素并完成五路归约。Ascend 编译器给出的错误是：

```text
ub overflow, requires 2621440 bits while 1572864 bits available
```

也就是需要约 320 KiB，而后端可用 UB 约 192 KiB。最终实现：

- `BLOCK_SIZE=4096`；
- 两个 program 分别计算一半输入；
- 每个 program 直接写回自己的 `grad_input`；
- 使用 FP32 `tl.atomic_add` 汇总一个 scale 梯度和四个 base 梯度。

这比盲目减少 `num_warps` 更有效，因为问题本质是活跃 tile 和中间值占用，不是线程配置本身。

## Attention tile 的重新选择

数学算法可以复用，但目标硬件上的最佳 tile 不同：

- Task03 因果 attention：`BLOCK_M` 从 16 调到 32，正式 Triton 时间为 0.303849 ms；64 行 tile 在稳定短测中更慢，因此回退。
- Task06 非因果 attention：`BLOCK_M` 从 64 调到 32，并使用 `num_warps=1`，正式 Triton 时间为 0.300920 ms。

调参时使用 20 次 warmup、100 次 repeat 比较中位数，只保留差异稳定且正确性通过的配置，最后才运行 200/500。

## 哪些优化可以直接复用

- Task01 的单调性消除 Softmax 分母和 Top-k 融合可以直接复用。
- Task02 的路由、激活、专家 GEMM 与聚合分阶段实现可以直接复用，并得到 11.446×。
- Task05、09 的逐元素/小归约融合可以直接复用。
- Task07 的大张量逐元素融合不需要 C500 的 channels-last 特殊处理。
- Task08 的固定 4×4 标量化 Sinkhorn 可以直接复用，并得到 9.203×。

## 推荐迁移顺序

1. 用 `npu-smi info` 确认设备健康且无其他进程占用。
2. 固定 CANN、torch_npu、Triton-Ascend 版本，不在系统 Python 上直接升级。
3. 运行 `smoke_test.py` 验证真实编译和执行。
4. 使用 `warmup=1/repeat=3` 逐题建立正确性基线。
5. 编译失败先读 UB、IR 和不支持 API 信息；运行超时先缩小融合边界或 tile。
6. 使用 20/100 比较有限数量的有依据参数组合。
7. 最终使用官方 200/500 全量回归，并记录 PyTorch、Triton 和加速比。

## 正式结果

十项全部通过，PyTorch 简单合计 17.297547 ms，Triton 简单合计 4.610002 ms，约 3.752×。逐题数据见 [Ascend 910B 平台 README](README.md)。合计仅用于直观汇总，不代表官方综合评分。
