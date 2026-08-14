#!/usr/bin/env bash
set -euo pipefail

VENV="${KS_ASCEND_VENV:-/data/venvs/kernelswift-ascend910b}"
TRITON_ASCEND_VERSION="${TRITON_ASCEND_VERSION:-3.2.1}"

python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install \
    --disable-pip-version-check \
    "triton-ascend==$TRITON_ASCEND_VERSION" \
    --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi

"$VENV/bin/python" - <<'PY'
import torch
import torch_npu
import triton

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("triton:", triton.__version__)
print("device:", torch.npu.get_device_name())
PY
