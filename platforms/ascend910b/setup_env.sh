#!/usr/bin/env bash

# Source this file before running the Ascend 910B benchmarks:
#   source platforms/ascend910b/setup_env.sh

KS_ASCEND_VENV="${KS_ASCEND_VENV:-/data/venvs/kernelswift-ascend910b}"

if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

if [[ ! -x "$KS_ASCEND_VENV/bin/python" ]]; then
    echo "Ascend virtual environment not found: $KS_ASCEND_VENV" >&2
    echo "Run platforms/ascend910b/install_env.sh first." >&2
    return 1 2>/dev/null || exit 1
fi

export PATH="$KS_ASCEND_VENV/bin:$PATH"
export PYTHONNOUSERSITE=1
