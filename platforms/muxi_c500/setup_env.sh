#!/usr/bin/env bash

# Source this file from interactive shells, non-interactive SSH commands, or CI:
#   source platforms/muxi_c500/setup_env.sh

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
fi

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export MACA_CLANG_PATH="${MACA_CLANG_PATH:-$MACA_PATH/mxgpu_llvm/bin}"
export PYTORCH_DEFAULT_NCHW="${PYTORCH_DEFAULT_NCHW:-1}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/ompi/lib:$MACA_PATH/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/opt/mxdriver/lib:${LIBRARY_PATH:-}"
export PATH="/opt/conda/bin:$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/ompi/bin:$MACA_PATH/ucx/bin:/opt/mxdriver/bin:$PATH"
