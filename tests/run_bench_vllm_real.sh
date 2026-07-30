#!/bin/bash
# Real vllm-xpu-kernels standalone ops benchmark (Python + torch)
#
# Prerequisites: source your oneAPI environment before running, e.g.:
#   source /opt/intel/oneapi/setvars.sh
#
# Usage:
#   ./run_bench_vllm_real.sh                  # default: llama3_8b
#   ./run_bench_vllm_real.sh --all            # all presets
#   ./run_bench_vllm_real.sh --preset qwen25_7b --m 4096

set -e

if ! command -v icpx &>/dev/null; then
    echo "Error: icpx not found. Source your oneAPI environment first." >&2
    exit 1
fi

SYCL_TLA_DIR="${SYCL_TLA_DIR:?Error: set SYCL_TLA_DIR to your sycl-tla checkout}"
XE_FUSE_DIR="${XE_FUSE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="$XE_FUSE_DIR/tests"
VENV_DIR="$XE_FUSE_DIR/.venv"

export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"

echo "============================================================"
echo "Real vllm-xpu-kernels Standalone Ops Benchmark"
echo "============================================================"
echo "Node: $(hostname)"
echo "Python: $VENV_DIR/bin/python"
echo "Args: $@"
echo ""

$VENV_DIR/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'XPU: {torch.xpu.get_device_name(0)}')"
$VENV_DIR/bin/python -c "import vllm_xpu_kernels._C; print('vllm-xpu-kernels: loaded OK')"
echo ""

BENCH_ARGS="${1:---all}"
$VENV_DIR/bin/python "$SCRIPT_DIR/bench_vllm_real.py" $BENCH_ARGS

echo ""
echo "============================================================"
echo "Real vllm benchmark DONE"
echo "============================================================"
