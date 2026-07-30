#!/bin/bash
# xe-fuse vs vllm-xpu-kernels: Three-way comparison benchmark
#
# Prerequisites: source your oneAPI environment before running, e.g.:
#   source /opt/intel/oneapi/setvars.sh
#
# Usage:
#   ./run_vllm_comparison.sh                     # default: llama3_8b
#   ./run_vllm_comparison.sh llama3_8b           # specify preset
#   ./run_vllm_comparison.sh gemma2_9b 4096      # preset + sequence length
#   ./run_vllm_comparison.sh llama3_8b 2048 200  # preset + M + iterations

set -e

if ! command -v icpx &>/dev/null; then
    echo "Error: icpx not found. Source your oneAPI environment first." >&2
    exit 1
fi

PRESET="${1:-llama3_8b}"
SEQ_LEN="${2:-2048}"
ITERATIONS="${3:-100}"

SYCL_TLA_DIR="${SYCL_TLA_DIR:?Error: set SYCL_TLA_DIR to your sycl-tla checkout}"
XE_FUSE_DIR="${XE_FUSE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MKL_INCLUDE="${MKLROOT:-/opt/intel/oneapi/mkl/latest}/include"
BUILD_DIR="/tmp/xe_vllm_cmp_$$"
mkdir -p "$BUILD_DIR"

export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export IGC_VectorAliasBBThreshold=100000000000

COMMON_FLAGS="-fsycl \
    -DCUTLASS_ENABLE_SYCL \
    -DSYCL_INTEL_TARGET \
    -I $SYCL_TLA_DIR/include \
    -I $SYCL_TLA_DIR/tools/util/include \
    -I $SYCL_TLA_DIR/examples/common \
    -I $XE_FUSE_DIR/include \
    -I $MKL_INCLUDE \
    -O2 -std=c++17 \
    -fno-sycl-instrument-device-code \
    -fsycl-targets=spir64_gen \
    -Xsycl-target-backend=spir64_gen \"-device bmg-g31\" \
    -Xspirv-translator \"-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate\""

echo "============================================================"
echo "xe-fuse vs vllm-xpu-kernels: Three-Way Comparison"
echo "============================================================"
echo "Node: $(hostname)"
icpx --version 2>&1 | head -1
echo "Preset: $PRESET"
echo "Sequence length: $SEQ_LEN"
echo "Iterations: $ITERATIONS"
echo ""

# Step 1: Generate model-specific C++
echo "=== Generating C++ for preset: $PRESET ==="
GEN_CPP="$BUILD_DIR/test_${PRESET}_comparison.cpp"
cd "$XE_FUSE_DIR/autotune"
uv run generate_pipeline.py --preset "$PRESET" --output "$GEN_CPP"
echo ""

# Step 2: Compile
echo "=== Building comparison test ==="
BUILD_START=$(date +%s)

eval icpx $COMMON_FLAGS \
    -o "$BUILD_DIR/test_comparison" "$GEN_CPP" \
    > "$BUILD_DIR/build.log" 2>&1

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

if [ -f "$BUILD_DIR/test_comparison" ]; then
    spills=$(grep -o "spilled around [0-9]*" "$BUILD_DIR/build.log" 2>/dev/null || true)
    echo "BUILD OK (${BUILD_TIME}s), ${spills:-no spills}"
else
    echo "BUILD FAILED (${BUILD_TIME}s)"
    tail -30 "$BUILD_DIR/build.log"
    exit 1
fi
echo ""

# Step 3: Correctness (small dims)
echo "=== Correctness: M=1024 ==="
"$BUILD_DIR/test_comparison" --m=1024 --iterations=0
echo ""

# Step 4: Benchmark
echo "============================================================"
echo "=== Benchmark: M=${SEQ_LEN}, ${ITERATIONS} iterations ==="
echo "============================================================"
"$BUILD_DIR/test_comparison" --m="$SEQ_LEN" --iterations="$ITERATIONS"
echo ""

echo "============================================================"
echo "Comparison test DONE ($PRESET)"
echo "============================================================"

rm -rf "$BUILD_DIR"
