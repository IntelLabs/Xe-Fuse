#!/bin/bash
# xe-fuse: compile and benchmark a generated kernel
#
# Prerequisites: source your oneAPI environment before running, e.g.:
#   source /opt/intel/oneapi/setvars.sh
#
# Usage:
#   ./run_kernel.sh <source.cpp> [extra_args...]
#
# Output is structured for machine parsing:
#   BUILD: OK|FAILED
#   SPILLS: <count>|none
#   <kernel_name>: [<tflops>]TFlop/s  (<time>)ms

set -e

if ! command -v icpx &>/dev/null; then
    echo "Error: icpx not found. Source your oneAPI environment first." >&2
    exit 1
fi

KERNEL_SRC="${1:?Usage: ./run_kernel.sh <source.cpp> [args...]}"
shift
EXTRA_ARGS="$@"

SYCL_TLA_DIR="${SYCL_TLA_DIR:?Error: set SYCL_TLA_DIR to your sycl-tla checkout}"
XE_FUSE_DIR="${XE_FUSE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MKL_INCLUDE="${MKLROOT:-/opt/intel/oneapi/mkl/latest}/include"
BUILD_DIR="/tmp/xe_fuse_autotune_$$"
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

BINARY="$BUILD_DIR/kernel"

echo "=== xe-fuse kernel ==="
echo "Source: $KERNEL_SRC"
echo "Node: $(hostname)"
echo ""

# Build
echo "--- Compiling ---"
eval icpx $COMMON_FLAGS -o "$BINARY" "$KERNEL_SRC" > "$BUILD_DIR/build.log" 2>&1
BUILD_RC=$?

if [ $BUILD_RC -ne 0 ]; then
    echo "BUILD: FAILED"
    cat "$BUILD_DIR/build.log"
    rm -rf "$BUILD_DIR"
    exit 1
fi

echo "BUILD: OK"
SPILLS=$(grep -o "spilled around [0-9]*" "$BUILD_DIR/build.log" 2>/dev/null | head -1 || true)
if [ -n "$SPILLS" ]; then
    echo "SPILLS: $(echo $SPILLS | grep -o '[0-9]*')"
else
    echo "SPILLS: none"
fi
echo ""

# Run benchmark at multiple sizes
for SIZE in "4096 4096 4096" "8192 4096 4096" "16384 4096 4096"; do
    read M N K <<< "$SIZE"
    echo "--- ${M}x${N}x${K} ---"
    "$BINARY" --m=$M --n=$N --k=$K --iterations=200 --verify=0 $EXTRA_ARGS
    echo ""
done

echo "=== DONE ==="

rm -rf "$BUILD_DIR"
