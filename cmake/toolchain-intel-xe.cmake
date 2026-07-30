# xe-fuse Intel Xe GPU toolchain
#
# Prerequisites — must run BEFORE cmake:
#   source /opt/intel/oneapi/setvars.sh
#   # or site-specific, e.g.:
#   source /swtools/intel/compiler/2025.3/env/vars.sh
#   source /swtools/intel-gpu/latest/intel_gpu_vars.sh
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-intel-xe.cmake \
#         -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 \
#         -B build .

set(CMAKE_CXX_COMPILER "icpx" CACHE STRING
    "Intel oneAPI DPC++ compiler for Xe GPU targets")

# Suppress common noisy warnings that appear with Intel headers
add_compile_options(
  -Wall
  -Wno-unused-variable
  -Wno-unused-local-typedef
  -Wno-unused-but-set-variable
  -Wno-uninitialized
  -Wno-reorder-ctor
  -Wno-logical-op-parentheses
  -Wno-unused-function
  -Wno-unknown-pragmas
)
