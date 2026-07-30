cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

# Re-entry guard for add_subdirectory / FetchContent nesting
if(xe_fuse_LOADED)
  return()
endif()
set(xe_fuse_LOADED ON)

project(XeFuse
  VERSION 0.1.0
  DESCRIPTION "Fused GEMM+epilogue kernels for Intel Xe (Battlemage/BMG)"
  LANGUAGES CXX)

# ── Options ────────────────────────────────────────────────────────────────────
option(XE_FUSE_BUILD_TESTS    "Build xe-fuse tests"    ${PROJECT_IS_TOP_LEVEL})
option(XE_FUSE_BUILD_EXAMPLES "Build xe-fuse examples" ${PROJECT_IS_TOP_LEVEL})

set(SYCL_TLA_DIR "" CACHE PATH
    "Path to an existing sycl-tla checkout. Leave empty to auto-fetch via FetchContent.")
set(SYCL_TLA_GIT_TAG "main" CACHE STRING
    "sycl-tla git tag/branch/SHA to fetch when SYCL_TLA_DIR is empty.")
set(DPCPP_SYCL_TARGET "intel_gpu_bmg_g31" CACHE STRING
    "Comma-separated SYCL device target(s): intel_gpu_bmg_g31, intel_gpu_bmg_g21, intel_gpu_pvc, spir64, ...")

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  message(WARNING
    "xe-fuse targets icpx (Intel oneAPI DPC++). "
    "Pass -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-intel-xe.cmake or set "
    "CMAKE_CXX_COMPILER=icpx. Other compilers are unsupported.")
endif()

# ── sycl-tla cmake options (must be set BEFORE sycl-tla's CMakeLists runs) ────
set(CUTLASS_ENABLE_SYCL         ON  CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_TOOLS        ON  CACHE BOOL "" FORCE)   # needed for cutlass_tools_util_includes
set(CUTLASS_ENABLE_HEADERS_ONLY OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_EXAMPLES     OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_TESTS        OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_LIBRARY      OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_BENCHMARKS   OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_PROFILER     OFF CACHE BOOL "" FORCE)

# ── Locate or fetch sycl-tla ──────────────────────────────────────────────────
if(SYCL_TLA_DIR)
  if(NOT EXISTS "${SYCL_TLA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "SYCL_TLA_DIR='${SYCL_TLA_DIR}' does not contain CMakeLists.txt. "
      "Check the path or leave SYCL_TLA_DIR empty to auto-fetch.")
  endif()
  message(STATUS "xe-fuse: using sycl-tla at ${SYCL_TLA_DIR}")
  add_subdirectory(${SYCL_TLA_DIR} ${CMAKE_BINARY_DIR}/_sycl_tla EXCLUDE_FROM_ALL)
else()
  message(STATUS "xe-fuse: fetching sycl-tla (tag=${SYCL_TLA_GIT_TAG}) via FetchContent")
  include(FetchContent)
  FetchContent_Declare(sycl_tla
    GIT_REPOSITORY https://github.com/intel/sycl-tla.git
    GIT_TAG        ${SYCL_TLA_GIT_TAG}
    GIT_SHALLOW    ON)
  FetchContent_MakeAvailable(sycl_tla)
  set(SYCL_TLA_DIR ${sycl_tla_SOURCE_DIR} CACHE PATH "" FORCE)
  message(STATUS "xe-fuse: sycl-tla fetched to ${SYCL_TLA_DIR}")
endif()

# ── DPCPP::DPCPP (normally created by sycl-tla; be defensive) ─────────────────
if(NOT TARGET DPCPP::DPCPP)
  list(APPEND CMAKE_MODULE_PATH "${SYCL_TLA_DIR}/cmake")
  find_package(DPCPP REQUIRED)
endif()

# ── xe_fuse INTERFACE library ─────────────────────────────────────────────────
add_library(xe_fuse INTERFACE)
add_library(xe_fuse::xe_fuse ALIAS xe_fuse)

target_include_directories(xe_fuse INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_definitions(xe_fuse INTERFACE
  CUTLASS_ENABLE_SYCL
  SYCL_INTEL_TARGET)

target_link_libraries(xe_fuse INTERFACE
  CUTLASS
  DPCPP::DPCPP)

# ── Internal harness target (tests + examples only, NOT installed) ─────────────
# Provides sycl_common.hpp / helper.h from sycl-tla examples/common and
# the CUTLASS util headers (tools/util/include + MKL).
add_library(_xe_fuse_test_harness INTERFACE)
target_include_directories(_xe_fuse_test_harness INTERFACE
  ${SYCL_TLA_DIR}/examples/common)
target_link_libraries(_xe_fuse_test_harness INTERFACE
  xe_fuse
  nvidia::cutlass::tools::util)

# ── Helper function ────────────────────────────────────────────────────────────
include(cmake/XeFuseHelpers.cmake)

# ── Subdirectories ─────────────────────────────────────────────────────────────
if(XE_FUSE_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(XE_FUSE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ── Install rules ──────────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS xe_fuse
  EXPORT XeFuseTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT XeFuseTargets
  FILE XeFuseTargets.cmake
  NAMESPACE xe_fuse::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/XeFuse)
configure_package_config_file(
  cmake/XeFuseConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/XeFuseConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/XeFuse)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/XeFuseConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/XeFuseConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/XeFuseConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/XeFuse)
