# xe-fuse wrapper Makefile
#
# Usage:
#   make                                   # configure + build (auto-fetches sycl-tla)
#   make SYCL_TLA_DIR=/path/to/sycl-tla   # use existing sycl-tla checkout
#   make SYCL_TARGET=intel_gpu_pvc         # different GPU target
#   make BUILD_TYPE=Debug                  # debug build
#   make tests                             # build only tests
#   make examples                          # build only examples
#   make install PREFIX=/opt/xe-fuse       # install headers + cmake package
#   make clean                             # remove build dir
#   make help                              # show this help

BUILD_DIR    ?= build
BUILD_TYPE   ?= Release
SYCL_TARGET  ?= intel_gpu_bmg_g31
SYCL_TLA_DIR ?=
PREFIX       ?= $(abspath install)
NPROC        := $(shell nproc 2>/dev/null || echo 8)

CMAKE_ARGS := \
  -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
  -DDPCPP_SYCL_TARGET=$(SYCL_TARGET) \
  -DXE_FUSE_BUILD_TESTS=ON \
  -DXE_FUSE_BUILD_EXAMPLES=ON

ifneq ($(SYCL_TLA_DIR),)
  CMAKE_ARGS += -DSYCL_TLA_DIR=$(abspath $(SYCL_TLA_DIR))
endif

.PHONY: all configure build tests examples install clean help

all: build

configure:
	cmake -B $(BUILD_DIR) \
	  -DCMAKE_TOOLCHAIN_FILE=$(abspath cmake/toolchain-intel-xe.cmake) \
	  $(CMAKE_ARGS) \
	  $(CURDIR)

build: configure
	cmake --build $(BUILD_DIR) --parallel $(NPROC)

tests: configure
	cmake --build $(BUILD_DIR) --parallel $(NPROC) --target $(shell \
	  cmake --build $(BUILD_DIR) --target help 2>/dev/null | grep '^test_\|^tile_sweep\|^gemm_\|^moe_tile' | awk '{print $$1}' | tr '\n' ' ')
	@echo ""
	@echo "Test binaries are in $(BUILD_DIR)/tests/"
	@echo "Submit GPU jobs with: sbatch tests/run_tests.sh"

examples: configure
	cmake --build $(BUILD_DIR) --parallel $(NPROC) --target moe_expert_builder moe_expert_fused
	@echo "Example binaries are in $(BUILD_DIR)/examples/"

install: build
	cmake --install $(BUILD_DIR) --prefix $(PREFIX)

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Targets : all configure build tests examples install clean"
	@echo ""
	@echo "Variables:"
	@echo "  BUILD_DIR    = $(BUILD_DIR)   (output directory)"
	@echo "  BUILD_TYPE   = $(BUILD_TYPE)  (Release | Debug)"
	@echo "  SYCL_TARGET  = $(SYCL_TARGET) (e.g. intel_gpu_bmg_g31, intel_gpu_pvc)"
	@echo "  SYCL_TLA_DIR = $(if $(SYCL_TLA_DIR),$(SYCL_TLA_DIR),(empty — auto-fetch))"
	@echo "  PREFIX       = $(PREFIX)      (cmake --install prefix)"
	@echo ""
	@echo "Examples:"
	@echo "  make SYCL_TLA_DIR=~/sycl-tla"
	@echo "  make SYCL_TARGET=intel_gpu_pvc BUILD_TYPE=Debug"
	@echo "  make install PREFIX=/opt/xe-fuse"
