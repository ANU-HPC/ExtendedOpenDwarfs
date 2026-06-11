SHELL := /usr/bin/env bash

APP := needle

TOP_LEVEL ?= ../../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

NVCC ?= nvcc
NVIDIA_NVCC ?= $(shell command -v nvcc 2>/dev/null || echo $(NVCC))
CUDA_NVCC ?= $(NVIDIA_NVCC)
CUDA_PATH ?=

SCALE_ROOT ?= $(TOP_LEVEL)/scale-1.7.0-Linux

CUDA_DEV_TARGET ?= sm_70
CUDA_ARCH ?= $(patsubst sm_%,%,$(CUDA_DEV_TARGET))
HIP_ARCH ?= gfx942

NVCCFLAGS ?= -O3 -std=c++17
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

BUILD_DIR ?= build/cuda

INCLUDES := \
	-I. \
	-I.. \
	-I../opencl \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

NVCC_OBJ := \
	$(BUILD_DIR)/needle_cuda.nvcc.o \
	$(BUILD_DIR)/needle_kernel.nvcc.o \
	$(BUILD_DIR)/needle_kernel_opt_gpu.nvcc.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help \
	run-needle-cuda-nvcc run-needle-cuda-scale-nvidia run-needle-cuda-scale-amd

all: needle-cuda-nvcc

help:
	@echo "Build targets:"
	@echo "  make needle-cuda-nvcc"
	@echo "  make needle-cuda-scale-nvidia"
	@echo "  make needle-cuda-scale-amd"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-needle-cuda-nvcc"
	@echo "  make run-needle-cuda-scale-nvidia"
	@echo "  make run-needle-cuda-scale-amd"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/needle_cuda.nvcc.o: needle_cuda.cu | $(BUILD_DIR)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-arch=sm_$(CUDA_ARCH) \
		-c $< \
		-o $@

$(BUILD_DIR)/needle_kernel.nvcc.o: needle_kernel.cu | $(BUILD_DIR)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-arch=sm_$(CUDA_ARCH) \
		-c $< \
		-o $@

$(BUILD_DIR)/needle_kernel_opt_gpu.nvcc.o: needle_kernel_opt_gpu.cu | $(BUILD_DIR)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-arch=sm_$(CUDA_ARCH) \
		-c $< \
		-o $@

needle-cuda-nvcc: $(NVCC_OBJ)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		-arch=sm_$(CUDA_ARCH) \
		-o $@ \
		$(NVCC_OBJ) \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

needle-cuda-scale-nvidia: needle_cuda.cu needle_kernel.cu needle_kernel_opt_gpu.cu
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-o $@ \
		needle_cuda.cu needle_kernel.cu needle_kernel_opt_gpu.cu \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

needle-cuda-scale-amd: needle_cuda.cu needle_kernel.cu needle_kernel_opt_gpu.cu
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-o $@ \
		needle_cuda.cu needle_kernel.cu needle_kernel_opt_gpu.cu \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

run-needle-cuda-nvcc:
	@if [[ ! -x "$(CURDIR)/needle-cuda-nvcc" ]]; then \
		$(MAKE) needle-cuda-nvcc; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/needle-cuda-nvcc" $(ARGS) $(N)

run-needle-cuda-scale-nvidia:
	@if [[ ! -x "$(CURDIR)/needle-cuda-scale-nvidia" ]]; then \
		$(MAKE) needle-cuda-scale-nvidia; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/needle-cuda-scale-nvidia" $(ARGS) $(N)

run-needle-cuda-scale-amd:
	@if [[ ! -x "$(CURDIR)/needle-cuda-scale-amd" ]]; then \
		$(MAKE) needle-cuda-scale-amd; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/needle-cuda-scale-amd" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f \
		needle-cuda-nvcc \
		needle-cuda-scale-nvidia \
		needle-cuda-scale-amd
