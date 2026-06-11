SHELL := /usr/bin/env bash

APP := lud

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

CC ?= /usr/bin/gcc
CFLAGS ?= -O3 -std=c99 -fPIC
NVCCFLAGS ?= -O3 -std=c++17

CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

BUILD_DIR ?= build/cuda

COMMON_SRC := ../opencl/common.c

INCLUDES := \
	-I. \
	-I.. \
	-I../opencl \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

NVCC_OBJ := \
	$(BUILD_DIR)/lud.nvcc.o \
	$(BUILD_DIR)/lud_kernel.nvcc.o \
	$(BUILD_DIR)/common.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help \
	run-lud-cuda-nvcc run-lud-cuda-scale-nvidia run-lud-cuda-scale-amd

all: lud-cuda-nvcc

help:
	@echo "Build targets:"
	@echo "  make lud-cuda-nvcc"
	@echo "  make lud-cuda-scale-nvidia"
	@echo "  make lud-cuda-scale-amd"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-lud-cuda-nvcc"
	@echo "  make run-lud-cuda-scale-nvidia"
	@echo "  make run-lud-cuda-scale-amd"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/common.o: $(COMMON_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/lud.nvcc.o: lud.cu | $(BUILD_DIR)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-arch=sm_$(CUDA_ARCH) \
		-c $< \
		-o $@

$(BUILD_DIR)/lud_kernel.nvcc.o: lud_kernel.cu | $(BUILD_DIR)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-arch=sm_$(CUDA_ARCH) \
		-c $< \
		-o $@

lud-cuda-nvcc: $(NVCC_OBJ)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		-arch=sm_$(CUDA_ARCH) \
		-o $@ \
		$(NVCC_OBJ) \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

lud-cuda-scale-nvidia: lud.cu lud_kernel.cu $(BUILD_DIR)/common.o
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-o $@ \
		lud.cu lud_kernel.cu $(BUILD_DIR)/common.o \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

lud-cuda-scale-amd: lud.cu lud_kernel.cu $(BUILD_DIR)/common.o
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-o $@ \
		lud.cu lud_kernel.cu $(BUILD_DIR)/common.o \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

run-lud-cuda-nvcc:
	@if [[ ! -x "$(CURDIR)/lud-cuda-nvcc" ]]; then \
		$(MAKE) lud-cuda-nvcc; \
	fi
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/lud-cuda-nvcc" $(ARGS) $(N)

run-lud-cuda-scale-nvidia:
	@if [[ ! -x "$(CURDIR)/lud-cuda-scale-nvidia" ]]; then \
		$(MAKE) lud-cuda-scale-nvidia; \
	fi
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/lud-cuda-scale-nvidia" $(ARGS) $(N)

run-lud-cuda-scale-amd:
	@if [[ ! -x "$(CURDIR)/lud-cuda-scale-amd" ]]; then \
		$(MAKE) lud-cuda-scale-amd; \
	fi
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/lud-cuda-scale-amd" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f \
		lud-cuda-nvcc \
		lud-cuda-scale-nvidia \
		lud-cuda-scale-amd
