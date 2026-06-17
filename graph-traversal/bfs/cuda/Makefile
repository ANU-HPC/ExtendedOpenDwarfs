SHELL := /usr/bin/env bash

APP := bfs

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
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

OBJ := \
	$(BUILD_DIR)/bfs.nvcc.o \
	$(BUILD_DIR)/bfs_kernel.nvcc.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help \
	bfs-cuda-nvcc bfs-cuda-scale-nvidia bfs-cuda-scale-amd \
	run-bfs-cuda-nvcc run-bfs-cuda-scale-nvidia run-bfs-cuda-scale-amd

all: bfs-cuda-nvcc

help:
	@echo "Build targets:"
	@echo "  make bfs-cuda-nvcc"
	@echo "  make bfs-cuda-scale-nvidia"
	@echo "  make bfs-cuda-scale-amd"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-bfs-cuda-nvcc"
	@echo "  make run-bfs-cuda-scale-nvidia"
	@echo "  make run-bfs-cuda-scale-amd"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/bfs.nvcc.o: bfs.cu | $(BUILD_DIR)
	$(CUDA_NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) -arch=sm_$(CUDA_ARCH) -c $< -o $@

$(BUILD_DIR)/bfs_kernel.nvcc.o: bfs_kernel.cu | $(BUILD_DIR)
	$(CUDA_NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) -arch=sm_$(CUDA_ARCH) -c $< -o $@

bfs-cuda-nvcc: $(OBJ)
	$(CUDA_NVCC) $(NVCCFLAGS) -arch=sm_$(CUDA_ARCH) -o $@ $(OBJ) $(LDFLAGS) $(LDLIBS) -lm

bfs-cuda-scale-nvidia: bfs.cu bfs_kernel.cu
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) \
		-o $@ bfs.cu bfs_kernel.cu \
		$(LDFLAGS) $(LDLIBS) -lm

bfs-cuda-scale-amd: bfs.cu bfs_kernel.cu
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) \
		-o $@ bfs.cu bfs_kernel.cu \
		$(LDFLAGS) $(LDLIBS) -lm

run-bfs-cuda-nvcc:
	@if [[ ! -x "$(CURDIR)/bfs-cuda-nvcc" ]]; then $(MAKE) bfs-cuda-nvcc; fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && $(RUN_ENV) "$(CURDIR)/bfs-cuda-nvcc" $(ARGS) $(N)

run-bfs-cuda-scale-nvidia:
	@if [[ ! -x "$(CURDIR)/bfs-cuda-scale-nvidia" ]]; then $(MAKE) bfs-cuda-scale-nvidia; fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/bfs-cuda-scale-nvidia" $(ARGS) $(N)

run-bfs-cuda-scale-amd:
	@if [[ ! -x "$(CURDIR)/bfs-cuda-scale-amd" ]]; then $(MAKE) bfs-cuda-scale-amd; fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/bfs-cuda-scale-amd" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f bfs-cuda-nvcc bfs-cuda-scale-nvidia bfs-cuda-scale-amd
