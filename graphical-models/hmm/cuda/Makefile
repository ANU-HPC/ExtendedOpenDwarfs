SHELL := /usr/bin/env bash

APP := bwa_hmm

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
	$(BUILD_DIR)/main_bwa_hmm.nvcc.o \
	$(BUILD_DIR)/bwa_hmm_cuda_kernel.nvcc.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help \
	bwa_hmm-cuda-nvcc bwa_hmm-cuda-scale-nvidia bwa_hmm-cuda-scale-amd \
	run-bwa_hmm-cuda-nvcc run-bwa_hmm-cuda-scale-nvidia run-bwa_hmm-cuda-scale-amd

all: bwa_hmm-cuda-nvcc

help:
	@echo "Build targets:"
	@echo "  make bwa_hmm-cuda-nvcc"
	@echo "  make bwa_hmm-cuda-scale-nvidia"
	@echo "  make bwa_hmm-cuda-scale-amd"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-bwa_hmm-cuda-nvcc"
	@echo "  make run-bwa_hmm-cuda-scale-nvidia"
	@echo "  make run-bwa_hmm-cuda-scale-amd"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/main_bwa_hmm.nvcc.o: main_bwa_hmm.cu | $(BUILD_DIR)
	$(CUDA_NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) -arch=sm_$(CUDA_ARCH) -c $< -o $@

$(BUILD_DIR)/bwa_hmm_cuda_kernel.nvcc.o: bwa_hmm_cuda_kernel.cu | $(BUILD_DIR)
	$(CUDA_NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) -arch=sm_$(CUDA_ARCH) -c $< -o $@

bwa_hmm-cuda-nvcc: $(OBJ)
	$(CUDA_NVCC) $(NVCCFLAGS) -arch=sm_$(CUDA_ARCH) -o $@ $(OBJ) $(LDFLAGS) $(LDLIBS) -lm

bwa_hmm-cuda-scale-nvidia: main_bwa_hmm.cu bwa_hmm_cuda_kernel.cu
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) \
		-o $@ main_bwa_hmm.cu bwa_hmm_cuda_kernel.cu \
		$(LDFLAGS) $(LDLIBS) -lm

bwa_hmm-cuda-scale-amd: main_bwa_hmm.cu bwa_hmm_cuda_kernel.cu
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) \
		-o $@ main_bwa_hmm.cu bwa_hmm_cuda_kernel.cu \
		$(LDFLAGS) $(LDLIBS) -lm

run-bwa_hmm-cuda-nvcc:
	@if [[ ! -x "$(CURDIR)/bwa_hmm-cuda-nvcc" ]]; then $(MAKE) bwa_hmm-cuda-nvcc; fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && $(RUN_ENV) "$(CURDIR)/bwa_hmm-cuda-nvcc" $(ARGS) $(N)

run-bwa_hmm-cuda-scale-nvidia:
	@if [[ ! -x "$(CURDIR)/bwa_hmm-cuda-scale-nvidia" ]]; then $(MAKE) bwa_hmm-cuda-scale-nvidia; fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/bwa_hmm-cuda-scale-nvidia" $(ARGS) $(N)

run-bwa_hmm-cuda-scale-amd:
	@if [[ ! -x "$(CURDIR)/bwa_hmm-cuda-scale-amd" ]]; then $(MAKE) bwa_hmm-cuda-scale-amd; fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/bwa_hmm-cuda-scale-amd" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f bwa_hmm-cuda-nvcc bwa_hmm-cuda-scale-nvidia bwa_hmm-cuda-scale-amd
