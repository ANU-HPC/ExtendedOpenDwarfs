SHELL := /usr/bin/env bash

APP := tdm

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

CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fPIC
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

CUDA_OBJ := \
	$(BUILD_DIR)/tdm_cuda.nvcc.o \
	$(BUILD_DIR)/tdm_cuda_kernel.nvcc.o

CXX_OBJ := \
	$(BUILD_DIR)/dataio.o

OBJ := $(CUDA_OBJ) $(CXX_OBJ)

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help \
	tdm-cuda-nvcc tdm-cuda-scale-nvidia tdm-cuda-scale-amd \
	run-tdm-cuda-nvcc run-tdm-cuda-scale-nvidia run-tdm-cuda-scale-amd

all: tdm-cuda-nvcc

help:
	@echo "Build targets:"
	@echo "  make tdm-cuda-nvcc"
	@echo "  make tdm-cuda-scale-nvidia"
	@echo "  make tdm-cuda-scale-amd"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-tdm-cuda-nvcc"
	@echo "  make run-tdm-cuda-scale-nvidia"
	@echo "  make run-tdm-cuda-scale-amd"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/tdm_cuda.nvcc.o: tdm_cuda.cu ../types.h ../global.h ../dataio.h | $(BUILD_DIR)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-arch=sm_$(CUDA_ARCH) \
		-c $< \
		-o $@

$(BUILD_DIR)/tdm_cuda_kernel.nvcc.o: tdm_cuda_kernel.cu ../types.h | $(BUILD_DIR)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-arch=sm_$(CUDA_ARCH) \
		-c $< \
		-o $@

$(BUILD_DIR)/dataio.o: ../dataio.cpp ../dataio.h ../types.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

tdm-cuda-nvcc: $(OBJ)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		-arch=sm_$(CUDA_ARCH) \
		-o $@ \
		$(OBJ) \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

tdm-cuda-scale-nvidia: tdm_cuda.cu tdm_cuda_kernel.cu ../dataio.cpp
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-o $@ \
		tdm_cuda.cu tdm_cuda_kernel.cu ../dataio.cpp \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

tdm-cuda-scale-amd: tdm_cuda.cu tdm_cuda_kernel.cu ../dataio.cpp
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-o $@ \
		tdm_cuda.cu tdm_cuda_kernel.cu ../dataio.cpp \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

run-tdm-cuda-nvcc:
	@if [[ ! -x "$(CURDIR)/tdm-cuda-nvcc" ]]; then \
		$(MAKE) tdm-cuda-nvcc; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/tdm-cuda-nvcc" $(ARGS) $(N)

run-tdm-cuda-scale-nvidia:
	@if [[ ! -x "$(CURDIR)/tdm-cuda-scale-nvidia" ]]; then \
		$(MAKE) tdm-cuda-scale-nvidia; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/tdm-cuda-scale-nvidia" $(ARGS) $(N)

run-tdm-cuda-scale-amd:
	@if [[ ! -x "$(CURDIR)/tdm-cuda-scale-amd" ]]; then \
		$(MAKE) tdm-cuda-scale-amd; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/tdm-cuda-scale-amd" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f \
		tdm-cuda-nvcc \
		tdm-cuda-scale-nvidia \
		tdm-cuda-scale-amd
