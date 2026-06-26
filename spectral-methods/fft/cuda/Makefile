SHELL := /usr/bin/env bash

TOP_LEVEL ?= ../../..
include $(TOP_LEVEL)/makefile_defs.mk

TARGET_NVCC := fft-cuda-nvcc
TARGET_SCALE_NVIDIA := fft-cuda-scale-nvidia
TARGET_SCALE_AMD := fft-cuda-scale-amd

BUILD_DIR := build/cuda
CUDA_OBJ := $(BUILD_DIR)/fft.nvcc.o

INCLUDES := \
	-I. \
	-I.. \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

COMMON_CPPFLAGS := $(CPPFLAGS) $(INCLUDES)

.PHONY: all clean run \
	$(TARGET_NVCC) $(TARGET_SCALE_NVIDIA) $(TARGET_SCALE_AMD) \
	run-$(TARGET_NVCC) run-$(TARGET_SCALE_NVIDIA) run-$(TARGET_SCALE_AMD)

all: $(TARGET_NVCC)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/fft.nvcc.o: fft.cu | $(BUILD_DIR)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		$(COMMON_CPPFLAGS) \
		-arch=sm_$(CUDA_ARCH) \
		-c $< \
		-o $@

$(TARGET_NVCC): $(CUDA_OBJ)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		-arch=sm_$(CUDA_ARCH) \
		-o $@ \
		$(CUDA_OBJ) \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

$(TARGET_SCALE_NVIDIA): fft.cu
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) \
		$(COMMON_CPPFLAGS) \
		-o $@ \
		fft.cu \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

$(TARGET_SCALE_AMD): fft.cu
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) \
		$(COMMON_CPPFLAGS) \
		-o $@ \
		fft.cu \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

run: run-$(TARGET_NVCC)

run-$(TARGET_NVCC): $(TARGET_NVCC)
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/$(TARGET_NVCC)" $(ARGS) $(N)

run-$(TARGET_SCALE_NVIDIA): $(TARGET_SCALE_NVIDIA)
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/$(TARGET_SCALE_NVIDIA)" $(ARGS) $(N)

run-$(TARGET_SCALE_AMD): $(TARGET_SCALE_AMD)
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/$(TARGET_SCALE_AMD)" $(ARGS) $(N)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET_NVCC) $(TARGET_SCALE_NVIDIA) $(TARGET_SCALE_AMD)
