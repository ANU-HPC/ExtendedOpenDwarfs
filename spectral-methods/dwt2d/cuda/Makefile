SHELL := /usr/bin/env bash

TOP_LEVEL ?= ../../..
include $(TOP_LEVEL)/makefile_defs.mk

TARGET_NVCC := dwt2d-cuda-nvcc
TARGET_SCALE_NVIDIA := dwt2d-cuda-scale-nvidia
TARGET_SCALE_AMD := dwt2d-cuda-scale-amd
BUILD_DIR := build/cuda

CUDA_OBJS := $(BUILD_DIR)/dwt2d.nvcc.o
SCALE_NVIDIA_CUDA_OBJS := $(BUILD_DIR)/scale-nvidia-dwt2d.o
SCALE_AMD_CUDA_OBJS := $(BUILD_DIR)/scale-amd-dwt2d.o

INCLUDES := \
	-I. \
	-I.. \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

COMMON_CPPFLAGS := $(CPPFLAGS) $(INCLUDES)

.PHONY: all clean run \
	run-$(TARGET_NVCC) run-$(TARGET_SCALE_NVIDIA) run-$(TARGET_SCALE_AMD) \
	$(TARGET_NVCC) $(TARGET_SCALE_NVIDIA) $(TARGET_SCALE_AMD)

all: $(TARGET_NVCC)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/dwt2d.nvcc.o: dwt2d.cu dwt2d_kernel.cuh | $(BUILD_DIR)
	$(CUDA_NVCC) $(NVCCFLAGS) $(COMMON_CPPFLAGS) -arch=sm_$(CUDA_ARCH) -c $< -o $@

$(TARGET_NVCC): $(CUDA_OBJS)
	$(CUDA_NVCC) $(NVCCFLAGS) -arch=sm_$(CUDA_ARCH) -o $@ $^ $(LDFLAGS) $(LDLIBS) -lm

$(BUILD_DIR)/scale-nvidia-dwt2d.o: dwt2d.cu dwt2d_kernel.cuh | $(BUILD_DIR)
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) $(COMMON_CPPFLAGS) \
		-c $< -o $@

$(BUILD_DIR)/scale-amd-dwt2d.o: dwt2d.cu dwt2d_kernel.cuh | $(BUILD_DIR)
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) $(COMMON_CPPFLAGS) \
		-c $< -o $@

$(TARGET_SCALE_NVIDIA): $(SCALE_NVIDIA_CUDA_OBJS)
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) \
		-o $@ $^ \
		$(LDFLAGS) $(LDLIBS) -lm

$(TARGET_SCALE_AMD): $(SCALE_AMD_CUDA_OBJS)
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) \
		-o $@ $^ \
		$(LDFLAGS) $(LDLIBS) -lm

run: run-$(TARGET_NVCC)

run-$(TARGET_NVCC): $(TARGET_NVCC)
	mkdir -p $(TOP_LEVEL)/results
	cd $(TOP_LEVEL)/results && $(RUN_ENV) $(CURDIR)/$(TARGET_NVCC) $(ARGS) $(N)

run-$(TARGET_SCALE_NVIDIA): $(TARGET_SCALE_NVIDIA)
	mkdir -p $(TOP_LEVEL)/results
	cd $(TOP_LEVEL)/results && source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) $(CURDIR)/$(TARGET_SCALE_NVIDIA) $(ARGS) $(N)

run-$(TARGET_SCALE_AMD): $(TARGET_SCALE_AMD)
	mkdir -p $(TOP_LEVEL)/results
	cd $(TOP_LEVEL)/results && source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) $(CURDIR)/$(TARGET_SCALE_AMD) $(ARGS) $(N)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET_NVCC) $(TARGET_SCALE_NVIDIA) $(TARGET_SCALE_AMD)
