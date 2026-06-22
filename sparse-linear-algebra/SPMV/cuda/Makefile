SHELL := /usr/bin/env bash

TOP_LEVEL ?= ../../..
include $(TOP_LEVEL)/makefile_defs.mk

TARGET_NVCC := csr-cuda-nvcc
TARGET_SCALE_NVIDIA := csr-cuda-scale-nvidia
TARGET_SCALE_AMD := csr-cuda-scale-amd
BUILD_DIR := build/cuda

COMMON_C_OBJS := \
	$(BUILD_DIR)/sparse_formats.o \
	$(BUILD_DIR)/ziggurat.o \
	$(BUILD_DIR)/common.o \
	$(BUILD_DIR)/common_args.o \
	$(BUILD_DIR)/opts.o

CUDA_OBJS := \
	$(BUILD_DIR)/csr.nvcc.o

SCALE_COMMON_C_OBJS := \
	$(BUILD_DIR)/scale-sparse_formats.o \
	$(BUILD_DIR)/scale-ziggurat.o \
	$(BUILD_DIR)/scale-common.o \
	$(BUILD_DIR)/scale-common_args.o \
	$(BUILD_DIR)/scale-opts.o

SCALE_NVIDIA_CUDA_OBJS := \
	$(BUILD_DIR)/scale-nvidia-csr.o

SCALE_AMD_CUDA_OBJS := \
	$(BUILD_DIR)/scale-amd-csr.o

INCLUDES := \
	-I. \
	-I.. \
	-I../inc \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

COMMON_CPPFLAGS := $(CPPFLAGS) $(INCLUDES)

.PHONY: all clean run \
	run-$(TARGET_NVCC) run-$(TARGET_SCALE_NVIDIA) run-$(TARGET_SCALE_AMD)

all: $(TARGET_NVCC)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/sparse_formats.o: ../src-common/sparse_formats.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/ziggurat.o: ../src-common/ziggurat.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/common.o: ../src-common/common.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -ffunction-sections -fdata-sections -Wno-deprecated-declarations -c $< -o $@


$(BUILD_DIR)/opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@


$(BUILD_DIR)/csr.nvcc.o: csr.cu | $(BUILD_DIR)
	$(CUDA_NVCC) $(NVCCFLAGS) $(COMMON_CPPFLAGS) -arch=sm_$(CUDA_ARCH) -c $< -o $@

$(TARGET_NVCC): $(COMMON_C_OBJS) $(CUDA_OBJS)
	$(CUDA_NVCC) $(NVCCFLAGS) -arch=sm_$(CUDA_ARCH) -o $@ $^ $(LDFLAGS) -Xlinker --gc-sections $(LDLIBS) -lm

$(BUILD_DIR)/scale-%.o: ../src-common/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/scale-common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -ffunction-sections -fdata-sections -Wno-deprecated-declarations -c $< -o $@


$(BUILD_DIR)/scale-opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@


$(BUILD_DIR)/scale-nvidia-csr.o: csr.cu | $(BUILD_DIR)
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) $(COMMON_CPPFLAGS) \
		-c $< -o $@

$(BUILD_DIR)/scale-amd-csr.o: csr.cu | $(BUILD_DIR)
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) $(COMMON_CPPFLAGS) \
		-c $< -o $@

$(TARGET_SCALE_NVIDIA): $(SCALE_COMMON_C_OBJS) $(SCALE_NVIDIA_CUDA_OBJS)
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) \
		-o $@ $^ \
		$(LDFLAGS) -Xlinker --gc-sections $(LDLIBS) -lm

$(TARGET_SCALE_AMD): $(SCALE_COMMON_C_OBJS) $(SCALE_AMD_CUDA_OBJS)
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) \
		-o $@ $^ \
		$(LDFLAGS) -Xlinker --gc-sections $(LDLIBS) -lm

run: run-$(TARGET_NVCC)

run-$(TARGET_NVCC):
	@if [ ! -x "$(CURDIR)/$(TARGET_NVCC)" ]; then \
		$(MAKE) $(TARGET_NVCC); \
	fi
	mkdir -p $(TOP_LEVEL)/results
	cd $(TOP_LEVEL)/results && $(RUN_ENV) $(CURDIR)/$(TARGET_NVCC) $(ARGS) $(N)

run-$(TARGET_SCALE_NVIDIA):
	@if [ ! -x "$(CURDIR)/$(TARGET_SCALE_NVIDIA)" ]; then \
		$(MAKE) $(TARGET_SCALE_NVIDIA); \
	fi
	mkdir -p $(TOP_LEVEL)/results
	cd $(TOP_LEVEL)/results && source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) $(CURDIR)/$(TARGET_SCALE_NVIDIA) $(ARGS) $(N)

run-$(TARGET_SCALE_AMD):
	@if [ ! -x "$(CURDIR)/$(TARGET_SCALE_AMD)" ]; then \
		$(MAKE) $(TARGET_SCALE_AMD); \
	fi
	mkdir -p $(TOP_LEVEL)/results
	cd $(TOP_LEVEL)/results && source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) $(CURDIR)/$(TARGET_SCALE_AMD) $(ARGS) $(N)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET_NVCC) $(TARGET_SCALE_NVIDIA) $(TARGET_SCALE_AMD)
