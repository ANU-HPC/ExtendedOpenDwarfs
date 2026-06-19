SHELL := /usr/bin/env bash

TOP_LEVEL ?= ../../..
include $(TOP_LEVEL)/makefile_defs.mk

TARGET_NVCC := gemnoui-cuda-nvcc
TARGET_SCALE_NVIDIA := gemnoui-cuda-scale-nvidia
TARGET_SCALE_AMD := gemnoui-cuda-scale-amd
BUILD_DIR := build/cuda

COMMON_C_OBJS := \
	$(BUILD_DIR)/dump_vertices.o \
	$(BUILD_DIR)/floating_centers.o \
	$(BUILD_DIR)/radix_sort.o \
	$(BUILD_DIR)/read_pqr.o \
	$(BUILD_DIR)/vector_math.o \
	$(BUILD_DIR)/write_grid.o \
	$(BUILD_DIR)/extrapolate_bonds.o \
	$(BUILD_DIR)/populate_stats.o \
	$(BUILD_DIR)/read_msms.o \
	$(BUILD_DIR)/run_msms.o \
	$(BUILD_DIR)/write_avs.o \
	$(BUILD_DIR)/write_xyzr.o

COMMON_CXX_OBJS := \
	$(BUILD_DIR)/check_cmdline.o \
	$(BUILD_DIR)/estimate_a.o \
	$(BUILD_DIR)/gem_no_ui.o \
	$(BUILD_DIR)/open_pqr_run_msms.o

CUDA_OBJS := \
	$(BUILD_DIR)/calculate_potential.nvcc.o

SCALE_COMMON_C_OBJS := \
	$(BUILD_DIR)/scale-dump_vertices.o \
	$(BUILD_DIR)/scale-floating_centers.o \
	$(BUILD_DIR)/scale-radix_sort.o \
	$(BUILD_DIR)/scale-read_pqr.o \
	$(BUILD_DIR)/scale-vector_math.o \
	$(BUILD_DIR)/scale-write_grid.o \
	$(BUILD_DIR)/scale-extrapolate_bonds.o \
	$(BUILD_DIR)/scale-populate_stats.o \
	$(BUILD_DIR)/scale-read_msms.o \
	$(BUILD_DIR)/scale-run_msms.o \
	$(BUILD_DIR)/scale-write_avs.o \
	$(BUILD_DIR)/scale-write_xyzr.o

SCALE_COMMON_CXX_OBJS := \
	$(BUILD_DIR)/scale-check_cmdline.o \
	$(BUILD_DIR)/scale-estimate_a.o \
	$(BUILD_DIR)/scale-gem_no_ui.o \
	$(BUILD_DIR)/scale-open_pqr_run_msms.o

SCALE_NVIDIA_CUDA_OBJS := \
	$(BUILD_DIR)/scale-nvidia-calculate_potential.o

SCALE_AMD_CUDA_OBJS := \
	$(BUILD_DIR)/scale-amd-calculate_potential.o

INCLUDES := \
	-I. \
	-I.. \
	-I../include \
	-I../include/visualize/dialogs \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

COMMON_CPPFLAGS := $(CPPFLAGS) $(INCLUDES) -DNO_UI

.PHONY: all clean run \
	run-$(TARGET_NVCC) run-$(TARGET_SCALE_NVIDIA) run-$(TARGET_SCALE_AMD) \
	$(TARGET_NVCC) $(TARGET_SCALE_NVIDIA) $(TARGET_SCALE_AMD)

all: $(TARGET_NVCC)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: ../%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: ../%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/calculate_potential.nvcc.o: calculate_potential.cu | $(BUILD_DIR)
	$(CUDA_NVCC) $(NVCCFLAGS) $(COMMON_CPPFLAGS) -arch=sm_$(CUDA_ARCH) -c $< -o $@

$(TARGET_NVCC): $(COMMON_C_OBJS) $(COMMON_CXX_OBJS) $(CUDA_OBJS)
	$(CUDA_NVCC) $(NVCCFLAGS) -arch=sm_$(CUDA_ARCH) -o $@ $^ $(LDFLAGS) $(LDLIBS) -lm

$(BUILD_DIR)/scale-%.o: ../%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/scale-%.o: ../%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/scale-nvidia-calculate_potential.o: calculate_potential.cu | $(BUILD_DIR)
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) $(COMMON_CPPFLAGS) \
		-c $< -o $@

$(BUILD_DIR)/scale-amd-calculate_potential.o: calculate_potential.cu | $(BUILD_DIR)
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) $(COMMON_CPPFLAGS) \
		-c $< -o $@

$(TARGET_SCALE_NVIDIA): $(SCALE_COMMON_C_OBJS) $(SCALE_COMMON_CXX_OBJS) $(SCALE_NVIDIA_CUDA_OBJS)
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) \
		-o $@ $^ \
		$(LDFLAGS) $(LDLIBS) -lm

$(TARGET_SCALE_AMD): $(SCALE_COMMON_C_OBJS) $(SCALE_COMMON_CXX_OBJS) $(SCALE_AMD_CUDA_OBJS)
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
