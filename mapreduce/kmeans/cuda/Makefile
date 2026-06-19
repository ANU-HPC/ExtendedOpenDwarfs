SHELL := /usr/bin/env bash

APP := kmeans

TOP_LEVEL ?= ../../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

include $(TOP_LEVEL)/makefile_defs.mk

BUILD_DIR ?= build/cuda

INCLUDES := \
	-I. \
	-I.. \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

COMMON_C_OBJS := \
	$(BUILD_DIR)/kmeans_main.o \
	$(BUILD_DIR)/cluster.o \
	$(BUILD_DIR)/getopt.o \
	$(BUILD_DIR)/kmeans_clustering.o \
	$(BUILD_DIR)/rmse.o

CUDA_OBJS := \
	$(BUILD_DIR)/kmeans_cuda.nvcc.o \
	$(BUILD_DIR)/kmeans_cuda_kernel.nvcc.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help \
	kmeans-cuda-nvcc kmeans-cuda-scale-nvidia kmeans-cuda-scale-amd \
	run-kmeans-cuda-nvcc run-kmeans-cuda-scale-nvidia run-kmeans-cuda-scale-amd

all: kmeans-cuda-nvcc

help:
	@echo "Build targets:"
	@echo "  make kmeans-cuda-nvcc"
	@echo "  make kmeans-cuda-scale-nvidia"
	@echo "  make kmeans-cuda-scale-amd"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-kmeans-cuda-nvcc"
	@echo "  make run-kmeans-cuda-scale-nvidia"
	@echo "  make run-kmeans-cuda-scale-amd"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/kmeans_main.o: ../kmeans.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cluster.o: ../cluster.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/getopt.o: ../getopt.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/kmeans_clustering.o: ../kmeans_clustering.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rmse.o: ../rmse.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/kmeans_cuda.nvcc.o: kmeans.cu | $(BUILD_DIR)
	$(CUDA_NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) -arch=sm_$(CUDA_ARCH) -c $< -o $@

$(BUILD_DIR)/kmeans_cuda_kernel.nvcc.o: kmeans_cuda_kernel.cu | $(BUILD_DIR)
	$(CUDA_NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) -arch=sm_$(CUDA_ARCH) -c $< -o $@

kmeans-cuda-nvcc: $(COMMON_C_OBJS) $(CUDA_OBJS)
	$(CUDA_NVCC) $(NVCCFLAGS) -arch=sm_$(CUDA_ARCH) -o $@ $^ $(LDFLAGS) $(LDLIBS) -lm

SCALE_COMMON_C_OBJS := \
	$(BUILD_DIR)/scale-kmeans_main.o \
	$(BUILD_DIR)/scale-cluster.o \
	$(BUILD_DIR)/scale-getopt.o \
	$(BUILD_DIR)/scale-kmeans_clustering.o \
	$(BUILD_DIR)/scale-rmse.o

SCALE_CUDA_OBJS := \
	$(BUILD_DIR)/scale-kmeans_cuda.o \
	$(BUILD_DIR)/scale-kmeans_cuda_kernel.o

$(BUILD_DIR)/scale-kmeans_main.o: ../kmeans.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/scale-cluster.o: ../cluster.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/scale-getopt.o: ../getopt.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/scale-kmeans_clustering.o: ../kmeans_clustering.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/scale-rmse.o: ../rmse.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/scale-kmeans_cuda.o: kmeans.cu | $(BUILD_DIR)
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) \
		-c $< -o $@

$(BUILD_DIR)/scale-kmeans_cuda_kernel.o: kmeans_cuda_kernel.cu | $(BUILD_DIR)
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) \
		-c $< -o $@

kmeans-cuda-scale-nvidia: $(SCALE_COMMON_C_OBJS) $(SCALE_CUDA_OBJS)
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) \
		-o $@ $^ \
		$(LDFLAGS) $(LDLIBS) -lm

$(BUILD_DIR)/scale-amd-kmeans_cuda.o: kmeans.cu | $(BUILD_DIR)
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) \
		-c $< -o $@

$(BUILD_DIR)/scale-amd-kmeans_cuda_kernel.o: kmeans_cuda_kernel.cu | $(BUILD_DIR)
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) \
		-c $< -o $@

kmeans-cuda-scale-amd: $(SCALE_COMMON_C_OBJS) $(BUILD_DIR)/scale-amd-kmeans_cuda.o $(BUILD_DIR)/scale-amd-kmeans_cuda_kernel.o
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) \
		-o $@ $^ \
		$(LDFLAGS) $(LDLIBS) -lm

run-kmeans-cuda-nvcc:
	@if [[ ! -x "$(CURDIR)/kmeans-cuda-nvcc" ]]; then $(MAKE) kmeans-cuda-nvcc; fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && $(RUN_ENV) "$(CURDIR)/kmeans-cuda-nvcc" $(ARGS) $(N)

run-kmeans-cuda-scale-nvidia:
	@if [[ ! -x "$(CURDIR)/kmeans-cuda-scale-nvidia" ]]; then $(MAKE) kmeans-cuda-scale-nvidia; fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/kmeans-cuda-scale-nvidia" $(ARGS) $(N)

run-kmeans-cuda-scale-amd:
	@if [[ ! -x "$(CURDIR)/kmeans-cuda-scale-amd" ]]; then $(MAKE) kmeans-cuda-scale-amd; fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/kmeans-cuda-scale-amd" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f kmeans-cuda-nvcc kmeans-cuda-scale-nvidia kmeans-cuda-scale-amd
