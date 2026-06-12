SHELL := /usr/bin/env bash

APP := swat

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

CC ?= gcc
CXX ?= g++

CFLAGS ?= -O3 -std=c99 -fPIC
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
	-I$(TOP_LEVEL)/opts \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

CUDA_OBJ := \
	$(BUILD_DIR)/swat.nvcc.o \
	$(BUILD_DIR)/kernels.nvcc.o

CXX_OBJ := \
	$(BUILD_DIR)/alignments.o \
	$(BUILD_DIR)/prints.o \
	$(BUILD_DIR)/sequences.o \
	$(BUILD_DIR)/param.o \
	$(BUILD_DIR)/timeRec.o

OBJ := $(CUDA_OBJ) $(CXX_OBJ)

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help \
	run-swat-cuda-nvcc run-swat-cuda-scale-nvidia run-swat-cuda-scale-amd

all: swat-cuda-nvcc

help:
	@echo "Build targets:"
	@echo "  make swat-cuda-nvcc"
	@echo "  make swat-cuda-scale-nvidia"
	@echo "  make swat-cuda-scale-amd"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-swat-cuda-nvcc"
	@echo "  make run-swat-cuda-scale-nvidia"
	@echo "  make run-swat-cuda-scale-amd"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/swat.nvcc.o: swat.cu | $(BUILD_DIR)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-arch=sm_$(CUDA_ARCH) \
		-c $< \
		-o $@

$(BUILD_DIR)/kernels.nvcc.o: kernels.cu | $(BUILD_DIR)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-arch=sm_$(CUDA_ARCH) \
		-c $< \
		-o $@

$(BUILD_DIR)/alignments.o: ../alignments.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/prints.o: ../prints.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/sequences.o: ../sequences.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/param.o: ../param.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/timeRec.o: ../timeRec.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

swat-cuda-nvcc: $(OBJ)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		-arch=sm_$(CUDA_ARCH) \
		-o $@ \
		$(OBJ) \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

swat-cuda-scale-nvidia: swat.cu kernels.cu ../alignments.cpp ../prints.cpp ../sequences.cpp ../param.cpp ../timeRec.cpp
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-o $@ \
		swat.cu kernels.cu \
		../alignments.cpp ../prints.cpp ../sequences.cpp ../param.cpp ../timeRec.cpp \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

swat-cuda-scale-amd: swat.cu kernels.cu ../alignments.cpp ../prints.cpp ../sequences.cpp ../param.cpp ../timeRec.cpp
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-o $@ \
		swat.cu kernels.cu \
		../alignments.cpp ../prints.cpp ../sequences.cpp ../param.cpp ../timeRec.cpp \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

run-swat-cuda-nvcc:
	@if [[ ! -x "$(CURDIR)/swat-cuda-nvcc" ]]; then \
		$(MAKE) swat-cuda-nvcc; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/swat-cuda-nvcc" $(ARGS) $(N)

run-swat-cuda-scale-nvidia:
	@if [[ ! -x "$(CURDIR)/swat-cuda-scale-nvidia" ]]; then \
		$(MAKE) swat-cuda-scale-nvidia; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/swat-cuda-scale-nvidia" $(ARGS) $(N)

run-swat-cuda-scale-amd:
	@if [[ ! -x "$(CURDIR)/swat-cuda-scale-amd" ]]; then \
		$(MAKE) swat-cuda-scale-amd; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/swat-cuda-scale-amd" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f \
		swat-cuda-nvcc \
		swat-cuda-scale-nvidia \
		swat-cuda-scale-amd
