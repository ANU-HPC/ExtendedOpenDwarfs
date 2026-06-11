SHELL := /usr/bin/env bash

APP := crc

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

CRC_COMMON_SRC := ../common/crc_formats.c
CRC_COMMON_OBJ := crc_formats.o

INCLUDES := \
	-I. \
	-I.. \
	-I../include \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help \
        crc-cuda-nvcc crc-cuda-scale-nvidia crc-cuda-scale-amd \
        run-crc-cuda-nvcc run-crc-cuda-scale-nvidia run-crc-cuda-scale-amd

all: crc-cuda-nvcc

help:
	@echo "Build targets:"
	@echo "  make crc-cuda-nvcc"
	@echo "  make crc-cuda-scale-nvidia"
	@echo "  make crc-cuda-scale-amd"

$(CRC_COMMON_OBJ): $(CRC_COMMON_SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

crc-cuda-nvcc: crc_algo.cu crc_kernel.cu $(CRC_COMMON_OBJ)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-arch=sm_$(CUDA_ARCH) \
		-o $@ \
		crc_algo.cu crc_kernel.cu $(CRC_COMMON_OBJ) \
		$(LDFLAGS) \
		$(LDLIBS)

crc-cuda-scale-nvidia: crc_algo.cu crc_kernel.cu $(CRC_COMMON_OBJ)
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-o $@ \
		crc_algo.cu crc_kernel.cu $(CRC_COMMON_OBJ) \
		$(LDFLAGS) \
		$(LDLIBS)

crc-cuda-scale-amd: crc_algo.cu crc_kernel.cu $(CRC_COMMON_OBJ)
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-o $@ \
		crc_algo.cu crc_kernel.cu $(CRC_COMMON_OBJ) \
		$(LDFLAGS) \
		$(LDLIBS)

run-crc-cuda-nvcc: crc-cuda-nvcc
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/crc-cuda-nvcc" $(ARGS) $(N)

run-crc-cuda-scale-nvidia: crc-cuda-scale-nvidia
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/crc-cuda-scale-nvidia" $(ARGS) $(N)

run-crc-cuda-scale-amd: crc-cuda-scale-amd
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/crc-cuda-scale-amd" $(ARGS) $(N)

clean:
	rm -f \
		crc-cuda-nvcc \
		crc-cuda-scale-nvidia \
		crc-cuda-scale-amd \
		$(CRC_COMMON_OBJ)
