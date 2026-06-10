SHELL := /usr/bin/env bash

APP := nqueens

SRC := main.cpp nqueen_cuda.cu nqueen_cpu.cpp kernels_nqueens.cu

TOP_LEVEL ?= ../../..

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

INCLUDES := -I. -I$(TOP_LEVEL)/include

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help \
        nqueens-cuda-nvcc nqueens-cuda-scale-nvidia nqueens-cuda-scale-amd \
        run-nqueens-cuda-nvcc run-nqueens-cuda-scale-nvidia run-nqueens-cuda-scale-amd

all: nqueens-cuda-nvcc

help:
	@echo "Build targets:"
	@echo "  make nqueens-cuda-nvcc"
	@echo "  make nqueens-cuda-scale-nvidia"
	@echo "  make nqueens-cuda-scale-amd"

nqueens-cuda-nvcc: $(SRC)
	$(CUDA_NVCC) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-arch=sm_$(CUDA_ARCH) \
		-x cu \
		-o $@ \
		main.cpp nqueen_cuda.cu nqueen_cpu.cpp kernels_nqueens.cu \
		$(LDFLAGS) \
		$(LDLIBS)

nqueens-cuda-scale-nvidia: $(SRC)
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
		$(if $(CUDA_PATH),--cuda-path="$(CUDA_PATH)") \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-x cu \
		-o $@ \
		main.cpp nqueen_cuda.cu nqueen_cpu.cpp kernels_nqueens.cu \
		$(LDFLAGS) \
		$(LDLIBS)

nqueens-cuda-scale-amd: $(SRC)
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc \
		$(NVCCFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-x cu \
		-o $@ \
		main.cpp nqueen_cuda.cu nqueen_cpu.cpp kernels_nqueens.cu \
		$(LDFLAGS) \
		$(LDLIBS)

run-nqueens-cuda-nvcc:
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/nqueens-cuda-nvcc" $(ARGS) $(N)

run-nqueens-cuda-scale-nvidia:
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/nqueens-cuda-scale-nvidia" $(ARGS) $(N)

run-nqueens-cuda-scale-amd:
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) "$(CURDIR)/nqueens-cuda-scale-amd" $(ARGS) $(N)

clean:
	rm -f \
		nqueens-cuda-nvcc \
		nqueens-cuda-scale-nvidia \
		nqueens-cuda-scale-amd
