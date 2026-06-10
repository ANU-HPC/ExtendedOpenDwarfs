# OpenDwarfsCUDA common Makefile definitions

HOST ?= $(shell hostname -s)
TOP_LEVEL ?= $(CURDIR)

# ------------------------------------------------------------
# Toolchains
# ------------------------------------------------------------

CC  ?= gcc
CXX ?= g++

CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
NVIDIA_NVCC ?= $(shell command -v nvcc 2>/dev/null || echo $(NVCC))
CUDA_NVCC ?= $(NVIDIA_NVCC)

SCALE_ROOT ?= $(TOP_LEVEL)/scale-1.7.0-Linux

# ------------------------------------------------------------
# Device targets
# ------------------------------------------------------------

CUDA_DEV_TARGET ?= sm_70
CUDA_ARCH ?= $(patsubst sm_%,%,$(CUDA_DEV_TARGET))

HIP_ARCH ?= gfx942

# ------------------------------------------------------------
# Flags
# ------------------------------------------------------------

CFLAGS ?= -O3
CXXFLAGS ?= -O3 -std=c++17
NVCCFLAGS ?= -O3 -std=c++17

CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

# ------------------------------------------------------------
# Benchmark / run defaults
# ------------------------------------------------------------

N ?= 14
ARGS ?=
RUN_ENV ?=

RESULTS_DIR ?= results

# ------------------------------------------------------------
# libLSB / libSciBench
# ------------------------------------------------------------
HOST ?= $(shell hostname -s)

LSB_GIT_URL ?= https://github.com/spcl/liblsb.git
LSB_SRC_DIR ?= $(TOP_LEVEL)/external/liblsb-src
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(HOST)
LSB_CONFIGURE_FLAGS ?= --without-mpi --without-papi

LSB_WITH_MPI ?= 0
LSB_WITH_PAPI ?= 0

LSB_CONFIGURE_FLAGS :=
ifeq ($(LSB_WITH_MPI),0)
  LSB_CONFIGURE_FLAGS += --without-mpi
endif
ifeq ($(LSB_WITH_PAPI),0)
  LSB_CONFIGURE_FLAGS += --without-papi
else
  LSB_CONFIGURE_FLAGS += --with-papi
endif

CPPFLAGS += -I$(LSB_INSTALL_ROOT)/include
LDFLAGS  += -L$(LSB_INSTALL_ROOT)/lib -Xlinker -rpath -Xlinker $(LSB_INSTALL_ROOT)/lib
LDLIBS   += -llsb

# ------------------------------------------------------------
# OpenCL 
# ------------------------------------------------------------
OCD_COMMON_ARGS_SRC ?= $(TOP_LEVEL)/include/common_args.c
OCD_OPTS_SRC ?= $(TOP_LEVEL)/opts/opts.c
OCD_RDTSC_SRC ?= $(TOP_LEVEL)/include/rdtsc.c
OPENCL_INC_DIR ?= /usr/include
OPENCL_LIB_DIR ?= /usr/lib/x86_64-linux-gnu
OPENCL_CPPFLAGS ?= -DOPENCL -DCL_TARGET_OPENCL_VERSION=120 -I$(OPENCL_INC_DIR)
OPENCL_LDFLAGS ?= -L$(OPENCL_LIB_DIR) -Xlinker -rpath -Xlinker $(OPENCL_LIB_DIR)
OPENCL_LDLIBS ?= -lOpenCL
