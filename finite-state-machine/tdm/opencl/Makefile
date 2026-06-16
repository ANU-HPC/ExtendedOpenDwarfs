SHELL := /usr/bin/env bash

APP := tdm

TOP_LEVEL ?= ../../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

CC ?= gcc
CXX ?= g++

CFLAGS ?= -O3 -std=c99
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

OPENCL_CPPFLAGS ?= -DOPENCL -DCL_TARGET_OPENCL_VERSION=120
OPENCL_LDFLAGS ?=
OPENCL_LDLIBS ?= -lOpenCL

OCD_COMMON_ARGS_SRC ?= $(TOP_LEVEL)/include/common_args.c
OCD_OPTS_SRC ?= $(TOP_LEVEL)/opts/opts.c
OCD_RDTSC_SRC ?= $(TOP_LEVEL)/include/rdtsc.c

BUILD_DIR ?= build/opencl

INCLUDES := \
	-I. \
	-I.. \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(TOP_LEVEL)/opts \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

CXX_OBJ := \
	$(BUILD_DIR)/tdm_ocl.o \
	$(BUILD_DIR)/dataio.o

C_OBJ := \
	$(BUILD_DIR)/common_args.o \
	$(BUILD_DIR)/opts.o \
	$(BUILD_DIR)/rdtsc.o

OBJ := $(CXX_OBJ) $(C_OBJ)

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help run-tdm-opencl

all: tdm-opencl

help:
	@echo "Build targets:"
	@echo "  make tdm-opencl"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-tdm-opencl"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/tdm_ocl.o: tdm_ocl.cpp ../types.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/dataio.o: ../dataio.cpp ../dataio.h ../types.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rdtsc.o: $(OCD_RDTSC_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

tdm-opencl: $(OBJ) tdm_ocl_kernel.cl ../types.h
	$(CXX) \
		$(CXXFLAGS) \
		-o $@ \
		$(OBJ) \
		$(LDFLAGS) \
		$(OPENCL_LDFLAGS) \
		$(LDLIBS) \
		$(OPENCL_LDLIBS) \
		-lm

run-tdm-opencl:
	@if [[ ! -x "$(CURDIR)/tdm-opencl" ]]; then \
		$(MAKE) tdm-opencl; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cp tdm_ocl_kernel.cl "$(TOP_LEVEL)/results/tdm_ocl_kernel.cl"
	@cp ../types.h "$(TOP_LEVEL)/results/types.h"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/tdm-opencl" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f tdm-opencl
