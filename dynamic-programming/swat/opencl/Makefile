SHELL := /usr/bin/env bash

APP := swat

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
	$(BUILD_DIR)/swat.o \
	$(BUILD_DIR)/alignments.o \
	$(BUILD_DIR)/prints.o \
	$(BUILD_DIR)/sequences.o \
	$(BUILD_DIR)/param.o \
	$(BUILD_DIR)/timeRec.o

C_OBJ := \
	$(BUILD_DIR)/common_args.o \
	$(BUILD_DIR)/opts.o \
	$(BUILD_DIR)/rdtsc.o

OBJ := $(CXX_OBJ) $(C_OBJ)

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help run-swat-opencl

all: swat-opencl

help:
	@echo "Build targets:"
	@echo "  make swat-opencl"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-swat-opencl"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/swat.o: swat.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/alignments.o: ../alignments.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/prints.o: ../prints.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/sequences.o: ../sequences.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/param.o: ../param.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/timeRec.o: ../timeRec.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rdtsc.o: $(OCD_RDTSC_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

swat-opencl: $(OBJ) kernels.cl
	$(CXX) \
		$(CXXFLAGS) \
		-o $@ \
		$(OBJ) \
		$(LDFLAGS) \
		$(OPENCL_LDFLAGS) \
		$(LDLIBS) \
		$(OPENCL_LDLIBS) \
		-lm

run-swat-opencl:
	@if [[ ! -x "$(CURDIR)/swat-opencl" ]]; then \
		$(MAKE) swat-opencl; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cp kernels.cl "$(TOP_LEVEL)/results/kernels.cl"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/swat-opencl" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f swat-opencl
