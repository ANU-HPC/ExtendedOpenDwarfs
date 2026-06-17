SHELL := /usr/bin/env bash

APP := bfs

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

OBJ := \
	$(BUILD_DIR)/bfs.o \
	$(BUILD_DIR)/common_args.o \
	$(BUILD_DIR)/opts.o \
	$(BUILD_DIR)/rdtsc.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help bfs-opencl run-bfs-opencl

all: bfs-opencl

help:
	@echo "Build targets:"
	@echo "  make bfs-opencl"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-bfs-opencl"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/bfs.o: bfs.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rdtsc.o: $(OCD_RDTSC_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

bfs-opencl: $(OBJ) bfs_kernel.cl
	$(CXX) \
		$(CXXFLAGS) \
		-o $@ \
		$(OBJ) \
		$(LDFLAGS) \
		$(OPENCL_LDFLAGS) \
		$(LDLIBS) \
		$(OPENCL_LDLIBS) \
		-lm

run-bfs-opencl:
	@if [[ ! -x "$(CURDIR)/bfs-opencl" ]]; then \
		$(MAKE) bfs-opencl; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cp bfs_kernel.cl "$(TOP_LEVEL)/results/bfs_kernel.cl"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/bfs-opencl" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f bfs-opencl
