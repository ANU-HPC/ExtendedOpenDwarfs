SHELL := /usr/bin/env bash

APP := needle

TOP_LEVEL ?= ../../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

CC ?= gcc

CFLAGS ?= -O3 -std=c99
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
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(TOP_LEVEL)/opts \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

OBJ := \
	$(BUILD_DIR)/needle.o \
	$(BUILD_DIR)/common_args.o \
	$(BUILD_DIR)/opts.o \
	$(BUILD_DIR)/rdtsc.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help run-needle-opencl

all: needle-opencl

help:
	@echo "Build targets:"
	@echo "  make needle-opencl"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-needle-opencl"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/needle.o: needle.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rdtsc.o: $(OCD_RDTSC_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

needle-opencl: $(OBJ) needle_kernel.cl needle_kernel_opt_gpu.cl
	$(CC) \
		$(CFLAGS) \
		-o $@ \
		$(OBJ) \
		$(LDFLAGS) \
		$(OPENCL_LDFLAGS) \
		$(LDLIBS) \
		$(OPENCL_LDLIBS) \
		-lm

run-needle-opencl:
	@if [[ ! -x "$(CURDIR)/needle-opencl" ]]; then \
		$(MAKE) needle-opencl; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cp needle_kernel.cl "$(TOP_LEVEL)/results/needle_kernel.cl"
	@cp needle_kernel_opt_gpu.cl "$(TOP_LEVEL)/results/needle_kernel_opt_gpu.cl"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/needle-opencl" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f needle-opencl
