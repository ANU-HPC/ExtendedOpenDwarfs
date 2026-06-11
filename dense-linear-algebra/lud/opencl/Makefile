SHELL := /usr/bin/env bash

APP := lud

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

COMMON_OBJ := \
	$(BUILD_DIR)/common.o \
	$(BUILD_DIR)/common_args.o \
	$(BUILD_DIR)/opts.o \
	$(BUILD_DIR)/rdtsc.o

LUD_OBJ := \
	$(BUILD_DIR)/lud.o \
	$(COMMON_OBJ)

LUD_PROFILE_OBJ := \
	$(BUILD_DIR)/lud_profiling_outer_loop.o \
	$(COMMON_OBJ)

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help \
	run-lud-opencl run-lud-profiling-outer-loop-opencl

all: lud-opencl

help:
	@echo "Build targets:"
	@echo "  make lud-opencl"
	@echo "  make lud-profiling-outer-loop-opencl"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-lud-opencl"
	@echo "  make run-lud-profiling-outer-loop-opencl"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/lud.o: lud.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/lud_profiling_outer_loop.o: lud.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) -DPROFILE_OUTER_LOOP $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/common.o: common.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rdtsc.o: $(OCD_RDTSC_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

lud-opencl: $(LUD_OBJ) lud_kernel.cl lud_kernel_opt_gpu.cl
	$(CC) \
		$(CFLAGS) \
		-o $@ \
		$(LUD_OBJ) \
		$(LDFLAGS) \
		$(OPENCL_LDFLAGS) \
		$(LDLIBS) \
		$(OPENCL_LDLIBS) \
		-lm

lud-profiling-outer-loop-opencl: $(LUD_PROFILE_OBJ) lud_kernel.cl lud_kernel_opt_gpu.cl
	$(CC) \
		$(CFLAGS) \
		-o $@ \
		$(LUD_PROFILE_OBJ) \
		$(LDFLAGS) \
		$(OPENCL_LDFLAGS) \
		$(LDLIBS) \
		$(OPENCL_LDLIBS) \
		-lm

run-lud-opencl:
	@if [[ ! -x "$(CURDIR)/lud-opencl" ]]; then \
		$(MAKE) lud-opencl; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cp lud_kernel.cl "$(TOP_LEVEL)/results/lud_kernel.cl"
	@cp lud_kernel_opt_gpu.cl "$(TOP_LEVEL)/results/lud_kernel_opt_gpu.cl"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/lud-opencl" $(ARGS) $(N)

run-lud-profiling-outer-loop-opencl:
	@if [[ ! -x "$(CURDIR)/lud-profiling-outer-loop-opencl" ]]; then \
		$(MAKE) lud-profiling-outer-loop-opencl; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cp lud_kernel.cl "$(TOP_LEVEL)/results/lud_kernel.cl"
	@cp lud_kernel_opt_gpu.cl "$(TOP_LEVEL)/results/lud_kernel_opt_gpu.cl"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/lud-profiling-outer-loop-opencl" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f \
		lud-opencl \
		lud-profiling-outer-loop-opencl
