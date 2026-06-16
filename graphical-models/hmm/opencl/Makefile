SHELL := /usr/bin/env bash

APP := bwa_hmm

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
	-I.. \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(TOP_LEVEL)/opts \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

OBJ := \
	$(BUILD_DIR)/main_bwa_hmm.o \
	$(BUILD_DIR)/common_args.o \
	$(BUILD_DIR)/opts.o \
	$(BUILD_DIR)/rdtsc.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help bwa_hmm-opencl run-bwa_hmm-opencl

all: bwa_hmm-opencl

help:
	@echo "Build targets:"
	@echo "  make bwa_hmm-opencl"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-bwa_hmm-opencl"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/main_bwa_hmm.o: main_bwa_hmm.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rdtsc.o: $(OCD_RDTSC_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

bwa_hmm-opencl: $(OBJ) bwa_hmm_opencl.cl
	$(CC) \
		$(CFLAGS) \
		-o $@ \
		$(OBJ) \
		$(LDFLAGS) \
		$(OPENCL_LDFLAGS) \
		$(LDLIBS) \
		$(OPENCL_LDLIBS) \
		-lm

run-bwa_hmm-opencl:
	@if [[ ! -x "$(CURDIR)/bwa_hmm-opencl" ]]; then \
		$(MAKE) bwa_hmm-opencl; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cp bwa_hmm_opencl.cl "$(TOP_LEVEL)/results/bwa_hmm_opencl.cl"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/bwa_hmm-opencl" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f bwa_hmm-opencl
