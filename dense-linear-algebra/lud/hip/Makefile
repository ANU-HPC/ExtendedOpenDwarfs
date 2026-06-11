SHELL := /usr/bin/env bash

APP := lud

TOP_LEVEL ?= ../../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

HIPCC ?= hipcc
ROCM_PATH ?= /opt/rocm
HIP_DEV_TARGET ?= gfx942
HIP_ARCH ?= $(HIP_DEV_TARGET)

CC ?= /usr/bin/gcc
CFLAGS ?= -O3 -std=c99 -fPIC
HIPFLAGS ?= -O3 -std=c++17

CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

BUILD_DIR ?= build/hip

COMMON_SRC := ../opencl/common.c

INCLUDES := \
	-I. \
	-I.. \
	-I../opencl \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

HIP_OBJ := \
	$(BUILD_DIR)/lud.hip.o \
	$(BUILD_DIR)/lud_kernel.hip.o \
	$(BUILD_DIR)/common.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help run-lud-hip-hipcc

all: lud-hip-hipcc

help:
	@echo "Build targets:"
	@echo "  make lud-hip-hipcc"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-lud-hip-hipcc"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/common.o: $(COMMON_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/lud.hip.o: lud.cpp | $(BUILD_DIR)
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		--offload-arch=$(HIP_ARCH) \
		-x hip \
		-c $< \
		-o $@

$(BUILD_DIR)/lud_kernel.hip.o: lud_kernel.hip | $(BUILD_DIR)
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		--offload-arch=$(HIP_ARCH) \
		-x hip \
		-c $< \
		-o $@

lud-hip-hipcc: $(HIP_OBJ)
	$(HIPCC) \
		$(HIPFLAGS) \
		--offload-arch=$(HIP_ARCH) \
		-o $@ \
		$(HIP_OBJ) \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

run-lud-hip-hipcc:
	@if [[ ! -x "$(CURDIR)/lud-hip-hipcc" ]]; then \
		$(MAKE) lud-hip-hipcc; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/lud-hip-hipcc" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f lud-hip-hipcc

