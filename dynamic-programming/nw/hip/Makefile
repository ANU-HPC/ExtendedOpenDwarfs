SHELL := /usr/bin/env bash

APP := needle

TOP_LEVEL ?= ../../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

HIPCC ?= hipcc
ROCM_PATH ?= /opt/rocm
HIP_DEV_TARGET ?= gfx942
HIP_ARCH ?= $(HIP_DEV_TARGET)

HIPFLAGS ?= -O3 -std=c++17
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

BUILD_DIR ?= build/hip

INCLUDES := \
	-I. \
	-I.. \
	-I../opencl \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

HIP_OBJ := \
	$(BUILD_DIR)/needle.hip.o \
	$(BUILD_DIR)/needle_kernel.hip.o \
	$(BUILD_DIR)/needle_kernel_opt_gpu.hip.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help run-needle-hip-hipcc

all: needle-hip-hipcc

help:
	@echo "Build targets:"
	@echo "  make needle-hip-hipcc"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-needle-hip-hipcc"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/needle.hip.o: needle.cpp | $(BUILD_DIR)
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		--offload-arch=$(HIP_ARCH) \
		-x hip \
		-c $< \
		-o $@

$(BUILD_DIR)/needle_kernel.hip.o: needle_kernel.hip | $(BUILD_DIR)
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		--offload-arch=$(HIP_ARCH) \
		-x hip \
		-c $< \
		-o $@

$(BUILD_DIR)/needle_kernel_opt_gpu.hip.o: needle_kernel_opt_gpu.hip | $(BUILD_DIR)
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		--offload-arch=$(HIP_ARCH) \
		-x hip \
		-c $< \
		-o $@

needle-hip-hipcc: $(HIP_OBJ)
	$(HIPCC) \
		$(HIPFLAGS) \
		--offload-arch=$(HIP_ARCH) \
		-o $@ \
		$(HIP_OBJ) \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

run-needle-hip-hipcc:
	@if [[ ! -x "$(CURDIR)/needle-hip-hipcc" ]]; then \
		$(MAKE) needle-hip-hipcc; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/needle-hip-hipcc" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f needle-hip-hipcc
