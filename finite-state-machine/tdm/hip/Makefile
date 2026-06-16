SHELL := /usr/bin/env bash

APP := tdm

TOP_LEVEL ?= ../../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

HIPCC ?= hipcc
ROCM_PATH ?= /opt/rocm
HIP_DEV_TARGET ?= gfx942
HIP_ARCH ?= $(HIP_DEV_TARGET)

CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fPIC
HIPFLAGS ?= -O3 -std=c++17

CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

BUILD_DIR ?= build/hip

INCLUDES := \
	-I. \
	-I.. \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

HIP_OBJ := \
	$(BUILD_DIR)/tdm_hip.o \
	$(BUILD_DIR)/tdm_hip_kernel.o

CXX_OBJ := \
	$(BUILD_DIR)/dataio.o

OBJ := $(HIP_OBJ) $(CXX_OBJ)

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help tdm-hip-hipcc run-tdm-hip-hipcc

all: tdm-hip-hipcc

help:
	@echo "Build targets:"
	@echo "  make tdm-hip-hipcc"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-tdm-hip-hipcc"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/tdm_hip.o: tdm_hip.cpp ../types.h ../global.h ../dataio.h | $(BUILD_DIR)
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		--offload-arch=$(HIP_ARCH) \
		-x hip \
		-c $< \
		-o $@

$(BUILD_DIR)/tdm_hip_kernel.o: tdm_hip_kernel.hip ../types.h | $(BUILD_DIR)
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		--offload-arch=$(HIP_ARCH) \
		-x hip \
		-c $< \
		-o $@

$(BUILD_DIR)/dataio.o: ../dataio.cpp ../dataio.h ../types.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

tdm-hip-hipcc: $(OBJ)
	$(HIPCC) \
		$(HIPFLAGS) \
		--offload-arch=$(HIP_ARCH) \
		-o $@ \
		$(OBJ) \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

run-tdm-hip-hipcc:
	@if [[ ! -x "$(CURDIR)/tdm-hip-hipcc" ]]; then \
		$(MAKE) tdm-hip-hipcc; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/tdm-hip-hipcc" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f tdm-hip-hipcc
