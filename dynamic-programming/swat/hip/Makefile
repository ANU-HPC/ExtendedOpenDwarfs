SHELL := /usr/bin/env bash

APP := swat

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
	-I$(TOP_LEVEL)/opts \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

HIP_OBJ := \
	$(BUILD_DIR)/swat.hip.o \
	$(BUILD_DIR)/kernels.hip.o

CXX_OBJ := \
	$(BUILD_DIR)/alignments.o \
	$(BUILD_DIR)/prints.o \
	$(BUILD_DIR)/sequences.o \
	$(BUILD_DIR)/param.o \
	$(BUILD_DIR)/timeRec.o

OBJ := $(HIP_OBJ) $(CXX_OBJ)

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help run-swat-hip-hipcc

all: swat-hip-hipcc

help:
	@echo "Build targets:"
	@echo "  make swat-hip-hipcc"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-swat-hip-hipcc"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/swat.hip.o: swat.cpp | $(BUILD_DIR)
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		--offload-arch=$(HIP_ARCH) \
		-x hip \
		-c $< \
		-o $@

$(BUILD_DIR)/kernels.hip.o: kernels.hip | $(BUILD_DIR)
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		--offload-arch=$(HIP_ARCH) \
		-x hip \
		-c $< \
		-o $@

$(BUILD_DIR)/alignments.o: ../alignments.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/prints.o: ../prints.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/sequences.o: ../sequences.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/param.o: ../param.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/timeRec.o: ../timeRec.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

swat-hip-hipcc: $(OBJ)
	$(HIPCC) \
		$(HIPFLAGS) \
		--offload-arch=$(HIP_ARCH) \
		-o $@ \
		$(OBJ) \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

run-swat-hip-hipcc:
	@if [[ ! -x "$(CURDIR)/swat-hip-hipcc" ]]; then \
		$(MAKE) swat-hip-hipcc; \
	fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/swat-hip-hipcc" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f swat-hip-hipcc
