SHELL := /usr/bin/env bash

APP := bwa_hmm

TOP_LEVEL ?= ../../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

HIPCC ?= hipcc
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
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

OBJ := \
	$(BUILD_DIR)/main_bwa_hmm.o \
	$(BUILD_DIR)/bwa_hmm_hip_kernel.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help bwa_hmm-hip-hipcc run-bwa_hmm-hip-hipcc

all: bwa_hmm-hip-hipcc

help:
	@echo "Build targets:"
	@echo "  make bwa_hmm-hip-hipcc"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-bwa_hmm-hip-hipcc"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/main_bwa_hmm.o: main_bwa_hmm.cpp | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) $(CPPFLAGS) $(INCLUDES) --offload-arch=$(HIP_ARCH) -x hip -c $< -o $@

$(BUILD_DIR)/bwa_hmm_hip_kernel.o: bwa_hmm_hip_kernel.hip | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) $(CPPFLAGS) $(INCLUDES) --offload-arch=$(HIP_ARCH) -x hip -c $< -o $@

bwa_hmm-hip-hipcc: $(OBJ)
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(HIP_ARCH) -o $@ $(OBJ) $(LDFLAGS) $(LDLIBS) -lm

run-bwa_hmm-hip-hipcc:
	@if [[ ! -x "$(CURDIR)/bwa_hmm-hip-hipcc" ]]; then $(MAKE) bwa_hmm-hip-hipcc; fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && $(RUN_ENV) "$(CURDIR)/bwa_hmm-hip-hipcc" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f bwa_hmm-hip-hipcc
