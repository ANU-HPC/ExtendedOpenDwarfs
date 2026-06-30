SHELL := /usr/bin/env bash

TOP_LEVEL ?= ../../..
include $(TOP_LEVEL)/makefile_defs.mk

TARGET := cwt-hip-hipcc
BUILD_DIR := build/hip

HIP_OBJ := $(BUILD_DIR)/cwt.hipcc.o

INCLUDES := \
	-I. \
	-I.. \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

COMMON_CPPFLAGS := $(CPPFLAGS) $(INCLUDES)

.PHONY: all clean run run-$(TARGET)

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/cwt.hipcc.o: cwt.hip.cpp | $(BUILD_DIR)
	$(HIPCC) \
		$(HIPFLAGS) \
		$(COMMON_CPPFLAGS) \
		--offload-arch=$(HIP_ARCH) \
		-c $< \
		-o $@

$(TARGET): $(HIP_OBJ)
	$(HIPCC) \
		$(HIPFLAGS) \
		--offload-arch=$(HIP_ARCH) \
		-o $@ \
		$^ \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

run: run-$(TARGET)

run-$(TARGET): $(TARGET)
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && $(RUN_ENV) "$(CURDIR)/$(TARGET)" $(ARGS) $(N)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET)
