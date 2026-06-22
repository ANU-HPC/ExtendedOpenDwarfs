SHELL := /usr/bin/env bash

TOP_LEVEL ?= ../../..
include $(TOP_LEVEL)/makefile_defs.mk

TARGET := csr-hip-hipcc
BUILD_DIR := build/hip

COMMON_C_OBJS := \
	$(BUILD_DIR)/sparse_formats.o \
	$(BUILD_DIR)/ziggurat.o \
	$(BUILD_DIR)/common.o \
	$(BUILD_DIR)/common_args.o \
	$(BUILD_DIR)/opts.o

HIP_OBJS := \
	$(BUILD_DIR)/csr.o

INCLUDES := \
	-I. \
	-I.. \
	-I../inc \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

COMMON_CPPFLAGS := $(CPPFLAGS) $(INCLUDES)

.PHONY: all clean run run-$(TARGET)

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/sparse_formats.o: ../src-common/sparse_formats.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/ziggurat.o: ../src-common/ziggurat.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/common.o: ../src-common/common.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -ffunction-sections -fdata-sections -Wno-deprecated-declarations -c $< -o $@

$(BUILD_DIR)/opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/csr.o: csr.cpp | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) $(COMMON_CPPFLAGS) --offload-arch=$(HIP_ARCH) -x hip -c $< -o $@

$(TARGET): $(COMMON_C_OBJS) $(HIP_OBJS)
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(HIP_ARCH) -o $@ $^ $(LDFLAGS) -Wl,--gc-sections $(LDLIBS) -lm

run: run-$(TARGET)

run-$(TARGET):
	@if [ ! -x "$(CURDIR)/$(TARGET)" ]; then \
		$(MAKE) $(TARGET); \
	fi
	mkdir -p $(TOP_LEVEL)/results
	cd $(TOP_LEVEL)/results && $(RUN_ENV) $(CURDIR)/$(TARGET) $(ARGS) $(N)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET)
