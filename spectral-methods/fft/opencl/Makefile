SHELL := /usr/bin/env bash

TOP_LEVEL ?= ../../..
include $(TOP_LEVEL)/makefile_defs.mk

TARGET := fft-opencl
BUILD_DIR := build/opencl

OBJS := \
	$(BUILD_DIR)/fft.o \
	$(BUILD_DIR)/common_args.o \
	$(BUILD_DIR)/opts.o \
	$(BUILD_DIR)/rdtsc.o

KERNELS := \
	fft_kernels.cl

INCLUDES := \
	-I. \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(TOP_LEVEL)/opts \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

COMMON_CPPFLAGS := $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES)

.PHONY: all clean run run-$(TARGET)

all: $(TARGET)

$(TARGET): $(OBJS) $(KERNELS)
	$(CXX) \
		$(CXXFLAGS) \
		-o $@ \
		$(OBJS) \
		$(LDFLAGS) \
		$(OPENCL_LDFLAGS) \
		$(LDLIBS) \
		$(OPENCL_LDLIBS) \
		-lm

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/fft.o: fft.cpp fft_kernels.cl | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rdtsc.o: $(OCD_RDTSC_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

run: run-$(TARGET)

run-$(TARGET):
	@if [[ ! -x "$(CURDIR)/$(TARGET)" ]]; then \
		$(MAKE) $(TARGET); \
	fi
	mkdir -p $(TOP_LEVEL)/results
	cp $(KERNELS) $(TOP_LEVEL)/results/fft_kernels.cl
	cd $(TOP_LEVEL)/results && $(RUN_ENV) $(CURDIR)/$(TARGET) $(ARGS) $(N)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET)
