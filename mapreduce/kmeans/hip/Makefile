SHELL := /usr/bin/env bash

APP := kmeans

TOP_LEVEL ?= ../../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

HIPCC ?= hipcc
HIP_DEV_TARGET ?= gfx942
HIP_ARCH ?= $(HIP_DEV_TARGET)

HIPFLAGS ?= -O3 -std=c++17
CFLAGS ?= -O3 -std=c99
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

BUILD_DIR ?= build/hip

C_CPPFLAGS := $(CPPFLAGS) -D_DEFAULT_SOURCE -D_GNU_SOURCE -include stdint.h

INCLUDES := \
	-I. \
	-I.. \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

COMMON_C_OBJS := \
	$(BUILD_DIR)/kmeans_main.o \
	$(BUILD_DIR)/cluster.o \
	$(BUILD_DIR)/getopt.o \
	$(BUILD_DIR)/kmeans_clustering.o \
	$(BUILD_DIR)/rmse.o

HIP_OBJS := \
	$(BUILD_DIR)/kmeans.o \
	$(BUILD_DIR)/kmeans_hip_kernel.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help kmeans-hip-hipcc run-kmeans-hip-hipcc

all: kmeans-hip-hipcc

help:
	@echo "Build targets:"
	@echo "  make kmeans-hip-hipcc"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-kmeans-hip-hipcc"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/kmeans_main.o: ../kmeans.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(C_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cluster.o: ../cluster.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(C_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/getopt.o: ../getopt.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(C_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/kmeans_clustering.o: ../kmeans_clustering.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(C_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rmse.o: ../rmse.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(C_CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/kmeans.o: kmeans.cpp | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) $(CPPFLAGS) $(INCLUDES) --offload-arch=$(HIP_ARCH) -x hip -c $< -o $@

$(BUILD_DIR)/kmeans_hip_kernel.o: kmeans_hip_kernel.hip | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) $(CPPFLAGS) $(INCLUDES) --offload-arch=$(HIP_ARCH) -x hip -c $< -o $@

kmeans-hip-hipcc: $(COMMON_C_OBJS) $(HIP_OBJS)
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(HIP_ARCH) -o $@ $^ $(LDFLAGS) $(LDLIBS) -lm

run-kmeans-hip-hipcc:
	@if [[ ! -x "$(CURDIR)/kmeans-hip-hipcc" ]]; then $(MAKE) kmeans-hip-hipcc; fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && $(RUN_ENV) "$(CURDIR)/kmeans-hip-hipcc" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f kmeans-hip-hipcc
