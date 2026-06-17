SHELL := /usr/bin/env bash

APP := bfs

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
	$(BUILD_DIR)/bfs.o \
	$(BUILD_DIR)/bfs_kernel.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help bfs-hip-hipcc run-bfs-hip-hipcc

all: bfs-hip-hipcc

help:
	@echo "Build targets:"
	@echo "  make bfs-hip-hipcc"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-bfs-hip-hipcc"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/bfs.o: bfs.cpp | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) $(CPPFLAGS) $(INCLUDES) --offload-arch=$(HIP_ARCH) -x hip -c $< -o $@

$(BUILD_DIR)/bfs_kernel.o: bfs_kernel.hip | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) $(CPPFLAGS) $(INCLUDES) --offload-arch=$(HIP_ARCH) -x hip -c $< -o $@

bfs-hip-hipcc: $(OBJ)
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(HIP_ARCH) -o $@ $(OBJ) $(LDFLAGS) $(LDLIBS) -lm

run-bfs-hip-hipcc:
	@if [[ ! -x "$(CURDIR)/bfs-hip-hipcc" ]]; then $(MAKE) bfs-hip-hipcc; fi
	@mkdir -p "$(TOP_LEVEL)/results"
	@cd "$(TOP_LEVEL)/results" && $(RUN_ENV) "$(CURDIR)/bfs-hip-hipcc" $(ARGS) $(N)

clean:
	rm -rf build
	rm -f bfs-hip-hipcc
