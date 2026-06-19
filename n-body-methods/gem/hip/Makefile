SHELL := /usr/bin/env bash

TOP_LEVEL ?= ../../..
include $(TOP_LEVEL)/makefile_defs.mk

TARGET := gemnoui-hip-hipcc
BUILD_DIR := build/hip

COMMON_C_OBJS := \
	$(BUILD_DIR)/dump_vertices.o \
	$(BUILD_DIR)/floating_centers.o \
	$(BUILD_DIR)/radix_sort.o \
	$(BUILD_DIR)/read_pqr.o \
	$(BUILD_DIR)/vector_math.o \
	$(BUILD_DIR)/write_grid.o \
	$(BUILD_DIR)/extrapolate_bonds.o \
	$(BUILD_DIR)/populate_stats.o \
	$(BUILD_DIR)/read_msms.o \
	$(BUILD_DIR)/run_msms.o \
	$(BUILD_DIR)/write_avs.o \
	$(BUILD_DIR)/write_xyzr.o

COMMON_CXX_OBJS := \
	$(BUILD_DIR)/check_cmdline.o \
	$(BUILD_DIR)/estimate_a.o \
	$(BUILD_DIR)/gem_no_ui.o \
	$(BUILD_DIR)/open_pqr_run_msms.o

HIP_OBJS := \
	$(BUILD_DIR)/calculate_potential.o

INCLUDES := \
	-I. \
	-I.. \
	-I../include \
	-I../include/visualize/dialogs \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

COMMON_CPPFLAGS := $(CPPFLAGS) $(INCLUDES) -DNO_UI

.PHONY: all clean run run-$(TARGET) $(TARGET)

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: ../%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: ../%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/calculate_potential.o: calculate_potential.cpp | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) $(COMMON_CPPFLAGS) --offload-arch=$(HIP_ARCH) -x hip -c $< -o $@

$(TARGET): $(COMMON_C_OBJS) $(COMMON_CXX_OBJS) $(HIP_OBJS)
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(HIP_ARCH) -o $@ $^ $(LDFLAGS) $(LDLIBS) -lm

run: run-$(TARGET)

run-$(TARGET): $(TARGET)
	mkdir -p $(TOP_LEVEL)/results
	cd $(TOP_LEVEL)/results && $(RUN_ENV) $(CURDIR)/$(TARGET) $(ARGS) $(N)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET)
