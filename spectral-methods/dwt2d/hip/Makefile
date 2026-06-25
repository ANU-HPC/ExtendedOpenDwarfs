SHELL := /usr/bin/env bash

TOP_LEVEL ?= ../../..
include $(TOP_LEVEL)/makefile_defs.mk

TARGET := dwt2d-hip-hipcc
BUILD_DIR := build/hip

INCLUDES := \
        -I. \
        -I.. \
        -I$(TOP_LEVEL) \
        -I$(TOP_LEVEL)/include \
        -I$(LSB_INSTALL_ROOT)/include \
        -I$(LSB_INSTALL_ROOT)/lib/include

SRC := dwt2d.cpp

.PHONY: all clean run

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/dwt2d.o: $(SRC) | $(BUILD_DIR)
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-c $< \
		-o $@

$(TARGET): $(BUILD_DIR)/dwt2d.o
	$(HIPCC) \
		$(HIPFLAGS) \
		-o $@ \
		$^ \
		$(LDFLAGS) \
		$(LDLIBS) \
		-lm

run: run-$(TARGET)
		$(MAKE); \
	fi
	mkdir -p $(TOP_LEVEL)/results
	cd $(TOP_LEVEL)/results && \
	$(RUN_ENV) $(CURDIR)/$(TARGET) $(ARGS) $(N)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET)
