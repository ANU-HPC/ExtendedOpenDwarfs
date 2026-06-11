SHELL := /usr/bin/env bash

APP := crc

TOP_LEVEL ?= ../../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

HIPCC ?= hipcc
ROCM_PATH ?= /opt/rocm
HIP_DEV_TARGET ?= gfx942
HIP_ARCH ?= $(HIP_DEV_TARGET)

CC ?= /usr/bin/gcc
CFLAGS ?= -O3 -std=c99 -fPIC
HIPFLAGS ?= -O3 -std=c++17

CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

CRC_COMMON_SRC := ../common/crc_formats.c

INCLUDES := \
	-I. \
	-I.. \
	-I../include \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

OBJ := \
	crc_algo.o \
	crc_kernel.o \
	crc_formats.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help crc-hip-hipcc run-crc-hip-hipcc

all: crc-hip-hipcc

help:
	@echo "Build targets:"
	@echo "  make crc-hip-hipcc"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-crc-hip-hipcc"

crc_formats.o: $(CRC_COMMON_SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

crc_algo.o: crc_algo.cpp
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		-c $< \
		-o $@

crc_kernel.o: crc_kernel.hip
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		--offload-arch=$(HIP_ARCH) \
		-x hip \
		-c $< \
		-o $@

crc-hip-hipcc: $(OBJ)
	$(HIPCC) \
		$(HIPFLAGS) \
		--offload-arch=$(HIP_ARCH) \
		-o $@ \
		$(OBJ) \
		$(LDFLAGS) \
		$(LDLIBS)

run-crc-hip-hipcc: crc-hip-hipcc
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/crc-hip-hipcc" $(ARGS) $(N)

clean:
	rm -f \
		crc-hip-hipcc \
		$(OBJ)
