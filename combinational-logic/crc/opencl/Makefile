SHELL := /usr/bin/env bash

APP := crc

TOP_LEVEL ?= ../../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

CC := $(or $(CC),/usr/bin/gcc)

CFLAGS ?= -O3 -std=c99 -fPIE
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

OPENCL_CPPFLAGS ?= -D_GNU_SOURCE -D_DEFAULT_SOURCE -DOPENCL -DCL_TARGET_OPENCL_VERSION=120
CRC_CPPFLAGS ?= -D__constant=const
OPENCL_LDLIBS ?= -lOpenCL
EXE_LDFLAGS ?=

OCD_COMMON_ARGS_SRC ?= $(TOP_LEVEL)/include/common_args.c
OCD_OPTS_SRC ?= $(TOP_LEVEL)/opts/opts.c
OCD_RDTSC_SRC ?= $(TOP_LEVEL)/include/rdtsc.c

CRC_COMMON_SRC := ../common/crc_formats.c
CREATECRC_SRC := ../test/createcrc.c

INCLUDES := \
	-I. \
	-I.. \
	-I../include \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(TOP_LEVEL)/opts \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

CRC_OBJ := \
	crc_algo.o \
	crc_formats.o \
	common_args.o \
	opts.o \
	rdtsc.o

CREATECRC_OBJ := \
	createcrc.o \
	createcrc_formats.o

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help crc-opencl createcrc run-crc-opencl

all: crc-opencl createcrc

help:
	@echo "Build targets:"
	@echo "  make crc-opencl"
	@echo "  make createcrc"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-crc-opencl"

crc_algo.o: crc_algo.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(CRC_CPPFLAGS) $(INCLUDES) -c $< -o $@

crc_formats.o: $(CRC_COMMON_SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(CRC_CPPFLAGS) $(INCLUDES) -c $< -o $@

common_args.o: $(OCD_COMMON_ARGS_SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

opts.o: $(OCD_OPTS_SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

rdtsc.o: $(OCD_RDTSC_SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

crc-opencl: $(CRC_OBJ) crc_kernel.cl
	$(CC) \
		$(CFLAGS) \
		$(EXE_LDFLAGS) \
		-o $@ \
		$(CRC_OBJ) \
		$(LDFLAGS) \
		$(OPENCL_LDFLAGS) \
		$(LDLIBS) \
		$(OPENCL_LDLIBS) \
		-lm

createcrc: $(CREATECRC_OBJ)
	$(CC) \
		$(CFLAGS) \
		$(EXE_LDFLAGS) \
		-o $@ \
		$(CREATECRC_OBJ) \
		-lm

run-crc-opencl: crc-opencl
	mkdir -p "$(TOP_LEVEL)/results"
	cp crc_kernel.cl "$(TOP_LEVEL)/results/crc_kernel.cl"
	cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/crc-opencl" $(ARGS) $(N)

clean:
	rm -f \
		crc-opencl \
		createcrc \
		$(CRC_OBJ) \
		$(CREATECRC_OBJ)
