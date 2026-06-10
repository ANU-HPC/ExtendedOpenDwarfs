SHELL := /usr/bin/env bash

APP := crc

TOP_LEVEL ?= ../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

CC ?= /usr/bin/gcc

CFLAGS ?= -O3 -std=c99
CPPFLAGS ?=

OCD_COMMON_ARGS_SRC ?= $(TOP_LEVEL)/include/common_args.c
OCD_OPTS_SRC ?= $(TOP_LEVEL)/opts/opts.c
OCD_RDTSC_SRC ?= $(TOP_LEVEL)/include/rdtsc.c

INCLUDES := \
	-I. \
	-Iinclude \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(TOP_LEVEL)/opts \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

OBJ := \
	createcrc.o \
	crc_formats.o \
	common_args.o \
	opts.o \
	rdtsc.o

.PHONY: all clean help createcrc datasets

all: createcrc

help:
	@echo "Build targets:"
	@echo "  make createcrc"
	@echo "  make datasets"
	@echo ""
	@echo "Environment:"
	@echo "  TOP_LEVEL=$(TOP_LEVEL)"
	@echo "  CC=$(CC)"

createcrc.o: test/createcrc.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

crc_formats.o: common/crc_formats.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

common_args.o: $(OCD_COMMON_ARGS_SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) -D_GNU_SOURCE -D_DEFAULT_SOURCE -DOPENCL -DCL_TARGET_OPENCL_VERSION=120 $(INCLUDES) -c $< -o $@

opts.o: $(OCD_OPTS_SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

rdtsc.o: $(OCD_RDTSC_SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) -D_GNU_SOURCE -D_DEFAULT_SOURCE -DOPENCL -DCL_TARGET_OPENCL_VERSION=120 $(INCLUDES) -c $< -o $@

createcrc: $(OBJ)
	$(CC) \
		$(CFLAGS) \
		-o $@ \
		$(OBJ) \
		-lm \
		-lOpenCL

CRC_DATASETS := \
	$(TOP_LEVEL)/test/combinational-logic/crc/crc_1000x2000.txt \
	$(TOP_LEVEL)/test/combinational-logic/crc/crc_1000x16000.txt \
	$(TOP_LEVEL)/test/combinational-logic/crc/crc_1000x524000.txt \
	$(TOP_LEVEL)/test/combinational-logic/crc/crc_1000x4194304.txt

datasets: $(CRC_DATASETS)

$(TOP_LEVEL)/test/combinational-logic/crc/crc_1000x2000.txt: | createcrc
	mkdir -p $(dir $@)
	./createcrc -n 1000 -s 2000 -f $@

$(TOP_LEVEL)/test/combinational-logic/crc/crc_1000x16000.txt: | createcrc
	mkdir -p $(dir $@)
	./createcrc -n 1000 -s 16000 -f $@

$(TOP_LEVEL)/test/combinational-logic/crc/crc_1000x524000.txt: | createcrc
	mkdir -p $(dir $@)
	./createcrc -n 1000 -s 524000 -f $@

$(TOP_LEVEL)/test/combinational-logic/crc/crc_1000x4194304.txt: | createcrc
	mkdir -p $(dir $@)
	./createcrc -n 1000 -s 4194304 -f $@

clean:
	rm -f \
		createcrc \
		$(OBJ)
