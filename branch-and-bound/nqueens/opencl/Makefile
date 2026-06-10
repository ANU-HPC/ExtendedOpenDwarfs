SHELL := /usr/bin/env bash

APP := nqueens

TOP_LEVEL ?= ../../..
LSB_INSTALL_ROOT ?= $(TOP_LEVEL)/external/liblsb-install/$(shell hostname -s)

CC ?= gcc
CXX ?= g++

CFLAGS ?= -O3 -std=c99
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

OPENCL_CPPFLAGS ?= -DOPENCL -DCL_TARGET_OPENCL_VERSION=120
OPENCL_LDFLAGS ?=
OPENCL_LDLIBS ?= -lOpenCL

OCD_COMMON_ARGS_SRC ?= $(TOP_LEVEL)/include/common_args.c
OCD_OPTS_SRC ?= $(TOP_LEVEL)/opts/opts.c
OCD_RDTSC_SRC ?= $(TOP_LEVEL)/include/rdtsc.c

INCLUDES := \
	-I. \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(TOP_LEVEL)/opts \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

CXX_OBJ := main.o nqueen_cl.o nqueen_cpu.o
C_OBJ := common_args.o opts.o rdtsc.o
OBJ := $(CXX_OBJ) $(C_OBJ)

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help nqueens-opencl run-nqueens-opencl

all: nqueens-opencl

help:
	@echo "Build targets:"
	@echo "  make nqueens-opencl"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-nqueens-opencl"

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

nqueen_cl.o: nqueen_cl.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

nqueen_cpu.o: nqueen_cpu.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

common_args.o: $(OCD_COMMON_ARGS_SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

opts.o: $(OCD_OPTS_SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

rdtsc.o: $(OCD_RDTSC_SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -c $< -o $@

nqueens-opencl: $(OBJ) kernels_nqueens.cl
	$(CXX) \
		$(CXXFLAGS) \
		-o $@ \
		$(OBJ) \
		$(LDFLAGS) \
		$(OPENCL_LDFLAGS) \
		$(LDLIBS) \
		$(OPENCL_LDLIBS)

run-nqueens-opencl: nqueens-opencl
	mkdir -p "$(TOP_LEVEL)/results"
	cp kernels_nqueens.cl "$(TOP_LEVEL)/results/kernels_nqueens.cl"
	cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/nqueens-opencl" $(ARGS) $(N)

clean:
	rm -f nqueens-opencl $(OBJ)
