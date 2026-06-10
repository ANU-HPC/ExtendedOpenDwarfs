SHELL := /usr/bin/env bash

APP := nqueens

SRC := main.cpp nqueen_hip.cpp nqueen_cpu.cpp kernels_nqueens.hip

TOP_LEVEL ?= ../../..

HIPCC ?= hipcc
ROCM_PATH ?= /opt/rocm
HIP_DEV_TARGET ?= gfx942
HIP_ARCH ?= $(HIP_DEV_TARGET)

HIPFLAGS ?= -O3 -std=c++17
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

INCLUDES := -I. -I$(TOP_LEVEL)/include

N ?=
ARGS ?=
RUN_ENV ?=

.PHONY: all clean help \
        nqueens-hip-hipcc \
        run-nqueens-hip-hipcc

all: nqueens-hip-hipcc

help:
	@echo "Build targets:"
	@echo "  make nqueens-hip-hipcc"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-nqueens-hip-hipcc"

nqueens-hip-hipcc: $(SRC)
	$(HIPCC) \
		$(HIPFLAGS) \
		$(CPPFLAGS) \
		$(INCLUDES) \
		--offload-arch=$(HIP_ARCH) \
		-x hip \
		-o $@ \
		main.cpp nqueen_hip.cpp nqueen_cpu.cpp kernels_nqueens.hip \
		$(LDFLAGS) \
		$(LDLIBS)

run-nqueens-hip-hipcc:
	mkdir -p "$(TOP_LEVEL)/results"
	cd "$(TOP_LEVEL)/results" && \
	$(RUN_ENV) "$(CURDIR)/nqueens-hip-hipcc" $(ARGS) $(N)

clean:
	rm -f nqueens-hip-hipcc
