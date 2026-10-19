include ./makefile_defs.mk

SHELL := /usr/bin/env bash

APP ?= nqueens
BACKEND ?= cuda
COMPILER ?= nvcc
SIZE ?= tiny
ITERS ?= 1

export BACKENDS=$(BACKEND)
# COMPILER is already defined above (?=); `export COMPILER=$(COMPILER)` here
# would make it a self-referential recursive variable and abort make.
export COMPILER

.PHONY: all build run profile clean ensure-liblsb install-liblsb clean-liblsb

all: build

build: ensure-liblsb
	. ./setup-backends.sh && \
	LSB_INSTALL_ROOT="$(LSB_INSTALL_ROOT)" \
	scripts/odw.py build \
		--app "$(APP)" \
		--backend "$(BACKEND)" \
		--compiler "$(COMPILER)"

run: ensure-liblsb build
	. ./setup-backends.sh && \
	LSB_INSTALL_ROOT="$(LSB_INSTALL_ROOT)" \
	scripts/odw.py run \
		--app "$(APP)" \
		--backend "$(BACKEND)" \
		--compiler "$(COMPILER)" \
		--size "$(SIZE)" \
		--iterations "$(ITERS)"

# Run one APP/COMPILER/SIZE under its profiler to examine a slowdown; output
# in profiles/ (see scripts/profile.sh). Timings from this run are not results.
profile:
	APP="$(APP)" COMPILER="$(COMPILER)" SIZE="$(SIZE)" scripts/profile.sh

clean:
	scripts/odw.py clean \
		--app "$(APP)" \
		--backend "$(BACKEND)"

LSB_STAMP := $(LSB_INSTALL_ROOT)/.installed

ensure-liblsb: $(LSB_STAMP)

install-liblsb: $(LSB_STAMP)

$(LSB_STAMP):
	@mkdir -p "$(dir $(LSB_SRC_DIR))"
	@if [ ! -d "$(LSB_SRC_DIR)/.git" ]; then \
		git clone "$(LSB_GIT_URL)" "$(LSB_SRC_DIR)"; \
	fi
	cd "$(LSB_SRC_DIR)" && \
		$(MAKE) distclean >/dev/null 2>&1 || true && \
		rm -rf .deps .libs tests/.deps tests/.libs && \
		autoreconf -fi && \
		env CC=/usr/bin/gcc CXX=/usr/bin/g++ \
		./configure --prefix="$(LSB_INSTALL_ROOT)" $(LSB_CONFIGURE_FLAGS) && \
		$(MAKE) CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
		$(MAKE) install
	@mkdir -p "$(LSB_INSTALL_ROOT)"
	@touch "$(LSB_STAMP)"

clean-liblsb:
	rm -rf "$(LSB_INSTALL_ROOT)"

clean-liblsb-all:
	rm -rf "$(LSB_SRC_DIR)" "$(TOP_LEVEL)/external/liblsb-install"
