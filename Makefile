include ./makefile_defs.mk

SHELL := /usr/bin/env bash

APP ?= nqueens
BACKEND ?= cuda
COMPILER ?= nvcc
SIZE ?= tiny
ITERS ?= 1

.PHONY: all build run clean ensure-liblsb install-liblsb clean-liblsb

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
