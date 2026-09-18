#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"

echo "Setting up CUDA/HIP/OpenCL backends"

append_ld_library_path() {
       local dir="$1"
       [ -n "$dir" ] || return 0
       [ -d "$dir" ] || return 0

       case ":${LD_LIBRARY_PATH:-}:" in
       *":$dir:"*) ;;
       *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$dir" ;;
       esac
}

prepend_path() {
       local dir="$1"
       [ -n "$dir" ] || return 0
       [ -d "$dir" ] || return 0

       case ":${PATH:-}:" in
       *":$dir:"*) ;;
       *) export PATH="$dir${PATH:+:$PATH}" ;;
       esac
}

# Keep these clean. Do not inherit generated compiler/linker/include flags.
unset CPPFLAGS
unset LDFLAGS
unset LDLIBS
unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset LIBRARY_PATH

# Keep host compilation stable across NVHPC module environments.
# Some systems export CC=nvc/CXX=nvc++, which produces host objects that
# require NVHPC runtime libraries when linked by nvcc. Default to GCC unless
# explicitly overridden for this workflow.
# TODO: Don't we want to use the matching host compiler when we can?
export CC="${ODW_HOST_CC:-/usr/bin/gcc}"
export CXX="${ODW_HOST_CXX:-/usr/bin/g++}"

export BACKENDS="${BACKENDS:?Must set backends under test}"
export COMPILER="${COMPILER:?Must set compiler flavour}"

# CUDA setup
if [[ "${BACKENDS}" == *"cuda"* ]]; then
       export CUDA_DEV_TARGET="${CUDA_DEV_TARGET:?Must set CUDA_DEV_TARGET when cuda backend set}"
       export CUDA_PATH="${CUDA_PATH:?Must set CUDA_ROOT when cuda backend set}"
       export NVHPC_ROOT="${NVHPC_ROOT:-}" # TODO: this will use nvidia's library even when using scale, riiight?

       if [[ "${ODW_USE_SCALE:-0}" != "1" ]]; then
               export CUDA_HOME="$CUDA_PATH"
               export CUDA_TOOLKIT_ROOT_PATH="$CUDA_PATH"
               prepend_path "$CUDA_PATH/bin"
       fi

       if [[ -n "${NVHPC_ROOT:-}" ]]; then
               append_ld_library_path "$NVHPC_ROOT/compilers/lib"
               append_ld_library_path "$NVHPC_ROOT/math_libs/lib64"
               append_ld_library_path "$NVHPC_ROOT/comm_libs/nccl/lib"
               append_ld_library_path "$NVHPC_ROOT/comm_libs/nvshmem/lib"
       fi

       append_ld_library_path "$CUDA_PATH/lib64"


       export NVCC="${NVCC:-$CUDA_PATH/bin/nvcc}"
       export CUDA_NVCC="${CUDA_NVCC:-${NVCC}}"
       export CUDA_ARCH="${CUDA_DEV_TARGET#sm_}"
fi

# HIP/ROCm setup
if [[ "${BACKENDS}" == *"hip"* ]]; then
       export HIP_DEV_TARGET="${HIP_DEV_TARGET:?Must set HIP_DEV_TARGET when hip backend used}"
       export ROCM_PATH="${ROCM_PATH:?Must set ROCM_PATH when hip backend used}"
       export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"

       prepend_path "$ROCM_PATH/bin"
       append_ld_library_path "$ROCM_PATH/lib"
       append_ld_library_path "$ROCM_PATH/lib64"
       append_ld_library_path "$ROCM_PATH/llvm/lib"

       export HIPCC="${HIPCC:-${HIP_PATH}/bin/hipcc}"
       export HIP_ARCH="${HIP_ARCH:-$HIP_DEV_TARGET}"
fi

# OpenCL setup
if [[ "${BACKENDS}" == *"opencl"* ]]; then
       export OPENCL_CPPFLAGS="${OPENCL_CPPFLAGS:--DOPENCL -DCL_TARGET_OPENCL_VERSION=120 -I$OPENCL_INC_DIR}"
       export OPENCL_LDFLAGS="${OPENCL_LDFLAGS:--L$OPENCL_LIB_DIR -Xlinker -rpath -Xlinker $OPENCL_LIB_DIR}"
       export OPENCL_LDLIBS="${OPENCL_LDLIBS:--lOpenCL}"

       append_ld_library_path "$OPENCL_LIB_DIR"

       # Prefer ROCm OpenCL unless overridden.
    if ( [ -z "${OPENCL_INC_DIR:-}" ] || [ -z "${OPENCL_LIB_DIR}" ] ) && [ -z "${ROCM_PATH:-}" ]; then
        echo "When using opencl backend, must set either OPENCL_INC_DIR and OPENCL_LIB_DIR, or set ROCM_PATH"
        exit 1
    fi
       export OPENCL_INC_DIR="${OPENCL_INC_DIR:-$ROCM_PATH/include}"
       export OPENCL_LIB_DIR="${OPENCL_LIB_DIR:-$ROCM_PATH/lib}"
fi

# CUDA 12.9/13.0 / SCALE workarounds for hosts with newer host compilers
# (GCC 13+) than NVIDIA officially supports for this CUDA version. This
# is a compiler/CUDA-version interaction, not something tied to any one
# site or machine -- originally added for hudson (ORNL/ExCL), but trill
# and alpha (Spectral Compute) hit the identical cfd/nvcc STL-include-order
# failure (basic_string.h __noinline__ error) despite being on different
# hardware, sites, and Ubuntu versions. Applied unconditionally rather
# than gated by hostname.
# Keep these compiler-specific: SCALE needs libstdc++ paths; plain NVCC
# only needs the CFD include-order workaround.
case "${COMPILER:-}" in
scale-nvidia | scale-amd)
       if [[ -n "${SCALE_CPLUS_INCLUDE_PATH:-}" ]]; then
               export CPLUS_INCLUDE_PATH="$SCALE_CPLUS_INCLUDE_PATH"
       fi
       if [[ -n "${SCALE_LIBRARY_PATH:-}" ]]; then
               export LIBRARY_PATH="$SCALE_LIBRARY_PATH"
       fi
       if [[ -n "${SCALE_LD_LIBRARY_PATH:-}" ]]; then
               export LD_LIBRARY_PATH="$SCALE_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
       fi

       export NVCCFLAGS="${NVCCFLAGS:-}"
       case " $NVCCFLAGS " in
       *" -include include/odw_cuda_compat.h "*) ;;
       *)
               export NVCCFLAGS="${NVCCFLAGS:+$NVCCFLAGS }-include ${SCRIPT_DIR}/include/odw_cuda_compat.h"
               ;;
       esac
       ;;
nvcc)
       # Was previously gated on `[[ "${APP:-}" == "cfd" ]]`, but $APP is only
       # ever the specific app name when the caller drives one app per `make`
       # invocation (as runner.sh does). scale-validation's 03-test.sh calls
       # `make run APP=all ...` as a single invocation; odw.py then loops over
       # every app internally and shells out to `make <app>-cuda-nvcc` per app
       # without ever re-sourcing this script, so an app-gated check here can
       # never fire under APP=all regardless of host. Forcing these two
       # includes ahead of any .cu file's own includes is harmless for the
       # other apps, so apply unconditionally instead of gating on $APP.
       export NVCCFLAGS="${NVCCFLAGS:-}"
       case " $NVCCFLAGS " in
       *" -include string "*) ;;
       *) export NVCCFLAGS="${NVCCFLAGS:+$NVCCFLAGS }-include string" ;;
       esac
       case " $NVCCFLAGS " in
       *" -include fstream "*) ;;
       *) export NVCCFLAGS="${NVCCFLAGS:+$NVCCFLAGS }-include fstream" ;;
       esac
       ;;
esac

echo "  BACKENDS=$BACKENDS"
echo "  CUDA_DEV_TARGET=${CUDA_DEV_TARGET:-}"
echo "  CUDA_PATH=${CUDA_PATH:-}"
echo "  HIP_DEV_TARGET=${HIP_DEV_TARGET:-}"
echo "  ROCM_PATH=${ROCM_PATH:-}"
echo "  OPENCL_INC_DIR=${OPENCL_INC_DIR:-}"
echo "  OPENCL_LIB_DIR=${OPENCL_LIB_DIR:-}"
