#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
HOST="$(hostname -s)"
export HOST
export MACHINE="${MACHINE:-$HOST}"
export EXCL_HOST="$HOST"

echo "Setting up CUDA/HIP/OpenCL backends for $HOST"

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

# Keep these clean. Do not inherit SYCL-era generated compiler flags.
unset CPPFLAGS
unset LDFLAGS
unset LDLIBS

export BACKENDS="${BACKENDS:-}"
export CUDA_DEV_TARGET="${CUDA_DEV_TARGET:-}"
export HIP_DEV_TARGET="${HIP_DEV_TARGET:-}"
export SCALE_ROOT="${SCALE_ROOT:-$SCRIPT_DIR/scale-1.7.1-Linux}"

case "$HOST" in
  trill)
    # NVIDIA Blackwell + AMD RDNA3
    export BACKENDS="cuda,hip,opencl"
    export CUDA_DEV_TARGET="${CUDA_DEV_TARGET:-sm_120}"
    export HIP_DEV_TARGET="${HIP_DEV_TARGET:-gfx1100}"
    export MACHINE="Blackwell+RDNA3"

    export NVHPC_ROOT="${NVHPC_ROOT:-/usr/local}"
    export CUDA_PATH="${CUDA_PATH:-$NVHPC_ROOT/cuda-13.0}"

    export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.1.0}"
    export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"

    # Prefer ROCm OpenCL on mixed AMD/NVIDIA hosts unless overridden.
    export OPENCL_INC_DIR="${OPENCL_INC_DIR:-$ROCM_PATH/include}"
    export OPENCL_LIB_DIR="${OPENCL_LIB_DIR:-$ROCM_PATH/lib}"    ;;

  alpha)
    # NVIDIA Blackwell + AMD RDNA3
    export BACKENDS="cuda,hip,opencl"
    export CUDA_DEV_TARGET="${CUDA_DEV_TARGET:-sm_120}"
    export HIP_DEV_TARGET="${HIP_DEV_TARGET:-gfx1100}"
    export MACHINE="Blackwell+RDNA3"

    export NVHPC_ROOT="${NVHPC_ROOT:-/usr/local}"
    export CUDA_PATH="${CUDA_PATH:-$NVHPC_ROOT/cuda-13.0}"

    export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.1.0}"
    export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"

    # Prefer ROCm OpenCL on mixed AMD/NVIDIA hosts unless overridden.
    export OPENCL_INC_DIR="${OPENCL_INC_DIR:-$ROCM_PATH/include}"
    export OPENCL_LIB_DIR="${OPENCL_LIB_DIR:-$ROCM_PATH/lib}"    ;;

  epsilon)
    # AMD RX 9070 XT/gfx1201 + Vega/gfx900 
    export BACKENDS="hip,opencl"
    export HIP_DEV_TARGET="${HIP_DEV_TARGET:-gfx1201}"
    export MACHINE="AMD RX 9070 XT"

    export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.1.0}"
    export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"

    # Prefer ROCm OpenCL on mixed AMD/NVIDIA hosts unless overridden.
    export OPENCL_INC_DIR="${OPENCL_INC_DIR:-$ROCM_PATH/include}"
    export OPENCL_LIB_DIR="${OPENCL_LIB_DIR:-$ROCM_PATH/lib}"    ;;

  beta)
    # AMD Radeon RX 6800 XT gfx1030 + Vega/gfx900 
    export BACKENDS="hip"
    export HIP_DEV_TARGET="${HIP_DEV_TARGET:-gfx1201}"
    export MACHINE="AMD RX 6800 XT"

    export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.1.0}"
    export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"

    # Prefer ROCm OpenCL on mixed AMD/NVIDIA hosts unless overridden.
    export OPENCL_INC_DIR="${OPENCL_INC_DIR:-$ROCM_PATH/include}"
    export OPENCL_LIB_DIR="${OPENCL_LIB_DIR:-$ROCM_PATH/lib}"    ;;

  andoria)
    # RTX 4070 Ti + RTX 5070 Ti
    export BACKENDS="cuda,opencl"
    export CUDA_DEV_TARGET="${CUDA_DEV_TARGET:-sm_89}"
    export MACHINE="RTX 4070 Ti"

    export NVHPC_ROOT="${NVHPC_ROOT:-/usr/local}"
    export CUDA_PATH="${CUDA_PATH:-$NVHPC_ROOT/cuda-13.0}"

    # Prefer ROCm OpenCL on mixed AMD/NVIDIA hosts unless overridden.
    export OPENCL_INC_DIR="${OPENCL_INC_DIR:-$CUDA_PATH/include}"
    export OPENCL_LIB_DIR="${OPENCL_LIB_DIR:-$CUDA_PATH/lib}"    ;;

  milan2)
    # Tesla V100-PCIE-32GB
    export BACKENDS="cuda,opencl"
    export CUDA_DEV_TARGET="${CUDA_DEV_TARGET:-sm_70}"
    export MACHINE="${MACHINE:-V100}"

    export NVHPC_ROOT="${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5}"
    export CUDA_PATH="${CUDA_PATH:-$NVHPC_ROOT/cuda/12.9}"

    export OPENCL_INC_DIR="${OPENCL_INC_DIR:-$CUDA_PATH/include}"
    export OPENCL_LIB_DIR="${OPENCL_LIB_DIR:-$CUDA_PATH/lib64}"
    ;;

  milan0)
    # NVIDIA A100
    export BACKENDS="cuda"
    export CUDA_DEV_TARGET="${CUDA_DEV_TARGET:-sm_80}"
    export MACHINE="${MACHINE:-A100}"

    export NVHPC_ROOT="${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/24.5}"
    export CUDA_PATH="${CUDA_PATH:-$NVHPC_ROOT/cuda}"

    export OPENCL_INC_DIR="${OPENCL_INC_DIR:-$CUDA_PATH/include}"
    export OPENCL_LIB_DIR="${OPENCL_LIB_DIR:-$CUDA_PATH/lib64}"
    ;;

  hudson)
    # NVIDIA H100
    export BACKENDS="cuda"
    export CUDA_DEV_TARGET="${CUDA_DEV_TARGET:-sm_90}"
    export MACHINE="${MACHINE:-H100}"

    export NVHPC_ROOT="${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/2026}"
    export CUDA_PATH="${CUDA_PATH:-$NVHPC_ROOT/cuda}"

    export OPENCL_INC_DIR="${OPENCL_INC_DIR:-$CUDA_PATH/include}"
    export OPENCL_LIB_DIR="${OPENCL_LIB_DIR:-$CUDA_PATH/lib64}"
    ;;

  faraday)
    # AMD MI300A
    export BACKENDS="hip"
    export HIP_DEV_TARGET="${HIP_DEV_TARGET:-gfx942}"
    export MACHINE="${MACHINE:-MI300A}"

    export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"

    export OPENCL_INC_DIR="${OPENCL_INC_DIR:-$ROCM_PATH/include}"
    export OPENCL_LIB_DIR="${OPENCL_LIB_DIR:-$ROCM_PATH/lib}"
    ;;

  cousteau)
    # AMD MI100
    export BACKENDS="hip"
    export HIP_DEV_TARGET="${HIP_DEV_TARGET:-gfx908}"
    export MACHINE="${MACHINE:-MI100}"

    export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"

    export OPENCL_INC_DIR="${OPENCL_INC_DIR:-$ROCM_PATH/include}"
    export OPENCL_LIB_DIR="${OPENCL_LIB_DIR:-$ROCM_PATH/lib}"
    ;;

  zenith)
    # NVIDIA Ampere + AMD RDNA2
    export BACKENDS="cuda,hip"
    export CUDA_DEV_TARGET="${CUDA_DEV_TARGET:-sm_86}"
    export HIP_DEV_TARGET="${HIP_DEV_TARGET:-gfx1030}"
    export MACHINE="${MACHINE:-Ampere+RDNA2}"

    export NVHPC_ROOT="${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.3}"
    export CUDA_PATH="${CUDA_PATH:-$NVHPC_ROOT/cuda/13.1}"

    export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"

    # Prefer ROCm OpenCL on mixed AMD/NVIDIA hosts unless overridden.
    export OPENCL_INC_DIR="${OPENCL_INC_DIR:-$ROCM_PATH/include}"
    export OPENCL_LIB_DIR="${OPENCL_LIB_DIR:-$ROCM_PATH/lib}"
    ;;

  *)
    echo "Unknown system ($HOST)." >&2
    echo "Set CUDA_DEV_TARGET/HIP_DEV_TARGET, CUDA_PATH/ROCM_PATH manually or add this host." >&2
    exit 1
    ;;
esac

# CUDA setup
if [[ "${BACKENDS}" == *"cuda"* ]]; then
  if [[ -z "${CUDA_PATH:-}" && -x "$(command -v nvcc 2>/dev/null || true)" ]]; then
    export CUDA_PATH="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi

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
fi

# HIP/ROCm setup
if [[ "${BACKENDS}" == *"hip"* ]]; then
  if [[ -z "${ROCM_PATH:-}" && -x "$(command -v hipcc 2>/dev/null || true)" ]]; then
    export ROCM_PATH="$(dirname "$(dirname "$(command -v hipcc)")")"
  fi

  export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"

  prepend_path "$ROCM_PATH/bin"
  append_ld_library_path "$ROCM_PATH/lib"
  append_ld_library_path "$ROCM_PATH/lib64"
  append_ld_library_path "$ROCM_PATH/llvm/lib"

  export CC="${CC:-/usr/bin/gcc}"
  export CXX="${CXX:-/usr/bin/g++}"
  export HIPCC="$(command -v hipcc)"
fi

# OpenCL setup
if [[ "${BACKENDS}" == *"opencl"* ]]; then
  export OPENCL_CPPFLAGS="${OPENCL_CPPFLAGS:--DOPENCL -DCL_TARGET_OPENCL_VERSION=120 -I$OPENCL_INC_DIR}"
  export OPENCL_LDFLAGS="${OPENCL_LDFLAGS:--L$OPENCL_LIB_DIR -Xlinker -rpath -Xlinker $OPENCL_LIB_DIR}"
  export OPENCL_LDLIBS="${OPENCL_LDLIBS:--lOpenCL}"

  append_ld_library_path "$OPENCL_LIB_DIR"
fi

# Convenience aliases consumed by Make/Python.
if [[ "${BACKENDS}" == *"cuda"* ]]; then
  export CUDA_ARCH="${CUDA_ARCH:-${CUDA_DEV_TARGET#sm_}}"
fi

if [[ "${BACKENDS}" == *"hip"* ]]; then
  export HIP_ARCH="${HIP_ARCH:-$HIP_DEV_TARGET}"
fi

export CC="${CC:-$(command -v gcc || true)}"
export CXX="${CXX:-$(command -v g++ || true)}"
export NVCC="${NVCC:-$(command -v nvcc || true)}"
export CUDA_NVCC="${CUDA_NVCC:-${NVCC:-$(command -v nvcc || true)}}"
export HIPCC="${HIPCC:-$(command -v hipcc || true)}"

echo "  MACHINE=$MACHINE"
echo "  BACKENDS=$BACKENDS"
echo "  SCALE_ROOT=${SCALE_ROOT:-}"
echo "  CUDA_DEV_TARGET=${CUDA_DEV_TARGET:-}"
echo "  CUDA_PATH=${CUDA_PATH:-}"
echo "  HIP_DEV_TARGET=${HIP_DEV_TARGET:-}"
echo "  ROCM_PATH=${ROCM_PATH:-}"
echo "  OPENCL_INC_DIR=${OPENCL_INC_DIR:-}"
echo "  OPENCL_LIB_DIR=${OPENCL_LIB_DIR:-}"
