#!/usr/bin/env bash
set -euo pipefail
HOST="$(hostname -s)"
APP="${APP:-all}"
BACKEND="${BACKEND:-all}"
COMPILER="${COMPILER:-all}"
SIZE="${SIZE:-all}"
ITERS="${ITERS:-5}"
SCALE_ROOT="${SCALE_ROOT:-}"
if [[ -z "$SCALE_ROOT" ]]; then
  case "$HOST" in
    zenith|milan0|hudson|faraday|cousteau|explorer)
      SCALE_ROOT="/home/9bj/Documents/2026/scale-1.7.1-Linux"
      ;;
    *)
      SCALE_ROOT="/home/beau/Documents/2026/scale-1.7.1-Linux"
      ;;
  esac
fi
run_case() {
  local label="$1"
  shift
  echo
  echo "============================================================"
  echo "Paper run: host=$HOST device=$label"
  echo "APP=$APP SIZE=$SIZE ITERS=$ITERS $*"
  echo "============================================================"
  env DEVICE_LABEL="$label" SCALE_ROOT="$SCALE_ROOT" "$@" \
    APP="$APP" SIZE="$SIZE" ITERS="$ITERS" \
    ./runner.sh --no-plots
}
case "$HOST" in
  trill)
    # NOTE: BACKEND=all previously meant every run_case call attempted
    # every backend regardless of the device it was labeling. On a mixed
    # NVIDIA+AMD box this silently mislabeled the OTHER card's results
    # under whichever device this call happened to be tagging. Each call
    # below is now restricted to exactly the backend+compiler pair valid
    # for the physical device it targets -- nothing is left to fall back
    # to a host-level default.
    run_case "rtx5090" BACKEND=cuda COMPILER=nvcc \
      CUDA_DEV_TARGET=sm_120 CUDA_ARCH=120 CUDA_VISIBLE_DEVICES=0
    run_case "rtx5090" BACKEND=cuda COMPILER=scale-nvidia \
      CUDA_DEV_TARGET=sm_120 CUDA_ARCH=120 CUDA_VISIBLE_DEVICES=0

    run_case "w7800" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx1100 HIP_ARCH=gfx1100 ROCR_VISIBLE_DEVICES=0
    # scale-amd is invoked via the CUDA code path in runner.sh (its SOURCE
    # is CUDA even though its TARGET is AMD ISA) -- BACKEND must be cuda
    # here to pass runner.sh's gate, but COMPILER=scale-amd still
    # correctly excludes nvcc/scale-nvidia from this call.
    run_case "w7800" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx1100 HIP_ARCH=gfx1100 ROCR_VISIBLE_DEVICES=0

    # TODO: verify OpenCL platform/device indices for each card on this
    # host (e.g. via clinfo) before re-enabling -- OPENCL_ARGS was
    # previously constant across both run_case calls, which likely means
    # every OpenCL run silently targeted the same physical device
    # regardless of which label it was tagged with.
    # run_case "rtx5090" BACKEND=opencl COMPILER=opencl OPENCL_ARGS="-p <nvidia_platform> -d <nvidia_device> -t 1 --"
    # run_case "w7800"   BACKEND=opencl COMPILER=opencl OPENCL_ARGS="-p <amd_platform> -d <amd_device> -t 1 --"
    ;;
  alpha)
    run_case "rtx5070ti" BACKEND=cuda COMPILER=nvcc \
      CUDA_DEV_TARGET=sm_120 CUDA_ARCH=120 CUDA_VISIBLE_DEVICES=0
    run_case "rtx5070ti" BACKEND=cuda COMPILER=scale-nvidia \
      CUDA_DEV_TARGET=sm_120 CUDA_ARCH=120 CUDA_VISIBLE_DEVICES=0

    run_case "rx7900xtx" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx1100 HIP_ARCH=gfx1100 ROCR_VISIBLE_DEVICES=0
    run_case "rx7900xtx" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx1100 HIP_ARCH=gfx1100 ROCR_VISIBLE_DEVICES=0

    # TODO: same OpenCL platform/device caveat as trill -- verify indices
    # before enabling per-device OpenCL collection here.
    ;;
  andoria)
    # This host has no HIP/ROCm at all in setup-backends.sh (BACKENDS=
    # "cuda,opencl", no HIP_DEV_TARGET default), so it isn't exposed to
    # the hip/scale-amd cross-contamination bug described above. It DOES
    # still have the OpenCL platform/device-index caveat: OPENCL_ARGS
    # wasn't previously varied between the two CUDA_VISIBLE_DEVICES
    # calls, so if OpenCL collection is added here, each call needs its
    # own explicit -d index matching CUDA_VISIBLE_DEVICES, verified via
    # clinfo rather than assumed.
    run_case "rtx4070ti" BACKEND=cuda COMPILER=nvcc \
      CUDA_DEV_TARGET=sm_89 CUDA_ARCH=89 CUDA_VISIBLE_DEVICES=0
    run_case "rtx4070ti" BACKEND=cuda COMPILER=scale-nvidia \
      CUDA_DEV_TARGET=sm_89 CUDA_ARCH=89 CUDA_VISIBLE_DEVICES=0

    # rtx5070ti on this host is intentionally NOT collected. alpha also
    # has an RTX 5070 Ti, and "device" is parsed purely from the GPU-model
    # token in the filename -- not which physical host it's plugged into.
    # Collecting the same device model on two hosts would silently pool
    # both machines' results together under one "rtx5070ti" column with
    # no way to tell them apart after the fact (this is exactly what was
    # happening here before this was disabled: andoria's missing
    # cuda/scale-nvidia results were being masked by alpha's). If andoria's
    # rtx5070ti specifically needs to be distinguished from alpha's someday,
    # that requires host-qualifying the device label (e.g. "rtx5070ti-andoria")
    # throughout the naming convention, not just here.
    # run_case "rtx5070ti" BACKEND=cuda COMPILER=nvcc \
    #   CUDA_DEV_TARGET=sm_120 CUDA_ARCH=120 CUDA_VISIBLE_DEVICES=1
    # run_case "rtx5070ti" BACKEND=cuda COMPILER=scale-nvidia \
    #   CUDA_DEV_TARGET=sm_120 CUDA_ARCH=120 CUDA_VISIBLE_DEVICES=1
    ;;
  epsilon)
    # Single AMD device on this host -- no cross-contamination risk since
    # there's nothing else present to be mislabeled as.
    run_case "rx9070xt" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx1201 HIP_ARCH=gfx1201 ROCR_VISIBLE_DEVICES=1 \
      OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    run_case "rx9070xt" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx1201 HIP_ARCH=gfx1201 ROCR_VISIBLE_DEVICES=1
    ;;
  beta)
    run_case "rx6800xt" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx1030 HIP_ARCH=gfx1030 ROCR_VISIBLE_DEVICES=0 \
      OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    run_case "rx6800xt" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx1030 HIP_ARCH=gfx1030 ROCR_VISIBLE_DEVICES=0
    ;;
  zenith)
    run_case "rtx3090" BACKEND=cuda COMPILER=nvcc \
      CUDA_DEV_TARGET=sm_86 CUDA_ARCH=86 CUDA_VISIBLE_DEVICES=0
    run_case "rtx3090" BACKEND=cuda COMPILER=scale-nvidia \
      CUDA_DEV_TARGET=sm_86 CUDA_ARCH=86 CUDA_VISIBLE_DEVICES=0

    run_case "rx6800" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx1030 HIP_ARCH=gfx1030 ROCR_VISIBLE_DEVICES=0
    run_case "rx6800" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx1030 HIP_ARCH=gfx1030 ROCR_VISIBLE_DEVICES=0

    # TODO: OpenCL platform/device caveat as above.
    ;;
  milan0)
    run_case "a100" BACKEND=cuda COMPILER=nvcc \
      CUDA_DEV_TARGET=sm_80 CUDA_ARCH=80 CUDA_VISIBLE_DEVICES=0
    run_case "a100" BACKEND=cuda COMPILER=scale-nvidia \
      CUDA_DEV_TARGET=sm_80 CUDA_ARCH=80 CUDA_VISIBLE_DEVICES=0
    ;;
  hudson)
    run_case "h100" BACKEND=cuda COMPILER=nvcc \
      CUDA_DEV_TARGET=sm_90 CUDA_ARCH=90 CUDA_VISIBLE_DEVICES=0
    run_case "h100" BACKEND=cuda COMPILER=scale-nvidia \
      CUDA_DEV_TARGET=sm_90 CUDA_ARCH=90 CUDA_VISIBLE_DEVICES=0
    ;;
  faraday)
    run_case "mi300a" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx942 HIP_ARCH=gfx942 ROCR_VISIBLE_DEVICES=0 \
      OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    run_case "mi300a" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx942 HIP_ARCH=gfx942 ROCR_VISIBLE_DEVICES=0
    ;;
  cousteau)
    run_case "mi100" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx908 HIP_ARCH=gfx908 ROCR_VISIBLE_DEVICES=0 \
      OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    run_case "mi100" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx908 HIP_ARCH=gfx908 ROCR_VISIBLE_DEVICES=0
    ;;
  explorer)
    run_case "mi60" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx906 HIP_ARCH=gfx906 ROCR_VISIBLE_DEVICES=0 \
      OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    run_case "mi60" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx906 HIP_ARCH=gfx906 ROCR_VISIBLE_DEVICES=0
    ;;
  troi)
    run_case "vega" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx900 HIP_ARCH=gfx900 ROCR_VISIBLE_DEVICES=0 \
      OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    run_case "vega" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx900 HIP_ARCH=gfx900 ROCR_VISIBLE_DEVICES=0
    ;;
  *)
    echo "No paper profile for host '$HOST'; add one in scripts/run_scale_eod_paper.sh" >&2
    exit 1
    ;;
esac
