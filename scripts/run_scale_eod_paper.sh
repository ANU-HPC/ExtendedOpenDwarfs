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
  echo "APP=$APP BACKEND=$BACKEND COMPILER=$COMPILER SIZE=$SIZE ITERS=$ITERS"
  echo "============================================================"

  env DEVICE_LABEL="$label" SCALE_ROOT="$SCALE_ROOT" "$@" \
    APP="$APP" BACKEND="$BACKEND" COMPILER="$COMPILER" SIZE="$SIZE" ITERS="$ITERS" \
    ./runner.sh --no-plots
}

case "$HOST" in
  trill)
    run_case "rtx5090" CUDA_VISIBLE_DEVICES=0
    run_case "w7800" HIP_DEV_TARGET=gfx1100 HIP_ARCH=gfx1100 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  alpha)
    run_case "rtx5070ti" CUDA_VISIBLE_DEVICES=0
    run_case "rx7900xtx" HIP_DEV_TARGET=gfx1100 HIP_ARCH=gfx1100 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  andoria)
    run_case "rtx4070ti" CUDA_VISIBLE_DEVICES=0
    run_case "rtx5070ti" CUDA_VISIBLE_DEVICES=1
    ;;

  epsilon)
    run_case "vega" HIP_DEV_TARGET=gfx900 HIP_ARCH=gfx900 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    run_case "rx9070xt" HIP_DEV_TARGET=gfx1201 HIP_ARCH=gfx1201 HIP_VISIBLE_DEVICES=1 ROCR_VISIBLE_DEVICES=1 OPENCL_ARGS="-p 0 -d 1 -t 1 --"
    ;;

  beta)
    run_case "rx6800xt" HIP_DEV_TARGET=gfx1030 HIP_ARCH=gfx1030 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    run_case "vega" HIP_DEV_TARGET=gfx900 HIP_ARCH=gfx900 HIP_VISIBLE_DEVICES=1 ROCR_VISIBLE_DEVICES=1 OPENCL_ARGS="-p 0 -d 1 -t 1 --"
    ;;

  zenith)
    run_case "rtx3090" CUDA_VISIBLE_DEVICES=0
    run_case "rx6800" HIP_DEV_TARGET=gfx1030 HIP_ARCH=gfx1030 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  milan0)
    run_case "a100" CUDA_VISIBLE_DEVICES=0
    ;;

  hudson)
    run_case "h100" CUDA_VISIBLE_DEVICES=0
    ;;

  faraday)
    run_case "mi300a" HIP_DEV_TARGET=gfx942 HIP_ARCH=gfx942 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  cousteau)
    run_case "mi100" HIP_DEV_TARGET=gfx908 HIP_ARCH=gfx908 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  explorer)
    run_case "mi60" HIP_DEV_TARGET=gfx906 HIP_ARCH=gfx906 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  troi)
    run_case "vega" HIP_DEV_TARGET=gfx900 HIP_ARCH=gfx900 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  *)
    echo "No paper profile for host '$HOST'; add one in scripts/run_scale_eod_paper.sh" >&2
    exit 1
    ;;
esac
