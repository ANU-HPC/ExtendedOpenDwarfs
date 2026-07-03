#!/usr/bin/env bash
set -euo pipefail

HOST="$(hostname -s)"
APP="${APP:-all}"
BACKEND="${BACKEND:-all}"
COMPILER="${COMPILER:-all}"
SIZE="${SIZE:-all}"
ITERS="${ITERS:-5}"

run_case() {
  local label="$1"
  shift

  echo
  echo "============================================================"
  echo "Paper run: host=$HOST device=$label"
  echo "APP=$APP BACKEND=$BACKEND COMPILER=$COMPILER SIZE=$SIZE ITERS=$ITERS"
  echo "============================================================"

  env DEVICE_LABEL="$label" "$@" \
    APP="$APP" BACKEND="$BACKEND" COMPILER="$COMPILER" SIZE="$SIZE" ITERS="$ITERS" \
    ./runner.sh --no-plots
}

case "$HOST" in
  trill)
    run_case "rtx5090" CUDA_VISIBLE_DEVICES=0
    run_case "w7800" HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  alpha)
    run_case "rtx5070ti" CUDA_VISIBLE_DEVICES=0
    run_case "rx7900xtx" HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  andoria)
    run_case "rtx4070ti" CUDA_VISIBLE_DEVICES=0
    run_case "rtx5070ti" CUDA_VISIBLE_DEVICES=1
    ;;

  epsilon)
    run_case "vega" HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    run_case "rx9070xt" HIP_VISIBLE_DEVICES=1 ROCR_VISIBLE_DEVICES=1 OPENCL_ARGS="-p 0 -d 1 -t 1 --"
    ;;

  beta)
    run_case "rx6800xt" HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    run_case "vega" HIP_VISIBLE_DEVICES=1 ROCR_VISIBLE_DEVICES=1 OPENCL_ARGS="-p 0 -d 1 -t 1 --"
    ;;

  zenith)
    run_case "rtx3090" CUDA_VISIBLE_DEVICES=0
    run_case "rx6800" HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  milan0)
    run_case "a100" CUDA_VISIBLE_DEVICES=0
    ;;

  hudson)
    run_case "h100" CUDA_VISIBLE_DEVICES=0
    ;;

  faraday)
    run_case "mi300a" HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  cousteau)
    run_case "mi100" HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  explorer)
    run_case "mi60" HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OPENCL_ARGS="-p 0 -d 0 -t 1 --"
    ;;

  *)
    echo "No paper profile for host '$HOST'; add one in scripts/run_scale_eod_paper.sh" >&2
    exit 1
    ;;
esac
