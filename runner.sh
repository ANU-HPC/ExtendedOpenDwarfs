#!/usr/bin/env bash
set -euo pipefail

APP="${APP:-nqueens}"
ITERS="${ITERS:-5}"
MODE="${MODE:-single}"
SIZE="${SIZE:-tiny}"

OPENCL_ARGS="${OPENCL_ARGS:--p 0 -d 0 -t 1 --}"

usage() {
  cat <<EOF
Usage:
  ./runner.sh [options]

Options:
  --app APP          Benchmark app, default: nqueens
  --size SIZE        tiny|small|medium|large|default
  --iters N          Repetitions per configuration, default: 5
  --full             Run tiny, small, medium, and large
  --sweep            Alias for --full
  --no-plots         Skip plot generation
  --help             Show this help

Environment:
  APP=nqueens SIZE=small ITERS=10 ./runner.sh
  MODE=full ./runner.sh
EOF
}

DO_PLOTS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app) APP="$2"; shift 2 ;;
    --size) SIZE="$2"; MODE="single"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --full|--sweep) MODE="full"; shift ;;
    --no-plots) DO_PLOTS=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$MODE" == "full" ]]; then
  SIZES=(tiny small medium large)
else
  SIZES=("$SIZE")
fi

echo "Running APP=$APP ITERS=$ITERS MODE=$MODE SIZES=${SIZES[*]}"

run_one() {
  local backend="$1"
  local compiler="$2"
  local size="$3"
  shift 3

  echo
  echo "==> APP=$APP SIZE=$size BACKEND=$backend COMPILER=$compiler"

  "$@" make clean \
    APP="$APP" \
    BACKEND="$backend" \
    COMPILER="$compiler" \
    SIZE="$size" \
    ITERS="$ITERS"

  "$@" make run \
    APP="$APP" \
    BACKEND="$backend" \
    COMPILER="$compiler" \
    SIZE="$size" \
    ITERS="$ITERS"
}

run_size() {
  local size="$1"

  echo
  echo "============================================================"
  echo "SIZE=$size"
  echo "============================================================"

  run_one opencl opencl "$size" env ARGS="$OPENCL_ARGS"

  if . ./setup-backends.sh >/dev/null && [[ "${BACKENDS:-}" == *"hip"* ]]; then
    run_one hip hipcc "$size" env
  fi

  if . ./setup-backends.sh >/dev/null && [[ "${BACKENDS:-}" == *"cuda"* ]]; then
    run_one cuda nvcc "$size" env
    run_one cuda scale-nvidia "$size" env
  fi

  if . ./setup-backends.sh >/dev/null && [[ -n "${HIP_DEV_TARGET:-}" ]]; then
    run_one cuda scale-amd "$size" env
  fi
}

for size in "${SIZES[@]}"; do
  run_size "$size"
done

echo
echo "Done. Results should be in ./results/"

if [[ "$DO_PLOTS" == "1" ]]; then
  echo "Generating plots in ./results/plots"
  pixi run plot-lsb
  tar -czf results/plots.tar.gz -C results plots
  echo "Done. Plots should be in ./results/plots"
fi
