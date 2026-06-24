#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP="${APP:-all}"
ITERS="${ITERS:-5}"
MODE="${MODE:-single}"
SIZE="${SIZE:-tiny}"
BACKEND="${BACKEND:-all}"

usage() {
  cat <<EOF
Usage:
  ./runner.sh [options]

Options:
  --app APP          Benchmark app, default: all
  --backend BACKEND  all|opencl|cuda|hip, default: all
  --size SIZE        tiny|small|medium|large|default
  --iters N          Repetitions per configuration, default: 5
  --full             Run tiny, small, medium, and large
  --sweep            Alias for --full
  --no-plots         Skip plot generation
  --plots-only       Only regenerate plots from existing results
  --help             Show this help

Environment:
  APP=crc BACKEND=cuda SIZE=tiny ITERS=10 ./runner.sh
  APP=all BACKEND=all ./runner.sh --plots-only
  MODE=full ./runner.sh
EOF
}

DO_PLOTS=1
PLOTS_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app) APP="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --size) SIZE="$2"; MODE="single"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --full|--sweep) MODE="full"; shift ;;
    --no-plots) DO_PLOTS=0; shift ;;
    --plots-only) PLOTS_ONLY=1; DO_PLOTS=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

PLOT_APP="$APP"
PLOT_BACKEND="$BACKEND"

generate_plots() {
  echo
  echo "Generating plots in ./results/plots"
  echo "Plot filter: app=$PLOT_APP backend=$PLOT_BACKEND"

  rm -rf results/plots
  mkdir -p results/plots

  pixi run Rscript scripts/plot_lsb.R \
    results \
    results/plots \
    --app "$PLOT_APP" \
    --backend "$PLOT_BACKEND"

  local plot_tarball="results/plots.tar.gz"
  rm -f "$plot_tarball"

  if [[ -d results/plots ]] && find results/plots -mindepth 1 -print -quit | grep -q .; then
    tar -czf "$plot_tarball" -C results plots
    echo "Done. Plots should be in ./results/plots"
    echo "Done. Plot archive written to ./$plot_tarball"
  else
    echo "Warning: results/plots is empty; not creating plot archive" >&2
  fi
}

if [[ "$PLOTS_ONLY" == "1" ]]; then
  generate_plots
  exit 0
fi

case "$APP" in
  all) APP_LIST=(srad csr gem kmeans bfs hmm swat nw lud crc nqueens) ;;
  *) APP_LIST=("$APP") ;;
esac

case "$BACKEND" in
  all|opencl|cuda|hip) ;;
  *)
    echo "Unknown backend: $BACKEND" >&2
    usage
    exit 1
    ;;
esac

if [[ "$MODE" == "full" ]]; then
  SIZES=(tiny small medium large)
else
  SIZES=("$SIZE")
fi

echo "Running APPS=${APP_LIST[*]} BACKEND=$BACKEND ITERS=$ITERS MODE=$MODE SIZES=${SIZES[*]}"

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

  "$@" make build \
    APP="$APP" \
    BACKEND="$backend" \
    COMPILER="$compiler" \
    SIZE="$size" \
    ITERS="$ITERS"

  for ((iter = 1; iter <= ITERS; iter++)); do
    echo
    echo "==> APP=$APP SIZE=$size BACKEND=$backend COMPILER=$compiler iter=$iter/$ITERS"

    ODW_SKIP_MISSING_SIZE=1 "$@" scripts/odw.py run \
      --app "$APP" \
      --backend "$backend" \
      --compiler "$compiler" \
      --size "$size" \
      --iterations 1
  done
}

prepare_app() {
  local app="$1"

  case "$app" in
    crc)
      echo
      echo "==> Preparing CRC datasets"
      make -C combinational-logic/crc clean
      make -C combinational-logic/crc datasets
      ;;

    cfd)
      echo
      echo "==> Preparing CFD datasets"

      [[ -f test/unstructured-grids/cfd/128.dat ]] || \
        python3 scripts/generate_cfd_dataset.py \
          test/unstructured-grids/cfd/fvcorr.domn.193K \
          test/unstructured-grids/cfd/128.dat \
          128

      [[ -f test/unstructured-grids/cfd/1284.dat ]] || \
        python3 scripts/generate_cfd_dataset.py \
          test/unstructured-grids/cfd/fvcorr.domn.193K \
          test/unstructured-grids/cfd/1284.dat \
          1284

      [[ -f test/unstructured-grids/cfd/45056.dat ]] || \
        python3 scripts/generate_cfd_dataset.py \
          test/unstructured-grids/cfd/fvcorr.domn.193K \
          test/unstructured-grids/cfd/45056.dat \
          45056

      [[ -f test/unstructured-grids/cfd/193474.dat ]] || \
        python3 scripts/generate_cfd_dataset.py \
          test/unstructured-grids/cfd/fvcorr.domn.193K \
          test/unstructured-grids/cfd/193474.dat \
          193474
      ;;
  esac
}

host_supports_backend() {
  local backend="$1"

  . ./setup-backends.sh >/dev/null

  case "$backend" in
    opencl)
      [[ "${BACKENDS:-}" == *"opencl"* ]]
      ;;
    cuda)
      [[ "${BACKENDS:-}" == *"cuda"* ]]
      ;;
    hip)
      [[ "${BACKENDS:-}" == *"hip"* ]]
      ;;
    scale-amd)
      [[ -n "${HIP_DEV_TARGET:-}" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

run_size() {
  local size="$1"

  echo
  echo "============================================================"
  echo "SIZE=$size"
  echo "============================================================"

  if [[ "$BACKEND" == "all" || "$BACKEND" == "opencl" ]]; then
    if host_supports_backend opencl; then
      run_one opencl opencl "$size" env
    else
      echo "Skipping OpenCL: not available on this host"
    fi
  fi

  if [[ "$BACKEND" == "all" || "$BACKEND" == "hip" ]]; then
    if host_supports_backend hip; then
      run_one hip hipcc "$size" env
    else
      echo "Skipping HIP: not available on this host"
    fi
  fi

  if [[ "$BACKEND" == "all" || "$BACKEND" == "cuda" ]]; then
    if host_supports_backend cuda; then
      run_one cuda nvcc "$size" env
      run_one cuda scale-nvidia "$size" env
    else
      echo "Skipping CUDA: not available on this host"
    fi

    if host_supports_backend scale-amd; then
      run_one cuda scale-amd "$size" env
    elif [[ "$BACKEND" == "cuda" ]]; then
      echo "Skipping SCALE AMD path: HIP target not available on this host"
    fi
  fi
}

for app in "${APP_LIST[@]}"; do
  APP="$app"
  prepare_app "$APP"

  for size in "${SIZES[@]}"; do
    run_size "$size"
  done
done

echo
echo "Done. Results should be in ./results/"

if [[ "$DO_PLOTS" == "1" ]]; then
  generate_plots
fi
