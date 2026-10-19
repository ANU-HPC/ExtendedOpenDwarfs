#!/usr/bin/env bash
# Profile one benchmark configuration, to examine a slowdown or regression
# that the LSB timings flagged (e.g. a SCALE-vs-native gap). Not for
# collecting results: profiling perturbs timings, so output goes to
# profiles/, never results/.
#
#   APP=bfs COMPILER=scale-amd SIZE=tiny scripts/profile.sh
#   make profile APP=bfs COMPILER=hipcc SIZE=tiny
#
# Same environment as `make run` (CUDA_DEV_TARGET/CUDA_PATH for
# nvcc/scale-nvidia, HIP_DEV_TARGET for scale-amd, ROCM_PATH for hipcc,
# SCALE_ROOT for scale-*). Profiler used, by COMPILER:
#   scale-amd     SCALE_PROFILE + redscale_profiler (kernels / API / memops)
#   hipcc         rocprofv3 (override flags with ROCPROF_FLAGS)
#   nvcc, scale-nvidia  nsys
# Run on a host that has the target GPU, and check it is idle first.
set -e
trap 'echo "FAILED: $BASH_COMMAND" >&2' ERR
cd "$(dirname "$0")/.."

APP="${APP:?usage: APP=<app> COMPILER=<nvcc|scale-nvidia|scale-amd|hipcc> [SIZE=tiny] $0}"
COMPILER="${COMPILER:?usage: APP=<app> COMPILER=<nvcc|scale-nvidia|scale-amd|hipcc> [SIZE=tiny] $0}"
SIZE="${SIZE:-tiny}"
case "$COMPILER" in
nvcc | scale-nvidia | scale-amd) BACKEND=cuda ;;
hipcc) BACKEND=hip ;;
*) echo "unknown COMPILER: $COMPILER (expected nvcc, scale-nvidia, scale-amd or hipcc)" >&2; exit 1 ;;
esac

export ODW_PROFILE_DIR="${ODW_PROFILE_DIR:-$PWD/profiles/${APP}_${COMPILER}_${SIZE}}"
mkdir -p "$ODW_PROFILE_DIR"
make run APP="$APP" BACKEND="$BACKEND" COMPILER="$COMPILER" SIZE="$SIZE" ITERS=1
echo "Profile written to $ODW_PROFILE_DIR"
