#!/usr/bin/env bash
set -euo pipefail
# Per-host, per-physical-GPU dispatch for the REGRESSION fleet (as invoked
# by scale-validation/ExtendedOpenDwarfs/regression/run-regression-fleet.sh).
#
# This is a sibling of run_scale_eod_paper.sh, not a replacement for it --
# run_scale_eod_paper.sh stays exactly as-is for reproducing the paper's
# original fixed-host results. This script targets the CURRENT regression
# fleet inventory (rescanned 2026-08-27), which has diverged from what
# run_scale_eod_paper.sh assumes in two ways worth calling out:
#   - "alpha" no longer exists under that name/hardware. It was NOT simply
#     renamed to "benzar" -- benzar's actual GPUs (2x RX 9070 XT + 1x
#     MI210) don't match what run_scale_eod_paper.sh's alpha) case
#     describes (RTX 5070 Ti + RX 7900 XTX) at all. Treat benzar as a new
#     host, not alpha's successor.
#   - epsilon and beta have each gained a second GPU (a Vega 56/64 on
#     each) that run_scale_eod_paper.sh's epsilon)/beta) cases don't know
#     about at all -- those cases' "single device, no cross-contamination
#     risk" comments are no longer accurate for the current hardware.
#
# Unlike run_scale_eod_paper.sh, this script does NOT try to avoid
# collecting the same GPU model from two different hosts (that script's
# andoria) case deliberately disabled its RTX 5070 Ti to avoid pooling
# with alpha's RTX 5070 Ti under one "rtx5070ti" device label). For the
# regression fleet, plot_lsb.R now pools by device model on purpose (see
# its `device` column) -- the same GPU model measured on two different
# boxes is treated as more samples of that GPU, not a conflict. The only
# case that mattered for this script's dead predecessor (alpha vs.
# andoria both having an RTX 5070 Ti) no longer exists now that alpha is
# gone, so every device below is collected without exception.
#
# NOT included here: the paper-only hosts (zenith, milan0, hudson,
# faraday, cousteau, explorer, troi) that appear in run_scale_eod_paper.sh
# but never showed up in the regression fleet scan. If/when one of those
# joins the regression fleet, add it the same way as any host below.
#
# SCALE_ROOT is intentionally NOT resolved with a per-host fallback path
# here (run_scale_eod_paper.sh hardcoded stale per-user absolute paths for
# this, which is exactly the kind of drift that made its alpha) case
# wrong). run-regression-fleet.sh's build_sweep_command always exports
# SCALE_ROOT before invoking this script (via ensure-scale.sh, or a
# distributed local build) -- if it's missing, something upstream broke
# and guessing a path would only hide that.
HOST="$(hostname -s)"
APP="${APP:-all}"
BACKEND="${BACKEND:-all}"
COMPILER="${COMPILER:-all}"
SIZE="${SIZE:-all}"
ITERS="${ITERS:-5}"
if [[ -z "${SCALE_ROOT:-}" ]]; then
  echo "error: SCALE_ROOT is not set. Expected run-regression-fleet.sh's build_sweep_command to have already exported it (via ensure-scale.sh or a distributed local build) before invoking this script." >&2
  exit 1
fi
run_case() {
  local label="$1"
  shift
  echo
  echo "============================================================"
  echo "Regression run: host=$HOST device=$label"
  echo "APP=$APP SIZE=$SIZE ITERS=$ITERS $*"
  echo "============================================================"
  env DEVICE_LABEL="$label" SCALE_ROOT="$SCALE_ROOT" "$@" \
    APP="$APP" SIZE="$SIZE" ITERS="$ITERS" \
    ./runner.sh --no-plots
}
case "$HOST" in
  trill)
    # 2x Radeon PRO W7800 (gfx1100) + 2x RTX 5090 (sm_120/compute 12.0).
    # Index 0 used for each -- the second card of each model is a
    # duplicate, not a different device, so it's not separately collected.
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
    # host (e.g. via clinfo) before enabling -- see run_scale_eod_paper.sh's
    # identical caveat.
    # run_case "rtx5090" BACKEND=opencl COMPILER=opencl OPENCL_ARGS="-p <nvidia_platform> -d <nvidia_device> -t 1 --"
    # run_case "w7800"   BACKEND=opencl COMPILER=opencl OPENCL_ARGS="-p <amd_platform> -d <amd_device> -t 1 --"
    ;;
  andoria)
    # RTX 4070 Ti (sm_89) + RTX 5070 Ti (sm_120/compute 12.0). Both
    # collected -- unlike run_scale_eod_paper.sh's andoria) case, there is
    # no other host with an RTX 5070 Ti in the current regression fleet
    # (alpha, the one that collided, no longer exists), so there's nothing
    # to pool ambiguously with.
    run_case "rtx4070ti" BACKEND=cuda COMPILER=nvcc \
      CUDA_DEV_TARGET=sm_89 CUDA_ARCH=89 CUDA_VISIBLE_DEVICES=0
    run_case "rtx4070ti" BACKEND=cuda COMPILER=scale-nvidia \
      CUDA_DEV_TARGET=sm_89 CUDA_ARCH=89 CUDA_VISIBLE_DEVICES=0
    run_case "rtx5070ti" BACKEND=cuda COMPILER=nvcc \
      CUDA_DEV_TARGET=sm_120 CUDA_ARCH=120 CUDA_VISIBLE_DEVICES=1
    run_case "rtx5070ti" BACKEND=cuda COMPILER=scale-nvidia \
      CUDA_DEV_TARGET=sm_120 CUDA_ARCH=120 CUDA_VISIBLE_DEVICES=1
    # TODO: OpenCL platform/device-index verification, as above.
    ;;
  benzar)
    # New host (not alpha's successor -- see file header). 2x RX 9070 XT
    # (gfx1201, rocm-smi GPU[0]/[1]) + 1x Instinct MI210 (gfx90a,
    # rocm-smi GPU[2]).
    run_case "rx9070xt" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx1201 HIP_ARCH=gfx1201 ROCR_VISIBLE_DEVICES=0
    run_case "rx9070xt" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx1201 HIP_ARCH=gfx1201 ROCR_VISIBLE_DEVICES=0
    run_case "mi210" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx90a HIP_ARCH=gfx90a ROCR_VISIBLE_DEVICES=2
    run_case "mi210" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx90a HIP_ARCH=gfx90a ROCR_VISIBLE_DEVICES=2
    # TODO: OpenCL platform/device-index verification, as above.
    ;;
  beta)
    # RX 6800 XT (gfx1030, GPU[0]) + Vega 56/64 (gfx900, GPU[1] -- newly
    # added here; run_scale_eod_paper.sh's beta) case predates this card).
    run_case "rx6800xt" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx1030 HIP_ARCH=gfx1030 ROCR_VISIBLE_DEVICES=0
    run_case "rx6800xt" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx1030 HIP_ARCH=gfx1030 ROCR_VISIBLE_DEVICES=0
    run_case "vega" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx900 HIP_ARCH=gfx900 ROCR_VISIBLE_DEVICES=1
    run_case "vega" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx900 HIP_ARCH=gfx900 ROCR_VISIBLE_DEVICES=1
    # TODO: OpenCL platform/device-index verification, as above.
    ;;
  delta)
    # New host. Single RX 6800 (gfx1030, GPU[0]) -- note this is "rx6800",
    # not "rx6800xt": same generation as beta's card but a distinct SKU,
    # so it's kept as its own device label rather than pooled with beta's.
    run_case "rx6800" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx1030 HIP_ARCH=gfx1030 ROCR_VISIBLE_DEVICES=0
    run_case "rx6800" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx1030 HIP_ARCH=gfx1030 ROCR_VISIBLE_DEVICES=0
    # TODO: OpenCL platform/device-index verification, as above.
    ;;
  epsilon)
    # Vega 56/64 (gfx900, GPU[0] -- newly added here, same caveat as
    # beta's) + RX 9070 XT (gfx1201, GPU[1] -- this one already existed in
    # run_scale_eod_paper.sh's epsilon) case at the same index).
    run_case "vega" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx900 HIP_ARCH=gfx900 ROCR_VISIBLE_DEVICES=0
    run_case "vega" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx900 HIP_ARCH=gfx900 ROCR_VISIBLE_DEVICES=0
    run_case "rx9070xt" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx1201 HIP_ARCH=gfx1201 ROCR_VISIBLE_DEVICES=1
    run_case "rx9070xt" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx1201 HIP_ARCH=gfx1201 ROCR_VISIBLE_DEVICES=1
    # TODO: OpenCL platform/device-index verification, as above.
    ;;
  gamma)
    # New host. RX 9070 XT (gfx1201, GPU[0]) + Radeon VII (gfx906,
    # GPU[1]) -- labelled "radeonvii", deliberately distinct from any
    # Instinct MI60 elsewhere: same die (gfx906) but a different, consumer
    # SKU, so it's not pooled with an MI60's numbers.
    run_case "rx9070xt" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx1201 HIP_ARCH=gfx1201 ROCR_VISIBLE_DEVICES=0
    run_case "rx9070xt" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx1201 HIP_ARCH=gfx1201 ROCR_VISIBLE_DEVICES=0
    run_case "radeonvii" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx906 HIP_ARCH=gfx906 ROCR_VISIBLE_DEVICES=1
    run_case "radeonvii" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx906 HIP_ARCH=gfx906 ROCR_VISIBLE_DEVICES=1
    # TODO: OpenCL platform/device-index verification, as above.
    ;;
  risa)
    # New host. 2x Radeon PRO W7800 (gfx1100) -- same model as trill's,
    # pools with trill's "w7800" results by device. Index 0 used; the
    # second card is a duplicate, not a different device.
    run_case "w7800" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx1100 HIP_ARCH=gfx1100 ROCR_VISIBLE_DEVICES=0
    run_case "w7800" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx1100 HIP_ARCH=gfx1100 ROCR_VISIBLE_DEVICES=0
    # TODO: OpenCL platform/device-index verification, as above.
    ;;
  tellar)
    # New host. Single Instinct MI210 (gfx90a) -- pools with benzar's
    # "mi210" results by device.
    run_case "mi210" BACKEND=hip COMPILER=hipcc \
      HIP_DEV_TARGET=gfx90a HIP_ARCH=gfx90a ROCR_VISIBLE_DEVICES=0
    run_case "mi210" BACKEND=cuda COMPILER=scale-amd \
      HIP_DEV_TARGET=gfx90a HIP_ARCH=gfx90a ROCR_VISIBLE_DEVICES=0
    # TODO: OpenCL platform/device-index verification, as above.
    ;;
  *)
    echo "No regression-fleet profile for host '$HOST'; add one in scripts/run_scale_eod_regression.sh" >&2
    exit 1
    ;;
esac
