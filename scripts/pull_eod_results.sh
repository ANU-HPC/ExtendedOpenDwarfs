#!/usr/bin/env bash
set -euo pipefail

LOCAL_BASE="$HOME/Documents/2026/ExtendedOpenDwarfs/results"

HOSTS=(
  alpha
  epsilon
  beta
  andoria
  risa
)

mkdir -p "$LOCAL_BASE"

RSYNC_OPTS=(
  -az
  --info=stats1,name1
  --partial
)

echo "==> Pulling EOD results into $LOCAL_BASE"

for host in "${HOSTS[@]}"; do
  echo "==> $host"
  mkdir -p "$LOCAL_BASE/$host"

  rsync "${RSYNC_OPTS[@]}" \
    "beau@${host}:~/Documents/2026/ExtendedOpenDwarfs/results/" \
    "$LOCAL_BASE/$host/"
done

echo "==> excl"
mkdir -p "$LOCAL_BASE/excl"

rsync "${RSYNC_OPTS[@]}" \
  "9bj@excl:~/Documents/2026/ExtendedOpenDwarfs/results/" \
  "$LOCAL_BASE/excl/"

echo "==> Done."
