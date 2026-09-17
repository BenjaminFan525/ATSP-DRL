#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Usage: launch_stage3_h3_canonical_matrix.sh ABSOLUTE_MANIFEST' >&2
  exit 2
fi
CANONICAL_MANIFEST=$(realpath "$1")
CANONICAL_DIR=$(dirname "$CANONICAL_MANIFEST")
CANONICAL_PYTHON=/home/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python
CANONICAL_UNIT="hkbz-h3-canonical-s0-e6-$(date +%s)"
if [[ -f "$CANONICAL_DIR/run_started.json" ]]; then
  echo 'This frozen matrix has already been started.' >&2
  exit 2
fi
systemd-run --user --unit="$CANONICAL_UNIT" \
  --property=Type=exec --property=KillMode=control-group \
  --property=RuntimeMaxSec=46800 --property=TimeoutStopSec=45 \
  --property=MemoryHigh=14G --property=MemoryMax=16G --property=MemorySwapMax=0 \
  --property='CPUAffinity=22-29 86-93' --property=AllowedCPUs=22-29,86-93 \
  --property=Nice=10 --property=CPUWeight=20 \
  --property="WorkingDirectory=$CANONICAL_DIR" \
  --property="StandardOutput=append:$CANONICAL_DIR/service.log" \
  --property="StandardError=append:$CANONICAL_DIR/service.log" \
  --setenv=CUDA_VISIBLE_DEVICES=GPU-744c1334-98c8-5318-e799-7ad15eea1fbf \
  --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --setenv="MPLCONFIGDIR=$CANONICAL_DIR/mplconfig" \
  --setenv=HKBZ_STAGE3_WORKSPACE_ROOT=/home/fanyx/HKBZ-environment \
  "$CANONICAL_PYTHON" -B -u "$CANONICAL_DIR/driver.py" run "$CANONICAL_MANIFEST"
printf '%s\n' "$CANONICAL_UNIT.service" > "$CANONICAL_DIR/service_unit.txt"
