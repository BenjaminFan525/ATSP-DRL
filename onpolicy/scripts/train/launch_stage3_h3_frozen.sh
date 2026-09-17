#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Usage: launch_stage3_h3_frozen.sh ABSOLUTE_MANIFEST' >&2
  exit 2
fi
STUDY_MANIFEST=$(realpath "$1")
STUDY_DIR=$(dirname "$STUDY_MANIFEST")
STUDY_PYTHON=/home/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python
STUDY_UNIT="hkbz-h3-frozen-r0-$(date +%s)"
systemd-run --user --unit="$STUDY_UNIT" --collect \
  --property=Type=exec --property=KillMode=control-group \
  --property=RuntimeMaxSec=1209600 --property=TimeoutStopSec=45 \
  --property=MemoryHigh=102G --property=MemoryMax=114G --property=MemorySwapMax=0 \
  --property='CPUAffinity=0-31 64-95' --property=AllowedCPUs=0-31,64-95 \
  --property="WorkingDirectory=$STUDY_DIR/source" \
  --property="StandardOutput=append:$STUDY_DIR/service.log" \
  --property="StandardError=append:$STUDY_DIR/service.log" \
  --setenv=CUDA_VISIBLE_DEVICES=GPU-744c1334-98c8-5318-e799-7ad15eea1fbf \
  --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --setenv="MPLCONFIGDIR=$STUDY_DIR/mplconfig" \
  --setenv=HKBZ_STAGE3_WORKSPACE_ROOT=/home/fanyx/HKBZ-environment \
  "$STUDY_PYTHON" -B -u "$STUDY_DIR/source/onpolicy/scripts/train/run_stage3_h3_frozen.py" \
  run "$STUDY_MANIFEST"
printf '%s\n' "$STUDY_UNIT.service" > "$STUDY_DIR/last_service_unit.txt"
