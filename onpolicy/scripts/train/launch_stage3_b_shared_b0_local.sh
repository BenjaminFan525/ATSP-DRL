#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: launch_stage3_b_shared_b0_local.sh ABSOLUTE_MANIFEST [run|resume]' >&2
  exit 2
fi
STUDY_MANIFEST=$(realpath "$1")
STUDY_MODE=${2:-run}
[[ "$STUDY_MODE" == run || "$STUDY_MODE" == resume ]]
STUDY_DIR=$(dirname "$STUDY_MANIFEST")
STUDY_PYTHON=/home/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python
STUDY_WORKSPACE=/home/fanyx/HKBZ-environment
STUDY_UNIT="hkbz-bshared-b0-$(basename "$STUDY_DIR")-$(date +%s)"
systemd-run --user --unit="$STUDY_UNIT" --collect \
  --property=Type=exec --property=KillMode=control-group \
  --property=RuntimeMaxSec=1209600 --property=TimeoutStopSec=45 \
  --property=MemoryHigh=96G --property=MemoryMax=128G --property=MemorySwapMax=0 \
  --property='CPUAffinity=0-31 64-95' --property=AllowedCPUs=0-31,64-95 \
  --property="WorkingDirectory=$STUDY_DIR/source" \
  --property="StandardOutput=append:$STUDY_DIR/service.log" \
  --property="StandardError=append:$STUDY_DIR/service.log" \
  --setenv=CUDA_VISIBLE_DEVICES=GPU-744c1334-98c8-5318-e799-7ad15eea1fbf \
  --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$STUDY_WORKSPACE" \
  "$STUDY_PYTHON" -B -u "$STUDY_DIR/source/onpolicy/scripts/train/run_stage3_b_shared_b0.py" \
  "$STUDY_MODE" "$STUDY_MANIFEST"
printf '%s\n' "$STUDY_UNIT.service" > "$STUDY_DIR/last_service_unit.txt"
