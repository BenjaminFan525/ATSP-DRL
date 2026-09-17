#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: launch_stage3_b_shared_b0_throughput.sh MANIFEST RUNTIME' >&2; exit 2; }
STUDY_MANIFEST=$(realpath "$1")
STUDY_RUNTIME=$(realpath "$2")
STUDY_DIR=$(dirname "$STUDY_MANIFEST")
STUDY_PYTHON=/home/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python
STUDY_ADAPTER=$("$STUDY_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["launcher"])' "$STUDY_RUNTIME")
STUDY_MODULE=$("$STUDY_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["module"])' "$STUDY_RUNTIME")
STUDY_UNIT="hkbz-b0-throughput-$(date +%s)"
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
  --setenv=HKBZ_STAGE3_WORKSPACE_ROOT=/home/fanyx/HKBZ-environment \
  "$STUDY_PYTHON" -B -u "$STUDY_ADAPTER" resume "$STUDY_MANIFEST" \
  --module "$STUDY_MODULE" --runtime "$STUDY_RUNTIME"
printf '%s\n' "$STUDY_UNIT.service" > "$STUDY_DIR/last_service_unit.txt"
