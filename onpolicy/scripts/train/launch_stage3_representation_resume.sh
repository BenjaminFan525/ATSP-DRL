#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/fanyx/HKBZ-environment
PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
MANIFEST=${1:?Usage: bash launch_stage3_representation_resume.sh ABSOLUTE_RECOVERY_MANIFEST}
test -f "$MANIFEST"
SUITE=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$MANIFEST")
ATTEMPT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["recovery"]["attempt_id"])' "$MANIFEST")
ATTEMPT_DIR=$(dirname "$MANIFEST")
CODE_ROOT="$ATTEMPT_DIR/source"
test -f "$CODE_ROOT/onpolicy/scripts/train/run_stage3_representation.py"
test ! -e "$ATTEMPT_DIR/started.json"
mkdir -p "$ATTEMPT_DIR/logs"
UNIT="hkbz-s3repr-$(basename "$SUITE")-$ATTEMPT"
systemd-run --user --unit="$UNIT" \
  --property="WorkingDirectory=$CODE_ROOT" \
  --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$ROOT" --setenv="PYTHONPATH=$CODE_ROOT" \
  --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 \
  --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 --setenv=MALLOC_ARENA_MAX=2 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --property=Type=exec --property=Restart=no --property=KillMode=control-group \
  --property=TimeoutStopSec=30 --property=RuntimeMaxSec=864000 \
  --property=LimitNOFILE=65536 --property=TasksMax=512 \
  --property=AllowedCPUs=0-127 --property=CPUAffinity=0-127 \
  --property=MemoryHigh=72G --property=MemoryMax=88G \
  --property="StandardOutput=append:$ATTEMPT_DIR/logs/controller.log" \
  --property="StandardError=append:$ATTEMPT_DIR/logs/controller.log" \
  "$PYTHON" -u "$CODE_ROOT/onpolicy/scripts/train/run_stage3_representation.py" \
  run --manifest "$MANIFEST" --resume-fit
systemctl --user show "$UNIT.service" -p Id -p ActiveState -p SubState -p MainPID \
  -p NRestarts -p AllowedCPUs -p CPUAffinity -p MemoryMax
