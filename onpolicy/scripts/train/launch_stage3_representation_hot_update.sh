#!/usr/bin/env bash
set -euo pipefail
PERF_MANIFEST=${1:?Usage: bash launch_stage3_representation_hot_update.sh ABSOLUTE_MANIFEST}
PERF_PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
PERF_ATTEMPT=$(dirname "$PERF_MANIFEST")
PERF_SOURCE="$PERF_ATTEMPT/source"
PERF_ID=$(basename "$PERF_ATTEMPT")
test -f "$PERF_MANIFEST"
test -f "$PERF_SOURCE/onpolicy/scripts/train/hot_update_stage3_representation.py"
test ! -e "$PERF_ATTEMPT/started.json"
mkdir -p "$PERF_ATTEMPT/logs"
systemd-run --user --unit="hkbz-s3repr-hot-$PERF_ID" \
  --property="WorkingDirectory=$PERF_SOURCE" \
  --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=/data/fanyx/HKBZ-environment" --setenv="PYTHONPATH=$PERF_SOURCE" \
  --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 --setenv=MALLOC_ARENA_MAX=2 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --property=Type=exec --property=Restart=no --property=KillMode=control-group \
  --property=TimeoutStopSec=60 --property=RuntimeMaxSec=864000 \
  --property=LimitNOFILE=65536 --property=TasksMax=512 \
  --property=AllowedCPUs=0-127 --property=CPUAffinity=0-127 \
  --property=MemoryHigh=72G --property=MemoryMax=88G \
  --property="StandardOutput=append:$PERF_ATTEMPT/logs/controller.log" \
  --property="StandardError=append:$PERF_ATTEMPT/logs/controller.log" \
  "$PERF_PYTHON" -u "$PERF_SOURCE/onpolicy/scripts/train/hot_update_stage3_representation.py" run --manifest "$PERF_MANIFEST"
systemctl --user show "hkbz-s3repr-hot-$PERF_ID.service" -p ActiveState -p SubState -p MainPID
