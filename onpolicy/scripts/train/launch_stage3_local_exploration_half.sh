#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/fanyx/HKBZ-environment
PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
SUITE=${1:?Usage: bash launch_stage3_local_exploration_half.sh ABSOLUTE_NEW_SUITE}
test -f "$SUITE/manifest.json"
test ! -e "$SUITE/status.json"
CODE_ROOT="$SUITE/source"
test -f "$CODE_ROOT/onpolicy/scripts/train/run_stage3_local_exploration.py"
for gpu in 0 1 2 3; do
  occupied=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -n "$occupied" ]]; then
    echo "GPU $gpu is occupied: $occupied; refusing collision" >&2
    exit 1
  fi
done
mkdir -p "$SUITE/logs"
UNIT="hkbz-s3local-$(basename "$SUITE")"
systemd-run --user --unit="$UNIT" \
  --property="WorkingDirectory=$CODE_ROOT" \
  --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$ROOT" --setenv="PYTHONPATH=$CODE_ROOT" \
  --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2,3 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 --setenv=MALLOC_ARENA_MAX=2 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --property=Type=exec --property=Restart=no --property=KillMode=control-group \
  --property=TimeoutStopSec=30 --property=RuntimeMaxSec=864000 \
  --property=LimitNOFILE=65536 --property=TasksMax=160 \
  --property=AllowedCPUs=0-31,64-95 --property=CPUAffinity=0-31,64-95 \
  --property=NUMAPolicy=bind --property=NUMAMask=0 \
  --property=MemoryHigh=32G --property=MemoryMax=40G \
  --property="StandardOutput=append:$SUITE/logs/controller.log" \
  --property="StandardError=append:$SUITE/logs/controller.log" \
  /usr/bin/taskset --cpu-list 0-31,64-95 "$PYTHON" -u \
  "$CODE_ROOT/onpolicy/scripts/train/run_stage3_local_exploration.py" run --manifest "$SUITE/manifest.json"
systemctl --user show "$UNIT.service" -p Id -p ActiveState -p SubState -p MainPID \
  -p NRestarts -p AllowedCPUs -p CPUAffinity -p MemoryMax
