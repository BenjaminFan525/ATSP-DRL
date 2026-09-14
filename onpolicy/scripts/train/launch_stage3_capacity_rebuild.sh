#!/usr/bin/env bash
set -euo pipefail
REBUILD_WORKSPACE=/data/fanyx/HKBZ-environment
REBUILD_ROOT="$REBUILD_WORKSPACE/result/hkbz_train_logs/stage3_representation_all_20260908_r1"
REBUILD_ATTEMPT="$REBUILD_ROOT/capacity_retest/uncapped_20260908_r1"
REBUILD_SOURCE="$REBUILD_ROOT/hot_updates/perf_20260908_r1/source"
REBUILD_PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
test ! -e "$REBUILD_ATTEMPT/rebuild_source"
mkdir "$REBUILD_ATTEMPT/rebuild_source"
cp --no-clobber "$REBUILD_WORKSPACE/onpolicy/scripts/train/rebuild_stage3_committed.py" "$REBUILD_ATTEMPT/rebuild_source/"
systemd-run --user --unit=hkbz-s3rebuild-capacity-20260908-r1 \
  --property="WorkingDirectory=$REBUILD_SOURCE" \
  --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$REBUILD_WORKSPACE" --setenv="PYTHONPATH=$REBUILD_SOURCE" \
  --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 --setenv=MALLOC_ARENA_MAX=2 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --property=Type=exec --property=Restart=no --property=KillMode=control-group \
  --property=TimeoutStopSec=45 --property=RuntimeMaxSec=7500 \
  --property=CPUAffinity=0-127 --property=TasksMax=256 \
  --property=MemoryHigh=30G --property=MemoryMax=36G --property=MemorySwapMax=0 \
  --property="StandardOutput=append:$REBUILD_ATTEMPT/rebuild.log" \
  --property="StandardError=append:$REBUILD_ATTEMPT/rebuild.log" \
  "$REBUILD_PYTHON" -B -u "$REBUILD_ATTEMPT/rebuild_source/rebuild_stage3_committed.py" suite \
    --manifest "$REBUILD_ROOT/hot_updates/perf_20260908_r1/manifest.json" \
    --pause-receipt "$REBUILD_ATTEMPT/pause_receipt.json" --output "$REBUILD_ATTEMPT/rebuild"
