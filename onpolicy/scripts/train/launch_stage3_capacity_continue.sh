#!/usr/bin/env bash
# Explicit recipe-version continuation; never overwrite prior study artifacts.
set -euo pipefail
CAPACITY_WORKSPACE=/data/fanyx/HKBZ-environment
CAPACITY_STUDY="$CAPACITY_WORKSPACE/result/hkbz_train_logs/stage3_representation_all_20260908_r1"
CAPACITY_ATTEMPT="$CAPACITY_STUDY/capacity_retest/uncapped_20260908_r1"
CAPACITY_BASE="$CAPACITY_STUDY/hot_updates/perf_20260908_r1"
CAPACITY_PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
CAPACITY_RECIPE=${1:?Supply b32_t8_c2048 or b32_t8_c4096 after completed normal and dense tests}
case "$CAPACITY_RECIPE" in b32_t8_c2048|b32_t8_c4096) ;; *) exit 2 ;; esac
CAPACITY_OUTPUT="$CAPACITY_STUDY/continuations/capacity_20260908_r1"
CAPACITY_DRIVER="$CAPACITY_ATTEMPT/continuation_source"
test ! -e "$CAPACITY_OUTPUT"
test ! -e "$CAPACITY_DRIVER"
test -f "$CAPACITY_ATTEMPT/probes/$CAPACITY_RECIPE/run/result.json"
test -f "$CAPACITY_ATTEMPT/probes/${CAPACITY_RECIPE}_dense/run/result.json"
test -f "$CAPACITY_ATTEMPT/pause_receipt.json"
mkdir "$CAPACITY_DRIVER"
cp --no-clobber "$CAPACITY_WORKSPACE/onpolicy/scripts/train/continue_stage3_capacity.py" "$CAPACITY_DRIVER/"
cp --no-clobber "$CAPACITY_WORKSPACE/onpolicy/scripts/train/stage3_capacity_protocol.py" "$CAPACITY_DRIVER/"
cp --no-clobber "$CAPACITY_WORKSPACE/onpolicy/scripts/train/launch_stage3_capacity_continue.sh" "$CAPACITY_DRIVER/"
cp --no-clobber "$CAPACITY_WORKSPACE/onpolicy/envs/HKBZ/test/test_stage3_capacity_continue.py" "$CAPACITY_DRIVER/"
cp --no-clobber "$CAPACITY_WORKSPACE/onpolicy/envs/HKBZ/test/test_stage3_capacity_protocol.py" "$CAPACITY_DRIVER/"
HKBZ_STAGE3_WORKSPACE_ROOT="$CAPACITY_WORKSPACE" PYTHONPATH="$CAPACITY_BASE/source" \
  "$CAPACITY_PYTHON" -B "$CAPACITY_DRIVER/continue_stage3_capacity.py" prepare \
  --manifest "$CAPACITY_BASE/manifest.json" \
  --normal "$CAPACITY_ATTEMPT/probes/$CAPACITY_RECIPE/run/result.json" \
  --dense "$CAPACITY_ATTEMPT/probes/${CAPACITY_RECIPE}_dense/run/result.json" \
  --boundaries "$CAPACITY_ATTEMPT/boundaries" --pause-receipt "$CAPACITY_ATTEMPT/pause_receipt.json" \
  --output "$CAPACITY_OUTPUT"
systemd-run --user --unit=hkbz-s3repr-capacity-20260908-r1 \
  --property="WorkingDirectory=$CAPACITY_BASE/source" \
  --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$CAPACITY_WORKSPACE" --setenv="PYTHONPATH=$CAPACITY_BASE/source" \
  --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 --setenv=MALLOC_ARENA_MAX=2 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --property=Type=exec --property=Restart=no --property=KillMode=control-group \
  --property=TimeoutStopSec=60 --property=RuntimeMaxSec=864000 \
  --property=LimitNOFILE=65536 --property=TasksMax=512 --property=OOMPolicy=kill \
  --property=CPUAffinity=0-127 --property=MemoryHigh=96G --property=MemoryMax=108G --property=MemorySwapMax=0 \
  --property="StandardOutput=append:$CAPACITY_OUTPUT/controller.log" \
  --property="StandardError=append:$CAPACITY_OUTPUT/controller.log" \
  "$CAPACITY_PYTHON" -B -u "$CAPACITY_DRIVER/continue_stage3_capacity.py" run --manifest "$CAPACITY_OUTPUT/manifest.json"
systemctl --user show hkbz-s3repr-capacity-20260908-r1.service \
  -p ActiveState -p SubState -p MainPID -p MemoryHigh -p MemoryMax
