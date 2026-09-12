#!/usr/bin/env bash
# Two logical policies; all eight training ranks share one two-worker validator pool.
set -euo pipefail
FULL_DATA_WORKSPACE=/data/fanyx/HKBZ-environment
FULL_DATA_PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
FULL_DATA_PRIOR="$FULL_DATA_WORKSPACE/result/hkbz_train_logs/stage3_conditional_improvement_all_20260911_r1_recovery1/manifest.json"
FULL_DATA_RUN=${1:-stage3_full_data_all_20260912_r1}
FULL_DATA_PREPARED=${2:-}
if [[ ! "$FULL_DATA_RUN" =~ ^stage3_full_data_all_[a-zA-Z0-9_]+$ ]]; then
    echo "Invalid full-data study identifier" >&2
    exit 2
fi
FULL_DATA_OUTPUT="$FULL_DATA_WORKSPACE/result/hkbz_train_logs/$FULL_DATA_RUN"
FULL_DATA_UNIT="hkbz-${FULL_DATA_RUN//_/-}"
cd "$FULL_DATA_WORKSPACE"
if [[ "$FULL_DATA_PREPARED" != "--prepared" ]]; then
    test ! -e "$FULL_DATA_OUTPUT"
    "$FULL_DATA_PYTHON" -B onpolicy/scripts/train/run_stage3_full_data.py prepare \
        --prior "$FULL_DATA_PRIOR" --output "$FULL_DATA_OUTPUT"
    echo "Prepared $FULL_DATA_OUTPUT; run source-matched CPU/GPU tests and seal-tests before --prepared launch."
    exit 0
fi
test -f "$FULL_DATA_OUTPUT/verification.json"
test ! -e "$FULL_DATA_OUTPUT/status.json"
systemd-run --user --unit="$FULL_DATA_UNIT" \
    --property="WorkingDirectory=$FULL_DATA_OUTPUT/source" \
    --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$FULL_DATA_WORKSPACE" \
    --setenv="PYTHONPATH=$FULL_DATA_OUTPUT/source" \
    --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
    --setenv=NUMEXPR_NUM_THREADS=1 --setenv=MALLOC_ARENA_MAX=2 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=TimeoutStopSec=60 --property=RuntimeMaxSec=1209600 \
    --property=LimitNOFILE=65536 --property=TasksMax=512 --property=OOMPolicy=kill \
    --property=CPUAffinity=0-127 --property=MemoryHigh=96G --property=MemoryMax=108G --property=MemorySwapMax=0 \
    --property="StandardOutput=append:$FULL_DATA_OUTPUT/controller.log" \
    --property="StandardError=append:$FULL_DATA_OUTPUT/controller.log" \
    "$FULL_DATA_PYTHON" -B -u "$FULL_DATA_OUTPUT/source/onpolicy/scripts/train/run_stage3_full_data.py" \
    run --manifest "$FULL_DATA_OUTPUT/manifest.json"
systemctl --user show "$FULL_DATA_UNIT.service" -p ActiveState -p SubState -p MainPID -p MemoryHigh -p MemoryMax
