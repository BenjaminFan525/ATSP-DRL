#!/usr/bin/env bash
set -euo pipefail
CONDITIONAL_STAGE3_WORKSPACE=/data/fanyx/HKBZ-environment
CONDITIONAL_STAGE3_PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
CONDITIONAL_STAGE3_RUN=${1:-stage3_conditional_improvement_all_20260911_r1}
if [[ ! "$CONDITIONAL_STAGE3_RUN" =~ ^stage3_conditional_improvement_all_[a-zA-Z0-9_]+$ ]]; then
    echo "Invalid experiment identifier" >&2
    exit 2
fi
CONDITIONAL_STAGE3_OUTPUT="$CONDITIONAL_STAGE3_WORKSPACE/result/hkbz_train_logs/$CONDITIONAL_STAGE3_RUN"
CONDITIONAL_STAGE3_UNIT="hkbz-${CONDITIONAL_STAGE3_RUN//_/-}"
test ! -e "$CONDITIONAL_STAGE3_OUTPUT"
cd "$CONDITIONAL_STAGE3_WORKSPACE"
"$CONDITIONAL_STAGE3_PYTHON" -B onpolicy/scripts/train/run_stage3_conditional_improvement.py prepare \
    --prior "$CONDITIONAL_STAGE3_WORKSPACE/result/hkbz_train_logs/stage3_local_improvement_all_20260910_r1/manifest.json" \
    --output "$CONDITIONAL_STAGE3_OUTPUT"
systemd-run --user --unit="$CONDITIONAL_STAGE3_UNIT" \
    --property="WorkingDirectory=$CONDITIONAL_STAGE3_OUTPUT/source" \
    --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$CONDITIONAL_STAGE3_WORKSPACE" \
    --setenv="PYTHONPATH=$CONDITIONAL_STAGE3_OUTPUT/source" \
    --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
    --setenv=NUMEXPR_NUM_THREADS=1 --setenv=MALLOC_ARENA_MAX=2 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=TimeoutStopSec=60 --property=RuntimeMaxSec=108000 \
    --property=LimitNOFILE=65536 --property=TasksMax=512 --property=OOMPolicy=kill \
    --property=CPUAffinity=0-127 --property=MemoryHigh=96G --property=MemoryMax=108G --property=MemorySwapMax=0 \
    --property="StandardOutput=append:$CONDITIONAL_STAGE3_OUTPUT/controller.log" \
    --property="StandardError=append:$CONDITIONAL_STAGE3_OUTPUT/controller.log" \
    "$CONDITIONAL_STAGE3_PYTHON" -B -u "$CONDITIONAL_STAGE3_OUTPUT/source/onpolicy/scripts/train/run_stage3_conditional_improvement.py" \
    run --manifest "$CONDITIONAL_STAGE3_OUTPUT/manifest.json"
systemctl --user show "$CONDITIONAL_STAGE3_UNIT.service" -p ActiveState -p SubState -p MainPID -p MemoryHigh -p MemoryMax
