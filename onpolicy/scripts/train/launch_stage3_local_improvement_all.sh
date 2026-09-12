#!/usr/bin/env bash
set -euo pipefail
LOCAL_IMPROVEMENT_WORKSPACE=/data/fanyx/HKBZ-environment
LOCAL_IMPROVEMENT_PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
LOCAL_IMPROVEMENT_RUN=${1:-stage3_local_improvement_all_20260910_r1}
if [[ ! "$LOCAL_IMPROVEMENT_RUN" =~ ^stage3_local_improvement_all_[a-zA-Z0-9_]+$ ]]; then
    echo "Invalid study identifier" >&2
    exit 2
fi
LOCAL_IMPROVEMENT_OUTPUT="$LOCAL_IMPROVEMENT_WORKSPACE/result/hkbz_train_logs/$LOCAL_IMPROVEMENT_RUN"
LOCAL_IMPROVEMENT_UNIT="hkbz-${LOCAL_IMPROVEMENT_RUN//_/-}"
test ! -e "$LOCAL_IMPROVEMENT_OUTPUT"
cd "$LOCAL_IMPROVEMENT_WORKSPACE"
"$LOCAL_IMPROVEMENT_PYTHON" -B onpolicy/scripts/train/run_stage3_local_improvement.py prepare \
    --prior "$LOCAL_IMPROVEMENT_WORKSPACE/result/hkbz_train_logs/stage3_full_policy_all_20260909_r1_recovery1/manifest.json" \
    --output "$LOCAL_IMPROVEMENT_OUTPUT"
systemd-run --user --unit="$LOCAL_IMPROVEMENT_UNIT" \
    --property="WorkingDirectory=$LOCAL_IMPROVEMENT_OUTPUT/source" \
    --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$LOCAL_IMPROVEMENT_WORKSPACE" \
    --setenv="PYTHONPATH=$LOCAL_IMPROVEMENT_OUTPUT/source" \
    --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
    --setenv=NUMEXPR_NUM_THREADS=1 --setenv=MALLOC_ARENA_MAX=2 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=TimeoutStopSec=60 --property=RuntimeMaxSec=864000 \
    --property=LimitNOFILE=65536 --property=TasksMax=512 --property=OOMPolicy=kill \
    --property=CPUAffinity=0-127 --property=MemoryHigh=96G --property=MemoryMax=108G --property=MemorySwapMax=0 \
    --property="StandardOutput=append:$LOCAL_IMPROVEMENT_OUTPUT/controller.log" \
    --property="StandardError=append:$LOCAL_IMPROVEMENT_OUTPUT/controller.log" \
    "$LOCAL_IMPROVEMENT_PYTHON" -B -u "$LOCAL_IMPROVEMENT_OUTPUT/source/onpolicy/scripts/train/run_stage3_local_improvement.py" \
    run --manifest "$LOCAL_IMPROVEMENT_OUTPUT/manifest.json"
systemctl --user show "$LOCAL_IMPROVEMENT_UNIT.service" -p ActiveState -p SubState -p MainPID -p MemoryHigh -p MemoryMax
