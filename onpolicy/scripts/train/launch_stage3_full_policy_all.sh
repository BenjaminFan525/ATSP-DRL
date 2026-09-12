#!/usr/bin/env bash
# A fresh, immutable three-arm study; all eight GPUs share one validator pool.
set -euo pipefail
FULL_POLICY_WORKSPACE=/data/fanyx/HKBZ-environment
FULL_POLICY_PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
FULL_POLICY_PRIOR="$FULL_POLICY_WORKSPACE/result/hkbz_train_logs/stage3_representation_all_20260908_r1/hot_updates/perf_20260908_r1/manifest.json"
FULL_POLICY_RUN=${1:-stage3_full_policy_all_20260909_r1}
FULL_POLICY_RECOVERY_PARENT=${2:-}
if [[ ! "$FULL_POLICY_RUN" =~ ^stage3_full_policy_all_[a-zA-Z0-9_]+$ ]]; then
    echo "Invalid study identifier" >&2
    exit 2
fi
FULL_POLICY_OUTPUT="$FULL_POLICY_WORKSPACE/result/hkbz_train_logs/$FULL_POLICY_RUN"
FULL_POLICY_UNIT="hkbz-${FULL_POLICY_RUN//_/-}"
test ! -e "$FULL_POLICY_OUTPUT"
cd "$FULL_POLICY_WORKSPACE"
if [[ -n "$FULL_POLICY_RECOVERY_PARENT" ]]; then
    test -f "$FULL_POLICY_RECOVERY_PARENT"
    "$FULL_POLICY_PYTHON" -B onpolicy/scripts/train/run_stage3_full_policy.py prepare \
        --prior "$FULL_POLICY_RECOVERY_PARENT" --output "$FULL_POLICY_OUTPUT" --recovery
else
    "$FULL_POLICY_PYTHON" -B onpolicy/scripts/train/run_stage3_full_policy.py prepare \
        --prior "$FULL_POLICY_PRIOR" --output "$FULL_POLICY_OUTPUT"
fi
systemd-run --user --unit="$FULL_POLICY_UNIT" \
    --property="WorkingDirectory=$FULL_POLICY_OUTPUT/source" \
    --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$FULL_POLICY_WORKSPACE" \
    --setenv="PYTHONPATH=$FULL_POLICY_OUTPUT/source" \
    --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
    --setenv=NUMEXPR_NUM_THREADS=1 --setenv=MALLOC_ARENA_MAX=2 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=TimeoutStopSec=60 --property=RuntimeMaxSec=864000 \
    --property=LimitNOFILE=65536 --property=TasksMax=512 --property=OOMPolicy=kill \
    --property=CPUAffinity=0-127 --property=MemoryHigh=96G --property=MemoryMax=108G --property=MemorySwapMax=0 \
    --property="StandardOutput=append:$FULL_POLICY_OUTPUT/controller.log" \
    --property="StandardError=append:$FULL_POLICY_OUTPUT/controller.log" \
    "$FULL_POLICY_PYTHON" -B -u "$FULL_POLICY_OUTPUT/source/onpolicy/scripts/train/run_stage3_full_policy.py" \
    run --manifest "$FULL_POLICY_OUTPUT/manifest.json"
systemctl --user show "$FULL_POLICY_UNIT.service" -p ActiveState -p SubState -p MainPID -p MemoryHigh -p MemoryMax
