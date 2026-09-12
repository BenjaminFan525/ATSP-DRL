#!/usr/bin/env bash
set -euo pipefail
RECOVERY_WORKSPACE=/data/fanyx/HKBZ-environment
RECOVERY_STUDY="$RECOVERY_WORKSPACE/result/hkbz_train_logs/stage3_representation_all_20260908_r1"
RECOVERY_PROBE="$RECOVERY_STUDY/probes/b32_t16_20260908_r2"
RECOVERY_SOURCE="$RECOVERY_STUDY/hot_updates/perf_20260908_r1/source"
RECOVERY_MANIFEST="$RECOVERY_STUDY/hot_updates/perf_20260908_r1/manifest.json"
RECOVERY_ATTEMPT="$RECOVERY_STUDY/recovery/probe_guard_resume_20260908_r1"
RECOVERY_PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
test ! -e "$RECOVERY_ATTEMPT"
test ! -e "$RECOVERY_PROBE/resume_stage3_after_probe.py"
test -f "$RECOVERY_PROBE/run/configuration.json"
cp --no-clobber "$RECOVERY_WORKSPACE/onpolicy/scripts/train/resume_stage3_after_probe.py" "$RECOVERY_PROBE/"
cp --no-clobber "$RECOVERY_WORKSPACE/onpolicy/scripts/train/launch_stage3_probe_recovery.sh" "$RECOVERY_PROBE/"
cp --no-clobber "$RECOVERY_WORKSPACE/onpolicy/envs/HKBZ/test/test_stage3_probe_recovery.py" "$RECOVERY_PROBE/"
systemd-run --user --unit=hkbz-s3repr-probe-guard-resume-20260908-r1 \
  --property="WorkingDirectory=$RECOVERY_SOURCE" \
  --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$RECOVERY_WORKSPACE" --setenv="PYTHONPATH=$RECOVERY_SOURCE" \
  --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 --setenv=MALLOC_ARENA_MAX=2 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --property=Type=exec --property=Restart=no --property=KillMode=control-group \
  --property=TimeoutStopSec=60 --property=RuntimeMaxSec=864000 \
  --property=LimitNOFILE=65536 --property=TasksMax=512 \
  --property=CPUAffinity=0-127 --property=MemoryHigh=72G --property=MemoryMax=88G \
  --property="StandardOutput=append:$RECOVERY_PROBE/recovery_controller.log" \
  --property="StandardError=append:$RECOVERY_PROBE/recovery_controller.log" \
  "$RECOVERY_PYTHON" -B -u "$RECOVERY_PROBE/resume_stage3_after_probe.py" \
    --manifest "$RECOVERY_MANIFEST" --attempt "$RECOVERY_ATTEMPT" --probe-output "$RECOVERY_PROBE/run"
systemctl --user show hkbz-s3repr-probe-guard-resume-20260908-r1.service \
  -p ActiveState -p SubState -p MainPID -p MemoryHigh -p MemoryMax
