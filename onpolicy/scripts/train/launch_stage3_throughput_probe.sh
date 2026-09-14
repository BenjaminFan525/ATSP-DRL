#!/usr/bin/env bash
set -euo pipefail
PROBE_WORKSPACE=/data/fanyx/HKBZ-environment
PROBE_STUDY="$PROBE_WORKSPACE/result/hkbz_train_logs/stage3_representation_all_20260908_r1"
PROBE_ATTEMPT="$PROBE_STUDY/probes/b32_t16_20260908_r2"
PROBE_MANIFEST="$PROBE_STUDY/hot_updates/perf_20260908_r1/manifest.json"
PROBE_SOURCE="$PROBE_STUDY/hot_updates/perf_20260908_r1/source"
PROBE_CHECKPOINT="$PROBE_STUDY/train/E2_T1_to960/models/episodes_000240.pt"
PROBE_PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
PROBE_GPU=GPU-ac97e983-8f9f-6ead-4da9-8c35ec0ef683
test -f "$PROBE_CHECKPOINT"
test -f "$PROBE_MANIFEST"
test ! -e "$PROBE_ATTEMPT"
mkdir -p "$PROBE_STUDY/probes"
mkdir "$PROBE_ATTEMPT"
cp --no-clobber "$PROBE_WORKSPACE/onpolicy/scripts/train/probe_stage3_throughput.py" "$PROBE_ATTEMPT/"
cp --no-clobber "$PROBE_WORKSPACE/onpolicy/envs/HKBZ/test/test_stage3_throughput_probe.py" "$PROBE_ATTEMPT/"
cp --no-clobber "$PROBE_WORKSPACE/onpolicy/scripts/train/launch_stage3_throughput_probe.sh" "$PROBE_ATTEMPT/"
systemd-run --user --unit=hkbz-s3probe-b32-t16-20260908-r2 \
  --property="WorkingDirectory=$PROBE_SOURCE" \
  --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$PROBE_WORKSPACE" --setenv="PYTHONPATH=$PROBE_SOURCE" \
  --setenv="CUDA_VISIBLE_DEVICES=$PROBE_GPU" --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 --setenv=MALLOC_ARENA_MAX=2 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --property=Type=exec --property=Restart=no --property=KillMode=control-group \
  --property=TimeoutStopSec=45 --property=RuntimeMaxSec=3600 \
  --property=LimitNOFILE=65536 --property=TasksMax=128 --property=OOMPolicy=kill \
  --property=CPUAffinity=11-17,75-81 \
  --property=MemoryHigh=20G --property=MemoryMax=24G --property=MemorySwapMax=0 \
  --property="StandardOutput=append:$PROBE_ATTEMPT/service.log" \
  --property="StandardError=append:$PROBE_ATTEMPT/service.log" \
  "$PROBE_PYTHON" -B -u "$PROBE_ATTEMPT/probe_stage3_throughput.py" \
    --manifest "$PROBE_MANIFEST" --checkpoint "$PROBE_CHECKPOINT" --output "$PROBE_ATTEMPT/run" \
    --gpu-uuid "$PROBE_GPU" --batch-trajectories 32 --tbptt-steps 16 --cache-mib 4096 \
    --max-seconds 3300 --host-headroom-gib 24
systemctl --user show hkbz-s3probe-b32-t16-20260908-r2.service \
  -p ActiveState -p SubState -p MainPID -p MemoryHigh -p MemoryMax -p MemorySwapMax
