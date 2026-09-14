#!/usr/bin/env bash
set -euo pipefail
CAPACITY_WORKSPACE=/data/fanyx/HKBZ-environment
CAPACITY_ROOT="$CAPACITY_WORKSPACE/result/hkbz_train_logs/stage3_representation_all_20260908_r1"
CAPACITY_ATTEMPT="$CAPACITY_ROOT/capacity_retest/uncapped_20260908_r1"
CAPACITY_SOURCE="$CAPACITY_ROOT/hot_updates/perf_20260908_r1/source"
CAPACITY_MANIFEST="$CAPACITY_ROOT/hot_updates/perf_20260908_r1/manifest.json"
CAPACITY_CHECKPOINT="$CAPACITY_ROOT/train/E2_T1_to960/models/episodes_000240.pt"
CAPACITY_PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
CAPACITY_LABEL=${1:-b32_t16_c4096}
CAPACITY_GPU_INDEX=${2:-1}
case "$CAPACITY_GPU_INDEX" in
  1) CAPACITY_GPU=GPU-ac97e983-8f9f-6ead-4da9-8c35ec0ef683; CAPACITY_CPUS=11-17,75-81 ;;
  2) CAPACITY_GPU=GPU-fcfa9bd7-815a-4b25-d479-f883d406744a; CAPACITY_CPUS=18-24,82-88 ;;
  3) CAPACITY_GPU=GPU-9882611a-10cd-020d-2695-4df59ecedece; CAPACITY_CPUS=25-31,89-95 ;;
  *) exit 2 ;;
esac
case "$CAPACITY_LABEL" in
  b32_t16_c4096) CAPACITY_TBPTT=16; CAPACITY_CACHE=4096; CAPACITY_EXTRA=(--archive-rollouts) ;;
  b32_t8_c2048) CAPACITY_TBPTT=8; CAPACITY_CACHE=2048;
    CAPACITY_EXTRA=(--reuse-archive "$CAPACITY_ATTEMPT/probes/b32_t16_c4096/run/trajectories.pkl") ;;
  b32_t8_c2048_dense) CAPACITY_TBPTT=8; CAPACITY_CACHE=2048;
    CAPACITY_EXTRA=(--reuse-archive "$CAPACITY_ATTEMPT/probes/b32_t16_c4096/run/trajectories.pkl" --dense-stress) ;;
  b32_t16_c4096_dense) CAPACITY_TBPTT=16; CAPACITY_CACHE=4096;
    CAPACITY_EXTRA=(--reuse-archive "$CAPACITY_ATTEMPT/probes/b32_t16_c4096/run/trajectories.pkl" --dense-stress) ;;
  b32_t16_c4096_full) CAPACITY_TBPTT=16; CAPACITY_CACHE=4096;
    CAPACITY_EXTRA=(--reuse-archive "$CAPACITY_ATTEMPT/probes/b32_t16_c4096/run/trajectories.pkl") ;;
  b32_t8_c4096) CAPACITY_TBPTT=8; CAPACITY_CACHE=4096;
    CAPACITY_EXTRA=(--reuse-archive "$CAPACITY_ATTEMPT/probes/b32_t16_c4096/run/trajectories.pkl") ;;
  b32_t8_c4096_dense) CAPACITY_TBPTT=8; CAPACITY_CACHE=4096;
    CAPACITY_EXTRA=(--reuse-archive "$CAPACITY_ATTEMPT/probes/b32_t16_c4096/run/trajectories.pkl" --dense-stress) ;;
  *) exit 2 ;;
esac
CAPACITY_OUTPUT="$CAPACITY_ATTEMPT/probes/$CAPACITY_LABEL"
test -f "$CAPACITY_ATTEMPT/pause_receipt.json"
test ! -e "$CAPACITY_OUTPUT"
"$CAPACITY_PYTHON" -B -c 'import json,pathlib; receipt=pathlib.Path(__import__("sys").argv[1]); r=json.loads(receipt.read_text());
for item in r["workers"]+[r["controller"]]:
 stat=pathlib.Path("/proc")/str(item["pid"])/"stat"
 row=stat.read_text().rsplit(")",1)[1].split() if stat.exists() else None
 if row and row[19]==item["start_ticks"] and row[0] in ("T","t"): continue
 if (not row or row[0] in ("Z","X")) and item.get("task","").startswith("train/"):
  arm=item["task"].split("/")[1][:5]; folder=receipt.parent/"boundaries"/arm
  result=json.loads((folder/"result.json").read_text()); proof=json.loads((folder/"boundary_proof.json").read_text())
  if result["completed"] and proof["verified"] and proof["committed_groups_discarded"]==0: continue
 raise SystemExit("Study must be safely paused or have a certified completed-group capture")' "$CAPACITY_ATTEMPT/pause_receipt.json"
mkdir -p "$CAPACITY_ATTEMPT/probes"
mkdir "$CAPACITY_OUTPUT"
cp --no-clobber "$CAPACITY_WORKSPACE/onpolicy/scripts/train/probe_stage3_throughput.py" "$CAPACITY_OUTPUT/"
cp --no-clobber "$CAPACITY_WORKSPACE/onpolicy/scripts/train/launch_stage3_capacity_retest.sh" "$CAPACITY_OUTPUT/"
systemd-run --user --unit="hkbz-s3probe-uncapped-$CAPACITY_LABEL-20260908-r1" \
  --property="WorkingDirectory=$CAPACITY_SOURCE" \
  --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$CAPACITY_WORKSPACE" --setenv="PYTHONPATH=$CAPACITY_SOURCE" \
  --setenv="CUDA_VISIBLE_DEVICES=$CAPACITY_GPU" --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONHASHSEED=0 \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 --setenv=MALLOC_ARENA_MAX=2 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --property=Type=exec --property=Restart=no --property=KillMode=control-group \
  --property=TimeoutStopSec=45 --property=RuntimeMaxSec=3600 \
  --property=LimitNOFILE=65536 --property=TasksMax=128 --property=OOMPolicy=kill \
  --property="CPUAffinity=$CAPACITY_CPUS" \
  --property=MemoryHigh=20G --property=MemoryMax=24G --property=MemorySwapMax=0 \
  --property="StandardOutput=append:$CAPACITY_OUTPUT/service.log" \
  --property="StandardError=append:$CAPACITY_OUTPUT/service.log" \
  "$CAPACITY_PYTHON" -B -u "$CAPACITY_OUTPUT/probe_stage3_throughput.py" \
    --manifest "$CAPACITY_MANIFEST" --checkpoint "$CAPACITY_CHECKPOINT" --output "$CAPACITY_OUTPUT/run" \
    --gpu-uuid "$CAPACITY_GPU" --batch-trajectories 32 --tbptt-steps "$CAPACITY_TBPTT" --cache-mib "$CAPACITY_CACHE" \
    --cuda-memory-fraction 1.0 --max-seconds 3300 --host-headroom-gib 24 "${CAPACITY_EXTRA[@]}"
systemctl --user show "hkbz-s3probe-uncapped-$CAPACITY_LABEL-20260908-r1.service" \
  -p ActiveState -p SubState -p MainPID -p MemoryMax
