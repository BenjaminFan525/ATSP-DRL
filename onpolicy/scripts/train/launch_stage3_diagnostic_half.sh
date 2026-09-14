#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/fanyx/HKBZ-environment
PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
SUITE=${1:?Usage: bash launch_stage3_diagnostic_half.sh ABSOLUTE_SUITE_DIRECTORY [RESUME_MANIFEST]}
MANIFEST=${2:-$SUITE/manifest.json}
test -f "$MANIFEST"
mapfile -t EXECUTION < <("$PYTHON" -c '
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2]).resolve()
assert pathlib.Path(m["root"]).resolve() == root
if "resume" in m:
    assert json.loads((root / "status.json").read_text())["status"] == "failed"
    assert pathlib.Path(m["execution"]["code_root"]).is_relative_to(root / "attempts")
else:
    assert not (root / "status.json").exists()
print(m.get("execution", {}).get("code_root", sys.argv[3]))
print(m.get("resume", {}).get("attempt_dir", str(root)))
print(m.get("resume", {}).get("id", "initial"))
' "$MANIFEST" "$SUITE" "$ROOT")
test "${#EXECUTION[@]}" -eq 3
CODE_ROOT=${EXECUTION[0]}
ATTEMPT=${EXECUTION[1]}
ATTEMPT_ID=${EXECUTION[2]}
test -f "$CODE_ROOT/onpolicy/scripts/train/run_stage3_diagnostic_suite.py"
# Refuse to collide with unrelated compute work on the selected half.
for gpu in 0 1 2 3; do
  occupied=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -n "$occupied" ]]; then
    echo "GPU $gpu already has compute processes: $occupied" >&2
    exit 1
  fi
done
mkdir -p "$ATTEMPT/logs"
UNIT="hkbz-s3diag-$(basename "$SUITE")"
if [[ "$ATTEMPT_ID" != initial ]]; then
  UNIT="$UNIT-$ATTEMPT_ID"
fi
# Aggregate cgroup covers controller, IGA workers, three trainers AND validator.
systemd-run --user --unit="$UNIT" \
  --property="WorkingDirectory=$CODE_ROOT" \
  --setenv="HKBZ_STAGE3_WORKSPACE_ROOT=$ROOT" --setenv="PYTHONPATH=$CODE_ROOT" \
  --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2,3 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=PYTHONHASHSEED=0 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
  --setenv=MALLOC_ARENA_MAX=2 --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --property=Type=exec --property=Restart=no --property=KillMode=control-group \
  --property=TimeoutStopSec=30 --property=RuntimeMaxSec=1209600 \
  --property=LimitNOFILE=65536 --property=TasksMax=512 \
  --property=AllowedCPUs=0-31,64-95 --property=CPUAffinity=0-31,64-95 \
  --property=NUMAPolicy=bind --property=NUMAMask=0 \
  --property=MemoryHigh=100G --property=MemoryMax=120G \
  --property="StandardOutput=append:$ATTEMPT/logs/controller.log" \
  --property="StandardError=append:$ATTEMPT/logs/controller.log" \
  /usr/bin/taskset --cpu-list 0-31,64-95 "$PYTHON" -u \
  "$CODE_ROOT/onpolicy/scripts/train/run_stage3_diagnostic_suite.py" "$MANIFEST"
systemctl --user show "$UNIT.service" -p Id -p ActiveState -p SubState -p MainPID -p NRestarts -p AllowedCPUs -p CPUAffinity
