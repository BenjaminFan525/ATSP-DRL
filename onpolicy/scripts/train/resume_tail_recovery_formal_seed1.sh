#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd -- "$(dirname -- "${SCRIPT_PATH}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python}"
SOURCE_COMMAND_JSON="${ROOT_DIR}/result/hkbz_train_logs/stage1_suite_tail_recovery_stage1_20260730_r1/formal_R3_tail_team_ratio_seed1.command.json"
OUTPUT_COMMAND_JSON="${ROOT_DIR}/result/hkbz_train_logs/stage1_suite_tail_recovery_stage1_20260730_r1/formal_R3_tail_team_ratio_seed1.resume_run3.command.json"
EXPERIMENT="tail_recovery_stage1_20260730_r1_formal_R3_tail_team_ratio_seed1"
RESULT_DIR="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/${EXPERIMENT}"
RECOVERY_CHECKPOINT="${RESULT_DIR}/run2/models/checkpoint_Recovery.pt"
POTENTIAL_WEIGHTS="${ROOT_DIR}/result/hkbz_train_logs/stage1_suite_tail_recovery_stage1_20260730_r1/iga_trajectory_analysis.json"
SERVICE_LOG_ROOT="${ROOT_DIR}/result/hkbz_train_logs/formal_runs"
SERVICE_LOG="${SERVICE_LOG_ROOT}/${EXPERIMENT}.resume_run3.service.log"

GPU="${GPU:-1}"
CPU_SET="${CPU_SET:-36-71,108-143}"
UNIT="${UNIT:-hkbz-tail-r3-seed1-resume}"

run_worker() {
  cd "${ROOT_DIR}"
  exec "${PYTHON}" -u - \
    "${GPU}" "${SOURCE_COMMAND_JSON}" "${OUTPUT_COMMAND_JSON}" \
    "${RECOVERY_CHECKPOINT}" "${POTENTIAL_WEIGHTS}" "${ROOT_DIR}" \
    "${PYTHON}" <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

(
    gpu_text,
    source_json,
    output_json,
    recovery_checkpoint,
    potential_weights,
    root_text,
    python_text,
) = sys.argv[1:]
root = Path(root_text).resolve()
experiment = "tail_recovery_stage1_20260730_r1_formal_R3_tail_team_ratio_seed1"

payload = json.loads(Path(source_json).read_text(encoding="utf-8"))
command = [
    str(item)
    .replace("/home/fanyx/HKBZ-environment", str(root))
    .replace("/home/fanyx/anaconda3/envs/maia/bin/python", python_text)
    for item in payload["command"]
]

def set_arg(flag, value):
    index = command.index(flag)
    command[index + 1] = str(value)

set_arg("--experiment_name", experiment)
set_arg("--seed", 1)
set_arg("--checkpoint_dir", recovery_checkpoint)
set_arg("--iga_potential_weights_path", potential_weights)
while "--reset_optimizers_on_resume" in command:
    command.remove("--reset_optimizers_on_resume")
if "--resume_stage1" not in command:
    command.append("--resume_stage1")

environment = os.environ.copy()
environment.update(
    {
        "CUDA_VISIBLE_DEVICES": gpu_text,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp",
    }
)

record = {
    "command": command,
    "shell_command": shlex.join(command),
    "environment": {
        key: environment[key]
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "PYTHONHASHSEED",
            "PYTORCH_CUDA_ALLOC_CONF",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "MPLCONFIGDIR",
        )
    },
    "resume": {
        "checkpoint": str(Path(recovery_checkpoint).resolve()),
        "exact_stage1_cursor": True,
        "reset_optimizers": False,
        "expected_epoch": 8,
        "expected_completed_shards": 10,
        "expected_total_shards": 16,
    },
}
output_path = Path(output_json)
temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
temporary.write_text(
    json.dumps(record, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
os.replace(temporary, output_path)
if environment.get("HKBZ_COMMAND_ONLY") == "1":
    print(json.dumps(record, indent=2, ensure_ascii=False))
    raise SystemExit(0)
os.execvpe(command[0], command, environment)
PY
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
fi

for required in \
  "${PYTHON}" "${SOURCE_COMMAND_JSON}" "${RECOVERY_CHECKPOINT}" \
  "${POTENTIAL_WEIGHTS}" /usr/bin/taskset; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done
if [[ -e "${RESULT_DIR}/run3" ]]; then
  echo "[Error] Refusing to overwrite existing resume directory: ${RESULT_DIR}/run3" >&2
  exit 1
fi
if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "[Error] User systemd manager is unavailable." >&2
  exit 1
fi
/usr/bin/taskset --cpu-list "${CPU_SET}" /bin/true

load_state="$(
  systemctl --user show "${UNIT}.service" \
    --property=LoadState --value 2>/dev/null || true
)"
if [[ -n "${load_state}" && "${load_state}" != "not-found" ]]; then
  echo "[Error] Refusing to reuse unit ${UNIT}.service." >&2
  exit 1
fi
busy_pids="$(
  nvidia-smi --id="${GPU}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true
)"
if [[ -n "${busy_pids//[[:space:]]/}" ]]; then
  echo "[Error] GPU${GPU} is busy; compute PIDs: ${busy_pids}" >&2
  exit 1
fi

mkdir -p "${SERVICE_LOG_ROOT}"
systemd-run --user \
  --unit="${UNIT}" \
  --collect \
  --same-dir \
  --property=Type=exec \
  --property=Restart=no \
  --property=KillMode=control-group \
  --property=TimeoutStopSec=120 \
  --property=LimitNOFILE=65536 \
  --property="AllowedCPUs=${CPU_SET}" \
  --property="CPUAffinity=${CPU_SET}" \
  --property=CPUWeight=100 \
  --property=Nice=0 \
  --property="StandardOutput=append:${SERVICE_LOG}" \
  --property="StandardError=append:${SERVICE_LOG}" \
  /usr/bin/taskset --cpu-list "${CPU_SET}" \
  /bin/bash "${SCRIPT_PATH}" --worker

echo "SEED=1 GPU=${GPU} CPU_AFFINITY=${CPU_SET}"
echo "UNIT=${UNIT}.service"
echo "LOG=${SERVICE_LOG}"
systemctl --user show "${UNIT}.service" \
  --property=Id --property=MainPID --property=ActiveState --property=SubState
