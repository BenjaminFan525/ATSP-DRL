#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python}"
SOURCE_COMMAND_JSON="${ROOT_DIR}/result/hkbz_train_logs/stage1_suite_tail_recovery_stage1_20260730_r1/formal_R3_tail_team_ratio_seed1.command.json"
SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/stage1_suite_tail_recovery_stage1_20260730_r1"
RESULT_ROOT="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo"
FOUNDATION="${RESULT_ROOT}/tail_recovery_stage1_20260730_r1_screen_R0_balanced_dagger_global_seed1/run1/models/checkpoint_PlaneBC.pt"
POTENTIAL_WEIGHTS="${SUITE_DIR}/iga_trajectory_analysis.json"
SERVICE_LOG_ROOT="${ROOT_DIR}/result/hkbz_train_logs/formal_runs"

CPU_SET_GPU0="${CPU_SET_GPU0:-0-35,72-107}"
CPU_SET_GPU1="${CPU_SET_GPU1:-36-71,108-143}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-30}"

experiment_name() {
  local seed="$1"
  printf 'tail_recovery_stage1_20260730_r1_formal_R3_tail_team_ratio_seed%s' "${seed}"
}

run_worker() {
  local seed="$1"
  local gpu="$2"
  local command_json="${SUITE_DIR}/formal_R3_tail_team_ratio_seed${seed}.command.json"

  cd "${ROOT_DIR}"
  exec "${PYTHON}" -u - \
    "${seed}" "${gpu}" "${SOURCE_COMMAND_JSON}" "${command_json}" \
    "${FOUNDATION}" "${POTENTIAL_WEIGHTS}" "${ROOT_DIR}" "${PYTHON}" <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

(
    seed_text,
    gpu_text,
    source_json,
    output_json,
    foundation,
    potential_weights,
    root_text,
    python_text,
) = sys.argv[1:]
seed = int(seed_text)
gpu = int(gpu_text)
root = Path(root_text).resolve()

payload = json.loads(Path(source_json).read_text(encoding="utf-8"))
command = [
    str(item)
    .replace("/home/fanyx/HKBZ-environment", str(root))
    .replace("/home/fanyx/conda/envs/maia-hkbz-cu124-20260903-hkbz-cu124-20260903/bin/python", python_text)
    for item in payload["command"]
]

def set_arg(flag, value):
    index = command.index(flag)
    command[index + 1] = str(value)

experiment = (
    f"tail_recovery_stage1_20260730_r1_formal_"
    f"R3_tail_team_ratio_seed{seed}"
)
set_arg("--experiment_name", experiment)
set_arg("--seed", seed)
set_arg("--checkpoint_dir", foundation)
set_arg("--iga_potential_weights_path", potential_weights)
if "--reset_optimizers_on_resume" not in command:
    command.append("--reset_optimizers_on_resume")

environment = os.environ.copy()
environment.update(
    {
        "CUDA_VISIBLE_DEVICES": str(gpu),
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
    "initialization": {
        "source": str(Path(foundation).resolve()),
        "reset_optimizers": True,
        "shared_plane_bc": True,
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
  if [[ "$#" != 3 ]]; then
    echo "[Error] Worker usage: $0 --worker SEED GPU" >&2
    exit 1
  fi
  run_worker "$2" "$3"
fi

if ! [[ "${START_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "[Error] START_DELAY_SECONDS must be a non-negative integer." >&2
  exit 1
fi
for required in \
  "${PYTHON}" "${SOURCE_COMMAND_JSON}" "${FOUNDATION}" \
  "${POTENTIAL_WEIGHTS}" /usr/bin/taskset; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done
if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "[Error] User systemd manager is unavailable." >&2
  exit 1
fi

for mapping in "2:0:${CPU_SET_GPU0}" "3:1:${CPU_SET_GPU1}"; do
  seed="${mapping%%:*}"
  remainder="${mapping#*:}"
  gpu="${remainder%%:*}"
  cpu_set="${remainder#*:}"
  experiment="$(experiment_name "${seed}")"
  result_dir="${RESULT_ROOT}/${experiment}"
  unit="hkbz-tail-r3-seed${seed}"

  /usr/bin/taskset --cpu-list "${cpu_set}" /bin/true
  if [[ -e "${result_dir}" ]]; then
    echo "[Error] Refusing to overwrite existing result directory: ${result_dir}" >&2
    exit 1
  fi
  load_state="$(
    systemctl --user show "${unit}.service" \
      --property=LoadState --value 2>/dev/null || true
  )"
  if [[ -n "${load_state}" && "${load_state}" != "not-found" ]]; then
    echo "[Error] Refusing to reuse unit ${unit}.service." >&2
    exit 1
  fi
  busy_pids="$(
    nvidia-smi --id="${gpu}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true
  )"
  if [[ -n "${busy_pids//[[:space:]]/}" ]]; then
    echo "[Error] GPU${gpu} is busy; compute PIDs: ${busy_pids}" >&2
    exit 1
  fi
done

mkdir -p "${SERVICE_LOG_ROOT}"

launch_seed() {
  local seed="$1"
  local gpu="$2"
  local cpu_set="$3"
  local unit="hkbz-tail-r3-seed${seed}"
  local log="${SERVICE_LOG_ROOT}/tail_recovery_stage1_20260730_r1_formal_R3_tail_team_ratio_seed${seed}.service.log"

  systemd-run --user \
    --unit="${unit}" \
    --collect \
    --same-dir \
    --property=Type=exec \
    --property=Restart=no \
    --property=KillMode=control-group \
    --property=TimeoutStopSec=120 \
    --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${cpu_set}" \
    --property="CPUAffinity=${cpu_set}" \
    --property=CPUWeight=100 \
    --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /usr/bin/taskset --cpu-list "${cpu_set}" \
    /bin/bash "$0" --worker "${seed}" "${gpu}"

  echo "SEED=${seed} GPU=${gpu} CPU_AFFINITY=${cpu_set}"
  echo "UNIT=${unit}.service"
  echo "LOG=${log}"
}

launch_seed 2 0 "${CPU_SET_GPU0}"
echo "[Launch] Waiting ${START_DELAY_SECONDS}s before seed3 initialization."
sleep "${START_DELAY_SECONDS}"
launch_seed 3 1 "${CPU_SET_GPU1}"

systemctl --user show hkbz-tail-r3-seed2.service hkbz-tail-r3-seed3.service \
  --property=Id --property=MainPID --property=ActiveState --property=SubState
