#!/usr/bin/env bash
set -euo pipefail

# One-command Stage2 Wave-3 controller.  It first runs a true five-trainer,
# full-validator 1000-graph canary, then launches the five causal arms on GPU0.
# All services share the audited 0-143 CPU pool; GPU1 is intentionally unused.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python3.11}"
PREPARE="${ROOT_DIR}/onpolicy/scripts/train/prepare_stage2_resource_research.py"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage2_resource_manifest_trial.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"

RUN_TAG="${RUN_TAG:-stage2_critical_wave3_20260821_r1}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s2-critical-w3-r1}"
CPU_POOL="${CPU_POOL:-0-143}"
VALIDATION_DIR="${VALIDATION_DIR:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint/tune}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-77824}"
HOST_MEMORY_MIN_AVAILABLE_MIB="${HOST_MEMORY_MIN_AVAILABLE_MIB:-65536}"
SWAP_GROWTH_LIMIT_MIB="${SWAP_GROWTH_LIMIT_MIB:-8192}"
CANARY_TIMEOUT_SECONDS="${CANARY_TIMEOUT_SECONDS:-21600}"
BC_WAIT_TIMEOUT_SECONDS="${BC_WAIT_TIMEOUT_SECONDS:-86400}"
SKIP_COMPLETED_CANARY="${SKIP_COMPLETED_CANARY:-0}"
METHODS=(
  C0_cmax_control
  C1_team_time
  C2_critical_slack
  C3_slack_arrival
  C4_slack_arrival_tail
)

for required in "${PYTHON}" "${PREPARE}" "${TRIAL_RUNNER}" \
  "${EVALUATOR}" "${VALIDATION_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing Stage2 Wave-3 input: ${required}" >&2
    exit 1
  fi
done
if [[ "$(nproc)" -ne 144 ]]; then
  echo "[Error] Expected 144 logical CPUs, got $(nproc)." >&2
  exit 1
fi

mkdir -p "${SUITE_DIR}/commands" "${SUITE_DIR}/service_logs" \
  "${SUITE_DIR}/records" "${SUITE_DIR}/shared_evaluator" \
  "${SUITE_DIR}/hardware"

profile_slug() {
  case "$1" in
    critical_canary) echo cc ;;
    critical_wave) echo cw ;;
    *) return 1 ;;
  esac
}

experiment_dir() {
  local profile="$1"
  local method="$2"
  echo "${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/${RUN_TAG}_${profile}_${method}_seed1"
}

device_bc_path() {
  local profile="$1"
  echo "$(experiment_dir "${profile}" C0_cmax_control)/run1/models/checkpoint_DeviceBC.pt"
}

unit_name() {
  local profile="$1"
  local suffix="$2"
  echo "${UNIT_PREFIX}-$(profile_slug "${profile}")-${suffix}"
}

trainer_lanes() {
  seq 0 $((${#METHODS[@]} - 1))
}

require_fresh_profile() {
  local profile="$1"
  local method path unit state lane
  for method in "${METHODS[@]}"; do
    path="$(experiment_dir "${profile}" "${method}")"
    if [[ -e "${path}" ]]; then
      echo "[Error] Refusing ambiguous runN reuse: ${path}" >&2
      exit 1
    fi
  done
  unit="$(unit_name "${profile}" eval).service"
  state="$(systemctl --user show "${unit}" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${state}" && "${state}" != not-found ]]; then
    echo "[Error] Existing unit ${unit}; choose a new UNIT_PREFIX." >&2
    exit 1
  fi
  for lane in $(trainer_lanes); do
    unit="$(unit_name "${profile}" "c${lane}").service"
    state="$(systemctl --user show "${unit}" --property=LoadState --value 2>/dev/null || true)"
    if [[ -n "${state}" && "${state}" != not-found ]]; then
      echo "[Error] Existing unit ${unit}; choose a new UNIT_PREFIX." >&2
      exit 1
    fi
  done
}

prepare_manifest() {
  local profile="$1"
  local manifest="${SUITE_DIR}/commands/${profile}.json"
  "${PYTHON}" "${PREPARE}" \
    --run-tag "${RUN_TAG}" --profile "${profile}" \
    --suite-dir "${SUITE_DIR}" --output "${manifest}"
  echo "${manifest}"
}

launch_evaluator() {
  local profile="$1"
  local manifest="$2"
  local workers="$3"
  local unit socket log
  unit="$(unit_name "${profile}" eval)"
  socket="/tmp/${unit}.sock"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  if [[ -e "${socket}" ]]; then
    echo "[Error] Evaluator socket already exists: ${socket}" >&2
    exit 1
  fi
  systemd-run --user --unit="${unit}" --same-dir \
    --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
    --property=Type=exec --property=Restart=no \
    --property=KillMode=control-group --property=TimeoutStopSec=180 \
    --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${CPU_POOL}" \
    --property="CPUAffinity=${CPU_POOL}" \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    "${PYTHON}" -u "${EVALUATOR}" \
      --source-command-json "${manifest}" \
      --socket-path "${socket}" --cpu-pool "${CPU_POOL}" \
      --run-dir "${SUITE_DIR}/shared_evaluator/${profile}" \
      --eval-dataset-dir "${VALIDATION_DIR}" \
      --max-eval-cases "${workers}" \
      --eval-partition-seed 20260811 \
      --eval-partition-stratify-by distribution >/dev/null
  echo "${socket}"
}

wait_evaluator() {
  local profile="$1"
  local socket="$2"
  local workers="$3"
  local unit deadline
  unit="$(unit_name "${profile}" eval)"
  deadline=$((SECONDS + 1200))
  while (( SECONDS < deadline )); do
    if [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" != active ]]; then
      echo "[Error] Evaluator exited before readiness: ${unit}.service" >&2
      return 1
    fi
    if [[ -S "${socket}" ]] && "${PYTHON}" - "${socket}" "${workers}" <<'PY'
import sys
from onpolicy.utils.shared_eval import SharedEvalClient
reply = SharedEvalClient(sys.argv[1], timeout_seconds=5.0).ping()
raise SystemExit(0 if int(reply.get('worker_count', -1)) == int(sys.argv[2]) else 1)
PY
    then
      echo "[Ready] ${unit}.service workers=${workers} CPUs=${CPU_POOL}"
      return 0
    fi
    sleep 2
  done
  echo "[Error] Evaluator readiness timed out: ${unit}.service" >&2
  return 1
}

launch_trainer() {
  local profile="$1"
  local manifest="$2"
  local socket="$3"
  local workers="$4"
  local lane="$5"
  local shared_bc="${6:-}"
  local method unit log
  method="${METHODS[${lane}]}"
  unit="$(unit_name "${profile}" "c${lane}")"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  local args=(
    "${PYTHON}" -u "${TRIAL_RUNNER}"
    --manifest "${manifest}" --command-key "${method}" --gpu 0
    --cpu-set "${CPU_POOL}" --shared-eval-socket "${socket}"
    --eval-workers "${workers}"
    --record "${SUITE_DIR}/records/${profile}_c${lane}.json"
  )
  if [[ -n "${shared_bc}" ]]; then
    args+=(
      --resource-bc-checkpoint "${shared_bc}"
      --resource-bc-wait-timeout-seconds "${BC_WAIT_TIMEOUT_SECONDS}"
    )
  fi
  systemd-run --user --unit="${unit}" --same-dir \
    --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
    --property=Type=exec --property=Restart=no \
    --property=KillMode=control-group --property=TimeoutStopSec=180 \
    --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${CPU_POOL}" \
    --property="CPUAffinity=${CPU_POOL}" \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    "${args[@]}" >/dev/null
  echo "[Launch] ${unit}.service ${method} GPU0 CPUs=${CPU_POOL} shared_bc=${shared_bc:-producer}"
}

stop_profile() {
  local profile="$1"
  local lane
  for lane in $(trainer_lanes); do
    systemctl --user stop "$(unit_name "${profile}" "c${lane}").service" >/dev/null 2>&1 || true
  done
  systemctl --user stop "$(unit_name "${profile}" eval).service" >/dev/null 2>&1 || true
}

swap_used_mib() {
  awk '/SwapTotal:/ {total=$2} /SwapFree:/ {free=$2} END {printf "%d", (total-free)/1024}' /proc/meminfo
}

mem_available_mib() {
  awk '/MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo
}

wait_gpu_idle() {
  local deadline busy
  deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    busy="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -z "${busy//[[:space:]]/}" ]]; then
      return 0
    fi
    sleep 2
  done
  echo "[Error] GPU0 processes did not release after canary." >&2
  return 1
}

wait_canary() {
  local profile=critical_canary
  local memory_log="${SUITE_DIR}/hardware/critical_canary_gpu_memory.csv"
  local deadline peak used util active lane unit status
  local available swap_now swap_start swap_growth min_available
  deadline=$((SECONDS + CANARY_TIMEOUT_SECONDS))
  peak=0
  min_available=999999999
  swap_start="$(swap_used_mib)"
  echo "unix_time,gpu_memory_used_mib,gpu_utilization_percent,mem_available_mib,swap_used_mib,active_trainers" >"${memory_log}"
  while (( SECONDS < deadline )); do
    used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    util="$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    available="$(mem_available_mib)"
    swap_now="$(swap_used_mib)"
    active=0
    for lane in $(trainer_lanes); do
      unit="$(unit_name "${profile}" "c${lane}")"
      if [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" == active ]]; then
        active=$((active + 1))
      fi
    done
    echo "$(date +%s),${used},${util},${available},${swap_now},${active}" >>"${memory_log}"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used > peak )); then peak="${used}"; fi
    if (( available < min_available )); then min_available="${available}"; fi
    if (( active == 0 )); then break; fi
    sleep 5
  done
  if (( SECONDS >= deadline )); then
    echo "[Error] Five-lane canary timed out." >&2
    return 1
  fi
  for lane in $(trainer_lanes); do
    unit="$(unit_name "${profile}" "c${lane}")"
    status="$(systemctl --user show "${unit}.service" --property=ExecMainStatus --value 2>/dev/null || echo 255)"
    if [[ "${status}" != 0 ]] || grep -Eqi \
      'out of memory|OutOfMemoryError|CUDA error|Traceback|No trainable PPO|no gradients|gradient.*missing' \
      "${SUITE_DIR}/service_logs/${unit}.log"; then
      echo "[Error] Canary trainer failed: ${unit}.service status=${status}" >&2
      return 1
    fi
    if ! grep -q '\[Memory\].*graphs_per_forward=1000' \
      "${SUITE_DIR}/service_logs/${unit}.log"; then
      echo "[Error] Canary did not exercise a 1000-graph update: ${unit}" >&2
      return 1
    fi
  done
  swap_growth=$((swap_now - swap_start))
  if (( peak > GPU_MEMORY_LIMIT_MIB )); then
    echo "[Error] Canary GPU peak ${peak} MiB exceeds ${GPU_MEMORY_LIMIT_MIB} MiB." >&2
    return 1
  fi
  if (( min_available < HOST_MEMORY_MIN_AVAILABLE_MIB )); then
    echo "[Error] Canary MemAvailable floor ${min_available} MiB is unsafe." >&2
    return 1
  fi
  if (( swap_growth > SWAP_GROWTH_LIMIT_MIB )); then
    echo "[Error] Canary swap grew ${swap_growth} MiB; refusing formal launch." >&2
    return 1
  fi
  echo "[Canary] passed gpu_peak_mib=${peak} mem_available_floor_mib=${min_available} swap_growth_mib=${swap_growth}"
}

validate_completed_canary() {
  local method status_path memory_log peak
  memory_log="${SUITE_DIR}/hardware/critical_canary_gpu_memory.csv"
  if [[ ! -s "${memory_log}" ]]; then
    echo "[Error] Missing completed canary memory evidence: ${memory_log}" >&2
    return 1
  fi
  peak="$(awk -F, 'NR>1 && $2+0>max {max=$2+0} END {print max+0}' "${memory_log}")"
  if (( peak <= 0 || peak > GPU_MEMORY_LIMIT_MIB )); then
    echo "[Error] Reused canary GPU peak is invalid: ${peak} MiB." >&2
    return 1
  fi
  for method in "${METHODS[@]}"; do
    status_path="$(experiment_dir critical_canary "${method}")/run1/run_status.json"
    "${PYTHON}" - "${status_path}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
ok = (
    payload.get('status') == 'completed'
    and payload.get('phase') == 'resource_joint_completed'
    and int(payload.get('completed_shard', 0))
        == int(payload.get('total_shards', -1))
    and int(payload.get('total_shards', 0)) > 0
)
raise SystemExit(0 if ok else 1)
PY
  done
  if [[ ! -f "$(device_bc_path critical_canary)" ]]; then
    echo "[Error] Reused canary lacks its shared DeviceBC checkpoint." >&2
    return 1
  fi
  echo "[Canary] reused completed five-lane evidence peak_gpu_memory_mib=${peak}"
}

run_controller() {
  local busy canary_manifest canary_socket canary_bc
  local formal_manifest formal_socket formal_bc lane
  busy="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "${busy//[[:space:]]/}" ]]; then
    echo "[Error] GPU0 is busy before launch; PIDs: ${busy}" >&2
    exit 1
  fi

  nvidia-smi -L >"${SUITE_DIR}/hardware/gpus.txt"
  lscpu >"${SUITE_DIR}/hardware/lscpu.txt"
  free -h >"${SUITE_DIR}/hardware/memory_before.txt"

  if [[ "${SKIP_COMPLETED_CANARY}" == 1 ]]; then
    validate_completed_canary
  else
    require_fresh_profile critical_canary
    canary_manifest="$(prepare_manifest critical_canary | tail -n 1)"
    canary_socket="$(launch_evaluator critical_canary "${canary_manifest}" 60 | tail -n 1)"
    wait_evaluator critical_canary "${canary_socket}" 60
    canary_bc="$(device_bc_path critical_canary)"
    launch_trainer critical_canary "${canary_manifest}" "${canary_socket}" 60 0
    for lane in 1 2 3 4; do
      launch_trainer critical_canary "${canary_manifest}" "${canary_socket}" 60 "${lane}" "${canary_bc}"
    done
    if ! wait_canary; then
      stop_profile critical_canary
      exit 1
    fi
    stop_profile critical_canary
    wait_gpu_idle
  fi

  require_fresh_profile critical_wave
  formal_manifest="$(prepare_manifest critical_wave | tail -n 1)"
  formal_socket="$(launch_evaluator critical_wave "${formal_manifest}" 60 | tail -n 1)"
  wait_evaluator critical_wave "${formal_socket}" 60
  formal_bc="$(device_bc_path critical_wave)"
  launch_trainer critical_wave "${formal_manifest}" "${formal_socket}" 60 0
  for lane in 1 2 3 4; do
    launch_trainer critical_wave "${formal_manifest}" "${formal_socket}" 60 "${lane}" "${formal_bc}"
  done

  echo "[Done] Formal Wave-3 launched: ${METHODS[*]}"
  echo "[Done] GPU policy: GPU0 only; GPU1 unused."
  echo "[Done] CPU policy: every service may use ${CPU_POOL}; 36 rollout workers per trainer."
  echo "[Done] Shared validator: $(unit_name critical_wave eval).service (60 workers)."
  echo "[Done] Shared DeviceBC producer: $(unit_name critical_wave c0).service"
  echo "[Done] Manifest: ${formal_manifest}"
}

if [[ "${HKBZ_CONTROLLER_CHILD:-0}" != 1 ]]; then
  controller="${UNIT_PREFIX}-controller"
  controller_log="${SUITE_DIR}/service_logs/${controller}.log"
  state="$(systemctl --user show "${controller}.service" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${state}" && "${state}" != not-found ]]; then
    echo "[Error] Existing controller unit ${controller}.service." >&2
    exit 1
  fi
  systemd-run --user --unit="${controller}" --same-dir \
    --setenv=HKBZ_CONTROLLER_CHILD=1 \
    --setenv="RUN_TAG=${RUN_TAG}" --setenv="SUITE_DIR=${SUITE_DIR}" \
    --setenv="UNIT_PREFIX=${UNIT_PREFIX}" --setenv="CPU_POOL=${CPU_POOL}" \
    --setenv="VALIDATION_DIR=${VALIDATION_DIR}" --setenv="PYTHON=${PYTHON}" \
    --setenv="GPU_MEMORY_LIMIT_MIB=${GPU_MEMORY_LIMIT_MIB}" \
    --setenv="HOST_MEMORY_MIN_AVAILABLE_MIB=${HOST_MEMORY_MIN_AVAILABLE_MIB}" \
    --setenv="SWAP_GROWTH_LIMIT_MIB=${SWAP_GROWTH_LIMIT_MIB}" \
    --setenv="SKIP_COMPLETED_CANARY=${SKIP_COMPLETED_CANARY}" \
    --property=Type=exec --property=Restart=no \
    --property=KillMode=control-group --property=TimeoutStopSec=180 \
    --property="AllowedCPUs=${CPU_POOL}" --property="CPUAffinity=${CPU_POOL}" \
    --property="StandardOutput=append:${controller_log}" \
    --property="StandardError=append:${controller_log}" \
    /usr/bin/bash "${BASH_SOURCE[0]}" >/dev/null
  echo "[Launch] controller=${controller}.service log=${controller_log}"
  exit 0
fi

run_controller
