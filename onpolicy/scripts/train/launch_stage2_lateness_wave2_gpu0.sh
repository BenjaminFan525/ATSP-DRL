#!/usr/bin/env bash
set -euo pipefail

# One-command Stage2 lateness round: a short three-process canary, one shared
# DeviceBC/DAgger warm-up, then P0/P1/P2 PPO on GPU0.  Every service may use
# the complete audited CPU pool; one persistent evaluator serializes all val.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python3.11}"
PREPARE="${ROOT_DIR}/onpolicy/scripts/train/prepare_stage2_resource_research.py"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage2_resource_manifest_trial.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"

RUN_TAG="${RUN_TAG:-stage2_lateness_wave2_20260820_r1}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s2-late-w2-r1}"
CPU_POOL="${CPU_POOL:-0-143}"
VALIDATION_DIR="${VALIDATION_DIR:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint/tune}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-73728}"
CANARY_TIMEOUT_SECONDS="${CANARY_TIMEOUT_SECONDS:-14400}"
BC_WAIT_TIMEOUT_SECONDS="${BC_WAIT_TIMEOUT_SECONDS:-86400}"
METHODS=(P0_cmax P1_total_lateness P2_critical_jit)

for required in "${PYTHON}" "${PREPARE}" "${TRIAL_RUNNER}" "${EVALUATOR}" "${VALIDATION_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing Stage2 lateness input: ${required}" >&2
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
    lateness_canary) echo lc ;;
    lateness_wave) echo lw ;;
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
  echo "$(experiment_dir "${profile}" P0_cmax)/run1/models/checkpoint_DeviceBC.pt"
}

unit_name() {
  local profile="$1"
  local suffix="$2"
  echo "${UNIT_PREFIX}-$(profile_slug "${profile}")-${suffix}"
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
  if [[ -n "${state}" && "${state}" != "not-found" ]]; then
    echo "[Error] Existing unit ${unit}; choose a new UNIT_PREFIX." >&2
    exit 1
  fi
  for lane in 0 1 2; do
    unit="$(unit_name "${profile}" "p${lane}").service"
    state="$(systemctl --user show "${unit}" --property=LoadState --value 2>/dev/null || true)"
    if [[ -n "${state}" && "${state}" != "not-found" ]]; then
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
  deadline=$((SECONDS + 900))
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
  unit="$(unit_name "${profile}" "p${lane}")"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  local args=(
    "${PYTHON}" -u "${TRIAL_RUNNER}"
    --manifest "${manifest}" --command-key "${method}" --gpu 0
    --cpu-set "${CPU_POOL}" --shared-eval-socket "${socket}"
    --eval-workers "${workers}"
    --record "${SUITE_DIR}/records/${profile}_p${lane}.json"
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
  for lane in 0 1 2; do
    systemctl --user stop "$(unit_name "${profile}" "p${lane}").service" >/dev/null 2>&1 || true
  done
  systemctl --user stop "$(unit_name "${profile}" eval).service" >/dev/null 2>&1 || true
}

wait_canary() {
  local profile=lateness_canary
  local memory_log="${SUITE_DIR}/hardware/lateness_canary_gpu_memory.csv"
  local deadline peak used active lane unit status
  deadline=$((SECONDS + CANARY_TIMEOUT_SECONDS))
  peak=0
  echo "unix_time,memory_used_mib" >"${memory_log}"
  while (( SECONDS < deadline )); do
    used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "${used}" =~ ^[0-9]+$ ]]; then
      echo "$(date +%s),${used}" >>"${memory_log}"
      if (( used > peak )); then peak="${used}"; fi
    fi
    active=0
    for lane in 0 1 2; do
      unit="$(unit_name "${profile}" "p${lane}")"
      if [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" == active ]]; then
        active=$((active + 1))
      fi
    done
    if (( active == 0 )); then break; fi
    sleep 5
  done
  if (( SECONDS >= deadline )); then
    echo "[Error] Lateness canary timed out." >&2
    return 1
  fi
  for lane in 0 1 2; do
    unit="$(unit_name "${profile}" "p${lane}")"
    status="$(systemctl --user show "${unit}.service" --property=ExecMainStatus --value 2>/dev/null || echo 255)"
    if [[ "${status}" != 0 ]] || grep -Eqi 'out of memory|CUDA error|Traceback' "${SUITE_DIR}/service_logs/${unit}.log"; then
      echo "[Error] Canary trainer failed: ${unit}.service status=${status}" >&2
      return 1
    fi
  done
  if (( peak > GPU_MEMORY_LIMIT_MIB )); then
    echo "[Error] Canary peak ${peak} MiB exceeds ${GPU_MEMORY_LIMIT_MIB} MiB." >&2
    return 1
  fi
  echo "[Canary] passed peak_gpu_memory_mib=${peak} shared_bc=$(device_bc_path "${profile}")"
}

run_controller() {
  local busy canary_manifest canary_socket canary_bc
  local formal_manifest formal_socket formal_bc
  busy="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "${busy//[[:space:]]/}" ]]; then
    echo "[Error] GPU0 is busy before launch; PIDs: ${busy}" >&2
    exit 1
  fi

  require_fresh_profile lateness_canary
  canary_manifest="$(prepare_manifest lateness_canary | tail -n 1)"
  canary_socket="$(launch_evaluator lateness_canary "${canary_manifest}" 12 | tail -n 1)"
  wait_evaluator lateness_canary "${canary_socket}" 12
  canary_bc="$(device_bc_path lateness_canary)"
  launch_trainer lateness_canary "${canary_manifest}" "${canary_socket}" 12 0
  launch_trainer lateness_canary "${canary_manifest}" "${canary_socket}" 12 1 "${canary_bc}"
  launch_trainer lateness_canary "${canary_manifest}" "${canary_socket}" 12 2 "${canary_bc}"
  if ! wait_canary; then
    stop_profile lateness_canary
    exit 1
  fi
  systemctl --user stop "$(unit_name lateness_canary eval).service" >/dev/null

  require_fresh_profile lateness_wave
  formal_manifest="$(prepare_manifest lateness_wave | tail -n 1)"
  formal_socket="$(launch_evaluator lateness_wave "${formal_manifest}" 60 | tail -n 1)"
  wait_evaluator lateness_wave "${formal_socket}" 60
  formal_bc="$(device_bc_path lateness_wave)"
  launch_trainer lateness_wave "${formal_manifest}" "${formal_socket}" 60 0
  launch_trainer lateness_wave "${formal_manifest}" "${formal_socket}" 60 1 "${formal_bc}"
  launch_trainer lateness_wave "${formal_manifest}" "${formal_socket}" 60 2 "${formal_bc}"

  echo "[Done] Formal lateness round launched: ${METHODS[*]}"
  echo "[Done] GPU policy: GPU0 only; CPU policy: every service may use ${CPU_POOL}."
  echo "[Done] Shared validator: $(unit_name lateness_wave eval).service"
  echo "[Done] Shared DeviceBC producer: $(unit_name lateness_wave p0).service"
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
    --setenv="VALIDATION_DIR=${VALIDATION_DIR}" \
    --setenv="PYTHON=${PYTHON}" \
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
