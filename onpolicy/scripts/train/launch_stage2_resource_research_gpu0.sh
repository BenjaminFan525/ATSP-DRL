#!/usr/bin/env bash
set -euo pipefail

# GPU0-only Stage2 research launcher.  All trainers intentionally share the
# complete CPU pool: this round optimizes method throughput, not seed-to-seed
# timing isolation.  BLAS/OpenMP remain single-threaded inside every worker.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python3.11}"
PREPARE="${ROOT_DIR}/onpolicy/scripts/train/prepare_stage2_resource_research.py"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage2_resource_manifest_trial.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"

RUN_TAG="${RUN_TAG:-stage2_resource_wave1_20260819_r9}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s2-w1-20260819-r9}"
CPU_POOL="${CPU_POOL:-0-143}"
VALIDATION_DIR="${VALIDATION_DIR:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint/tune}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-73728}"
CANARY_TIMEOUT_SECONDS="${CANARY_TIMEOUT_SECONDS:-14400}"
CANARY_ONLY="${CANARY_ONLY:-false}"
REUSE_COMPLETED_CANARY="${REUSE_COMPLETED_CANARY:-false}"

METHODS=(M0_heuristic_uniform M1_iga_uniform M2_iga_role_balanced)

for required in "${PYTHON}" "${PREPARE}" "${TRIAL_RUNNER}" "${EVALUATOR}" "${VALIDATION_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing Stage2 input: ${required}" >&2
    exit 1
  fi
done
if [[ "$(nproc)" -ne 144 ]]; then
  echo "[Error] Expected the audited 144-logical-CPU host, got $(nproc)." >&2
  exit 1
fi

mkdir -p "${SUITE_DIR}/commands" "${SUITE_DIR}/service_logs" \
  "${SUITE_DIR}/records" "${SUITE_DIR}/shared_evaluator" \
  "${SUITE_DIR}/hardware"

ensure_units_absent() {
  local profile="$1"
  local attempt="$2"
  local names=("${UNIT_PREFIX}-${profile}-${attempt}-eval.service")
  for lane in 0 1 2; do
    names+=("${UNIT_PREFIX}-${profile}-${attempt}-m${lane}.service")
  done
  local unit state
  for unit in "${names[@]}"; do
    state="$(systemctl --user show "${unit}" --property=LoadState --value 2>/dev/null || true)"
    if [[ -n "${state}" && "${state}" != "not-found" ]]; then
      echo "[Error] Refusing to reuse existing unit ${unit}." >&2
      exit 1
    fi
  done
}

prepare_manifest() {
  local profile="$1"
  local attempt="$2"
  local low_memory="$3"
  local manifest="${SUITE_DIR}/commands/${profile}_${attempt}.json"
  local args=(
    "${PYTHON}" "${PREPARE}"
    --run-tag "${RUN_TAG}" --profile "${profile}"
    --suite-dir "${SUITE_DIR}" --output "${manifest}"
  )
  if [[ "${low_memory}" == "true" ]]; then
    args+=(--low-memory)
  fi
  "${args[@]}"
  echo "${manifest}"
}

launch_evaluator() {
  local profile="$1"
  local attempt="$2"
  local manifest="$3"
  local eval_cases="$4"
  local unit="${UNIT_PREFIX}-${profile}-${attempt}-eval"
  local socket_path="/tmp/${unit}.sock"
  local log="${SUITE_DIR}/service_logs/${unit}.log"
  if [[ -e "${socket_path}" ]]; then
    echo "[Error] Refusing to replace existing evaluator socket ${socket_path}." >&2
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
      --socket-path "${socket_path}" --cpu-pool "${CPU_POOL}" \
      --run-dir "${SUITE_DIR}/shared_evaluator/${profile}_${attempt}" \
      --eval-dataset-dir "${VALIDATION_DIR}" \
      --max-eval-cases "${eval_cases}" \
      --eval-partition-seed 20260811 \
      --eval-partition-stratify-by distribution >/dev/null
  echo "${socket_path}"
}

wait_evaluator() {
  local profile="$1"
  local attempt="$2"
  local socket_path="$3"
  local expected_workers="$4"
  local unit="${UNIT_PREFIX}-${profile}-${attempt}-eval"
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" != "active" ]]; then
      echo "[Error] Stage2 evaluator exited before readiness: ${unit}.service" >&2
      return 1
    fi
    if [[ -S "${socket_path}" ]] && "${PYTHON}" - "${socket_path}" "${expected_workers}" <<'PY'
import sys
from onpolicy.utils.shared_eval import SharedEvalClient
reply = SharedEvalClient(sys.argv[1], timeout_seconds=5.0).ping()
if int(reply.get('worker_count', -1)) != int(sys.argv[2]):
    raise SystemExit(1)
PY
    then
      echo "[Ready] ${unit}.service workers=${expected_workers} CPUs=${CPU_POOL}"
      return 0
    fi
    sleep 2
  done
  echo "[Error] Timed out waiting for ${unit}.service." >&2
  return 1
}

launch_trainers() {
  local profile="$1"
  local attempt="$2"
  local manifest="$3"
  local socket_path="$4"
  local eval_workers="$5"
  local lane method unit log delay
  for lane in 0 1 2; do
    method="${METHODS[${lane}]}"
    unit="${UNIT_PREFIX}-${profile}-${attempt}-m${lane}"
    log="${SUITE_DIR}/service_logs/${unit}.log"
    delay=$((lane * 15))
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
      "${PYTHON}" -u "${TRIAL_RUNNER}" \
        --manifest "${manifest}" --command-key "${method}" --gpu 0 \
        --cpu-set "${CPU_POOL}" --shared-eval-socket "${socket_path}" \
        --eval-workers "${eval_workers}" \
        --start-delay-seconds "${delay}" \
        --record "${SUITE_DIR}/records/${profile}_${attempt}_m${lane}.json" >/dev/null
    echo "[Launch] ${unit}.service ${method} full_cpu_pool=${CPU_POOL} delay=${delay}s"
  done
}

stop_evaluator() {
  local profile="$1"
  local attempt="$2"
  local unit="${UNIT_PREFIX}-${profile}-${attempt}-eval.service"
  systemctl --user stop "${unit}" >/dev/null 2>&1 || true
}

wait_canary() {
  local attempt="$1"
  local memory_log="${SUITE_DIR}/hardware/canary_${attempt}_gpu_memory.csv"
  local deadline=$((SECONDS + CANARY_TIMEOUT_SECONDS))
  local peak=0 used active_count lane unit status
  echo "unix_time,memory_used_mib" >"${memory_log}"
  while (( SECONDS < deadline )); do
    used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "${used}" =~ ^[0-9]+$ ]]; then
      echo "$(date +%s),${used}" >>"${memory_log}"
      if (( used > peak )); then peak="${used}"; fi
    fi
    active_count=0
    for lane in 0 1 2; do
      unit="${UNIT_PREFIX}-canary-${attempt}-m${lane}.service"
      if [[ "$(systemctl --user is-active "${unit}" 2>/dev/null || true)" == "active" ]]; then
        active_count=$((active_count + 1))
      fi
    done
    if (( active_count == 0 )); then
      break
    fi
    sleep 5
  done
  if (( SECONDS >= deadline )); then
    echo "[Error] Stage2 three-process canary timed out." >&2
    return 2
  fi
  for lane in 0 1 2; do
    unit="${UNIT_PREFIX}-canary-${attempt}-m${lane}.service"
    status="$(systemctl --user show "${unit}" --property=ExecMainStatus --value 2>/dev/null || echo 255)"
    if grep -Eqi 'out of memory|CUDA error|CUDNN_STATUS_ALLOC_FAILED' \
      "${SUITE_DIR}/service_logs/${unit%.service}.log"; then
      echo "[Error] ${unit} reported a CUDA allocation failure." >&2
      return 1
    fi
    if [[ "${status}" != "0" ]]; then
      echo "[Error] ${unit} failed with status ${status}; this is not an OOM fallback condition." >&2
      return 2
    fi
  done
  echo "[Canary] attempt=${attempt} peak_gpu_memory_mib=${peak} limit=${GPU_MEMORY_LIMIT_MIB}"
  if (( peak > GPU_MEMORY_LIMIT_MIB )); then
    echo "[Error] Canary exceeded the reserved GPU safety limit." >&2
    return 1
  fi
  return 0
}

run_canary_attempt() {
  local attempt="$1"
  local low_memory="$2"
  ensure_units_absent canary "${attempt}"
  local manifest
  manifest="$(prepare_manifest canary "${attempt}" "${low_memory}" | tail -n 1)"
  local socket_path
  socket_path="$(launch_evaluator canary "${attempt}" "${manifest}" 12 | tail -n 1)"
  wait_evaluator canary "${attempt}" "${socket_path}" 12
  launch_trainers canary "${attempt}" "${manifest}" "${socket_path}" 12
  local result=0
  wait_canary "${attempt}" || result=$?
  stop_evaluator canary "${attempt}"
  return "${result}"
}

verify_completed_canary() {
  local attempt="$1"
  local memory_log="${SUITE_DIR}/hardware/canary_${attempt}_gpu_memory.csv"
  local lane unit status result peak
  if [[ ! -s "${memory_log}" ]]; then
    echo "[Error] Missing completed canary memory record: ${memory_log}" >&2
    return 1
  fi
  peak="$(awk -F, 'NR > 1 && $2 + 0 > peak { peak=$2 + 0 } END { print peak + 0 }' "${memory_log}")"
  if (( peak <= 0 || peak > GPU_MEMORY_LIMIT_MIB )); then
    echo "[Error] Reused canary peak is invalid or unsafe: ${peak} MiB." >&2
    return 1
  fi
  for lane in 0 1 2; do
    unit="${UNIT_PREFIX}-canary-${attempt}-m${lane}.service"
    status="$(systemctl --user show "${unit}" --property=ExecMainStatus --value 2>/dev/null || echo 255)"
    result="$(systemctl --user show "${unit}" --property=Result --value 2>/dev/null || echo unknown)"
    if [[ "${status}" != "0" || "${result}" != "success" ]]; then
      echo "[Error] Reused canary unit did not succeed: ${unit} status=${status} result=${result}." >&2
      return 1
    fi
    if grep -Eqi 'out of memory|CUDA error|CUDNN_STATUS_ALLOC_FAILED|Traceback' \
      "${SUITE_DIR}/service_logs/${unit%.service}.log"; then
      echo "[Error] Reused canary log contains a fatal signature: ${unit}." >&2
      return 1
    fi
  done
  echo "[Canary] Reusing completed attempt=${attempt} peak_gpu_memory_mib=${peak}."
}

busy_pids="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -n "${busy_pids//[[:space:]]/}" ]]; then
  echo "[Error] GPU0 is busy before Stage2 canary; PIDs: ${busy_pids}" >&2
  exit 1
fi

selected_attempt=standard
selected_low_memory=false
canary_result=0
if [[ "${REUSE_COMPLETED_CANARY}" == "true" ]]; then
  verify_completed_canary standard
else
  run_canary_attempt standard false || canary_result=$?
  if (( canary_result == 1 )); then
    echo "[Fallback] Retrying three-process canary with 800-graph microbatches."
    selected_attempt=lowmem
    selected_low_memory=true
    canary_result=0
    run_canary_attempt lowmem true || canary_result=$?
    if (( canary_result != 0 )); then
      echo "[Error] Both three-process Stage2 canaries failed; formal Wave1 was not launched." >&2
      exit 1
    fi
  elif (( canary_result != 0 )); then
    echo "[Error] Stage2 canary failed for a non-memory reason; formal Wave1 was not launched." >&2
    exit "${canary_result}"
  fi
fi

if [[ "${CANARY_ONLY}" == "true" ]]; then
  echo "[Done] Stage2 three-process canary completed; CANARY_ONLY=true, formal launch skipped."
  exit 0
fi

ensure_units_absent wave1 "${selected_attempt}"
formal_manifest="$(prepare_manifest wave1 "${selected_attempt}" "${selected_low_memory}" | tail -n 1)"
formal_socket="$(launch_evaluator wave1 "${selected_attempt}" "${formal_manifest}" 60 | tail -n 1)"
wait_evaluator wave1 "${selected_attempt}" "${formal_socket}" 60
launch_trainers wave1 "${selected_attempt}" "${formal_manifest}" "${formal_socket}" 60

echo "[Done] Stage2 Wave1 launched on GPU0: ${METHODS[*]}"
echo "[Done] CPU policy: all services share ${CPU_POOL}; no inter-experiment CPU isolation."
echo "[Done] Manifest: ${formal_manifest}"
systemctl --user show \
  "${UNIT_PREFIX}-wave1-${selected_attempt}-eval.service" \
  "${UNIT_PREFIX}-wave1-${selected_attempt}-m0.service" \
  "${UNIT_PREFIX}-wave1-${selected_attempt}-m1.service" \
  "${UNIT_PREFIX}-wave1-${selected_attempt}-m2.service" \
  --property=Id --property=MainPID --property=ActiveState \
  --property=SubState --property=AllowedCPUs --property=CPUAffinity
