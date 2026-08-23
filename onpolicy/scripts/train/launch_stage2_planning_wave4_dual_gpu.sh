#!/usr/bin/env bash
set -euo pipefail

# One-command Stage2 Wave-4 controller:
#   1. generate the ten-arm canary manifest;
#   2. launch five isolated trainers + one shared validator on each A800;
#   3. require a real concurrent 1000-graph PPO update without OOM/swap;
#   4. generate the calibrated formal manifest and launch all ten screens.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python3.11}"
PREPARE="${ROOT_DIR}/onpolicy/scripts/train/prepare_stage2_resource_research.py"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage2_resource_manifest_trial.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"

RUN_TAG="${RUN_TAG:-stage2_planning_wave4_20260822_r1}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s2-plan-w4-r1}"
VALIDATION_DIR="${VALIDATION_DIR:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint/tune}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-77824}"
HOST_MEMORY_MIN_AVAILABLE_MIB="${HOST_MEMORY_MIN_AVAILABLE_MIB:-65536}"
SWAP_GROWTH_LIMIT_MIB="${SWAP_GROWTH_LIMIT_MIB:-8192}"
CANARY_TIMEOUT_SECONDS="${CANARY_TIMEOUT_SECONDS:-28800}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-10}"
RESUME_AFTER_CANARY="${RESUME_AFTER_CANARY:-0}"

METHODS_GPU0=(
  A0_legacy_c1 A1_release_eta A2_frontier
  A3_soft_reservation A4_hard_reservation
)
METHODS_GPU1=(
  B0_soft_control B1_iga_flow_bc B2_wait_constraint
  B3_iga_constraint B4_gradual_shared
)
TRAIN_CPUS_GPU0=(
  0-5,72-77 6-11,78-83 12-17,84-89 18-23,90-95 24-29,96-101
)
TRAIN_CPUS_GPU1=(
  36-41,108-113 42-47,114-119 48-53,120-125
  54-59,126-131 60-65,132-137
)
VAL_CPUS_GPU0="30-34,102-106"
VAL_CPUS_GPU1="66-70,138-142"
OS_CPUS="35,71,107,143"

for required in "${PYTHON}" "${PREPARE}" "${TRIAL_RUNNER}" \
  "${EVALUATOR}" "${VALIDATION_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing Wave-4 input: ${required}" >&2
    exit 1
  fi
done
if [[ "$(nproc)" -ne 144 ]]; then
  echo "[Error] Expected 144 logical CPUs, got $(nproc)." >&2
  exit 1
fi

mkdir -p "${SUITE_DIR}"/{commands,service_logs,records,shared_evaluator,hardware}

profile_slug() {
  case "$1" in
    planning_canary) echo pc ;;
    planning_wave) echo pw ;;
    *) return 1 ;;
  esac
}

unit_name() {
  local profile="$1" suffix="$2"
  echo "${UNIT_PREFIX}-$(profile_slug "${profile}")-${suffix}"
}

experiment_dir() {
  local profile="$1" method="$2"
  echo "${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/${RUN_TAG}_${profile}_${method}_seed1"
}

method_for_lane() {
  local gpu="$1" lane="$2"
  if [[ "${gpu}" -eq 0 ]]; then
    echo "${METHODS_GPU0[${lane}]}"
  else
    echo "${METHODS_GPU1[${lane}]}"
  fi
}

cpu_for_lane() {
  local gpu="$1" lane="$2"
  if [[ "${gpu}" -eq 0 ]]; then
    echo "${TRAIN_CPUS_GPU0[${lane}]}"
  else
    echo "${TRAIN_CPUS_GPU1[${lane}]}"
  fi
}

val_cpus() {
  [[ "$1" -eq 0 ]] && echo "${VAL_CPUS_GPU0}" || echo "${VAL_CPUS_GPU1}"
}

gpu_numa() {
  [[ "$1" -eq 0 ]] && echo 0 || echo 1
}

verify_hardware() {
  local rows row index name memory pci expected_pci expected_suffix node busy
  mapfile -t rows < <(nvidia-smi \
    --query-gpu=index,name,memory.total,pci.bus_id \
    --format=csv,noheader,nounits)
  if [[ "${#rows[@]}" -ne 2 ]]; then
    echo "[Error] Expected exactly two CUDA-visible compute GPUs; got ${#rows[@]}." >&2
    exit 1
  fi
  for row in "${rows[@]}"; do
    IFS=',' read -r index name memory pci <<<"${row}"
    index="${index//[[:space:]]/}"
    memory="${memory//[[:space:]]/}"
    name="${name# }"
    pci="${pci//[[:space:]]/}"
    if [[ "${index}" != 0 && "${index}" != 1 ]] \
      || [[ "${name}" != *"A800 80GB"* ]] \
      || (( memory < 80000 )); then
      echo "[Error] Refusing non-A800 compute target: ${row}" >&2
      exit 1
    fi
    expected_pci="0000:1b:00.0"; expected_suffix=":1b:00.0"
    if [[ "${index}" -eq 1 ]]; then
      expected_pci="0000:b6:00.0"; expected_suffix=":b6:00.0"
    fi
    pci="${pci,,}"
    if [[ "${pci}" != *"${expected_suffix}" ]]; then
      echo "[Error] GPU${index} PCI mismatch: ${pci} != *${expected_suffix}." >&2
      exit 1
    fi
    node="$(<"/sys/bus/pci/devices/${expected_pci}/numa_node")"
    if [[ "${node}" -ne "${index}" ]]; then
      echo "[Error] GPU${index} NUMA mismatch: node=${node}." >&2
      exit 1
    fi
    busy="$(nvidia-smi --id="${index}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "${busy//[[:space:]]/}" ]]; then
      echo "[Error] GPU${index} has compute processes: ${busy}" >&2
      exit 1
    fi
  done
}

verify_cpu_partition() {
  "${PYTHON}" - "${OS_CPUS}" "${VAL_CPUS_GPU0}" "${VAL_CPUS_GPU1}" \
    "${TRAIN_CPUS_GPU0[@]}" "${TRAIN_CPUS_GPU1[@]}" <<'PY'
import os, sys

def expand(spec):
    values = set()
    for part in spec.split(','):
        lo, sep, hi = part.partition('-')
        values.update(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    return values

groups = [expand(value) for value in sys.argv[1:]]
union = set()
for index, group in enumerate(groups):
    if len(group) != (4 if index == 0 else 10 if index in {1, 2} else 12):
        raise SystemExit(f'invalid CPU group width index={index}: {sorted(group)}')
    overlap = union.intersection(group)
    if overlap:
        raise SystemExit(f'CPU partition overlap: {sorted(overlap)}')
    union.update(group)
expected = set(range(os.cpu_count()))
if union != expected:
    raise SystemExit(f'CPU partition mismatch missing={sorted(expected-union)} extra={sorted(union-expected)}')
print('[CPU] ten isolated trainer lanes + two validators cover 0-143 exactly')
PY
}

require_fresh_profile() {
  local profile="$1" gpu lane method path unit state
  for gpu in 0 1; do
    for lane in 0 1 2 3 4; do
      method="$(method_for_lane "${gpu}" "${lane}")"
      path="$(experiment_dir "${profile}" "${method}")"
      if [[ -e "${path}" ]]; then
        echo "[Error] Refusing ambiguous run reuse: ${path}" >&2
        exit 1
      fi
      unit="$(unit_name "${profile}" "g${gpu}l${lane}").service"
      state="$(systemctl --user show "${unit}" -p LoadState --value 2>/dev/null || true)"
      if [[ -n "${state}" && "${state}" != not-found ]]; then
        echo "[Error] Existing unit ${unit}; change UNIT_PREFIX." >&2
        exit 1
      fi
    done
    unit="$(unit_name "${profile}" "g${gpu}eval").service"
    state="$(systemctl --user show "${unit}" -p LoadState --value 2>/dev/null || true)"
    if [[ -n "${state}" && "${state}" != not-found ]]; then
      echo "[Error] Existing unit ${unit}; change UNIT_PREFIX." >&2
      exit 1
    fi
  done
}

prepare_manifest() {
  local profile="$1"
  local manifest="${SUITE_DIR}/commands/${profile}.json"
  "${PYTHON}" "${PREPARE}" --run-tag "${RUN_TAG}" --profile "${profile}" \
    --suite-dir "${SUITE_DIR}" --output "${manifest}"
  echo "${manifest}"
}

launch_evaluator() {
  local profile="$1" manifest="$2" gpu="$3"
  local cpus node unit socket log
  cpus="$(val_cpus "${gpu}")"
  node="$(gpu_numa "${gpu}")"
  unit="$(unit_name "${profile}" "g${gpu}eval")"
  socket="/tmp/${unit}.sock"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  [[ ! -e "${socket}" ]] || { echo "[Error] Existing socket ${socket}" >&2; exit 1; }
  systemd-run --user --unit="${unit}" --same-dir \
    --setenv="CUDA_VISIBLE_DEVICES=${gpu}" --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=TimeoutStopSec=180 --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${cpus}" --property="CPUAffinity=${cpus}" \
    --property=NUMAPolicy=bind --property="NUMAMask=${node}" \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    "${PYTHON}" -u "${EVALUATOR}" --source-command-json "${manifest}" \
      --socket-path "${socket}" --cpu-pool "${cpus}" \
      --run-dir "${SUITE_DIR}/shared_evaluator/${profile}_gpu${gpu}" \
      --eval-dataset-dir "${VALIDATION_DIR}" --max-eval-cases 60 \
      --eval-partition-seed 20260811 \
      --eval-partition-stratify-by distribution >/dev/null
  echo "${socket}"
}

wait_evaluator() {
  local profile="$1" gpu="$2" socket="$3" unit deadline
  unit="$(unit_name "${profile}" "g${gpu}eval")"
  deadline=$((SECONDS + 1200))
  while (( SECONDS < deadline )); do
    if [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" != active ]]; then
      echo "[Error] Evaluator exited before readiness: ${unit}.service" >&2
      return 1
    fi
    if [[ -S "${socket}" ]] && "${PYTHON}" - "${socket}" <<'PY'
import sys
from onpolicy.utils.shared_eval import SharedEvalClient
reply = SharedEvalClient(sys.argv[1], timeout_seconds=5.0).ping()
raise SystemExit(0 if int(reply.get('worker_count', -1)) == 10 else 1)
PY
    then
      echo "[Ready] ${unit}.service GPU${gpu} workers=10 CPUs=$(val_cpus "${gpu}")"
      return 0
    fi
    sleep 2
  done
  echo "[Error] Evaluator readiness timeout: ${unit}.service" >&2
  return 1
}

launch_trainer() {
  local profile="$1" manifest="$2" gpu="$3" lane="$4" socket="$5"
  local method cpus eval_cpus node unit log delay
  method="$(method_for_lane "${gpu}" "${lane}")"
  cpus="$(cpu_for_lane "${gpu}" "${lane}")"
  eval_cpus="$(val_cpus "${gpu}")"
  node="$(gpu_numa "${gpu}")"
  unit="$(unit_name "${profile}" "g${gpu}l${lane}")"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  delay=$((lane * START_STAGGER_SECONDS))
  systemd-run --user --unit="${unit}" --same-dir \
    --setenv="CUDA_VISIBLE_DEVICES=${gpu}" --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=TimeoutStopSec=180 --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${cpus}" --property="CPUAffinity=${cpus}" \
    --property=NUMAPolicy=bind --property="NUMAMask=${node}" \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    "${PYTHON}" -u "${TRIAL_RUNNER}" --manifest "${manifest}" \
      --command-key "${method}" --gpu "${gpu}" --cpu-set "${cpus}" \
      --shared-eval-socket "${socket}" --shared-eval-cpu-set "${eval_cpus}" \
      --eval-workers 10 --start-delay-seconds "${delay}" \
      --record "${SUITE_DIR}/records/${profile}_g${gpu}l${lane}.json" >/dev/null
  echo "[Launch] ${unit}.service ${method} GPU${gpu} CPUs=${cpus} delay=${delay}s"
}

stop_profile() {
  local profile="$1" gpu lane
  for gpu in 0 1; do
    for lane in 0 1 2 3 4; do
      systemctl --user stop "$(unit_name "${profile}" "g${gpu}l${lane}").service" >/dev/null 2>&1 || true
    done
    systemctl --user stop "$(unit_name "${profile}" "g${gpu}eval").service" >/dev/null 2>&1 || true
  done
}

swap_used_mib() {
  awk '/SwapTotal:/ {total=$2} /SwapFree:/ {free=$2} END {printf "%d", (total-free)/1024}' /proc/meminfo
}

mem_available_mib() {
  awk '/MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo
}

latest_run_status() {
  local profile="$1" method="$2" run_dir
  local -a candidates=()
  run_dir="$(experiment_dir "${profile}" "${method}")"
  shopt -s nullglob
  candidates=("${run_dir}"/run*/run_status.json)
  shopt -u nullglob
  (( ${#candidates[@]} > 0 )) || return 1
  printf '%s\n' "${candidates[@]}" | sort -V | tail -n 1
}

validate_canary_completions() {
  local gpu lane method status_file unit
  for gpu in 0 1; do
    for lane in 0 1 2 3 4; do
      method="$(method_for_lane "${gpu}" "${lane}")"
      unit="$(unit_name planning_canary "g${gpu}l${lane}")"
      status_file="$(latest_run_status planning_canary "${method}")" || {
        echo "[Error] Canary has no run status: ${method}" >&2
        return 1
      }
      if ! jq -e '
        .status == "completed"
        and .event == "completed"
        and (.completed_shard | tonumber) == (.total_shards | tonumber)
        and (.actor_step_completion_rate | tonumber) >= 0.9
      ' "${status_file}" >/dev/null; then
        echo "[Error] Canary run did not complete cleanly: ${status_file}" >&2
        return 1
      fi
      if ! grep -q '\[Memory\].*graphs_per_forward=1000' \
        "${SUITE_DIR}/service_logs/${unit}.log"; then
        echo "[Error] Canary missed live 1000-graph update: ${unit}" >&2
        return 1
      fi
    done
  done
}

validate_canary_memory_record() {
  local log="${SUITE_DIR}/hardware/planning_canary_gpu_memory.csv"
  local peak0 peak1 min_available swap_start swap_end
  [[ -s "${log}" ]] || {
    echo "[Error] Missing canary memory record: ${log}" >&2
    return 1
  }
  read -r peak0 peak1 min_available swap_start swap_end < <(
    awk -F, '
      NR == 2 {min_available=$7; swap_start=$8}
      NR > 1 {
        if ($3 > peak0) peak0=$3
        if ($4 > peak1) peak1=$4
        if ($7 < min_available) min_available=$7
        swap_end=$8
      }
      END {
        if (NR < 2) exit 1
        printf "%d %d %d %d %d\n", peak0, peak1, min_available, swap_start, swap_end
      }
    ' "${log}"
  )
  if (( peak0 > GPU_MEMORY_LIMIT_MIB || peak1 > GPU_MEMORY_LIMIT_MIB )); then
    echo "[Error] Canary GPU peaks ${peak0}/${peak1} MiB exceed ${GPU_MEMORY_LIMIT_MIB}." >&2
    return 1
  fi
  if (( min_available < HOST_MEMORY_MIN_AVAILABLE_MIB )); then
    echo "[Error] Canary MemAvailable floor ${min_available} MiB is unsafe." >&2
    return 1
  fi
  if (( swap_end - swap_start > SWAP_GROWTH_LIMIT_MIB )); then
    echo "[Error] Canary swap grew $((swap_end - swap_start)) MiB." >&2
    return 1
  fi
  echo "[Canary] evidence accepted peaks=${peak0}/${peak1}MiB mem_floor=${min_available}MiB swap_growth=$((swap_end-swap_start))MiB"
}

wait_canary() {
  local profile=planning_canary log="${SUITE_DIR}/hardware/planning_canary_gpu_memory.csv"
  local deadline peak0=0 peak1=0 min_available=999999999 swap_start swap_now
  local active gpu lane unit available util0 util1 used0 used1 maximum
  deadline=$((SECONDS + CANARY_TIMEOUT_SECONDS))
  swap_start="$(swap_used_mib)"
  echo "unix_time,max_gpu_memory_used_mib,gpu0_memory_used_mib,gpu1_memory_used_mib,gpu0_util_percent,gpu1_util_percent,mem_available_mib,swap_used_mib,active_trainers" >"${log}"
  while (( SECONDS < deadline )); do
    mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    mapfile -t util < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    used0="${used[0]}"; used1="${used[1]}"; util0="${util[0]}"; util1="${util[1]}"
    maximum="${used0}"; (( used1 > maximum )) && maximum="${used1}"
    (( used0 > peak0 )) && peak0="${used0}"
    (( used1 > peak1 )) && peak1="${used1}"
    available="$(mem_available_mib)"; swap_now="$(swap_used_mib)"
    (( available < min_available )) && min_available="${available}"
    active=0
    for gpu in 0 1; do
      for lane in 0 1 2 3 4; do
        unit="$(unit_name "${profile}" "g${gpu}l${lane}")"
        [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" == active ]] && active=$((active + 1))
      done
    done
    echo "$(date +%s),${maximum},${used0},${used1},${util0},${util1},${available},${swap_now},${active}" >>"${log}"
    (( active == 0 )) && break
    sleep 5
  done
  (( SECONDS < deadline )) || { echo "[Error] ten-lane canary timeout." >&2; return 1; }
  validate_canary_completions
  if (( peak0 > GPU_MEMORY_LIMIT_MIB || peak1 > GPU_MEMORY_LIMIT_MIB )); then
    echo "[Error] Canary GPU peaks ${peak0}/${peak1} MiB exceed ${GPU_MEMORY_LIMIT_MIB}." >&2
    return 1
  fi
  if (( min_available < HOST_MEMORY_MIN_AVAILABLE_MIB )); then
    echo "[Error] Canary MemAvailable floor ${min_available} MiB is unsafe." >&2
    return 1
  fi
  if (( swap_now - swap_start > SWAP_GROWTH_LIMIT_MIB )); then
    echo "[Error] Canary swap grew $((swap_now - swap_start)) MiB." >&2
    return 1
  fi
  echo "[Canary] passed peaks=${peak0}/${peak1}MiB mem_floor=${min_available}MiB swap_growth=$((swap_now-swap_start))MiB"
}

launch_profile() {
  local profile="$1" manifest="$2" socket0 socket1 gpu lane
  socket0="$(launch_evaluator "${profile}" "${manifest}" 0)"
  socket1="$(launch_evaluator "${profile}" "${manifest}" 1)"
  wait_evaluator "${profile}" 0 "${socket0}"
  wait_evaluator "${profile}" 1 "${socket1}"
  for gpu in 0 1; do
    for lane in 0 1 2 3 4; do
      if [[ "${gpu}" -eq 0 ]]; then
        launch_trainer "${profile}" "${manifest}" "${gpu}" "${lane}" "${socket0}"
      else
        launch_trainer "${profile}" "${manifest}" "${gpu}" "${lane}" "${socket1}"
      fi
    done
  done
}

main() {
  local canary_manifest formal_manifest
  verify_hardware
  verify_cpu_partition
  if [[ "${RESUME_AFTER_CANARY}" == 1 ]]; then
    validate_canary_completions
    validate_canary_memory_record
  else
    require_fresh_profile planning_canary
    canary_manifest="$(prepare_manifest planning_canary | tail -n 1)"
    trap 'stop_profile planning_canary' ERR INT TERM
    launch_profile planning_canary "${canary_manifest}"
    wait_canary
  fi
  stop_profile planning_canary
  sleep 10

  require_fresh_profile planning_wave
  formal_manifest="$(prepare_manifest planning_wave | tail -n 1)"
  launch_profile planning_wave "${formal_manifest}"
  trap - ERR INT TERM
  echo "[Started] Wave-4 formal ten-arm screen"
  echo "[Started] suite=${SUITE_DIR}"
  echo "[Started] GPU0=${METHODS_GPU0[*]}"
  echo "[Started] GPU1=${METHODS_GPU1[*]}"
}

main "$@"
