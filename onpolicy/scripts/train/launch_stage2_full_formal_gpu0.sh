#!/usr/bin/env bash
set -euo pipefail

# One-command Stage2 full-data formal launcher.
#
# Five trainers share GPU0 and one persistent validation service.  Only NUMA0
# is available to this suite (72/144 logical CPUs).  Every trainer owns six
# complete physical cores (both SMT siblings), the evaluator owns five, and
# one physical core is reserved for the lightweight monitor/host work.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python3.11}"
PREPARE="${ROOT_DIR}/onpolicy/scripts/train/prepare_stage2_full_formal.py"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage2_resource_manifest_trial.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"

RUN_TAG="${RUN_TAG:-stage2_full_formal_20260823_r1}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s2-full-20260823-r1}"
VALIDATION_DIR="${VALIDATION_DIR:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint/tune}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-15}"
BC_WAIT_TIMEOUT_SECONDS="${BC_WAIT_TIMEOUT_SECONDS:-604800}"
DRY_RUN="${DRY_RUN:-0}"

METHODS=(
  F0_soft_control
  F1_hard_reservation
  F2_iga_flow_bc
  F3_wait_constraint
  F4_iga_constraint
)
TRAIN_CPUS=(
  0-5,72-77
  6-11,78-83
  12-17,84-89
  18-23,90-95
  24-29,96-101
)
VAL_CPUS="30-34,102-106"
MONITOR_CPUS="35,107"

manifest_path() {
  echo "${SUITE_DIR}/commands/full_formal.json"
}

trainer_unit() {
  echo "${UNIT_PREFIX}-g0l$1"
}

evaluator_unit() {
  echo "${UNIT_PREFIX}-g0eval"
}

monitor_unit() {
  echo "${UNIT_PREFIX}-monitor"
}

experiment_dir() {
  local method="$1"
  echo "${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/${RUN_TAG}_planning_formal_full_${method}_seed1"
}

resource_bc_checkpoint() {
  local method="$1"
  echo "$(experiment_dir "${method}")/run1/models/checkpoint_DeviceBC.pt"
}

bc_source_for_method() {
  case "$1" in
    F3_wait_constraint) echo F0_soft_control ;;
    F4_iga_constraint) echo F2_iga_flow_bc ;;
    *) return 1 ;;
  esac
}

verify_inputs() {
  local required
  for required in "${PYTHON}" "${PREPARE}" "${TRIAL_RUNNER}" \
    "${EVALUATOR}" "${VALIDATION_DIR}"; do
    if [[ ! -e "${required}" ]]; then
      echo "[Error] Missing Stage2 formal input: ${required}" >&2
      exit 1
    fi
  done
  if [[ "$(nproc)" -ne 144 ]]; then
    echo "[Error] Expected 144 logical CPUs, got $(nproc)." >&2
    exit 1
  fi
}

verify_cpu_partition() {
  "${PYTHON}" - "${TRAIN_CPUS[@]}" "${VAL_CPUS}" "${MONITOR_CPUS}" <<'PY'
import os
import sys

def expand(spec):
    values = set()
    for part in spec.split(','):
        lo, sep, hi = part.partition('-')
        values.update(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    return values

if os.cpu_count() != 144:
    raise SystemExit(f'expected 144 logical CPUs, got {os.cpu_count()}')
groups = [expand(value) for value in sys.argv[1:]]
widths = [12, 12, 12, 12, 12, 10, 2]
if [len(group) for group in groups] != widths:
    raise SystemExit(
        f'CPU group widths mismatch: {[len(group) for group in groups]} != {widths}'
    )
union = set()
for index, group in enumerate(groups):
    overlap = union.intersection(group)
    if overlap:
        raise SystemExit(f'CPU group {index} overlaps at {sorted(overlap)}')
    union.update(group)
    for cpu in group:
        sibling = cpu + 72 if cpu < 72 else cpu - 72
        if sibling not in group:
            raise SystemExit(
                f'CPU group {index} splits physical siblings {cpu}/{sibling}'
            )
expected = set(range(36)) | set(range(72, 108))
if union != expected:
    raise SystemExit(
        f'NUMA0 partition mismatch missing={sorted(expected-union)} '
        f'extra={sorted(union-expected)}'
    )
trainer_union = set().union(*groups[:5])
if len(trainer_union) != 60 or len(union) != 72:
    raise SystemExit('Stage2 suite exceeds its 50% CPU budget.')
print('[CPU] 5x12 trainer + 10 validator + 2 monitor/host = 72/144 logical CPUs')
PY
}

verify_gpu0() {
  local row index name memory pci node busy
  row="$(nvidia-smi --id=0 \
    --query-gpu=index,name,memory.total,pci.bus_id \
    --format=csv,noheader,nounits)"
  IFS=',' read -r index name memory pci <<<"${row}"
  index="${index//[[:space:]]/}"
  memory="${memory//[[:space:]]/}"
  name="${name# }"
  pci="${pci//[[:space:]]/}"
  if [[ "${index}" != 0 || "${name}" != *"A800 80GB"* ]] \
    || (( memory < 80000 )); then
    echo "[Error] GPU0 is not the calibrated A800 80GB: ${row}" >&2
    exit 1
  fi
  if [[ "${pci,,}" != *":1b:00.0" ]]; then
    echo "[Error] GPU0 PCI/NUMA identity changed: ${pci}." >&2
    exit 1
  fi
  node="$(< /sys/bus/pci/devices/0000:1b:00.0/numa_node)"
  if [[ "${node}" -ne 0 ]]; then
    echo "[Error] GPU0 must be local to NUMA0, got node=${node}." >&2
    exit 1
  fi
  busy="$(nvidia-smi --id=0 --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "${busy//[[:space:]]/}" ]]; then
    echo "[Error] GPU0 still has compute processes: ${busy}" >&2
    exit 1
  fi
}

require_fresh_run() {
  local lane method path unit state
  if [[ -e "${SUITE_DIR}/commands/full_formal.json" ]]; then
    echo "[Error] Refusing existing formal suite: ${SUITE_DIR}" >&2
    exit 1
  fi
  for lane in 0 1 2 3 4; do
    method="${METHODS[${lane}]}"
    path="$(experiment_dir "${method}")"
    if [[ -e "${path}" ]]; then
      echo "[Error] Refusing ambiguous experiment reuse: ${path}" >&2
      exit 1
    fi
    unit="$(trainer_unit "${lane}").service"
    state="$(systemctl --user show "${unit}" -p LoadState --value 2>/dev/null || true)"
    if [[ -n "${state}" && "${state}" != not-found ]]; then
      echo "[Error] Existing systemd unit ${unit}; change UNIT_PREFIX." >&2
      exit 1
    fi
  done
  for unit in "$(evaluator_unit).service" "$(monitor_unit).service"; do
    state="$(systemctl --user show "${unit}" -p LoadState --value 2>/dev/null || true)"
    if [[ -n "${state}" && "${state}" != not-found ]]; then
      echo "[Error] Existing systemd unit ${unit}; change UNIT_PREFIX." >&2
      exit 1
    fi
  done
}

prepare_manifest() {
  mkdir -p "${SUITE_DIR}"/{commands,service_logs,records,shared_evaluator,hardware}
  "${PYTHON}" "${PREPARE}" \
    --run-tag "${RUN_TAG}" --suite-dir "${SUITE_DIR}" \
    --output "$(manifest_path)"
}

launch_evaluator() {
  local unit socket log
  unit="$(evaluator_unit)"
  socket="/tmp/${unit}.sock"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  [[ ! -e "${socket}" ]] || {
    echo "[Error] Shared-evaluator socket already exists: ${socket}" >&2
    exit 1
  }
  systemd-run --user --unit="${unit}" --same-dir \
    --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=TimeoutStopSec=180 --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${VAL_CPUS}" --property="CPUAffinity=${VAL_CPUS}" \
    --property=NUMAPolicy=bind --property=NUMAMask=0 \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    "${PYTHON}" -u "${EVALUATOR}" \
      --source-command-json "$(manifest_path)" --socket-path "${socket}" \
      --cpu-pool "${VAL_CPUS}" \
      --run-dir "${SUITE_DIR}/shared_evaluator/full_formal_gpu0" \
      --eval-dataset-dir "${VALIDATION_DIR}" --max-eval-cases 60 \
      --eval-partition-seed 20260811 \
      --eval-partition-stratify-by distribution >/dev/null
  echo "${socket}"
}

wait_evaluator() {
  local socket="$1" unit deadline
  unit="$(evaluator_unit)"
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
      echo "[Ready] ${unit}.service GPU0 workers=10 CPUs=${VAL_CPUS}"
      return 0
    fi
    sleep 2
  done
  echo "[Error] Evaluator readiness timeout: ${unit}.service" >&2
  return 1
}

launch_trainer() {
  local lane="$1" socket="$2"
  local method cpus unit log delay bc_source
  local -a reuse_args=()
  method="${METHODS[${lane}]}"
  cpus="${TRAIN_CPUS[${lane}]}"
  unit="$(trainer_unit "${lane}")"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  delay=$((lane * START_STAGGER_SECONDS))
  if bc_source="$(bc_source_for_method "${method}" 2>/dev/null)"; then
    reuse_args=(
      --resource-bc-checkpoint "$(resource_bc_checkpoint "${bc_source}")"
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
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=TimeoutStopSec=180 --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${cpus}" --property="CPUAffinity=${cpus}" \
    --property=NUMAPolicy=bind --property=NUMAMask=0 \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    "${PYTHON}" -u "${TRIAL_RUNNER}" \
      --manifest "$(manifest_path)" --command-key "${method}" \
      --gpu 0 --cpu-set "${cpus}" \
      --shared-eval-socket "${socket}" --shared-eval-cpu-set "${VAL_CPUS}" \
      --eval-workers 10 --start-delay-seconds "${delay}" \
      --record "${SUITE_DIR}/records/full_formal_g0l${lane}.json" \
      "${reuse_args[@]}" >/dev/null
  if [[ ${#reuse_args[@]} -gt 0 ]]; then
    echo "[Launch] ${unit}.service ${method} CPUs=${cpus} waits_for_BC=${bc_source}"
  else
    echo "[Launch] ${unit}.service ${method} CPUs=${cpus} full_BC_leader=true"
  fi
}

stop_started_units() {
  local lane
  systemctl --user stop "$(monitor_unit).service" >/dev/null 2>&1 || true
  for lane in 0 1 2 3 4; do
    systemctl --user stop "$(trainer_unit "${lane}").service" >/dev/null 2>&1 || true
  done
  systemctl --user stop "$(evaluator_unit).service" >/dev/null 2>&1 || true
}

monitor_main() {
  local output lane active state gpu_used gpu_util mem_available swap_used
  output="${SUITE_DIR}/hardware/runtime_usage.csv"
  echo "unix_time,gpu0_memory_used_mib,gpu0_util_percent,mem_available_mib,swap_used_mib,active_trainers" >"${output}"
  while true; do
    active=0
    for lane in 0 1 2 3 4; do
      state="$(systemctl --user is-active "$(trainer_unit "${lane}").service" 2>/dev/null || true)"
      [[ "${state}" == active || "${state}" == activating ]] && active=$((active + 1))
    done
    gpu_used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    gpu_util="$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    mem_available="$(awk '/MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo)"
    swap_used="$(awk '/SwapTotal:/ {t=$2} /SwapFree:/ {f=$2} END {printf "%d", (t-f)/1024}' /proc/meminfo)"
    echo "$(date +%s),${gpu_used},${gpu_util},${mem_available},${swap_used},${active}" >>"${output}"
    if (( active == 0 )); then
      systemctl --user stop "$(evaluator_unit).service" >/dev/null 2>&1 || true
      break
    fi
    sleep 30
  done
}

launch_monitor() {
  local unit log
  unit="$(monitor_unit)"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  systemd-run --user --unit="${unit}" --same-dir \
    --setenv=HKBZ_STAGE2_FULL_MONITOR=1 \
    --setenv="RUN_TAG=${RUN_TAG}" --setenv="SUITE_DIR=${SUITE_DIR}" \
    --setenv="UNIT_PREFIX=${UNIT_PREFIX}" \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property="AllowedCPUs=${MONITOR_CPUS}" \
    --property="CPUAffinity=${MONITOR_CPUS}" \
    --property=NUMAPolicy=bind --property=NUMAMask=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /bin/bash "${BASH_SOURCE[0]}" >/dev/null
}

verify_initial_services() {
  local lane unit state cpus expected deadline
  deadline=$((SECONDS + 120))
  while (( SECONDS < deadline )); do
    for lane in 0 1 2 3 4; do
      state="$(systemctl --user is-active "$(trainer_unit "${lane}").service" 2>/dev/null || true)"
      [[ "${state}" == active || "${state}" == activating ]] || break 2
    done
    sleep 2
    break
  done
  for lane in 0 1 2 3 4; do
    unit="$(trainer_unit "${lane}")"
    state="$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)"
    if [[ "${state}" != active && "${state}" != activating ]]; then
      echo "[Error] Trainer failed initial health check: ${unit}.service state=${state}" >&2
      return 1
    fi
    cpus="$(systemctl --user show "${unit}.service" -p CPUAffinity --value)"
    expected="${TRAIN_CPUS[${lane}]}"
    echo "[Healthy] ${unit}.service method=${METHODS[${lane}]} CPUs=${expected} state=${state}"
  done
}

main() {
  local socket
  verify_inputs
  verify_cpu_partition
  require_fresh_run
  prepare_manifest
  if [[ "${DRY_RUN}" == 1 ]]; then
    echo "[DryRun] manifest=$(manifest_path)"
    echo "[DryRun] GPU0 methods=${METHODS[*]}"
    echo "[DryRun] CPU budget=72/144 logical CPUs; no service launched"
    return 0
  fi
  verify_gpu0
  trap stop_started_units ERR INT TERM
  socket="$(launch_evaluator)"
  wait_evaluator "${socket}"
  for lane in 0 1 2 3 4; do
    launch_trainer "${lane}" "${socket}"
  done
  verify_initial_services
  launch_monitor
  trap - ERR INT TERM
  echo "[Started] Stage2 full-data formal five-arm suite"
  echo "[Started] suite=${SUITE_DIR}"
  echo "[Started] manifest=$(manifest_path)"
  echo "[Started] GPU0 only; 70 training/eval + 2 monitor logical CPUs"
}

if [[ "${HKBZ_STAGE2_FULL_MONITOR:-0}" == 1 ]]; then
  monitor_main
else
  main "$@"
fi
