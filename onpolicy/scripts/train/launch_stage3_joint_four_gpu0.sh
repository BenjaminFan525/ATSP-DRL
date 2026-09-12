#!/usr/bin/env bash
set -euo pipefail

# Run four pure-RL Stage3 seeds on GPU0 with one persistent validation pool.
# Preparation is the default; set START=1 after reviewing the manifests.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python3.11}"
PREPARE="${ROOT_DIR}/onpolicy/scripts/train/prepare_stage3_joint_finetune.py"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage3_joint_manifest_trial.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"

PROFILE="${PROFILE:-canary}"
START="${START:-0}"
GPU="${GPU:-0}"
RUN_TAG="${RUN_TAG:-stage3_joint_rl_four_20260823_r2}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s3-j4-r2-g${GPU}}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
VALIDATION_DIR="${VALIDATION_DIR:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint/tune}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage2_planning_wave4_20260822_r2_planning_wave_B2_wait_constraint_seed1/run1/models/checkpoint_Best.pt}"

GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-77824}"
GRAPHS_PER_FORWARD=800
HOST_MEMORY_MIN_AVAILABLE_MIB="${HOST_MEMORY_MIN_AVAILABLE_MIB:-98304}"
SWAP_GROWTH_LIMIT_MIB="${SWAP_GROWTH_LIMIT_MIB:-4096}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-20}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-}"

SEEDS=(1 2 3 4)
CPU_LAYOUT="${CPU_LAYOUT:-auto}"
if [[ "${CPU_LAYOUT}" == auto ]]; then
  [[ "${GPU}" == 1 ]] && CPU_LAYOUT=gpu1_local || CPU_LAYOUT=full_machine
fi
case "${CPU_LAYOUT}" in
  full_machine)
    # Four 16-core lanes span both NUMA nodes.  Use only when no other suite
    # owns CPUs on this host.
    TRAIN_CPUS=(
      "0-7,36-43,72-79,108-115"
      "8-15,44-51,80-87,116-123"
      "16-23,52-59,88-95,124-131"
      "24-31,60-67,96-103,132-139"
    )
    VAL_CPUS="32-34,68-69,104-106,140-141"
    OS_CPUS="35,70-71,107,142-143"
    EXPECTED_CPU_SET="0-143"
    EXPECTED_CPU_WIDTHS="6,10,32,32,32,32"
    NUMA_POLICY=interleave
    NUMA_MASK=0,1
    ;;
  gpu1_local)
    # GPU1 is local to NUMA node 1 (physical cores 36-71).  GPU0's active
    # Stage2 suite owns cores 0-34; core 35 is left to the host.  Each Stage3
    # lane owns eight complete physical cores including both SMT siblings.
    TRAIN_CPUS=(
      "36-43,108-115"
      "44-51,116-123"
      "52-59,124-131"
      "60-67,132-139"
    )
    VAL_CPUS="68-71,140-143"
    OS_CPUS="35,107"
    EXPECTED_CPU_SET="35-71,107-143"
    EXPECTED_CPU_WIDTHS="2,8,16,16,16,16"
    NUMA_POLICY=preferred
    NUMA_MASK=1
    ;;
  *) echo "[Error] CPU_LAYOUT must be auto, full_machine, or gpu1_local." >&2; exit 1 ;;
esac
GPU_PHASE_LOCK="${GPU_PHASE_LOCK:-/tmp/${UNIT_PREFIX}-gpu${GPU}.ppo.lock}"

case "${PROFILE}" in
  memory_canary) MAX_EVAL_CASES=10 ;;
  canary) MAX_EVAL_CASES=20 ;;
  formal) MAX_EVAL_CASES=60 ;;
  *) echo "[Error] PROFILE must be memory_canary, canary, or formal." >&2; exit 1 ;;
esac
if [[ -z "${WAIT_FOR_COMPLETION}" ]]; then
  [[ "${PROFILE}" == formal ]] && WAIT_FOR_COMPLETION=0 || WAIT_FOR_COMPLETION=1
fi

for required in "${PYTHON}" "${PREPARE}" "${TRIAL_RUNNER}" \
  "${EVALUATOR}" "${SOURCE_CHECKPOINT}" "${VALIDATION_DIR}"; do
  [[ -e "${required}" ]] || { echo "[Error] Missing Stage3 input: ${required}" >&2; exit 1; }
done
mkdir -p "${SUITE_DIR}"/{commands,records,service_logs,shared_evaluator,hardware}

unit_name() {
  local suffix="$1"
  echo "${UNIT_PREFIX}-${PROFILE}-${suffix}"
}

manifest_for_lane() {
  local lane="$1"
  echo "${SUITE_DIR}/commands/${PROFILE}_seed${SEEDS[${lane}]}.json"
}

experiment_name_for_lane() {
  local lane="$1"
  echo "${RUN_TAG}_${PROFILE}_seed${SEEDS[${lane}]}"
}

experiment_dir_for_lane() {
  local lane="$1"
  echo "${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/$(experiment_name_for_lane "${lane}")"
}

verify_cpu_partition() {
  "${PYTHON}" -c '
import os, sys
def expand(spec):
    values = set()
    for item in spec.split(","):
        lo, sep, hi = item.partition("-")
        values.update(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    return values
expected = expand(sys.argv[1])
expected_widths = [int(value) for value in sys.argv[2].split(",")]
groups = [expand(item) for item in sys.argv[3:]]
union = set()
for index, (group, width) in enumerate(zip(groups, expected_widths)):
    if len(group) != width:
        raise SystemExit(f"CPU group {index} width={len(group)} expected={width}")
    overlap = union & group
    if overlap:
        raise SystemExit(f"CPU overlap: {sorted(overlap)}")
    union.update(group)
if union != expected:
    raise SystemExit(
        f"CPU coverage mismatch missing={sorted(expected-union)} extra={sorted(union-expected)}"
    )
print(
    f"[CPU] isolated groups cover {len(expected)} logical CPUs; "
    f"trainer_width={expected_widths[2]} validator_width={expected_widths[1]}"
)
' "${EXPECTED_CPU_SET}" "${EXPECTED_CPU_WIDTHS}" \
    "${OS_CPUS}" "${VAL_CPUS}" "${TRAIN_CPUS[@]}"
}

verify_hardware() {
  [[ "$(nproc)" -eq 144 ]] || { echo "[Error] Expected 144 logical CPUs." >&2; exit 1; }
  local total busy
  total="$(nvidia-smi --id="${GPU}" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
  (( total >= 80000 )) || { echo "[Error] GPU${GPU} is not an 80-GiB training target." >&2; exit 1; }
  busy="$(nvidia-smi --id="${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -z "${busy//[[:space:]]/}" ]] || { echo "[Error] GPU${GPU} is busy: ${busy}" >&2; exit 1; }
  verify_cpu_partition
}

prepare_manifests() {
  local lane manifest experiment
  for lane in 0 1 2 3; do
    manifest="$(manifest_for_lane "${lane}")"
    experiment="$(experiment_name_for_lane "${lane}")"
    "${PYTHON}" "${PREPARE}" --profile "${PROFILE}" \
      --source-checkpoint "${SOURCE_CHECKPOINT}" --python "${PYTHON}" \
      --experiment-name "${experiment}" --seed "${SEEDS[${lane}]}" \
      --output "${manifest}"
    "${PYTHON}" "${ROOT_DIR}/onpolicy/scripts/train/run_stage3_joint_manifest.py" \
      "${manifest}" --check-only
  done
}

require_fresh_outputs() {
  local lane path unit state
  for lane in 0 1 2 3; do
    path="$(experiment_dir_for_lane "${lane}")"
    [[ ! -e "${path}" ]] || { echo "[Error] Refusing output reuse: ${path}" >&2; exit 1; }
    unit="$(unit_name "l${lane}").service"
    state="$(systemctl --user show "${unit}" -p LoadState --value 2>/dev/null || true)"
    [[ -z "${state}" || "${state}" == not-found ]] || { echo "[Error] Existing unit ${unit}." >&2; exit 1; }
  done
}

require_canary_gate() {
  [[ "${PROFILE}" != formal || "${ALLOW_FORMAL_WITHOUT_CANARY:-0}" == 1 ]] && return 0
  local lane status
  for lane in 0 1 2 3; do
    status="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/${RUN_TAG}_canary_seed${SEEDS[${lane}]}/run1/run_status.json"
    [[ -s "${status}" ]] || { echo "[Error] Missing canary status: ${status}" >&2; exit 1; }
    jq -e '
      .status == "completed"
      and .training_stage == "joint_finetune"
      and .phase == "joint_finetune_completed"
      and ((.actor_step_completion_rate // 0) | tonumber) >= 0.9
      and ((.last_eval_completion_rate // 0) | tonumber) == 1
      and ((.last_eval_cycle_count // 1) | tonumber) == 0
      and ((.last_eval_timeout_count // 1) | tonumber) == 0
    ' "${status}" >/dev/null || { echo "[Error] Canary gate failed: ${status}" >&2; exit 1; }
  done
}

launch_evaluator() {
  local unit socket log source_manifest
  unit="$(unit_name eval)"
  socket="/tmp/${unit}.sock"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  source_manifest="$(manifest_for_lane 0)"
  [[ ! -e "${socket}" ]] || { echo "[Error] Existing socket: ${socket}" >&2; exit 1; }
  systemd-run --user --unit="${unit}" --same-dir \
    --setenv="CUDA_VISIBLE_DEVICES=${GPU}" --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONHASHSEED=0 --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=TimeoutStopSec=180 --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${VAL_CPUS}" --property="CPUAffinity=${VAL_CPUS}" \
    --property="NUMAPolicy=${NUMA_POLICY}" --property="NUMAMask=${NUMA_MASK}" \
    --property=MemoryHigh=32G --property=MemoryMax=48G \
    --property="StandardOutput=append:${log}" --property="StandardError=append:${log}" \
    "${PYTHON}" -u "${EVALUATOR}" --source-command-json "${source_manifest}" \
      --socket-path "${socket}" --cpu-pool "${VAL_CPUS}" \
      --run-dir "${SUITE_DIR}/shared_evaluator/${PROFILE}_gpu${GPU}" \
      --eval-dataset-dir "${VALIDATION_DIR}" --max-eval-cases "${MAX_EVAL_CASES}" \
      --eval-case-offset 0 --eval-partition-seed 20260803 \
      --eval-partition-stratify-by profile --cuda-memory-fraction 0.06 >/dev/null
  echo "${socket}"
}

wait_evaluator() {
  local socket="$1" unit deadline
  unit="$(unit_name eval)"
  deadline=$((SECONDS + 1200))
  while (( SECONDS < deadline )); do
    [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" == active ]] \
      || { echo "[Error] Shared evaluator exited: ${unit}.service" >&2; return 1; }
    if [[ -S "${socket}" ]] && "${PYTHON}" -c '
import sys
from onpolicy.utils.shared_eval import SharedEvalClient
reply = SharedEvalClient(sys.argv[1], timeout_seconds=5).ping()
raise SystemExit(0 if reply.get("worker_count") == 10 else 1)
' "${socket}"; then
      echo "[Ready] ${unit}.service workers=10 CPUs=${VAL_CPUS}"
      return 0
    fi
    sleep 2
  done
  echo "[Error] Shared evaluator readiness timeout." >&2
  return 1
}

launch_trainers() {
  local socket="$1" lane unit log delay
  for lane in 0 1 2 3; do
    unit="$(unit_name "l${lane}")"
    log="${SUITE_DIR}/service_logs/${unit}.log"
    delay=$((lane * START_STAGGER_SECONDS))
    systemd-run --user --unit="${unit}" --same-dir \
      --setenv="CUDA_VISIBLE_DEVICES=${GPU}" --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
      --setenv=PYTHONHASHSEED=0 --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
      --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
      --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
      --property=Type=exec --property=Restart=no --property=KillMode=control-group \
      --property=TimeoutStopSec=180 --property=LimitNOFILE=65536 \
      --property="AllowedCPUs=${TRAIN_CPUS[${lane}]}" \
      --property="CPUAffinity=${TRAIN_CPUS[${lane}]}" \
      --property="NUMAPolicy=${NUMA_POLICY}" --property="NUMAMask=${NUMA_MASK}" \
      --property=MemoryHigh=72G --property=MemoryMax=88G \
      --property="StandardOutput=append:${log}" --property="StandardError=append:${log}" \
      "${PYTHON}" -u "${TRIAL_RUNNER}" --manifest "$(manifest_for_lane "${lane}")" \
        --gpu "${GPU}" --cpu-set "${TRAIN_CPUS[${lane}]}" \
        --shared-eval-socket "${socket}" --shared-eval-cpu-set "${VAL_CPUS}" \
        --shared-gpu-phase-lock "${GPU_PHASE_LOCK}" \
        --eval-workers 10 --start-delay-seconds "${delay}" \
        --record "${SUITE_DIR}/records/${PROFILE}_g${GPU}l${lane}.json" >/dev/null
    echo "[Launch] ${unit}.service seed=${SEEDS[${lane}]} CPUs=${TRAIN_CPUS[${lane}]} delay=${delay}s"
  done
}

stop_profile() {
  local lane
  for lane in 0 1 2 3; do
    systemctl --user stop "$(unit_name "l${lane}").service" >/dev/null 2>&1 || true
  done
  systemctl --user stop "$(unit_name eval).service" >/dev/null 2>&1 || true
}

mem_available_mib() {
  awk '/MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo
}

swap_used_mib() {
  awk '/SwapTotal:/ {total=$2} /SwapFree:/ {free=$2} END {printf "%d", (total-free)/1024}' /proc/meminfo
}

wait_and_monitor() {
  local output="${SUITE_DIR}/hardware/${PROFILE}_gpu_memory.csv"
  local active lane unit used util available swap_now swap_start min_available peak=0
  swap_start="$(swap_used_mib)"
  min_available=999999999
  echo "unix_time,gpu_memory_used_mib,gpu_util_percent,mem_available_mib,swap_used_mib,active_trainers" >"${output}"
  while true; do
    used="$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    util="$(nvidia-smi --id="${GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    available="$(mem_available_mib)"; swap_now="$(swap_used_mib)"
    (( used > peak )) && peak="${used}"
    (( available < min_available )) && min_available="${available}"
    active=0
    for lane in 0 1 2 3; do
      unit="$(unit_name "l${lane}")"
      [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" == active ]] \
        && active=$((active + 1))
    done
    echo "$(date +%s),${used},${util},${available},${swap_now},${active}" >>"${output}"
    if (( used > GPU_MEMORY_LIMIT_MIB )); then
      echo "[Error] GPU memory ${used} MiB exceeds ${GPU_MEMORY_LIMIT_MIB} MiB." >&2
      stop_profile
      return 1
    fi
    (( active == 0 )) && break
    sleep 5
  done
  (( min_available >= HOST_MEMORY_MIN_AVAILABLE_MIB )) \
    || { echo "[Error] Host MemAvailable floor ${min_available} MiB." >&2; return 1; }
  (( swap_now - swap_start <= SWAP_GROWTH_LIMIT_MIB )) \
    || { echo "[Error] Swap grew $((swap_now-swap_start)) MiB." >&2; return 1; }
  echo "[Memory] peak_gpu=${peak}MiB mem_floor=${min_available}MiB swap_growth=$((swap_now-swap_start))MiB"
}

validate_completions() {
  local lane status log unit
  for lane in 0 1 2 3; do
    status="$(experiment_dir_for_lane "${lane}")/run1/run_status.json"
    log="${SUITE_DIR}/service_logs/$(unit_name "l${lane}").log"
    jq -e '
      .status == "completed"
      and .training_stage == "joint_finetune"
      and .phase == "joint_finetune_completed"
      and ((.actor_step_completion_rate // 0) | tonumber) >= 0.9
    ' "${status}" >/dev/null || { echo "[Error] Incomplete Stage3 lane: ${status}" >&2; return 1; }
    grep -q "\\[Memory\\].*graphs_per_forward=${GRAPHS_PER_FORWARD}" "${log}" \
      || { echo "[Error] Lane missed a real ${GRAPHS_PER_FORWARD}-graph update: ${log}" >&2; return 1; }
  done
  echo "[Complete] Four Stage3 ${PROFILE} lanes passed update-health checks."
}

main() {
  local socket
  prepare_manifests
  if [[ "${START}" != 1 ]]; then
    echo "[Ready] Prepared four ${PROFILE} manifests under ${SUITE_DIR}/commands"
    echo "[Ready] Launch with START=1 PROFILE=${PROFILE} $0"
    return 0
  fi
  verify_hardware
  require_canary_gate
  require_fresh_outputs
  trap stop_profile ERR INT TERM
  socket="$(launch_evaluator)"
  wait_evaluator "${socket}"
  launch_trainers "${socket}"
  if [[ "${WAIT_FOR_COMPLETION}" == 1 ]]; then
    wait_and_monitor
    validate_completions
    systemctl --user stop "$(unit_name eval).service" >/dev/null 2>&1 || true
  fi
  trap - ERR INT TERM
  echo "[Started] Stage3 ${PROFILE}: seeds=${SEEDS[*]} GPU${GPU} shared_eval=${socket}"
}

main "$@"
