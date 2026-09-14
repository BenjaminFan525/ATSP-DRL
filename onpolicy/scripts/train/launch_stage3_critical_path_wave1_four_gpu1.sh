#!/usr/bin/env bash
set -euo pipefail

# Stage3 critical-path/rendezvous causal screen on GPU1:
#   C0: exact E2-style control (legacy wait objective, horizon1/frontier2)
#   C1: policy-invariant critical-slack potential, legacy horizon
#   C2: legacy objective, horizon3/frontier4 soft forecast
#   C3: critical-slack potential + horizon3/frontier4
# All arms share one Valid60 service, use graph=1000, own whole SMT sibling
# pairs, and serialize PPO updates through one GPU-local lock.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python3.11}"
PREPARE="${ROOT_DIR}/onpolicy/scripts/train/prepare_stage3_joint_finetune.py"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage3_joint_manifest_trial.py"
MANIFEST_CHECKER="${ROOT_DIR}/onpolicy/scripts/train/run_stage3_joint_manifest.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"
AUTOPILOT="${ROOT_DIR}/onpolicy/scripts/train/stage3_critical_path_wave1_autopilot.py"

GPU=1
PROFILE="${PROFILE:-critical_wave1}"
START="${START:-0}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-0}"
METHOD_FILTER="${METHOD_FILTER:-all}"
AUTO_ANALYZE="${AUTO_ANALYZE:-1}"
RUN_TAG="${RUN_TAG:-stage3_critical_path_wave1_20260825_r1}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s3crit-r1-g1}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
VALIDATION_DIR="${VALIDATION_DIR:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint/tune}"
RESULTS_ROOT="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${RESULTS_ROOT}/stage2_planning_wave4_20260822_r2_planning_wave_B2_wait_constraint_seed1/run1/models/checkpoint_Best.pt}"

GRAPHS_PER_FORWARD=1000
EVAL_WORKERS=10
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-20}"
GPU_PHASE_LOCK="${GPU_PHASE_LOCK:-/tmp/${UNIT_PREFIX}-${PROFILE}.ppo.lock}"
METHODS=(C0 C1 C2 C3)

# CPU x and x+72 are SMT siblings of one physical core.  No physical or
# logical core overlaps between a trainer and the shared validator.
TRAIN_CPUS=(
  "36-42,108-114"
  "43-49,115-121"
  "50-56,122-128"
  "57-63,129-135"
)
VAL_CPUS="64-71,136-143"
EXPECTED_CPU_SET="36-71,108-143"
EXPECTED_CPU_WIDTHS="16,14,14,14,14"

case "${PROFILE}" in
  memory_canary) MAX_EVAL_CASES=10 ;;
  canary) MAX_EVAL_CASES=20 ;;
  critical_wave1) MAX_EVAL_CASES=60 ;;
  *) echo "[Error] PROFILE must be memory_canary, canary, or critical_wave1." >&2; exit 1 ;;
esac

selected_lane() {
  local lane="$1"
  [[ "${METHOD_FILTER}" == all || "${METHODS[${lane}]}" == "${METHOD_FILTER}" ]]
}

unit_name() { echo "${UNIT_PREFIX}-${PROFILE}-$1"; }
manifest_for_lane() {
  local method="${METHODS[$1]}"
  echo "${SUITE_DIR}/commands/${PROFILE}_${method}.json"
}
experiment_name_for_lane() {
  local method="${METHODS[$1]}"
  echo "${RUN_TAG}_${PROFILE}_${method}_seed1"
}
experiment_dir_for_lane() {
  echo "${RESULTS_ROOT}/$(experiment_name_for_lane "$1")"
}

verify_cpu_partition() {
  "${PYTHON}" -c '
import sys
def expand(spec):
    values=set()
    for item in spec.split(","):
        lo, sep, hi=item.partition("-")
        values.update(range(int(lo), int(hi)+1) if sep else [int(lo)])
    return values
expected=expand(sys.argv[1]); widths=[int(x) for x in sys.argv[2].split(",")]
groups=[expand(x) for x in sys.argv[3:]]; union=set()
for index,(group,width) in enumerate(zip(groups,widths)):
    if len(group)!=width: raise SystemExit(f"CPU group {index}: {len(group)} != {width}")
    overlap=union & group
    if overlap: raise SystemExit(f"CPU overlap: {sorted(overlap)}")
    union.update(group)
if union!=expected:
    raise SystemExit(f"CPU coverage mismatch missing={sorted(expected-union)} extra={sorted(union-expected)}")
for logical in expected:
    sibling=logical+72 if logical<72 else logical-72
    if sibling not in expected: raise SystemExit(f"Split physical core at CPU {logical}")
print("[CPU] GPU1 partition: validator=16, four trainers=14 logical CPUs each")
' "${EXPECTED_CPU_SET}" "${EXPECTED_CPU_WIDTHS}" "${VAL_CPUS}" "${TRAIN_CPUS[@]}"
}

verify_hardware() {
  [[ "$(nproc --all)" -eq 144 ]] || {
    echo "[Error] Expected 144 host logical CPUs, got $(nproc --all)." >&2; exit 1;
  }
  local total busy
  total="$(nvidia-smi --id=1 --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
  (( total >= 80000 )) || { echo "[Error] GPU1 is not the 80-GiB target." >&2; exit 1; }
  busy="$(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -z "${busy//[[:space:]]/}" ]] || { echo "[Error] GPU1 is busy: ${busy}" >&2; exit 1; }
  verify_cpu_partition
}

prepare_manifests() {
  local lane method manifest
  mkdir -p "${SUITE_DIR}"/{commands,records,service_logs,shared_evaluator,analysis,autopilot}
  for lane in 0 1 2 3; do
    method="${METHODS[${lane}]}"
    manifest="$(manifest_for_lane "${lane}")"
    MPLCONFIGDIR=/tmp/hkbz-mpl "${PYTHON}" "${PREPARE}" \
      --profile "${PROFILE}" --method "${method}" \
      --max-graphs-per-forward "${GRAPHS_PER_FORWARD}" \
      --source-checkpoint "${SOURCE_CHECKPOINT}" --python "${PYTHON}" \
      --experiment-name "$(experiment_name_for_lane "${lane}")" \
      --seed 1 --output "${manifest}"
    PYTHON="${PYTHON}" "${PYTHON}" "${MANIFEST_CHECKER}" \
      "${manifest}" --check-only
  done
}

require_fresh_outputs() {
  local lane unit state path
  for lane in 0 1 2 3; do
    selected_lane "${lane}" || continue
    path="$(experiment_dir_for_lane "${lane}")"
    [[ ! -e "${path}" ]] || { echo "[Error] Refusing output reuse: ${path}" >&2; exit 1; }
    unit="$(unit_name "${METHODS[${lane}]}")"
    state="$(systemctl --user show "${unit}.service" -p LoadState --value 2>/dev/null || true)"
    [[ -z "${state}" || "${state}" == not-found ]] || { echo "[Error] Existing unit ${unit}.service." >&2; exit 1; }
  done
}

launch_evaluator() {
  local unit socket log
  unit="$(unit_name eval)"
  socket="/tmp/${unit}.sock"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  [[ ! -e "${socket}" ]] || { echo "[Error] Existing evaluator socket: ${socket}" >&2; exit 1; }
  systemd-run --user --unit="${unit}" --same-dir \
    --setenv=CUDA_VISIBLE_DEVICES=1 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONHASHSEED=0 --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=TimeoutStopSec=180 --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${VAL_CPUS}" --property="CPUAffinity=${VAL_CPUS}" \
    --property=NUMAPolicy=preferred --property=NUMAMask=1 \
    --property=MemoryHigh=32G --property=MemoryMax=48G \
    --property="StandardOutput=append:${log}" --property="StandardError=append:${log}" \
    "${PYTHON}" -u "${EVALUATOR}" \
      --source-command-json "$(manifest_for_lane 0)" \
      --socket-path "${socket}" --cpu-pool "${VAL_CPUS}" \
      --run-dir "${SUITE_DIR}/shared_evaluator/${PROFILE}_gpu1" \
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
    [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" == active ]] || {
      echo "[Error] Shared evaluator exited: ${unit}.service" >&2; return 1;
    }
    if [[ -S "${socket}" ]] && "${PYTHON}" -c '
import sys
from onpolicy.utils.shared_eval import SharedEvalClient
reply=SharedEvalClient(sys.argv[1], timeout_seconds=5).ping()
raise SystemExit(0 if reply.get("worker_count")==10 else 1)
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
  local socket="$1" lane method unit log cpu_set delay
  for lane in 0 1 2 3; do
    selected_lane "${lane}" || continue
    method="${METHODS[${lane}]}"
    unit="$(unit_name "${method}")"
    log="${SUITE_DIR}/service_logs/${unit}.log"
    cpu_set="${TRAIN_CPUS[${lane}]}"
    delay=$((lane * START_STAGGER_SECONDS))
    systemd-run --user --unit="${unit}" --same-dir \
      --setenv=CUDA_VISIBLE_DEVICES=1 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
      --setenv=PYTHONHASHSEED=0 --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
      --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
      --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
      --property=Type=exec --property=Restart=no --property=KillMode=control-group \
      --property=TimeoutStopSec=180 --property=LimitNOFILE=65536 \
      --property="AllowedCPUs=${cpu_set}" --property="CPUAffinity=${cpu_set}" \
      --property=NUMAPolicy=preferred --property=NUMAMask=1 \
      --property=MemoryHigh=72G --property=MemoryMax=88G \
      --property="StandardOutput=append:${log}" --property="StandardError=append:${log}" \
      "${PYTHON}" -u "${TRIAL_RUNNER}" \
        --manifest "$(manifest_for_lane "${lane}")" --gpu 1 --cpu-set "${cpu_set}" \
        --shared-eval-socket "${socket}" --shared-eval-cpu-set "${VAL_CPUS}" \
        --shared-gpu-phase-lock "${GPU_PHASE_LOCK}" --eval-workers "${EVAL_WORKERS}" \
        --start-delay-seconds "${delay}" \
        --record "${SUITE_DIR}/records/${PROFILE}_${method}.json" >/dev/null
    echo "[Launch] ${unit}.service method=${method} CPUs=${cpu_set} delay=${delay}s"
  done
}

stop_profile() {
  local lane
  for lane in 0 1 2 3; do
    systemctl --user stop "$(unit_name "${METHODS[${lane}]}").service" >/dev/null 2>&1 || true
  done
  systemctl --user stop "$(unit_name eval).service" >/dev/null 2>&1 || true
}

wait_for_trainers() {
  local active lane
  while true; do
    active=0
    for lane in 0 1 2 3; do
      selected_lane "${lane}" || continue
      [[ "$(systemctl --user is-active "$(unit_name "${METHODS[${lane}]}").service" 2>/dev/null || true)" == active ]] && active=$((active+1))
    done
    (( active == 0 )) && break
    sleep 10
  done
}

validate_completions() {
  local lane method status log
  for lane in 0 1 2 3; do
    selected_lane "${lane}" || continue
    method="${METHODS[${lane}]}"
    status="$(experiment_dir_for_lane "${lane}")/run1/run_status.json"
    log="${SUITE_DIR}/service_logs/$(unit_name "${method}").log"
    jq -e '
      .status == "completed" and .training_stage == "joint_finetune"
      and .phase == "joint_finetune_completed" and (.canary_rejected == false)
      and ((.actor_update_health.step_completion_rate // 0) | tonumber) >= 0.95
      and ((.actor_update_health.zero_update_shards // 0) | tonumber) == 0
      and ((.actor_update_health.empty_replay_fraction // 0) | tonumber) <= 0.01
      and ((.eval_completion_rate // 0) | tonumber) == 1
      and ((.eval_cycle_count // 1) | tonumber) == 0
      and ((.eval_timeout_count // 1) | tonumber) == 0
    ' "${status}" >/dev/null || { echo "[Error] Incomplete lane: ${status}" >&2; return 1; }
    grep -q "\[Memory\].*graphs_per_forward=${GRAPHS_PER_FORWARD}" "${log}" || {
      echo "[Error] No real graph=${GRAPHS_PER_FORWARD} update: ${log}" >&2; return 1;
    }
  done
}

launch_autopilot() {
  local unit log
  unit="${UNIT_PREFIX}-${PROFILE}-autopilot"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  systemd-run --user --unit="${unit}" --same-dir \
    --setenv=PYTHONHASHSEED=0 --setenv=OMP_NUM_THREADS=1 \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=MemoryHigh=1G --property=MemoryMax=2G \
    --property="StandardOutput=append:${log}" --property="StandardError=append:${log}" \
    "${PYTHON}" -u "${AUTOPILOT}" \
      --suite "${SUITE_DIR}" --results-root "${RESULTS_ROOT}" \
      --run-tag "${RUN_TAG}" --profile "${PROFILE}" \
      --unit-prefix "${UNIT_PREFIX}" --epochs 4 >/dev/null
  echo "[Launch] ${unit}.service will audit C0-C3 and select the Wave-2 candidate."
}

main() {
  local socket
  [[ "${METHOD_FILTER}" == all || " ${METHODS[*]} " == *" ${METHOD_FILTER} "* ]] || {
    echo "[Error] METHOD_FILTER must be all/C0/C1/C2/C3." >&2; exit 1;
  }
  [[ "${AUTO_ANALYZE}" == 0 || "${AUTO_ANALYZE}" == 1 ]] || {
    echo "[Error] AUTO_ANALYZE must be 0 or 1." >&2; exit 1;
  }
  if [[ "${PROFILE}" == critical_wave1 && "${AUTO_ANALYZE}" == 1 && "${METHOD_FILTER}" != all ]]; then
    echo "[Error] Automatic causal selection requires all C0-C3 arms." >&2
    exit 1
  fi
  prepare_manifests
  if [[ "${START}" != 1 ]]; then
    echo "[Ready] Prepared Stage3 ${PROFILE} manifests in ${SUITE_DIR}/commands"
    return 0
  fi
  verify_hardware
  require_fresh_outputs
  trap stop_profile ERR INT TERM
  socket="$(launch_evaluator)"
  wait_evaluator "${socket}"
  launch_trainers "${socket}"
  if [[ "${WAIT_FOR_COMPLETION}" == 1 ]]; then
    wait_for_trainers
    validate_completions
    systemctl --user stop "$(unit_name eval).service" >/dev/null 2>&1 || true
  elif [[ "${PROFILE}" == critical_wave1 && "${AUTO_ANALYZE}" == 1 ]]; then
    launch_autopilot
  fi
  trap - ERR INT TERM
  echo "[Started] Stage3 ${PROFILE} GPU1 methods=${METHOD_FILTER} graph=${GRAPHS_PER_FORWARD} shared_eval=${socket}"
}

main "$@"
