#!/usr/bin/env bash
set -euo pipefail

# Stage3 strict PPO-gain screen on GPU1.  Four trainers own disjoint complete
# SMT sibling pairs, share one Valid60 service, and serialize graph=1000 PPO
# updates so the active updater can use the large GPU allocator safely.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python3.11}"
PREPARE="${ROOT_DIR}/onpolicy/scripts/train/prepare_stage3_joint_finetune.py"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage3_joint_manifest_trial.py"
MANIFEST_CHECKER="${ROOT_DIR}/onpolicy/scripts/train/run_stage3_joint_manifest.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"
AUTOPILOT="${ROOT_DIR}/onpolicy/scripts/train/stage3_ppo_gain_autopilot.py"
LAUNCHER="${ROOT_DIR}/onpolicy/scripts/train/launch_stage3_ppo_gain_four_gpu1.sh"

GPU="${GPU:-1}"
PROFILE="${PROFILE:-ppo_gain_wave1}"
START="${START:-0}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-0}"
METHOD_FILTER="${METHOD_FILTER:-all}"
AUTO_ANALYZE="${AUTO_ANALYZE:-1}"
AUTO_NEXT_WAVE="${AUTO_NEXT_WAVE:-1}"
BASE_CREDIT_METHOD="${BASE_CREDIT_METHOD:-P3}"
INHERIT_SOURCE_CONTRACTS="${INHERIT_SOURCE_CONTRACTS:-0}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
RUN_TAG="${RUN_TAG:-stage3_ppo_gain_wave1_20260826_r1}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s3gain-r1-g1}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
RESULTS_LOG_ROOT="${ROOT_DIR}/result/hkbz_train_logs"
VALIDATION_DIR="${VALIDATION_DIR:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint/tune}"
RESULTS_ROOT="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${RESULTS_ROOT}/stage2_planning_wave4_20260822_r2_planning_wave_B2_wait_constraint_seed1/run1/models/checkpoint_Best.pt}"
SEED="${SEED:-1}"

GRAPHS_PER_FORWARD=1000
EVAL_WORKERS=10
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-20}"
GPU_PHASE_LOCK="${GPU_PHASE_LOCK:-/tmp/${UNIT_PREFIX}-${PROFILE}.ppo.lock}"

case "${PROFILE}" in
  ppo_gain_memory_canary)
    METHODS=(P0 P1 P2 P3)
    MAX_EVAL_CASES=10
    ;;
  ppo_gain_wave1)
    METHODS=(P0 P1 P2 P3)
    MAX_EVAL_CASES=60
    ;;
  ppo_gain_wave2)
    METHODS=(Q0 Q1 Q2 Q3)
    MAX_EVAL_CASES=60
    ;;
  credit_happo_memory_canary)
    METHODS=(N0 N1 N2 N3)
    MAX_EVAL_CASES=10
    ;;
  credit_happo_wave1)
    METHODS=(N0 N1 N2 N3)
    MAX_EVAL_CASES=60
    ;;
  gradient_conflict_memory_canary)
    METHODS=(G0 G1 G2 G3)
    MAX_EVAL_CASES=10
    ;;
  gradient_conflict_wave1)
    METHODS=(G0 G1 G2 G3)
    MAX_EVAL_CASES=60
    ;;
  *)
    echo "[Error] unsupported four-lane Stage3 PROFILE=${PROFILE}." >&2
    exit 1
    ;;
esac

# CPU x and x+72 are the two SMT threads of one physical core.  Every group
# owns whole physical cores and neither trainers nor evaluator overlap.
if [[ "${GPU}" == 0 ]]; then
  TRAIN_CPUS=(
    "0-6,72-78"
    "7-13,79-85"
    "14-20,86-92"
    "21-27,93-99"
  )
  VAL_CPUS="28-35,100-107"
  EXPECTED_CPU_SET="0-35,72-107"
  NUMA_NODE=0
else
  TRAIN_CPUS=(
    "36-42,108-114"
    "43-49,115-121"
    "50-56,122-128"
    "57-63,129-135"
  )
  VAL_CPUS="64-71,136-143"
  EXPECTED_CPU_SET="36-71,108-143"
  NUMA_NODE=1
fi
EXPECTED_CPU_WIDTHS="16,14,14,14,14"

selected_lane() {
  local lane="$1"
  [[ "${METHOD_FILTER}" == all || "${METHODS[${lane}]}" == "${METHOD_FILTER}" ]]
}

unit_name() { echo "${UNIT_PREFIX}-${PROFILE}-$1"; }
manifest_for_lane() {
  echo "${SUITE_DIR}/commands/${PROFILE}_${METHODS[$1]}.json"
}
experiment_name_for_lane() {
  echo "${RUN_TAG}_${PROFILE}_${METHODS[$1]}_seed${SEED}"
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
print("[CPU] validator=16 logical; four trainers=14 logical each; whole SMT pairs")
' "${EXPECTED_CPU_SET}" "${EXPECTED_CPU_WIDTHS}" "${VAL_CPUS}" "${TRAIN_CPUS[@]}"
}

verify_hardware() {
  [[ "$(nproc --all)" -eq 144 ]] || {
    echo "[Error] Expected 144 host logical CPUs, got $(nproc --all)." >&2
    exit 1
  }
  local total busy
  total="$(nvidia-smi --id=${GPU} --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
  (( total >= 80000 )) || {
    echo "[Error] GPU${GPU} is not the 80-GiB target." >&2
    exit 1
  }
  busy="$(nvidia-smi --id=${GPU} --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -z "${busy//[[:space:]]/}" || "${ALLOW_BUSY_GPU}" == 1 ]] || {
    echo "[Error] GPU${GPU} is busy; preserving existing PIDs: ${busy}" >&2
    exit 1
  }
  if [[ -n "${busy//[[:space:]]/}" ]]; then
    echo "[Warning] GPU${GPU} co-scheduling explicitly enabled; existing PIDs: ${busy}"
  fi
  verify_cpu_partition
}

prepare_manifests() {
  local lane method manifest
  local -a contract_args=()
  if [[ "${INHERIT_SOURCE_CONTRACTS}" == 1 ]]; then
    contract_args+=(--inherit-source-contracts)
  fi
  mkdir -p "${SUITE_DIR}"/{commands,records,service_logs,shared_evaluator,analysis,autopilot}
  for lane in 0 1 2 3; do
    method="${METHODS[${lane}]}"
    manifest="$(manifest_for_lane "${lane}")"
    MPLCONFIGDIR=/tmp/hkbz-mpl "${PYTHON}" "${PREPARE}" \
      --profile "${PROFILE}" --method "${method}" \
      --base-credit-method "${BASE_CREDIT_METHOD}" \
      --max-graphs-per-forward "${GRAPHS_PER_FORWARD}" \
      --source-checkpoint "${SOURCE_CHECKPOINT}" --python "${PYTHON}" \
      --experiment-name "$(experiment_name_for_lane "${lane}")" \
      --seed "${SEED}" "${contract_args[@]}" --output "${manifest}"
    PYTHON="${PYTHON}" "${PYTHON}" "${MANIFEST_CHECKER}" \
      "${manifest}" --check-only
  done
}

require_fresh_outputs() {
  local lane unit state path
  for lane in 0 1 2 3; do
    selected_lane "${lane}" || continue
    path="$(experiment_dir_for_lane "${lane}")"
    [[ ! -e "${path}" ]] || {
      echo "[Error] Refusing output reuse: ${path}" >&2
      exit 1
    }
    unit="$(unit_name "${METHODS[${lane}]}")"
    state="$(systemctl --user show "${unit}.service" -p LoadState --value 2>/dev/null || true)"
    [[ -z "${state}" || "${state}" == not-found ]] || {
      echo "[Error] Existing unit ${unit}.service." >&2
      exit 1
    }
  done
}

launch_evaluator() {
  local unit socket log
  unit="$(unit_name eval)"
  socket="/tmp/${unit}.sock"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  [[ ! -e "${socket}" ]] || {
    echo "[Error] Existing evaluator socket: ${socket}" >&2
    exit 1
  }
  systemd-run --user --collect --unit="${unit}" --same-dir \
    --setenv=CUDA_VISIBLE_DEVICES=${GPU} --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONHASHSEED=0 --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=MALLOC_ARENA_MAX=2 --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=TimeoutStopSec=180 --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${VAL_CPUS}" --property="CPUAffinity=${VAL_CPUS}" \
    --property=NUMAPolicy=preferred --property="NUMAMask=${NUMA_NODE}" \
    --property=MemoryHigh=32G --property=MemoryMax=48G \
    --property="StandardOutput=append:${log}" --property="StandardError=append:${log}" \
    "${PYTHON}" -u "${EVALUATOR}" \
      --source-command-json "$(manifest_for_lane 0)" \
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
    [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" == active ]] || {
      echo "[Error] Shared evaluator exited: ${unit}.service" >&2
      return 1
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
    systemd-run --user --collect --unit="${unit}" --same-dir \
      --setenv=CUDA_VISIBLE_DEVICES=${GPU} --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
      --setenv=PYTHONHASHSEED=0 --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
      --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
      --setenv=MALLOC_ARENA_MAX=2 --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
      --property=Type=exec --property=Restart=no --property=KillMode=control-group \
      --property=TimeoutStopSec=180 --property=LimitNOFILE=65536 \
      --property="AllowedCPUs=${cpu_set}" --property="CPUAffinity=${cpu_set}" \
      --property=NUMAPolicy=preferred --property="NUMAMask=${NUMA_NODE}" \
      --property=MemoryHigh=72G --property=MemoryMax=88G \
      --property="StandardOutput=append:${log}" --property="StandardError=append:${log}" \
      "${PYTHON}" -u "${TRIAL_RUNNER}" \
        --manifest "$(manifest_for_lane "${lane}")" --gpu ${GPU} --cpu-set "${cpu_set}" \
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

launch_autopilot() {
  local unit log next_tag next_prefix
  local -a command
  unit="${UNIT_PREFIX}-${PROFILE}-autopilot"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  next_tag="stage3_ppo_gain_wave2_20260826_r1"
  next_prefix="hkbz-s3gain-r1w2-g1"
  command=(
    "${PYTHON}" -u "${AUTOPILOT}"
    --suite "${SUITE_DIR}" --results-root "${RESULTS_ROOT}"
    --results-log-root "${RESULTS_LOG_ROOT}" --run-tag "${RUN_TAG}"
    --profile "${PROFILE}" --unit-prefix "${UNIT_PREFIX}"
  )
  if [[ "${PROFILE}" == ppo_gain_wave1 && "${AUTO_NEXT_WAVE}" == 1 ]]; then
    command+=(
      --auto-launch-next-wave --launcher "${LAUNCHER}"
      --next-run-tag "${next_tag}" --next-unit-prefix "${next_prefix}"
    )
  fi
  systemd-run --user --collect --unit="${unit}" --same-dir \
    --setenv=PYTHONHASHSEED=0 --setenv=OMP_NUM_THREADS=1 \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=MemoryHigh=2G --property=MemoryMax=4G \
    --property="StandardOutput=append:${log}" --property="StandardError=append:${log}" \
    "${command[@]}" >/dev/null
  echo "[Launch] ${unit}.service will apply strict own-Pre selection."
}

main() {
  local socket
  [[ "${SEED}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[Error] SEED must be a positive integer, got ${SEED}." >&2
    exit 1
  }
  [[ "${METHOD_FILTER}" == all || " ${METHODS[*]} " == *" ${METHOD_FILTER} "* ]] || {
    echo "[Error] METHOD_FILTER is not valid for ${PROFILE}." >&2
    exit 1
  }
  [[ "${AUTO_ANALYZE}" =~ ^[01]$ && "${AUTO_NEXT_WAVE}" =~ ^[01]$ ]] || {
    echo "[Error] AUTO_ANALYZE/AUTO_NEXT_WAVE must be 0 or 1." >&2
    exit 1
  }
  if [[ "${AUTO_ANALYZE}" == 1 && "${METHOD_FILTER}" != all ]]; then
    echo "[Error] Automatic selection requires all four arms." >&2
    exit 1
  fi
  prepare_manifests
  if [[ "${START}" != 1 ]]; then
    echo "[Ready] Prepared ${PROFILE} manifests in ${SUITE_DIR}/commands"
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
    systemctl --user stop "$(unit_name eval).service" >/dev/null 2>&1 || true
  elif [[ "${AUTO_ANALYZE}" == 1 && "${PROFILE}" != ppo_gain_memory_canary && "${PROFILE}" != credit_happo_memory_canary && "${PROFILE}" != gradient_conflict_memory_canary && "${PROFILE}" != gradient_conflict_wave1 ]]; then
    launch_autopilot
  fi
  trap - ERR INT TERM
  echo "[Started] ${PROFILE} GPU${GPU} seed=${SEED} graph=${GRAPHS_PER_FORWARD} methods=${METHODS[*]} shared_eval=${socket}"
}

main "$@"
