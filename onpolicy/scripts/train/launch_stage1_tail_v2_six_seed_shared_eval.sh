#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python}"
CONTROLLER="${ROOT_DIR}/onpolicy/scripts/train/run_stage1_tail_robustness_v2.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"

PARENT_RUN_TAG="${PARENT_RUN_TAG:-stage1_tail_robustness_v2_20260803_r1}"
RUN_TAG="${RUN_TAG:-${PARENT_RUN_TAG}_extension_n1_n2_20260805_r1}"
PARENT_SUITE="${ROOT_DIR}/result/hkbz_train_logs/${PARENT_RUN_TAG}"
SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-tail-v2-shared6}"
FORMAL_EPOCHS="${FORMAL_EPOCHS:-8}"
REUSE_EVALUATORS="${REUSE_EVALUATORS:-0}"
TRAINER_SEEDS="${TRAINER_SEEDS:-1,2,3}"
SEED2_DELAY_GPU0="${SEED2_DELAY_GPU0:-120}"
SEED2_DELAY_GPU1="${SEED2_DELAY_GPU1:-150}"
SEED3_DELAY_GPU0="${SEED3_DELAY_GPU0:-240}"
SEED3_DELAY_GPU1="${SEED3_DELAY_GPU1:-270}"

SOURCE_SUITE="${ROOT_DIR}/result/hkbz_train_logs/stage1_suite_tail_recovery_stage1_20260730_r1"
SOURCE_COMMAND="${SOURCE_SUITE}/formal_R3_tail_team_ratio_seed1.command.json"
OLD_POTENTIAL="${SOURCE_SUITE}/iga_trajectory_analysis.json"
OLD_BC="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/tail_recovery_stage1_20260730_r1_screen_R0_balanced_dagger_global_seed1/run1/models/checkpoint_PlaneBC.pt"
TEACHER_DIR="${ROOT_DIR}/result/hkbz_train_logs/iga_teachers/fjspv3_t600_train_p20_g20_t1800_a3_s1_verified_v1"
EVAL_ROOT="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803"
VALIDATION_DIR="${EVAL_ROOT}/validation"
BLIND_TEST_DIR="${EVAL_ROOT}/test"
NEW_POTENTIAL="${PARENT_SUITE}/iga_tail_cv_potential.json"

CPU_POOL_GPU0="0-35,72-107"
CPU_POOL_GPU1="36-71,108-143"
CPU_N1_SEED1="0-11,72-83"
CPU_N1_SEED2="12-23,84-95"
CPU_N1_SEED3="24-35,96-107"
CPU_N2_SEED1="36-47,108-119"
CPU_N2_SEED2="48-59,120-131"
CPU_N2_SEED3="60-71,132-143"

# Keep the AF_UNIX path safely below Linux's 108-byte sockaddr limit.
SOCKET_GPU0="${SOCKET_GPU0:-/tmp/hkbz-v2-shared6-g0.sock}"
SOCKET_GPU1="${SOCKET_GPU1:-/tmp/hkbz-v2-shared6-g1.sock}"
SOURCE_N1="${SUITE_DIR}/commands/formal_N1_tail_cv_potential_seed1.json"
SOURCE_N2="${SUITE_DIR}/commands/formal_N2_safe_tail_dagger_seed1.json"

for required in \
  "${PYTHON}" "${CONTROLLER}" "${EVALUATOR}" "${SOURCE_COMMAND}" \
  "${OLD_POTENTIAL}" "${OLD_BC}" "${TEACHER_DIR}" \
  "${VALIDATION_DIR}" "${BLIND_TEST_DIR}" "${NEW_POTENTIAL}" \
  "${SUITE_DIR}" "${SOURCE_N1}" "${SOURCE_N2}" /usr/bin/taskset; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done
if ! [[ "${FORMAL_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[Error] FORMAL_EPOCHS must be a positive integer." >&2
  exit 1
fi
if [[ "${REUSE_EVALUATORS}" != "0" && "${REUSE_EVALUATORS}" != "1" ]]; then
  echo "[Error] REUSE_EVALUATORS must be 0 or 1." >&2
  exit 1
fi
if [[ "${TRAINER_SEEDS}" != "1,2,3" && "${TRAINER_SEEDS}" != "2,3" && "${TRAINER_SEEDS}" != "1" ]]; then
  echo "[Error] TRAINER_SEEDS must be 1,2,3, 2,3, or 1." >&2
  exit 1
fi
if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "[Error] User systemd manager is unavailable." >&2
  exit 1
fi

"${PYTHON}" - "${CPU_POOL_GPU0}" "${CPU_N1_SEED1}" "${CPU_N1_SEED2}" "${CPU_N1_SEED3}" \
  "${CPU_POOL_GPU1}" "${CPU_N2_SEED1}" "${CPU_N2_SEED2}" "${CPU_N2_SEED3}" <<'PY'
import sys

def expand(spec):
    cpus = set()
    for part in spec.split(','):
        bounds = [int(value) for value in part.split('-', 1)]
        cpus.update(range(bounds[0], bounds[-1] + 1))
    return cpus

for offset in (0, 4):
    pool = expand(sys.argv[1 + offset])
    slices = [expand(value) for value in sys.argv[2 + offset:5 + offset]]
    if len(pool) != 72 or any(len(cpu_set) != 24 for cpu_set in slices):
        raise SystemExit('Expected one 72-CPU pool and three 24-CPU slices.')
    if set.union(*slices) != pool:
        raise SystemExit('Three seed slices must exactly cover the GPU-local pool.')
    if any(slices[i] & slices[j] for i in range(3) for j in range(i + 1, 3)):
        raise SystemExit('Seed CPU slices overlap.')
    for cpu_set in slices:
        physical = {cpu if cpu < 72 else cpu - 72 for cpu in cpu_set}
        if len(physical) != 12:
            raise SystemExit('Each seed must own both SMT threads of 12 physical cores.')
PY

UNITS_TO_CREATE=()
if [[ "${REUSE_EVALUATORS}" == "0" ]]; then
  UNITS_TO_CREATE+=(
    "${UNIT_PREFIX}-eval-g0.service" "${UNIT_PREFIX}-eval-g1.service"
  )
fi
for seed in ${TRAINER_SEEDS//,/ }; do
  UNITS_TO_CREATE+=(
    "${UNIT_PREFIX}-n1-s${seed}.service"
    "${UNIT_PREFIX}-n2-s${seed}.service"
  )
done
for unit in "${UNITS_TO_CREATE[@]}"; do
  state="$(systemctl --user show "${unit}" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${state}" && "${state}" != "not-found" ]]; then
    echo "[Error] Refusing to reuse existing unit ${unit}." >&2
    exit 1
  fi
done
if [[ "${REUSE_EVALUATORS}" == "0" ]]; then
  for socket_path in "${SOCKET_GPU0}" "${SOCKET_GPU1}"; do
    if [[ -e "${socket_path}" ]]; then
      echo "[Error] Refusing to replace existing socket ${socket_path}." >&2
      exit 1
    fi
  done
  for gpu in 0 1; do
    busy_pids="$(
      nvidia-smi --id="${gpu}" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null || true
    )"
    if [[ -n "${busy_pids//[[:space:]]/}" ]]; then
      echo "[Error] GPU${gpu} is busy; compute PIDs: ${busy_pids}" >&2
      exit 1
    fi
  done
fi

mkdir -p "${SUITE_DIR}/service_logs" "${SUITE_DIR}/shared_evaluator"

launch_evaluator() {
  local gpu="$1"
  local cpu_pool="$2"
  local socket_path="$3"
  local source_json="$4"
  local unit="${UNIT_PREFIX}-eval-g${gpu}"
  local log="${SUITE_DIR}/service_logs/${unit}.log"
  systemd-run --user \
    --unit="${unit}" --collect --same-dir \
    --setenv="CUDA_VISIBLE_DEVICES=${gpu}" \
    --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --property=Type=exec --property=Restart=no \
    --property=KillMode=control-group --property=TimeoutStopSec=180 \
    --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${cpu_pool}" \
    --property="CPUAffinity=${cpu_pool}" \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /usr/bin/taskset --cpu-list "${cpu_pool}" \
    "${PYTHON}" -u "${EVALUATOR}" \
      --source-command-json "${source_json}" \
      --socket-path "${socket_path}" \
      --cpu-pool "${cpu_pool}" \
      --run-dir "${SUITE_DIR}/shared_evaluator/gpu${gpu}"
  echo "[Launch] evaluator GPU${gpu} CPUs=${cpu_pool} unit=${unit}.service"
}

wait_evaluator() {
  local unit="$1"
  local socket_path="$2"
  local deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    if [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" != "active" ]]; then
      echo "[Error] Evaluator unit exited before readiness: ${unit}.service" >&2
      exit 1
    fi
    if [[ -S "${socket_path}" ]] && "${PYTHON}" - "${socket_path}" <<'PY'
import sys
from onpolicy.utils.shared_eval import SharedEvalClient
reply = SharedEvalClient(sys.argv[1], timeout_seconds=5.0).ping()
if reply.get('worker_count') != 60:
    raise SystemExit(f'Expected 60 shared evaluation workers, got {reply}')
PY
    then
      echo "[Ready] ${unit}.service socket=${socket_path}"
      return
    fi
    sleep 2
  done
  echo "[Error] Timed out waiting for ${unit}.service." >&2
  exit 1
}

COMMON_ARGS=(
  --run_tag "${RUN_TAG}"
  --suite_dir "${SUITE_DIR}"
  --source_command_json "${SOURCE_COMMAND}"
  --old_bc_checkpoint "${OLD_BC}"
  --old_potential "${OLD_POTENTIAL}"
  --new_potential "${NEW_POTENTIAL}"
  --teacher_dir "${TEACHER_DIR}"
  --validation_dir "${VALIDATION_DIR}"
  --blind_test_dir "${BLIND_TEST_DIR}"
  --screen_epochs 2
  --formal_epochs "${FORMAL_EPOCHS}"
  --partition_seed 20260803
  --safe_pipeline_all_phases
)

launch_trainer() {
  local lane="$1"
  local gpu="$2"
  local seed="$3"
  local cpu_set="$4"
  local variant="$5"
  local socket_path="$6"
  local delay="$7"
  local short_variant="n$((lane + 1))"
  local status_id="${short_variant}-s${seed}"
  local unit="${UNIT_PREFIX}-${status_id}"
  local log="${SUITE_DIR}/service_logs/${unit}.log"
  systemd-run --user \
    --unit="${unit}" --collect --same-dir \
    --property=Type=exec --property=Restart=no \
    --property=KillMode=control-group --property=TimeoutStopSec=180 \
    --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${cpu_set}" \
    --property="CPUAffinity=${cpu_set}" \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /usr/bin/taskset --cpu-list "${cpu_set}" \
    "${PYTHON}" -u "${CONTROLLER}" \
      --lane "${lane}" --gpu "${gpu}" --status-id "${status_id}" \
      --formal-only-variant "${variant}" --formal-seeds "${seed}" \
      --shared-eval-socket "${socket_path}" \
      --shared-eval-cpu-set "${cpu_set}" \
      --start-delay-seconds "${delay}" \
      "${COMMON_ARGS[@]}"
  echo "[Launch] ${variant} seed${seed} GPU${gpu} CPUs=${cpu_set} delay=${delay}s unit=${unit}.service"
}

if [[ "${REUSE_EVALUATORS}" == "0" ]]; then
  launch_evaluator 0 "${CPU_POOL_GPU0}" "${SOCKET_GPU0}" "${SOURCE_N1}"
  wait_evaluator "${UNIT_PREFIX}-eval-g0" "${SOCKET_GPU0}"
  launch_evaluator 1 "${CPU_POOL_GPU1}" "${SOCKET_GPU1}" "${SOURCE_N2}"
  wait_evaluator "${UNIT_PREFIX}-eval-g1" "${SOCKET_GPU1}"
else
  for mapping in \
    "${UNIT_PREFIX}-eval-g0:${SOCKET_GPU0}" \
    "${UNIT_PREFIX}-eval-g1:${SOCKET_GPU1}"; do
    unit="${mapping%%:*}"
    socket_path="${mapping#*:}"
    if [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" != "active" || ! -S "${socket_path}" ]]; then
      echo "[Error] Reused evaluator is not active: ${unit}.service socket=${socket_path}" >&2
      exit 1
    fi
    echo "[Ready] reusing active ${unit}.service socket=${socket_path}"
  done
fi

if [[ ",${TRAINER_SEEDS}," == *",1,"* ]]; then
  launch_trainer 0 0 1 "${CPU_N1_SEED1}" N1_tail_cv_potential "${SOCKET_GPU0}" 0
  launch_trainer 1 1 1 "${CPU_N2_SEED1}" N2_safe_tail_dagger "${SOCKET_GPU1}" 30
fi
if [[ ",${TRAINER_SEEDS}," == *",2,"* ]]; then
  launch_trainer 0 0 2 "${CPU_N1_SEED2}" N1_tail_cv_potential "${SOCKET_GPU0}" "${SEED2_DELAY_GPU0}"
  launch_trainer 1 1 2 "${CPU_N2_SEED2}" N2_safe_tail_dagger "${SOCKET_GPU1}" "${SEED2_DELAY_GPU1}"
fi
if [[ ",${TRAINER_SEEDS}," == *",3,"* ]]; then
  launch_trainer 0 0 3 "${CPU_N1_SEED3}" N1_tail_cv_potential "${SOCKET_GPU0}" "${SEED3_DELAY_GPU0}"
  launch_trainer 1 1 3 "${CPU_N2_SEED3}" N2_safe_tail_dagger "${SOCKET_GPU1}" "${SEED3_DELAY_GPU1}"
fi

systemctl --user show "${UNITS_TO_CREATE[@]}" \
  --property=Id --property=MainPID --property=ActiveState --property=SubState
echo "[Done] Six isolated trainers and two shared validation pools launched."
