#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/fanyx/HKBZ-environment"
PYTHON="/home/fanyx/anaconda3/envs/maia/bin/python"
SUITE="${ROOT_DIR}/onpolicy/scripts/train/run_stage1_research_suite.py"
TRAIN_DATA="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train"
TEACHER_ROOT="${ROOT_DIR}/result/hkbz_train_logs/iga_teachers"
TEACHER_1800="${TEACHER_ROOT}/fjspv3_t600_train_p20_g20_t1800_a3_s1_verified_v1"
SERVICE_LOG_ROOT="${ROOT_DIR}/result/hkbz_train_logs/formal_runs"

RUN_TAG="${RUN_TAG:-tail_recovery_stage1_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
STUDY="${STUDY:-tail_recovery}"
SCREEN_EPOCHS="${SCREEN_EPOCHS:-2}"
FORMAL_EPOCHS="${FORMAL_EPOCHS:-8}"
FORMAL_SEEDS="${FORMAL_SEEDS:-1,2,3}"
TOP_CONFIGS="${TOP_CONFIGS:-1}"
ROLLOUT_THREADS="${ROLLOUT_THREADS:-60}"
EVAL_THREADS="${EVAL_THREADS:-60}"
PLANE_BC_EPOCHS="${PLANE_BC_EPOCHS:-4}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
TRAJECTORY_WORKERS="${TRAJECTORY_WORKERS:-32}"
TEACHER_WORKERS="${TEACHER_WORKERS:-48}"
POTENTIAL_RIDGE="${POTENTIAL_RIDGE:-0.001}"
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-1800}"
IPC_TIMEOUT_SECONDS="${IPC_TIMEOUT_SECONDS:-300}"
RESUME="${RESUME:-0}"
CPU_AFFINITY="${CPU_AFFINITY:-0-31,64-95}"
CPU_WEIGHT="${CPU_WEIGHT:-50}"
NICE_LEVEL="${NICE_LEVEL:-5}"
PLANE_BC_CHECKPOINT="${PLANE_BC_CHECKPOINT:-${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/global_dagger_stage1_20260729_r1_screen_G0_local_teacher_seed1/run1/models/checkpoint_PlaneBC.pt}"

run_suite() {
  cd "${ROOT_DIR}"

  suite_command=(
    "${PYTHON}" -u "${SUITE}"
    --run_tag "${RUN_TAG}"
    --study "${STUDY}"
    --plane_bc_checkpoint "${PLANE_BC_CHECKPOINT}"
    --gpu "${GPU}"
    --screen_epochs "${SCREEN_EPOCHS}"
    --formal_epochs "${FORMAL_EPOCHS}"
    --formal_seeds "${FORMAL_SEEDS}"
    --top_configs "${TOP_CONFIGS}"
    --rollout_threads "${ROLLOUT_THREADS}"
    --eval_threads "${EVAL_THREADS}"
    --teacher_workers "${TEACHER_WORKERS}"
    --teacher_time_budget 1800
    --teacher_pop_size 20
    --teacher_generations 20
    --teacher_max_attempts 3
    --teacher_dir "${TEACHER_1800}"
    --plane_bc_epochs "${PLANE_BC_EPOCHS}"
    --monitor_interval "${MONITOR_INTERVAL}"
    --stall_timeout_seconds "${STALL_TIMEOUT_SECONDS}"
    --ipc_timeout_seconds "${IPC_TIMEOUT_SECONDS}"
    --trajectory_workers "${TRAJECTORY_WORKERS}"
    --potential_ridge "${POTENTIAL_RIDGE}"
  )
  if [[ "${RESUME}" == "1" ]]; then
    suite_command+=(--resume)
  fi
  exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp \
    "${suite_command[@]}"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_suite
fi

if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "[Error] Invalid RUN_TAG: ${RUN_TAG}" >&2
  exit 1
fi
if [[ "${GPU}" != "0" ]]; then
  echo "[Error] The research suite is intentionally pinned to physical GPU 0." >&2
  exit 1
fi
if ! [[ "${CPU_WEIGHT}" =~ ^[0-9]+$ ]] \
   || (( CPU_WEIGHT < 1 || CPU_WEIGHT > 10000 )); then
  echo "[Error] CPU_WEIGHT must be an integer in [1, 10000]." >&2
  exit 1
fi
if ! [[ "${NICE_LEVEL}" =~ ^-?[0-9]+$ ]] \
   || (( NICE_LEVEL < 0 || NICE_LEVEL > 19 )); then
  echo "[Error] NICE_LEVEL must be an integer in [0, 19]." >&2
  exit 1
fi
if ! [[ "${TEACHER_WORKERS}" =~ ^[0-9]+$ ]] \
   || (( TEACHER_WORKERS < 1 )); then
  echo "[Error] TEACHER_WORKERS must be a positive integer." >&2
  exit 1
fi
if ! /usr/bin/taskset --cpu-list "${CPU_AFFINITY}" /bin/true; then
  echo "[Error] Invalid or unavailable CPU_AFFINITY: ${CPU_AFFINITY}" >&2
  exit 1
fi
AFFINITY_CPU_COUNT="$(
  /usr/bin/taskset --cpu-list "${CPU_AFFINITY}" \
    "${PYTHON}" -c 'import os; print(len(os.sched_getaffinity(0)))'
)"
if (( TEACHER_WORKERS > AFFINITY_CPU_COUNT )); then
  echo "[Error] TEACHER_WORKERS=${TEACHER_WORKERS} exceeds isolated CPUs=${AFFINITY_CPU_COUNT}." >&2
  exit 1
fi
if (( ROLLOUT_THREADS > AFFINITY_CPU_COUNT )); then
  echo "[Error] ROLLOUT_THREADS=${ROLLOUT_THREADS} exceeds isolated CPUs=${AFFINITY_CPU_COUNT}." >&2
  exit 1
fi
if (( EVAL_THREADS > AFFINITY_CPU_COUNT )); then
  echo "[Error] EVAL_THREADS=${EVAL_THREADS} exceeds isolated CPUs=${AFFINITY_CPU_COUNT}." >&2
  exit 1
fi
for required in "${PYTHON}" "${SUITE}" "${TRAIN_DATA}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done
if [[ "${STUDY}" == "ppo_recovery" && ! -s "${PLANE_BC_CHECKPOINT}" ]]; then
  echo "[Error] Missing PlaneBC initialization checkpoint: ${PLANE_BC_CHECKPOINT}" >&2
  exit 1
fi
if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "[Error] User systemd manager is unavailable." >&2
  exit 1
fi

mkdir -p "${SERVICE_LOG_ROOT}"
UNIT_NAME="hkbz-${RUN_TAG}"
SERVICE_LOG="${SERVICE_LOG_ROOT}/${RUN_TAG}.service.log"
LOAD_STATE="$(
  systemctl --user show "${UNIT_NAME}.service" \
    --property=LoadState --value 2>/dev/null || true
)"
if [[ -n "${LOAD_STATE}" && "${LOAD_STATE}" != "not-found" ]]; then
  echo "[Error] Refusing to reuse existing unit ${UNIT_NAME}.service." >&2
  exit 1
fi

systemd-run --user \
  --unit="${UNIT_NAME}" \
  --collect \
  --same-dir \
  --property=Type=exec \
  --property=Restart=no \
  --property=KillMode=control-group \
  --property=TimeoutStopSec=120 \
  --property=LimitNOFILE=65536 \
  --property="CPUWeight=${CPU_WEIGHT}" \
  --property="Nice=${NICE_LEVEL}" \
  --property="StandardOutput=append:${SERVICE_LOG}" \
  --property="StandardError=append:${SERVICE_LOG}" \
  --setenv="RUN_TAG=${RUN_TAG}" \
  --setenv="GPU=${GPU}" \
  --setenv="STUDY=${STUDY}" \
  --setenv="SCREEN_EPOCHS=${SCREEN_EPOCHS}" \
  --setenv="FORMAL_EPOCHS=${FORMAL_EPOCHS}" \
  --setenv="FORMAL_SEEDS=${FORMAL_SEEDS}" \
  --setenv="TOP_CONFIGS=${TOP_CONFIGS}" \
  --setenv="ROLLOUT_THREADS=${ROLLOUT_THREADS}" \
  --setenv="EVAL_THREADS=${EVAL_THREADS}" \
  --setenv="PLANE_BC_EPOCHS=${PLANE_BC_EPOCHS}" \
  --setenv="MONITOR_INTERVAL=${MONITOR_INTERVAL}" \
  --setenv="STALL_TIMEOUT_SECONDS=${STALL_TIMEOUT_SECONDS}" \
  --setenv="TRAJECTORY_WORKERS=${TRAJECTORY_WORKERS}" \
  --setenv="TEACHER_WORKERS=${TEACHER_WORKERS}" \
  --setenv="POTENTIAL_RIDGE=${POTENTIAL_RIDGE}" \
  --setenv="IPC_TIMEOUT_SECONDS=${IPC_TIMEOUT_SECONDS}" \
  --setenv="RESUME=${RESUME}" \
  --setenv="CPU_AFFINITY=${CPU_AFFINITY}" \
  --setenv="CPU_WEIGHT=${CPU_WEIGHT}" \
  --setenv="NICE_LEVEL=${NICE_LEVEL}" \
  --setenv="PLANE_BC_CHECKPOINT=${PLANE_BC_CHECKPOINT}" \
  --setenv="PYTHONHASHSEED=0" \
  /usr/bin/taskset --cpu-list "${CPU_AFFINITY}" \
  /bin/bash "${ROOT_DIR}/onpolicy/scripts/train/launch_stage1_iga_weekly_suite.sh" --worker

systemctl --user show "${UNIT_NAME}.service" \
  --property=MainPID \
  --property=ActiveState \
  --property=SubState \
  --property=ExecMainStatus

echo "RUN_TAG=${RUN_TAG}"
echo "UNIT=${UNIT_NAME}.service"
echo "LOG=${SERVICE_LOG}"
echo "TEACHER=${TEACHER_1800}"
echo "STUDY=${STUDY}"
echo "PLANE_BC_CHECKPOINT=${PLANE_BC_CHECKPOINT}"
echo "CPU_AFFINITY=${CPU_AFFINITY}"
echo "AFFINITY_CPU_COUNT=${AFFINITY_CPU_COUNT}"
echo "TEACHER_WORKERS=${TEACHER_WORKERS}"
echo "CPU_WEIGHT=${CPU_WEIGHT}"
echo "NICE_LEVEL=${NICE_LEVEL}"
