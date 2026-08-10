#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PIPELINE="${ROOT_DIR}/result/hkbz_train_logs/run_four_stage_fjsp_v2.sh"
SERVICE_LOG_DIR="${ROOT_DIR}/result/hkbz_train_logs/formal_runs"

RUN_TAG="${RUN_TAG:-teamcmax_casebal_$(date +%Y%m%d_%H%M%S)}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
case "${CUDA_VISIBLE_DEVICES}" in
  0) DEFAULT_CPU_AFFINITY="0-35,72-107" ;;
  1) DEFAULT_CPU_AFFINITY="36-71,108-143" ;;
  *) DEFAULT_CPU_AFFINITY="" ;;
esac
CPU_AFFINITY="${CPU_AFFINITY:-${DEFAULT_CPU_AFFINITY}}"
SEED="${SEED:-1}"
STAGE1_RESUME_CHECKPOINT="${STAGE1_RESUME_CHECKPOINT:-}"
STAGE1_SELECTION_CHECKPOINT="${STAGE1_SELECTION_CHECKPOINT:-}"
STAGE1_RESET_OPTIMIZERS_ON_RESUME="${STAGE1_RESET_OPTIMIZERS_ON_RESUME:-0}"
PLANE_CANARY_INTERVAL_SHARDS="${PLANE_CANARY_INTERVAL_SHARDS:-2}"
PLANE_CANARY_MAX_REGRESSION="${PLANE_CANARY_MAX_REGRESSION:-0.02}"
PLANE_ACTOR_WARMUP_SHARDS="${PLANE_ACTOR_WARMUP_SHARDS:-1}"
PLANE_CANARY_STOP_ON_REGRESSION="${PLANE_CANARY_STOP_ON_REGRESSION:-1}"
PLANE_EPOCHS_OVERRIDE="${PLANE_EPOCHS_OVERRIDE:-}"
PLANE_ACTOR_GRAD_ACCUMULATION_STEPS="${PLANE_ACTOR_GRAD_ACCUMULATION_STEPS:-32}"
PLANE_TAU="${PLANE_TAU:-0.3}"
N_ROLLOUT_THREADS_OVERRIDE="${N_ROLLOUT_THREADS_OVERRIDE:-}"
N_EVAL_THREADS_OVERRIDE="${N_EVAL_THREADS_OVERRIDE:-}"
MAX_TRAIN_CASES_OVERRIDE="${MAX_TRAIN_CASES_OVERRIDE:-}"
MAX_EVAL_CASES_OVERRIDE="${MAX_EVAL_CASES_OVERRIDE:-}"
MINI_BATCH_SIZE_OVERRIDE="${MINI_BATCH_SIZE_OVERRIDE:-}"
DATA_CHUNK_LENGTH_OVERRIDE="${DATA_CHUNK_LENGTH_OVERRIDE:-}"
GRAD_ACCUMULATION_STEPS_OVERRIDE="${GRAD_ACCUMULATION_STEPS_OVERRIDE:-}"
MAX_GRAPHS_PER_FORWARD_OVERRIDE="${MAX_GRAPHS_PER_FORWARD_OVERRIDE:-}"

if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "[Error] RUN_TAG may contain only letters, digits, '.', '_', and '-'." >&2
  exit 1
fi
if [[ ! "${STOP_AFTER_STAGE}" =~ ^[1-4]$ ]]; then
  echo "[Error] STOP_AFTER_STAGE must be one of 1, 2, 3, or 4." >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES}" != "0" && "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  echo "[Error] CUDA_VISIBLE_DEVICES must select physical GPU 0 or 1." >&2
  exit 1
fi
if ! /usr/bin/taskset --cpu-list "${CPU_AFFINITY}" /bin/true; then
  echo "[Error] Invalid or unavailable CPU_AFFINITY: ${CPU_AFFINITY}" >&2
  exit 1
fi
if [[ ! -x "${PIPELINE}" ]]; then
  echo "[Error] Pipeline is missing or not executable: ${PIPELINE}" >&2
  exit 1
fi
if ! systemctl --user is-system-running >/dev/null; then
  echo "[Error] The user systemd manager is unavailable; refusing a fragile terminal-bound launch." >&2
  exit 1
fi

mkdir -p "${SERVICE_LOG_DIR}"
UNIT_NAME="hkbz-${RUN_TAG}"
SERVICE_LOG="${SERVICE_LOG_DIR}/${RUN_TAG}.service.log"

UNIT_LOAD_STATE="$(
  systemctl --user show "${UNIT_NAME}.service" --property=LoadState --value 2>/dev/null \
    || true
)"
if [[ -n "${UNIT_LOAD_STATE}" && "${UNIT_LOAD_STATE}" != "not-found" ]]; then
  echo "[Error] Refusing to reuse existing service unit ${UNIT_NAME}.service." >&2
  exit 1
fi

echo "[Info] Launching ${UNIT_NAME}.service"
echo "[Info] RUN_TAG=${RUN_TAG}, STOP_AFTER_STAGE=${STOP_AFTER_STAGE}, GPU=${CUDA_VISIBLE_DEVICES}, SEED=${SEED}"
echo "[Info] CPU_AFFINITY=${CPU_AFFINITY}"
if [[ -n "${STAGE1_RESUME_CHECKPOINT}" ]]; then
  echo "[Info] Stage 1 resume checkpoint: ${STAGE1_RESUME_CHECKPOINT}"
fi
if [[ -n "${STAGE1_SELECTION_CHECKPOINT}" ]]; then
  echo "[Info] Stage 1 selection checkpoint: ${STAGE1_SELECTION_CHECKPOINT}"
fi
if [[ -n "${PLANE_EPOCHS_OVERRIDE}" ]]; then
  echo "[Info] Stage 1 epoch override: ${PLANE_EPOCHS_OVERRIDE}"
fi
echo "[Info] Stage1 policy controls: actor_grad_accumulation=${PLANE_ACTOR_GRAD_ACCUMULATION_STEPS}, fixed_tau=${PLANE_TAU}."
echo "[Info] Runtime overrides: rollout_threads=${N_ROLLOUT_THREADS_OVERRIDE:-default}, eval_threads=${N_EVAL_THREADS_OVERRIDE:-default}, mini_batch=${MINI_BATCH_SIZE_OVERRIDE:-default}, chunk=${DATA_CHUNK_LENGTH_OVERRIDE:-default}, grad_accumulation=${GRAD_ACCUMULATION_STEPS_OVERRIDE:-default}, max_graphs=${MAX_GRAPHS_PER_FORWARD_OVERRIDE:-default}."
echo "[Info] Service log: ${SERVICE_LOG}"

systemd-run --user \
  --unit="${UNIT_NAME}" \
  --collect \
  --same-dir \
  --property=Type=exec \
  --property=Restart=no \
  --property=KillMode=control-group \
  --property=TimeoutStopSec=120 \
  --property=LimitNOFILE=65536 \
  --property="AllowedCPUs=${CPU_AFFINITY}" \
  --property="CPUAffinity=${CPU_AFFINITY}" \
  --property="StandardOutput=append:${SERVICE_LOG}" \
  --property="StandardError=append:${SERVICE_LOG}" \
  --setenv="RUN_TAG=${RUN_TAG}" \
  --setenv="STOP_AFTER_STAGE=${STOP_AFTER_STAGE}" \
  --setenv="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
  --setenv="CUDA_DEVICE_ORDER=PCI_BUS_ID" \
  --setenv="CPU_AFFINITY=${CPU_AFFINITY}" \
  --setenv="OMP_NUM_THREADS=1" \
  --setenv="MKL_NUM_THREADS=1" \
  --setenv="OPENBLAS_NUM_THREADS=1" \
  --setenv="NUMEXPR_NUM_THREADS=1" \
  --setenv="SEED=${SEED}" \
  --setenv="STAGE1_RESUME_CHECKPOINT=${STAGE1_RESUME_CHECKPOINT}" \
  --setenv="STAGE1_SELECTION_CHECKPOINT=${STAGE1_SELECTION_CHECKPOINT}" \
  --setenv="STAGE1_RESET_OPTIMIZERS_ON_RESUME=${STAGE1_RESET_OPTIMIZERS_ON_RESUME}" \
  --setenv="PLANE_CANARY_INTERVAL_SHARDS=${PLANE_CANARY_INTERVAL_SHARDS}" \
  --setenv="PLANE_CANARY_MAX_REGRESSION=${PLANE_CANARY_MAX_REGRESSION}" \
  --setenv="PLANE_ACTOR_WARMUP_SHARDS=${PLANE_ACTOR_WARMUP_SHARDS}" \
  --setenv="PLANE_CANARY_STOP_ON_REGRESSION=${PLANE_CANARY_STOP_ON_REGRESSION}" \
  --setenv="PLANE_EPOCHS_OVERRIDE=${PLANE_EPOCHS_OVERRIDE}" \
  --setenv="PLANE_ACTOR_GRAD_ACCUMULATION_STEPS=${PLANE_ACTOR_GRAD_ACCUMULATION_STEPS}" \
  --setenv="PLANE_TAU=${PLANE_TAU}" \
  --setenv="N_ROLLOUT_THREADS_OVERRIDE=${N_ROLLOUT_THREADS_OVERRIDE}" \
  --setenv="N_EVAL_THREADS_OVERRIDE=${N_EVAL_THREADS_OVERRIDE}" \
  --setenv="MAX_TRAIN_CASES_OVERRIDE=${MAX_TRAIN_CASES_OVERRIDE}" \
  --setenv="MAX_EVAL_CASES_OVERRIDE=${MAX_EVAL_CASES_OVERRIDE}" \
  --setenv="MINI_BATCH_SIZE_OVERRIDE=${MINI_BATCH_SIZE_OVERRIDE}" \
  --setenv="DATA_CHUNK_LENGTH_OVERRIDE=${DATA_CHUNK_LENGTH_OVERRIDE}" \
  --setenv="GRAD_ACCUMULATION_STEPS_OVERRIDE=${GRAD_ACCUMULATION_STEPS_OVERRIDE}" \
  --setenv="MAX_GRAPHS_PER_FORWARD_OVERRIDE=${MAX_GRAPHS_PER_FORWARD_OVERRIDE}" \
  --setenv="SMOKE_TEST=${SMOKE_TEST:-0}" \
  --setenv="PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  /usr/bin/taskset --cpu-list "${CPU_AFFINITY}" \
  /bin/bash "${PIPELINE}"

systemctl --user show "${UNIT_NAME}.service" \
  --property=MainPID \
  --property=ActiveState \
  --property=SubState \
  --property=ExecMainStatus

echo "RUN_TAG=${RUN_TAG}"
echo "UNIT=${UNIT_NAME}.service"
echo "LOG=${SERVICE_LOG}"
