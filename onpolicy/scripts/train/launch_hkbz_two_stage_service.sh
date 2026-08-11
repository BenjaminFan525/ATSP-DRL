#!/usr/bin/env bash
set -euo pipefail

# Canonical two-stage service entry point.  Stage 1 is an externally completed
# Stage-1 M2 artifact; this launcher only registers it.  Stage 2 always starts
# from that immutable M2 source and lets the Python controller perform the
# resource-BC warm-up followed by frozen resource_joint PPO.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python3.11}"
CONTROLLER="${ROOT_DIR}/onpolicy/scripts/train/run_hkbz_two_stage_pipeline.py"

RUN_TAG="${RUN_TAG:-hkbz_two_stage_$(date +%Y%m%d_%H%M%S)}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-2}"
DRY_RUN="${DRY_RUN:-0}"
STAGE1_M2_CHECKPOINT="${STAGE1_M2_CHECKPOINT:-${STAGE1_M2_PATH:-${M2_CHECKPOINT:-${STAGE1_SELECTION_CHECKPOINT:-}}}}"
STAGE1_HANDOFF="${STAGE1_HANDOFF:-${ROOT_DIR}/onpolicy/config/stage1_m2_handoff.json}"
MANIFEST_PATH="${MANIFEST_PATH:-${ROOT_DIR}/result/hkbz_train_logs/two_stage/${RUN_TAG}.json}"
ARTIFACT_DIR="${STAGE2_ARTIFACT_DIR:-${ARTIFACT_DIR:-}}"
SOURCE_COMMAND_JSON="${SOURCE_COMMAND_JSON:-${STAGE1_COMMAND_JSON:-}}"
SEED="${SEED:-1}"
BC_EPOCHS="${BC_EPOCHS:-2}"
PPO_EPOCHS="${PPO_EPOCHS:-8}"
PPO_EPOCH="${PPO_EPOCH:-3}"
BC_MIN_LABELS="${BC_MIN_LABELS:-64}"
BC_MIN_ROLLOUTS="${BC_MIN_ROLLOUTS:-1}"
BC_MAX_ROLLOUTS="${BC_MAX_ROLLOUTS:-20}"
BC_LR="${BC_LR:-0}"
PLANE_ORDER_MODE="${PLANE_ORDER_MODE:-fixed}"
PLANE_PAIR_DECODER="${PLANE_PAIR_DECODER:-joint_pair}"
GLOBAL_FEATURE_MODE="${GLOBAL_FEATURE_MODE:-f1f2}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${DRY_RUN,,}" in
  1|true|yes|on) DRY_RUN=1 ;;
  0|false|no|off|"") DRY_RUN=0 ;;
  *) echo "[Error] DRY_RUN must be 0/1 or a boolean value." >&2; exit 2 ;;
esac
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "[Error] RUN_TAG may contain only letters, digits, '.', '_', and '-'." >&2
  exit 2
fi
if [[ "${STOP_AFTER_STAGE}" != "1" && "${STOP_AFTER_STAGE}" != "2" ]]; then
  echo "[Error] STOP_AFTER_STAGE must be 1 or 2; Stage 3/4 are retired." >&2
  exit 2
fi
if [[ -n "${STAGE1_M2_CHECKPOINT}" && ! -f "${STAGE1_M2_CHECKPOINT}" ]]; then
  echo "[Error] Stage-1 M2 checkpoint is missing: ${STAGE1_M2_CHECKPOINT}" >&2
  exit 2
fi
if [[ -z "${STAGE1_M2_CHECKPOINT}" && ! -f "${STAGE1_HANDOFF}" ]]; then
  echo "[Error] Stage-1 hand-off is missing: ${STAGE1_HANDOFF}" >&2
  exit 2
fi
if [[ "${PYTHON}" == */* ]]; then
  if [[ ! -x "${PYTHON}" ]]; then
    echo "[Error] Python executable is missing or not executable: ${PYTHON}" >&2
    exit 2
  fi
elif ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "[Error] Python executable is not on PATH: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${CONTROLLER}" ]]; then
  echo "[Error] Controller is missing: ${CONTROLLER}" >&2
  exit 2
fi

COMMAND_ARGS=(
  --run-tag "${RUN_TAG}"
  --manifest "${MANIFEST_PATH}"
  --artifact-dir "${ARTIFACT_DIR}"
  --seed "${SEED}"
  --bc-epochs "${BC_EPOCHS}"
  --ppo-epochs "${PPO_EPOCHS}"
  --ppo-epoch "${PPO_EPOCH}"
  --bc-min-labels "${BC_MIN_LABELS}"
  --bc-min-rollouts "${BC_MIN_ROLLOUTS}"
  --bc-max-rollouts "${BC_MAX_ROLLOUTS}"
  --bc-lr "${BC_LR}"
)
if [[ -n "${STAGE1_M2_CHECKPOINT}" ]]; then
  COMMAND_ARGS+=(--source-m2 "${STAGE1_M2_CHECKPOINT}")
else
  COMMAND_ARGS+=(--stage1-handoff "${STAGE1_HANDOFF}" --source-seed "${SEED}")
fi
if [[ -n "${SOURCE_COMMAND_JSON}" ]]; then
  COMMAND_ARGS+=(--source-command-json "${SOURCE_COMMAND_JSON}")
fi
if [[ -n "${PLANE_ORDER_MODE}" ]]; then
  COMMAND_ARGS+=(--plane-order-mode "${PLANE_ORDER_MODE}")
fi
if [[ -n "${PLANE_PAIR_DECODER}" ]]; then
  COMMAND_ARGS+=(--plane-pair-decoder "${PLANE_PAIR_DECODER}")
fi
if [[ -n "${GLOBAL_FEATURE_MODE}" ]]; then
  COMMAND_ARGS+=(--global-feature-mode "${GLOBAL_FEATURE_MODE}")
fi

if [[ "${STOP_AFTER_STAGE}" == "1" ]]; then
  CONTROLLER_ACTION=(register)
else
  CONTROLLER_ACTION=(run)
fi

# Dry-run is deliberately resolved before every systemd check and call.  It
# registers/audits the exact command through the controller but cannot create
# a service, spawn training, or query a systemd manager.
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DryRun] Auditing the two-stage command; no service will be created."
  if [[ "${STOP_AFTER_STAGE}" == "2" ]]; then
    COMMAND_ARGS+=(--dry-run)
  fi
  exec "${PYTHON}" -u "${CONTROLLER}" "${CONTROLLER_ACTION[@]}" "${COMMAND_ARGS[@]}"
fi

if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "[Error] User systemd manager is unavailable; use DRY_RUN=1 for an audit." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
case "${CUDA_VISIBLE_DEVICES}" in
  0) DEFAULT_CPU_AFFINITY="0-35,72-107" ;;
  1) DEFAULT_CPU_AFFINITY="36-71,108-143" ;;
  *) DEFAULT_CPU_AFFINITY="" ;;
esac
CPU_AFFINITY="${CPU_AFFINITY:-${DEFAULT_CPU_AFFINITY}}"
if [[ -z "${CPU_AFFINITY}" ]]; then
  echo "[Error] CUDA_VISIBLE_DEVICES must select physical GPU 0 or 1." >&2
  exit 2
fi
if ! /usr/bin/taskset --cpu-list "${CPU_AFFINITY}" /bin/true; then
  echo "[Error] Invalid or unavailable CPU_AFFINITY: ${CPU_AFFINITY}" >&2
  exit 2
fi

UNIT_NAME="hkbz-${RUN_TAG}"
SERVICE_LOG_DIR="${ROOT_DIR}/result/hkbz_train_logs/two_stage/service_logs"
SERVICE_LOG="${SERVICE_LOG_DIR}/${RUN_TAG}.service.log"
UNIT_LOAD_STATE="$(systemctl --user show "${UNIT_NAME}.service" --property=LoadState --value 2>/dev/null || true)"
if [[ -n "${UNIT_LOAD_STATE}" && "${UNIT_LOAD_STATE}" != "not-found" ]]; then
  echo "[Error] Refusing to reuse existing service unit ${UNIT_NAME}.service." >&2
  exit 1
fi
mkdir -p "${SERVICE_LOG_DIR}"

echo "[Info] Launching ${UNIT_NAME}.service"
echo "[Info] RUN_TAG=${RUN_TAG}, STOP_AFTER_STAGE=${STOP_AFTER_STAGE}, GPU=${CUDA_VISIBLE_DEVICES}"
if [[ -n "${STAGE1_M2_CHECKPOINT}" ]]; then
  echo "[Info] Stage-1 M2=${STAGE1_M2_CHECKPOINT}"
else
  echo "[Info] Stage-1 hand-off=${STAGE1_HANDOFF}, source seed=${SEED}"
fi
echo "[Info] Manifest=${MANIFEST_PATH}"

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
  --setenv="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
  --setenv="CUDA_DEVICE_ORDER=PCI_BUS_ID" \
  --setenv="OMP_NUM_THREADS=1" \
  --setenv="MKL_NUM_THREADS=1" \
  --setenv="OPENBLAS_NUM_THREADS=1" \
  --setenv="NUMEXPR_NUM_THREADS=1" \
  --setenv="PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" \
  --setenv="STAGE1_M2_CHECKPOINT=${STAGE1_M2_CHECKPOINT}" \
  --setenv="STAGE1_HANDOFF=${STAGE1_HANDOFF}" \
  --setenv="MANIFEST_PATH=${MANIFEST_PATH}" \
  --setenv="PYTHONHASHSEED=0" \
  /usr/bin/taskset --cpu-list "${CPU_AFFINITY}" \
  "${PYTHON}" -u "${CONTROLLER}" "${CONTROLLER_ACTION[@]}" "${COMMAND_ARGS[@]}"

systemctl --user show "${UNIT_NAME}.service" \
  --property=MainPID \
  --property=ActiveState \
  --property=SubState \
  --property=ExecMainStatus
echo "RUN_TAG=${RUN_TAG}"
echo "UNIT=${UNIT_NAME}.service"
echo "MANIFEST=${MANIFEST_PATH}"
