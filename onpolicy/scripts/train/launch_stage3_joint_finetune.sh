#!/usr/bin/env bash
set -euo pipefail

# Stage3 is a strict Stage2 hand-off followed by pure joint PPO.  The default
# invocation only prepares and audits the immutable command; set START=1 to run.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python3.11}"
PROFILE="${PROFILE:-canary}"
START="${START:-0}"
GPU="${GPU:-0}"
CPU_SET="${CPU_SET:-0-143}"
MEMORY_HIGH="${MEMORY_HIGH:-220G}"
MEMORY_MAX="${MEMORY_MAX:-300G}"
RUN_TAG="${RUN_TAG:-stage3_joint_rl_20260823_r1}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage2_planning_wave4_20260822_r2_planning_wave_B2_wait_constraint_seed1/run1/models/checkpoint_Best.pt}"
PREPARE="${ROOT_DIR}/onpolicy/scripts/train/prepare_stage3_joint_finetune.py"
RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage3_joint_manifest.py"
MANIFEST="${SUITE_DIR}/commands/${PROFILE}.json"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RUN_TAG}_${PROFILE}_seed1}"
UNIT="${UNIT:-hkbz-s3-joint-${PROFILE}-g${GPU}}"
LOG="${SUITE_DIR}/service_logs/${UNIT}.log"

if [[ "${PROFILE}" != canary && "${PROFILE}" != formal ]]; then
  echo "[Error] PROFILE must be canary or formal." >&2
  exit 1
fi
for required in "${PYTHON}" "${SOURCE_CHECKPOINT}" "${PREPARE}" "${RUNNER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing Stage3 input: ${required}" >&2
    exit 1
  fi
done
mkdir -p "${SUITE_DIR}/commands" "${SUITE_DIR}/service_logs"

"${PYTHON}" "${PREPARE}" \
  --profile "${PROFILE}" \
  --source-checkpoint "${SOURCE_CHECKPOINT}" \
  --python "${PYTHON}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --output "${MANIFEST}"
"${PYTHON}" "${RUNNER}" --check-only "${MANIFEST}"

if [[ "${START}" != 1 ]]; then
  echo "[Ready] Stage3 ${PROFILE} manifest: ${MANIFEST}"
  echo "[Ready] Launch explicitly with START=1 $0"
  exit 0
fi
if [[ "${PROFILE}" == formal && "${ALLOW_FORMAL_WITHOUT_CANARY:-0}" != 1 ]]; then
  CANARY_STATUS="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/${RUN_TAG}_canary_seed1/run1/run_status.json"
  if [[ ! -s "${CANARY_STATUS}" ]]; then
    echo "[Error] Formal launch requires the completed canary status: ${CANARY_STATUS}" >&2
    exit 1
  fi
  "${PYTHON}" - "${CANARY_STATUS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("status") != "completed"
    or payload.get("training_stage") != "joint_finetune"
    or payload.get("phase") != "joint_finetune_completed"
):
    raise SystemExit(f"Stage3 canary did not complete its full contract: {path}")
PY
fi
if [[ -n "$(nvidia-smi --id="${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)" ]]; then
  echo "[Error] GPU${GPU} already has a compute process." >&2
  exit 1
fi
if [[ -e "${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/${EXPERIMENT_NAME}" ]]; then
  echo "[Error] Refusing to reuse experiment output: ${EXPERIMENT_NAME}" >&2
  exit 1
fi
UNIT_STATE="$(systemctl --user show "${UNIT}.service" -p LoadState --value 2>/dev/null || true)"
if [[ -n "${UNIT_STATE}" && "${UNIT_STATE}" != not-found ]]; then
  echo "[Error] Existing service name: ${UNIT}.service" >&2
  exit 1
fi

systemd-run --user --unit="${UNIT}" --same-dir \
  --setenv="PYTHON=${PYTHON}" \
  --setenv="CUDA_VISIBLE_DEVICES=${GPU}" \
  --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=PYTHONHASHSEED=0 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
  --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
  --property=Type=exec --property=Restart=no --property=KillMode=control-group \
  --property=TimeoutStopSec=180 --property=LimitNOFILE=65536 \
  --property="AllowedCPUs=${CPU_SET}" --property="CPUAffinity=${CPU_SET}" \
  --property=NUMAPolicy=interleave --property=NUMAMask=0,1 \
  --property="MemoryHigh=${MEMORY_HIGH}" --property="MemoryMax=${MEMORY_MAX}" \
  --property="StandardOutput=append:${LOG}" \
  --property="StandardError=append:${LOG}" \
  "${PYTHON}" -u "${RUNNER}" "${MANIFEST}" >/dev/null

echo "[Started] ${UNIT}.service GPU${GPU} log=${LOG}"
