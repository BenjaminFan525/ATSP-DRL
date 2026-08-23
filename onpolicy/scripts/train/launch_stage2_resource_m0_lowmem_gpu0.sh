#!/usr/bin/env bash
set -euo pipefail

# Relaunch only the failed Stage2 M0 lane while M1/M2 and their shared
# evaluator remain alive.  The 12-env x 50-step PPO slice caps one forward at
# 600 graphs.  Actor accumulation of 3 preserves an 1800-graph effective
# batch; critic accumulation of 7 preserves the original 4200-graph batch.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python3.11}"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage2_resource_manifest_trial.py"
RUN_TAG="${RUN_TAG:-stage2_resource_wave1_20260819_r9}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
MANIFEST="${MANIFEST:-${SUITE_DIR}/commands/wave1_standard.json}"
CPU_POOL="${CPU_POOL:-0-143}"
EVAL_WORKERS="${EVAL_WORKERS:-60}"
EVAL_UNIT="${EVAL_UNIT:-hkbz-s2-w1-20260819-r9-wave1-standard-eval.service}"
EVAL_SOCKET="${EVAL_SOCKET:-/tmp/hkbz-s2-w1-20260819-r9-wave1-standard-eval.sock}"
RESTART_INDEX="${RESTART_INDEX:-2}"
UNIT="${UNIT:-hkbz-s2-w1-20260819-r9-wave1-m0-lowmem600-r${RESTART_INDEX}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-stage2_resource_wave1_20260819_r9_wave1_M0_heuristic_uniform_lowmem600_restart${RESTART_INDEX}_seed1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20480}"
LOG="${SUITE_DIR}/service_logs/${UNIT}.log"
RECORD="${SUITE_DIR}/records/wave1_m0_lowmem600_restart${RESTART_INDEX}.json"

for required in "${PYTHON}" "${TRIAL_RUNNER}" "${MANIFEST}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing M0 relaunch input: ${required}" >&2
    exit 1
  fi
done
if [[ "$(systemctl --user is-active "${EVAL_UNIT}" 2>/dev/null || true)" != "active" ]]; then
  echo "[Error] Shared evaluator is not active: ${EVAL_UNIT}" >&2
  exit 1
fi
if [[ ! -S "${EVAL_SOCKET}" ]]; then
  echo "[Error] Shared evaluator socket is missing: ${EVAL_SOCKET}" >&2
  exit 1
fi
if [[ "$(systemctl --user show "${UNIT}.service" --property=LoadState --value 2>/dev/null || true)" != "not-found" ]]; then
  echo "[Error] Refusing to reuse existing unit ${UNIT}.service." >&2
  exit 1
fi

"${PYTHON}" - "${EVAL_SOCKET}" "${EVAL_WORKERS}" <<'PY'
import sys
from onpolicy.utils.shared_eval import SharedEvalClient

reply = SharedEvalClient(sys.argv[1], timeout_seconds=5.0).ping()
if int(reply.get("worker_count", -1)) != int(sys.argv[2]):
    raise SystemExit(
        f"shared evaluator worker mismatch: {reply.get('worker_count')} != {sys.argv[2]}"
    )
PY

free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
  echo "[Error] GPU0 free memory ${free_mib:-unknown} MiB is below the ${MIN_FREE_MIB} MiB relaunch floor." >&2
  exit 1
fi

mkdir -p "${SUITE_DIR}/service_logs" "${SUITE_DIR}/records"
systemd-run --user --unit="${UNIT}" --same-dir \
  --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=PYTHONHASHSEED=0 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
  --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
  --property=Type=exec --property=Restart=no \
  --property=KillMode=control-group --property=TimeoutStopSec=180 \
  --property=LimitNOFILE=65536 \
  --property="AllowedCPUs=${CPU_POOL}" \
  --property="CPUAffinity=${CPU_POOL}" \
  --property=CPUWeight=100 --property=Nice=0 \
  --property="StandardOutput=append:${LOG}" \
  --property="StandardError=append:${LOG}" \
  "${PYTHON}" -u "${TRIAL_RUNNER}" \
    --manifest "${MANIFEST}" --command-key M0_heuristic_uniform --gpu 0 \
    --cpu-set "${CPU_POOL}" --shared-eval-socket "${EVAL_SOCKET}" \
    --eval-workers "${EVAL_WORKERS}" --start-delay-seconds 0 \
    --record "${RECORD}" --experiment-name "${EXPERIMENT_NAME}" \
    --mini-batch-size 12 --data-chunk-length 50 \
    --max-graphs-per-forward 600 \
    --grad-accumulation-steps 7 \
    --actor-grad-accumulation-steps 3 >/dev/null

echo "[Launch] ${UNIT}.service M0_heuristic_uniform GPU0 free_before=${free_mib}MiB"
echo "[Launch] PPO geometry: 12x50=600 graphs, actor_accum=3, critic_accum=7"
echo "[Launch] Log: ${LOG}"
systemctl --user show "${UNIT}.service" \
  --property=Id --property=MainPID --property=ActiveState \
  --property=SubState --property=AllowedCPUs --property=CPUAffinity
