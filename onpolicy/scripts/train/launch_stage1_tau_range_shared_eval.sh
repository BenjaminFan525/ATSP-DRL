#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python3.11}"
RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage1_tau_range_trial.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"
RUN_TAG="${RUN_TAG:-stage1_tau_range_shared_eval_20260807_r1}"
SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-tau-range-r1}"
GPU="${GPU:-0}"
FIXED_SEED="${FIXED_SEED:-1}"
MID_DELAY="${MID_DELAY:-120}"
WIDE_DELAY="${WIDE_DELAY:-240}"

SOURCE_JSON="${ROOT_DIR}/result/hkbz_train_logs/stage1_tail_robustness_v2_20260803_r1_extension_n1_n2_20260805_r1/commands/formal_N1_tail_cv_potential_seed3.json"
INITIAL_CHECKPOINT="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/tail_recovery_stage1_20260730_r1_screen_R0_balanced_dagger_global_seed1/run1/models/checkpoint_PlaneBC.pt"
POTENTIAL_PATH="${ROOT_DIR}/result/hkbz_train_logs/stage1_tail_robustness_v2_20260803_r1/iga_tail_cv_potential.json"
VALIDATION_DIR="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/validation"

# GPU0/NUMA0 owns physical cores 0-35.  Each trial receives both SMT siblings
# of twelve whole cores; the three sets are disjoint and exactly cover NUMA0.
CPU_POOL="0-35,72-107"
CPU_FIXED="0-11,72-83"
CPU_MID="12-23,84-95"
CPU_WIDE="24-35,96-107"
SOCKET_PATH="${SOCKET_PATH:-/tmp/hkbz-tau-range-g0.sock}"

if [[ "${GPU}" != "0" ]]; then
  echo "[Error] This topology-checked launcher currently supports GPU=0 only." >&2
  exit 1
fi
if ! [[ "${FIXED_SEED}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[Error] FIXED_SEED must be a positive integer." >&2
  exit 1
fi
for delay in "${MID_DELAY}" "${WIDE_DELAY}"; do
  if ! [[ "${delay}" =~ ^[0-9]+$ ]]; then
    echo "[Error] Start delays must be non-negative integers." >&2
    exit 1
  fi
done
for required in \
  "${PYTHON}" "${RUNNER}" "${EVALUATOR}" "${SOURCE_JSON}" \
  "${INITIAL_CHECKPOINT}" "${POTENTIAL_PATH}" "${VALIDATION_DIR}" \
  /usr/bin/taskset; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done
if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "[Error] User systemd manager is unavailable." >&2
  exit 1
fi

"${PYTHON}" - "${CPU_POOL}" "${CPU_FIXED}" "${CPU_MID}" "${CPU_WIDE}" <<'PY'
import sys

def expand(spec):
    cpus = set()
    for part in spec.split(','):
        bounds = [int(value) for value in part.split('-', 1)]
        cpus.update(range(bounds[0], bounds[-1] + 1))
    return cpus

pool = expand(sys.argv[1])
slices = [expand(value) for value in sys.argv[2:]]
if len(pool) != 72 or any(len(cpu_set) != 24 for cpu_set in slices):
    raise SystemExit('Expected one 72-CPU pool and three 24-CPU slices.')
if set.union(*slices) != pool:
    raise SystemExit('The three trial slices must exactly cover the GPU-local pool.')
if any(slices[i] & slices[j] for i in range(3) for j in range(i + 1, 3)):
    raise SystemExit('Trial CPU slices overlap.')
for cpu_set in slices:
    physical = {cpu if cpu < 72 else cpu - 72 for cpu in cpu_set}
    if len(physical) != 12:
        raise SystemExit('Each trial must own both SMT threads of 12 cores.')
PY

UNITS=(
  "${UNIT_PREFIX}-eval-g0.service"
  "${UNIT_PREFIX}-fixed.service"
  "${UNIT_PREFIX}-mid.service"
  "${UNIT_PREFIX}-wide.service"
)
for unit in "${UNITS[@]}"; do
  state="$(systemctl --user show "${unit}" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${state}" && "${state}" != "not-found" ]]; then
    echo "[Error] Refusing to reuse existing unit ${unit}." >&2
    exit 1
  fi
done
if [[ -e "${SOCKET_PATH}" ]]; then
  echo "[Error] Refusing to replace existing socket ${SOCKET_PATH}." >&2
  exit 1
fi
busy_pids="$(
  nvidia-smi --id="${GPU}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true
)"
if [[ -n "${busy_pids//[[:space:]]/}" ]]; then
  echo "[Error] GPU${GPU} is busy; compute PIDs: ${busy_pids}" >&2
  exit 1
fi

mkdir -p \
  "${SUITE_DIR}/service_logs" \
  "${SUITE_DIR}/shared_evaluator/gpu0"

EVAL_UNIT="${UNIT_PREFIX}-eval-g0"
EVAL_LOG="${SUITE_DIR}/service_logs/${EVAL_UNIT}.log"
systemd-run --user \
  --unit="${EVAL_UNIT}" --collect --same-dir \
  --setenv="CUDA_VISIBLE_DEVICES=${GPU}" \
  --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=PYTHONHASHSEED=0 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
  --property=Type=exec --property=Restart=no \
  --property=KillMode=control-group --property=TimeoutStopSec=180 \
  --property=LimitNOFILE=65536 \
  --property="AllowedCPUs=${CPU_POOL}" \
  --property="CPUAffinity=${CPU_POOL}" \
  --property=CPUWeight=100 --property=Nice=0 \
  --property="StandardOutput=append:${EVAL_LOG}" \
  --property="StandardError=append:${EVAL_LOG}" \
  /usr/bin/taskset --cpu-list "${CPU_POOL}" \
  "${PYTHON}" -u "${EVALUATOR}" \
    --source-command-json "${SOURCE_JSON}" \
    --socket-path "${SOCKET_PATH}" \
    --cpu-pool "${CPU_POOL}" \
    --run-dir "${SUITE_DIR}/shared_evaluator/gpu0"
echo "[Launch] shared evaluator GPU${GPU} CPUs=${CPU_POOL}"

deadline=$((SECONDS + 300))
evaluator_ready=0
while (( SECONDS < deadline )); do
  if [[ "$(systemctl --user is-active "${EVAL_UNIT}.service" 2>/dev/null || true)" != "active" ]]; then
    echo "[Error] Evaluator exited before readiness: ${EVAL_UNIT}.service" >&2
    exit 1
  fi
  if [[ -S "${SOCKET_PATH}" ]] && "${PYTHON}" - "${SOCKET_PATH}" <<'PY'
import sys
from onpolicy.utils.shared_eval import SharedEvalClient

reply = SharedEvalClient(sys.argv[1], timeout_seconds=5.0).ping()
if reply.get('worker_count') != 60:
    raise SystemExit(f'Expected 60 shared evaluation workers, got {reply}')
PY
  then
    echo "[Ready] ${EVAL_UNIT}.service socket=${SOCKET_PATH} workers=60"
    evaluator_ready=1
    break
  fi
  sleep 2
done
if [[ "${evaluator_ready}" != "1" ]]; then
  echo "[Error] Timed out waiting for ${EVAL_UNIT}.service." >&2
  exit 1
fi

launch_trial() {
  local short_name="$1"
  local variant="$2"
  local cpu_set="$3"
  local delay="$4"
  local unit="${UNIT_PREFIX}-${short_name}"
  local log="${SUITE_DIR}/service_logs/${unit}.log"
  systemd-run --user \
    --unit="${unit}" --collect --same-dir \
    --setenv="CUDA_VISIBLE_DEVICES=${GPU}" \
    --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --property=Type=exec --property=Restart=no \
    --property=KillMode=control-group --property=TimeoutStopSec=180 \
    --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${cpu_set}" \
    --property="CPUAffinity=${cpu_set}" \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /usr/bin/taskset --cpu-list "${cpu_set}" \
    "${PYTHON}" -u "${RUNNER}" \
      --variant "${variant}" --seed "${FIXED_SEED}" \
      --gpu "${GPU}" --cpu-set "${cpu_set}" \
      --shared-eval-socket "${SOCKET_PATH}" \
      --start-delay-seconds "${delay}" \
      --run-tag "${RUN_TAG}" --suite-dir "${SUITE_DIR}" \
      --source-command-json "${SOURCE_JSON}" \
      --initial-checkpoint "${INITIAL_CHECKPOINT}" \
      --potential-path "${POTENTIAL_PATH}" \
      --validation-dir "${VALIDATION_DIR}" \
      --epochs 4 --train-sampling-size 480 --partition-seed 20260803
  echo "[Launch] ${variant} seed${FIXED_SEED} GPU${GPU} CPUs=${cpu_set} delay=${delay}s"
}

launch_trial fixed fixed_030 "${CPU_FIXED}" 0
launch_trial mid range_050_030 "${CPU_MID}" "${MID_DELAY}"
launch_trial wide range_080_030 "${CPU_WIDE}" "${WIDE_DELAY}"

systemctl --user show "${UNITS[@]}" \
  --property=Id --property=MainPID --property=ActiveState --property=SubState \
  --property=AllowedCPUs --property=CPUAffinity
echo "[Done] Three fixed-seed tau arms launched on GPU0; blind test remains sealed."
