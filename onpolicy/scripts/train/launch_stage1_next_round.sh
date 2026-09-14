#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python3.11}"
CONTROLLER="${ROOT_DIR}/onpolicy/scripts/train/run_stage1_next_round_controller.py"
RUN_TAG="${RUN_TAG:-stage1_next_round_20260808_r1}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-nr-0808r1}"
SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}"
CONTROLLER_UNIT="${UNIT_PREFIX}-controller"
CONTROLLER_LOG="${SUITE_DIR}/service_logs/${CONTROLLER_UNIT}.log"

SOURCE_JSON="${ROOT_DIR}/result/hkbz_train_logs/stage1_tail_robustness_v2_20260803_r1_extension_n1_n2_20260805_r1/commands/formal_N2_safe_tail_dagger_seed3.json"
INITIAL_CHECKPOINT="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage1_tail_robustness_v2_20260803_r1_screen_N2_safe_tail_dagger_seed1/run2/models/checkpoint_PlaneBC.pt"
POTENTIAL_PATH="${ROOT_DIR}/result/hkbz_train_logs/stage1_suite_tail_recovery_stage1_20260730_r1/iga_trajectory_analysis.json"
VALIDATION_DIR="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/validation"
TEST_DIR="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/test"
IGA_180_JSON="${ROOT_DIR}/result/hkbz_train_logs/stage1_tail_robustness_v2_20260803_r1_extension_n1_n2_20260805_r1/formal/IGA_180_evaluation.json"
IGA_1800_JSON="${ROOT_DIR}/result/hkbz_train_logs/stage1_tail_robustness_v2_20260803_r1_extension_n1_n2_20260805_r1/formal/IGA_1800_evaluation.json"

for required in \
  "${PYTHON}" "${CONTROLLER}" "${SOURCE_JSON}" "${INITIAL_CHECKPOINT}" \
  "${POTENTIAL_PATH}" "${VALIDATION_DIR}" "${TEST_DIR}" \
  "${IGA_180_JSON}" "${IGA_1800_JSON}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done

"${PYTHON}" - <<'PY'
from pathlib import Path

slices = (
    '0-11,72-83', '12-23,84-95', '24-35,96-107',
    '36-47,108-119', '48-59,120-131', '60-71,132-143',
)

def expand(spec):
    values = set()
    for part in spec.split(','):
        ends = [int(value) for value in part.split('-', 1)]
        values.update(range(ends[0], ends[-1] + 1))
    return values

sets = [expand(spec) for spec in slices]
if any(len(values) != 24 for values in sets):
    raise SystemExit('Each trainer must own exactly 24 logical CPUs.')
if any(sets[i] & sets[j] for i in range(6) for j in range(i + 1, 6)):
    raise SystemExit('Trainer CPU sets overlap.')
if set.union(*sets) != set(range(144)):
    raise SystemExit('The six trainer slices must exactly cover CPUs 0-143.')
for values in sets:
    physical = {cpu if cpu < 72 else cpu - 72 for cpu in values}
    if len(physical) != 12:
        raise SystemExit('Each slice must own both SMT siblings of 12 cores.')
    for core in physical:
        topology = Path(
            f'/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list'
        ).read_text(encoding='utf-8').strip()
        if topology != f'{core},{core + 72}':
            raise SystemExit(
                f'Unexpected SMT topology for core {core}: {topology}'
            )
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DryRun] topology and required inputs validated"
  echo "[DryRun] run_tag=${RUN_TAG} unit=${CONTROLLER_UNIT}"
  exit 0
fi

if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "[Error] User systemd manager is unavailable." >&2
  exit 1
fi
if [[ -e "${SUITE_DIR}" ]]; then
  echo "[Error] Refusing to reuse suite directory ${SUITE_DIR}." >&2
  exit 1
fi
state="$(systemctl --user show "${CONTROLLER_UNIT}.service" --property=LoadState --value 2>/dev/null || true)"
if [[ -n "${state}" && "${state}" != "not-found" ]]; then
  echo "[Error] Refusing to reuse ${CONTROLLER_UNIT}.service." >&2
  exit 1
fi
OLD_TAU_TRIAL_UNITS=(
  hkbz-tau-range-r1-fixed
  hkbz-tau-range-r1-mid
  hkbz-tau-range-r1-wide
)
OLD_TAU_EVALUATOR_UNIT=hkbz-tau-range-r1-eval-g0
handoff_args=()
old_tau_active=0
for unit in "${OLD_TAU_TRIAL_UNITS[@]}"; do
  if systemctl --user is-active --quiet "${unit}.service"; then
    old_tau_active=1
    handoff_args+=(--predecessor-trial-unit "${unit}")
  fi
done
if systemctl --user is-active --quiet "${OLD_TAU_EVALUATOR_UNIT}.service"; then
  old_tau_active=1
  handoff_args+=(--predecessor-evaluator-unit "${OLD_TAU_EVALUATOR_UNIT}")
fi

for gpu in 0 1; do
  if ! busy_pids="$(
    nvidia-smi --id="${gpu}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null
  )"; then
    echo "[Error] Cannot query GPU${gpu}; refusing an unverified launch." >&2
    exit 1
  fi
  if [[ -n "${busy_pids//[[:space:]]/}" ]]; then
    if [[ "${old_tau_active}" != "1" ]]; then
      echo "[Error] GPU${gpu} is busy; compute PIDs: ${busy_pids}" >&2
      exit 1
    fi
    echo "[Handoff] GPU${gpu} is occupied by predecessor work; new controller will wait."
  fi
done

mkdir -p "${SUITE_DIR}/service_logs"
systemd-run --user \
  --unit="${CONTROLLER_UNIT}" --collect --same-dir \
  --setenv=PYTHONHASHSEED=0 \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
  --property=Type=exec --property=Restart=no \
  --property=KillMode=control-group --property=TimeoutStopSec=300 \
  --property=LimitNOFILE=65536 \
  --property=AllowedCPUs=0-143 --property=CPUAffinity=0-143 \
  --property=CPUWeight=10 --property=Nice=10 \
  --property="StandardOutput=append:${CONTROLLER_LOG}" \
  --property="StandardError=append:${CONTROLLER_LOG}" \
  /usr/bin/taskset --cpu-list 0-143 \
  "${PYTHON}" -u "${CONTROLLER}" \
    --run-tag "${RUN_TAG}" \
    --unit-prefix "${UNIT_PREFIX}" \
    --suite-dir "${SUITE_DIR}" \
    --source-command-json "${SOURCE_JSON}" \
    --initial-checkpoint "${INITIAL_CHECKPOINT}" \
    --potential-path "${POTENTIAL_PATH}" \
    --validation-dir "${VALIDATION_DIR}" \
    --test-dir "${TEST_DIR}" \
    --iga-180-json "${IGA_180_JSON}" \
    --iga-1800-json "${IGA_1800_JSON}" \
    --screen-epochs 4 --formal-epochs 8 \
    --screen-sampling-size 480 --formal-sampling-size 0 \
    --partition-seed 20260803 --start-delay-step-seconds 30 \
    "${handoff_args[@]}"

systemctl --user show "${CONTROLLER_UNIT}.service" \
  --property=Id --property=MainPID --property=ActiveState --property=SubState \
  --property=AllowedCPUs --property=CPUAffinity
echo "[Launched] ${CONTROLLER_UNIT}.service"
echo "[Plan] ${SUITE_DIR}/experiment_plan.json"
echo "[Status] ${SUITE_DIR}/controller_status.json"
if [[ "${old_tau_active}" == "1" ]]; then
  echo "[Queued] Waiting for the existing tau trials to finish before resource handoff."
fi
