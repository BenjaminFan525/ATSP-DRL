#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python3.11}"
PREPARE="${ROOT_DIR}/onpolicy/scripts/train/prepare_stage1_departure_research.py"
UPGRADE="${ROOT_DIR}/onpolicy/scripts/train/upgrade_stage1_checkpoint_contract.py"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage1_departure_manifest_trial.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"

WAVE="${WAVE:-1}"
if [[ "${WAVE}" != "1" && "${WAVE}" != "2" ]]; then
  echo "[Error] WAVE must be 1 or 2." >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-stage1_departure_reward_screen_20260813_r1}"
SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-dep-ppo-r2-w${WAVE}}"
SOURCE_COMMAND="${ROOT_DIR}/result/hkbz_train_logs/stage1_next_round_20260808_r1/commands/formal_M2_bc_kl_anneal_seed1.json"
DATASET_DIR="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train"
TEACHER_DIR="${ROOT_DIR}/result/hkbz_train_logs/iga_teachers/fjspv3_t600_train_s1_progressive_departure_r014_20260812_v2_p20_g20_t1800_a5_s1"
POTENTIAL_PATH="${ROOT_DIR}/result/hkbz_train_logs/stage1_departure_research_20260812_r1/iga_potential_v2.json"
RAW_BC_CHECKPOINT="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage1_departure_research_20260812_r1_bc_screen_B0_legacy_dagger_warm_seed1/run1/models/checkpoint_PlaneBC.pt"
BC_CHECKPOINT="${SUITE_DIR}/artifacts/checkpoint_PlaneBC.pt"
VALIDATION_DIR="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/validation"

MANIFEST_SEED1="${SUITE_DIR}/commands/wave${WAVE}_seed1.json"
MANIFEST_SEED2="${SUITE_DIR}/commands/wave${WAVE}_seed2.json"
SOCKET_G0="${SOCKET_G0:-/tmp/hkbz-dep-ppo-r2-w${WAVE}-g0.sock}"
SOCKET_G1="${SOCKET_G1:-/tmp/hkbz-dep-ppo-r2-w${WAVE}-g1.sock}"

CPU_POOL_G0="0-35,72-107"
CPU_POOL_G1="36-71,108-143"
CPU_G0_L0="0-11,72-83"
CPU_G0_L1="12-23,84-95"
CPU_G0_L2="24-35,96-107"
CPU_G1_L0="36-47,108-119"
CPU_G1_L1="48-59,120-131"
CPU_G1_L2="60-71,132-143"

for required in \
  "${PYTHON}" "${PREPARE}" "${UPGRADE}" "${TRIAL_RUNNER}" \
  "${EVALUATOR}" "${SOURCE_COMMAND}" "${DATASET_DIR}" \
  "${TEACHER_DIR}" "${POTENTIAL_PATH}" "${RAW_BC_CHECKPOINT}" \
  "${VALIDATION_DIR}" /usr/bin/taskset; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done

"${PYTHON}" - \
  "${CPU_POOL_G0}" "${CPU_G0_L0}" "${CPU_G0_L1}" "${CPU_G0_L2}" \
  "${CPU_POOL_G1}" "${CPU_G1_L0}" "${CPU_G1_L1}" "${CPU_G1_L2}" <<'PY'
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
    if len(pool) != 72 or any(len(value) != 24 for value in slices):
        raise SystemExit('Expected 72 logical CPUs and three 24-CPU slices.')
    if set.union(*slices) != pool:
        raise SystemExit('Trainer slices must exactly cover the NUMA pool.')
    if any(slices[i] & slices[j] for i in range(3) for j in range(i + 1, 3)):
        raise SystemExit('Trainer CPU slices overlap.')
    for cpus in slices:
        physical = {cpu if cpu < 72 else cpu - 72 for cpu in cpus}
        if len(physical) != 12:
            raise SystemExit('Every trainer must own both SMT siblings of 12 cores.')
PY

UNITS=(
  "${UNIT_PREFIX}-eval-g0.service" "${UNIT_PREFIX}-eval-g1.service"
  "${UNIT_PREFIX}-g0-l0.service" "${UNIT_PREFIX}-g0-l1.service"
  "${UNIT_PREFIX}-g0-l2.service" "${UNIT_PREFIX}-g1-l0.service"
  "${UNIT_PREFIX}-g1-l1.service" "${UNIT_PREFIX}-g1-l2.service"
)
for unit in "${UNITS[@]}"; do
  state="$(systemctl --user show "${unit}" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${state}" && "${state}" != "not-found" ]]; then
    echo "[Error] Refusing to reuse existing unit ${unit}." >&2
    exit 1
  fi
done
for socket_path in "${SOCKET_G0}" "${SOCKET_G1}"; do
  if [[ -e "${socket_path}" ]]; then
    echo "[Error] Refusing to replace existing socket ${socket_path}." >&2
    exit 1
  fi
done
for gpu in 0 1; do
  busy_pids="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "${busy_pids//[[:space:]]/}" ]]; then
    echo "[Error] GPU${gpu} is busy; compute PIDs: ${busy_pids}" >&2
    exit 1
  fi
done

mkdir -p \
  "${SUITE_DIR}/artifacts" "${SUITE_DIR}/commands" \
  "${SUITE_DIR}/service_logs" "${SUITE_DIR}/records" \
  "${SUITE_DIR}/shared_evaluator/wave${WAVE}/gpu0" \
  "${SUITE_DIR}/shared_evaluator/wave${WAVE}/gpu1"

"${PYTHON}" "${UPGRADE}" \
  --source "${RAW_BC_CHECKPOINT}" --output "${BC_CHECKPOINT}" \
  --global-feature-mode f1f2 --plane-order-mode fixed \
  --plane-pair-decoder joint_pair --reuse-if-valid

prepare_manifest() {
  local seed="$1"
  local output="$2"
  "${PYTHON}" "${PREPARE}" \
    --phase ppo_screen --ppo-wave "${WAVE}" --run-tag "${RUN_TAG}" \
    --source-command-json "${SOURCE_COMMAND}" \
    --dataset-dir "${DATASET_DIR}" --teacher-dir "${TEACHER_DIR}" \
    --potential-path "${POTENTIAL_PATH}" \
    --bc-checkpoint "${BC_CHECKPOINT}" \
    --expected-cases 600 --screen-seed "${seed}" --screen-epochs 4 \
    --output "${output}"
}

prepare_manifest 1 "${MANIFEST_SEED1}"
prepare_manifest 2 "${MANIFEST_SEED2}"

"${PYTHON}" - "${MANIFEST_SEED1}" "${MANIFEST_SEED2}" "${WAVE}" <<'PY'
import json
import sys

expected = {
    '1': {'P0_action_cmax', 'P1_action_potential', 'P2_team_cmax'},
    '2': {
        'P3_team_time',
        'P4_team_time_potential_ramp',
        'P5_team_time_potential_fixed',
    },
}[sys.argv[3]]
for path in sys.argv[1:3]:
    payload = json.load(open(path, encoding='utf-8'))
    if set(payload.get('commands', {})) != expected:
        raise SystemExit(f'Manifest {path} does not contain exact wave methods.')
    modes = {
        item['argv'][item['argv'].index('--global_feature_mode') + 1]
        for item in payload['commands'].values()
    }
    if modes != {'f1f2'}:
        raise SystemExit(f'Manifest {path} mixes observation semantics: {modes}')
PY

launch_evaluator() {
  local gpu="$1"
  local cpu_pool="$2"
  local socket_path="$3"
  local manifest="$4"
  local unit="${UNIT_PREFIX}-eval-g${gpu}"
  local log="${SUITE_DIR}/service_logs/${unit}.log"
  systemd-run --user --unit="${unit}" --collect --same-dir \
    --setenv="CUDA_VISIBLE_DEVICES=${gpu}" --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
    --property=Type=exec --property=Restart=no \
    --property=KillMode=control-group --property=TimeoutStopSec=180 \
    --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${cpu_pool}" --property="CPUAffinity=${cpu_pool}" \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /usr/bin/taskset --cpu-list "${cpu_pool}" \
    "${PYTHON}" -u "${EVALUATOR}" \
      --source-command-json "${manifest}" \
      --socket-path "${socket_path}" --cpu-pool "${cpu_pool}" \
      --run-dir "${SUITE_DIR}/shared_evaluator/wave${WAVE}/gpu${gpu}" \
      --eval-dataset-dir "${VALIDATION_DIR}" --max-eval-cases 60 \
      --eval-partition-seed 20260803 --eval-partition-stratify-by profile
  echo "[Launch] shared evaluator GPU${gpu} CPUs=${cpu_pool}"
}

wait_evaluator() {
  local unit="$1"
  local socket_path="$2"
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" != "active" ]]; then
      echo "[Error] Evaluator exited before readiness: ${unit}.service" >&2
      exit 1
    fi
    if [[ -S "${socket_path}" ]] && "${PYTHON}" - "${socket_path}" <<'PY'
import sys
from onpolicy.utils.shared_eval import SharedEvalClient
reply = SharedEvalClient(sys.argv[1], timeout_seconds=5.0).ping()
if reply.get('worker_count') != 60:
    raise SystemExit(f'Expected 60 workers, got {reply!r}.')
PY
    then
      echo "[Ready] ${unit}.service socket=${socket_path} workers=60"
      return
    fi
    sleep 2
  done
  echo "[Error] Timed out waiting for ${unit}.service." >&2
  exit 1
}

launch_trial() {
  local gpu="$1"
  local lane="$2"
  local key="$3"
  local cpu_set="$4"
  local socket_path="$5"
  local manifest="$6"
  local delay="$7"
  local unit="${UNIT_PREFIX}-g${gpu}-l${lane}"
  local log="${SUITE_DIR}/service_logs/${unit}.log"
  systemd-run --user --unit="${unit}" --collect --same-dir \
    --setenv="CUDA_VISIBLE_DEVICES=${gpu}" --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=MPLCONFIGDIR=/tmp/hkbz-mpl \
    --property=Type=exec --property=Restart=no \
    --property=KillMode=control-group --property=TimeoutStopSec=180 \
    --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${cpu_set}" --property="CPUAffinity=${cpu_set}" \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /usr/bin/taskset --cpu-list "${cpu_set}" \
    "${PYTHON}" -u "${TRIAL_RUNNER}" \
      --manifest "${manifest}" --command-key "${key}" --gpu "${gpu}" \
      --cpu-set "${cpu_set}" --shared-eval-socket "${socket_path}" \
      --start-delay-seconds "${delay}" \
      --record "${SUITE_DIR}/records/w${WAVE}_g${gpu}_l${lane}.json"
  echo "[Launch] GPU${gpu} lane${lane} ${key} CPUs=${cpu_set} delay=${delay}s"
}

launch_evaluator 0 "${CPU_POOL_G0}" "${SOCKET_G0}" "${MANIFEST_SEED1}"
wait_evaluator "${UNIT_PREFIX}-eval-g0" "${SOCKET_G0}"
launch_evaluator 1 "${CPU_POOL_G1}" "${SOCKET_G1}" "${MANIFEST_SEED2}"
wait_evaluator "${UNIT_PREFIX}-eval-g1" "${SOCKET_G1}"

if [[ "${WAVE}" == "1" ]]; then
  launch_trial 0 0 P0_action_cmax "${CPU_G0_L0}" "${SOCKET_G0}" "${MANIFEST_SEED1}" 0
  launch_trial 0 1 P1_action_potential "${CPU_G0_L1}" "${SOCKET_G0}" "${MANIFEST_SEED2}" 60
  launch_trial 0 2 P2_team_cmax "${CPU_G0_L2}" "${SOCKET_G0}" "${MANIFEST_SEED1}" 120
  launch_trial 1 0 P0_action_cmax "${CPU_G1_L0}" "${SOCKET_G1}" "${MANIFEST_SEED2}" 30
  launch_trial 1 1 P1_action_potential "${CPU_G1_L1}" "${SOCKET_G1}" "${MANIFEST_SEED1}" 90
  launch_trial 1 2 P2_team_cmax "${CPU_G1_L2}" "${SOCKET_G1}" "${MANIFEST_SEED2}" 150
else
  launch_trial 0 0 P3_team_time "${CPU_G0_L0}" "${SOCKET_G0}" "${MANIFEST_SEED2}" 0
  launch_trial 0 1 P4_team_time_potential_ramp "${CPU_G0_L1}" "${SOCKET_G0}" "${MANIFEST_SEED1}" 60
  launch_trial 0 2 P5_team_time_potential_fixed "${CPU_G0_L2}" "${SOCKET_G0}" "${MANIFEST_SEED2}" 120
  launch_trial 1 0 P3_team_time "${CPU_G1_L0}" "${SOCKET_G1}" "${MANIFEST_SEED1}" 30
  launch_trial 1 1 P4_team_time_potential_ramp "${CPU_G1_L1}" "${SOCKET_G1}" "${MANIFEST_SEED2}" 90
  launch_trial 1 2 P5_team_time_potential_fixed "${CPU_G1_L2}" "${SOCKET_G1}" "${MANIFEST_SEED1}" 150
fi

systemctl --user show "${UNITS[@]}" \
  --property=Id --property=MainPID --property=ActiveState --property=SubState \
  --property=AllowedCPUs --property=CPUAffinity
echo "[Done] Wave ${WAVE}: six PPO trainers and two shared validation pools launched."
