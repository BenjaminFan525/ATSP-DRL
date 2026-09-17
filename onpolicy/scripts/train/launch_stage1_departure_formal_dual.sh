#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python3.11}"
PREPARE="${ROOT_DIR}/onpolicy/scripts/train/prepare_stage1_departure_research.py"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage1_departure_manifest_trial.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"
POSTFORMAL_IGA_CONTROLLER="${ROOT_DIR}/onpolicy/scripts/train/run_stage1_departure_postformal_iga.py"
IGA_LABEL_LAUNCHER="${ROOT_DIR}/onpolicy/scripts/train/launch_departure_iga_labels.sh"

RUN_TAG="${RUN_TAG:-stage1_departure_reward_formal_dual_20260816_r1}"
SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-dep-formal-p3p5-r1}"
SCHED_PROFILE="${HKBZ_SCHED_PROFILE:-hard_staggered}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

PRIMARY_VARIANT="P3_team_time"
CHALLENGER_VARIANT="P5_team_time_potential_fixed"
FORMAL_EPOCHS="${FORMAL_EPOCHS:-8}"

SOURCE_COMMAND="${ROOT_DIR}/result/hkbz_train_logs/stage1_next_round_20260808_r1/commands/formal_M2_bc_kl_anneal_seed1.json"
DATASET_DIR="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train"
TEACHER_DIR="${ROOT_DIR}/result/hkbz_train_logs/iga_teachers/fjspv3_t600_train_s1_progressive_departure_r014_20260812_v2_p20_g20_t1800_a5_s1"
POTENTIAL_PATH="${ROOT_DIR}/result/hkbz_train_logs/stage1_departure_research_20260812_r1/iga_potential_v2.json"
BC_CHECKPOINT="${ROOT_DIR}/result/hkbz_train_logs/stage1_departure_reward_screen_20260813_r1/artifacts/checkpoint_PlaneBC.pt"
VALIDATION_DIR="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/validation"
TEST60_DIR="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/test"
TEST60_IGA_LABEL_VERSION="${TEST60_IGA_LABEL_VERSION:-s1_progressive_departure_r014_test60_20260817_v1}"
TEST60_IGA_OUTPUT_ROOT="${TEST60_IGA_OUTPUT_ROOT:-${ROOT_DIR}/result/hkbz_train_logs/iga_labeling_${TEST60_IGA_LABEL_VERSION}}"
TEST60_IGA_TEACHER_ROOT="${TEST60_IGA_TEACHER_ROOT:-${ROOT_DIR}/result/hkbz_train_logs/iga_teachers}"
TEST60_IGA_UNIT="${TEST60_IGA_UNIT:-hkbz-s1-iga-test60-progressive-departure-20260817-v1}"
TEST60_IGA_WORKERS="${TEST60_IGA_WORKERS:-144}"
FULL_CPU_SET="0-143"

MANIFEST_G0="${SUITE_DIR}/commands/formal_P3.json"
MANIFEST_G1="${SUITE_DIR}/commands/formal_P5.json"
LAUNCH_CONTRACT="${SUITE_DIR}/launch_contract.json"
SOCKET_G0="${SOCKET_G0:-/tmp/${UNIT_PREFIX}-g0.sock}"
SOCKET_G1="${SOCKET_G1:-/tmp/${UNIT_PREFIX}-g1.sock}"
POSTFORMAL_IGA_UNIT="${UNIT_PREFIX}-postformal-iga"
POSTFORMAL_IGA_STATUS="${SUITE_DIR}/postformal_iga/status.json"

CPU_POOL_G0="0-35,72-107"
CPU_POOL_G1="36-71,108-143"
CPU_G0_L0="0-11,72-83"
CPU_G0_L1="12-23,84-95"
CPU_G0_L2="24-35,96-107"
CPU_G1_L0="36-47,108-119"
CPU_G1_L1="48-59,120-131"
CPU_G1_L2="60-71,132-143"

# One third of the measured shard period separates adjacent lanes.  This only
# changes wall-clock phase; every seed keeps the exact pre-registered training
# and validation contract.
DELAY_G0_L0=0
DELAY_G0_L1=480
DELAY_G0_L2=960
DELAY_G1_L0=30
DELAY_G1_L1=510
DELAY_G1_L2=990

if [[ "${SCHED_PROFILE}" != "hard_staggered" ]]; then
  echo "[Error] Formal currently permits only HKBZ_SCHED_PROFILE=hard_staggered." >&2
  exit 2
fi
if [[ "${PREFLIGHT_ONLY}" != "0" && "${PREFLIGHT_ONLY}" != "1" ]]; then
  echo "[Error] PREFLIGHT_ONLY must be 0 or 1." >&2
  exit 2
fi
if ! [[ "${FORMAL_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[Error] FORMAL_EPOCHS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${TEST60_IGA_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[Error] TEST60_IGA_WORKERS must be a positive integer." >&2
  exit 2
fi

for required in \
  "${PYTHON}" "${PREPARE}" "${TRIAL_RUNNER}" "${EVALUATOR}" \
  "${POSTFORMAL_IGA_CONTROLLER}" "${IGA_LABEL_LAUNCHER}" \
  "${SOURCE_COMMAND}" "${DATASET_DIR}" "${TEACHER_DIR}" \
  "${POTENTIAL_PATH}" "${BC_CHECKPOINT}" "${VALIDATION_DIR}" "${TEST60_DIR}" \
  /usr/bin/taskset; do
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
        raise SystemExit(
            'Expected one 72-logical-CPU NUMA pool and three 24-CPU slices.'
        )
    if set.union(*slices) != pool:
        raise SystemExit('Trainer slices must exactly cover their NUMA pool.')
    if any(
        slices[i] & slices[j]
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        raise SystemExit('Trainer CPU slices overlap.')
    for cpus in slices:
        physical = {cpu if cpu < 72 else cpu - 72 for cpu in cpus}
        if len(physical) != 12:
            raise SystemExit(
                'Every trainer must own both SMT siblings of 12 physical cores.'
            )
PY

mkdir -p \
  "${SUITE_DIR}/commands" "${SUITE_DIR}/service_logs" \
  "${SUITE_DIR}/records" "${SUITE_DIR}/shared_evaluator/gpu0" \
  "${SUITE_DIR}/shared_evaluator/gpu1" "${SUITE_DIR}/postformal_iga"

prepare_manifest() {
  local variant="$1"
  local output="$2"
  "${PYTHON}" "${PREPARE}" \
    --phase formal --formal-variant "${variant}" \
    --run-tag "${RUN_TAG}" \
    --source-command-json "${SOURCE_COMMAND}" \
    --dataset-dir "${DATASET_DIR}" --teacher-dir "${TEACHER_DIR}" \
    --potential-path "${POTENTIAL_PATH}" \
    --bc-checkpoint "${BC_CHECKPOINT}" --expected-cases 600 \
    --formal-epochs "${FORMAL_EPOCHS}" --output "${output}"
}

prepare_manifest "${PRIMARY_VARIANT}" "${MANIFEST_G0}"
prepare_manifest "${CHALLENGER_VARIANT}" "${MANIFEST_G1}"

"${PYTHON}" - \
  "${MANIFEST_G0}" "${MANIFEST_G1}" "${LAUNCH_CONTRACT}" \
  "${PRIMARY_VARIANT}" "${CHALLENGER_VARIANT}" "${FORMAL_EPOCHS}" \
  "${RUN_TAG}" "${UNIT_PREFIX}" "${SCHED_PROFILE}" \
  "${CPU_POOL_G0}" "${CPU_G0_L0}" "${CPU_G0_L1}" "${CPU_G0_L2}" \
  "${CPU_POOL_G1}" "${CPU_G1_L0}" "${CPU_G1_L1}" "${CPU_G1_L2}" \
  "${DELAY_G0_L0}" "${DELAY_G0_L1}" "${DELAY_G0_L2}" \
  "${DELAY_G1_L0}" "${DELAY_G1_L1}" "${DELAY_G1_L2}" \
  "${TEST60_DIR}" "${TEST60_IGA_LABEL_VERSION}" \
  "${TEST60_IGA_OUTPUT_ROOT}" "${POSTFORMAL_IGA_STATUS}" \
  "${TEST60_IGA_UNIT}" <<'PY'
import hashlib
import json
import os
import sys
import time
from pathlib import Path

(
    manifest_g0,
    manifest_g1,
    output,
    primary,
    challenger,
    formal_epochs,
    run_tag,
    unit_prefix,
    sched_profile,
    pool_g0,
    g0_l0,
    g0_l1,
    g0_l2,
    pool_g1,
    g1_l0,
    g1_l1,
    g1_l2,
    d_g0_l0,
    d_g0_l1,
    d_g0_l2,
    d_g1_l0,
    d_g1_l1,
    d_g1_l2,
    test60_dir,
    test60_iga_label_version,
    test60_iga_output_root,
    postformal_iga_status,
    test60_iga_unit,
) = sys.argv[1:]
formal_epochs = int(formal_epochs)


def option(argv, flag):
    if argv.count(flag) != 1:
        raise SystemExit(f'Expected exactly one {flag}: {argv.count(flag)}')
    return argv[argv.index(flag) + 1]


def audit(path, variant):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('phase') != 'formal':
        raise SystemExit(f'{path} is not a Formal manifest.')
    expected = {f'{variant}_seed{seed}' for seed in (1, 2, 3)}
    if set(payload.get('commands', {})) != expected:
        raise SystemExit(f'{path} command keys do not match {sorted(expected)}.')
    for seed in (1, 2, 3):
        argv = payload['commands'][f'{variant}_seed{seed}']['argv']
        checks = {
            '--seed': str(seed),
            '--num_episodes': str(formal_epochs),
            '--global_feature_mode': 'f1f2',
            '--selection_metric': 'composite',
            '--eval_interval': '1',
            '--early_stop_patience': '0',
            '--hindsight_terminal_cmax_coef': '1.0',
            '--gamma': '1.0',
            '--iga_potential_gamma': '1.0',
        }
        for flag, expected_value in checks.items():
            actual = option(argv, flag)
            if actual != expected_value:
                raise SystemExit(
                    f'{variant} seed{seed}: {flag}={actual}, '
                    f'expected {expected_value}.'
                )
        reward = option(argv, '--hindsight_reward_mode')
        expected_reward = (
            'team_time' if variant == 'P3_team_time'
            else 'team_time_potential'
        )
        if reward != expected_reward:
            raise SystemExit(
                f'{variant} seed{seed}: reward={reward}, '
                f'expected {expected_reward}.'
            )
        expected_beta = '0.0' if variant == 'P3_team_time' else '0.1'
        if option(argv, '--iga_potential_beta') != expected_beta:
            raise SystemExit(f'{variant} seed{seed}: potential beta mismatch.')
        if '--joint_team_ppo' not in argv:
            raise SystemExit(f'{variant} seed{seed}: joint-team PPO is disabled.')
        if '--reset_optimizers_on_resume' not in argv:
            raise SystemExit(f'{variant} seed{seed}: optimizer reset is disabled.')
        if '--reset_value_normalizer_on_resume' not in argv:
            raise SystemExit(
                f'{variant} seed{seed}: ValueNorm reset is disabled.'
            )
    return payload


payload_g0 = audit(manifest_g0, primary)
payload_g1 = audit(manifest_g1, challenger)
bc_g0 = payload_g0.get('bc_checkpoint_audit', {})
bc_g1 = payload_g1.get('bc_checkpoint_audit', {})
if not bc_g0.get('sha256') or bc_g0.get('sha256') != bc_g1.get('sha256'):
    raise SystemExit('P3 and P5 do not start from the exact same BC checkpoint.')

contract = {
    'schema_version': 1,
    'status': 'prepared',
    'created_unix_time': time.time(),
    'run_tag': run_tag,
    'unit_prefix': unit_prefix,
    'phase': 'formal_dual_confirmatory',
    'primary': {
        'variant': primary,
        'gpu': 0,
        'seeds': [1, 2, 3],
        'manifest': str(Path(manifest_g0).resolve()),
    },
    'challenger': {
        'variant': challenger,
        'gpu': 1,
        'seeds': [1, 2, 3],
        'manifest': str(Path(manifest_g1).resolve()),
        'interpretation': 'parallel challenger; P3 remains the screen-selected primary',
    },
    'formal_epochs': formal_epochs,
    'selection_metric': 'composite',
    'validation_interval_epochs': 1,
    'test_policy': 'evaluate test60 only after training; never tune from test60',
    'post_training_iga': {
        'trigger': 'all six Formal trials completed with epoch_8 validation',
        'dataset_dir': str(Path(test60_dir).resolve()),
        'expected_cases': 60,
        'environment_semantics_version': 'progressive-departure-r014-pipeline-v2',
        'budgets_seconds': [180, 1800],
        'execution_order': ['IGA-180', 'replay verification', 'IGA-1800'],
        'label_version': test60_iga_label_version,
        'output_root': str(Path(test60_iga_output_root).resolve()),
        'controller_status': str(Path(postformal_iga_status).resolve()),
        'systemd_unit': test60_iga_unit,
    },
    'bc_checkpoint_sha256': bc_g0['sha256'],
    'scheduler': {
        'profile': sched_profile,
        'shared_evaluators_per_gpu': 1,
        'trainers_per_gpu': 3,
        'gpu0_cpu_pool': pool_g0,
        'gpu0_cpu_slices': [g0_l0, g0_l1, g0_l2],
        'gpu1_cpu_pool': pool_g1,
        'gpu1_cpu_slices': [g1_l0, g1_l1, g1_l2],
        'gpu0_start_delays_seconds': [
            int(d_g0_l0), int(d_g0_l1), int(d_g0_l2)
        ],
        'gpu1_start_delays_seconds': [
            int(d_g1_l0), int(d_g1_l1), int(d_g1_l2)
        ],
        'omp_threads': 1,
        'allocator': 'expandable_segments:True',
    },
}
target = Path(output)
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_name(f'.{target.name}.tmp.{os.getpid()}')
temporary.write_text(
    json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
    encoding='utf-8',
)
os.replace(temporary, target)
print(
    '[FormalPreflight] exact P3/P5 contracts verified; '
    f'BC sha256={bc_g0["sha256"]}'
)
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "[Done] Preflight only; no systemd service was started."
  exit 0
fi

UNITS=(
  "${UNIT_PREFIX}-eval-g0.service" "${UNIT_PREFIX}-eval-g1.service"
  "${UNIT_PREFIX}-g0-s1.service" "${UNIT_PREFIX}-g0-s2.service"
  "${UNIT_PREFIX}-g0-s3.service" "${UNIT_PREFIX}-g1-s1.service"
  "${UNIT_PREFIX}-g1-s2.service" "${UNIT_PREFIX}-g1-s3.service"
  "${POSTFORMAL_IGA_UNIT}.service"
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
  busy_pids="$(
    nvidia-smi --id="${gpu}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true
  )"
  if [[ -n "${busy_pids//[[:space:]]/}" ]]; then
    echo "[Error] GPU${gpu} is busy; compute PIDs: ${busy_pids}" >&2
    exit 1
  fi
done

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
    --property="AllowedCPUs=${cpu_pool}" \
    --property="CPUAffinity=${cpu_pool}" \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /usr/bin/taskset --cpu-list "${cpu_pool}" \
    "${PYTHON}" -u "${EVALUATOR}" \
      --source-command-json "${manifest}" \
      --socket-path "${socket_path}" --cpu-pool "${cpu_pool}" \
      --run-dir "${SUITE_DIR}/shared_evaluator/gpu${gpu}" \
      --eval-dataset-dir "${VALIDATION_DIR}" --max-eval-cases 60 \
      --eval-partition-seed 20260803 --eval-partition-stratify-by profile
  echo "[Launch] ${unit} GPU${gpu} CPUs=${cpu_pool}"
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
  local seed="$2"
  local variant="$3"
  local cpu_set="$4"
  local socket_path="$5"
  local manifest="$6"
  local delay="$7"
  local unit="${UNIT_PREFIX}-g${gpu}-s${seed}"
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
    --property="AllowedCPUs=${cpu_set}" \
    --property="CPUAffinity=${cpu_set}" \
    --property=CPUWeight=100 --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /usr/bin/taskset --cpu-list "${cpu_set}" \
    "${PYTHON}" -u "${TRIAL_RUNNER}" \
      --manifest "${manifest}" --command-key "${variant}_seed${seed}" \
      --gpu "${gpu}" --cpu-set "${cpu_set}" \
      --shared-eval-socket "${socket_path}" \
      --start-delay-seconds "${delay}" \
      --record "${SUITE_DIR}/records/g${gpu}_${variant}_seed${seed}.json"
  echo "[Launch] ${unit} ${variant} seed${seed} CPUs=${cpu_set} delay=${delay}s"
}

launch_postformal_iga_controller() {
  local unit="${POSTFORMAL_IGA_UNIT}"
  local log="${SUITE_DIR}/service_logs/${unit}.log"
  systemd-run --user --unit="${unit}" --collect --same-dir \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --property=Type=exec --property=Restart=no \
    --property=KillMode=control-group --property=TimeoutStopSec=120 \
    --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${FULL_CPU_SET}" \
    --property=CPUAffinity=0 \
    --property=CPUWeight=1 --property=Nice=19 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /usr/bin/taskset --cpu-list 0 \
    "${PYTHON}" -u "${POSTFORMAL_IGA_CONTROLLER}" \
      --launch-contract "${LAUNCH_CONTRACT}" \
      --results-root "${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo" \
      --experiment-name "${RUN_TAG}_formal_${PRIMARY_VARIANT}_seed1" \
      --experiment-name "${RUN_TAG}_formal_${PRIMARY_VARIANT}_seed2" \
      --experiment-name "${RUN_TAG}_formal_${PRIMARY_VARIANT}_seed3" \
      --experiment-name "${RUN_TAG}_formal_${CHALLENGER_VARIANT}_seed1" \
      --experiment-name "${RUN_TAG}_formal_${CHALLENGER_VARIANT}_seed2" \
      --experiment-name "${RUN_TAG}_formal_${CHALLENGER_VARIANT}_seed3" \
      --trainer-unit "${UNIT_PREFIX}-g0-s1.service" \
      --trainer-unit "${UNIT_PREFIX}-g0-s2.service" \
      --trainer-unit "${UNIT_PREFIX}-g0-s3.service" \
      --trainer-unit "${UNIT_PREFIX}-g1-s1.service" \
      --trainer-unit "${UNIT_PREFIX}-g1-s2.service" \
      --trainer-unit "${UNIT_PREFIX}-g1-s3.service" \
      --evaluator-unit "${UNIT_PREFIX}-eval-g0.service" \
      --evaluator-unit "${UNIT_PREFIX}-eval-g1.service" \
      --formal-epochs "${FORMAL_EPOCHS}" \
      --validation-dataset-dir "${VALIDATION_DIR}" \
      --expected-validation-cases 60 \
      --test-dataset-dir "${TEST60_DIR}" --expected-test-cases 60 \
      --iga-launcher "${IGA_LABEL_LAUNCHER}" \
      --iga-label-version "${TEST60_IGA_LABEL_VERSION}" \
      --iga-output-root "${TEST60_IGA_OUTPUT_ROOT}" \
      --iga-teacher-root "${TEST60_IGA_TEACHER_ROOT}" \
      --iga-unit "${TEST60_IGA_UNIT}" \
      --iga-cpu-set "${FULL_CPU_SET}" \
      --iga-workers "${TEST60_IGA_WORKERS}" \
      --python "${PYTHON}" --status-path "${POSTFORMAL_IGA_STATUS}"
  echo "[Launch] ${unit} waits for all E${FORMAL_EPOCHS} validations, then submits test60 IGA-180/1800."
}

launch_evaluator 0 "${CPU_POOL_G0}" "${SOCKET_G0}" "${MANIFEST_G0}"
wait_evaluator "${UNIT_PREFIX}-eval-g0" "${SOCKET_G0}"
launch_evaluator 1 "${CPU_POOL_G1}" "${SOCKET_G1}" "${MANIFEST_G1}"
wait_evaluator "${UNIT_PREFIX}-eval-g1" "${SOCKET_G1}"

launch_trial 0 1 "${PRIMARY_VARIANT}" "${CPU_G0_L0}" "${SOCKET_G0}" "${MANIFEST_G0}" "${DELAY_G0_L0}"
launch_trial 0 2 "${PRIMARY_VARIANT}" "${CPU_G0_L1}" "${SOCKET_G0}" "${MANIFEST_G0}" "${DELAY_G0_L1}"
launch_trial 0 3 "${PRIMARY_VARIANT}" "${CPU_G0_L2}" "${SOCKET_G0}" "${MANIFEST_G0}" "${DELAY_G0_L2}"
launch_trial 1 1 "${CHALLENGER_VARIANT}" "${CPU_G1_L0}" "${SOCKET_G1}" "${MANIFEST_G1}" "${DELAY_G1_L0}"
launch_trial 1 2 "${CHALLENGER_VARIANT}" "${CPU_G1_L1}" "${SOCKET_G1}" "${MANIFEST_G1}" "${DELAY_G1_L1}"
launch_trial 1 3 "${CHALLENGER_VARIANT}" "${CPU_G1_L2}" "${SOCKET_G1}" "${MANIFEST_G1}" "${DELAY_G1_L2}"
launch_postformal_iga_controller

systemctl --user show "${UNITS[@]}" \
  --property=Id --property=MainPID --property=ActiveState \
  --property=SubState --property=AllowedCPUs --property=CPUAffinity
echo "[Done] P3 seeds 1/2/3 on GPU0 and P5 seeds 1/2/3 on GPU1 launched."
echo "[PostFormal] test60 IGA baseline is armed; status=${POSTFORMAL_IGA_STATUS}"
