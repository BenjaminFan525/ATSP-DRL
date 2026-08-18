#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python3.11}"
PREPARE="${ROOT_DIR}/onpolicy/scripts/train/prepare_stage1_departure_research.py"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage1_departure_manifest_trial.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"

RUN_TAG="${RUN_TAG:-stage1_departure_research_20260812_r1}"
SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-departure-bc-r1}"
SOURCE_COMMAND="${ROOT_DIR}/result/hkbz_train_logs/stage1_next_round_20260808_r1/commands/formal_M2_bc_kl_anneal_seed1.json"
DATASET_DIR="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train"
TEACHER_DIR="${ROOT_DIR}/result/hkbz_train_logs/iga_teachers/fjspv3_t600_train_s1_progressive_departure_r014_20260812_v2_p20_g20_t1800_a5_s1"
POTENTIAL_PATH="${SUITE_DIR}/iga_potential_v2.json"
WARM_CHECKPOINT="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage1_next_round_20260808_r1_formal_M2_bc_kl_anneal_seed1/run1/models/checkpoint_Best.pt"
VALIDATION_DIR="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/validation"

MANIFEST_G0="${SUITE_DIR}/commands/bc_screen_seed1.json"
MANIFEST_G1="${SUITE_DIR}/commands/bc_screen_seed2.json"
SOCKET_G0="${SOCKET_G0:-/tmp/hkbz-dep-bc-r1-g0.sock}"
SOCKET_G1="${SOCKET_G1:-/tmp/hkbz-dep-bc-r1-g1.sock}"

CPU_POOL_G0="0-35,72-107"
CPU_POOL_G1="36-71,108-143"
CPU_G0_B0="0-11,72-83"
CPU_G0_B1="12-23,84-95"
CPU_G0_B2="24-35,96-107"
CPU_G1_B0="36-47,108-119"
CPU_G1_B1="48-59,120-131"
CPU_G1_B2="60-71,132-143"

for required in \
  "${PYTHON}" "${PREPARE}" "${TRIAL_RUNNER}" "${EVALUATOR}" \
  "${SOURCE_COMMAND}" "${DATASET_DIR}" "${TEACHER_DIR}" \
  "${POTENTIAL_PATH}" "${WARM_CHECKPOINT}" "${VALIDATION_DIR}" \
  /usr/bin/taskset; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done

"${PYTHON}" - "${CPU_POOL_G0}" "${CPU_G0_B0}" "${CPU_G0_B1}" "${CPU_G0_B2}" \
  "${CPU_POOL_G1}" "${CPU_G1_B0}" "${CPU_G1_B1}" "${CPU_G1_B2}" <<'PY'
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
        raise SystemExit('Trainer slices must exactly cover the NUMA-local pool.')
    if any(slices[i] & slices[j] for i in range(3) for j in range(i + 1, 3)):
        raise SystemExit('Trainer CPU slices overlap.')
    for cpus in slices:
        physical = {cpu if cpu < 72 else cpu - 72 for cpu in cpus}
        if len(physical) != 12:
            raise SystemExit('Every trainer must own both SMT siblings of 12 cores.')
PY

UNITS=(
  "${UNIT_PREFIX}-eval-g0.service" "${UNIT_PREFIX}-eval-g1.service"
  "${UNIT_PREFIX}-g0-b0.service" "${UNIT_PREFIX}-g0-b1.service"
  "${UNIT_PREFIX}-g0-b2.service" "${UNIT_PREFIX}-g1-b0.service"
  "${UNIT_PREFIX}-g1-b1.service" "${UNIT_PREFIX}-g1-b2.service"
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

mkdir -p "${SUITE_DIR}/commands" "${SUITE_DIR}/service_logs" \
  "${SUITE_DIR}/records" "${SUITE_DIR}/shared_evaluator/gpu0" \
  "${SUITE_DIR}/shared_evaluator/gpu1"

prepare_manifest() {
  local seed="$1"
  local output="$2"
  "${PYTHON}" "${PREPARE}" \
    --phase bc_screen --run-tag "${RUN_TAG}" \
    --source-command-json "${SOURCE_COMMAND}" \
    --dataset-dir "${DATASET_DIR}" --teacher-dir "${TEACHER_DIR}" \
    --potential-path "${POTENTIAL_PATH}" \
    --warm-start-checkpoint "${WARM_CHECKPOINT}" \
    --expected-cases 600 --screen-seed "${seed}" --screen-epochs 4 \
    --output "${output}"
}

prepare_manifest 1 "${MANIFEST_G0}"
prepare_manifest 2 "${MANIFEST_G1}"

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
      --run-dir "${SUITE_DIR}/shared_evaluator/gpu${gpu}" \
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
  local short="$2"
  local key="$3"
  local cpu_set="$4"
  local socket_path="$5"
  local manifest="$6"
  local delay="$7"
  local unit="${UNIT_PREFIX}-g${gpu}-${short}"
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
      --record "${SUITE_DIR}/records/g${gpu}_${short}.json"
  echo "[Launch] GPU${gpu} ${key} CPUs=${cpu_set} delay=${delay}s"
}

launch_evaluator 0 "${CPU_POOL_G0}" "${SOCKET_G0}" "${MANIFEST_G0}"
wait_evaluator "${UNIT_PREFIX}-eval-g0" "${SOCKET_G0}"
launch_evaluator 1 "${CPU_POOL_G1}" "${SOCKET_G1}" "${MANIFEST_G1}"
wait_evaluator "${UNIT_PREFIX}-eval-g1" "${SOCKET_G1}"

launch_trial 0 b0 B0_legacy_dagger_warm "${CPU_G0_B0}" "${SOCKET_G0}" "${MANIFEST_G0}" 0
launch_trial 0 b1 B1_phase_dagger_warm "${CPU_G0_B1}" "${SOCKET_G0}" "${MANIFEST_G0}" 120
launch_trial 0 b2 B2_phase_dagger_cold "${CPU_G0_B2}" "${SOCKET_G0}" "${MANIFEST_G0}" 240
launch_trial 1 b0 B0_legacy_dagger_warm "${CPU_G1_B0}" "${SOCKET_G1}" "${MANIFEST_G1}" 30
launch_trial 1 b1 B1_phase_dagger_warm "${CPU_G1_B1}" "${SOCKET_G1}" "${MANIFEST_G1}" 150
launch_trial 1 b2 B2_phase_dagger_cold "${CPU_G1_B2}" "${SOCKET_G1}" "${MANIFEST_G1}" 270

systemctl --user show "${UNITS[@]}" \
  --property=Id --property=MainPID --property=ActiveState --property=SubState \
  --property=AllowedCPUs --property=CPUAffinity
echo "[Done] Six BC-screen trainers and two shared validation pools launched."
