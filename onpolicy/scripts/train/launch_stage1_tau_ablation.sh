#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python3.11}"
RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage1_tau_ablation.py"
RUN_TAG="${RUN_TAG:-stage1_tau_schedule_screen_20260807_r1}"
SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-tau-screen-r1}"

SOURCE_JSON="${ROOT_DIR}/result/hkbz_train_logs/stage1_tail_robustness_v2_20260803_r1_extension_n1_n2_20260805_r1/commands/formal_N1_tail_cv_potential_seed3.json"
INITIAL_CHECKPOINT="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/tail_recovery_stage1_20260730_r1_screen_R0_balanced_dagger_global_seed1/run1/models/checkpoint_PlaneBC.pt"
POTENTIAL_PATH="${ROOT_DIR}/result/hkbz_train_logs/stage1_tail_robustness_v2_20260803_r1/iga_tail_cv_potential.json"
VALIDATION_DIR="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/validation"

CPU_GPU0="0-35,72-107"
CPU_GPU1="36-71,108-143"

for required in \
  "${PYTHON}" "${RUNNER}" "${SOURCE_JSON}" "${INITIAL_CHECKPOINT}" \
  "${POTENTIAL_PATH}" "${VALIDATION_DIR}" /usr/bin/taskset; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done
if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "[Error] User systemd manager is unavailable." >&2
  exit 1
fi

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

mkdir -p "${SUITE_DIR}/service_logs"

launch_lane() {
  local lane="$1"
  local gpu="$2"
  local cpu_set="$3"
  local sequence="$4"
  local delay="$5"
  local unit="${UNIT_PREFIX}-lane${lane}"
  local log="${SUITE_DIR}/service_logs/${unit}.log"
  local state
  state="$(systemctl --user show "${unit}.service" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${state}" && "${state}" != "not-found" ]]; then
    echo "[Error] Refusing to reuse existing unit ${unit}.service." >&2
    exit 1
  fi
  systemd-run --user \
    --unit="${unit}" --collect --same-dir \
    --setenv="CUDA_VISIBLE_DEVICES=${gpu}" \
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
      --lane "${lane}" --gpu "${gpu}" --cpu-set "${cpu_set}" \
      --sequence "${sequence}" --start-delay-seconds "${delay}" \
      --run-tag "${RUN_TAG}" --suite-dir "${SUITE_DIR}" \
      --source-command-json "${SOURCE_JSON}" \
      --initial-checkpoint "${INITIAL_CHECKPOINT}" \
      --potential-path "${POTENTIAL_PATH}" \
      --validation-dir "${VALIDATION_DIR}" \
      --epochs 4 --train-sampling-size 480 --partition-seed 20260803
  echo "[Launch] lane${lane} GPU${gpu} CPUs=${cpu_set} sequence=${sequence}"
}

# Cross over the variants between GPUs so hardware cannot masquerade as a tau
# effect: seed1 is fixed/annealed on GPU0/GPU1; seed2 swaps that assignment.
launch_lane 0 0 "${CPU_GPU0}" "fixed:1,annealed:2" 0
launch_lane 1 1 "${CPU_GPU1}" "annealed:1,fixed:2" 30

systemctl --user show \
  "${UNIT_PREFIX}-lane0.service" "${UNIT_PREFIX}-lane1.service" \
  --property=Id --property=MainPID --property=ActiveState --property=SubState
echo "[Done] Tau ablation launched; blind test remains sealed."
