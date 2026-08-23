#!/usr/bin/env bash
set -euo pipefail

# Resume M0/M1/M2 from their latest completed-shard checkpoints. Emergency
# checkpoints are intentionally not accepted because they have no safe cursor.
# Two 30 x 50 slices cover all 60 rollout workers. Graph-target accumulation
# keeps optimizer mass close to the proven 1000-graph profile.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python3.11}"
TRIAL_RUNNER="${ROOT_DIR}/onpolicy/scripts/train/run_stage2_resource_manifest_trial.py"
EVALUATOR="${ROOT_DIR}/onpolicy/scripts/train/shared_hkbz_evaluator.py"
RUN_TAG="${RUN_TAG:-stage2_resource_wave1_20260819_r9}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
MANIFEST="${MANIFEST:-${SUITE_DIR}/commands/wave1_standard.json}"
CPU_POOL="${CPU_POOL:-0-143}"
EVAL_WORKERS="${EVAL_WORKERS:-60}"
VALIDATION_DIR="${VALIDATION_DIR:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint/tune}"
RESUME_INDEX="${RESUME_INDEX:-1}"
LEGACY_RESUME_MAX_INDEX="${LEGACY_RESUME_MAX_INDEX:-2}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s2-w1-20260819-r9-resume1500-r${RESUME_INDEX}}"
EVAL_UNIT="${UNIT_PREFIX}-eval"
EVAL_SOCKET="/tmp/${EVAL_UNIT}.sock"
MIN_FREE_MIB="${MIN_FREE_MIB:-78000}"

METHODS=(M0_heuristic_uniform M1_iga_uniform M2_iga_role_balanced)
BASE_EXPERIMENTS=(
  stage2_resource_wave1_20260819_r9_wave1_M0_heuristic_uniform_seed1
  stage2_resource_wave1_20260819_r9_wave1_M1_iga_uniform_seed1
  stage2_resource_wave1_20260819_r9_wave1_M2_iga_role_balanced_seed1
)

for required in "${PYTHON}" "${TRIAL_RUNNER}" "${EVALUATOR}" \
  "${MANIFEST}" "${VALIDATION_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing Stage2 resume input: ${required}" >&2
    exit 1
  fi
done
if [[ "$(nproc)" -ne 144 ]]; then
  echo "[Error] Expected 144 logical CPUs, got $(nproc)." >&2
  exit 1
fi
if [[ -e "${EVAL_SOCKET}" ]]; then
  echo "[Error] Refusing to replace evaluator socket ${EVAL_SOCKET}." >&2
  exit 1
fi

units=("${EVAL_UNIT}.service")
for lane in 0 1 2; do
  units+=("${UNIT_PREFIX}-m${lane}.service")
done
for unit in "${units[@]}"; do
  state="$(systemctl --user show "${unit}" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${state}" && "${state}" != "not-found" ]]; then
    echo "[Error] Refusing to reuse existing unit ${unit}." >&2
    exit 1
  fi
done

busy_pids="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -n "${busy_pids//[[:space:]]/}" ]]; then
  echo "[Error] GPU0 still has compute processes: ${busy_pids}" >&2
  exit 1
fi
free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
  echo "[Error] GPU0 free memory ${free_mib:-unknown} MiB is below ${MIN_FREE_MIB} MiB." >&2
  exit 1
fi

mkdir -p "${SUITE_DIR}/service_logs" "${SUITE_DIR}/records" \
  "${SUITE_DIR}/shared_evaluator"

eval_log="${SUITE_DIR}/service_logs/${EVAL_UNIT}.log"
systemd-run --user --unit="${EVAL_UNIT}" --same-dir \
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
  --property="StandardOutput=append:${eval_log}" \
  --property="StandardError=append:${eval_log}" \
  "${PYTHON}" -u "${EVALUATOR}" \
    --source-command-json "${MANIFEST}" \
    --socket-path "${EVAL_SOCKET}" --cpu-pool "${CPU_POOL}" \
    --run-dir "${SUITE_DIR}/shared_evaluator/resume1500_r${RESUME_INDEX}" \
    --eval-dataset-dir "${VALIDATION_DIR}" \
    --max-eval-cases 60 --eval-partition-seed 20260811 \
    --eval-partition-stratify-by distribution >/dev/null

deadline=$((SECONDS + 900))
while (( SECONDS < deadline )); do
  if [[ "$(systemctl --user is-active "${EVAL_UNIT}.service" 2>/dev/null || true)" != "active" ]]; then
    echo "[Error] Shared evaluator exited before readiness." >&2
    exit 1
  fi
  if [[ -S "${EVAL_SOCKET}" ]] && "${PYTHON}" -c \
    'import sys; from onpolicy.utils.shared_eval import SharedEvalClient; reply=SharedEvalClient(sys.argv[1], timeout_seconds=5.0).ping(); raise SystemExit(0 if int(reply.get("worker_count", -1)) == int(sys.argv[2]) else 1)' \
    "${EVAL_SOCKET}" "${EVAL_WORKERS}"; then
    break
  fi
  sleep 2
done
if [[ ! -S "${EVAL_SOCKET}" ]]; then
  echo "[Error] Timed out waiting for evaluator socket ${EVAL_SOCKET}." >&2
  exit 1
fi

for lane in 0 1 2; do
  method="${METHODS[${lane}]}"
  base_experiment="${BASE_EXPERIMENTS[${lane}]}"
  source_experiment=""
  checkpoint=""
  # Prefer the newest 1500-graph resume. The first 1500 run falls back through
  # the completed 1000-graph chain created before sparse pair observations.
  for ((source_index=RESUME_INDEX-1; source_index>=1; source_index--)); do
    candidate_experiment="${base_experiment}_resume1500_r${source_index}"
    candidate_checkpoint="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/${candidate_experiment}/run1/models/checkpoint_Recovery.pt"
    if [[ -s "${candidate_checkpoint}" ]]; then
      source_experiment="${candidate_experiment}"
      checkpoint="${candidate_checkpoint}"
      break
    fi
  done
  if [[ -z "${checkpoint}" ]]; then
    for ((source_index=LEGACY_RESUME_MAX_INDEX; source_index>=0; source_index--)); do
      candidate_experiment="${base_experiment}"
      if (( source_index > 0 )); then
        candidate_experiment+="_resume1000_r${source_index}"
      fi
      candidate_checkpoint="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/${candidate_experiment}/run1/models/checkpoint_Recovery.pt"
      if [[ -s "${candidate_checkpoint}" ]]; then
        source_experiment="${candidate_experiment}"
        checkpoint="${candidate_checkpoint}"
        break
      fi
    done
  fi
  if [[ ! -s "${checkpoint}" ]]; then
    echo "[Error] Missing recovery checkpoint for ${method}: ${checkpoint}" >&2
    exit 1
  fi
  unit="${UNIT_PREFIX}-m${lane}"
  experiment_name="${base_experiment}_resume1500_r${RESUME_INDEX}"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  record="${SUITE_DIR}/records/resume1500_r${RESUME_INDEX}_m${lane}.json"
  delay=$((lane * 15))
  systemd-run --user --unit="${unit}" --same-dir \
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
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    "${PYTHON}" -u "${TRIAL_RUNNER}" \
      --manifest "${MANIFEST}" --command-key "${method}" --gpu 0 \
      --cpu-set "${CPU_POOL}" --shared-eval-socket "${EVAL_SOCKET}" \
      --eval-workers "${EVAL_WORKERS}" --start-delay-seconds "${delay}" \
      --record "${record}" --experiment-name "${experiment_name}" \
      --resume-checkpoint "${checkpoint}" \
      --mini-batch-size 30 --data-chunk-length 50 \
      --max-graphs-per-forward 1500 \
      --grad-accumulation-steps 3 \
      --actor-grad-accumulation-steps 2 \
      --grad-accumulation-target-graphs 5000 \
      --actor-grad-accumulation-target-graphs 2000 >/dev/null
  echo "[Launch] ${unit}.service ${method} source=${source_experiment} cursor=${checkpoint} delay=${delay}s"
done

echo "[Done] Stage2 M0/M1/M2 resume launched with 30x50=1500 graphs per forward."
systemctl --user show "${EVAL_UNIT}.service" \
  "${UNIT_PREFIX}-m0.service" "${UNIT_PREFIX}-m1.service" \
  "${UNIT_PREFIX}-m2.service" \
  --property=Id --property=MainPID --property=ActiveState \
  --property=SubState --property=AllowedCPUs --property=CPUAffinity
