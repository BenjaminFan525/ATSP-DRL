#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python}"
WORKER="${ROOT_DIR}/onpolicy/scripts/train/run_stage1_tail_robustness_v2.py"
BUILD_EVAL="${ROOT_DIR}/onpolicy/envs/HKBZ/experiment/build_stage1_v2_eval_dataset.py"
REFIT="${ROOT_DIR}/onpolicy/envs/HKBZ/experiment/analyze_iga_teacher_trajectories.py"

RUN_TAG="${RUN_TAG:-stage1_tail_robustness_v2_20260803_r1}"
SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-tail-v2}"
CPU_SET_GPU0="${CPU_SET_GPU0:-0-35,72-107}"
CPU_SET_GPU1="${CPU_SET_GPU1:-36-71,108-143}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-30}"
RESUME="${RESUME:-0}"
SAFE_PIPELINE_ALL_PHASES="${SAFE_PIPELINE_ALL_PHASES:-1}"

SOURCE_SUITE="${ROOT_DIR}/result/hkbz_train_logs/stage1_suite_tail_recovery_stage1_20260730_r1"
SOURCE_COMMAND="${SOURCE_SUITE}/formal_R3_tail_team_ratio_seed1.command.json"
OLD_POTENTIAL="${SOURCE_SUITE}/iga_trajectory_analysis.json"
SOURCE_TRAJECTORY="${SOURCE_SUITE}/iga_teacher_trajectories.jsonl"
OLD_BC="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/tail_recovery_stage1_20260730_r1_screen_R0_balanced_dagger_global_seed1/run1/models/checkpoint_PlaneBC.pt"
TEACHER_DIR="${ROOT_DIR}/result/hkbz_train_logs/iga_teachers/fjspv3_t600_train_p20_g20_t1800_a3_s1_verified_v1"
TRAIN_DIR="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train"
EVAL_ROOT="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803"
VALIDATION_DIR="${EVAL_ROOT}/validation"
BLIND_TEST_DIR="${EVAL_ROOT}/test"
NEW_POTENTIAL="${SUITE_DIR}/iga_tail_cv_potential.json"

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "[Error] RESUME must be 0 or 1." >&2
  exit 1
fi
if [[ "${SAFE_PIPELINE_ALL_PHASES}" != "0" && "${SAFE_PIPELINE_ALL_PHASES}" != "1" ]]; then
  echo "[Error] SAFE_PIPELINE_ALL_PHASES must be 0 or 1." >&2
  exit 1
fi
if [[ -e "${SUITE_DIR}" && "${RESUME}" != "1" ]]; then
  echo "[Error] Refusing to reuse existing suite directory without RESUME=1: ${SUITE_DIR}" >&2
  exit 1
fi
if [[ ! -e "${SUITE_DIR}" && "${RESUME}" == "1" ]]; then
  echo "[Error] Cannot resume a missing suite directory: ${SUITE_DIR}" >&2
  exit 1
fi
for required in \
  "${PYTHON}" "${WORKER}" "${BUILD_EVAL}" "${REFIT}" \
  "${SOURCE_COMMAND}" "${OLD_POTENTIAL}" "${SOURCE_TRAJECTORY}" \
  "${OLD_BC}" "${TEACHER_DIR}" "${TRAIN_DIR}" /usr/bin/taskset; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done
if ! [[ "${START_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "[Error] START_DELAY_SECONDS must be a non-negative integer." >&2
  exit 1
fi

"${PYTHON}" -c '
import sys

def expand(spec):
    cpus = set()
    for part in spec.split(","):
        bounds = [int(value) for value in part.split("-", 1)]
        cpus.update(range(bounds[0], bounds[-1] + 1))
    return cpus

left, right = expand(sys.argv[1]), expand(sys.argv[2])
if left & right:
    raise SystemExit(f"CPU sets overlap: {sorted(left & right)}")
if len(left) != 72 or len(right) != 72:
    raise SystemExit(f"Expected 72 logical CPUs per lane, got {len(left)}/{len(right)}")
' "${CPU_SET_GPU0}" "${CPU_SET_GPU1}"

for mapping in "0:${CPU_SET_GPU0}" "1:${CPU_SET_GPU1}"; do
  gpu="${mapping%%:*}"
  cpu_set="${mapping#*:}"
  /usr/bin/taskset --cpu-list "${cpu_set}" /bin/true
  busy_pids="$(
    nvidia-smi --id="${gpu}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true
  )"
  if [[ -n "${busy_pids//[[:space:]]/}" ]]; then
    echo "[Error] GPU${gpu} is busy; compute PIDs: ${busy_pids}" >&2
    exit 1
  fi
done
if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "[Error] User systemd manager is unavailable." >&2
  exit 1
fi
for lane in 0 1; do
  unit="${UNIT_PREFIX}-lane${lane}.service"
  load_state="$(
    systemctl --user show "${unit}" --property=LoadState --value 2>/dev/null || true
  )"
  if [[ -n "${load_state}" && "${load_state}" != "not-found" ]]; then
    echo "[Error] Refusing to reuse existing unit ${unit}." >&2
    exit 1
  fi
done

if [[ ! -f "${EVAL_ROOT}/manifest.json" ]]; then
  echo "[Prepare] Generating independent validation120/blind-test60 dataset."
  "${PYTHON}" "${BUILD_EVAL}" \
    --output "${EVAL_ROOT}" \
    --validation_cases 120 \
    --test_cases 60 \
    --seed 20260803
fi

mkdir -p "${SUITE_DIR}/service_logs"
if [[ "${RESUME}" == "1" ]]; then
  if [[ ! -f "${NEW_POTENTIAL}" ]]; then
    echo "[Error] Resume potential is missing: ${NEW_POTENTIAL}" >&2
    exit 1
  fi
  echo "[Resume] Reusing prepared dataset and tail potential from ${SUITE_DIR}."
else
  echo "[Prepare] Refitting case/profile-balanced tail potential with case-level CV."
  "${PYTHON}" "${REFIT}" \
    --dataset_dir "${TRAIN_DIR}" \
    --teacher_dir "${TEACHER_DIR}" \
    --env_config "${ROOT_DIR}/onpolicy/config/env_plane_pretrain.yaml" \
    --output_json "${NEW_POTENTIAL}" \
    --trajectory_jsonl "${SOURCE_TRAJECTORY}" \
    --reuse_trajectory_jsonl "${SOURCE_TRAJECTORY}" \
    --source_analysis_json "${OLD_POTENTIAL}" \
    --ridge 0.001 \
    --seed 20260803 \
    --distribution_weights "iid=0.50,ood_stress=0.45,ood_scale=0.05" \
    --profile_balance \
    --tail_start_fraction 0.75 \
    --tail_weight 4.0 \
    --final_tail_start_fraction 0.90 \
    --final_tail_weight 8.0 \
    --cv_folds 5
fi

"${PYTHON}" -c '
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cv = payload["calibration"]["case_level_cross_validation"]
if payload.get("status") != "completed" or not cv.get("case_disjoint"):
    raise SystemExit("Tail potential validation failed")
if not math.isfinite(float(cv["metrics"]["r2"])):
    raise SystemExit("Tail potential CV R2 is not finite")
overall_r2 = float(cv["metrics"]["r2"])
tail_r2 = float(cv["phase_metrics"]["final_tail_0_90_1_00"]["r2"])
print(
    "[Prepare] potential CV R2="
    f"{overall_r2:.4f}, "
    "final-tail CV R2="
    f"{tail_r2:.4f}"
)
' "${NEW_POTENTIAL}"

COMMON_ARGS=(
  --run_tag "${RUN_TAG}"
  --suite_dir "${SUITE_DIR}"
  --source_command_json "${SOURCE_COMMAND}"
  --old_bc_checkpoint "${OLD_BC}"
  --old_potential "${OLD_POTENTIAL}"
  --new_potential "${NEW_POTENTIAL}"
  --teacher_dir "${TEACHER_DIR}"
  --validation_dir "${VALIDATION_DIR}"
  --blind_test_dir "${BLIND_TEST_DIR}"
  --screen_epochs 2
  --formal_epochs 8
  --partition_seed 20260803
)
if [[ "${SAFE_PIPELINE_ALL_PHASES}" == "1" ]]; then
  COMMON_ARGS+=(--safe_pipeline_all_phases)
fi

launch_lane() {
  local lane="$1"
  local gpu="$2"
  local cpu_set="$3"
  local unit="${UNIT_PREFIX}-lane${lane}"
  local log="${SUITE_DIR}/service_logs/lane${lane}.service.log"
  systemd-run --user \
    --unit="${unit}" \
    --collect \
    --same-dir \
    --property=Type=exec \
    --property=Restart=no \
    --property=KillMode=control-group \
    --property=TimeoutStopSec=120 \
    --property=LimitNOFILE=65536 \
    --property="AllowedCPUs=${cpu_set}" \
    --property="CPUAffinity=${cpu_set}" \
    --property=CPUWeight=100 \
    --property=Nice=0 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /usr/bin/taskset --cpu-list "${cpu_set}" \
    "${PYTHON}" -u "${WORKER}" \
      --lane "${lane}" --gpu "${gpu}" "${COMMON_ARGS[@]}"
  echo "[Launch] lane=${lane} GPU=${gpu} CPUs=${cpu_set} unit=${unit}.service"
  echo "[Launch] log=${log}"
}

launch_lane 0 0 "${CPU_SET_GPU0}"
echo "[Launch] Waiting ${START_DELAY_SECONDS}s before GPU1 initialization."
sleep "${START_DELAY_SECONDS}"
launch_lane 1 1 "${CPU_SET_GPU1}"

systemctl --user show \
  "${UNIT_PREFIX}-lane0.service" "${UNIT_PREFIX}-lane1.service" \
  --property=Id --property=MainPID --property=ActiveState --property=SubState
echo "[Done] Stage-1 tail-robustness v2 launched: ${SUITE_DIR}"
