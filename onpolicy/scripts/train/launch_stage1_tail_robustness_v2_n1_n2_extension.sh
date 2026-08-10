#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python}"
WORKER="${ROOT_DIR}/onpolicy/scripts/train/run_stage1_tail_robustness_v2.py"

PARENT_RUN_TAG="${PARENT_RUN_TAG:-stage1_tail_robustness_v2_20260803_r1}"
RUN_TAG="${RUN_TAG:-${PARENT_RUN_TAG}_extension_n1_n2_20260805_r1}"
PARENT_SUITE="${ROOT_DIR}/result/hkbz_train_logs/${PARENT_RUN_TAG}"
SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-tail-v2-n1-n2-ext}"
CPU_SET_GPU0="${CPU_SET_GPU0:-0-35,72-107}"
CPU_SET_GPU1="${CPU_SET_GPU1:-36-71,108-143}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-30}"
FORMAL_SEEDS="${FORMAL_SEEDS:-1,2,3}"
FORMAL_EPOCHS="${FORMAL_EPOCHS:-8}"
RESUME="${RESUME:-0}"

SOURCE_SUITE="${ROOT_DIR}/result/hkbz_train_logs/stage1_suite_tail_recovery_stage1_20260730_r1"
SOURCE_COMMAND="${SOURCE_SUITE}/formal_R3_tail_team_ratio_seed1.command.json"
OLD_POTENTIAL="${SOURCE_SUITE}/iga_trajectory_analysis.json"
OLD_BC="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/tail_recovery_stage1_20260730_r1_screen_R0_balanced_dagger_global_seed1/run1/models/checkpoint_PlaneBC.pt"
TEACHER_DIR="${ROOT_DIR}/result/hkbz_train_logs/iga_teachers/fjspv3_t600_train_p20_g20_t1800_a3_s1_verified_v1"
EVAL_ROOT="${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803"
VALIDATION_DIR="${EVAL_ROOT}/validation"
BLIND_TEST_DIR="${EVAL_ROOT}/test"
NEW_POTENTIAL="${PARENT_SUITE}/iga_tail_cv_potential.json"
NEW_BC_POINTER="${PARENT_SUITE}/new_plane_bc_checkpoint.txt"

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "[Error] RESUME must be 0 or 1." >&2
  exit 1
fi
if [[ -e "${SUITE_DIR}" && "${RESUME}" != "1" ]]; then
  echo "[Error] Refusing to reuse extension directory without RESUME=1: ${SUITE_DIR}" >&2
  exit 1
fi
if [[ ! -e "${SUITE_DIR}" && "${RESUME}" == "1" ]]; then
  echo "[Error] Cannot resume a missing extension directory: ${SUITE_DIR}" >&2
  exit 1
fi
for required in \
  "${PYTHON}" "${WORKER}" "${SOURCE_COMMAND}" "${OLD_POTENTIAL}" \
  "${OLD_BC}" "${TEACHER_DIR}" "${VALIDATION_DIR}" "${BLIND_TEST_DIR}" \
  "${NEW_POTENTIAL}" "${NEW_BC_POINTER}" /usr/bin/taskset; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done
if ! [[ "${START_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "[Error] START_DELAY_SECONDS must be a non-negative integer." >&2
  exit 1
fi
if ! [[ "${FORMAL_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[Error] FORMAL_EPOCHS must be a positive integer." >&2
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
  if ! nvidia-smi --id="${gpu}" --query-gpu=index --format=csv,noheader,nounits \
    >/dev/null 2>&1; then
    echo "[Error] Cannot query GPU${gpu} through nvidia-smi." >&2
    exit 1
  fi
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

mkdir -p "${SUITE_DIR}/service_logs"
cp -- "${NEW_BC_POINTER}" "${SUITE_DIR}/new_plane_bc_checkpoint.txt"
"${PYTHON}" -c '
import json
import sys
import time
from pathlib import Path

output = Path(sys.argv[1])
run_tag, parent_run_tag, seeds, epochs = sys.argv[2:6]
manifest = {
    "schema_version": 1,
    "enabled": True,
    "formal_only": True,
    "run_tag": run_tag,
    "safe_async_graph_clone_workers": 4,
    "safe_graph_batch_pipeline": True,
    "safe_dagger_teacher_overlap": True,
    "extension": {
        "parent_run_tag": parent_run_tag,
        "reason": "User-authorized post-selection continuation; original gates remain unchanged.",
        "lanes": {
            "0": {"gpu": 0, "variant": "N1_tail_cv_potential", "seeds": seeds},
            "1": {"gpu": 1, "variant": "N2_safe_tail_dagger", "seeds": seeds},
        },
        "formal_epochs": int(epochs),
        "created_unix_time": time.time(),
    },
}
output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
' "${SUITE_DIR}/formal_safe_pipeline.json" "${RUN_TAG}" "${PARENT_RUN_TAG}" "${FORMAL_SEEDS}" "${FORMAL_EPOCHS}"

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
  --formal_epochs "${FORMAL_EPOCHS}"
  --formal-seeds "${FORMAL_SEEDS}"
  --partition_seed 20260803
  --safe_pipeline_all_phases
)

launch_lane() {
  local lane="$1"
  local gpu="$2"
  local cpu_set="$3"
  local variant="$4"
  local unit="${UNIT_PREFIX}-lane${lane}"
  local log="${SUITE_DIR}/service_logs/lane${lane}_${variant}.service.log"
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
      --lane "${lane}" --gpu "${gpu}" \
      --formal-only-variant "${variant}" \
      "${COMMON_ARGS[@]}"
  echo "[Launch] lane=${lane} GPU=${gpu} variant=${variant} CPUs=${cpu_set} unit=${unit}.service"
  echo "[Launch] log=${log}"
}

launch_lane 0 0 "${CPU_SET_GPU0}" N1_tail_cv_potential
echo "[Launch] Waiting ${START_DELAY_SECONDS}s before GPU1 initialization."
sleep "${START_DELAY_SECONDS}"
launch_lane 1 1 "${CPU_SET_GPU1}" N2_safe_tail_dagger

systemctl --user show \
  "${UNIT_PREFIX}-lane0.service" "${UNIT_PREFIX}-lane1.service" \
  --property=Id --property=MainPID --property=ActiveState --property=SubState
echo "[Done] N1/N2 formal extension launched: ${SUITE_DIR}"
