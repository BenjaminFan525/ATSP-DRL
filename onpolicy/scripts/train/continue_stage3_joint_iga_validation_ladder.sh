#!/usr/bin/env bash
set -euo pipefail

# Continue the corrected full-joint Stage3 validation IGA from the verified
# IGA-1800 incumbent.  The second continuation is conditional: IGA-5400 is
# only launched when the completed IGA-3600 mean Cmax is still above the
# configured threshold.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python}"
GENERATOR="${ROOT_DIR}/onpolicy/envs/HKBZ/experiment/generate_stage3_joint_iga_labels.py"
DATASET="${DATASET:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint/tune}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/result/hkbz_train_logs/stage3_encoder_wave1_20260824_r3/joint_iga_tune60}"
CPU_SET="${CPU_SET:-36-71,108-143}"
WORKERS="${WORKERS:-60}"
THRESHOLD="${THRESHOLD:-8750}"
POPULATION="${POPULATION:-20}"
MAX_GENERATIONS="${MAX_GENERATIONS:-100000}"
MAX_STEPS="${MAX_STEPS:-4000}"
IGA_SEED="${IGA_SEED:-20260824}"
PHASE_ATTEMPTS="${PHASE_ATTEMPTS:-3}"
CONTINUE_TO_7200="${CONTINUE_TO_7200:-0}"
CONTINUE_TO_BUDGET="${CONTINUE_TO_BUDGET:-0}"
STATE_PATH="${OUTPUT_ROOT}/budget_ladder_state.json"

SOURCE_1800="${OUTPUT_ROOT}/iga1800"
OUTPUT_3600="${OUTPUT_ROOT}/iga3600"
OUTPUT_5400="${OUTPUT_ROOT}/iga5400"
OUTPUT_7200="${OUTPUT_ROOT}/iga7200"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${OUTPUT_ROOT}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

write_state() {
  local phase="$1"
  local mean_cmax="${2:-}"
  local selected_budget="${3:-}"
  "${PYTHON}" - "${STATE_PATH}" "${phase}" "${mean_cmax}" \
    "${selected_budget}" "${THRESHOLD}" "${CPU_SET}" "${WORKERS}" <<'PY'
import datetime
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {}
if path.is_file():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
payload.update(
    {
        "schema_version": 1,
        "updated_at": datetime.datetime.now().astimezone().isoformat(),
        "phase": sys.argv[2],
        "threshold_mean_cmax": float(sys.argv[5]),
        "cpu_set": sys.argv[6],
        "workers": int(sys.argv[7]),
    }
)
if sys.argv[3]:
    payload["latest_mean_cmax"] = float(sys.argv[3])
if sys.argv[4]:
    payload["selected_cumulative_budget_seconds"] = float(sys.argv[4])
if (
    sys.argv[2].startswith("completed_at_iga")
    and sys.argv[3]
    and sys.argv[4]
):
    budget_key = str(int(float(sys.argv[4])))
    payload.setdefault("checkpoints", {})[budget_key] = {
        "mean_cmax": float(sys.argv[3]),
        "completed_at": payload["updated_at"],
    }
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

verify_phase() {
  local source_dir="$1"
  local output_dir="$2"
  local source_budget="$3"
  local target_budget="$4"
  "${PYTHON}" - "${source_dir}" "${output_dir}" "${source_budget}" \
    "${target_budget}" <<'PY'
import json
import math
import pathlib
import statistics
import sys

source_dir = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])
source_budget = float(sys.argv[3])
target_budget = float(sys.argv[4])
additional_budget = target_budget - source_budget
summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
required = {
    "status": "completed",
    "teacher_scope": "stage3_full_joint_policy",
    "teacher_method": "joint_iga_all",
    "environment_semantics_version": "progressive-departure-r014-pipeline-v2",
    "expected_case_count": 60,
    "completed_case_count": 60,
    "missing_cases": [],
}
mismatch = {
    key: (summary.get(key), value)
    for key, value in required.items()
    if summary.get(key) != value
}
if not math.isclose(
    float(summary.get("nominal_cumulative_budget_seconds", -1)),
    target_budget,
    abs_tol=1e-9,
):
    mismatch["nominal_cumulative_budget_seconds"] = (
        summary.get("nominal_cumulative_budget_seconds"),
        target_budget,
    )
if mismatch:
    raise SystemExit(f"invalid summary contract: {mismatch}")

source_teachers = sorted((source_dir / "teachers").glob("case_*.json"))
target_teachers = sorted((output_dir / "teachers").glob("case_*.json"))
if len(source_teachers) != 60 or len(target_teachers) != 60:
    raise SystemExit(
        f"teacher count mismatch: source={len(source_teachers)} target={len(target_teachers)}"
    )

regressions = []
makespans = []
for source_path, target_path in zip(source_teachers, target_teachers):
    if source_path.name != target_path.name:
        raise SystemExit(f"case ordering mismatch: {source_path.name} != {target_path.name}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))
    warm = target.get("search", {}).get("warm_start", {})
    search = target.get("search", {})
    if warm.get("kind") != "verified_stage3_nested_incumbent":
        raise SystemExit(f"missing verified nested incumbent: {target_path}")
    if not math.isclose(
        float(warm.get("source_nominal_budget_seconds", -1)),
        source_budget,
        abs_tol=1e-9,
    ):
        raise SystemExit(f"wrong warm-start budget: {target_path}")
    if not math.isclose(
        float(search.get("configured_additional_budget_seconds", -1)),
        additional_budget,
        abs_tol=1e-9,
    ):
        raise SystemExit(f"wrong additional budget: {target_path}")
    source_value = float(source["makespan"])
    target_value = float(target["makespan"])
    if target_value > source_value + 1e-9:
        regressions.append((target_path.stem, source_value, target_value))
    makespans.append(target_value)
if regressions:
    raise SystemExit(f"nested continuation regressed: {regressions}")

mean_cmax = statistics.fmean(makespans)
if not math.isclose(mean_cmax, float(summary["makespan_mean"]), abs_tol=1e-9):
    raise SystemExit(
        f"summary mean mismatch: recomputed={mean_cmax} summary={summary['makespan_mean']}"
    )
print(f"{mean_cmax:.12f}")
PY
}

run_phase() {
  local source_dir="$1"
  local output_dir="$2"
  local source_budget="$3"
  local target_budget="$4"
  local additional_budget
  additional_budget="$((target_budget - source_budget))"

  local attempt
  for ((attempt = 1; attempt <= PHASE_ATTEMPTS; attempt++)); do
    echo "[Stage3IGA-Ladder] target=${target_budget}s attempt=${attempt}/${PHASE_ATTEMPTS}"
    if /usr/bin/taskset --cpu-list "${CPU_SET}" \
      "${PYTHON}" -u "${GENERATOR}" \
        --dataset-dir "${DATASET}" \
        --output-dir "${output_dir}" \
        --warm-start-dir "${source_dir}" \
        --warm-start-budget-seconds "${source_budget}" \
        --time-budget-seconds "${additional_budget}" \
        --cumulative-budget-seconds "${target_budget}" \
        --population "${POPULATION}" \
        --max-generations "${MAX_GENERATIONS}" \
        --workers "${WORKERS}" \
        --seed "${IGA_SEED}" \
        --max-steps "${MAX_STEPS}" \
        --max-plane-agents 24 \
        --max-device-num 80 \
        --device-lookahead-safety-margin 60; then
      return 0
    fi
    echo "[Stage3IGA-Ladder] target=${target_budget}s failed; resume incomplete cases"
  done
  return 1
}

if [[ ! -f "${SOURCE_1800}/summary.json" ]]; then
  echo "Missing verified IGA-1800 source: ${SOURCE_1800}" >&2
  exit 2
fi

if [[ ! "${CONTINUE_TO_BUDGET}" =~ ^[0-9]+$ ]]; then
  echo "CONTINUE_TO_BUDGET must be a non-negative integer" >&2
  exit 2
fi

if ((CONTINUE_TO_BUDGET > 7200)); then
  if (((CONTINUE_TO_BUDGET - 7200) % 1800 != 0)); then
    echo "CONTINUE_TO_BUDGET must advance from 7200 in 1800-second steps" >&2
    exit 2
  fi
  if [[ ! -f "${OUTPUT_7200}/summary.json" ]]; then
    echo "Missing verified IGA-7200 source: ${OUTPUT_7200}" >&2
    exit 2
  fi

  SOURCE_MEAN="$(verify_phase "${OUTPUT_5400}" "${OUTPUT_7200}" 5400 7200)"
  write_state "completed_at_iga7200" "${SOURCE_MEAN}" 7200
  SOURCE_DIR="${OUTPUT_7200}"
  SOURCE_BUDGET=7200

  while ((SOURCE_BUDGET < CONTINUE_TO_BUDGET)); do
    TARGET_BUDGET=$((SOURCE_BUDGET + 1800))
    TARGET_DIR="${OUTPUT_ROOT}/iga${TARGET_BUDGET}"
    write_state "iga${TARGET_BUDGET}_running" "${SOURCE_MEAN}" "${SOURCE_BUDGET}"
    echo "[Stage3IGA-Ladder] continue verified IGA-${SOURCE_BUDGET} to IGA-${TARGET_BUDGET}"
    run_phase "${SOURCE_DIR}" "${TARGET_DIR}" "${SOURCE_BUDGET}" "${TARGET_BUDGET}"
    TARGET_MEAN="$(verify_phase "${SOURCE_DIR}" "${TARGET_DIR}" "${SOURCE_BUDGET}" "${TARGET_BUDGET}")"
    write_state "completed_at_iga${TARGET_BUDGET}" "${TARGET_MEAN}" "${TARGET_BUDGET}"
    echo "[Stage3IGA-Ladder] checkpoint IGA-${TARGET_BUDGET} validation60 mean Cmax=${TARGET_MEAN}"
    SOURCE_DIR="${TARGET_DIR}"
    SOURCE_BUDGET="${TARGET_BUDGET}"
    SOURCE_MEAN="${TARGET_MEAN}"
  done
  exit 0
fi

if [[ "${CONTINUE_TO_7200}" == "1" ]]; then
  if [[ ! -f "${OUTPUT_5400}/summary.json" ]]; then
    echo "Missing verified IGA-5400 source: ${OUTPUT_5400}" >&2
    exit 2
  fi
  MEAN_5400="$(verify_phase "${OUTPUT_3600}" "${OUTPUT_5400}" 3600 5400)"
  write_state "iga7200_running" "${MEAN_5400}" 5400
  echo "[Stage3IGA-Ladder] continue verified IGA-5400 to IGA-7200"
  run_phase "${OUTPUT_5400}" "${OUTPUT_7200}" 5400 7200
  MEAN_7200="$(verify_phase "${OUTPUT_5400}" "${OUTPUT_7200}" 5400 7200)"
  write_state "completed_at_iga7200" "${MEAN_7200}" 7200
  echo "[Stage3IGA-Ladder] completed at IGA-7200; validation60 mean Cmax=${MEAN_7200}"
  exit 0
fi

write_state "iga3600_running"
run_phase "${SOURCE_1800}" "${OUTPUT_3600}" 1800 3600
MEAN_3600="$(verify_phase "${SOURCE_1800}" "${OUTPUT_3600}" 1800 3600)"
echo "[Stage3IGA-Ladder] IGA-3600 validation60 mean Cmax=${MEAN_3600} threshold=${THRESHOLD}"

if "${PYTHON}" -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)' \
    "${MEAN_3600}" "${THRESHOLD}"; then
  write_state "iga5400_running" "${MEAN_3600}" 3600
  echo "[Stage3IGA-Ladder] mean remains above threshold; continue to IGA-5400"
  run_phase "${OUTPUT_3600}" "${OUTPUT_5400}" 3600 5400
  MEAN_5400="$(verify_phase "${OUTPUT_3600}" "${OUTPUT_5400}" 3600 5400)"
  write_state "completed_at_iga5400" "${MEAN_5400}" 5400
  echo "[Stage3IGA-Ladder] completed at IGA-5400; validation60 mean Cmax=${MEAN_5400}"
else
  write_state "stopped_at_iga3600_threshold_met" "${MEAN_3600}" 3600
  echo "[Stage3IGA-Ladder] threshold met; stop after IGA-3600"
fi
