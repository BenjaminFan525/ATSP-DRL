#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python}"
RUNNER="${ROOT_DIR}/onpolicy/envs/HKBZ/experiment/run_fjsp_v2_evolutionary_parallel.py"
DATASET="${DATASET:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train}"

if [[ -z "${DATASET_TAG:-}" ]]; then
  case "$(basename -- "${DATASET}")" in
    train) DATASET_TAG="train600" ;;
    test) DATASET_TAG="test60" ;;
    *) DATASET_TAG="$(basename -- "${DATASET}")" ;;
  esac
fi
if ! [[ "${DATASET_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[Error] DATASET_TAG contains unsafe path characters: ${DATASET_TAG}" >&2
  exit 2
fi
case "${DATASET_TAG}" in
  train600) DEFAULT_TEACHER_DATASET_PREFIX="fjspv3_t600_train" ;;
  test60) DEFAULT_TEACHER_DATASET_PREFIX="fjspv3_test60" ;;
  *) DEFAULT_TEACHER_DATASET_PREFIX="fjspv3_${DATASET_TAG}" ;;
esac
TEACHER_DATASET_PREFIX="${TEACHER_DATASET_PREFIX:-${DEFAULT_TEACHER_DATASET_PREFIX}}"

LABEL_VERSION="${LABEL_VERSION:-s1_progressive_departure_r014_20260812_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/result/hkbz_train_logs/iga_labeling_${LABEL_VERSION}}"
TEACHER_ROOT="${TEACHER_ROOT:-${ROOT_DIR}/result/hkbz_train_logs/iga_teachers}"
WORKERS="${WORKERS:-144}"
POPULATION="${POPULATION:-20}"
GENERATIONS="${GENERATIONS:-20}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
IGA_SEED="${IGA_SEED:-1}"
RUNNER_PASSES="${RUNNER_PASSES:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-30}"

# The two budgets run sequentially, so both may use every logical CPU.  IGA-180
# is a hard prerequisite: IGA-1800 is not started until all IGA-180 cases and
# exported teachers pass independent replay verification.
CPU_SET="${CPU_SET:-0-143}"
UNIT_SEQUENCE="${UNIT_SEQUENCE:-hkbz-s1-iga-departure-sequential-20260812-v2}"

OUTPUT_180="${OUTPUT_ROOT}/iga180_${DATASET_TAG}.json"
OUTPUT_1800="${OUTPUT_ROOT}/iga1800_${DATASET_TAG}.json"
TEACHER_180="${TEACHER_ROOT}/${TEACHER_DATASET_PREFIX}_${LABEL_VERSION}_p${POPULATION}_g${GENERATIONS}_t180_a${MAX_ATTEMPTS}_s${IGA_SEED}"
TEACHER_1800="${TEACHER_ROOT}/${TEACHER_DATASET_PREFIX}_${LABEL_VERSION}_p${POPULATION}_g${GENERATIONS}_t1800_a${MAX_ATTEMPTS}_s${IGA_SEED}"
if [[ "${DATASET_TAG}" == "train600" ]]; then
  LOG_SEQUENCE="${OUTPUT_ROOT}/iga180_then_1800.service.log"
else
  LOG_SEQUENCE="${OUTPUT_ROOT}/iga180_then_1800_${DATASET_TAG}.service.log"
fi

verify_label_set() {
  local budget="$1"
  local output_json="$2"
  local teacher_dir="$3"

  "${PYTHON}" - \
    "${DATASET}" "${output_json}" "${teacher_dir}" "${budget}" \
    "${POPULATION}" "${GENERATIONS}" "${MAX_ATTEMPTS}" \
    "${IGA_SEED}" <<'PY'
import json
from pathlib import Path
import sys

(
    dataset_arg,
    output_arg,
    teacher_arg,
    budget_arg,
    population_arg,
    generations_arg,
    attempts_arg,
    seed_arg,
) = sys.argv[1:]
dataset = Path(dataset_arg).resolve()
output = Path(output_arg).resolve()
teacher_dir = Path(teacher_arg).resolve()
expected_semantics = "progressive-departure-r014-pipeline-v2"
expected_cases = sorted(path.name for path in dataset.glob("case_*") if path.is_dir())

with output.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("status") != "completed":
    raise SystemExit(f"label payload is not completed: {payload.get('status')}")
if payload.get("teacher_scope") != "stage1_plane_policy":
    raise SystemExit("label payload is not for the Stage-1 plane policy")
if payload.get("resource_policy") != "heuristic":
    raise SystemExit("Stage-1 label payload did not use heuristic resources")
if payload.get("environment_semantics_version") != expected_semantics:
    raise SystemExit("label payload uses the wrong environment semantics")
if int(payload.get("case_count", -1)) != len(expected_cases):
    raise SystemExit("label payload case_count does not match the dataset")
if int(payload.get("seed", -1)) != int(seed_arg):
    raise SystemExit("label payload seed mismatch")
expected_budget = {
    "population": int(population_arg),
    "generations": int(generations_arg),
    "time_seconds": float(budget_arg),
    "max_attempts": int(attempts_arg),
}
if payload.get("legacy_budget") != expected_budget:
    raise SystemExit("label payload budget mismatch")

method = payload.get("methods", {}).get("IGA", {})
records = method.get("cases", [])
summary = method.get("summary", {})
if method.get("status") != "completed":
    raise SystemExit(f"IGA status is not completed: {method.get('status')}")
if int(summary.get("completed_count", -1)) != len(expected_cases):
    raise SystemExit("not every IGA case completed")
if int(summary.get("verified_count", -1)) != len(expected_cases):
    raise SystemExit("not every IGA case was replay-verified")
if len(records) != len(expected_cases):
    raise SystemExit("IGA record count mismatch")
by_case = {str(record.get("case")): record for record in records}
if sorted(by_case) != expected_cases:
    raise SystemExit("IGA record case set mismatch")

teacher_files = sorted(teacher_dir.glob("case_*.json"))
if [path.stem for path in teacher_files] != expected_cases:
    raise SystemExit("teacher file set mismatch")
for case_name in expected_cases:
    record = by_case[case_name]
    if not record.get("completed") or not record.get("completion_verified"):
        raise SystemExit(f"unverified IGA record: {case_name}")
    with (teacher_dir / f"{case_name}.json").open("r", encoding="utf-8") as handle:
        teacher = json.load(handle)
    if teacher.get("case") != case_name:
        raise SystemExit(f"teacher case mismatch: {case_name}")
    if teacher.get("teacher_scope") != "stage1_plane_policy":
        raise SystemExit(f"teacher is not for Stage-1: {case_name}")
    if teacher.get("resource_policy") != "heuristic":
        raise SystemExit(f"teacher resource policy mismatch: {case_name}")
    if teacher.get("environment_semantics_version") != expected_semantics:
        raise SystemExit(f"teacher semantics mismatch: {case_name}")
    if not teacher.get("completion_verified"):
        raise SystemExit(f"teacher replay verification missing: {case_name}")
    job_priorities = teacher.get("job_priorities", [])
    site_priorities = teacher.get("site_priorities", [])
    if (
        not job_priorities
        or len(job_priorities) != len(site_priorities)
        or any(len(row) != 20 for row in job_priorities)
        or any(not row for row in site_priorities)
    ):
        raise SystemExit(f"invalid Stage-1 priority tensors: {case_name}")
print(
    f"[Verify] IGA-{budget_arg}: {len(expected_cases)}/{len(expected_cases)} "
    "completed and replay-verified",
    flush=True,
)
PY
}

run_budget() {
  local budget="$1"
  local output_json="$2"
  local teacher_dir="$3"
  local pass_number
  local mpl_dir="/tmp/hkbz-mpl-iga-${budget}-${LABEL_VERSION}"

  mkdir -p "${mpl_dir}" "${teacher_dir}"
  for ((pass_number = 1; pass_number <= RUNNER_PASSES; pass_number++)); do
    echo "[Run] IGA-${budget} pass ${pass_number}/${RUNNER_PASSES}"
    if env \
      PYTHONUNBUFFERED=1 \
      PYTHONHASHSEED=0 \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      MPLCONFIGDIR="${mpl_dir}" \
      /usr/bin/taskset --cpu-list "${CPU_SET}" \
      "${PYTHON}" -u "${RUNNER}" \
        --dataset_test_dir "${DATASET}" \
        --output_json "${output_json}" \
        --methods iga \
        --workers "${WORKERS}" \
        --time_budget "${budget}" \
        --iga_pop_size "${POPULATION}" \
        --iga_generations "${GENERATIONS}" \
        --iga_max_attempts "${MAX_ATTEMPTS}" \
        --iga_teacher_dir "${teacher_dir}" \
        --seed "${IGA_SEED}" \
        --resume; then
      verify_label_set "${budget}" "${output_json}" "${teacher_dir}"
      return 0
    fi
    if (( pass_number < RUNNER_PASSES )); then
      echo "[Retry] IGA-${budget} has failed/unverified cases; retrying only those cases in ${RETRY_DELAY_SECONDS}s."
      sleep "${RETRY_DELAY_SECONDS}"
    fi
  done

  echo "[Error] IGA-${budget} still has failed or unverified cases after ${RUNNER_PASSES} passes." >&2
  return 1
}

run_sequence() {
  echo "[Sequence] Starting IGA-180 first with CPUs=${CPU_SET}, workers=${WORKERS}."
  run_budget 180 "${OUTPUT_180}" "${TEACHER_180}"
  echo "[Sequence] IGA-180 passed its full verification gate; starting IGA-1800."
  run_budget 1800 "${OUTPUT_1800}" "${TEACHER_1800}"
  echo "[Done] IGA-180 and IGA-1800 both completed and passed replay verification."
}

if [[ "${1:-}" == "--sequential-worker" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 --sequential-worker" >&2
    exit 2
  fi
  run_sequence
  exit 0
fi

for required in "${PYTHON}" "${RUNNER}" "${DATASET}" /usr/bin/taskset; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done
for value_name in WORKERS POPULATION GENERATIONS MAX_ATTEMPTS IGA_SEED RUNNER_PASSES RETRY_DELAY_SECONDS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "[Error] ${value_name} must be a non-negative integer." >&2
    exit 1
  fi
done
if (( WORKERS < 1 || POPULATION < 1 || GENERATIONS < 1 || MAX_ATTEMPTS < 1 || RUNNER_PASSES < 1 )); then
  echo "[Error] Worker/search/retry counts must be positive." >&2
  exit 1
fi

"${PYTHON}" - "${CPU_SET}" "${WORKERS}" <<'PY'
import subprocess
import sys


def expand(spec):
    result = set()
    for item in spec.split(","):
        bounds = [int(value) for value in item.split("-", 1)]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


selected = expand(sys.argv[1])
workers = int(sys.argv[2])
rows = subprocess.check_output(
    ["lscpu", "-p=CPU,CORE,SOCKET,ONLINE"], text=True
).splitlines()
siblings = {}
online = set()
for row in rows:
    if not row or row.startswith("#"):
        continue
    cpu, core, socket, is_online = row.split(",")
    if is_online.lower() not in {"y", "yes", "1"}:
        continue
    cpu, core, socket = int(cpu), int(core), int(socket)
    online.add(cpu)
    siblings.setdefault((socket, core), set()).add(cpu)
if selected != online:
    missing = sorted(online - selected)
    extra = sorted(selected - online)
    raise SystemExit(
        "CPU_SET must cover every online logical CPU for sequential full-host "
        f"labeling; missing={missing}, extra={extra}"
    )
if workers > len(selected):
    raise SystemExit(
        f"workers={workers} exceeds selected logical CPUs={len(selected)}"
    )
partial = {
    core: sorted(core_cpus & selected)
    for core, core_cpus in siblings.items()
    if core_cpus & selected and not core_cpus <= selected
}
if partial:
    raise SystemExit(f"CPU_SET contains partial SMT cores: {partial}")
print(
    f"[CPU] {len(selected)} logical CPUs, {len(siblings)} physical cores, "
    f"{workers} IGA workers",
    flush=True,
)
PY

/usr/bin/taskset --cpu-list "${CPU_SET}" /bin/true
if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "[Error] User systemd manager is unavailable." >&2
  exit 1
fi
load_state="$(
  systemctl --user show "${UNIT_SEQUENCE}.service" \
    --property=LoadState --value 2>/dev/null || true
)"
if [[ -n "${load_state}" && "${load_state}" != "not-found" ]]; then
  echo "[Error] Refusing to reuse existing unit ${UNIT_SEQUENCE}.service." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${TEACHER_ROOT}"
systemd-run --user \
  --unit="${UNIT_SEQUENCE}" \
  --collect \
  --same-dir \
  --property=Type=exec \
  --property=Restart=no \
  --property=KillMode=control-group \
  --property=TimeoutStopSec=120 \
  --property=LimitNOFILE=65536 \
  --property="AllowedCPUs=${CPU_SET}" \
  --property="CPUAffinity=${CPU_SET}" \
  --property=CPUWeight=100 \
  --property="StandardOutput=append:${LOG_SEQUENCE}" \
  --property="StandardError=append:${LOG_SEQUENCE}" \
  --setenv="PYTHON=${PYTHON}" \
  --setenv="DATASET=${DATASET}" \
  --setenv="DATASET_TAG=${DATASET_TAG}" \
  --setenv="TEACHER_DATASET_PREFIX=${TEACHER_DATASET_PREFIX}" \
  --setenv="LABEL_VERSION=${LABEL_VERSION}" \
  --setenv="OUTPUT_ROOT=${OUTPUT_ROOT}" \
  --setenv="TEACHER_ROOT=${TEACHER_ROOT}" \
  --setenv="WORKERS=${WORKERS}" \
  --setenv="POPULATION=${POPULATION}" \
  --setenv="GENERATIONS=${GENERATIONS}" \
  --setenv="MAX_ATTEMPTS=${MAX_ATTEMPTS}" \
  --setenv="IGA_SEED=${IGA_SEED}" \
  --setenv="RUNNER_PASSES=${RUNNER_PASSES}" \
  --setenv="RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS}" \
  --setenv="CPU_SET=${CPU_SET}" \
  /bin/bash "$0" --sequential-worker

echo "[Launch] Submitted ${UNIT_SEQUENCE}.service."
echo "[Order] IGA-180 -> full verification gate -> IGA-1800."
echo "[Dataset] tag=${DATASET_TAG}, path=${DATASET}"
echo "[CPU] CPUs=${CPU_SET}, workers=${WORKERS}; no concurrent labeling job."
echo "[Log] ${LOG_SEQUENCE}"
echo "[Output] ${OUTPUT_180}"
echo "[Output] ${OUTPUT_1800}"
