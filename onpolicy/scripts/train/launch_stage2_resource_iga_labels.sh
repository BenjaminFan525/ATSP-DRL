#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python}"
RUNNER="${RUNNER:-${ROOT_DIR}/onpolicy/envs/HKBZ/experiment/generate_stage2_resource_iga_labels.py}"
DATASET="${DATASET:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage1_departure_reward_formal_dual_20260816_r1_formal_P5_team_time_potential_fixed_seed3/run1/models/checkpoint_Best.pt}"
SOURCE_COMMAND="${SOURCE_COMMAND:-${ROOT_DIR}/result/hkbz_train_logs/stage1_departure_reward_formal_dual_20260816_r1/commands/formal_P5.json}"
SOURCE_COMMAND_KEY="${SOURCE_COMMAND_KEY:-P5_team_time_potential_fixed_seed3}"
HANDOFF="${HANDOFF:-${ROOT_DIR}/onpolicy/config/stage1_m2_handoff.json}"

RUN_TAG="${RUN_TAG:-stage2_resource_iga_p5_seed3_case_parallel_20260818_r3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
OUTPUT_180="${OUTPUT_ROOT}/iga180"
OUTPUT_1800="${OUTPUT_ROOT}/iga1800"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s2-iga-p5s3-case-20260818-r3}"
POPULATION="${POPULATION:-20}"
CASE_BATCH_SIZE="${CASE_BATCH_SIZE:-16}"
MAX_GENERATIONS="${MAX_GENERATIONS:-100000}"
IGA_SEED="${IGA_SEED:-20260818}"
MAX_STEPS="${MAX_STEPS:-4000}"
MAX_CASES="${MAX_CASES:-0}"
EVALUATION_TAU="${EVALUATION_TAU:-0.3}"
LOOKAHEAD_MARGIN="${LOOKAHEAD_MARGIN:-60}"
POLL_SECONDS="${POLL_SECONDS:-30}"
SHARD_COUNT=8

# Four independent graph-collation / inference pipelines share each GPU.
# Every pipeline owns both SMT siblings of nine physical cores and advances
# different cases.  The measured 6x18 layout left about fourteen physical-core
# equivalents idle; 8x16 fills that gap without crossing NUMA boundaries.
SHARD_GPUS=(0 0 0 0 1 1 1 1)
SHARD_NUMAS=(0 0 0 0 1 1 1 1)
SHARD_CPUS=(
  "0-8,72-80"
  "9-17,81-89"
  "18-26,90-98"
  "27-35,99-107"
  "36-44,108-116"
  "45-53,117-125"
  "54-62,126-134"
  "63-71,135-143"
)

common_args() {
  printf '%s\n' \
    --dataset-dir "${DATASET}" \
    --checkpoint "${CHECKPOINT}" \
    --source-command "${SOURCE_COMMAND}" \
    --source-command-key "${SOURCE_COMMAND_KEY}" \
    --handoff "${HANDOFF}" \
    --device cuda:0 \
    --evaluation-tau "${EVALUATION_TAU}" \
    --device-lookahead-safety-margin "${LOOKAHEAD_MARGIN}" \
    --population "${POPULATION}" \
    --case-batch-size "${CASE_BATCH_SIZE}" \
    --max-generations "${MAX_GENERATIONS}" \
    --seed "${IGA_SEED}" \
    --max-steps "${MAX_STEPS}" \
    --max-cases "${MAX_CASES}"
}

run_stage() {
  local shard="$1"
  local cpu_set="$2"
  local output_dir="$3"
  local additional_budget="$4"
  local cumulative_budget="$5"
  shift 5
  local args=()
  mapfile -t args < <(common_args)
  /usr/bin/taskset --cpu-list "${cpu_set}" \
    "${PYTHON}" -u "${RUNNER}" \
      "${args[@]}" \
      --output-dir "${output_dir}" \
      --time-budget-seconds "${additional_budget}" \
      --cumulative-budget-seconds "${cumulative_budget}" \
      --shard-index "${shard}" \
      --shard-count "${SHARD_COUNT}" \
      "$@"
}

worker() {
  local shard="$1"
  local cpu_set="$2"
  echo "[Worker] shard=${shard} starting nested IGA-180 -> IGA-1800"
  run_stage "${shard}" "${cpu_set}" "${OUTPUT_180}" 180 180
  echo "[Worker] shard=${shard} IGA-180 complete and replay verified"
  run_stage \
    "${shard}" "${cpu_set}" "${OUTPUT_1800}" 1620 1800 \
    --warm-start-dir "${OUTPUT_180}" \
    --warm-start-budget-seconds 180
  echo "[Worker] shard=${shard} IGA-1800 complete and replay verified"
}

completed_shards() {
  local output_dir="$1"
  local count=0
  local shard
  for ((shard = 0; shard < SHARD_COUNT; shard++)); do
    if [[ -s "${output_dir}/workers/shard_$(printf '%02d' "${shard}").json" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%d\n' "${count}"
}

summarize_stage() {
  local output_dir="$1"
  local additional_budget="$2"
  local cumulative_budget="$3"
  shift 3
  local args=()
  mapfile -t args < <(common_args)
  /usr/bin/taskset --cpu-list 0 \
    "${PYTHON}" -u "${RUNNER}" \
      "${args[@]}" \
      --output-dir "${output_dir}" \
      --time-budget-seconds "${additional_budget}" \
      --cumulative-budget-seconds "${cumulative_budget}" \
      --summarize-only \
      "$@"
}

coordinator() {
  local summarized_180=0
  local done_180 done_1800 shard unit active_state
  while true; do
    done_180="$(completed_shards "${OUTPUT_180}")"
    done_1800="$(completed_shards "${OUTPUT_1800}")"
    if (( done_180 == SHARD_COUNT && summarized_180 == 0 )); then
      summarize_stage "${OUTPUT_180}" 180 180
      summarized_180=1
      echo "[Coordinator] IGA-180: 600/600 teachers verified"
    fi
    if (( done_1800 == SHARD_COUNT )); then
      summarize_stage \
        "${OUTPUT_1800}" 1620 1800 \
        --warm-start-dir "${OUTPUT_180}" \
        --warm-start-budget-seconds 180
      echo "[Coordinator] IGA-1800: 600/600 teachers verified"
      return 0
    fi
    for ((shard = 0; shard < SHARD_COUNT; shard++)); do
      unit="${UNIT_PREFIX}-shard${shard}.service"
      active_state="$(systemctl --user show "${unit}" --property=ActiveState --value 2>/dev/null || true)"
      if [[ "${active_state}" == "failed" ]]; then
        echo "[Coordinator] ${unit} failed before full verification" >&2
        return 1
      fi
    done
    echo "[Coordinator] IGA-180 shards=${done_180}/${SHARD_COUNT}; IGA-1800 shards=${done_1800}/${SHARD_COUNT}"
    sleep "${POLL_SECONDS}"
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 --worker SHARD CPU_SET" >&2
    exit 2
  fi
  worker "$2" "$3"
  exit 0
fi

if [[ "${1:-}" == "--coordinator" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 --coordinator" >&2
    exit 2
  fi
  coordinator
  exit 0
fi

if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

for required in "${PYTHON}" "${RUNNER}" "${DATASET}" "${CHECKPOINT}" "${SOURCE_COMMAND}" "${HANDOFF}" /usr/bin/taskset /usr/bin/nvidia-smi; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] Missing required input: ${required}" >&2
    exit 1
  fi
done
if ! [[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ && "${UNIT_PREFIX}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[Error] RUN_TAG and UNIT_PREFIX must be path/unit safe." >&2
  exit 1
fi
for value_name in POPULATION CASE_BATCH_SIZE MAX_GENERATIONS IGA_SEED MAX_STEPS POLL_SECONDS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "[Error] ${value_name} must be a positive integer." >&2
    exit 1
  fi
done
if (( POPULATION < 2 )); then
  echo "[Error] POPULATION must be at least 2." >&2
  exit 1
fi
if (( CASE_BATCH_SIZE > 16 )); then
  echo "[Error] CASE_BATCH_SIZE must not exceed 16 in an 18-CPU shard." >&2
  exit 1
fi
if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "[Error] User systemd manager is unavailable." >&2
  exit 1
fi
compute_pids="$(/usr/bin/nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "${compute_pids}" ]]; then
  echo "[Error] Refusing to launch while GPU compute processes exist: ${compute_pids//$'\n'/,}" >&2
  exit 1
fi

# Fail closed if the topology has changed or any physical/SMT CPU is reused.
"${PYTHON}" - "${SHARD_CPUS[@]}" <<'PY'
import subprocess
import sys


def expand(spec):
    result = set()
    for item in spec.split(","):
        bounds = [int(value) for value in item.split("-", 1)]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


specs = sys.argv[1:]
if len(specs) != 8:
    raise SystemExit("exactly eight GPU-local CPU sets are required")
sets = [expand(spec) for spec in specs]
if any(len(selected) != 18 for selected in sets):
    raise SystemExit("every shard must own exactly 18 logical CPUs")
if len(set().union(*sets)) != sum(map(len, sets)):
    raise SystemExit("CPU sets overlap")
rows = subprocess.check_output(
    ["lscpu", "-p=CPU,CORE,SOCKET,NODE,ONLINE"], text=True
).splitlines()
cpu_to_core = {}
online = set()
for row in rows:
    if not row or row.startswith("#"):
        continue
    cpu, core, socket, node, is_online = row.split(",")
    if is_online.lower() not in {"y", "yes", "1"}:
        continue
    cpu = int(cpu)
    online.add(cpu)
    cpu_to_core[cpu] = (int(socket), int(core), int(node))
if set().union(*sets) != online:
    raise SystemExit("eight CPU sets must cover every online logical CPU")
for index, selected in enumerate(sets):
    physical = {(cpu_to_core[cpu][0], cpu_to_core[cpu][1]) for cpu in selected}
    nodes = {cpu_to_core[cpu][2] for cpu in selected}
    if len(physical) != 9 or len(nodes) != 1:
        raise SystemExit(
            f"shard {index} is not nine complete physical cores on one NUMA node"
        )
print("[Topology] 8 disjoint shards, 72 physical / 144 logical CPUs", flush=True)
PY

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_180}" "${OUTPUT_1800}"
for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  unit="${UNIT_PREFIX}-shard${shard}"
  load_state="$(systemctl --user show "${unit}.service" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${load_state}" && "${load_state}" != "not-found" ]]; then
    echo "[Error] Refusing to reuse existing unit ${unit}.service." >&2
    exit 1
  fi
done
coordinator_unit="${UNIT_PREFIX}-coordinator"
load_state="$(systemctl --user show "${coordinator_unit}.service" --property=LoadState --value 2>/dev/null || true)"
if [[ -n "${load_state}" && "${load_state}" != "not-found" ]]; then
  echo "[Error] Refusing to reuse existing unit ${coordinator_unit}.service." >&2
  exit 1
fi

for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  gpu="${SHARD_GPUS[$shard]}"
  numa="${SHARD_NUMAS[$shard]}"
  cpu_set="${SHARD_CPUS[$shard]}"
  unit="${UNIT_PREFIX}-shard${shard}"
  log_path="${OUTPUT_ROOT}/logs/shard_$(printf '%02d' "${shard}").log"
  tmp_dir="/tmp/${UNIT_PREFIX}-shard${shard}"
  mpl_dir="${tmp_dir}/matplotlib"
  mkdir -p "${mpl_dir}"
  systemd-run --user \
    --unit="${unit}" \
    --collect \
    --same-dir \
    --property=Type=exec \
    --property=Restart=on-failure \
    --property=RestartSec=30 \
    --property=KillMode=control-group \
    --property=TimeoutStopSec=120 \
    --property=LimitNOFILE=262144 \
    --property=TasksMax=512 \
    --property="AllowedCPUs=${cpu_set}" \
    --property="CPUAffinity=${cpu_set}" \
    --property=CPUWeight=100 \
    --property=NUMAPolicy=preferred \
    --property="NUMAMask=${numa}" \
    --property="StandardOutput=append:${log_path}" \
    --property="StandardError=append:${log_path}" \
    --setenv="PYTHON=${PYTHON}" \
    --setenv="RUNNER=${RUNNER}" \
    --setenv="DATASET=${DATASET}" \
    --setenv="CHECKPOINT=${CHECKPOINT}" \
    --setenv="SOURCE_COMMAND=${SOURCE_COMMAND}" \
    --setenv="SOURCE_COMMAND_KEY=${SOURCE_COMMAND_KEY}" \
    --setenv="HANDOFF=${HANDOFF}" \
    --setenv="RUN_TAG=${RUN_TAG}" \
    --setenv="OUTPUT_ROOT=${OUTPUT_ROOT}" \
    --setenv="UNIT_PREFIX=${UNIT_PREFIX}" \
    --setenv="POPULATION=${POPULATION}" \
    --setenv="CASE_BATCH_SIZE=${CASE_BATCH_SIZE}" \
    --setenv="MAX_GENERATIONS=${MAX_GENERATIONS}" \
    --setenv="IGA_SEED=${IGA_SEED}" \
    --setenv="MAX_STEPS=${MAX_STEPS}" \
    --setenv="MAX_CASES=${MAX_CASES}" \
    --setenv="EVALUATION_TAU=${EVALUATION_TAU}" \
    --setenv="LOOKAHEAD_MARGIN=${LOOKAHEAD_MARGIN}" \
    --setenv="POLL_SECONDS=${POLL_SECONDS}" \
    --setenv="CUDA_VISIBLE_DEVICES=${gpu}" \
    --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=PYTHONUNBUFFERED=1 \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=OMP_NUM_THREADS=1 \
    --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 \
    --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512" \
    --setenv="MPLCONFIGDIR=${mpl_dir}" \
    --setenv="TMPDIR=${tmp_dir}" \
    /bin/bash "$0" --worker "${shard}" "${cpu_set}"
  echo "[Launch] ${unit}.service GPU=${gpu} NUMA=${numa} CPUs=${cpu_set}"
done

coordinator_log="${OUTPUT_ROOT}/logs/coordinator.log"
systemd-run --user \
  --unit="${coordinator_unit}" \
  --collect \
  --same-dir \
  --property=Type=exec \
  --property=Restart=no \
  --property=KillMode=control-group \
  --property="AllowedCPUs=0,72" \
  --property="CPUAffinity=0,72" \
  --property="StandardOutput=append:${coordinator_log}" \
  --property="StandardError=append:${coordinator_log}" \
  --setenv="PYTHON=${PYTHON}" \
  --setenv="RUNNER=${RUNNER}" \
  --setenv="DATASET=${DATASET}" \
  --setenv="CHECKPOINT=${CHECKPOINT}" \
  --setenv="SOURCE_COMMAND=${SOURCE_COMMAND}" \
  --setenv="SOURCE_COMMAND_KEY=${SOURCE_COMMAND_KEY}" \
  --setenv="HANDOFF=${HANDOFF}" \
  --setenv="RUN_TAG=${RUN_TAG}" \
  --setenv="OUTPUT_ROOT=${OUTPUT_ROOT}" \
  --setenv="UNIT_PREFIX=${UNIT_PREFIX}" \
  --setenv="POPULATION=${POPULATION}" \
  --setenv="CASE_BATCH_SIZE=${CASE_BATCH_SIZE}" \
  --setenv="MAX_GENERATIONS=${MAX_GENERATIONS}" \
  --setenv="IGA_SEED=${IGA_SEED}" \
  --setenv="MAX_STEPS=${MAX_STEPS}" \
  --setenv="MAX_CASES=${MAX_CASES}" \
  --setenv="EVALUATION_TAU=${EVALUATION_TAU}" \
  --setenv="LOOKAHEAD_MARGIN=${LOOKAHEAD_MARGIN}" \
  --setenv="POLL_SECONDS=${POLL_SECONDS}" \
  --setenv=OMP_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 \
  /bin/bash "$0" --coordinator

echo "[Launch] ${coordinator_unit}.service"
echo "[Policy] P5 seed3 validation-Best: ${CHECKPOINT}"
echo "[Dataset] ${DATASET} (600 cases)"
echo "[Parallelism] 2 GPUs x 4 isolated CPU shards x ${CASE_BATCH_SIZE} independent cases; population=${POPULATION} is sequential within each case"
echo "[Budgets] nested IGA-180 then +1620s refinement = cumulative IGA-1800"
echo "[Output] ${OUTPUT_ROOT}"
