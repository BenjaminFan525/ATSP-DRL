#!/usr/bin/env bash
set -euo pipefail

# Two-phase Stage3 full-joint IGA orchestration:
#   1. use one NUMA node opportunistically while Wave-3 is still in shared BC;
#   2. pre-empt before five-way PPO and resume with all 144 logical CPUs only
#      after every Wave-3 arm has completed.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python}"
RUNNER="${ROOT_DIR}/onpolicy/envs/HKBZ/experiment/generate_stage3_joint_iga_labels.py"
DATASET="${DATASET:-${ROOT_DIR}/onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train}"

S1_180="${S1_180:-${ROOT_DIR}/result/hkbz_train_logs/iga_teachers/fjspv3_t600_train_s1_progressive_departure_r014_20260812_v2_p20_g20_t180_a5_s1}"
S2_180="${S2_180:-${ROOT_DIR}/result/hkbz_train_logs/stage2_resource_iga_p5_seed3_case_parallel_20260818_r3/iga180}"
RUN_TAG="${RUN_TAG:-stage3_joint_iga_20260821_r1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}}"
OUTPUT_CANARY="${OUTPUT_ROOT}/canary"
OUTPUT_180="${OUTPUT_ROOT}/iga180"
OUTPUT_1800="${OUTPUT_ROOT}/iga1800"

UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s3-iga-20260821-r1}"
OPPORTUNISTIC_UNIT="${UNIT_PREFIX}-opportunistic"
MONITOR_UNIT="${UNIT_PREFIX}-preempt-monitor"
POSTTRAIN_UNIT="${UNIT_PREFIX}-posttrain"
OPPORTUNISTIC_CPUS="${OPPORTUNISTIC_CPUS:-36-71,108-143}"
ALL_CPUS="${ALL_CPUS:-0-143}"
OPPORTUNISTIC_WORKERS="${OPPORTUNISTIC_WORKERS:-72}"
FULL_WORKERS="${FULL_WORKERS:-144}"
POPULATION="${POPULATION:-20}"
MAX_GENERATIONS="${MAX_GENERATIONS:-100000}"
MAX_STEPS="${MAX_STEPS:-4000}"
IGA_SEED="${IGA_SEED:-20260821}"
POLL_SECONDS="${POLL_SECONDS:-5}"

WAVE3_TAG="${WAVE3_TAG:-stage2_critical_wave3_20260821_r1}"
WAVE3_PREFIX="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/${WAVE3_TAG}_critical_wave"
METHODS=(
  C0_cmax_control
  C1_team_time
  C2_critical_slack
  C3_slack_arrival
  C4_slack_arrival_tail
)

status_path() {
  local method="$1"
  printf '%s_%s_seed1/run1/run_status.json\n' "${WAVE3_PREFIX}" "${method}"
}

device_bc_path() {
  printf '%s_%s_seed1/run1/models/checkpoint_DeviceBC.pt\n' \
    "${WAVE3_PREFIX}" "${METHODS[0]}"
}

common_args() {
  printf '%s\n' \
    --dataset-dir "${DATASET}" \
    --population "${POPULATION}" \
    --max-generations "${MAX_GENERATIONS}" \
    --max-steps "${MAX_STEPS}" \
    --max-plane-agents 24 \
    --max-device-num 80 \
    --device-lookahead-safety-margin 60 \
    --seed "${IGA_SEED}"
}

run_iga180() {
  local cpu_set="$1"
  local workers="$2"
  local args=()
  mapfile -t args < <(common_args)
  /usr/bin/taskset --cpu-list "${cpu_set}" \
    "${PYTHON}" -u "${RUNNER}" "${args[@]}" \
      --output-dir "${OUTPUT_180}" \
      --plane-warm-dir "${S1_180}" \
      --resource-warm-dir "${S2_180}" \
      --time-budget-seconds 180 \
      --cumulative-budget-seconds 180 \
      --workers "${workers}"
}

run_iga1800() {
  local cpu_set="$1"
  local workers="$2"
  local args=()
  mapfile -t args < <(common_args)
  /usr/bin/taskset --cpu-list "${cpu_set}" \
    "${PYTHON}" -u "${RUNNER}" "${args[@]}" \
      --output-dir "${OUTPUT_1800}" \
      --warm-start-dir "${OUTPUT_180}" \
      --warm-start-budget-seconds 180 \
      --time-budget-seconds 1620 \
      --cumulative-budget-seconds 1800 \
      --workers "${workers}"
}

training_cpu_phase_started() {
  local status
  if [[ -s "$(device_bc_path)" ]]; then
    return 0
  fi
  for method in "${METHODS[@]:1}"; do
    status="$(status_path "${method}")"
    if [[ -s "${status}" ]]; then
      return 0
    fi
  done
  status="$(status_path "${METHODS[0]}")"
  if [[ -s "${status}" ]] && ! "${PYTHON}" - "${status}" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("phase") == "resource_bc_warmup" else 1)
PY
  then
    return 0
  fi
  return 1
}

all_trainers_completed() {
  local paths=()
  local method
  for method in "${METHODS[@]}"; do
    paths+=("$(status_path "${method}")")
  done
  "${PYTHON}" - "${paths[@]}" <<'PY'
import json, pathlib, sys
for value in sys.argv[1:]:
    path = pathlib.Path(value)
    if not path.is_file():
        raise SystemExit(1)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise SystemExit(1)
    if payload.get("status") != "completed":
        raise SystemExit(1)
    if payload.get("phase") != "resource_joint_completed":
        raise SystemExit(1)
raise SystemExit(0)
PY
}

opportunistic() {
  if training_cpu_phase_started; then
    echo "[Stage3IGA] Wave-3 PPO/admission has begun; skip opportunistic phase."
    return 0
  fi
  local args=()
  mapfile -t args < <(common_args)
  echo "[Stage3IGA] four-case full-joint canary on CPUs=${OPPORTUNISTIC_CPUS}"
  /usr/bin/taskset --cpu-list "${OPPORTUNISTIC_CPUS}" \
    "${PYTHON}" -u "${RUNNER}" "${args[@]}" \
      --output-dir "${OUTPUT_CANARY}" \
      --plane-warm-dir "${S1_180}" \
      --resource-warm-dir "${S2_180}" \
      --time-budget-seconds 5 \
      --cumulative-budget-seconds 5 \
      --population 4 --max-generations 4 \
      --workers 4 --max-cases 4
  if training_cpu_phase_started; then
    echo "[Stage3IGA] training claimed CPU after canary; defer IGA-180."
    return 0
  fi
  echo "[Stage3IGA] canary passed; start interruptible IGA-180 with ${OPPORTUNISTIC_WORKERS} workers"
  run_iga180 "${OPPORTUNISTIC_CPUS}" "${OPPORTUNISTIC_WORKERS}"
}

monitor() {
  echo "[Stage3IGA] monitor waits for DeviceBC/PPO admission."
  while ! training_cpu_phase_started; do
    if ! systemctl --user is-active --quiet "${OPPORTUNISTIC_UNIT}.service"; then
      echo "[Stage3IGA] opportunistic service finished before pre-emption."
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
  echo "[Stage3IGA] Wave-3 needs CPU; pre-empt ${OPPORTUNISTIC_UNIT}.service"
  systemctl --user stop "${OPPORTUNISTIC_UNIT}.service" || true
}

posttrain() {
  echo "[Stage3IGA] post-train worker waiting for all five Wave-3 arms."
  while ! all_trainers_completed; do
    sleep 30
  done
  echo "[Stage3IGA] Wave-3 complete; resume IGA-180 on all ${FULL_WORKERS} logical CPUs."
  systemctl --user stop "${OPPORTUNISTIC_UNIT}.service" >/dev/null 2>&1 || true
  run_iga180 "${ALL_CPUS}" "${FULL_WORKERS}"
  echo "[Stage3IGA] IGA-180 verified for train600; start nested IGA-1800."
  run_iga1800 "${ALL_CPUS}" "${FULL_WORKERS}"
  echo "[Stage3IGA] IGA-1800 verified for train600."
}

launch_service() {
  local unit="$1"
  local cpu_set="$2"
  local weight="$3"
  local nice="$4"
  local mode="$5"
  local restart="$6"
  local log="${OUTPUT_ROOT}/logs/${unit}.log"
  systemd-run --user --unit="${unit}" --collect --same-dir \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=MALLOC_ARENA_MAX=2 \
    --setenv="MPLCONFIGDIR=${OUTPUT_ROOT}/mplconfig" \
    --property=Type=exec --property="Restart=${restart}" \
    --property=RestartSec=60 --property=KillMode=control-group \
    --property=TimeoutStopSec=20 --property=LimitNOFILE=262144 \
    --property=TasksMax=2048 --property="AllowedCPUs=${cpu_set}" \
    --property="CPUAffinity=${cpu_set}" --property="CPUWeight=${weight}" \
    --property="Nice=${nice}" --property=IOSchedulingClass=idle \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /usr/bin/bash "${BASH_SOURCE[0]}" "--${mode}" >/dev/null
  echo "[Launch] ${unit}.service mode=${mode} CPUs=${cpu_set} weight=${weight} nice=${nice}"
}

case "${1:-}" in
  --opportunistic) opportunistic; exit $? ;;
  --monitor) monitor; exit $? ;;
  --posttrain) posttrain; exit $? ;;
  "") ;;
  *) echo "Usage: $0 [--opportunistic|--monitor|--posttrain]" >&2; exit 2 ;;
esac

for required in "${PYTHON}" "${RUNNER}" "${DATASET}" "${S1_180}" \
  "${S2_180}" /usr/bin/taskset /usr/bin/bash; do
  if [[ ! -e "${required}" ]]; then
    echo "[Error] missing required input: ${required}" >&2
    exit 1
  fi
done
if [[ "$(nproc)" -ne 144 ]]; then
  echo "[Error] expected 144 online logical CPUs, found $(nproc)." >&2
  exit 1
fi
if [[ "${OPPORTUNISTIC_WORKERS}" -lt 72 || "${OPPORTUNISTIC_WORKERS}" -gt 96 ]]; then
  echo "[Error] opportunistic workers must be in [72, 96]." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/mplconfig" \
  "${OUTPUT_CANARY}" "${OUTPUT_180}" "${OUTPUT_1800}"
for unit in "${OPPORTUNISTIC_UNIT}" "${MONITOR_UNIT}" "${POSTTRAIN_UNIT}"; do
  state="$(systemctl --user show "${unit}.service" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${state}" && "${state}" != "not-found" ]]; then
    echo "[Error] refusing to reuse existing unit ${unit}.service (${state})." >&2
    exit 1
  fi
done

# Register the durable continuation first; a controller interruption cannot
# lose the all-CPU post-training phase.
launch_service "${POSTTRAIN_UNIT}" "${ALL_CPUS}" 10 10 posttrain on-failure
launch_service "${OPPORTUNISTIC_UNIT}" "${OPPORTUNISTIC_CPUS}" 10 10 opportunistic no
launch_service "${MONITOR_UNIT}" "0" 10 10 monitor no
echo "[Stage3IGA] two-phase schedule registered under ${OUTPUT_ROOT}"
