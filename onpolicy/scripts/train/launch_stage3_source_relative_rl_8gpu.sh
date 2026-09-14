#!/usr/bin/env bash
set -euo pipefail

# One complete train+validation process per physical GPU. The factorial order
# balances every binary factor within each NUMA node.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11}"
PHASE="${PHASE:-canary}"
RUN_TAG="${RUN_TAG:-stage3_source_relative_rl_4090_20260904_r1}"
START="${START:-1}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s3sr-r1}"
SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}"
SOURCE_BASELINE="${SOURCE_BASELINE:-${SUITE_DIR}/artifacts/source_checkpoint_train600.json}"
PREPARER="${SCRIPT_DIR}/prepare_stage3_source_relative_rl.py"
RUNNER="${SCRIPT_DIR}/run_stage3_source_relative_manifest.py"
GATE="${CANARY_GATE:-${SUITE_DIR}/canary_gate.json}"

if [[ "${PHASE}" != "canary" && "${PHASE}" != "wave1" ]]; then
  echo "[Error] PHASE must be canary or wave1." >&2
  exit 2
fi
if [[ "${START}" != "0" && "${START}" != "1" ]]; then
  echo "[Error] START must be 0 or 1." >&2
  exit 2
fi
if [[ ! -f "${SOURCE_BASELINE}" ]]; then
  echo "[Error] Verified C0 train600 baseline is required: ${SOURCE_BASELINE}" >&2
  exit 2
fi
if [[ "${PHASE}" == "wave1" && ! -f "${GATE}" ]]; then
  echo "[Error] Formal Wave 1 requires the 8/8 gate: ${GATE}" >&2
  exit 2
fi

ARMS=(
  U0K0R0 U0K1R1 U1K0R1 U1K1R0
  U0K0R1 U0K1R0 U1K0R0 U1K1R1
)
CPU_SETS=(
  "0-6,64-70"
  "7-13,71-77"
  "14-20,78-84"
  "21-27,85-91"
  "32-38,96-102"
  "39-45,103-109"
  "46-52,110-116"
  "53-59,117-123"
)
NUMA_NODES=(0 0 0 0 1 1 1 1)

mkdir -p \
  "${SUITE_DIR}/commands" \
  "${SUITE_DIR}/records" \
  "${SUITE_DIR}/service_logs" \
  "${SUITE_DIR}/hardware"

manifest_path() {
  local gpu="$1"
  printf '%s/commands/%s_g%s_%s.json' \
    "${SUITE_DIR}" "${PHASE}" "${gpu}" "${ARMS[${gpu}]}"
}

experiment_name() {
  local gpu="$1"
  printf '%s_%s_%s_seed3' "${RUN_TAG}" "${PHASE}" "${ARMS[${gpu}]}"
}

unit_name() {
  local gpu="$1" phase_label
  phase_label="can"
  [[ "${PHASE}" == "wave1" ]] && phase_label="w1"
  printf '%s-%s-g%s' "${UNIT_PREFIX}" "${phase_label}" "${gpu}"
}

for gpu in 0 1 2 3 4 5 6 7; do
  output="$(manifest_path "${gpu}")"
  command=(
    "${PYTHON}" "${PREPARER}"
    --phase "${PHASE}"
    --arm "${ARMS[${gpu}]}"
    --physical-gpu "${gpu}"
    --cpu-affinity "${CPU_SETS[${gpu}]}"
    --numa-node "${NUMA_NODES[${gpu}]}"
    --run-tag "${RUN_TAG}"
    --source-baseline "${SOURCE_BASELINE}"
    --output "${output}"
    --python "${PYTHON}"
  )
  if [[ "${PHASE}" == "wave1" ]]; then
    command+=(--canary-gate "${GATE}")
  fi
  "${command[@]}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHON="${PYTHON}" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    taskset --cpu-list "${CPU_SETS[${gpu}]}" \
    "${PYTHON}" "${RUNNER}" "${output}" \
      --expected-gpu "${gpu}" \
      --expected-cpu-affinity "${CPU_SETS[${gpu}]}" \
      --check-only
done

echo "[Prepared] ${PHASE}: 8 immutable source-relative manifests passed checks."
if [[ "${START}" == "0" ]]; then
  exit 0
fi

online_cpus="$(getconf _NPROCESSORS_ONLN)"
if (( online_cpus < 128 )); then
  echo "[Error] Expected at least 128 online logical CPUs; found ${online_cpus}." >&2
  exit 1
fi

nvidia-smi topo -m > "${SUITE_DIR}/hardware/${PHASE}_nvidia_topology.txt"
mapfile -t gpu_inventory < <(
  nvidia-smi --query-gpu=index,pci.bus_id,memory.total \
    --format=csv,noheader,nounits
)
if (( ${#gpu_inventory[@]} < 8 )); then
  echo "[Error] Eight GPUs are required; found ${#gpu_inventory[@]}." >&2
  exit 1
fi
for gpu in 0 1 2 3 4 5 6 7; do
  IFS=',' read -r observed_index bus memory_mib <<< "${gpu_inventory[${gpu}]}"
  observed_index="${observed_index//[[:space:]]/}"
  bus="${bus//[[:space:]]/}"
  memory_mib="${memory_mib//[[:space:]]/}"
  if [[ "${observed_index}" != "${gpu}" ]] || (( memory_mib < 23000 )); then
    echo "[Error] GPU${gpu} inventory mismatch or VRAM < 23 GiB." >&2
    exit 1
  fi
  short_bus="0000:${bus##*:}"
  if [[ "${bus}" =~ ^00000000:(.*)$ ]]; then
    short_bus="0000:${BASH_REMATCH[1]}"
  elif [[ "${bus}" =~ ^0000:(.*)$ ]]; then
    short_bus="${bus}"
  fi
  short_bus="${short_bus,,}"
  numa_file="/sys/bus/pci/devices/${short_bus}/numa_node"
  if [[ ! -r "${numa_file}" ]]; then
    echo "[Error] Cannot resolve GPU${gpu} NUMA node from ${bus}." >&2
    exit 1
  fi
  observed_numa="$(<"${numa_file}")"
  if [[ "${observed_numa}" != "${NUMA_NODES[${gpu}]}" ]]; then
    echo "[Error] GPU${gpu} NUMA=${observed_numa}; expected ${NUMA_NODES[${gpu}]}" >&2
    exit 1
  fi
done

busy="$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid \
  --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ "${ALLOW_BUSY_GPU}" != "1" && -n "${busy//[[:space:]]/}" ]]; then
  echo "[Error] Refusing to share the eight GPUs with existing compute jobs:" >&2
  echo "${busy}" >&2
  exit 1
fi

for gpu in 0 1 2 3 4 5 6 7; do
  unit="$(unit_name "${gpu}")"
  load_state="$(systemctl --user show "${unit}.service" \
    --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${load_state}" && "${load_state}" != "not-found" ]]; then
    echo "[Error] Refusing to reuse systemd unit ${unit}.service." >&2
    exit 1
  fi
  result_parent="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/$(experiment_name "${gpu}")"
  if [[ -e "${result_parent}" ]]; then
    echo "[Error] Refusing to mix results in ${result_parent}." >&2
    exit 1
  fi
done

runtime="12h"
[[ "${PHASE}" == "wave1" ]] && runtime="96h"
for gpu in 0 1 2 3 4 5 6 7; do
  unit="$(unit_name "${gpu}")"
  manifest="$(manifest_path "${gpu}")"
  log="${SUITE_DIR}/service_logs/${unit}.log"
  record="${SUITE_DIR}/records/${PHASE}_g${gpu}_${ARMS[${gpu}]}.json"
  delay=$((gpu * 15))
  systemd-run --user --collect --unit="${unit}" --same-dir \
    --setenv="CUDA_VISIBLE_DEVICES=${gpu}" \
    --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv="HKBZ_PHYSICAL_GPU=${gpu}" \
    --setenv="PYTHON=${PYTHON}" \
    --setenv=PYTHONHASHSEED=0 \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=1 \
    --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 \
    --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=MALLOC_ARENA_MAX=2 \
    --setenv="MPLCONFIGDIR=/tmp/hkbz-s3sr-mpl-g${gpu}" \
    --property=Type=exec \
    --property=Restart=no \
    --property=KillMode=control-group \
    --property=TimeoutStopSec=180 \
    --property="RuntimeMaxSec=${runtime}" \
    --property=LimitNOFILE=65536 \
    --property=TasksMax=512 \
    --property="AllowedCPUs=${CPU_SETS[${gpu}]}" \
    --property="CPUAffinity=${CPU_SETS[${gpu}]}" \
    --property=NUMAPolicy=bind \
    --property="NUMAMask=${NUMA_NODES[${gpu}]}" \
    --property=MemoryHigh=20G \
    --property=MemoryMax=28G \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    /usr/bin/taskset --cpu-list "${CPU_SETS[${gpu}]}" \
      "${PYTHON}" -u "${RUNNER}" "${manifest}" \
      --expected-gpu "${gpu}" \
      --expected-cpu-affinity "${CPU_SETS[${gpu}]}" \
      --record "${record}" \
      --start-delay "${delay}" >/dev/null
  echo "[Launch] ${unit}.service GPU${gpu} ${ARMS[${gpu}]} CPUs=${CPU_SETS[${gpu}]} NUMA=${NUMA_NODES[${gpu}]} delay=${delay}s"
done

echo "[Started] ${PHASE} has 8 independent source-relative services; no retry."
