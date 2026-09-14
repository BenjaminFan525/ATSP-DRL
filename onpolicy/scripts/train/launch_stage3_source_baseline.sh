#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11}"
RUN_TAG="${RUN_TAG:-stage3_source_relative_rl_4090_20260904_r1}"
START="${START:-1}"
GPU="${GPU:-0}"
CPU_SET="${CPU_SET:-0-31,64-95}"
NUMA_NODE="${NUMA_NODE:-0}"
EVAL_WORKERS="${EVAL_WORKERS:-30}"
UNIT="${UNIT:-hkbz-s3sr-r1-baseline}"
SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}"
OUTPUT="${SOURCE_BASELINE:-${SUITE_DIR}/artifacts/source_checkpoint_train600.json}"
OUTPUT_DIR="${SUITE_DIR}/source_baseline_eval"
STATUS="${SUITE_DIR}/source_baseline_status.json"
LOG="${SUITE_DIR}/service_logs/${UNIT}.log"
BUILDER="${SCRIPT_DIR}/build_stage3_source_baseline.py"

if [[ "${START}" != "0" && "${START}" != "1" ]]; then
  echo "[Error] START must be 0 or 1." >&2
  exit 2
fi
if [[ "${GPU}" != "0" || "${NUMA_NODE}" != "0" ]]; then
  echo "[Error] The audited baseline lane is GPU0/NUMA0." >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "[Error] Python environment is unavailable: ${PYTHON}" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" || -e "${OUTPUT_DIR}" || -e "${STATUS}" ]]; then
  echo "[Error] Refusing to overwrite an existing source-baseline artifact." >&2
  exit 1
fi

mkdir -p "${SUITE_DIR}/artifacts" "${SUITE_DIR}/service_logs" "${SUITE_DIR}/hardware"
command=(
  "${PYTHON}" -u "${BUILDER}"
  --output-dir "${OUTPUT_DIR}"
  --output "${OUTPUT}"
  --status "${STATUS}"
  --expected-cases 600
  --n-eval-rollout-threads "${EVAL_WORKERS}"
  --eval-partition-seed 20260803
  --heartbeat-seconds 30
)
printf '[Prepared]'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "${START}" == "0" ]]; then
  exit 0
fi

nvidia-smi topo -m > "${SUITE_DIR}/hardware/source_baseline_nvidia_topology.txt"
memory_mib="$(nvidia-smi --id="${GPU}" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
if (( memory_mib < 23000 )); then
  echo "[Error] GPU${GPU} has less than 23 GiB VRAM." >&2
  exit 1
fi
busy="$(nvidia-smi --id="${GPU}" --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -n "${busy//[[:space:]]/}" ]]; then
  echo "[Error] Refusing to share GPU${GPU} with existing compute jobs:" >&2
  echo "${busy}" >&2
  exit 1
fi
load_state="$(systemctl --user show "${UNIT}.service" --property=LoadState --value 2>/dev/null || true)"
if [[ -n "${load_state}" && "${load_state}" != "not-found" ]]; then
  echo "[Error] Refusing to reuse systemd unit ${UNIT}.service." >&2
  exit 1
fi

systemd-run --user --collect --unit="${UNIT}" --same-dir \
  --setenv="CUDA_VISIBLE_DEVICES=${GPU}" \
  --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv="PYTHON=${PYTHON}" \
  --setenv=PYTHONHASHSEED=0 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --setenv=OMP_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 \
  --setenv=MALLOC_ARENA_MAX=2 \
  --setenv=MPLCONFIGDIR=/tmp/hkbz-s3sr-baseline-mpl \
  --property=Type=exec \
  --property=Restart=no \
  --property=KillMode=control-group \
  --property=TimeoutStopSec=180 \
  --property=RuntimeMaxSec=12h \
  --property=LimitNOFILE=65536 \
  --property=TasksMax=512 \
  --property="AllowedCPUs=${CPU_SET}" \
  --property="CPUAffinity=${CPU_SET}" \
  --property=NUMAPolicy=bind \
  --property="NUMAMask=${NUMA_NODE}" \
  --property=MemoryHigh=64G \
  --property=MemoryMax=96G \
  --property="StandardOutput=append:${LOG}" \
  --property="StandardError=append:${LOG}" \
  /usr/bin/taskset --cpu-list "${CPU_SET}" "${command[@]}" >/dev/null

echo "[Started] ${UNIT}.service on GPU${GPU}; status=${STATUS}; no automatic retry."
