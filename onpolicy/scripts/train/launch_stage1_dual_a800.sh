#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia/bin/python}"
SINGLE_LAUNCHER="${ROOT_DIR}/onpolicy/scripts/train/launch_stage1_iga_weekly_suite.sh"

BASE_TAG="${BASE_TAG:-stage1_dual_a800_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG_GPU0="${RUN_TAG_GPU0:-${BASE_TAG}_gpu0}"
RUN_TAG_GPU1="${RUN_TAG_GPU1:-${BASE_TAG}_gpu1}"
FORMAL_SEEDS_GPU0="${FORMAL_SEEDS_GPU0:-1}"
FORMAL_SEEDS_GPU1="${FORMAL_SEEDS_GPU1:-2}"
RESUME_GPU0="${RESUME_GPU0:-0}"
RESUME_GPU1="${RESUME_GPU1:-0}"
CPU_SET_GPU0="${CPU_SET_GPU0:-0-35,72-107}"
CPU_SET_GPU1="${CPU_SET_GPU1:-36-71,108-143}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-30}"

for required in "${PYTHON}" "${SINGLE_LAUNCHER}" /usr/bin/taskset; do
  if [[ ! -x "${required}" ]]; then
    echo "[Error] Missing executable: ${required}" >&2
    exit 1
  fi
done
if [[ "${RUN_TAG_GPU0}" == "${RUN_TAG_GPU1}" ]]; then
  echo "[Error] The two run tags must be different." >&2
  exit 1
fi
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
overlap = sorted(left & right)
if overlap:
    raise SystemExit(f"CPU sets overlap: {overlap}")
if len(left) != 72 or len(right) != 72:
    raise SystemExit(
        f"Expected 72 logical CPUs per A800: got {len(left)} and {len(right)}"
    )
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

echo "[Launch] GPU0: tag=${RUN_TAG_GPU0}, seeds=${FORMAL_SEEDS_GPU0}, CPUs=${CPU_SET_GPU0}"
env \
  RUN_TAG="${RUN_TAG_GPU0}" \
  GPU=0 \
  CPU_AFFINITY="${CPU_SET_GPU0}" \
  FORMAL_SEEDS="${FORMAL_SEEDS_GPU0}" \
  RESUME="${RESUME_GPU0}" \
  /bin/bash "${SINGLE_LAUNCHER}"

echo "[Launch] Waiting ${START_DELAY_SECONDS}s before GPU1 initialization."
sleep "${START_DELAY_SECONDS}"

echo "[Launch] GPU1: tag=${RUN_TAG_GPU1}, seeds=${FORMAL_SEEDS_GPU1}, CPUs=${CPU_SET_GPU1}"
env \
  RUN_TAG="${RUN_TAG_GPU1}" \
  GPU=1 \
  CPU_AFFINITY="${CPU_SET_GPU1}" \
  FORMAL_SEEDS="${FORMAL_SEEDS_GPU1}" \
  RESUME="${RESUME_GPU1}" \
  /bin/bash "${SINGLE_LAUNCHER}"

echo "[Done] Both isolated Stage-1 services were submitted."
