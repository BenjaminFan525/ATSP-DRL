#!/usr/bin/env bash
set -euo pipefail

# Wait for exclusive GPU0 ownership, execute the four-lane graph=1000 memory
# canary, then start and supervise the scientific G0-G3 screen.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/launch_stage3_gradient_conflict_four_gpu0.sh"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python3.11}"
RESULTS_ROOT="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo"
POLL_SECONDS="${POLL_SECONDS:-30}"
FREE_SAMPLES_REQUIRED="${FREE_SAMPLES_REQUIRED:-3}"
SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-0}"
CANARY_RUN_TAG="${CANARY_RUN_TAG:-stage3_gradient_conflict_canary_20260831_r1}"
CANARY_UNIT_PREFIX="${CANARY_UNIT_PREFIX:-hkbz-s3grad-canary-r1-g0}"
FORMAL_RUN_TAG="${FORMAL_RUN_TAG:-stage3_gradient_conflict_20260831_r1}"
FORMAL_UNIT_PREFIX="${FORMAL_UNIT_PREFIX:-hkbz-s3grad-r1-g0}"

gpu_compute_pids() {
  nvidia-smi --id=0 --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}

wait_for_exclusive_gpu0() {
  local free_samples=0 pids last_report=0
  if [[ "${SKIP_GPU_WAIT}" == 1 ]]; then
    echo "[GPUWait] Exclusive wait bypassed by explicit user request."
    return 0
  fi
  while (( free_samples < FREE_SAMPLES_REQUIRED )); do
    pids="$(gpu_compute_pids || true)"
    if [[ -z "${pids}" ]]; then
      free_samples=$((free_samples + 1))
      echo "[GPUWait] GPU0 free sample ${free_samples}/${FREE_SAMPLES_REQUIRED}"
    else
      free_samples=0
      if (( SECONDS - last_report >= 300 || last_report == 0 )); then
        echo "[GPUWait] GPU0 still occupied by PIDs: $(tr '\n' ' ' <<<"${pids}")"
        last_report=${SECONDS}
      fi
    fi
    (( free_samples >= FREE_SAMPLES_REQUIRED )) || sleep "${POLL_SECONDS}"
  done
}

validate_canary() {
  local method status_path status
  for method in G0 G1 G2 G3; do
    status_path="${RESULTS_ROOT}/${CANARY_RUN_TAG}_gradient_conflict_memory_canary_${method}_seed1/run1/run_status.json"
    [[ -f "${status_path}" ]] || {
      echo "[Error] Missing canary status: ${status_path}" >&2
      return 1
    }
    status="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${status_path}")"
    [[ "${status}" == completed ]] || {
      echo "[Error] Canary ${method} ended with status=${status}" >&2
      return 1
    }
    [[ -f "${status_path%/run_status.json}/models/checkpoint_Epoch1.pt" ]] || {
      echo "[Error] Canary ${method} produced no Epoch1 checkpoint." >&2
      return 1
    }
  done
  echo "[Canary] G0-G3 completed graph=1000 unfrozen updates without OOM."
}

main() {
  [[ "${POLL_SECONDS}" =~ ^[0-9]+$ && "${FREE_SAMPLES_REQUIRED}" =~ ^[0-9]+$ ]] || {
    echo "[Error] Poll settings must be positive integers." >&2
    exit 1
  }
  (( POLL_SECONDS > 0 && FREE_SAMPLES_REQUIRED > 0 )) || exit 1
  wait_for_exclusive_gpu0

  START=1 WAIT_FOR_COMPLETION=1 AUTO_ANALYZE=0 \
    PROFILE=gradient_conflict_memory_canary \
    RUN_TAG="${CANARY_RUN_TAG}" UNIT_PREFIX="${CANARY_UNIT_PREFIX}" \
    "${LAUNCHER}"
  validate_canary

  wait_for_exclusive_gpu0
  START=1 WAIT_FOR_COMPLETION=1 AUTO_ANALYZE=0 \
    PROFILE=gradient_conflict_wave1 \
    RUN_TAG="${FORMAL_RUN_TAG}" UNIT_PREFIX="${FORMAL_UNIT_PREFIX}" \
    "${LAUNCHER}"
  echo "[Complete] Stage3 G0-G3 gradient-conflict screen finished."
}

main "$@"
