#!/usr/bin/env bash
set -euo pipefail

# Compatibility shim for operators who still have the historical four-stage
# command in a notebook or scheduler.  The old Stage 3/4 semantics no longer
# exist: canonical Stage 2 is resource_joint and the run ends with frozen
# plane/shared PPO.  Keep Stage 1/2 forwarding explicit so a typo cannot
# silently launch a retired stage.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
NEW_LAUNCHER="${ROOT_DIR}/onpolicy/scripts/train/launch_hkbz_two_stage_service.sh"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-2}"

echo "[Deprecated] launch_hkbz_four_stage_service.sh is a compatibility shim. " >&2
echo "[Deprecated] Use launch_hkbz_two_stage_service.sh; only STOP_AFTER_STAGE=1 or 2 is supported." >&2

if [[ "${STOP_AFTER_STAGE}" != "1" && "${STOP_AFTER_STAGE}" != "2" ]]; then
  echo "[Error] STOP_AFTER_STAGE=${STOP_AFTER_STAGE} is retired; choose 1 or 2." >&2
  exit 2
fi
if [[ ! -x "${NEW_LAUNCHER}" ]]; then
  echo "[Error] Canonical two-stage launcher is missing or not executable: ${NEW_LAUNCHER}" >&2
  exit 2
fi

export STOP_AFTER_STAGE
exec /bin/bash "${NEW_LAUNCHER}" "$@"
