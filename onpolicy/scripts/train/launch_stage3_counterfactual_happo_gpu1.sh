#!/usr/bin/env bash
set -euo pipefail

# One-command Stage3 N0-N3 workflow on GPU1:
#   1) four-lane graph=1000 memory/update canary;
#   2) fail closed unless every canary completes;
#   3) launch the scientific five-epoch Wave1 and its shared Valid60 service;
#   4) analyze all four arms automatically after completion.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_LAUNCHER="${ROOT_DIR}/onpolicy/scripts/train/launch_stage3_ppo_gain_four_gpu1.sh"
PYTHON="${PYTHON:-${ROOT_DIR}/../conda/envs/maia-hkbz-cu124-20260903/bin/python3.11}"
CANARY_FIRST="${CANARY_FIRST:-1}"
RUN_TAG="${RUN_TAG:-stage3_counterfactual_happo_wave1_20260826_r1}"
UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s3cfh-r1-g1}"
CANARY_TAG="${CANARY_TAG:-stage3_counterfactual_happo_canary_20260826_r1}"
CANARY_PREFIX="${CANARY_PREFIX:-hkbz-s3cfh-canary-r1-g1}"
RESULTS_ROOT="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo"

if [[ "${CANARY_FIRST}" == 1 ]]; then
  env \
    PROFILE=credit_happo_memory_canary START=1 WAIT_FOR_COMPLETION=1 \
    AUTO_ANALYZE=0 AUTO_NEXT_WAVE=0 METHOD_FILTER=all \
    RUN_TAG="${CANARY_TAG}" UNIT_PREFIX="${CANARY_PREFIX}" \
    SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${CANARY_TAG}" \
    "${BASE_LAUNCHER}"

  "${PYTHON}" -c '
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); tag=sys.argv[2]
bad=[]
for method in ("N0", "N1", "N2", "N3"):
    path=root/f"{tag}_credit_happo_memory_canary_{method}_seed1"/"run1"/"run_status.json"
    try: status=json.loads(path.read_text())
    except Exception as error: bad.append(f"{method}: unreadable {path}: {error}"); continue
    if status.get("status") != "completed": bad.append(f"{method}: {status.get('"'"'status'"'"')} / {status.get('"'"'error'"'"', '"'"''"'"')}")
if bad: raise SystemExit("Stage3 canary gate failed:\n" + "\n".join(bad))
print("[Canary] N0-N3 graph=1000 update and memory contracts passed.")
' "${RESULTS_ROOT}" "${CANARY_TAG}"
fi

env \
  PROFILE=credit_happo_wave1 START=1 WAIT_FOR_COMPLETION=0 \
  AUTO_ANALYZE=1 AUTO_NEXT_WAVE=0 METHOD_FILTER=all \
  RUN_TAG="${RUN_TAG}" UNIT_PREFIX="${UNIT_PREFIX}" \
  SUITE_DIR="${ROOT_DIR}/result/hkbz_train_logs/${RUN_TAG}" \
  "${BASE_LAUNCHER}"

