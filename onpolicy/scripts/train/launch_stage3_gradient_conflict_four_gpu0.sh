#!/usr/bin/env bash
set -euo pipefail

# Stage3 shared-encoder gradient-conflict screen.  GPU0 is paired with NUMA0;
# four trainers own disjoint complete SMT sibling pairs, share one Valid60
# process, and serialize graph=1000 PPO updates through one GPU-local lock.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RESULTS_ROOT="${ROOT_DIR}/onpolicy/scripts/results/HKBZ/simple/gnn_mappo"

export GPU=0
export PROFILE="${PROFILE:-gradient_conflict_wave1}"
export RUN_TAG="${RUN_TAG:-stage3_gradient_conflict_20260831_r1}"
export UNIT_PREFIX="${UNIT_PREFIX:-hkbz-s3grad-r1-g0}"
export AUTO_ANALYZE="${AUTO_ANALYZE:-0}"
export AUTO_NEXT_WAVE=0
export INHERIT_SOURCE_CONTRACTS=1
export SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${RESULTS_ROOT}/stage2_adaptive_trust_20260830_r4_wave1_T0_backtrack_no_bc_seed1/run1/models/checkpoint_Best.pt}"

exec "${SCRIPT_DIR}/launch_stage3_ppo_gain_four_gpu1.sh" "$@"
