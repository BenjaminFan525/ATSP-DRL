# Stage3 IGA frozen references

Decision, 2026-09-18: IGA is frozen. These artifacts are read-only comparison
references; no further IGA solving is scheduled and no future round re-runs the
solver. Comparisons must read the frozen summaries below instead of launching a
new search.

The reference uses the contract family: H3/F4 soft full-joint
(`stage3_full_joint_policy` / `joint_iga_all`, backends `{ordinary: iga,
transporter: iga}`), environment semantics
`progressive-departure-r014-pipeline-v2`, base seed `20260824`, and planning
contract horizon 3 / frontier 4 / request-capacity-per-plane 5 / safety margin
60s / soft reservation with 300s grace / release-aware ETA / slack forecast 0.
The solver budget is wall-clock: IGA-180 stops at 180s cumulative, IGA-1800 is a
nested continuation that stops at 1800s cumulative (1620s additional).

## Primary reference: Validation120 (case-matched to the RL evaluation split)

- Run root: `result/hkbz_train_logs/stage3_matched_iga_h3f4_validation120_20260918_r1/full`
- Dataset: `onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/validation`, 120 cases
- Execution: 120 workers, cpu-set `0-127`, one thread per worker; completed
  2026-09-18T17:11:21+08:00
- Verification: `nested_regression_count = 0`
- Frozen digests (sha256):
  - `iga180/summary.json` = `4744cf4a31019ec9bd98e34ee61cd095e5674c6e11483cc11c065dada64044a7`
  - `iga1800/summary.json` = `b1500a6703e31072b800e4a5ddd47c0fbaedc403da57ee08207f4518da5555a1`
  - `pipeline_state.json` = `a182ca0c21c0f230969693b85687558ea14d75da141f4962fc615e29ef4aa72d`

| Variant | mean makespan | median | evaluated candidates/case |
|---|---:|---:|---:|
| IGA-180 | 8791.0856 | 8393.5000 | 5.19 |
| IGA-1800 | 8310.6556 | 8017.0000 | 51.07 |

The 120-case split shares no case with Tune60 (`case_sha256` disjoint), so this
reference is the only admissible IGA comparison for Validation120. Case identity
was re-checked against the RL validation split: 120/120 `case_sha256` matches,
zero mismatches.

Comparison rule, 2026-09-18: Tune60 is no longer a comparison basis. Do not
report RL-versus-IGA gaps on Tune60 in validation or final reports. The IGA
Tune60 run directory was deleted by user instruction; its file inventory and
frozen digests are recorded in
`result/stage3_analysis/iga_tune60_deletion_20260918/receipt.json`.

## Retired reference: Tune60 (deleted 2026-09-18)

`result/hkbz_train_logs/stage3_matched_iga_h3f4_tune60_20260831_r1` was removed
at the user's request. For the record it had reported IGA-1800 mean 8182.7711 /
median 8148.5000 at 104.82 evaluated candidates per case with 36 workers, and
IGA-180 mean 8570.8167 / median 8449.5000. Those files no longer exist and those
values are not a comparison basis.

The sealed Stage3 H3 R0 manifest still embeds the same 60 Tune60 makespans inline
in `frozen_iga.json`. That sealed file is left untouched for audit, but it must
not be used for new comparisons.

Consequence to expect: re-running `verify_manifest(..., inputs=True)` on that
sealed suite, or the `h3_r0_final_20260917` integrity pass that counted 122
`frozen_iga` files, will now report missing Tune60 teacher files. The inline
makespans and the deletion receipt are the surviving record.

Recorded search depth is an outcome of the wall-clock budget under the stated
concurrency, not a configurable knob. Report the depth and the worker count
(51.07 candidates per case at 120 workers for the Validation120 reference)
alongside any IGA-1800 claim.

## Paired RL values cited from another run (unverified here)

`result/hkbz_train_logs/stage3_tau_twoarms_no_kl_20260917_r2/learning_curves.json`
does not exist in this workspace, so the following numbers are unverified
secondary citations and are not used in any comparison.

| Split | B0 | T1_ANNEAL10 | T3_ANNEAL10_RISK |
|---|---:|---:|---:|
| Validation120 | 8398.0178 | 8271.3417 | 8318.6806 |
