# Stage1 IGA-180 / IGA-1800 rerun, 2026-09-20

Requested rerun: reuse the Stage3 IGA search settings for Stage1, evaluate
validation120, and extract the exact validation60 used by the learning baselines.

Started 2026-09-20 10:41:30 +08:00 as user service
`hkbz-stage1-iga-validation120-20260920-r1.service`.
Live completion status is authoritative in `full/pipeline_state.json`.

## Frozen contract

- Stage1 plane operation/site priorities only; resource backend `heuristic`.
  No Stage3 resource genes or H3/F4 lookahead controls are enabled.
- Environment: `progressive-departure-r014-pipeline-v2`, domain randomization off,
  environment seed 42, maximum plane count at least 24, max devices 80,
  cycle/no-progress/relocation limits 8/120/40, rollout limit 4000.
- 120 independent case workers, one numerical thread each, CPU affinity 0–127.
- Base search seed 20260824. Per-case/phase seed derivation and genetic operators
  are imported from the Stage3 generator frozen in `code/`.
- Population 20, maximum 100000 generations, real-valued SBX eta 15,
  polynomial mutation eta 20, tournament size 2, elitism, anytime best retention.
- First phase: 180 seconds of search per case. Second phase: 1620 additional
  seconds, initialized from that case's replay-verified first-phase incumbent.
  Cumulative nominal budget: 1800 seconds, with no score-based restart selection.
- As in Stage3, setup and final deterministic replay are outside the search
  timer. A search without any feasible incumbent lets a candidate complete;
  any resulting budget overshoot is reported explicitly.
- Every final incumbent must complete a deterministic replay with identical
  makespan. Second-phase makespan must not exceed the first-phase value.

## Data and provenance

`inputs/validation` is a byte-for-byte snapshot of
`onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/validation`.
All 120 case fingerprints match the frozen Stage3 validation120 reference.
Canonical JSON file fingerprints and raw file digests are both retained.

`inputs/aligned_validation60.json` is the P5 seed1 epoch8 evaluation used to
identify the exact learning-baseline panel: 29 IID, 29 OOD-stress, 2 OOD-scale.
Its 60 cases are matched by case directory and all five content fingerprints.
They are not the first 60 cases in the dataset. The manifest records their names.
The historical evaluation's distribution labels differ from actual case metadata
for 32 of these cases, because the runner applied overrides from the legacy
`fjsp_v3_t600_v120_test60/manifest.json`. Historical counts are 29/29/2; actual
dataset counts for this same panel are 30/27/3. These are reporting labels and
do not change case payloads or the IGA search environment.
The IGA case hash uses `fingerprints.case_sha256`, matching the Stage3 reference;
legacy evaluation records also contain a different top-level hash convention.

`snapshot_manifest.json` records hashes of the frozen code and inputs;
`launch_contract.json` records the exact command and environment;
`hardware_preflight.json` and `startup_audit.json` record launch checks.
The prior Stage3 reference remains unchanged.

## Outputs

- `full/iga180/summary.json` and `full/iga1800/summary.json`: separate full-panel
  and aligned-panel mean, median, p95, worst case, distribution means, and
  composite `0.50 IID + 0.45 OOD-stress + 0.05 OOD-scale`.
  `aligned_validation60` preserves historical learning-baseline grouping for
  score compatibility only. `aligned_validation60_case_metadata` reports actual
  dataset groups. `validation120` always uses actual dataset labels.
- `full/iga*/cases/`: per-case makespan, completion, search depth and timing.
- `full/iga*/teachers/`: replay-verified chromosome, trajectories, and warm-start
  source hash. Results are atomically saved as cases finish.
- `full/nested_verification.json`: final paired improvement/regression counts.
- `logs/pipeline.log`: launch, case completions, phase transitions and errors.
- `full/report_status.json`: completion of automatic label-aligned reporting.
  The separate `hkbz-stage1-iga-validation120-20260920-r1-report.service` waits
  for each phase summary, preserves its original version, and publishes both
  label conventions. Its frozen code is in `reporting_code/`, with a separate
  hash manifest. Active search code, teachers and case results are untouched.

The two-case 10+10-second smoke run is under `smoke/` and is excluded from
formal results. Both replayed successfully with zero nested regressions.
Four regression tests passed, covering continuation across generation zero,
incumbent preservation, warm-start identity/budget checks, payload mutation,
and keeping historical versus actual distribution groups separate.

IGA-180 completed 120/120 cases at 10:44:56 +08:00, with all replay checks
passing. Search wall time was 180.0008–180.2157 seconds per case, and mean search
depth was 8.875 completed candidate evaluations. Validation120 mean makespan was
9324.1833; aligned60 historical-group composite was 9368.4759. IGA-1800 then
started automatically with an additional 1620-second budget.

This is one search seed per case, not a three-seed IGA experiment. Validation
results must not be merged into the historical independent test60 ranking.
