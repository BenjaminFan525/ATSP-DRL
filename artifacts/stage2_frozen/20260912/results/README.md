# Stage2 frozen handoff — 2026-09-12

Stage2 development is frozen, not scientifically declared successful. The
handoff is the exactly preserved B0 checkpoint with Hungarian decoding, H2/F4,
soft reservations (grace 300 s, safety 60 s), request capacity 5 per plane,
Ready injection `none`, release/deadline/departure-aware lookahead enabled.

The 60-case historical B0 mean makespan is 8352.922222 seconds. Historical
same-H/F IGA180 and IGA1800 means are 8200.438889 and 8094.988889 seconds:
B0 remains 1.86% and 3.19% worse respectively. No Stage2 configuration has
established the requested IGA superiority. Those IGA comparisons are historical
solution-quality references, not a new hardware-equal runtime certification.

The latest frozen-model HF sweep completed 14 jobs / 840 case-episodes (12 HF
configurations and two additional controls) on the same 60 cases. H2/F4 had the
lowest observed mean. Source and terminal replays passed. It performed zero
actor/critic updates and zero teacher queries. Repeated cases are not 840
independent samples; this is not independent BC retraining or confirmation.

`history/experiment_results.tar.gz` retains every selected original JSON record
byte-for-byte, including large engineering records. `history/results_index.json`
records the exact original path, SHA256, size and tar member for each. The
selection includes all Stage2 run-root JSON, evaluations, commands, records,
analyses, checks, named summaries, model-run evaluations, and historical HF IGA
per-case results. No metrics, parameter summaries, or failed runs were silently
projected away. Old physical rules and development gates differ across runs;
do not pool their raw scores as a common benchmark.

All original raw artifacts remain locally unchanged in the ignored `result/`
and `onpolicy/scripts/results/` trees. Dataset content, teacher trajectories,
TensorBoard/log streams and non-selected candidate checkpoints are not copied
into the portable bundle. The teacher index is preserved for provenance only;
its referenced training label files are not required for frozen inference.
Stage3 supplies its own data and training protocol. Original command paths are
historical evidence, not portable executable commands.

Use the frozen Stage2 helper and verification entry point for portable loading.
The embedded old BC-improvement `stage2_scientific_gate.passed` is not an IGA-goal
pass; the release manifest explicitly records `scientific_goal_confirmed=false`.
