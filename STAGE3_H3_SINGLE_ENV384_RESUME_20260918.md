## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: user-authorized sampling-width change and committed-state resume
- Authorization: increase parallel environments to 384 and continue the latest checkpoint
- Verification: pending selected tests and startup; speed benefit unverified

# Single-flow 384-environment restart from latest committed continuation

Stop the env192 system service and preserve its artifacts. Its latest complete
checkpoint is local epoch 1 (cumulative data epoch 9), with 108 PPO steps:
`stage3_h3_single_parent_20260918_r2_env192_gpu0/attempts/20260918T003443_3985957/C03/epoch_0001/checkpoint.pt`.
SHA256: `bbb335adb694b354c874075a1473f837ffa0856241fc324a8795e15cf6b34988`.
Local epoch 2 was interrupted without a complete commit and must be collected again.

Create `stage3_h3_single_parent_20260918_r3_env384_gpu0`. Only rollout_workers
changes from 192 to 384: one collection of 384 full trajectories per epoch,
64 CPU environment processes, one GPU model. Keep minibatch/microbatch 64/64,
tau .03, two PPO passes/12 updates, Train240, input cache 4096 MiB, learning rates,
source-relative advantage, normalization, original case/seed/optimizer schedule,
canonical evaluation and existing continuation gates.

Explicitly validate the previous suite/arm/commit/checkpoint hashes. Create a
derived checkpoint with only manifest_sha256 and recipe_sha256 rebound to the
new sampling profile. Compare every other payload field exactly after a CPU
save/reload, including model, nonempty Adam states, ValueNorm, RNG and epoch/update
cursors. Preserve the original checkpoint bytes. Import committed epoch/update
ledgers and matched completed evaluation results with provenance. Keep the old
budget clock and the eight-epoch overall target; do not reset the run budget.

One native canary and exact fresh-process restore now initialize from the imported
latest checkpoint, not from R0 E8. After admission, skip committed epoch 1 and
resume local epoch 2 (cumulative data epoch 10). At most seven epochs remain,
with E2/E4 harm/extension gates unchanged. No T10 or separate throughput study.

Use GPU0 and CPU 0-31,64-95. Retain disabled experiment RAM/swap caps and effective
oomd exemption; host-wide services remain unchanged. GPU1 is outside this run.
IGA stays frozen with zero solver queries. Full-epoch timing and memory peaks are
recorded by formal training; no unmeasured speedup is claimed.
