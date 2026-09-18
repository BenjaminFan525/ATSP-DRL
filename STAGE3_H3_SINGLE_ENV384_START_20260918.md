## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: authorized continuation from the most recent committed checkpoint
- Verification: 85 selected tests passed; full payload state preserved exactly across migration; GPU resume confirmed at 108 updates
- Launch status: single canary/restore admission running; formal epoch 2 follows automatically
- Scientific benefit and env384 speed: not yet measured

# 384-environment single continuation launched

Started 2026-09-18 08:20:15 CST as system service
`hkbz-stage3-h3-single-parent-20260918-r3-env384-gpu0.service`.
Run root: `result/hkbz_train_logs/stage3_h3_single_parent_20260918_r3_env384_gpu0`.
Controller PID 146303; first admission model PID 146322. Exactly one arm (C03).

The env192 service is inactive. Its latest full checkpoint, local epoch 1
(cumulative data epoch 9, 108 PPO steps), was restored. Original SHA256:
`bbb335adb694b354c874075a1473f837ffa0856241fc324a8795e15cf6b34988`.
Only manifest/recipe identity metadata was rebound for the new sampling width;
all other fields were compared exactly after serialization, including nonempty
Adam states, model, ValueNorm and RNG. The original checkpoint remains intact.

GPU loading already confirmed resume_epoch=1 and resumed_updates=108 in
`arms/C03/canary/fork.json`. Disposable native canary and fresh-process restore
checks precede formal epoch 2. The interrupted partial epoch 2 is recollected.
Epoch 1's completed evaluation and commit ledger were retained with provenance.
The original 8-epoch budget, wall clock and conditional gates are preserved.

384 simultaneous environments; 64 CPU environment processes; one GPU model;
384 visits/epoch; minibatch/microbatch 64/64; two PPO passes; tau .03. Only GPU0
and CPU 0-31,64-95. Experiment RAM protection remains disabled and actual
root-owned cgroup omission is verified. IGA solver queries remain zero.

Evidence: `resume_migration.json`, `launch_verification.json`, `tests.xml`,
`run_status.json`, `arms/C03/canary/fork.json`, `arms/C03/canary/passed.json`
(when admission completes), and `arms/C03/commits/`.
