## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: authorized single-flow continuation
- Verification: 74 selected CPU tests passed; frozen source and parent parameters verified; service running
- Training status at launch: one-arm native canary and restore admission; formal continuation follows automatically
- Scientific benefit: unverified

# Single-flow continuation launched 2026-09-18

Stopped `hkbz-stage3-h3-continuation-20260917-r3-env96-gpu0.service` and verified
inactive/dead, MainPID 0 and no remaining compute processes. The stopped run had
zero formal committed epochs; its source and collected data remain intact.

New run: `result/hkbz_train_logs/stage3_h3_single_parent_20260918_r1_env240_gpu0`.
System service: `hkbz-stage3-h3-single-parent-20260918-r1-env240-gpu0.service`.
Started 2026-09-18 00:28:39 CST; controller PID 3966258. Exactly one registered
arm exists (internal continuation label C03); T10 is not scheduled at any stage.

Parent: previous R0 selected epoch 8, checkpoint SHA256
`14d63bb44ee96eea38c993bb7d1d1d1ae5f0aa59f979b14a9337e6f96215a4e6`.
Model, actor/critic Adam, ValueNorm and RNG are restored. New local epoch 1 means
cumulative data epoch 9, starting with 96 inherited PPO updates.

Previous training parameters restored: 240 environments, 64 CPU environment
workers, one GPU model, Train240, 384 visits/epoch, minibatch/microbatch 64/64,
two PPO passes, tau .03, 4096 MiB input cache, unchanged learning rates and loss.
The original sampling schedule and optimizer shuffle advance into epochs 9-16.
The existing continuation risk/extension gates bound the maximum extra eight epochs.

GPU0 only; CPU 0-31,64-95. Root-owned system service runs as fanyx. Per-experiment
RAM/swap limits remain disabled and effective oomd omission is verified; global
OOM services are unchanged. Canonical fixed-rule evaluation remains in use.
Five completed baseline evaluation groups are reused; IGA solver calls remain zero.

Evidence in the new run: `manifest.json`, `tests.xml`, `parent_parameter_audit.json`,
`launch_attempt.json`, `launch_verification.json`, `attempts/*/memory_runtime.json`.
Consult live `run_status.json` and `arms/C03/canary/passed.json` for current
admission/completion state; this document records the launch snapshot only.
