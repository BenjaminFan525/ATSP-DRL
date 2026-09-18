## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: authorized execution configuration change
- Authorization: user requested 192 parallel environments for the single training flow
- Verification: pending selected checks and startup; no scientific benefit claimed

# Single R0 E8 continuation with 192 environments

Supersedes the env240 launch. That service was stopped during its disposable
canary, before any formal epoch/checkpoint was committed. Preserve all its files.
Continue from the same previous best R0 E8 checkpoint, SHA256
`14d63bb44ee96eea38c993bb7d1d1d1ae5f0aa59f979b14a9337e6f96215a4e6`.

The only training execution change is rollout_workers 240 -> 192. A complete
epoch still collects 384 trajectories, now in two equal pools of 192, with 64 CPU
environment workers and one GPU model. Train240, microbatch/minibatch 64/64,
two PPO passes, 12 updates/epoch, tau .03, input cache 4096 MiB, learning rates,
parent optimizer/ValueNorm/RNG, original epochs 9-16 case/seed schedule, canonical
evaluation and existing continuation gates all remain as registered in
`STAGE3_H3_SINGLE_PARENT_RESTART_20260918.md`.

GPU0 and CPU 0-31,64-95 only; one arm, no T10. Per-experiment memory protection
remains disabled with actual root-owned cgroup omission verified. IGA frozen,
zero solver queries. Sampling batch width can change stochastic trajectories;
the case/seed schedule is preserved, not a claim of identical sampled actions.

New immutable run: `stage3_h3_single_parent_20260918_r2_env192_gpu0`.
Reuse only completed, identity-checked baseline evaluation groups. The interrupted
canary is not a committed checkpoint and is not reused as a passed admission.
Run the single native canary and exact fresh-process restore, then automatically
start formal local epoch 1 (cumulative data epoch 9). No microbatch sweep.
