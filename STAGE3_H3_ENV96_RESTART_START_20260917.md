## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: authorized restart
- Verification: 91 selected CPU tests passed; service running; actual memory policy and CPU affinity verified
- Scientific result: unverified; GPU canary and capacity admission running before formal epoch 1

# Stage3 96-environment restart

The new experiment started at **2026-09-17 22:36:57 CST**. Service:
`hkbz-stage3-h3-continuation-20260917-r3-env96-gpu0.service` (system manager,
controller PID 3853888). Training runs as `fanyx`, on GPU0 only.

| Setting | C03 | T10 |
|---|---|---|
| Parallel environments / CPU workers | 96 / 32 | 96 / 32 |
| CPU logical IDs | 0–15,64–79 | 16–31,80–95 |
| Train tau | 0.03 | 0.1 |
| Train cases / visits per epoch | 240 / 384 | 240 / 384 |
| Sampling groups per epoch | 4 | 4 |
| Optimizer minibatch / physical microbatch | 64 / 64 | 64 / 64 |
| PPO updates per epoch | 12 | 12 |

Memory policy is verified from the kernel cgroup, not only the requested service
properties: `memory.high=max`, `memory.max=max`, `memory.swap.max=max` for this
service and its ancestors below the cgroup root. The service cgroup is owned by
root and has `user.oomd_omit=1`; `ManagedOOMPreference=omit` is therefore effective
on this host's systemd 249. The global oomd service and kernel global OOM behavior
remain unchanged. The training processes are ordinary user processes. See the
[systemd 249 implementation](https://raw.githubusercontent.com/systemd/systemd/v249/src/oom/oomd-util.c)
for the root-owned-cgroup requirement.

Both arms restart from the preserved R0 E8 checkpoint, SHA256
`14d63bb44ee96eea38c993bb7d1d1d1ae5f0aa59f979b14a9337e6f96215a4e6`,
with 96 inherited PPO updates. The interrupted r2 run had no formal continuation
checkpoint. Its files and frozen source were preserved.

The run manifest is
`49514b61ac6ba22f46c67277e916fe5c5d902ad5f44a7acb54bd0d75000058b9`.
All 489 frozen source files were checked after startup. Five completed baseline
groups, comprising 264 evaluation trajectories, were reused with identity checks.
IGA solver queries remain zero.

At startup verification, both two-case canaries completed sampling and entered
likelihood replay. Fresh-process recovery and the native 96-slot GPU window probe
precede automatic entry into formal epoch 1. These checks do not claim full-batch
CPU memory capacity or a training speedup. CPU affinity is applied before Python
imports; every live trainer initialization thread passed its partition check.

Evidence directory:
`result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r3_env96_gpu0/`

- `manifest.json`, `prepared.json`, `tests.xml`
- `launch_attempt.json`, `launch_verification.json`
- `attempts/20260917T223658_3853888/memory_runtime.json`
- `run_status.json`, `resource_samples.jsonl`, `controller.log`

Launch entry point: `launch_stage3_h3_env96_20260917.py`. Its dry-run prints the
exact systemd command; repeated starts are rejected while active or once the run
has acquired a budget clock, to avoid an accidental hidden retry.
