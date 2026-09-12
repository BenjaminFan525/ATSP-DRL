# Stage3 isolated B32 / TBPTT16 resource probe

User-authorized single-GPU diagnostic, 2026-09-08. This is not a ninth scientific
arm and its checkpoint must never initialize or enter the live study.

- Parent: E2_T1 episodes_000240.pt, full optimizer/normalizer/RNG resume.
- Execution: the existing perf_20260908_r1 frozen source; production code untouched.
- One aligned T1 macroblock: four cases × eight fresh trajectories, collected in
  four width-8 waves without an intervening policy update; one B32 PPO update.
- TBPTT 8 → 16; frozen-feature cache 1024 → 4096 MiB. PPO epochs, learning rates,
  precision, reward, clipping and KL guards retain the parent's settings.
- GPU1 co-location, seven physical CPU cores plus SMT. No existing process is
  stopped or moved. Timing is contention-dependent, not a matched speedup result.
- Probe-only MemoryHigh 20 GiB, MemoryMax 24 GiB, swap disabled; preserve 24 GiB
  host MemAvailable. CUDA allocator limit 75% (~18 GiB), excluding non-PyTorch
  context overhead. Guard at cgroup 23 GiB; 55-minute probe deadline and 60-minute
  service deadline. Limits apply to the probe and all eight environment children.
- Resource samples every ~5 seconds plus wave/update boundaries: cgroup current /
  sampled peak (this host lacks kernel memory.peak), summed RSS and PSS, host available RAM, own-process GPU memory,
  torch allocation/reservation peaks, whole-GPU utilization/memory/power.
- Preserve configuration, schedule/seeds/action hashes, wave timings, update,
  diagnostic checkpoint and failure/guard receipts. No production validator jobs.
- A completed group establishes one observed peak, not a worst-case memory bound
  or evidence of better RL quality. An incomplete probe cannot establish safety.

Verification before relaunch: eleven CPU unit tests passed (schedule balance/freshness,
cursor alignment, size limits, cgroup isolation, memory/runtime guards, JSON and
defaults, old-kernel peak fallback, cleanup before monitor startup). The launcher is syntax-checked. Frozen inputs and checkpoint provenance
are checked at runtime before training and again after the measured update.

Attempt r1 failed at monitor initialization before loading the engine because the
kernel lacks memory.peak; its artifacts are preserved. Attempt r2 uses sampled
host/cgroup peaks (brief spikes can be missed), alongside exact PyTorch allocator
peak counters and the unchanged cgroup hard limit.

Material Passport: academic-research-suite/experiment-agent; mode run;
version resource_probe_v1; resources/results unverified until runtime receipt.

## Coordination incident and explicit recovery

At 21:57 CST the existing rolling controller classified the independent probe
as an external GPU job and exited with `External GPU jobs appeared`, stopping
its five migrated trainers, validators and the legacy controller's three remaining
trainers. This was caused by the probe launch, not by memory pressure. The launch
preflight missed that the controller required exclusive ownership of all GPUs.

Checkpoint recovery necessarily rewinds completed updates since the last full
checkpoint; all their artifacts remain intact:

| Arm | Last committed trajectories | Full resume point | Recompute |
| --- | ---: | ---: | ---: |
| E0_T0 | 584 | 480 | 104 |
| E0_T1 | 552 | 480 | 72 |
| E1_T0 | 544 | 480 | 64 |
| E1_T1 | 472 | 240 | 232 |
| E2_T0 | 408 | 240 | 168 |
| E2_T1 | 392 | 240 | 152 |
| E3_T0 | 536 | 480 | 56 |
| E3_T1 | 488 | 480 | 8 |
| Total | | | 856 |

The explicit recovery wrapper retains the frozen worker code, scientific recipe,
eight-arm schedule and original screen/pilot gates. New output directories preserve
all previous runs. One exact GPU1 probe identity (PID, birth token, command,
affinity and cgroup) is authorized for co-location; unrelated jobs are not exempted.
The interrupted E3_T1@480 validation failure is archived and the same checkpoint's
request is republished to the existing shared two-process validator pool.
The probe's initial timing includes a period without training contention and must
not be reported as a matched production speedup.

## Measured outcome

Probe r2 ended at 22:15:13 CST: all 32 rollouts completed, but epoch 0 of the B32 /
TBPTT16 update hit the 75% CUDA allocator limit (17.64 GiB) before completing.
CPU cgroup sampled peak 9.64 GiB; probe GPU sampled peak 17.70 GiB and OOM-message
usage 18.05 GiB, with 3.76 GiB still free on the physical GPU. This establishes
failure under the shared allocation budget, not impossibility on an exclusive
24 GiB card. No completed-update throughput or RL improvement claim is supported.
The probe service exited; the recovered eight-arm study and shared validators
remained active. Full evidence and the coordination incident are documented in
`result/hkbz_train_logs/stage3_representation_all_20260908_r1/probes/b32_t16_20260908_r2/REPORT.md`.
