## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: implementation / run
- Verification Status: CPU_TESTED; GPU_ADMISSION_RUNNING
- Started: 2026-09-17T15:10:38.098401+08:00
- Scientific result: UNVERIFIED

# H3 continuation launch

已按用户后续澄清改用 GPU0。GPU1 的 Stage2 实验保持运行。

- Service: `hkbz-stage3-h3-continuation-20260917-r1-gpu0.service`
- Root: `/home/fanyx/HKBZ-environment/result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r1_gpu0`
- Manifest SHA256: `73f4f2a45972d0a417f3253c9fd239631ce3d55915d08018ea113f9b3f0a56f8`
- GPU0 UUID: `GPU-744c1334-98c8-5318-e799-7ad15eea1fbf`
- CPU half: `0-31,64-95` (64 logical CPUs, 32 complete physical cores)
- MemoryHigh / MemoryMax / swap: 102 GiB / 114 GiB / 0
- Selected CPU regression tests: 68 passed; not a full repository test suite.

## Implemented

Full model/Adam/ValueNorm continuation fork; sampler/replay/resume temperature binding; C03 (tau 0.03) and T10 (tau 0.1) with common Train240/384-visit schedules; fixed optimizer/microbatch 64 and two PPO passes; passive role/distribution decision statistics; per-group optimizer update statistics; Train24 action/time traces; immutable evaluation bridge using the admitted canonical execution code; parent selection, bounded stage gates, checkpoint ledger, cache identities and final Tune reporting.

State-value advantages are implemented as an explicit separate estimator with pre-update values. They are not enabled in this temperature comparison. Conditional P2/P3 studies remain separately registered comparisons after the P1 result; this service executes P0 and the bounded P1 plan.

## Execution order

1. Reproduce 12 E6 legacy and 12 canonical full trajectories using the frozen evaluator.
2. Complete E8 canonical Validation120, choose E8/E6 using Validation only, then canonical baseline Tune evaluations.
3. Run disposable two-case/two-update canaries at each temperature and verify exact fresh-process restoration of model/Adam/ValueNorm/RNG.
4. Run C03 to epoch 2, then T10 to epoch 2; compare at equal visit/update budgets. Extend to epoch 4 and, only when pre-registered gates pass, epoch 8.
5. Freeze the Validation-selected checkpoint before final Tune evaluation. No IGA solving and no independent IGA confirmation claim.

The normal training budget is 1,536 new visits / 48 PPO updates for the screen; at most 6,144 / 192 for both full branches. The historical microbatch 32/64 throughput benchmark is disabled. Canary PPO updates are discarded and are not counted as formal training.

Cached S0/E6 evaluation snapshots have identical computation files; the only differing file is the old command-line driver, which the new standalone evaluation adapter never imports or executes. All computation files and case identities are bound and checked.

## Evidence

- [Manifest](result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r1_gpu0/manifest.json)
- [Live status](result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r1_gpu0/run_status.json)
- [Launch receipt](result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r1_gpu0/launch_evidence.json)
- [CPU tests](result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r1_gpu0/tests.xml)

This document records launch state, not completion of admission, PPO training, or a measured research improvement.
