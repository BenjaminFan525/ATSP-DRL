## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: implementation / run
- Verification: 78 selected CPU tests passed; both native GPU replay and exact full-state restore passed; both formal epoch-1 workers running
- Scientific result: unverified

# Dual-arm GPU0 continuation launch

The user requested two simultaneous experiments on one GPU. The old serial service was stopped during its disposable canary with zero formal continuation commits. Five completed evaluation groups (264 full trajectories) were rebound after frozen-source, input, dependency and request identity checks. No IGA solving occurred.

| Arm | Train tau | CPUs | Environments / CPU workers | Optimizer / physical minibatch | Torch memory ceiling |
|---|---:|---|---:|---:|---:|
| C03 | 0.03 | 0–15,64–79 | 128 / 32 | 64 / 64 | 18 GiB |
| T10 | 0.10 | 16–31,80–95 | 128 / 32 | 64 / 64 | 18 GiB |

- GPU0 UUID: `GPU-744c1334-98c8-5318-e799-7ad15eea1fbf`
- Shared service RAM limits: MemoryHigh 164 GiB, MemoryMax 176 GiB, swap 0.
- Service: `hkbz-stage3-h3-continuation-20260917-r2-dual-gpu0.service`
- Manifest: `45039aa78f69e1cd113fea79a68c4bbc4726efd667971fc107bad904cf9cf5f6`
- Full-state parent: previous R0 E8, 96 inherited optimizer updates.
- Same Train240 / 384 visits / 12 optimizer updates per arm per epoch.
- Same canonical fixed-12 evaluation and E2/E4/E8 gates. Both training processes exit before serial validation.

The new sampling width is part of the paired physical configuration. It does not promise identical stochastic trajectories to the previous width of 240. Deferred replay reductions retain each original per-step scalar and summation order. CPU tests checked exact replay metrics and optimizer updates, process overlap, failure cleanup, resource partitions and existing Stage3 contracts. This was not the complete repository test suite.

The GPU gate compares native old/new replay metrics and CUDA RNG, then repeats both complete two-case/two-update canaries in fresh processes. Subsequent single-versus-parallel probes use repeated real-case 128-slot inference and 64-trajectory TBPTT windows without optimizer steps; their throughput is not a full-epoch speed measurement.

Evidence:

- [Live status](/home/fanyx/HKBZ-environment/result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r2_dual_gpu0/run_status.json)
- [Launch receipt](/home/fanyx/HKBZ-environment/result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r2_dual_gpu0/launch_evidence.json)
- [Frozen manifest](/home/fanyx/HKBZ-environment/result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r2_dual_gpu0/manifest.json)
- [CPU test report](/home/fanyx/HKBZ-environment/result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r2_dual_gpu0/tests.xml)

## Admission and initial execution results

- Both temperatures reproduced native replay metrics and CUDA RNG exactly. Both fresh-process restores reproduced model, Adam, ValueNorm and RNG exactly.
- Formal paired epoch 1 started at 2026-09-17 16:56:36 CST, trainer PIDs 3757026 and 3757027. All 64 environment workers were within their respective CPU partitions.
- Window benchmark: C03 single 35.071 s, T10 single 36.066 s; paired compute span 69.380 s. Aggregate window throughput ratio 1.02534 (about +2.5%, approximately unchanged). No complete-epoch speedup claim.
- Parallel window allocated/reserved peaks: C03 6.777/12.107 GiB; T10 6.886/12.012 GiB. At initial formal rollout, process VRAM was about 14.4/14.0 GiB and can continue growing within the allocator budgets.
- [Admission evidence](/home/fanyx/HKBZ-environment/result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r2_dual_gpu0/validation_evidence.json)
- [Capacity and timing evidence](/home/fanyx/HKBZ-environment/result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r2_dual_gpu0/dual_capacity.json)
