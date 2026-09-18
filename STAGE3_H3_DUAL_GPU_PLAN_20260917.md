## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: implementation and authorized execution
- Date: 2026-09-17
- Scope: user-requested two concurrent continuation arms on GPU0
- Research benefit: unverified; resource and recovery admission precede formal PPO

# H3 temperature continuation on one GPU with two arms

This execution amendment replaces the sequential scheduling and physical resource
settings in `STAGE3_B_SHARED_IGA1800_CONTINUATION_PLAN_20260917.md`. The research
questions, data, checkpoints, learning budget, evaluation and promotion rules stay
registered as before. The serial run stopped during its disposable canary, before
any new formal PPO checkpoint. Its completed evaluation results are preserved.

| Setting | C03 | T10 |
|---|---|---|
| Physical GPU | GPU0 | GPU0 |
| CPU logical IDs | 0–15,64–79 | 16–31,80–95 |
| Training temperature | 0.03 | 0.1 |
| Concurrent environments / CPU workers | 128 / 32 | 128 / 32 |
| Optimizer minibatch / physical microbatch | 64 / 64 | 64 / 64 |
| Torch allocator ceiling | 18 GiB | 18 GiB |
| Immutable GPU input cache | 1 GiB | 1 GiB |
| Unique training cases / visits per epoch | 240 / 384 | 240 / 384 |
| PPO passes / Adam steps per epoch | 2 / 12 | 2 / 12 |

The two lanes share the previously authorized CPU half, not an additional half
each. GPU1 and its Stage2 processes are outside this change. The combined service
uses MemoryHigh=164 GiB, MemoryMax=176 GiB, MemorySwapMax=0. Host available memory
was checked before choosing this aggregate ceiling; it is higher than the old
single-arm 114 GiB ceiling because both retained trajectory batches coexist.
Each CPU pool exits after collection. Free host malloc arenas are released after
logical collection without modifying retained observations.

All three complete probability checks per epoch remain. Scalar GPU-to-CPU
transfers in these checks are grouped into TBPTT windows while retaining the
original per-step reductions and Python summation order. Native replay metrics
and RNG are compared exactly against the previous reduction implementation.

The new inference batch width changes the sampling stream relative to a
hypothetical serial 240-environment run. Both arms use the same new physical
layout and the same registered case/seed schedule. This is not a claim of
240-versus-128 trajectory identity. Actor/critic learning rates, loss weighting,
gradient clipping, Adam inheritance and ValueNorm are unchanged.

Admission and execution:

1. Verify the old suite, frozen evaluator bytes, dependencies, GPU/CPU identity,
   case order/content and completed result identities before reusing the five
   baseline evaluations. Bind original result hashes in every migrated result.
2. Execute both temperatures' two-case/two-update canaries simultaneously. In
   separate fresh processes, reproduce model, Adam, ValueNorm and RNG exactly.
3. Benchmark identical native 128-slot inference and 64-trajectory TBPTT windows
   individually and simultaneously, recording per-process allocated/reserved
   memory, paired wall time, and zero optimizer updates. These repeated real-case
   windows are a bounded capacity/throughput probe, not a full-epoch speed claim.
4. Start both formal epoch-1 workers together. Both must exit before fixed-12
   canonical validation starts. Resume each arm only from its own complete epoch.
5. Apply the original E2/E4/E8 stopping and promotion rules with equal learning
   budgets. A failed worker terminates unfinished peers and preserves already
   committed checkpoints and failure evidence. No hidden retry/fallback profile.

IGA stays frozen; no new solver calls. The first full paired epoch determines
the actual combined training throughput and updated ETA. Two resident processes
alone do not establish a speedup or a research improvement.
