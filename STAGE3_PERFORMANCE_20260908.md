# Stage3 execution-only performance update

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + reproducibility validation
- Origin Date: 2026-09-08
- Verification Status: See immutable CPU/GPU receipts in the active hot-update attempt.
- Version Label: stage3_execution_perf_v1

## Scope

This is an execution update of the existing single-seed E0–E3 × T0/T1 study,
not a new algorithm arm. Seed, case schedule, 8 fresh trajectories per update,
case/trajectory/time reduction, FP32/TF32 settings, recurrent history, TBPTT8,
PPO2, optimizer ownership, all learning rates, KL guards, evaluation cases and
960/1440/1920 gates are unchanged. No extra seeds or FinalBlind evaluation.

Opt-in options: disable encoder activation recomputation; cache graph batches
and E0 encodings / E1–E3 frozen prefixes within one update group (1024 MiB
admission cap, plus one transient batch); reuse a detached actor encoding for
the immediately following critic pass; transfer scalar statistics once per
TBPTT window, preserving per-step FP32 reductions and Python FP64 sum order.
The cache never retains trainable encoder outputs across optimizer steps.
Graph modules must be in evaluation mode. Caches clear even on exceptions.

Not enabled: batched stochastic decoding, asynchronous policy-version mixing,
larger batches, longer TBPTT, mixed precision, TF32, torch.compile, CUDA Graphs,
unbounded shared-memory IPC. These require separate numerical/algorithm checks.

## Rolling deployment

Running Python workers do not support import reload or cooperative checkpoint
signals. The new service suspends only the old controller (not trainers), runs
a bounded GPU equivalence gate, then drains and replaces the two validators.
Both validator consumers share the same durable queue; only the two explicitly
recorded execution identities are admitted. No requests are dropped or retried.

Each old trainer keeps running until its next complete 240-trajectory checkpoint.
The updater suspends that trainer and verifies that the latest published update
has exactly the checkpoint's trajectory count. If a later update was committed,
it resumes the trainer and waits for the following boundary. After validating
model, Adam, normalizers, RNG, arm, schedule, seed and cursor, it terminates only
that recorded process group and resumes the new worker in a distinct output
directory. Any uncommitted work after the checkpoint remains diagnostic-only.
SIGTERM diagnostic failure states/logs from this administrative handoff are not
algorithm failures or valid training resume sources. Old source and artifacts
are never overwritten. Mutable dashboard files are archived before replacement.

The last old trainer's handoff retires the old controller. The new controller
then takes the controller lock and applies the original screen/pilot gates.
Two validators are co-located on GPU0 / GPU4, one per NUMA node. Each NUMA node
has 4 trainers × 7 physical cores, 3 validator cores and 1 coordination core;
all SMT siblings stay together. External jobs are never signaled or preempted.

The current updater deliberately accepts only the audited eight-GPU topology
and one migration from the current screen_960 run. It refuses implicit retries.
A GPU gate failure before any old worker is retired resumes the original
controller. An irreversible migration failure stops owned study jobs and retains
all checkpoints; it requires diagnosis, not an automatic retry.

## Verification and speed claims

The equivalence canary uses real environment observations with a deliberately
synthetic 16-step, B8 ragged prefix fixture. It compares loss/KL/masks, gradients,
model and warm Adam states, RNG and sampled actions for all four encoders, with
atol 2e-7 / rtol 2e-6 (not the research skill's generic 5% tolerance). GPU tests
use deterministic algorithms to isolate execution changes; production retains
its original CUDA settings. Peak memory is checked in the bounded canary.

This does not establish worst-case full-group peak memory or end-to-end speedup.
Runtime update records include execution-cache counters and peak reserved memory.
Compare real full-group seconds and matched case/macroblock exposure after the
update. The canary's timing ratio is a microbenchmark only, not an ETA multiplier.

Entry points: `hot_update_stage3_representation.py prepare ...` creates a new
immutable source/manifest after CPU evidence; the hot-update launcher starts an
independently owned systemd controller and performs the GPU gate before handoff.
