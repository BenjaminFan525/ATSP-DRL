# Stage3 B_SHARED single-model sampling continuation

The user requested replacing replicated sampling models with a shared model,
then restarting the experiment. The earlier explicit numerical-equivalence
waiver remains in force. This continuation retains GPU0 and CPUs
`0-31,64-95`, 64 environments, global batch 64, PPO microbatch 64, 4096 MiB
input cache, and two PPO passes per committed group.

## Execution change

`stage3_b0_single_model.ThroughputEngine` inherits the existing PPO execution
adapter. Its collector owns 64 CPU environment workers and calls the same
policy instance used for training once per live environment batch. Workers
receive only environment configuration and actions; they neither construct
policies nor load checkpoints. CUDA is hidden during their interpreter startup
and worker execution. Each worker is pinned to an SMT core pair inside the
authorized half of the host. Environment processes close before PPO; the policy,
Adam state, normalizer, and CUDA context remain in the trainer.

Training uses stable live-slot ordering, separate per-environment histories and
GRU states, authoritative action history, and per-agent terminal resets.
Actions, log probabilities, masks and new hidden states transfer to CPU once per
output tensor. The native fixed-12 deployment evaluator retains completed slots
and its original deterministic dispatch. The separate validator may own its own
evaluation model; the single-model restriction concerns sampling and training.

## Sampling and restart contract

The new contract is `single_model_batched_v1` with
`checkpoint_global_torch_live_slot_order_v1`. Environment seeds and the complete
scheduled visit order remain unchanged. Stochastic actions consume the Torch
CUDA RNG restored from the checkpoint in batch order. They do not reproduce the
former per-visit batch-one RNG streams. Model parameters, actor/critic Adam,
normalizer, counters, and all checkpoint RNG payloads are restored exactly.

The parent is `stage3_b_shared_b0_local_a6000_20260913_r4_resume64_gpu0`.
It was stopped before the uncommitted tenth group finished. Its latest formal
commit is batch 9: 288 visits, 18 updates, checkpoint SHA256
`0131683982682ff6fd16ac50ba99747c4df0bd1ca7539eff16107b0305a27259`.
The continuation imports the nine completed groups without changing their
training tensors. Remaining work is 7392 visits in 116 groups / 232 updates.
Existing B0 baseline and evaluation admission evidence is reused with provenance.

## Validation and evidence scope

CPU regressions cover batched ragged trajectories, action/slot routing,
per-agent hidden-state resets, authoritative history, restored RNG replay,
CPU-only worker startup, unchanged PPO method inheritance, checkpoint identity,
and rejection of stale or invalid admission evidence.

`probe_stage3_b0_single_model.py` loads the latest committed checkpoint on GPU0,
collects the next complete 64-visit group, repeats the first stochastic forward
from identical RNG state, and replays all retained action probabilities and
decision masks. It makes no optimizer update and commits no training visits.
The probe binds the sampler, collector, inherited update module and probe source
hashes. The restart admission requires its completed result and a maximum
behavior log-probability error no greater than 0.002.

Historical full micro64 finite-update evidence admits only unchanged PPO math
and capacity. The restart verifier compares all base-engine methods other than
the collector against the parent, and binds the unchanged update adapter hash.
It does not claim that the new sampler passed legacy numerical equivalence or
that an end-to-end speedup has already been measured. Every formal update still
checks finite values, legal masks, behavior replay, KL limits, and frozen weights
before a complete checkpoint can be committed.

Probe artifacts: `result/hkbz_train_logs/stage3_b_shared_b0_single_model_probe_20260913`.
The probe's hardware audit observed one GPU0 compute process and 64 CPU workers
with empty CUDA visibility and affinities inside the requested half of the host.

The complete GPU probe passed on 2026-09-13. It collected 64 complete trajectories
in 533.14 seconds, executing 30,541 environment steps in 656 batched forwards
(57.29 environment steps per second including pool startup and shutdown).
Full replay took 304.09 seconds. All 54,656 legal decisions had exactly matching
log probabilities and masks: maximum log-probability error 0 and KL 0 before
any update. Exact checkpoint/CUDA RNG restoration and repeat-forward RNG checks
passed. These timings describe sampling and replay, not a complete two-PPO
training group. The workspace regression suite passed all 48 tests.

## Launched continuation

Run: `result/hkbz_train_logs/stage3_b_shared_b0_local_a6000_20260913_r5_single_model_gpu0`.
Service: `hkbz-b0-resume64-1789269780.service`, started at approximately
11:23 CST on 2026-09-13. The frozen source independently passed all 48 tests,
manifest/runtime admission, and fresh-process checkpoint migration checks.

The actual GPU trainer verified exact model, both Adam states, ValueNorm and all
RNG state, including CUDA, before collection. Startup observation reached batch
10, rollout step 51 with 64 live trajectories. All 64 environment workers had
empty CUDA visibility and remained within CPUs `0-31,64-95`; GPU0 had one
sampling/training model process and the independent validator process.
MemoryMax is 114 GiB. `restart_launched.json` records live process, affinity,
CUDA visibility and restore evidence. The parent `execution_transition.json`
points to this continuation. These are startup observations, not a claim that
the first new two-PPO group has already committed.
