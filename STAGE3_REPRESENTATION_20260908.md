# Stage3 representation study — single seed

This protocol implements the agreed single-seed revision. It does not restart
the stopped R_PPO/J_PPO run or consume Finalblind. Research results remain
unverified until the relevant phase has completed.

## Causal matrix

| Encoder | Frozen | Trainable graph component |
| --- | --- | --- |
| E0 | all graph parameters | none; role GRUs/actors still train |
| E1 | embeddings and all but last message-passing block | one shared final block and its readout/context fusion |
| E2 | same lower graph trunk | three independent copies of the final block and readout, one per role |
| E3 | same lower graph trunk | E1 plus shared zero-output residual MLPs, trainable count within 5% of E2 |

All begin with **original C0**, never Fit, J960 or failure_state. E3 hidden
weights are nonzero and output weights zero: initial function is unchanged,
but added capacity can learn. It controls parameter count, not inductive bias.
The single joint autoregressive decoder retains request reservation order.
Graph-prefix copies and private tails have disjoint Parameter storage; actor
and critic Adam ownership cannot overlap. The shared workspace remains editable
by baseline work; only the run's frozen snapshot/inputs are enforced strictly.

T0 = 1 case x8 trajectories/update; T1 = 4 cases x2/update. Each four-update
macroblock gives both arms eight trajectories per case and exactly the same
case/replica seeds. Every update samples its own fresh eight trajectories from
the current policy. No 32-trajectory stale macro-batch is reused.

## Fixed training protocol

Eight configurations E0..E3 x T0/T1, **one common seed 2026090803**. J exploration
with tau .03 for all three roles. Actor base LR 5e-6; actual plane/device/
transporter LRs 1.25e-6/5e-6/2.5e-6. Source-relative advantage
`.01*(C0_case_cost - sampled_cost)`, gamma1, two PPO epochs, clip .2,
global gradient norm clip1, TBPTT8, critic LR1e-4. Actor reduction is case mean,
then trajectory mean, then time/role **sum**, with no length normalization.
Critics do not supply the source-relative actor baseline. O0 optimizer recipe
is retained; diagnostic LR/clipping/TBPTT variants do not silently change it.

Screen all arms to960 trajectories, save complete checkpoints every240 and
evaluate greedy Tune60 at480/960. At most two arms with gain >=.5% and all risk
bounds pass may extend. Pilot requires **both1440 and1920 gain >=1%**. If1440
fails, no1920 chunk is launched and it is not reported as a completed1920 run.
Risk: OODstress and stress_joint cost regression <=.5%; worst10% mean cost
regression <=1%; fraction cases worsening >5% <=5%. Desired effect remains2%.
No multi-seed confirmation, automatic4800 extension, or Finalblind access.

## Diagnosis and admission

1. CPU/GPU interface tests: graph/logit/action/logp/hidden equivalence, private
   gradients, optimizer ownership, queue concurrency and paired exposure.
2. Four architecture canaries: source greedy reproduction; actual4x2 on-policy
   batch, full recurrent replay, real PPO update, full optimizer/RNG restore and
   identical next update. Capacity, frozen-parameter and post-update KL checks.
   On this real T1 batch, capture full-history per-role and per-case graph
   gradients and pairwise cosine similarities before applying any update.
   One shared-pool C0 Tune60 evaluation must reproduce **every** source case.
3. Historical R/J groups1,121,181 regenerated on-policy from original C0/960/
   1440 and exact archived seeds. Compare full-history replay with stored-hidden
   probes; pre-clip positive/negative/winner and role gradients. Fixed-batch
   O0/LRx2/LRx4/per-group-clip/TBPTT32 updates and same-case closed-loop greedy
   evaluations are diagnostic-only. Historical frozen graph has zero gradient;
   this is reported as inapplicable, not evidence against role conflict.
4. Same archived J teachers for every structure: two separate single-case
   C0 fits (16 updates each), then a **fresh C0** Fit16 run (5 passes). Measure
   teacher NLL, closed-loop cost, Probe64 and recovery fraction
   `(sum(C0)-sum(greedy))/(sum(C0)-sum(teacher))`. Recovery target .5 applies
   only to the fit route. Fit failure never vetoes pure PPO; no fitted weights
   enter the main screen. These are fixed diagnostic budgets, not new seeds.
5. Main screen and gated pilot. Hard-contract IGA reruns are not automatically
   submitted in this representation screen; historical soft-contract costs
   are never treated as directly comparable. A later IGA check must report
   cold vs C0-warm search separately and match wall-time/case/contract budgets.

## Resource ownership and asynchronous validation

Eligible pool: all8 GPUs, all64 physical/128 logical CPU cores. Budget at full
availability:8 trainers x7 physical cores, shared validator pool2 workers x3
cores, controller2 cores; keep SMT siblings together. Rollout width8/train,
width3/validator; one Torch/BLAS thread per process. Validation workers co-locate
with training GPUs (allocator cap .65 trainer+.15 validator <=.80). No
per-arm evaluator, no dedicated validation GPU, no unbounded validation queue.

Existing external CUDA processes and their CPU affinity retain leases until
they exit, including temporarily idle CUDA periods. No external process is
stopped or resized. Freed resources are leased automatically. Thus8 arms can
be queued while baseline jobs still own cards; eight-way execution requires
eight available cards. Controller/child commands and current leases are saved.
CPU budget means eligibility, not forcing 100% utilization during sequential
GPU decisions. Host memory high72GiB/max88GiB preserves baseline headroom.

One durable validator queue has two globally shared consumers. Claim scans
are file-locked but evaluations run concurrently. Cache identity includes
checkpoint SHA, representation, frozen code/protocol, cases, contract,
exploration and decoder. Immutable checkpoint publication precedes requests;
max2 pending/arm and16 globally impose backpressure without dropping gates.
Individual workers never mark another worker's running request interrupted.

Checkpoints preserve model, both optimizers, value normalizers, Python/NumPy/
Torch/CUDA RNG, policy version, case cursor, complete schedule hash, structure,
seed, exploration and pending validation identities. Boundary extensions
resume complete previous endpoints only. Interrupted partial-group states are
saved diagnostic-only; no implicit retry or overwrite. All raw rollouts,
diagnostic checkpoints, evaluation results and rejected arms are retained.

## Run commands

```bash
/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11 onpolicy/scripts/train/run_stage3_representation.py prepare --prior result/hkbz_train_logs/stage3_local_exploration_half_20260906_r1/manifest.json --output result/hkbz_train_logs/stage3_representation_all_20260908_r1
bash onpolicy/scripts/train/launch_stage3_representation_all.sh /data/fanyx/HKBZ-environment/result/hkbz_train_logs/stage3_representation_all_20260908_r1
```

Durable systemd user service, no restart by default, maximum10 days. Per-phase
hard limits: contract12h, historical mechanism24h, fit48h, training chunk72h,
each validation request6h. Progress advisory after15min without actual work;
long updates are not killed solely for lack of log growth. No result is called
passed from NLL, best-of8, or an interrupted checkpoint alone.

Material passport: ARS experiment-agent/run;2026-09-08;representation_v1_single_seed;
internal/local data; implementation verification recorded separately from
research outcomes. This document/manifest are preregistered before execution.

Initial parameter audit: E0/E1 total graph598,657 parameters; E1 trainable176,769.
E2 total952,195/trainable530,307. E3 total952,437/trainable530,549 (548 hidden
units per residual channel), E3 vs E2 trainable-count difference0.0456%.

## Explicit pre-Fit recovery (2026-09-08)

The original run completed all four architecture contracts, all six historical
mechanism jobs and C0 Tune60 validation, then failed at02:57 while entering Fit.
Both phases used the task name E0, colliding at `commands/E0.json`; logs and the
in-memory job table had the same missing phase namespace. The fix uses
`phase/name` consistently for commands, logs, job tracking and wait conditions.
Same-phase duplicates still fail instead of overwriting earlier work.

Recovery is explicit and one-shot. It keeps the original manifest and source
snapshot intact, inventories previous artifacts, archives the failed root status
and resource dashboard, and creates a new snapshot under `recovery/<attempt>/`.
Only controller/recovery/launch/test/documentation files may differ. Network,
PPO, environment, case splits, seed, training budget, gates and validator worker
implementation remain unchanged. Completed diagnostic evidence and C0 validation
are checked by hash and reused; no fitted checkpoint becomes an RL initializer.
Recovery refuses any partial Fit/PPO output or unresolved validation queue.

```bash
/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11 onpolicy/scripts/train/run_stage3_representation.py prepare-recovery --manifest result/hkbz_train_logs/stage3_representation_all_20260908_r1/manifest.json --attempt-id fit_resume_20260908_r1
bash onpolicy/scripts/train/launch_stage3_representation_resume.sh /data/fanyx/HKBZ-environment/result/hkbz_train_logs/stage3_representation_all_20260908_r1/recovery/fit_resume_20260908_r1/manifest.json
```

Workers receive the explicit recovery manifest path, not the original manifest.
The shared validator queue stays in place; two newly named pool workers preserve
the old worker statuses. `active_execution.json` and the current root `status.json`
point to the active recovery manifest. Controller logs belong to the recovery
attempt; Fit and training logs are phase-scoped in `logs/fit/` and `logs/train/`.
No retry is automatic. This recovery changes orchestration, not the study design.
