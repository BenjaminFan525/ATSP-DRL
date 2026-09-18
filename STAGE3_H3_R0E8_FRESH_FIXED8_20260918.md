# Stage3: fresh R0 epoch 8 continuation, fixed eight epochs

## Material Passport

- Origin: academic-research-suite / experiment-agent; local execution contract.
- Authorization: restart from original R0 epoch 8; disable outcome early stopping,
  train eight local epochs, retain numerical and GPU memory admission checks.
- Resources: one training flow on GPU0, CPU 0-31,64-95; IGA frozen, zero solver queries.
- Verification: selected CPU regression suite, native checkpoint/optimizer/RNG
  restoration, fresh-process CUDA canary, and 128-wide real-graph capacity probe.
- Research status: throughput and learning benefit of this fresh run are unverified.

## Initialization and budget

The only training parent is original R0 epoch 8:
`result/hkbz_train_logs/stage3_b_shared_h3_r0_frozen_iga_20260915_r1_gpu0/attempts/20260915T012204_2081156/train/models/epoch_0008.pt`.
SHA256: `14d63bb44ee96eea38c993bb7d1d1d1ae5f0aa59f979b14a9337e6f96215a4e6`.
Restore weights, both Adam optimizers and step counters, ValueNorm, Python,
NumPy, Torch CPU and CUDA RNG. Inherited PPO updates: 96. No recent continuation
checkpoint or continuation data cursor is imported. Old experiments remain intact.

Run tag: `stage3_h3_r0e8_fresh_20260918_r5_env384_mb128_gpu0`.
Execution profile: `single_parent_env384_mb128_fresh_v1`.
Stopping policy: `fixed_eight_epochs_numeric_gates_v1`.
Train240 contains 192 IID, 43 OOD stress and 5 OOD scale cases. Each epoch visits
384 trajectories through the unchanged IID/OOD replication schedule. Eight local
epochs correspond to original schedule epochs 9-16: 3072 visits, 48 new optimizer
updates, 144 cumulative updates. The new wall budget starts at launch: 120 hours.

## Training parameters

B_SHARED, H3/F4/soft, C03 only. Concurrent environments 384, CPU environment
processes 64, one shared GPU model. Global batch 384, optimizer minibatch 128,
physical microbatch 128 from local epoch 1, two PPO passes (six updates per epoch).
Actor LR 1e-5, critic LR 1e-4, role LR scales [1, .25, 1, .5], PPO clip .2,
gradient clip 1, TBPTT 8, source-relative advantages, training tau .03. No LR
compensation for the larger optimizer minibatch. Input cache 4096 MiB and encoder
activation checkpointing are preserved. Data seeds use the original absolute
epoch schedule; initial RNG is restored from the parent without reseeding.

## Evaluation and checks

Validation120 still runs at local epochs 1, 2, 4, 6 and 8. Canonical H exact raw
matching with lexicographic ties and evaluation tau .3 remains unchanged.
Two consecutive adverse validations do not stop training. The epoch 4 improvement
threshold does not gate epochs 5-8. Validation still selects the final candidate;
the original parent remains eligible. Only the selected candidate receives final
Tune60 comparison with the frozen H3/F4/soft IGA1800 reference.

Numerical and execution guards remain: full-state restore consistency, behavior
likelihood/mask consistency, finite values, soft KL .02, hard KL .04, complete PPO
update accounting, frozen-component checks and GPU identity/resource checks.
If numerical guards prevent the registered PPO budget from completing, record an
abnormal stop; never report it as completion of eight epochs. GPU memory admission
uses 384 inference slots and 128 backward slots on three real-graph windows, plus
the existing 6144 MiB GPU headroom and 1024 MiB engine margin. This capacity test
does not establish full-trajectory memory or numerical equivalence to 64/64.

Reuse only completed baseline/admission evaluations with identical checkpoint,
cases, evaluator source, rules and hashes. Do not reuse any continuation training
state. CUDA restore and capacity admission execute for the new run.

GPU0 UUID: `GPU-744c1334-98c8-5318-e799-7ad15eea1fbf`.
Trainer CPUs: 0-21,64-85; validator: 22-29,86-93; controller: 30-31,94-95.
The experiment cgroup retains the previously authorized unlimited host memory/swap
settings and oomd exemption. Global oomd and the unrelated GPU1 experiment are unchanged.
