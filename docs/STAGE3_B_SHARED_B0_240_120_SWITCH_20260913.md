# Stage3 B_SHARED: minibatch 120 after batch 11

The user clarified that the change starts at batch 12, immediately after the
current batch 11 checkpoint is committed. This is inside data epoch 1; the next
data epoch starts at batch 14.

The target keeps GPU0, CPUs `0-31,64-95`, 240 environment slots, 64 CPU environment
workers, one sampling/training model, global batch 240, two PPO passes, TBPTT 8,
the 4 GiB input cache, and GNN activation recomputation. Physical PPO microbatch
changes from 80 to 120, giving two accumulated microbatches per full group. The
complete visit schedule, optimizer state, normalization and RNG state are
preserved. The original numerical-equivalence waiver remains recorded.

The switch supervisor requires the final `commits/batch_0011.json`, validates
its checkpoint/update hashes and the contiguous committed prefix, verifies the
original process identity, then stops only the original Stage3 service. A
partially started batch 12 is never imported. Capacity probing begins only
after GPU0 is free. The probe performs a fresh 26-step sampling/replay window
and two dense 32-step backward windows at microbatch 120; it makes no Adam
updates. Successful admission freezes a new source snapshot, performs exact
checkpoint restore checks and launches the continuation at batch 12. A failed
probe or preparation resumes the original microbatch 80 configuration and
records the failure; it never reports minibatch 120 as active.

The parameter/admission and boundary tests passed: 62 tests. The running r6
source and runtime remain frozen. These changes affect only the prepared
continuation and its separate supervisor.

At 2026-09-13 14:38 CST, the supervisor was active in `waiting_checkpoint`, with
10 committed batches. The original trainer remained active in batch 11's final
likelihood replay. Minibatch 120 had **not** started.

- Supervisor: `hkbz-b0-switch-mb120-1789281450.service`.
- Control: `result/hkbz_train_logs/stage3_b_shared_b0_switch_mb120_20260913/`.
- Parent: `result/hkbz_train_logs/stage3_b_shared_b0_local_a6000_20260913_r6_env240_mb80_gpu0/`.
- Planned continuation: `result/hkbz_train_logs/stage3_b_shared_b0_local_a6000_20260913_r7_env240_mb120_gpu0/`.
- Planned probe: `result/hkbz_train_logs/stage3_b_shared_b0_capacity240_120_recompute_20260913/`.

Use the control directory's `status.json` for the transition phase,
`boundary_checkpoint.json` for the captured checkpoint and `result.json` for
verified launch/restoration. `failure.json` and `fallback.json` identify a
failed change and any recovery to the original recipe. After a successful
switch, read the r7 `run_status.json`, latest attempt heartbeat and commits for
training progress.

## Recovery and verified launch at 15:14 CST

Batch 11 committed at 14:52:58 with 592 visits and 22 updates. The 240/120
capacity probe passed, and the frozen r7 continuation passed 57 tests plus the
exact CPU restore audit. The switch supervisor then failed before launch:
its child launch receipt reused the supervisor's `launch_command.json` name.
The automatic fallback resumed microbatch 80 from the saved batch 11.

The receipt collision was fixed by using `step_<name>_command.json` for child
commands. All six boundary/launch-record regression tests passed. The original
failure, fallback and pre-recovery status remain in the control directory.
The uncommitted fallback sampling (about 16.45 minutes) was stopped, and the
already verified r7 continuation started at 15:14:39 from batch 11.

Active r7 service: `hkbz-b0-resume240-1789283675.service`. The live trainer
confirmed exact model, Adam, normalization and CPU/CUDA RNG restoration,
240 live environment slots, one model, and batch 12. Runtime microbatch is 120.
The control directory's `result.json` and the r7
`scheduled_switch_verified.json` record the completed recovery. The original
`failure.json` is historical and does not describe the current run.
