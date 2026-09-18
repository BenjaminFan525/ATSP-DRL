## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: run; authorized checkpoint-boundary update
- Status at 2026-09-18 11:18 CST: WAITING_FOR_CHECKPOINT, not yet using 128/128
- Verification: 106 selected CPU tests passed; complete CPU migration preview passed using real epoch-1 state
- Native GPU restore, 128-wide capacity and full-epoch throughput: pending the checkpoint boundary

# Scheduled update

Service `hkbz-stage3-h3-single-parent-20260918-r4-env384-mb128-gpu0.service`
started successfully at 11:18 CST. PID 250883 is a CPU-only watcher, using about
24 MiB host memory and no GPU allocation. It verified root cgroup ownership,
`user.oomd_omit=1`, unlimited memory/swap cgroup limits and CPU 0-31,64-95.

Original controller PID 146303 and GPU0 model PID 152945 are still running.
At verification the original run was epoch 2, PPO pass 2, minibatch 2.
No complete epoch-2 checkpoint exists yet, so no switch or migration has occurred.
The earlier sudo stop command failed without stopping the training service;
no process signal was sent during setup. The final systemd-run launch succeeded
through the host service manager without sudo.

Target directory:
`result/hkbz_train_logs/stage3_h3_single_parent_20260918_r4_env384_mb128_gpu0`.
Its source contains 495 frozen files. Request SHA256:
`b35646922abcccc3dae558ad369a1510effa8491f962fcb65d75bed3c48386db`.

The watcher requires the completed epoch-2 checkpoint and validated atomic
commit ledger (120 cumulative PPO steps) before preparing a new manifest.
It compares copied checkpoint model/Adam/normalizer/RNG/cursor state exactly,
then stops only the bound old controller and starts the new controller.
Next training epoch is 3, using minibatch/microbatch 128/128 and six PPO updates
per epoch; 384 environments, one model, GPU0 and the original half-CPU allocation
remain in place. Missing Validation120 and existing promotion gates remain active.

Live evidence: `switch_status.json`, `switch_memory_runtime.json`,
`cpu_migration_preview.json`, `launch_attempt.json`, `tests.xml` and
`switch_request.json` in the target directory. After application also inspect
`resume_migration.json`, `switch_applied.json` and `run_status.json`.
