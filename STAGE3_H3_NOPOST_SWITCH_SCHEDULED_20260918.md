# Post-pass replay removal + epoch-10 extension scheduled

Verified at 2026-09-19 00:49 CST. The CPU-only watcher is active
(PID 473856, phase `waiting_for_checkpoint`); the origin trainer is untouched
(controller 356900, worker 457453, GPU0 37.3 GiB).

- Origin: `stage3_h3_r0e8_fresh_20260918_r6_env384_mb192_gpu0`.
- Target: `stage3_h3_r0e8_fresh_20260918_r7_env384_mb192_nopost_gpu0`.
- Unit: `hkbz-stage3-h3-r0e8-fresh-20260918-r7-env384-mb192-nopost-gpu0.service`.
- Request SHA256: `97b7841244e74449ca391be0c828542371be4fbefc8ad54fe09c7564d2474e28`.
- Frozen source: 498 files; nine reviewed files differ from the origin snapshot.
- Validation: 156 selected CPU tests passed.
- Changes: drop the two post-pass whole-batch replays (the pre-update replay and every
  minibatch backward replay stay); extend the local budget to 10 epochs with
  evaluation epochs 1,2,4,6,8,10.
- Accounting: planned new updates 6,10,14,18,22,26,30,34,38,42 for epochs 1-10;
  the run ends at 138 cumulative updates and 3840 training visits.
- Trigger: complete epoch-3 checkpoint plus matching atomic commit (14 new updates).
- Effective from epoch 4; the original 120-hour budget clock is inherited.
- Expected epoch time: 3.01 h to about 2.34 h.

Plan: `STAGE3_H3_NOPOST_SWITCH_20260918.md`.
Audit: `result/hkbz_switch_audits/20260918_nopost_after_e3`.
Monitor `/home/fanyx/HKBZ-environment/result/hkbz_train_logs/stage3_h3_r0e8_fresh_20260918_r7_env384_mb192_nopost_gpu0/switch_status.json`; after application `switch_applied.json`,
`resume_migration.json` and `run_status.json` carry the evidence.
