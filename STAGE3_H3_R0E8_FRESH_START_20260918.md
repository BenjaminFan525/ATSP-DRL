# Fresh R0 E8 Stage3 launch record

- Started: 2026-09-18T14:10:12.527225+08:00.
- Run: `result/hkbz_train_logs/stage3_h3_r0e8_fresh_20260918_r5_env384_mb128_gpu0`.
- Systemd unit: `hkbz-stage3-h3-r0e8-fresh-20260918-r5-env384-mb128-gpu0.service`.
- Controller PID at launch: 322121; initial CUDA canary worker: 322125.
- Suite manifest SHA256: `9b9db69fd1157230cdfbbae95d3e86654c1766591495705255d6bd4deb5a015a`.
- Arm manifest SHA256: `32ebc3a18208751eac531e38be1fb0203781be6b7473722ea2a591b455c32232`.
- Source snapshot: 496 files verified; selected CPU tests: 116 passed.
- Original R0 E8 restored with 96 inherited updates; physical microbatch 128 confirmed.
- 384 environments, optimizer/physical batch 128/128, one C03 flow, Train240.
- Eight local epochs (3072 visits, 48 new PPO updates); outcome early stop and epoch4 extension gate disabled.
- GPU0 and CPU 0-31,64-95 verified in live systemd allocation. GPU1 runs a separate experiment.
- Completed baseline evaluations reused by matching source, checkpoint, cases and hashes. No continuation training state imported.
- Launch-time phase: CUDA restore canary. New-process restore comparison and 128-wide capacity admission precede formal epoch1 automatically. Their completion is not claimed in this launch record.
- Numerical, KL and GPU memory checks are retained; IGA stays frozen.

Plan: `STAGE3_H3_R0E8_FRESH_FIXED8_20260918.md`.
Audit: `result/hkbz_switch_audits/20260918_r0e8_fresh_mb128_fixed8` (tests XML, prior source copies, scoped diff and launch verification).
Monitor `run_status.json`, `arms/C03/canary/passed.json`, `arms/C03/capacity/result.json`,
`attempts/*/C03/epoch_0001/status.json` and the live systemd service for current progress.
