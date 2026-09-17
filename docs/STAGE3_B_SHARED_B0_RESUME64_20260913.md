# Stage3 B_SHARED: resume with global64 / micro64

The user authorized ignoring the complete microbatch numerical-equivalence comparison and restarting the preferred throughput configuration from the latest committed checkpoint.

- New run: `result/hkbz_train_logs/stage3_b_shared_b0_local_a6000_20260913_r4_resume64_gpu0`.
- Service: `hkbz-b0-resume64-1789267312.service`, started on 2026-09-13 at approximately 10:41 CST.
- Parent: `stage3_b_shared_b0_local_a6000_20260912_r3_tuned_gpu0`, batch 9, checkpoint SHA256 `f9f521ff3ed52e33af6bf33e1454cd45d9861695fb003f521185d44709252bb1`.
- Execution: GPU0 only; CPUs `0-31,64-95`; 64 environments; 16 sampling processes; global batch 64; microbatch 64; 4096 MiB cache; two PPO passes per batch.

The original nine checkpoints and their logs are retained. Copies in `resume_import` change only the manifest and recipe identity fields. Model parameters, Adam states, normalization statistics, counters, and all RNG payloads are unchanged. A fresh CPU process verified the load; the actual GPU training worker independently verified all restored state including CUDA RNG before collecting the next batch. See `restore_check.json` and the attempt's `train/resume_verified.json`.

The first nine groups still contain 32 visits each: 288 visits and 18 updates are already complete. Only the remaining visits were regrouped, preserving every case, seed, visit ID, ordering, and data-epoch boundary. The first remaining group has 64 visits; the end of the first epoch has a 32-visit group. Remaining work is 7392 visits in 116 groups and 232 updates. The complete mixed-batch schedule therefore has 125 groups and 250 updates, with epoch-ending group indices `[20,35,50,65,80,95,110,125]`.

The numerical-equivalence rejection of the capacity candidate remains recorded as a failure. The amended admission explicitly records `numerical_equivalence_passed: false` and `numerical_equivalence_gate: waived_by_user`; it does not convert that rejection to a numerical pass. Existing finite-value, legal-mask, PPO KL, source identity, checkpoint integrity, and resource guards remain active. B0 baseline, zero-update, and AR-control evidence are reused with provenance.

Validation: 41 CPU tests passed against the frozen source; checkpoint migration and actual GPU restore checks passed; the service entered batch 10 with all 16 sampling lanes active and 64 visits scheduled. `restart_launched.json` records the launch evidence. The parent run's `execution_transition.json` points to this continuation.
