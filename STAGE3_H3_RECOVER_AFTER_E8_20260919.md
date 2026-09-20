# Stage3 recovery: resume the epoch-10 run from the epoch-8 checkpoint

## Material Passport

- Mode: source-only recovery continuation after a worker crash.
- Authorization: 2026-09-19, "续训到epoch10" (this recovery serves the same instruction).
- Verification: selected CPU suite (160 tests, including a regression test for the
  exact crash), exact serialized state comparison during migration, then canary and
  capacity admission in the new suite.

## What happened

`stage3_h3_r0e8_fresh_20260918_r8_env384_mb192_nopost_e10_gpu0` trained epochs 1-8
and committed all eight checkpoints, but its epoch-9 worker exited immediately with
`ValueError: Unregistered epoch`.

Cause: `Controller.arm_manifest()` rebuilds the arm recipe from the parent's recipe
and forwarded the execution profile, the historical boundary and the stopping policy
but not the epoch budget. The suite manifest carried `epochs = 10` while the arm
recipe silently fell back to the default 8, so the worker rejected epoch 9. The same
class of omission had already dropped the budget once (the r7 switch); the fix here
closes it at the arm level and is guarded by a regression test.

## Recovery

Origin: `stage3_h3_r0e8_fresh_20260918_r8_env384_mb192_nopost_e10_gpu0` (failed, GPU idle)
Target: `stage3_h3_r0e8_fresh_20260919_r9_env384_mb192_nopost_e10_gpu0`

The new suite copies the same execution profile (192/192, 384 environments, no
post-pass whole-batch replays) and the same 10-epoch budget, but from the corrected
source snapshot. Migration imports the eight committed epochs and the completed
Validation120 results for epochs 1, 2, 4, 6 and 8; model, both Adam states,
ValueNorm, RNG and cursor must compare exactly and only manifest/recipe metadata is
rebound.

Training resumes at epoch 9 and runs to epoch 10, with Validation120 at epoch 10 and
the final selection plus Tune60 comparison afterwards. The original 120-hour budget
clock is inherited. GPU0 and CPU 0-31,64-95 stay bound; effect early stopping stays
disabled; IGA remains frozen with Validation120 as the only comparison basis.

## Evidence preserved

The failed r8 run keeps its eight commits, its five Validation120 results and its
worker log; nothing is overwritten or deleted. The epoch-9 worker directory contains
the original traceback.

| checkpoint | Validation120 mean | vs parent R0 E8 |
|---|---:|---:|
| r8 epoch 4 | 8249.5 | −0.360% |
| r8 epoch 6 | 8227.7 | −0.623% |
| r8 epoch 8 | see `arms/C03/validation/epoch_0008.json` | — |
