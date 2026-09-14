# Stage2 frozen / Stage3 continuing integration — 2026-09-12

Stage2 is imported from `stage2-frozen-20260912`, commit
`7fb8559557dde94fd4c4bc61bb55346f08af2e40`. The publication is based on its
remote history, which descends from `main` at
`dafbbf405342d4cc454f092167ff57c15c1589e8`. The server's older local Git
history is unrelated after the remote history rewrite; it is not force-pushed.

## Merge boundary

- The frozen Stage2 bundle and its original checkpoint/report bytes are intact.
  Stage2 development launch guards and the explicit portable B0 handoff are retained.
- The server's Stage3 modules, launchers, tests and research plans are preserved.
  Shared code keeps private/full-depth encoders, role-specific routing, execution
  caches, per-group gradient clipping, counterfactual diagnostics, explicit C0 KL
  references, critical-path-v2 credit and the shared asynchronous validator.
- Shared modules integrate Stage2's optional Ready-head extensions and historical
  resource interfaces. Defaults preserve the legacy model tensor names and
  Hungarian tie-breaking. Stage2's resource-specific routing does not replace
  Stage3's private-role encodings.
- Existing local Stage1 and retired Stage2 development work is not deleted.
  Retired Stage2 launchers are not republished as active research entrypoints.
  Stage1 helpers required by Stage3 regression tests are retained as dependencies.
- No new server datasets, labels, training checkpoints, experiment logs, credentials
  or migration archives are published. The already-published, SHA-bound Stage2
  release under `artifacts/stage2_frozen/20260912` is the explicit exception.

## Checkpoint and running-experiment boundary

The continuing full-data Stage3 experiment uses the historical C0 with SHA-256
`41cfc782c1ff908bb527c0219b75b73a1421d48a2dafa02d449015abbe5ab030`.
The frozen Stage2 B0 has SHA-256
`b7740b2b688c83dc7fa65bed6381243cdf2f6675f4b96808ff05a41d4e44e7e8`.
They are not interchangeable baselines.

`--stage2_frozen_manifest` explicitly initializes a **new** Stage3 run, with
fresh optimizer/normalization state; it cannot be combined with `--checkpoint_dir`.
Existing Stage3 studies retain their original manifests and source checkpoints.
Their experiment-specific data, source lineage and admission checks remain
required; checking out this repository does not recreate those private artifacts.

The running `stage3_full_data_all_20260912_r1_recovery1` service executes its
immutable `result/hkbz_train_logs/.../source` snapshot, not the mutable workspace.
This merge does not restart it, edit its manifest, or replace its checkpoints.

## Verification

Use a matching PyTorch/PyG environment. CPU checks can be run with
`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

```bash
python onpolicy/scripts/train/verify_stage2_frozen.py
python scripts/export_stage2_frozen.py --verify-only
python -m pytest -q onpolicy/envs/HKBZ/test/test_stage2_frozen.py \
  onpolicy/envs/HKBZ/test/test_stage2_freeze_guard.py \
  onpolicy/envs/HKBZ/test/test_stage2_stage3_integration.py
```

Stage3 regression coverage includes full/shared encoders, replay/cache update
equivalence, checkpoint continuation, sampling and CPU multi-rank gradient
synchronization. Real distributed tests require permitted local loopback sockets.
GPU equivalence canaries require explicit hardware coordination; they are not
automatically launched by the merge or by default CPU tests.
