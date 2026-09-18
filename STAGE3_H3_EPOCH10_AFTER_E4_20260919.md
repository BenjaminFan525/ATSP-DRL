# Stage3: extend the local budget to epoch 10 at the epoch-4 boundary

## Material Passport

- Mode: user-authorized epoch-budget extension applied at a committed checkpoint boundary.
- Authorization: 2026-09-19, "续训到epoch10".
- Verification: selected CPU suite (159 tests), exact serialized state comparison
  during migration, then native CUDA canary and 192-wide capacity admission.

## Why a second switch is needed

The switch that took over at the epoch-3 boundary (`r7`) applied the
post-pass-replay removal correctly, but its recipe was built with the default
budget of 8 epochs: the prepared request carried `requested_epochs = 10` and
`prepare_at_boundary` never passed it into `recipe()`. The frozen r7 suite
therefore runs epochs 4-8 with `max_new_ppo_steps = 34`. The wiring is fixed in
this revision and the budget is extended at the next committed boundary.

## What changes

Origin: `stage3_h3_r0e8_fresh_20260918_r7_env384_mb192_nopost_gpu0`
Target: `stage3_h3_r0e8_fresh_20260918_r8_env384_mb192_nopost_e10_gpu0`

Only the epoch budget changes: `epochs` 8 to 10 and `evaluation_epochs` from
[1,2,4,6,8] to [1,2,4,6,8,10]. The execution profile, the removed post-pass
replays, 192/192 batches, 384 environments, learning rates, KL gates, TBPTT,
cache settings, sampling schedule and the checkpoint format are all unchanged.
`planned_new_steps` becomes 6,10,14,18,22,26,30,34,38,42 for local epochs 1-10,
so the run ends at 138 cumulative PPO updates and 3840 training visits.

The visit schedule is a strict extension: the first eight rows are identical to
the origin's and epochs 9-10 continue the same parent-absolute-epoch seeds.

## Boundary, admission and budget

The watcher waits for the complete epoch-4 checkpoint (18 new updates, 114
cumulative) plus its atomic commit before touching anything. The migration copies
the committed epochs 1-4, requiring model, both Adam states, ValueNorm, RNG and
cursor to compare exactly; only manifest/recipe metadata is rebound.

The new suite runs its own canary, canary-resume and 192-wide capacity probe
before formal training resumes at epoch 5. Validation120 still runs at epochs 5,
6, 8 and 10 (epoch 4's validation is redone by the new suite only if the origin
was interrupted before it completed). The original 120-hour budget clock is
inherited, so epochs 9-10 consume remaining wall budget rather than opening a new
one.

GPU0 (`GPU-744c1334-98c8-5318-e799-7ad15eea1fbf`) and CPU 0-31,64-95 stay bound.
Effect early stopping stays disabled and IGA remains frozen with Validation120 as
the only comparison basis.
