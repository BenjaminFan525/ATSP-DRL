# Stage3: drop the two post-pass full replays at a checkpoint boundary

## Material Passport

- Mode: user-authorized protocol change applied at a committed checkpoint boundary.
- Authorization: 2026-09-18, "去掉两次轮后整批回放，从最近的checkpoint继续训练"
  and "续训到epoch10".
- Verification: selected CPU suite (133 tests), exact serialized state comparison
  during migration, then native CUDA canary and 192-wide capacity admission.
- Research status: throughput benefit predicted from the measured epoch structure;
  learning effect is unverified and remains an open question for this round.

## Epoch budget extension

The same switch extends the local budget from 8 to 10 epochs, so the origin epoch-3
checkpoint continues through epochs 4-10 with no second migration. Registered
budgets are {8: epochs 1,2,4,6,8} and {10: epochs 1,2,4,6,8,10}; nothing else is
accepted. `planned_new_steps` becomes 6,10,14,18,22,26,30,34,38,42 for local
epochs 1-10, so the run ends at 138 cumulative PPO updates and 3840 training
visits. The visit schedule is a strict extension of the origin schedule: the
first eight rows are byte-identical and epochs 9-10 continue the same
parent-absolute-epoch seeding formula.

The original 120-hour budget clock is inherited unchanged, so the extension
consumes the remaining wall budget rather than opening a new one; the projected
finish stays well inside it.

## What changes

Origin: `stage3_h3_r0e8_fresh_20260918_r6_env384_mb192_gpu0`
Target: `stage3_h3_r0e8_fresh_20260918_r7_env384_mb192_nopost_gpu0`
Profile: `single_parent_env384_mb192_nopost_v1`, flag
`post_pass_replay = skip_minibatch_derived_v1`.

Every epoch currently runs three whole-batch replays: one before the PPO passes
(behaviour-likelihood contract check) and one after each of the two passes
(whole-batch post-update KL). The change removes only the two post-pass replays.
The pre-update replay, every minibatch replay inside the backward pass, the
optimizer, the collection and the checkpoint format are untouched.

The per-pass record becomes a derived summary of the applied minibatch pre-step
records of the same pass. It carries `source = minibatch_pre_step_derived`,
`post_pass_replay = skipped` and `excludes_final_step = true`, because a
minibatch pre-step value is measured before that minibatch's optimizer step.
`decisions`, `kl` and `clip_fraction` keep their names so existing health
analysis keeps working.

## Why the removed gate was already redundant

The pass gate compared the whole-batch KL against soft 0.02 / hard 0.04. Each
minibatch already computes the same quantity - drift relative to the recorded
rollout policy - and applies both thresholds before its own Adam step. The
derived pass value is the maximum over exactly those minibatch values, so the
pass-level gate could never fire independently of the gate that still runs.

Measured margins in this round: max pre-step KL 5.7e-5 / 6.6e-5, max derived
pass KL 1.3e-4 / 2.1e-4, clip fraction 0.027% / 0.004%. The thresholds are two
orders of magnitude away and have never fired.

## What is given up

After the change there is no whole-batch KL measurement after the final
optimizer step of each pass. The last step of a pass is covered indirectly by
the next minibatch pre-step check, and the final step of the whole update is not
measured at all. Case-level churn in this round was already +/-10% on 8-16 of
120 cases with no net effect, so this is a real but narrow reduction in
monitoring, not a change to the gradient path.

## Accounting and admission

Step accounting is identical to the origin for every epoch: local epoch 1 was 6
updates at 128/128, epochs 2 and beyond are 4 updates at 192/192. The historical
boundary stays at epoch 1, so `planned_new_steps` gives 6, 10, 14, 18, 22, 26,
30, 34 for epochs 1-8 and the origin ledger validates unchanged.

The watcher waits for the complete epoch-3 checkpoint plus its atomic commit, and
only then prepares and verifies the migration. Model, both Adam states,
ValueNorm, RNG and the cursor must compare exactly; only manifest/recipe metadata
is rebound. The new suite then runs its own canary, canary-resume and 192-wide
capacity probe before formal training continues at epoch 4.

GPU0 (`GPU-744c1334-98c8-5318-e799-7ad15eea1fbf`) and CPU 0-31,64-95 stay
bound with the same trainer/validator/controller subdivisions. The original
120-hour budget clock is inherited. Effect early stopping stays disabled;
Validation120 still runs at epochs 4, 6 and 8.

## Expected effect

The two removed replays measured 1,308 s and 1,209 s in epoch 2, so the epoch is
expected to fall from 3.01 h to about 2.31 h (roughly 23% shorter). That is a
throughput change only; whether the resulting six epochs reach a better
Validation120 checkpoint than the parent is exactly what this round is meant to
measure.
