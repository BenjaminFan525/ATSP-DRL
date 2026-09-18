## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: authorized implementation and restart
- Authorization: stop current round; continue only the previous best checkpoint using previous parameters
- Verification: pending execution checks; no new training benefit claimed

# Single continuation from the previous selected R0 E8

The user stopped the C03/T10 r3 env96 experiment. It has no committed formal
epoch. Preserve its source, logs, rollouts and stop receipt. Create a fresh run:
`stage3_h3_single_parent_20260918_r1_env240_gpu0`.

The previous R0 final selection is epoch 8, SHA256
`14d63bb44ee96eea38c993bb7d1d1d1ae5f0aa59f979b14a9337e6f96215a4e6`.
Use this exact checkpoint, including actor/critic Adam, ValueNorm and RNG; inherit
96 completed PPO updates. No T10 arm or alternate parent is launched.

Restore the previous learning/execution recipe: B_SHARED, H3/F4/soft, train tau
0.03, 240 concurrent environments, 64 CPU environment workers, one GPU model,
384 trajectories per epoch, optimizer minibatch/microbatch 64/64, two PPO passes
and 12 optimizer steps per epoch. Input cache is 4096 MiB. Actor LR 1e-5 with
role scales [1,.25,1,.5], critic LR 1e-4, PPO clip .2, gradient clip 1, TBPTT 8,
soft/hard KL .02/.04 and source-relative advantages are unchanged.

Keep Train240 and IID once / OOD four-times weighting. Continue the original
seed 2026091502 and original case-order algorithm at absolute data epochs 9-16.
Optimizer shuffling continues from seed 2026091506 + absolute epoch. Restore
parent RNG without resetting to the canceled temperature-study seed. New local
epochs 1-8 correspond to cumulative epochs 9-16. Maximum extra budget is 3072
visits and 96 optimizer steps, with the existing E2/E4/early-stop allocation rules.

Only GPU0 and CPU 0-31,64-95 are authorized. Use taskset before Python starts,
then preserve the previous trainer/validator CPU layout. Retain the user's
disabled experiment RAM limits and effective systemd-oomd exemption; do not
change the host-wide oomd service or kernel OOM handling. Use a root-owned
systemd cgroup with training User=fanyx and a 120-hour hard deadline.

Use canonical H evaluation with fixed numerical/tie rules, tau .3 and 12 fixed
slots. Reuse the five bound baseline evaluation groups when hashes/contracts
match. Run one two-case canary and its fresh-process restore check for C03 only;
do not repeat a microbatch sweep or run T10 admission. Save each complete epoch;
Validation120 at local epochs 1,2,4,6,8; select by Validation before final Tune60.
IGA remains frozen, solver queries zero, Confirmation unopened.

The E2 harm and E4 extension gates from the registered continuation remain:
two consecutive harmful validations stop; extending beyond E4 requires at least
0.3% mean improvement versus the parent, nonworse OOD stress and risk eligibility.
Only this single arm is eligible. Follow-on algorithm studies are not automatic.
