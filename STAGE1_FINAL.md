# Stage 1 final hand-off

Stage 1 is closed. The retained training framework is M2 (`BC + PPO`, annealed
BC-reference KL) with three formal seeds. The later closure screen did not
produce an arm that improved the common M2 baseline, so it must not be used as
a new formal starting point.

## Final result

- Winner: `M2_bc_kl_anneal`.
- Formal held-out three-seed mean Cmax: `8140.40`.
- Blind-test three-seed mean/std Cmax: `8044.06 / 50.38`.
- IGA-180 mean Cmax: `8186.00`; M2 is better by `141.94`.
- IGA-1800 mean Cmax: `7737.85`; M2 remains worse by `306.21` (`3.96%`).
- Completion rate: `100%`; cycle count: `0`.
- The predeclared “near IGA-1800” and “reach IGA-1800” goals were not met.

The authoritative checkpoint paths, SHA256 values, architecture semantics and
the matching formal command hashes, and the transition decision are frozen in
`onpolicy/config/stage1_m2_handoff.json`. A Stage-2 job must verify that file
and start from the matching seed's `checkpoint_Best.pt`; it must never use a
closure-screen checkpoint or `checkpoint_DeviceBC.pt` as its source.

## Retained Stage-1 framework

The retained path includes deterministic balanced sampling, DAgger PlaneBC,
tail-weighted BC, IGA potential shaping, joint-team PPO, annealed BC-reference
KL, shared evaluation, safe graph batching/prefetch, asynchronous rollout
support, checkpoint recovery and the M2 three-seed artifacts.

Residual replay, profile-only BC-reference anchors and the P0-P5 closure
orchestration were research branches, not successful framework components.
Their runtime switches, buffers, controllers and raw run artifacts have been
removed.

The final two-seed closure screen used a common pre-PPO composite Cmax of
`8294.08`. Its epoch-2 results were:

| Arm | Mean composite Cmax | Decision |
| --- | ---: | --- |
| P0 | 8408.10 | reject |
| P1 | 8456.93 | reject |
| P2 | 8373.84 | reject |
| P3 | unavailable | failed before training |
| P4 | 8481.62 | reject; configured tail BC was a no-op because PlaneBC epochs were zero |
| P5 | 8448.76 | reject |

All completed arms had 100% completion and no cycles, but every one regressed
against the common source. No closure arm was authorized for a formal run.

## Stage-2 boundary

Stage 2 is the single canonical `resource_joint` stage:

1. restore and strictly validate the immutable M2 plane/shared tensors;
2. warm up only the resource actor with resource BC;
3. reset optimizers;
4. run resource PPO while keeping the shared encoder and plane actor frozen;
5. require bitwise protected-parameter evidence and source-M2 lineage in the
   warm-up, Best, Last and run-status artifacts.

There is no planned full-joint/S4 transition. Tune, gate and final-blind
resource-joint evaluation sets remain separate; final-blind is opened only
after the Stage-2 winner is locked.
