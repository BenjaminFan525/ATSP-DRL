## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: implementation and authorized restart
- Authorization: user requested 96 parallel environments per experiment, memory protection disabled, restart
- Verification: pending execution checks; no scientific benefit claimed

# H3 continuation restart with 96 environments per arm

The interrupted r2 run has no formal continuation checkpoint. Both arms restart
from the same frozen R0 E8 parent with 96 inherited PPO updates. Preserve its
source, diagnostics, logs and failure evidence; create a new immutable r3 run.

| Setting | C03 | T10 |
|---|---|---|
| GPU | GPU0 | GPU0 |
| CPU logical IDs | 0–15,64–79 | 16–31,80–95 |
| Train tau | 0.03 | 0.1 |
| Parallel environments / CPU workers | 96 / 32 | 96 / 32 |
| Unique train cases / visits per epoch | 240 / 384 | 240 / 384 |
| Sampling groups per epoch | 4 | 4 |
| Optimizer minibatch / physical microbatch | 64 / 64 | 64 / 64 |
| PPO passes / updates per epoch | 2 / 12 | 2 / 12 |
| PyTorch allocator / input cache | 18 GiB / 1 GiB | 18 GiB / 1 GiB |

CPU affinity is set with taskset before Python starts, covering initialization
threads and descendants. GPU1 and CPU 32–63,96–127 remain outside this run.
The learning recipe, case/seed schedule, E2/E4/E8 gates and frozen IGA references
are preserved. Changing sampling width changes stochastic trajectories; this is
not a claim of 128-versus-96 trajectory equivalence.

The user-requested memory setting is scoped to this experiment: no memory.high,
memory.max or memory.swap.max limit; systemd-oomd omission on a root-owned service
cgroup. Training itself still runs as user fanyx. The root systemd service is
necessary because this host's systemd 249 ignores omit attributes on user-owned
cgroups. Keep the host-wide oomd service and kernel global OOM behavior unchanged.
The controller verifies actual cgroup ownership, omission and every ancestor's
limits before starting GPU work; a nominal configuration is insufficient.

Each arm still retains a complete 384-trajectory batch. Lowering concurrent
environments does not make that retained batch 96 trajectories. Resource sampling
records RAM, swap, pressure and limit events in addition to GPU usage.

Reuse only the five completed evaluation baselines after existing source/input/
request identity checks. Run both two-case canaries, fresh-process state restores
and native 96-slot/64-trajectory GPU window checks before formal epoch 1. The
window probe does not establish complete-epoch CPU memory capacity or speed.
Formal training provides the first complete-batch memory observation; no memory
pressure auto-kill or hidden restart is added. Existing research wall-time limits
and peer-failure cancellation remain in force. IGA solver queries stay zero.
