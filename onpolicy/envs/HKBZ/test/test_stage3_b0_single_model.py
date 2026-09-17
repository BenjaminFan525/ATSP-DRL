"""Single-policy collection contracts, including ragged recurrent trajectories."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from onpolicy.runner.shared.stage3_b_shared_b0_engine import BSharedB0Engine
from onpolicy.utils.stage3_b0_single_model import ThroughputEngine, CpuEnvironmentPool
from onpolicy.utils.stage3_b0_throughput import ThroughputEngine as UpdateEngine


class Pool:
    def __init__(self):
        self.steps = [0, 0]
        self.actions = [[], []]

    def info(self, i):
        return dict(active_agents=np.ones(104), agent_types=np.arange(104) % 3,
            env_total_time=self.steps[i], last_op_indices=np.full(24, 10+i),
            last_site_indices=np.full(24, 20+i))

    def call(self, commands):
        result = {}
        for i, command, payload in commands:
            if command == 'summary':
                result[i] = dict(completed=True, makespan=float(self.steps[i]))
                continue
            done = np.zeros(104, bool)
            if command == 'step':
                self.actions[i].append(payload.copy())
                self.steps[i] += 1
                done[0] = True
                if self.steps[i] >= (1 if i == 0 else 3):
                    done[:] = True
            result[i] = (i, None, done, self.info(i)) if command == 'step' else (i, None, self.info(i))
        return result


def toy(batched=True):
    r = object.__new__(BSharedB0Engine)
    r.batched_sampling = batched
    r.pool = Pool()
    r.device = torch.device('cpu')
    r.config = dict(rollout_max_steps=10)
    r.policy_updates = 18
    r.environment_config = lambda case, seed: dict(seed=seed)
    calls = []

    def get(graph, hidden, active, op, site, **kw):
        calls.append(dict(graph=graph[:], hidden=hidden.copy(), op=op.copy(), site=site.copy()))
        n = len(graph)
        actions = torch.randint(0, 100, (n, 104, 3))
        logp = torch.zeros(n, 104)
        mask = torch.as_tensor(active).float()
        return None, actions, logp, torch.as_tensor(hidden)+1, mask

    r.policy = SimpleNamespace(ac=SimpleNamespace(tau=.03), get_actions=get)
    return r, calls


def collect(r):
    return r._collect([dict(path=str(i), profile='test', distribution='iid') for i in range(2)],
        [42, 99], deterministic=False, retain=True, decoder='AR_sample', native=False,
        heartbeat=None, record_times=False)


def test_batched_collection_keeps_ragged_slots_history_and_per_agent_resets():
    r, calls = toy()
    rows = collect(r)
    assert [c['graph'] for c in calls] == [[0, 1], [1], [1]]
    assert [row['steps'] for row in rows] == [1, 3]
    assert np.all(calls[1]['hidden'][0, 0] == 0)
    assert np.all(calls[1]['hidden'][0, 1:] == 1)
    assert np.all(calls[2]['hidden'][0, 1:] == 2)
    assert np.all(calls[0]['op'][0, :24] == 10)
    assert np.all(calls[1]['site'][0, :24] == 21)
    for i, row in enumerate(rows):
        assert all(np.array_equal(state['action'], action)
            for state, action in zip(row['states'], r.pool.actions[i]))
        assert all(state['mask'].sum() == 104 for state in row['states'])


def test_batched_collection_consumes_and_replays_the_checkpoint_rng():
    torch.manual_seed(771)
    before = torch.get_rng_state()
    a, _ = toy()
    first = collect(a)
    after = torch.get_rng_state()
    assert not torch.equal(before, after)
    torch.rand(70)
    torch.set_rng_state(before)
    b, _ = toy()
    second = collect(b)
    assert [r['actions_sha256'] for r in first] == [r['actions_sha256'] for r in second]
    assert torch.equal(after, torch.get_rng_state())


def test_legacy_sampler_remains_batch_one_and_seeded_per_visit():
    r, calls = toy(batched=False)
    collect(r)
    assert [c['graph'] for c in calls] == [[0], [1], [1], [1]]


def test_single_model_reuses_the_admitted_update_and_replay_methods():
    assert ThroughputEngine.update is UpdateEngine.update
    assert ThroughputEngine._backward is UpdateEngine._backward
    assert ThroughputEngine._replay is UpdateEngine._replay
    assert ThroughputEngine.replay_metrics is UpdateEngine.replay_metrics


def test_cpu_pool_hides_cuda_during_spawn_and_restores_parent(monkeypatch):
    import os
    from onpolicy.utils import stage3_b0_single_model as mod
    observed = []
    class Connection:
        def close(self): pass
    class Process:
        def __init__(self, *, target, args, daemon):
            assert target is mod._cpu_environment and daemon
            self.affinity = args[1]
        def start(self): observed.append((os.environ['CUDA_VISIBLE_DEVICES'], self.affinity))
    context = SimpleNamespace(Pipe=lambda: (Connection(), Connection()), Process=Process)
    monkeypatch.setattr(mod.mp, 'get_context', lambda _: context)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU0-test')
    pool = CpuEnvironmentPool(4, timeout=1, affinity=[0, 1, 64, 65])
    assert len(pool.processes) == 4
    assert observed == [('', {0, 64}), ('', {1, 65}), ('', {0, 64}), ('', {1, 65})]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU0-test'


def test_single_model_rejects_multiple_model_lanes_before_spawning(tmp_path):
    r = object.__new__(ThroughputEngine)
    r.training = True
    r.runtime = dict(sampling_lanes=16)
    with pytest.raises(ValueError, match='one inference owner'):
        r.parallel_collect([{}], [42])


def test_single_model_admission_rejects_wrong_checkpoint_masks_or_changed_ppo(tmp_path):
    import copy
    from onpolicy.utils.stage3_b0_restart import MODE, verify_single_model_sample
    from onpolicy.utils.stage3_research import atomic_json, digest_file
    paths = ('onpolicy/utils/stage3_b0_single_model.py',
             'onpolicy/runner/shared/stage3_b_shared_b0_engine.py',
             'onpolicy/utils/stage3_b0_throughput.py',
             'onpolicy/scripts/train/probe_stage3_b0_single_model.py')
    old, new = tmp_path/'old', tmp_path/'new'
    code = {}
    for root in (old, new):
        for relative in paths:
            f = root/relative
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text('class BSharedB0Engine:\n    def update(self): return 1\n')
            code[relative] = digest_file(f)
    def bind(name, content):
        f = tmp_path/name
        atomic_json(f, content)
        return dict(path=str(f), sha256=digest_file(f))
    a = dict(mode=MODE, numerical_equivalence_gate='waived_by_user', user_instruction='authorized',
        completed_batches=9, sampling_architecture='single_model_batched_v1',
        parent_manifest=bind('parent.json', dict(manifest_sha256='parent', source_root=str(old))),
        parent_commit=bind('commit.json', dict(checkpoint_sha256='checkpoint', next_batch=9)),
        sampling_probe=dict(path='probe', sha256='proof'))
    m = dict(resume_amendment=a, source_root=str(new))
    runtime = dict(sampling_lanes=1, global_batch=64,
        module=str(new/paths[0]), proofs=dict(sample=a['sampling_probe']))
    sample = dict(passed=True, architecture='single_model_batched_v1', complete_trajectories=64,
        environment_count=64, model_copies=1, first_forward_batch=64,
        environment_cuda_visible_devices='', checkpoint_restore_exact=True,
        cuda_rng_restored_exact=True, repeated_forward_and_rng_exact=True,
        rng_contract='checkpoint_global_torch_live_slot_order_v1', parent_manifest_sha256='parent',
        checkpoint_sha256='checkpoint', next_batch=9, updates_executed=0,
        replay=dict(decisions=12, max_logp_error=.0001), code_files=code)
    verify_single_model_sample(m, runtime, sample)
    for key, value in [('checkpoint_sha256', 'other'), ('model_copies', 16),
                       ('first_forward_batch', 1), ('complete_trajectories', 63),
                       ('repeated_forward_and_rng_exact', False),
                       ('replay', dict(decisions=12, max_logp_error=float('nan')))]:
        broken = copy.deepcopy(sample)
        broken[key] = value
        with pytest.raises(ValueError, match='proof failed'):
            verify_single_model_sample(m, runtime, broken)
    f = new/paths[1]
    f.write_text('class BSharedB0Engine:\n    def update(self): return 2\n')
    with pytest.raises(ValueError, match='source changed'):
        verify_single_model_sample(m, runtime, sample)
    sample['code_files'][paths[1]] = digest_file(f)
    with pytest.raises(ValueError, match='changed inherited PPO'):
        verify_single_model_sample(m, runtime, sample)
