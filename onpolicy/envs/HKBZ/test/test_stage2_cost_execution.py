"""Exact fanout/cache tests independent of GPU availability or pilot outputs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cloudpickle
import numpy as np

from onpolicy.utils.stage2_cost_execution import (
    ExactCostCache, ShadowFanout, array_digest, coalesce_queries, json_digest,
    run_fanout, semantic_outcome,
)


class ToyEnv:
    def __init__(self, target=3):
        self.target, self.position, self.total_time = target, 0, 10.
        self.history = []
        self.np_random = np.random.default_rng(23)

    def _is_schedule_complete(self):
        return self.position >= self.target

    def step(self, actions):
        self.history.append(np.asarray(actions).copy())
        self.position += 1
        self.total_time += int(actions[0, 0]) + 1 + float(self.np_random.random())
        return (np.array([self.position, self.total_time]), 0.,
                np.array([self._is_schedule_complete()]),
                {'active_agents': np.array([1]), 'agent_types': np.array([0])})


class ToyVector:
    def __init__(self, targets):
        self.snapshots = [cloudpickle.dumps(ToyEnv(target)) for target in targets]

    def call(self, method, ids):
        assert method == 'stage2_cost_fanout_restore'
        self.fanouts = [ShadowFanout(snapshot, ids) for snapshot in self.snapshots]

    def call_each(self, method, args):
        assert method == 'stage2_cost_fanout_step'
        return [fanout.step(requests[0]) for fanout, requests in zip(self.fanouts, args)]

    @staticmethod
    def stack_infos(rows):
        return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def context(width=2):
    return dict(policy_sha256='b' * 64, batch_snapshot_sha256=['c' * 64] * width,
                post_forward_recurrent_sha256='d' * 64, history_sha256='e' * 64,
                batch_width=width, max_steps=20, arithmetic='cuda0_full_batch_test')


class CostExecutionTest(unittest.TestCase):
    def test_actor_lane_bounds_reject_before_cuda_allocation(self):
        from onpolicy.utils.stage2_cost_actor_pool import PrivateActorPool
        for lanes in (0, 1, 5):
            with self.assertRaises(ValueError):
                PrivateActorPool(None, lanes)

    def test_array_key_includes_dtype_shape_and_contents(self):
        a = np.array([[1, 2]], np.int64)
        for b in (a.astype(np.int32), a.reshape(-1), a + 1):
            self.assertNotEqual(array_digest(a), array_digest(b))
        self.assertEqual(array_digest(a), array_digest(a.copy()))

    def test_only_identical_full_batch_actions_coalesce(self):
        a = np.zeros((2, 1, 2), np.int64)
        b = a.copy()
        b[1, 0, 0] = 1
        jobs = coalesce_queries([(a, 0), (a.copy(), 1), (b, 0)])
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]['consumers'], [(0, 0), (1, 1)])

    def test_forks_preserve_private_rng_and_live_snapshot(self):
        env = ToyEnv()
        raw = cloudpickle.dumps(env)
        fanout = ShadowFanout(raw, [0, 1, 2])
        actions = [np.array([[x, 0]], np.int64) for x in (0, 0, 1)]
        result = fanout.step([{'job': i, 'action': a} for i, a in enumerate(actions)])
        self.assertEqual(result['requested_steps'], 3)
        self.assertEqual(result['executed_steps'], 2)
        self.assertEqual(result['forks'], 1)
        self.assertIs(fanout.heads[0], fanout.heads[1])
        self.assertIsNot(fanout.heads[0], fanout.heads[2])
        for i, action in enumerate(actions):
            original = cloudpickle.loads(raw)
            expected = original.step(action)
            actual = result['outputs'][result['refs'][i]]
            np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(env.position, 0)

    def test_equal_current_actions_do_not_merge_diverged_histories(self):
        fanout = ShadowFanout(cloudpickle.dumps(ToyEnv()), [0, 1])
        fanout.step([{'job': i, 'action': np.array([[i, 0]], np.int64)} for i in (0, 1)])
        result = fanout.step([{'job': i, 'action': np.zeros((1, 2), np.int64)} for i in (0, 1)])
        self.assertEqual(result['executed_steps'], 2)
        self.assertNotEqual(fanout.heads[0].total_time, fanout.heads[1].total_time)

    def test_finished_jobs_release_states_and_unknown_jobs_fail(self):
        fanout = ShadowFanout(cloudpickle.dumps(ToyEnv()), [0, 1])
        fanout.step([{'job': 1, 'action': np.zeros((1, 2), np.int64)}])
        self.assertEqual(set(fanout.heads), {1})
        with self.assertRaises(ValueError):
            fanout.step([{'job': 0, 'action': np.zeros((1, 2), np.int64)}])

    def test_duplicate_job_and_action_cast_fail(self):
        fanout = ShadowFanout(cloudpickle.dumps(ToyEnv()), [0])
        with self.assertRaises(ValueError):
            fanout.step([{'job': 0, 'action': np.zeros((1, 2), np.int32)}])
        with self.assertRaises(ValueError):
            fanout.step([{'job': 0, 'action': np.zeros((1, 2), np.int64)}] * 2)

    def test_complete_cache_round_trip_and_all_identity_fields(self):
        with TemporaryDirectory() as directory:
            cache = ExactCostCache(directory, 'a' * 64)
            actions = np.zeros((2, 1, 2), np.int64)
            binding = cache.binding(context(), actions, 0)
            expected = dict(completed=True, makespan=100., steps=5, reason='completed', wall_seconds=5.)
            self.assertIsNone(cache.get(binding))
            cache.put(binding, expected)
            cached = cache.get(binding)
            self.assertEqual(semantic_outcome(cached), semantic_outcome(expected))
            self.assertTrue(cached['cache_hit'])
            cache.put(binding, dict(expected, wall_seconds=999.))
            for field, value in [('policy_sha256', 'f' * 64), ('history_sha256', 'f' * 64),
                                 ('post_forward_recurrent_sha256', 'f' * 64),
                                 ('batch_snapshot_sha256', ['f' * 64, 'c' * 64]),
                                 ('arithmetic', 'cpu_proxy'), ('max_steps', 21)]:
                changed = dict(context(), **{field: value})
                self.assertIsNone(cache.get(cache.binding(changed, actions, 0)))
            self.assertIsNone(cache.get(cache.binding(context(), actions, 1)))
            self.assertIsNone(cache.get(cache.binding(context(), actions + 1, 0)))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                cache.put(binding, dict(expected, makespan=101.))

    def test_incomplete_never_cached_and_corruption_fails(self):
        with TemporaryDirectory() as directory:
            cache = ExactCostCache(directory, 'a' * 64)
            binding = cache.binding(context(), np.zeros((2, 1, 2), np.int64), 0)
            cache.put(binding, dict(completed=False, makespan=None, steps=5, reason='wall_timeout'))
            self.assertIsNone(cache.get(binding))
            cache.put(binding, dict(completed=True, makespan=100., steps=5, reason='completed'))
            path = cache.directory / (json_digest(binding) + '.json')
            record = json.loads(path.read_text())
            record['outcome']['makespan'] = 101.
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'changed'):
                cache.get(binding)

    def test_missing_companions_rejected(self):
        with TemporaryDirectory() as directory:
            cache = ExactCostCache(directory, 'a' * 64)
            with self.assertRaises(ValueError):
                cache.binding(dict(context(), batch_snapshot_sha256=['c' * 64]),
                              np.zeros((2, 1, 2), np.int64), 0)

    def test_invalid_complete_is_rejected_before_publication(self):
        with TemporaryDirectory() as directory:
            cache = ExactCostCache(directory, 'a' * 64)
            binding = cache.binding(context(), np.zeros((2, 1, 2), np.int64), 0)
            for changes in ({'makespan': float('nan')}, {'steps': 100}, {'reason': 'wall_timeout'}):
                row = dict(completed=True, makespan=100., steps=5, reason='completed')
                row.update(changes)
                with self.assertRaises(ValueError):
                    cache.put(binding, row)
                self.assertIsNone(cache.get(binding))

    def test_batch_width_and_target_are_part_of_cache_guard(self):
        with TemporaryDirectory() as directory:
            cache = ExactCostCache(directory, 'a' * 64)
            for actions, target in [(np.zeros((1, 1, 2), np.int64), 0),
                                    (np.zeros((2, 1, 2), np.int64), 2)]:
                with self.assertRaises(ValueError):
                    cache.binding(context(), actions, target)

    def test_fanout_matches_serial_target_termination_and_reuses_cache(self):
        actions = np.zeros((2, 1, 2), np.int64)
        changed = actions.copy()
        changed[0, 0, 0] = 2
        queries = [(actions, 0), (actions.copy(), 1), (changed, 0)]
        targets = [3, 5]
        vector = ToyVector(targets)
        runner = SimpleNamespace(envs=vector, episode_length=20,
            policy=SimpleNamespace(ac=SimpleNamespace(max_plane_agents=1)),
            _active_masks_from_info=lambda infos: infos['active_agents'][..., None],
            _authoritative_policy_history=lambda infos, past, n: past)
        iterator = SimpleNamespace(runner=runner, next_rnn=np.zeros((2, 1, 1, 2), np.float32),
            policy=object(), args=SimpleNamespace(stage2_cost_branch_timeout_seconds=600.))

        def forward(policy, graphs, hidden, active, history, kinds):
            self.assertEqual(len(graphs), 2)  # Never fuse candidate GPU batches.
            return np.zeros_like(actions), hidden + 1

        with TemporaryDirectory() as directory, patch(
                'onpolicy.utils.stage2_cost_improvement.actor_forward', side_effect=forward):
            cache = ExactCostCache(directory, 'a' * 64)
            actual, metrics = run_fanout(iterator, queries, width=4, context=context(), cache=cache)
            self.assertEqual(metrics['logical_queries'], 3)
            self.assertEqual(metrics['physical_jobs'], 2)
            self.assertEqual(metrics['coalesced_queries'], 1)
            self.assertLess(metrics['executed_environment_steps'], metrics['requested_environment_steps'])
            for (first, target), outcome in zip(queries, actual):
                env = ToyEnv(targets[target])
                env.step(first[target])
                for _ in range(1, targets[target]):
                    env.step(np.zeros((1, 2), np.int64))
                self.assertEqual(outcome['steps'], targets[target])
                self.assertEqual(outcome['makespan'], env.total_time)
                self.assertTrue(outcome['completed'])
            reused, metrics = run_fanout(iterator, queries, width=4, context=context(), cache=cache)
            self.assertEqual(metrics['cache_hits'], 3)
            self.assertEqual(metrics['physical_jobs'], 0)
            self.assertEqual([semantic_outcome(x) for x in actual], [semantic_outcome(x) for x in reused])

    def test_executor_uses_independent_actor_pool_interface(self):
        vector = ToyVector([2, 3])
        runner = SimpleNamespace(envs=vector, episode_length=20,
            policy=SimpleNamespace(ac=SimpleNamespace(max_plane_agents=1)),
            _active_masks_from_info=lambda infos: infos['active_agents'][..., None],
            _authoritative_policy_history=lambda infos, past, n: past)
        calls = []

        def dispatch(requests):
            calls.append(len(requests))
            return [(key, np.zeros((2, 1, 2), np.int64), hidden + 1, .1)
                    for key, graphs, hidden, active, history, kinds in requests]

        iterator = SimpleNamespace(runner=runner, next_rnn=np.zeros((2, 1, 1, 2), np.float32),
            actor_pool=SimpleNamespace(run=dispatch),
            args=SimpleNamespace(stage2_cost_branch_timeout_seconds=600.))
        first = np.zeros((2, 1, 2), np.int64)
        changed = first.copy()
        changed[0, 0, 0] = 2
        outcomes, stats = run_fanout(iterator, [(first, 1), (changed, 1)], width=4, context=context())
        self.assertTrue(all(row['completed'] for row in outcomes))
        self.assertEqual(calls, [2, 2, 0])
        self.assertAlmostEqual(stats['actor_seconds'], .4)


if __name__ == '__main__':
    unittest.main()
