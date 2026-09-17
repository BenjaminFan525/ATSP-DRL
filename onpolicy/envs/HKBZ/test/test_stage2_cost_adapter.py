"""Exercise complete evidence persistence/accounting through the opt-in adapter."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

from onpolicy.utils.stage2_cost_accelerated import AcceleratedCostIteration, FanoutAircraftScheduleEnv
from onpolicy.utils.stage2_cost_execution import ExactCostCache
from onpolicy.utils.stage2_cost_improvement import policy_digest


class SmallEnv(FanoutAircraftScheduleEnv):
    def __init__(self):
        self.total_time = 10.
        self.current_case_path = 'train/case_adapter'
        self.position = 0

    def _is_schedule_complete(self):
        return self.position >= 2

    def step(self, action):
        self.position += 1
        self.total_time += 1 + int(action[0, 0])
        return (graph(), np.array([0.]), np.array([self._is_schedule_complete()]),
                {'active_agents': np.array([1.]), 'agent_types': np.array([0])})


def graph():
    return Data(request_mask_matrix=torch.tensor([[True, True]]),
                request_is_lookahead=torch.tensor([True, True]))


class SmallVector:
    def __init__(self):
        self.env = SmallEnv()

    def call(self, name, *args):
        return [getattr(self.env, name)(*args)]

    def call_each(self, name, args):
        return [getattr(self.env, name)(*args[0])]

    @staticmethod
    def stack_infos(rows):
        return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


class CostAdapterTest(unittest.TestCase):
    def test_complete_candidate_evidence_summary_cache_and_live_isolation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ac = SimpleNamespace(max_plane_agents=0, state_dict=lambda: {'weight': torch.tensor([1.])})
            policy = SimpleNamespace(ac=ac)
            vector = SmallVector()
            runner = SimpleNamespace(log_dir=root, policy=policy, envs=vector, episode_length=10,
                _active_masks_from_info=lambda infos: infos['active_agents'][..., None],
                _authoritative_policy_history=lambda infos, past, n: past)
            iterator = AcceleratedCostIteration.__new__(AcceleratedCostIteration)
            hidden = np.zeros((1, 1, 1, 2), np.float32)
            zero = np.zeros((1, 1, 2), np.int64)
            iterator.__dict__.update(runner=runner, policy=policy, enabled=True, selected=[0],
                epoch=1, step=4, rollout=1, case_counts={}, model_sha256=policy_digest(policy),
                target_path=root / 'target.pt', target_file_sha256='f' * 64,
                path=root / 'cost_evidence_epoch1.jsonl', rnn=hidden.copy(),
                next_rnn=hidden.copy(), learner_rnn=hidden.copy(), frozen_actions=zero.copy(), prefix=[],
                execution_config={'fanout_width': 4}, arithmetic={'actor_lanes': 1, 'fanout_width': 4},
                actor_pool=None, cache=ExactCostCache(root / 'cache', 'a' * 64),
                args=SimpleNamespace(stage2_cost_branch_timeout_seconds=600.),
                counts=dict(candidate_queries=0, continuation_vector_steps=0, continuation_wall_seconds=0.,
                            completed=0, incomplete=0, selected_states=0, snapshot_bytes=0),
                execution_counts=dict(batches=0, logical_queries=0, cache_hits=0, physical_jobs=0,
                    coalesced_queries=0, vector_steps=0, requested_environment_steps=0,
                    executed_environment_steps=0, environment_forks=0,
                    environment_rpc_seconds=0., actor_seconds=0., wall_seconds=0.))
            teacher = zero.copy()
            teacher[0, 0, 0] = 1
            infos = {'case_id': np.array(['train/case_adapter']), 'agent_types': np.array([[0]])}
            with patch('onpolicy.utils.stage2_cost_improvement.actor_forward',
                       side_effect=lambda policy, graphs, rnn, active, past, kinds: (zero.copy(), rnn + 1)):
                rows = iterator.evidence([graph()], np.ones((1, 1, 1), np.float32),
                                         -np.ones_like(zero), infos, zero, teacher)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['costs'], [12., 13.])
            self.assertEqual(rows[0]['origins'], ['student', 'derived_teacher'])
            self.assertEqual(iterator.counts['candidate_queries'], 2)
            self.assertEqual(iterator.execution_counts['logical_queries'], 2)
            self.assertEqual(vector.env.position, 0)
            self.assertEqual(vector.env.total_time, 10.)
            self.assertFalse(hasattr(vector.env, '_stage2_cost_fanout'))
            self.assertFalse(hasattr(vector.env, '_stage2_cost_snapshot'))
            self.assertTrue(Path(rows[0]['state_path']).is_file())
            self.assertEqual(len(list(iterator.cache.directory.glob('*.json'))), 2)
            iterator.finish_epoch()
            summary = json.loads((root / 'cost_summary_epoch1.json').read_text())
            execution = json.loads((root / 'cost_execution_epoch1.json').read_text())
            self.assertEqual(summary['counts']['incomplete'], 0)
            self.assertEqual(execution['counts']['batches'], 1)
            self.assertEqual(execution['counts']['physical_jobs'], 2)


if __name__ == '__main__':
    unittest.main()
