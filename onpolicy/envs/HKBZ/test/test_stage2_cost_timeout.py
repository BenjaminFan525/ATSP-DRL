"""Reproduce overlapping-latency timeouts without waiting or allocating CUDA."""
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from onpolicy.envs.HKBZ.test.test_stage2_cost_execution import ToyVector, context
from onpolicy.utils.stage2_cost_execution import run_fanout
from onpolicy.utils.stage2_cost_timeout import (
    HardProgressDeadline, LEGACY_TIMEOUT_CONTRACT, TIMEOUT_CONTRACT, shared_wall_charges,
)


class CostTimeoutTest(unittest.TestCase):
    def test_shared_wall_is_never_multiplied_by_concurrency(self):
        for weights in ([4.] * 16, [0.] * 16, [1., 7.], [3.]):
            charges = shared_wall_charges(4., weights)
            self.assertAlmostEqual(sum(charges), 4.)
        self.assertEqual(shared_wall_charges(4., [4.] * 16), [.25] * 16)
        self.assertEqual(shared_wall_charges(4., []), [])
        for wall, weights in ((float('nan'), [1]), (-1., [1]), (1., [float('inf')]), (1., [-1])):
            with self.assertRaises(ValueError):
                shared_wall_charges(wall, weights)

    def simulate(self, contract, jobs=16, terminal=4, budget=8.):
        clock = [0.]
        vector = ToyVector([terminal] * 2)
        original_rpc = vector.call_each

        def rpc(*args):
            clock[0] += 1.
            return original_rpc(*args)

        def forward(requests):
            if requests:
                clock[0] += 4.
            return [(key, np.zeros((2, 1, 2), np.int64), hidden + 1, 4.)
                    for key, graphs, hidden, active, history, kinds in requests]

        vector.call_each = rpc
        runner = SimpleNamespace(envs=vector, episode_length=100,
            policy=SimpleNamespace(ac=SimpleNamespace(max_plane_agents=1)),
            _active_masks_from_info=lambda infos: infos['active_agents'][..., None],
            _authoritative_policy_history=lambda infos, past, n: past)
        iterator = SimpleNamespace(runner=runner, next_rnn=np.zeros((2, 1, 1, 2), np.float32),
            actor_pool=SimpleNamespace(run=forward),
            args=SimpleNamespace(stage2_cost_branch_timeout_seconds=budget))
        queries = []
        for number in range(jobs):
            first = np.zeros((2, 1, 2), np.int64)
            first[0, 0, 0] = number
            queries.append((first, 1))
        with patch('onpolicy.utils.stage2_cost_execution.time.monotonic', side_effect=lambda: clock[0]):
            return run_fanout(iterator, queries, width=16,
                              context=dict(context(), arithmetic={'timeout_contract': contract}))

    def test_sixteen_live_branches_no_longer_false_timeout(self):
        old, _ = self.simulate(LEGACY_TIMEOUT_CONTRACT)
        fixed, metrics = self.simulate(TIMEOUT_CONTRACT)
        self.assertTrue(all(row['reason'] == 'wall_timeout' for row in old))
        self.assertTrue(all(row['completed'] and row['steps'] == 4 for row in fixed))
        self.assertEqual(metrics['peak_live_jobs'], 16)
        self.assertEqual(metrics['legacy_timeout_avoided_queries'], 16)
        self.assertEqual(metrics['charged_work_seconds'], 16.)
        self.assertEqual(sum(row['charged_work_seconds'] for row in fixed), 16.)
        self.assertTrue(all(row['legacy_charged_seconds'] > 8. for row in fixed))

    def test_genuine_single_job_budget_exhaustion_still_abstains(self):
        fixed, _ = self.simulate(TIMEOUT_CONTRACT, jobs=1, terminal=20)
        self.assertFalse(fixed[0]['completed'])
        self.assertIsNone(fixed[0]['makespan'])
        self.assertEqual(fixed[0]['reason'], 'wall_timeout')

    def test_watchdog_restores_handler_and_does_not_override_existing_alarm(self):
        original = signal.getsignal(signal.SIGALRM)
        with HardProgressDeadline(30.):
            with self.assertRaises(ValueError):
                with HardProgressDeadline(30.):
                    pass
        self.assertEqual(signal.getsignal(signal.SIGALRM), original)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))

    def test_hung_operation_exits_124_without_hanging_cleanup(self):
        result = subprocess.run([sys.executable, '-c',
            'import time; from onpolicy.utils.stage2_cost_timeout import HardProgressDeadline; '
            '\nwith HardProgressDeadline(.05): time.sleep(10)'],
            cwd=Path(__file__).resolve().parents[4], capture_output=True, text=True, timeout=3.)
        self.assertEqual(result.returncode, 124)
        self.assertIn('CostProgressDeadline', result.stderr)


if __name__ == '__main__':
    unittest.main()
