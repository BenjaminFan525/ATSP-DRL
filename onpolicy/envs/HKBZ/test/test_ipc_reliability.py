import gc
import os
import time
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from onpolicy.config.config import get_config
from onpolicy.envs.env_wrappers import GraphSubprocVecEnv
from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner


class _IPCStressEnv:
    """Small worker env whose observation has production-like tensor count."""

    def __init__(self):
        self.step_index = 0

    def _observation(self):
        graph = HeteroData()
        for index in range(32):
            graph["node"][f"feature_{index}"] = torch.full(
                (2, 2), float(self.step_index + index), dtype=torch.float32
            )
        return graph

    @staticmethod
    def _info():
        return {"active_agents": np.array([True], dtype=bool)}

    def reset(self):
        self.step_index = 0
        return self._observation(), np.array([False]), self._info()

    def step(self, _action):
        self.step_index += 1
        return (
            self._observation(),
            np.zeros(1, dtype=np.float32),
            np.array([False]),
            self._info(),
        )

    def close(self):
        return None




class _HungIPCStressEnv(_IPCStressEnv):
    def step(self, _action):
        time.sleep(30.0)
        return super().step(_action)


class _CrashIPCStressEnv(_IPCStressEnv):
    def step(self, _action):
        os._exit(7)


class _FakeRemote:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.recv_calls = 0
        self.sent = []
        self.closed = False

    def recv(self):
        self.recv_calls += 1
        if self.error is not None:
            raise self.error
        return self.result

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self):
        self.join_calls = 0
        self.terminated = False

    def join(self, timeout=None):
        self.join_calls += 1

    def is_alive(self):
        return False

    def terminate(self):
        self.terminated = True


class _ScalarOnlyWriter:
    def __init__(self):
        self.calls = []

    def add_scalar(self, *args):
        self.calls.append(args)

    def add_scalars(self, *args):
        raise AssertionError("add_scalars creates one writer/file per metric")


def _shared_tensor_fd_count():
    count = 0
    fd_dir = "/proc/self/fd"
    for name in os.listdir(fd_dir):
        try:
            target = os.readlink(os.path.join(fd_dir, name))
        except FileNotFoundError:
            continue
        if target.startswith("/dev/shm/torch_"):
            count += 1
    return count


class IPCReliabilityTest(unittest.TestCase):
    def test_default_sharing_strategy_avoids_manager_per_worker(self):
        args = get_config().parse_args([])
        self.assertEqual(args.torch_mp_sharing_strategy, "file_descriptor")
        self.assertEqual(args.ipc_timeout_seconds, 300.0)

    def test_received_graphs_are_private_and_fd_count_is_stable(self):
        previous_strategy = torch.multiprocessing.get_sharing_strategy()
        torch.multiprocessing.set_sharing_strategy("file_descriptor")
        env = GraphSubprocVecEnv([lambda: _IPCStressEnv() for _ in range(20)])
        baseline_fds = _shared_tensor_fd_count()
        try:
            observations, _, _ = env.reset()
            actions = np.zeros((20, 1, 2), dtype=np.int64)
            for _ in range(30):
                observations, _, _, _ = env.step(actions)

            for store in observations[0].stores:
                for value in store.values():
                    if torch.is_tensor(value):
                        self.assertFalse(value.is_shared())
            gc.collect()
            self.assertLessEqual(
                _shared_tensor_fd_count() - baseline_fds,
                32,
            )
        finally:
            started = time.monotonic()
            env.close()
            self.assertLess(time.monotonic() - started, 10.0)
            torch.multiprocessing.set_sharing_strategy(previous_strategy)

    def test_ipc_timeout_fails_fast_and_close_kills_hung_worker(self):
        env = GraphSubprocVecEnv(
            [lambda: _HungIPCStressEnv()], ipc_timeout_seconds=0.2
        )
        try:
            env.reset()
            actions = np.zeros((1, 1, 2), dtype=np.int64)
            with self.assertRaisesRegex(TimeoutError, "Timed out waiting"):
                env.step(actions)
        finally:
            started = time.monotonic()
            env.close()
            self.assertLess(time.monotonic() - started, 18.0)
            self.assertFalse(env.ps[0].is_alive())

    def test_dead_worker_is_reported_instead_of_blocking_parent(self):
        env = GraphSubprocVecEnv(
            [lambda: _CrashIPCStressEnv()], ipc_timeout_seconds=2.0
        )
        try:
            env.reset()
            actions = np.zeros((1, 1, 2), dtype=np.int64)
            with self.assertRaisesRegex(RuntimeError, "worker"):
                env.step(actions)
        finally:
            env.close()
            self.assertFalse(env.ps[0].is_alive())

    def test_partial_receive_failure_does_not_reread_during_close(self):
        graph = HeteroData()
        graph["node"].x = torch.ones(1, 1)
        remotes = (
            _FakeRemote((graph, np.zeros(1), np.array([False]), {"x": 1})),
            _FakeRemote(error=RuntimeError("simulated IPC failure")),
            _FakeRemote((graph, np.zeros(1), np.array([False]), {"x": 1})),
        )
        env = GraphSubprocVecEnv.__new__(GraphSubprocVecEnv)
        env.remotes = remotes
        env.ps = tuple(_FakeProcess() for _ in remotes)
        env.closed = False
        env.waiting = True
        env._pending_remote_indices = {0, 1, 2}

        with self.assertRaisesRegex(RuntimeError, "simulated IPC failure"):
            env._recv_all(clone_graph_payload=True)
        self.assertEqual(env._pending_remote_indices, {2})
        self.assertEqual([remote.recv_calls for remote in remotes], [1, 1, 0])

        env.close()
        self.assertEqual([remote.recv_calls for remote in remotes], [1, 1, 0])
        self.assertTrue(env.closed)
        self.assertFalse(env.waiting)

    def test_resume_preserves_a_stronger_cross_run_selection_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_checkpoint = root / "prior_best.pt"
            output_dir = root / "models"
            output_dir.mkdir()
            torch.save(
                {
                    "eval_makespan": 88.0,
                    "stage": "pre_ppo_baseline",
                    "training_stage": "plane_pretrain",
                    "model": {},
                },
                source_checkpoint,
            )

            events = []
            runner = HKBZ_Runner.__new__(HKBZ_Runner)
            runner.selection_checkpoint_dir = str(source_checkpoint)
            runner.checkpoint_dir = "resume_last.pt"
            runner.best_eval_makespan = 125.0
            runner.save_dir = str(output_dir)
            runner.total_num_steps = 0
            runner.current_epoch = -1
            runner.current_shard = -1
            runner.progress_callback = lambda **event: events.append(event)
            runner.use_wandb = False
            runner.writter = _ScalarOnlyWriter()

            self.assertTrue(runner._seed_best_from_selection_checkpoint())
            self.assertEqual(runner.best_eval_makespan, 88.0)
            copied = torch.load(output_dir / "checkpoint_Best.pt", map_location="cpu")
            self.assertEqual(copied["eval_makespan"], 88.0)
            self.assertTrue(copied["selection_seeded_across_resume"])
            self.assertEqual(events[-1]["event"], "selection_baseline_seeded")

    def test_shard_canary_stops_after_three_percent_regression(self):
        runner = HKBZ_Runner.__new__(HKBZ_Runner)
        runner.best_eval_makespan = 100.0
        runner.canary_max_regression = 0.03
        runner.canary_stop_on_regression = True
        runner.canary_rejected = False
        runner.canary_rejection_info = {}
        runner.total_num_steps = 42
        saved = []
        events = []
        logged = []
        runner.eval = lambda **_kwargs: 104.0
        runner._evaluation_log_info = lambda makespan: {
            "eval_makespan": makespan
        }
        runner.log_train = lambda info, steps: logged.append((info, steps))
        runner._report_progress = lambda event, **payload: events.append({
            "event": event, **payload
        })
        runner.save = lambda episode, filename=None, extra=None: saved.append(
            (episode, filename, extra)
        )

        self.assertTrue(runner._run_shard_canary(0, 6))
        self.assertTrue(runner.canary_rejected)
        self.assertEqual(saved[0][1], "checkpoint_CanaryRejected.pt")
        self.assertEqual(saved[0][2]["stage"], "canary_rejected")
        self.assertAlmostEqual(
            runner.canary_rejection_info["canary_relative_regression"], 0.04
        )
        self.assertEqual(events[-1]["event"], "canary_regression_stop_requested")
        self.assertEqual(logged[0][1], 42)

    def test_training_metrics_use_one_tensorboard_writer(self):
        runner = HKBZ_Runner.__new__(HKBZ_Runner)
        runner.use_wandb = False
        runner.writter = _ScalarOnlyWriter()
        runner.log_train({"policy_loss": 1.25, "approx_kl": 0.01}, 42)
        self.assertEqual(
            runner.writter.calls,
            [("policy_loss", 1.25, 42), ("approx_kl", 0.01, 42)],
        )


if __name__ == "__main__":
    unittest.main()
