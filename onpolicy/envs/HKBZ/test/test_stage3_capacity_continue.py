"""CPU-only continuation flow tests; no fake numerical or GPU evidence."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import torch

from onpolicy.scripts.train import continue_stage3_capacity as driver
from onpolicy.scripts.train import stage3_capacity_protocol as protocol
from onpolicy.utils.stage3_research import digest_file, digest_json, read_json


class FakeRunner:
    def __init__(self):
        self.policy_updates = 10
        self.waves, self.updates, self.checkpoints = [], [], []
        self.closed = False

    def resume(self, *args, **kwargs):
        self.resumed = (args, kwargs)

    def rollout(self, cases, seeds, **kwargs):
        self.waves.append((len(cases), self.policy_updates))
        return [dict(case=case, seed=seed, policy_updates=self.policy_updates)
                for case, seed in zip(cases, seeds)]

    def update(self, group, *args, **kwargs):
        if len({row["policy_updates"] for row in group}) != 1:
            raise ValueError("Mixed behavior policy")
        self.updates.append((copy.deepcopy(group), kwargs))
        self.policy_updates += 2
        return {"test": True}

    def save(self, path, **metadata):
        self.checkpoints.append((path, metadata))

    def close(self):
        self.closed = True


class CapacityContinuationTests(unittest.TestCase):
    def run_fake(self, arm, *, continuing=False, guard=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "boundary.pt"
            checkpoint.touch()
            cases = [{"path": str(i), "content_sha256": str(i)} for i in range(120)]
            manifest = {"root": str(root), "manifest_sha256": "new", "code": {"sha256": "code"},
                "source": {"sha256": "source"}, "source_costs": {}, "splits": {"train_pilot120": cases},
                "training": {"seed": 1, "ppo_epochs": 2, "clip_mode": "joint", "post_update_hard_kl": .04},
                "capacity_recipe": {"tbptt_steps": 8},
                "initial_checkpoints": {arm: {"checkpoint": str(checkpoint), "checkpoint_sha256": digest_file(checkpoint)}}}
            payload = {"protocol_sha256": "old", "source_sha256": "source", "arm": arm,
                       "safe_boundary_proof": {"verified": True}, "training_episodes": 832}
            if continuing:
                plan = protocol.phase_schedule(cases, 1, arm[-2:], 832, 960)
                payload.update(protocol_sha256="new", training_episodes=864, capacity_stage_start=832,
                    capacity_stage_until=960, capacity_next_group=1, capacity_schedule_sha256=digest_json(plan))
            runner = FakeRunner()
            mocks = {
                "stage3_capacity_protocol": protocol,
                "onpolicy.scripts.train.run_stage3_sampling_audit": SimpleNamespace(save_group=Mock()),
                "onpolicy.scripts.train.stage3_local_worker": SimpleNamespace(frozen_state=Mock(), assert_frozen=Mock()),
                "onpolicy.scripts.train.stage3_representation_worker": SimpleNamespace(max_role_kl=lambda _: .1 if guard else .001),
            }
            args = SimpleNamespace(arm=arm, until=960, output=root / "train", checkpoint=checkpoint)
            with patch.dict(sys.modules, mocks), patch("torch.load", return_value=payload), \
                    patch("torch.cuda.reset_peak_memory_stats"), patch.object(driver, "engine", return_value=runner), \
                    patch.object(driver, "submit", return_value=f"{arm}_e000960") as submit:
                driver.train(args, manifest, {}, Mock())
            self.assertTrue(runner.closed)
            result = read_json(args.output / "result.json")
            if guard:
                self.assertFalse(result["completed"])
                self.assertEqual(result["training_episodes"], 832)
                self.assertEqual(len(runner.checkpoints), 1)
                self.assertTrue(runner.checkpoints[0][1]["diagnostic_only"])
                submit.assert_not_called()
                return
            expected = [896, 928, 960] if continuing else [864, 896, 928, 960]
            self.assertEqual([row[1]["training_episodes"] for row in runner.checkpoints], expected)
            self.assertEqual([row[1]["capacity_next_group"] for row in runner.checkpoints],
                             [2, 3, 4] if continuing else [1, 2, 3, 4])
            self.assertTrue(all(size == 8 for size, _ in runner.waves))
            self.assertEqual(len(runner.waves), 4 * len(expected))
            for group, kwargs in runner.updates:
                self.assertEqual(len(group), 32)
                self.assertEqual(len({row["case"]["path"] for row in group}), 1 if arm.endswith("T0") else 4)
                self.assertEqual((kwargs["epochs"], kwargs["chunk"]), (2, 8))
            submit.assert_called_once()
            self.assertTrue(result["completed"])
            self.assertEqual(result["training_episodes"], 960)

    def test_complete_groups_preserve_t0_t1_and_submit_after_save(self):
        for arm in ("E0_T0", "E2_T1"):
            with self.subTest(arm=arm):
                self.run_fake(arm)

    def test_mid_stage_resume_keeps_schedule_cursor(self):
        self.run_fake("E2_T1", continuing=True)

    def test_kl_guard_never_admits_partial_update(self):
        self.run_fake("E2_T1", guard=True)


if __name__ == "__main__":
    unittest.main()
