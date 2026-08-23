"""Synthetic, no-training tests for the Stage-2 controller/service contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest import mock

import yaml

from onpolicy.scripts.train import run_hkbz_two_stage_pipeline as pipeline


PYTHON = "/mnt/eb20f54b-f016-4501-a5b2-5dd6afff9d90/fanyixuan_files/conda/envs/maia/bin/python3.11"
ROOT = Path(__file__).resolve().parents[4]


def option_map(command):
    """Parse the value-taking subset of a generated command for assertions."""

    values = {}
    switches = set()
    tokens = list(command)
    while tokens and not tokens[0].startswith("--"):
        tokens.pop(0)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise AssertionError(f"Unexpected command token: {token!r}")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            if token in values:
                raise AssertionError(f"Duplicate option: {token}")
            values[token] = tokens[index + 1]
            index += 2
        else:
            if token in switches:
                raise AssertionError(f"Duplicate switch: {token}")
            switches.add(token)
            index += 1
    return values, switches


class TwoStageOrchestrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "stage1_M2.pt"
        self.source.write_bytes(b"synthetic-stage1-m2")
        self.manifest = self.root / "two_stage_manifest.json"
        self.models = self.root / "models"
        self.models.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_env_inherits_fjsp_v3_and_enables_resource_joint_semantics(self):
        env_plane = yaml.safe_load(
            (ROOT / "onpolicy/config/env_plane_pretrain.yaml").read_text(
                encoding="utf-8"
            )
        )
        env_stage2 = yaml.safe_load(
            (ROOT / "onpolicy/config/env_resource_joint.yaml").read_text(
                encoding="utf-8"
            )
        )
        for key in (
            "batch_num",
            "plane_num_per_batch",
            "n_agents",
            "max_device_num",
            "plane_cycle_repeat_limit",
            "plane_no_progress_limit",
            "plane_relocation_limit",
            "plane_first_completion_bonus",
            "plane_repeat_relocation_penalty",
            "plane_reset_job_penalty",
            "plane_no_progress_penalty",
            "plane_cycle_penalty",
        ):
            self.assertEqual(env_stage2[key], env_plane[key], key)
        self.assertEqual(env_stage2["resource_policy"], "drl")
        self.assertTrue(env_stage2["device_lookahead_dispatch"])
        self.assertIn("fjsp_v3", env_stage2["dataset_dir"])
        self.assertNotIn("fjsp_v2", json.dumps(env_stage2))
        for value in env_stage2.values():
            if isinstance(value, str):
                self.assertFalse(value.startswith("/"), value)

    def test_stage1_source_command_is_sanitized_and_stage2_is_explicit(self):
        old = [
            "python",
            "train_hkbz.py",
            "--seed",
            "23",
            "--resource_policy",
            "heuristic",
            "--plane_bc_pretrain_epochs",
            "8",
            "--resume_stage1",
            "--bc_reference_kl_coef",
            "0.4",
            "--bc_reference_hard_gate",
        ]
        command = pipeline.build_stage2_command(
            self.source,
            run_tag="migration_test",
            source_command=old,
        )
        values, switches = option_map(command)
        self.assertEqual(values["--training_stage"], "resource_joint")
        self.assertEqual(values["--resource_policy"], "drl")
        self.assertEqual(values["--checkpoint_dir"], str(self.source.resolve()))
        self.assertEqual(values["--env_config"], str(pipeline.ENV_CONFIG))
        self.assertEqual(values["--seed"], "1")
        self.assertGreater(int(values["--device_bc_pretrain_epochs"]), 0)
        self.assertGreater(int(values["--num_episodes"]), 0)
        self.assertEqual(values["--gnn_freeze_epochs"], values["--num_episodes"])
        self.assertEqual(values["--plane_freeze_epochs"], values["--num_episodes"])
        self.assertNotIn("--device_bc_train_gnn", switches)
        self.assertNotIn("--no_device_bc_reset_optim", switches)
        self.assertNotIn("--resume_stage1", switches)
        self.assertNotIn("--joint_team_ppo", switches)
        self.assertNotIn("--central_team_critic", switches)
        self.assertIn("--device_lookahead_dispatch", switches)
        self.assertIn("--strict_checkpoint_contract", switches)
        self.assertEqual(
            values["--device_lookahead_safety_margin"], "60.0"
        )
        self.assertEqual(values["--hindsight_reward_mode"], "team_cmax")
        self.assertEqual(values["--hindsight_terminal_cmax_coef"], "1.0")
        self.assertNotIn("--bc_reference_kl_coef", values)
        self.assertNotIn("--plane_bc_pretrain_epochs", values)

    def test_dry_run_writes_atomic_manifest_without_subprocess(self):
        with mock.patch.object(pipeline.subprocess, "run") as run:
            manifest = pipeline.run_stage2(
                self.source,
                run_tag="dry_run_test",
                manifest_path=self.manifest,
                artifact_dir=self.models,
                dry_run=True,
                seed=7,
            )
        run.assert_not_called()
        self.assertEqual(manifest["status"], "dry_run")
        self.assertEqual(manifest["training_stage"], "resource_joint")
        self.assertEqual(
            manifest["source_m2_sha256"],
            hashlib.sha256(self.source.read_bytes()).hexdigest(),
        )
        self.assertTrue(self.manifest.is_file())
        self.assertFalse(list(self.manifest.parent.glob(".two_stage_manifest.json.tmp.*")))
        observed = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(observed["migration_contract"]["to"], "resource_joint")
        self.assertEqual(observed["migration_contract"]["device_bc_train_gnn"], False)
        self.assertEqual(observed["migration_contract"]["device_bc_reset_optim"], True)
        self.assertEqual(
            observed["migration_contract"]["device_lookahead_dispatch"], True
        )
        self.assertEqual(
            observed["migration_contract"]["device_lookahead_safety_margin"],
            60.0,
        )
        self.assertEqual(
            observed["migration_contract"]["environment_semantics_version"],
            "progressive-departure-r014-pipeline-v2",
        )
        self.assertEqual(
            observed["migration_contract"]["observation_schema_id"],
            "hkbz-global-ad3c0982aef57ac78d1d",
        )

    def test_keyed_suite_manifest_resolves_exact_source_command(self):
        command_path = self.root / "suite.json"
        command_path.write_text(
            json.dumps(
                {
                    "commands": {
                        "seed1": {"argv": ["python", "train.py", "--seed", "1"]},
                        "seed3": {"argv": ["python", "train.py", "--seed", "3"]},
                    }
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "requires source_command_key"):
            pipeline._load_source_command(command_path)
        self.assertEqual(
            pipeline._load_source_command(command_path, "seed3"),
            ["python", "train.py", "--seed", "3"],
        )

    def test_verified_handoff_resolves_seed_and_semantics(self):
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        command_path = self.root / "source_command.json"
        command_path.write_text(
            json.dumps(
                {
                    "command": [
                        "python",
                        "train_hkbz.py",
                        "--seed",
                        "7",
                        "--plane_order_mode",
                        "fixed",
                        "--plane_pair_decoder",
                        "joint_pair",
                        "--global_feature_mode",
                        "f1f2",
                    ]
                }
            ),
            encoding="utf-8",
        )
        handoff = self.root / "stage1_handoff.json"
        handoff.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage1_status": "closed",
                    "training_stage": "plane_pretrain",
                    "semantic_contract": {
                        "plane_order_mode": "fixed",
                        "plane_pair_decoder": "joint_pair",
                        "global_feature_mode": "f1f2",
                    },
                    "checkpoints": {
                        "7": {
                            "path": str(self.source),
                            "sha256": digest,
                            "size_bytes": self.source.stat().st_size,
                            "source_command_path": str(command_path),
                            "source_command_sha256": hashlib.sha256(
                                command_path.read_bytes()
                            ).hexdigest(),
                            "source_command_size_bytes": command_path.stat().st_size,
                        }
                    },
                    "decision": {
                        "resource_joint_transition_authorized": True
                    },
                }
            ),
            encoding="utf-8",
        )
        path, semantics = pipeline.resolve_stage1_handoff_checkpoint(7, handoff)
        self.assertEqual(path, str(self.source.resolve()))
        self.assertEqual(semantics["global_feature_mode"], "f1f2")
        entry, _ = pipeline.resolve_stage1_handoff_entry(7, handoff)
        self.assertEqual(entry["source_command_path"], str(command_path.resolve()))
        self.source.write_bytes(b"replaced-source")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            pipeline.resolve_stage1_handoff_checkpoint(7, handoff)

    @staticmethod
    def _summary(digest):
        return {"sha256": digest, "count": 1, "numel": 1, "scope": ["synthetic"]}

    def _source_identity(self):
        return {
            "path": str(self.source.resolve()),
            "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
        }

    def _write_lineage_checkpoint(self, name, phase, *, role):
        protected = self._summary("protected-unchanged")
        path = self.models / name
        checkpoint = {
            "training_stage": "resource_joint",
            "phase": phase,
            "source_m2_checkpoint": self._source_identity(),
            "resource_bc_optimizer_reset": True,
            "resource_bc_total_labels": 12,
            "protected_parameter_summary": protected,
            "protected_parameter_summary_before_bc": protected,
            "protected_parameter_summary_after_bc": protected,
            "resource_actor_summary_before_bc": self._summary("actor-bc-before"),
            "resource_actor_summary_after_bc": self._summary("actor-bc-after"),
        }
        if role == "last":
            checkpoint.update(
                {
                    "protected_parameter_summary_after_ppo": protected,
                    "resource_actor_summary_before_ppo": self._summary("actor-ppo-before"),
                    "resource_actor_summary_after_ppo": self._summary("actor-ppo-after"),
                }
            )
        elif role == "best":
            # A Best checkpoint is allowed to be the post-BC baseline.  It is
            # still written after the canonical resource_joint_ppo phase is
            # entered, but may not yet contain post-PPO actor evidence.
            checkpoint["resource_actor_summary_before_ppo"] = self._summary("actor-ppo-before")
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
        return path

    def test_mocked_execution_requires_and_records_warmup_best_last_lineage(self):
        self._write_lineage_checkpoint(
            "checkpoint_DeviceBC.pt", "resource_bc_warmup_completed", role="warmup"
        )
        self._write_lineage_checkpoint(
            "checkpoint_Best.pt", "resource_joint_ppo", role="best"
        )
        self._write_lineage_checkpoint(
            "checkpoint_Last.pt", "resource_joint_ppo", role="last"
        )
        self._write_run_status()
        calls = []

        def fake_run(command, **kwargs):
            calls.append((list(command), kwargs))
            return mock.Mock(returncode=0)

        manifest = pipeline.run_stage2(
            self.source,
            run_tag="mocked_execution",
            manifest_path=self.manifest,
            artifact_dir=self.models,
            runner=fake_run,
            seed=3,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["check"], True)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["lineage_status"], "validated")
        self.assertEqual(manifest["lineage"]["run_status"]["status"], "validated")
        self.assertEqual(
            manifest["run_status"]["path"],
            str((self.models.parent / "run_status.json").resolve()),
        )
        self.assertEqual(
            manifest["run_status"]["sha256"],
            hashlib.sha256((self.models.parent / "run_status.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(manifest["run_status_path"], manifest["run_status"]["path"])
        self.assertEqual(manifest["run_status_sha256"], manifest["run_status"]["sha256"])
        self.assertEqual(
            {entry["status"] for entry in manifest["lineage"].values()},
            {"validated"},
        )
        self.assertTrue(
            any(
                entry["operation"] == "audit_stage2_lineage"
                for entry in manifest["operation_log"]
            )
        )

    def _write_run_status(self):
        protected = self._summary("protected-unchanged")
        status = {
            "status": "completed",
            "training_stage": "resource_joint",
            "phase": "resource_joint_completed",
            "source_m2_checkpoint": self._source_identity(),
            "protected_parameter_summary": protected,
            "resource_actor_summary_before_bc": self._summary("actor-bc-before"),
            "resource_actor_summary_after_bc": self._summary("actor-bc-after"),
            "resource_actor_summary_before_ppo": self._summary("actor-ppo-before"),
            "resource_actor_summary_after_ppo": self._summary("actor-ppo-after"),
            "resource_bc_total_labels": 12,
            "resource_bc_optimizer_reset": True,
        }
        (self.models.parent / "run_status.json").write_text(
            json.dumps(status), encoding="utf-8"
        )

    def test_three_warmup_checkpoints_without_run_status_fail_closed(self):
        # Distinct files alone are not final-run evidence.  In particular,
        # three warm-up-shaped artifacts must not be promoted to completed.
        self._write_lineage_checkpoint(
            "checkpoint_DeviceBC.pt", "resource_bc_warmup_completed", role="warmup"
        )
        self._write_lineage_checkpoint(
            "checkpoint_Best.pt", "resource_bc_warmup_completed", role="warmup"
        )
        self._write_lineage_checkpoint(
            "checkpoint_Last.pt", "resource_bc_warmup_completed", role="warmup"
        )
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return mock.Mock(returncode=0)

        with self.assertRaisesRegex(FileNotFoundError, "run_status"):
            pipeline.run_stage2(
                self.source,
                run_tag="missing_run_status",
                manifest_path=self.manifest,
                artifact_dir=self.models,
                runner=fake_run,
            )
        self.assertEqual(len(calls), 1)
        observed = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(observed["status"], "failed")
        self.assertNotEqual(observed.get("lineage_status"), "validated")
        self.assertFalse("run_status" in observed and observed["run_status"].get("status") == "completed")

    def test_service_dry_run_does_not_call_systemd_or_training(self):
        launcher = ROOT / "onpolicy/scripts/train/launch_hkbz_two_stage_service.sh"
        env = dict(os.environ)
        env.update(
            {
                "PYTHON": PYTHON,
                "RUN_TAG": "launcher_dry_run",
                "STOP_AFTER_STAGE": "2",
                "DRY_RUN": "1",
                "STAGE1_M2_CHECKPOINT": str(self.source),
                "MANIFEST_PATH": str(self.manifest),
                "ARTIFACT_DIR": str(self.models),
            }
        )
        completed = subprocess.run(
            ["/bin/bash", str(launcher)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertNotIn("systemd-run", completed.stdout)
        self.assertNotIn("[Stage2Command]", completed.stdout)
        self.assertEqual(
            json.loads(self.manifest.read_text(encoding="utf-8"))["status"],
            "dry_run",
        )

    def test_four_stage_shim_rejects_retired_stage(self):
        shim = ROOT / "onpolicy/scripts/train/launch_hkbz_four_stage_service.sh"
        env = dict(os.environ)
        env.update({"STOP_AFTER_STAGE": "3", "DRY_RUN": "1"})
        completed = subprocess.run(
            ["/bin/bash", str(shim)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("retired", completed.stdout)


if __name__ == "__main__":
    unittest.main()
