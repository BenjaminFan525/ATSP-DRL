"""Focused Stage3 pure-RL and offline joint-teacher contract tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.scripts.train.prepare_stage3_joint_finetune import (
    MINI_BATCH_SIZE,
    GRAPHS_PER_FORWARD,
    TRAINER_CUDA_MEMORY_FRACTION,
    build_command,
    profile_settings,
)
from onpolicy.scripts.train.run_stage3_joint_manifest import (
    _validate_pure_rl_command,
)
from onpolicy.utils.gpu_phase_lock import exclusive_gpu_phase


ROOT = Path(__file__).resolve().parents[4]
CASE = ROOT / (
    "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/"
    "train/case_0046"
)
TEACHERS = ROOT / (
    "result/hkbz_train_logs/stage3_joint_iga_20260821_r1/"
    "iga1800/teachers"
)
PLANNING = {
    "device_lookahead_dispatch": True,
    "device_lookahead_safety_margin": 60.0,
    "device_deadline_aware_dispatch": True,
    "device_future_intent_horizon": 1,
    "device_future_intent_mode": "bounded_frontier",
    "device_frontier_max_requests": 2,
    "resource_release_aware_eta": True,
    "device_lookahead_reservation_mode": "soft",
    "device_reservation_grace_seconds": 300.0,
    "device_departure_lookahead": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Stage3JointFinetuneTest(unittest.TestCase):
    def test_gpu_phase_lock_supports_disabled_and_absolute_modes(self):
        with exclusive_gpu_phase("", "disabled") as waited:
            self.assertEqual(waited, 0.0)
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu.lock"
            with exclusive_gpu_phase(str(lock_path), "unit") as waited:
                self.assertGreaterEqual(waited, 0.0)
                self.assertTrue(lock_path.is_file())
        with self.assertRaisesRegex(ValueError, "absolute"):
            with exclusive_gpu_phase("relative.lock", "invalid"):
                pass

    def test_canonical_command_is_pure_joint_ppo(self):
        command = build_command(SimpleNamespace(
            profile="canary",
            python=Path("/usr/bin/python3"),
            experiment_name="stage3_pure_rl_contract_test",
            source_checkpoint=Path("/tmp/stage2_source.pt"),
            seed=1,
        ))
        _validate_pure_rl_command(command)
        self.assertNotIn("--joint_iga_teacher_dir", command)
        self.assertNotIn("--joint_iga_teacher_index", command)
        self.assertNotIn("--resource_bc_checkpoint", command)

        for flag in (
            "--plane_bc_pretrain_epochs",
            "--device_bc_pretrain_epochs",
            "--bc_reference_kl_coef",
            "--bc_reference_kl_coef_schedule",
            "--bc_reference_target_kl",
            "--plane_bc_dagger_schedule",
            "--plane_bc_staging_dagger_schedule",
            "--device_bc_dagger_schedule",
        ):
            expected = "0" if "epochs" in flag else "0.0"
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertIn("--joint_team_ppo", command)
        self.assertEqual(
            command[command.index("--joint_team_ppo_scope") + 1], "all"
        )
        self.assertEqual(
            float(command[command.index("--cuda_memory_fraction") + 1]),
            TRAINER_CUDA_MEMORY_FRACTION,
        )
        self.assertEqual(
            int(command[command.index("--max_graphs_per_forward") + 1]),
            GRAPHS_PER_FORWARD,
        )
        self.assertIn("--clear_cuda_cache_after_update", command)
        self.assertEqual(
            command[command.index("--eval_partition_stratify_by") + 1],
            "profile",
        )
        self.assertEqual(
            int(command[command.index("--eval_partition_seed") + 1]),
            20260803,
        )

    def test_latest_stage2_hard_contract_is_inherited_by_pure_rl_n0(self):
        hard_planning = {
            **PLANNING,
            "device_lookahead_reservation_mode": "hard",
        }
        pure_reward = {
            "hindsight_reward_mode": "team_time",
            "resource_critical_lateness_coef": 0.0,
            "resource_earliness_coef": 0.0,
            "resource_wait_constraint_target": 0.0,
            "resource_wait_dual_lr": 0.0,
            "resource_wait_dual_max": 0.05,
        }
        command = build_command(SimpleNamespace(
            profile="credit_happo_wave1",
            python=Path("/usr/bin/python3"),
            experiment_name="stage3_latest_stage2_pure_rl_n0",
            source_checkpoint=Path("/tmp/stage2_source.pt"),
            seed=1,
            method="N0",
            max_graphs_per_forward=1000,
            inherit_source_contracts=True,
            standalone_eval=True,
            source_planning_contract=hard_planning,
            source_reward_contract=pure_reward,
        ))
        _validate_pure_rl_command(command, allow_staged_shared=True)
        self.assertEqual(
            command[command.index("--device_lookahead_reservation_mode") + 1],
            "hard",
        )
        self.assertEqual(
            int(command[command.index("--device_future_intent_horizon") + 1]),
            3,
        )
        self.assertEqual(
            int(command[command.index("--device_frontier_max_requests") + 1]),
            4,
        )
        for flag in (
            "--resource_critical_lateness_coef",
            "--resource_earliness_coef",
            "--resource_wait_constraint_target",
            "--resource_wait_dual_lr",
        ):
            self.assertEqual(float(command[command.index(flag) + 1]), 0.0)
        self.assertEqual(
            float(command[command.index("--resource_wait_dual_max") + 1]),
            0.05,
        )
        for flag in (
            "--canary_eval_interval_shards",
            "--canary_eval_max_per_epoch",
            "--canary_eval_max_cases",
        ):
            self.assertEqual(int(command[command.index(flag) + 1]), 0)
        self.assertNotIn("--canary_stop_on_regression", command)

    def test_four_lane_profiles_preserve_data_and_reduce_cpu_contention(self):
        self.assertEqual(profile_settings("memory_canary"), {
            "episodes": 1,
            "n_rollout_threads": 20,
            "max_train_cases": 20,
            "max_eval_cases": 10,
            "ppo_epoch": 1,
            "train_sampling_size": 20,
        })
        formal = profile_settings("formal")
        self.assertEqual(formal["episodes"], 6)
        self.assertEqual(formal["n_rollout_threads"], 40)
        self.assertEqual(formal["max_train_cases"], 0)
        self.assertEqual(formal["max_eval_cases"], 60)
        self.assertEqual(formal["ppo_epoch"], 2)
        self.assertEqual(formal["train_sampling_size"], 960)

        wave1 = profile_settings("wave1")
        self.assertEqual(wave1["episodes"], 2)
        self.assertEqual(wave1["n_rollout_threads"], 20)
        self.assertEqual(wave1["max_train_cases"], 120)
        self.assertEqual(wave1["max_eval_cases"], 60)
        self.assertEqual(wave1["ppo_epoch"], 1)
        self.assertEqual(wave1["train_sampling_size"], 120)

        encoder_wave1 = profile_settings("encoder_wave1")
        self.assertEqual(encoder_wave1["episodes"], 4)
        self.assertEqual(encoder_wave1["n_rollout_threads"], 20)
        self.assertEqual(encoder_wave1["max_train_cases"], 120)
        self.assertEqual(encoder_wave1["max_eval_cases"], 60)

        for profile in (
            "memory_canary", "canary", "wave1", "encoder_wave1", "formal"
        ):
            settings = profile_settings(profile)
            self.assertGreaterEqual(
                settings["n_rollout_threads"], MINI_BATCH_SIZE
            )
            if settings["max_train_cases"]:
                self.assertGreaterEqual(
                    settings["max_train_cases"],
                    settings["n_rollout_threads"],
                )

        memory_command = build_command(SimpleNamespace(
            profile="memory_canary",
            python=Path("/usr/bin/python3"),
            experiment_name="stage3_memory_canary_contract_test",
            source_checkpoint=Path("/tmp/stage2_source.pt"),
            seed=1,
        ))
        self.assertNotIn("--canary_stop_on_regression", memory_command)
        self.assertEqual(
            int(memory_command[
                memory_command.index("--canary_eval_interval_shards") + 1
            ]),
            0,
        )

    def test_role_clock_phase1_arms_are_strictly_incremental(self):
        expected = {
            "A0": set(),
            "A1": {"--role_atomic_ppo"},
            "A2": {"--role_atomic_ppo", "--role_event_returns"},
            "A3": {
                "--role_atomic_ppo", "--role_event_returns", "--role_valuenorm"
            },
        }
        switches = {
            "--role_atomic_ppo", "--role_event_returns", "--role_valuenorm"
        }
        for method, enabled in expected.items():
            command = build_command(SimpleNamespace(
                profile="canary",
                python=Path("/usr/bin/python3"),
                experiment_name=f"stage3_role_clock_{method}",
                source_checkpoint=Path("/tmp/stage2_source.pt"),
                seed=1,
                method=method,
                max_graphs_per_forward=1000,
            ))
            _validate_pure_rl_command(command, allow_shared_frozen=True)
            self.assertEqual({flag for flag in switches if flag in command}, enabled)
            self.assertEqual(
                int(command[command.index("--max_graphs_per_forward") + 1]),
                1000,
            )
            self.assertEqual(
                int(command[command.index("--mini_batch_size") + 1]), 20
            )

    def test_role_clock_phase2_is_a_clean_two_by_two_screen(self):
        expected = {
            "B0": (False, "fixed", 1.0),
            "B1": (False, "sqrt_event", 1.0),
            "B2": (True, "fixed", 0.95),
            "B3": (True, "sqrt_event", 0.95),
        }
        for method, (event_returns, weighting, gae_lambda) in expected.items():
            command = build_command(SimpleNamespace(
                profile="wave1",
                python=Path("/usr/bin/python3"),
                experiment_name=f"stage3_role_credit_{method}",
                source_checkpoint=Path("/tmp/stage2_source.pt"),
                seed=1,
                method=method,
                max_graphs_per_forward=1000,
            ))
            _validate_pure_rl_command(command, allow_shared_frozen=True)
            self.assertIn("--role_atomic_ppo", command)
            self.assertEqual("--role_event_returns" in command, event_returns)
            self.assertEqual(
                command[command.index("--role_loss_weighting") + 1],
                weighting,
            )
            self.assertAlmostEqual(
                float(command[
                    command.index("--role_event_gae_lambda") + 1
                ]),
                gae_lambda,
            )
            self.assertEqual(
                int(command[command.index("--max_train_cases") + 1]), 120
            )
            self.assertEqual(
                int(command[command.index("--max_eval_cases") + 1]), 60
            )
            self.assertEqual(
                int(command[command.index("--canary_eval_max_cases") + 1]),
                20,
            )

    def test_encoder_wave1_arms_only_change_shared_adaptation(self):
        expected = {
            "E0": (4, "0.0,0.0,0.0,0.0", False),
            "E1": (0, "0.02,0.02,0.02,0.02", False),
            "E2": (1, "0.0,0.01,0.025,0.05", False),
            "E3": (1, "0.0,0.01,0.025,0.05", True),
        }
        for method, (freeze_epochs, schedule, pcgrad) in expected.items():
            command = build_command(SimpleNamespace(
                profile="encoder_wave1",
                python=Path("/usr/bin/python3"),
                experiment_name=f"stage3_encoder_{method}",
                source_checkpoint=Path("/tmp/stage2_source.pt"),
                seed=1,
                method=method,
                max_graphs_per_forward=1000,
            ))
            _validate_pure_rl_command(
                command,
                allow_shared_frozen=method == "E0",
                allow_staged_shared=method != "E0",
            )
            self.assertIn("--role_atomic_ppo", command)
            self.assertNotIn("--role_event_returns", command)
            self.assertNotIn("--role_valuenorm", command)
            self.assertEqual(
                command[command.index("--role_loss_weighting") + 1],
                "sqrt_event",
            )
            self.assertEqual(
                int(command[command.index("--gnn_freeze_epochs") + 1]),
                freeze_epochs,
            )
            self.assertEqual(
                command[
                    command.index("--shared_actor_lr_scale_schedule") + 1
                ],
                schedule,
            )
            self.assertEqual("--shared_encoder_pcgrad" in command, pcgrad)
            self.assertEqual(
                "--shared_encoder_activation_checkpoint" in command,
                method != "E0",
            )
            self.assertEqual(
                int(command[command.index("--max_graphs_per_forward") + 1]),
                1000,
            )

    def test_supervision_cannot_be_reintroduced_via_manifest_command(self):
        command = build_command(SimpleNamespace(
            profile="canary",
            python=Path("/usr/bin/python3"),
            experiment_name="stage3_pure_rl_contract_test",
            source_checkpoint=Path("/tmp/stage2_source.pt"),
            seed=1,
        ))
        command.extend(["--joint_iga_teacher_index", "/tmp/teacher.json"])
        with self.assertRaisesRegex(ValueError, "forbidden flag"):
            _validate_pure_rl_command(command)

    def _config(self, index_path: Path) -> dict:
        config = {
            "jobs_path": str(CASE / "job.json"),
            "fixed_res_path": str(CASE / "fixed_resources.json"),
            "mobile_res_path": str(CASE / "mobile_resources.json"),
            "sites_path": str(CASE / "sites.json"),
            "flights_path": str(CASE / "flights.json"),
            "resource_policy": "drl",
            "n_agents": 24,
            "max_device_num": 80,
            "global_feature_mode": "f1f2",
            "use_domain_rand": False,
            "joint_iga_teacher_dir": str(TEACHERS.resolve()),
            "joint_iga_teacher_index": str(index_path.resolve()),
            **PLANNING,
        }
        return config

    def test_joint_chromosome_decodes_both_roles_against_live_masks(self):
        teacher_path = TEACHERS / "case_0046.json"
        teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
        metadata = json.loads(
            (CASE / "metadata.json").read_text(encoding="utf-8")
        )
        case_sha = (
            metadata.get("case_sha256")
            or metadata["fingerprints"]["case_sha256"]
        )
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "index.json"
            index_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "status": "completed",
                    "teacher_scope": "stage3_full_joint_policy",
                    "teacher_method": "joint_iga_all",
                    "environment_semantics_version": (
                        AircraftScheduleEnv.SEMANTICS_VERSION
                    ),
                    "teacher_dir": str(TEACHERS.resolve()),
                    "planning_contract": PLANNING,
                    "entries": {
                        "case_0046": {
                            "case_sha256": case_sha,
                            "teacher_sha256": _sha256(teacher_path),
                            # Independently produced by the target-contract
                            # replay binder.  It differs from the Wave-3
                            # teacher objective (8851), so this assertion also
                            # catches accidental execution under old masks.
                            "replay_makespan": 8775.000000000015,
                            "replay_verified": True,
                        }
                    },
                }),
                encoding="utf-8",
            )
            env = AircraftScheduleEnv(self._config(index_path))
            try:
                env.reset()
                plane = env.joint_iga_plane_teacher_actions(
                    return_info=True
                )
                resource = env.joint_iga_resource_teacher_actions(
                    return_info=True
                )
                self.assertTrue(plane["info"]["available"])
                self.assertGreater(plane["info"]["active_planes"], 0)
                self.assertTrue(resource["info"]["available"])
                self.assertEqual(
                    plane["info"]["teacher_sha256"],
                    resource["info"]["teacher_sha256"],
                )

                dones = np.zeros(env.n_agents, dtype=bool)
                steps = 0
                while not np.all(dones) and steps < 5000:
                    plane = env.joint_iga_plane_teacher_actions(
                        return_info=True
                    )
                    resource = env.joint_iga_resource_teacher_actions(
                        return_info=True
                    )
                    actions = plane["actions"].copy()
                    actions[env.n_plane_agents :, :2] = resource["actions"][
                        env.n_plane_agents :, :2
                    ]
                    _, _, dones, _ = env.step(actions)
                    steps += 1
                self.assertTrue(np.all(dones))
                self.assertLess(steps, 5000)
                self.assertAlmostEqual(
                    env.total_time, 8775.000000000015, places=8
                )
            finally:
                env.close()

    def test_planning_contract_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "index.json"
            index_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "teacher_scope": "stage3_full_joint_policy",
                    "teacher_method": "joint_iga_all",
                    "environment_semantics_version": (
                        AircraftScheduleEnv.SEMANTICS_VERSION
                    ),
                    "teacher_dir": str(TEACHERS.resolve()),
                    "planning_contract": {
                        **PLANNING,
                        "device_future_intent_mode": "legacy_one",
                    },
                    "entries": {},
                }),
                encoding="utf-8",
            )
            env = AircraftScheduleEnv(self._config(index_path))
            try:
                env.reset()
                with self.assertRaisesRegex(ValueError, "planning-incompatible"):
                    env.joint_iga_plane_teacher_actions(return_info=True)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
