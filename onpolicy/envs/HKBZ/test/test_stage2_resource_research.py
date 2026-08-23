"""Focused regressions for live Stage2 IGA BC and method scheduling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from onpolicy.algorithms.gnn_mappo.gnn_mappo import MAPPO_Trainer
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner
from onpolicy.scripts.train.prepare_stage2_resource_research import (
    CANARY_TRAINING_STEPS,
    CRITICAL_ACTOR_ACCUMULATION_TARGET_GRAPHS,
    CRITICAL_CRITIC_ACCUMULATION_TARGET_GRAPHS,
    CRITICAL_GRAPHS_PER_FORWARD,
    CRITICAL_METHODS,
    CRITICAL_MINI_BATCH_SIZE,
    CRITICAL_ROLLOUT_THREADS,
    FORMAL_MINI_BATCH_SIZE,
    FORMAL_ROLLOUT_THREADS,
    FORMAL_STANDARD_GRAPHS,
    FORMAL_ACTOR_ACCUMULATION_TARGET_GRAPHS,
    FORMAL_CRITIC_ACCUMULATION_TARGET_GRAPHS,
    LATENESS_METHODS,
    PLANNING_ACTOR_ACCUMULATION_TARGET_GRAPHS,
    PLANNING_CRITIC_ACCUMULATION_TARGET_GRAPHS,
    PLANNING_GRAPHS_PER_FORWARD,
    PLANNING_METHODS,
    PLANNING_ROLLOUT_THREADS,
    configure_command,
    formal_gpu_calibration,
    natural_evaluator_command,
)
from onpolicy.scripts.train.run_stage2_resource_manifest_trial import (
    apply_runtime_overrides,
)
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.training_stage import (
    protected_parameter_summary,
    validate_stage2_recovery_checkpoint,
)


ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = ROOT / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train/case_0001"
SOURCE_TEACHER = (
    ROOT
    / "result/hkbz_train_logs/"
    "stage2_resource_iga_p5_seed3_case_parallel_20260818_r3/"
    "iga1800/teachers/case_0001.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(**extra):
    result = {
        "jobs_path": str(CASE_DIR / "job.json"),
        "fixed_res_path": str(CASE_DIR / "fixed_resources.json"),
        "mobile_res_path": str(CASE_DIR / "mobile_resources.json"),
        "sites_path": str(CASE_DIR / "sites.json"),
        "flights_path": str(CASE_DIR / "flights.json"),
        "seed": 1,
        "use_domain_rand": False,
        "resource_policy": "drl",
        "n_agents": 24,
        "max_device_num": 80,
        "device_lookahead_dispatch": True,
        "device_lookahead_safety_margin": 60.0,
    }
    result.update(extra)
    return result


class Stage2ResourceResearchTest(unittest.TestCase):
    def test_release_aware_eta_includes_mobile_resource_service(self):
        env = AircraftScheduleEnv(config(resource_release_aware_eta=True))
        try:
            env.reset()
            device = env.device_list[0]
            site = device.site
            device.resource.on_service[:] = [site.code]
            device.resource.available = False
            site.onging_jobs['TEST_RELEASE'] = [123.0, device.resource.code]
            self.assertFalse(device.resource.is_available())
            self.assertAlmostEqual(
                env._device_service_release_seconds(device), 123.0
            )
            self.assertAlmostEqual(
                env._device_eta_seconds(device, site), 123.0
            )
            env.resource_release_aware_eta = False
            self.assertAlmostEqual(env._device_release_seconds(device), 0.0)
        finally:
            env.close()

    def test_soft_lease_is_stealable_but_hard_lease_is_not(self):
        env = AircraftScheduleEnv(config(
            resource_release_aware_eta=True,
            device_future_intent_horizon=1,
            device_lookahead_reservation_mode='soft',
        ))
        try:
            env.reset()
            device = next(
                item for item in env.device_list
                if env._device_is_dispatchable(item)
            )
            request = {
                'plane_id': 'Plane_0_1',
                'job_code': 'ZY02',
                'site_code': device.site.code,
                'needed_res_types': [device.resource.type],
                'is_lookahead': False,
            }
            device.lookahead_reservation = {
                'identity': ['Plane_0_0', 'ZY07', device.site.code],
                'mode': 'soft',
                'expires_at': env.total_time + 300.0,
            }
            self.assertTrue(env._device_is_dispatchable(device))
            self.assertTrue(env._device_can_serve(device, request))
            request['is_lookahead'] = True
            self.assertFalse(env._device_can_serve(device, request))
            device.lookahead_reservation['mode'] = 'hard'
            self.assertFalse(env._device_is_dispatchable(device))
        finally:
            env.close()

    def test_departure_lookahead_lease_is_cancelled_after_stand_change(self):
        env = AircraftScheduleEnv(config(
            resource_release_aware_eta=True,
            device_future_intent_horizon=1,
            device_departure_lookahead=True,
            device_lookahead_reservation_mode='soft',
        ))
        try:
            env.reset()
            plane = env.planes['Plane_0_0']
            device = next(
                item for item in env.device_list
                if item.resource.type == env.TRANSPORTER_RESOURCE_TYPE
            )
            old_site = device.site.code
            plane.site = next(
                site for site in env.sites.values() if site.code != old_site
            )
            plane.departure_staging_decided = True
            device.lookahead_reservation = {
                'identity': [plane.code, env.TRANSFER_JOB_CODE, old_site],
                'plane_id': plane.code,
                'job_code': env.TRANSFER_JOB_CODE,
                'site_code': old_site,
                'mode': 'soft',
                'expires_at': env.total_time + 300.0,
            }
            env._reconcile_lookahead_reservations()
            self.assertIsNone(device.lookahead_reservation)
            self.assertNotIn(plane.code, env.departure_transporter_by_plane)
        finally:
            env.close()

    def test_planning_wave_has_ten_1000_graph_causal_arms(self):
        base = [
            'python', 'train_hkbz.py', '--rollout_until_done',
            '--episode_length', '150', '--rollout_max_steps', '4000',
        ]
        self.assertEqual(len(PLANNING_METHODS), 10)
        self.assertEqual(
            [method['gpu_group'] for method in PLANNING_METHODS.values()].count(0),
            5,
        )
        self.assertEqual(
            [method['gpu_group'] for method in PLANNING_METHODS.values()].count(1),
            5,
        )
        commands = {}
        for method_id, method in PLANNING_METHODS.items():
            command = configure_command(
                base,
                method_id,
                method,
                run_tag='test',
                profile='planning_wave',
                teacher_dir=Path('/tmp/unused-teachers'),
                teacher_index=Path('/tmp/unused-index.json'),
                low_memory=False,
            )
            value = lambda flag: command[command.index(flag) + 1]
            self.assertEqual(
                int(value('--n_rollout_threads')), PLANNING_ROLLOUT_THREADS
            )
            self.assertEqual(
                int(value('--max_graphs_per_forward')),
                PLANNING_GRAPHS_PER_FORWARD,
            )
            self.assertEqual(
                int(value('--grad_accumulation_target_graphs')),
                PLANNING_CRITIC_ACCUMULATION_TARGET_GRAPHS,
            )
            self.assertEqual(
                int(value('--actor_grad_accumulation_target_graphs')),
                PLANNING_ACTOR_ACCUMULATION_TARGET_GRAPHS,
            )
            self.assertEqual(value('--selection_metric'), 'composite_tail')
            self.assertNotIn('--skip_pre_ppo_eval', command)
            commands[method_id] = command

        self.assertNotIn('--resource_release_aware_eta', commands['A0_legacy_c1'])
        self.assertIn('--resource_release_aware_eta', commands['A1_release_eta'])
        self.assertEqual(
            commands['A2_frontier'][
                commands['A2_frontier'].index('--device_future_intent_mode') + 1
            ],
            'bounded_frontier',
        )
        self.assertEqual(
            commands['A3_soft_reservation'][
                commands['A3_soft_reservation'].index(
                    '--device_lookahead_reservation_mode'
                ) + 1
            ],
            'soft',
        )
        self.assertIn('--stage2_allow_shared_unfreeze', commands['B4_gradual_shared'])

    def test_critical_slack_potential_is_nonpositive_and_request_sensitive(self):
        env = AircraftScheduleEnv(config(
            hindsight_reward_mode='team_time_resource_potential',
            iga_potential_beta=0.10,
            iga_potential_gamma=1.0,
            resource_slack_criticality_seconds=1800.0,
            resource_slack_forecast_seconds=0.0,
        ))
        try:
            env.reset()
            plane = next(iter(env.planes.values()))
            device = env.device_list[0]
            env.request_list = [{
                'id': 0,
                'is_noop': True,
            }, {
                'id': 1,
                'job_code': 'ZY02',
                'site_code': plane.site.code,
                'plane_id': plane.code,
                'needed_res_types': [device.resource.type],
                'waiting_time': 120.0,
                'is_noop': False,
                'is_lookahead': False,
            }]
            components = env.get_resource_slack_potential_components()
            self.assertGreaterEqual(components['total_cost'], 120.0)
            self.assertEqual(components['blocking_request_count'], 1)
            self.assertAlmostEqual(
                env._resource_slack_potential_value(),
                -components['total_cost'],
            )
        finally:
            env.close()

    def test_composite_tail_selection_penalizes_worst_cases(self):
        runner = HKBZ_Runner.__new__(HKBZ_Runner)
        runner.selection_metric = 'composite_tail'
        runner.selection_weights = {
            'iid': 0.50,
            'ood_stress': 0.45,
            'ood_scale': 0.05,
        }
        runner.selection_tail_fraction = 0.20
        runner.selection_tail_weight = 0.25
        distributions = (
            ['iid'] * 4 + ['ood_stress'] * 4 + ['ood_scale'] * 2
        )
        runner.last_eval_records = [{
            'case_id': f'case_{index:04d}',
            'case_key': f'case_{index:04d}',
            'makespan': float(index),
            'distribution': distribution,
            'split': 'validation',
        } for index, distribution in enumerate(distributions, start=1)]
        composite = 0.50 * 2.5 + 0.45 * 6.5 + 0.05 * 9.5
        score = runner._selection_metrics_from_records(
            5.5, evaluation_label='epoch_1'
        )
        self.assertAlmostEqual(runner.last_eval_tail_makespan, 9.5)
        self.assertAlmostEqual(score, 0.75 * composite + 0.25 * 9.5)

    def test_wave3_has_five_causal_arms_and_1000_graph_geometry(self):
        base = [
            'python', 'train_hkbz.py', '--rollout_until_done',
            '--episode_length', '150', '--rollout_max_steps', '4000',
        ]
        commands = {}
        for method_id, method in CRITICAL_METHODS.items():
            command = configure_command(
                base,
                method_id,
                method,
                run_tag='test',
                profile='critical_wave',
                teacher_dir=Path('/tmp/unused-teachers'),
                teacher_index=Path('/tmp/unused-index.json'),
                low_memory=False,
            )
            value = lambda flag: command[command.index(flag) + 1]
            self.assertEqual(
                int(value('--n_rollout_threads')), CRITICAL_ROLLOUT_THREADS
            )
            self.assertEqual(
                int(value('--mini_batch_size')), CRITICAL_MINI_BATCH_SIZE
            )
            self.assertEqual(
                int(value('--max_graphs_per_forward')),
                CRITICAL_GRAPHS_PER_FORWARD,
            )
            self.assertEqual(
                int(value('--grad_accumulation_target_graphs')),
                CRITICAL_CRITIC_ACCUMULATION_TARGET_GRAPHS,
            )
            self.assertEqual(
                int(value('--actor_grad_accumulation_target_graphs')),
                CRITICAL_ACTOR_ACCUMULATION_TARGET_GRAPHS,
            )
            self.assertEqual(value('--selection_metric'), 'composite_tail')
            self.assertEqual(value('--device_future_intent_horizon'), '1')
            commands[method_id] = command

        self.assertNotIn('--skip_pre_ppo_eval', commands['C0_cmax_control'])
        for method_id in tuple(CRITICAL_METHODS)[1:]:
            self.assertIn('--skip_pre_ppo_eval', commands[method_id])
        self.assertEqual(
            commands['C0_cmax_control'][
                commands['C0_cmax_control'].index('--hindsight_reward_mode') + 1
            ],
            'team_cmax',
        )
        self.assertEqual(
            commands['C1_team_time'][
                commands['C1_team_time'].index('--hindsight_reward_mode') + 1
            ],
            'team_time',
        )
        self.assertEqual(
            commands['C2_critical_slack'][
                commands['C2_critical_slack'].index('--hindsight_reward_mode') + 1
            ],
            'team_time_resource_potential',
        )

    def test_deadline_window_wakes_before_need_time(self):
        env = AircraftScheduleEnv(config(
            device_deadline_aware_dispatch=True,
            device_future_intent_horizon=1,
            device_departure_lookahead=True,
        ))
        try:
            env.reset()
            device = next(
                item for item in env.device_list
                if env._device_is_dispatchable(item)
            )
            target = next(
                site for site in env.sites.values()
                if site != device.site
            )
            travel = env._device_travel_seconds(device, target)
            request = {
                'id': 1,
                'job_code': 'ZY02',
                'site_code': target.code,
                'plane_id': 'Plane_0_0',
                'needed_res_types': [device.resource.type],
                'is_lookahead': True,
                'lead_time': travel + env.device_lookahead_safety_margin + 17.0,
            }
            env.request_list = [
                {'id': 0, 'is_noop': True}, request
            ]
            self.assertFalse(env._device_can_dispatch(device, request))
            self.assertAlmostEqual(
                env._next_lookahead_dispatch_dt(), 17.0, places=6
            )
            request['lead_time'] -= 17.0
            self.assertTrue(env._device_can_dispatch(device, request))
        finally:
            env.close()

    def test_one_edge_future_intent_exposes_mobile_successor(self):
        env = AircraftScheduleEnv(config(
            device_deadline_aware_dispatch=True,
            device_future_intent_horizon=1,
            device_departure_lookahead=True,
        ))
        try:
            env.reset()
            plane = env.planes['Plane_0_0']
            target_site = next(
                site for site in env.sites.values()
                if (
                    site.update_resources() is None
                    and 'R011' not in site.res_avail
                )
            )
            plane.site = target_site
            plane.is_busy = True
            plane.current_jobs = ['ZY02']
            plane.left_jobs = ['ZY07', *env.departure_job_code_list]
            plane.finished_jobs = []
            plane.site.left_job_time = 90.0
            request = env._next_mobile_job_request(plane)
            self.assertIsNotNone(request)
            self.assertEqual(request['job_code'], 'ZY07')
            self.assertEqual(request['request_kind'], 'next_mobile_job')
            self.assertEqual(request['lead_time'], 90.0)
            self.assertTrue(request['is_lookahead'])
        finally:
            env.close()

    def test_lateness_round_has_true_dagger_and_three_credit_arms(self):
        base = [
            'python', 'train_hkbz.py', '--rollout_until_done',
            '--episode_length', '150', '--rollout_max_steps', '4000',
        ]
        observed = {}
        for method_id, method in LATENESS_METHODS.items():
            command = configure_command(
                base,
                method_id,
                method,
                run_tag='test',
                profile='lateness_wave',
                teacher_dir=Path('/tmp/unused-teachers'),
                teacher_index=Path('/tmp/unused-index.json'),
                low_memory=False,
            )
            value = lambda flag: command[command.index(flag) + 1]
            self.assertEqual(value('--device_bc_dagger_schedule'), '1.0,0.7,0.4,0.1')
            self.assertEqual(value('--device_bc_pretrain_epochs'), '4')
            self.assertEqual(value('--device_future_intent_horizon'), '1')
            self.assertIn('--device_deadline_aware_dispatch', command)
            self.assertIn('--device_departure_lookahead', command)
            observed[method_id] = (
                float(value('--resource_lateness_coef')),
                float(value('--resource_critical_lateness_coef')),
                float(value('--resource_earliness_coef')),
            )
        self.assertEqual(observed['P0_cmax'], (0.0, 0.0, 0.0))
        self.assertGreater(observed['P1_total_lateness'][0], 0.0)
        self.assertGreater(observed['P2_critical_jit'][1], 0.0)
        self.assertGreater(observed['P2_critical_jit'][2], 0.0)

    def test_one_lane_runtime_override_records_safe_graph_geometry(self):
        command = [
            "python",
            "train_hkbz.py",
            "--experiment_name",
            "original",
            "--mini_batch_size",
            "24",
            "--data_chunk_length",
            "50",
            "--max_graphs_per_forward",
            "1200",
            "--grad_accumulation_steps",
            "4",
            "--actor_grad_accumulation_steps",
            "2",
        ]
        args = SimpleNamespace(
            experiment_name="m0_lowmem600",
            mini_batch_size=12,
            data_chunk_length=50,
            max_graphs_per_forward=600,
            grad_accumulation_steps=7,
            actor_grad_accumulation_steps=3,
            resume_checkpoint=None,
        )

        overrides = apply_runtime_overrides(command, args)

        self.assertEqual(overrides["max_graphs_per_forward"], 600)
        self.assertEqual(
            command[command.index("--experiment_name") + 1],
            "m0_lowmem600",
        )
        self.assertEqual(
            int(command[command.index("--mini_batch_size") + 1]), 12
        )
        self.assertEqual(
            int(command[command.index("--max_graphs_per_forward") + 1]),
            600,
        )

        args.max_graphs_per_forward = 599
        with self.assertRaisesRegex(ValueError, "exceeds"):
            apply_runtime_overrides(command, args)

    def test_runtime_override_enables_explicit_stage2_recovery(self):
        command = [
            "python",
            "train_hkbz.py",
            "--checkpoint_dir",
            "/stage1.pt",
            "--reset_optimizers_on_resume",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            recovery = Path(temporary) / "checkpoint_Recovery.pt"
            recovery.touch()
            args = SimpleNamespace(
                experiment_name=None,
                mini_batch_size=None,
                data_chunk_length=None,
                max_graphs_per_forward=None,
                grad_accumulation_steps=None,
                actor_grad_accumulation_steps=None,
                resume_checkpoint=recovery,
            )
            overrides = apply_runtime_overrides(command, args)

        self.assertEqual(
            command[command.index("--checkpoint_dir") + 1],
            str(recovery.resolve()),
        )
        self.assertIn("--resume_stage2", command)
        self.assertNotIn("--reset_optimizers_on_resume", command)
        self.assertEqual(
            overrides["resume_checkpoint"], str(recovery.resolve())
        )

    def test_stage2_recovery_requires_complete_post_shard_state(self):
        target = {
            "encoder.weight": torch.ones(2, 2),
            "device_actor.weight": torch.arange(4.0).reshape(2, 2),
        }
        checkpoint = {
            "training_stage": "resource_joint",
            "stage": "post_shard_recovery",
            "phase": "resource_joint_ppo",
            "episodes": 1,
            "completed_shard": 2,
            "total_shards": 3,
            "total_num_steps": 123,
            "plane_order_mode": "fixed",
            "plane_pair_decoder": "joint_pair",
            "global_feature_mode": "f1f2",
            "source_m2_path": "/source.pt",
            "source_m2_sha256": "a" * 64,
            "model": {key: value.clone() for key, value in target.items()},
        }
        checkpoint["protected_parameter_summary"] = (
            protected_parameter_summary(checkpoint["model"])
        )

        result = validate_stage2_recovery_checkpoint(
            checkpoint,
            target,
            plane_order_mode="fixed",
            plane_pair_decoder="joint_pair",
            global_feature_mode="f1f2",
        )
        self.assertEqual(result["completed_shards"], 2)
        self.assertEqual(result["total_shards"], 3)

        checkpoint["stage"] = "emergency"
        with self.assertRaisesRegex(ValueError, "post_shard_recovery"):
            validate_stage2_recovery_checkpoint(
                checkpoint,
                target,
                plane_order_mode="fixed",
                plane_pair_decoder="joint_pair",
                global_feature_mode="f1f2",
            )

    def test_stage2_recovery_can_protect_plane_subset_for_shared_unfreeze(self):
        target = {
            "encoder.weight": torch.zeros(2, 2),
            "plane_sel_enc.weight": torch.ones(2, 2),
            "device_actor.weight": torch.arange(4.0).reshape(2, 2),
        }
        checkpoint = {
            "training_stage": "resource_joint",
            "stage": "post_shard_recovery",
            "phase": "resource_joint_ppo",
            "episodes": 3,
            "completed_shard": 1,
            "total_shards": 3,
            "total_num_steps": 456,
            "plane_order_mode": "fixed",
            "plane_pair_decoder": "joint_pair",
            "global_feature_mode": "f1f2",
            "source_m2_path": "/source.pt",
            "source_m2_sha256": "b" * 64,
            "model": {key: value.clone() for key, value in target.items()},
        }
        checkpoint["model"]["encoder.weight"].fill_(2.0)
        plane_prefixes = (
            "plane_sel_enc.", "actor.", "plane_order_actor.",
        )
        checkpoint["protected_parameter_summary"] = (
            protected_parameter_summary(
                checkpoint["model"], prefixes=plane_prefixes
            )
        )

        result = validate_stage2_recovery_checkpoint(
            checkpoint,
            target,
            plane_order_mode="fixed",
            plane_pair_decoder="joint_pair",
            global_feature_mode="f1f2",
            protected_prefixes=plane_prefixes,
        )
        self.assertEqual(
            result["protected_summary"],
            checkpoint["protected_parameter_summary"],
        )

    def test_graph_batches_are_time_major_for_exact_accumulation(self):
        buffer = SharedReplayBuffer.__new__(SharedReplayBuffer)
        steps, environments, agents = 4, 3, 1
        buffer.episode_length = steps
        buffer.filled_steps = steps
        buffer.data_chunk_length = 2
        buffer.rewards = np.zeros(
            (steps, environments, agents, 1), dtype=np.float32
        )
        buffer.graph_obs = [
            [(step, env) for env in range(environments)]
            for step in range(steps + 1)
        ]
        buffer.rnn_states = np.zeros(
            (steps + 1, environments, agents, 1, 1), dtype=np.float32
        )
        buffer.actions = np.zeros(
            (steps, environments, agents, 3), dtype=np.float32
        )
        buffer.value_preds = np.zeros(
            (steps + 1, environments, agents, 1), dtype=np.float32
        )
        buffer.returns = np.zeros_like(buffer.value_preds)
        buffer.active_masks = np.ones_like(buffer.value_preds)
        buffer.policy_masks = np.ones_like(buffer.value_preds)
        buffer.agent_types = np.zeros(
            (steps + 1, environments, agents), dtype=np.int64
        )
        buffer.policy_sample_weights = np.ones_like(buffer.rewards)
        buffer.value_sample_weights = np.ones_like(buffer.rewards)
        buffer.action_log_probs = np.zeros_like(buffer.rewards)
        advantages = np.zeros_like(buffer.rewards)

        samples = list(buffer.graph_recurrent_generator(advantages, 2))
        self.assertEqual([len(sample[0]) for sample in samples], [4, 2, 4, 2])
        self.assertEqual(
            [[graph[0] for graph in sample[0]] for sample in samples],
            [[0, 0, 1, 1], [0, 1], [2, 2, 3, 3], [2, 3]],
        )

    def test_graph_target_accumulation_preserves_optimizer_mass(self):
        samples = [([None] * 1500, index) for index in range(4)]
        actor_groups = MAPPO_Trainer._accumulation_groups(
            samples, fixed_steps=2, target_graphs=2000
        )
        actor_graphs = [
            sum(len(sample[0]) for sample in group)
            for group in actor_groups
        ]
        self.assertEqual(len(actor_groups), 3)
        self.assertEqual(sum(actor_graphs), 6000)
        self.assertEqual(sum(actor_graphs) / len(actor_graphs), 2000)
        self.assertEqual(
            [sample[1] for group in actor_groups for sample in group],
            list(range(4)),
        )

        critic_samples = [([None] * 1500, index) for index in range(6)]
        critic_groups = MAPPO_Trainer._accumulation_groups(
            critic_samples, fixed_steps=3, target_graphs=5000
        )
        self.assertEqual(
            [sum(len(sample[0]) for sample in group) for group in critic_groups],
            [4500, 4500],
        )

        fixed_groups = MAPPO_Trainer._accumulation_groups(
            samples, fixed_steps=2, target_graphs=0
        )
        self.assertEqual([len(group) for group in fixed_groups], [2, 2])

    def test_canary_truncates_training_but_not_validation(self):
        base = [
            "python",
            "train_hkbz.py",
            "--rollout_until_done",
            "--episode_length",
            "150",
            "--rollout_max_steps",
            "4000",
        ]
        method = {"teacher": "heuristic", "role_balanced": False}
        canary = configure_command(
            base,
            "M0_heuristic_uniform",
            method,
            run_tag="test",
            profile="canary",
            teacher_dir=Path("/tmp/unused-teachers"),
            teacher_index=Path("/tmp/unused-index.json"),
            low_memory=False,
        )
        self.assertIn("--no_rollout_until_done", canary)
        self.assertNotIn("--rollout_until_done", canary)
        self.assertIn("--skip_pre_ppo_eval", canary)
        self.assertIn("--skip_epoch_eval", canary)
        self.assertEqual(
            int(canary[canary.index("--actor_warmup_shards") + 1]),
            0,
        )
        self.assertEqual(
            int(canary[canary.index("--episode_length") + 1]),
            CANARY_TRAINING_STEPS,
        )

        evaluator = natural_evaluator_command(canary)
        self.assertIn("--rollout_until_done", evaluator)
        self.assertNotIn("--no_rollout_until_done", evaluator)
        self.assertEqual(
            int(evaluator[evaluator.index("--rollout_max_steps") + 1]),
            4000,
        )

        formal = configure_command(
            base,
            "M0_heuristic_uniform",
            method,
            run_tag="test",
            profile="wave1",
            teacher_dir=Path("/tmp/unused-teachers"),
            teacher_index=Path("/tmp/unused-index.json"),
            low_memory=False,
        )
        self.assertIn("--rollout_until_done", formal)
        self.assertNotIn("--no_rollout_until_done", formal)
        self.assertNotIn("--skip_pre_ppo_eval", formal)
        self.assertNotIn("--skip_epoch_eval", formal)
        self.assertEqual(
            int(formal[formal.index("--actor_warmup_shards") + 1]),
            1,
        )
        self.assertEqual(
            int(formal[formal.index("--n_rollout_threads") + 1]),
            FORMAL_ROLLOUT_THREADS,
        )
        self.assertEqual(
            int(formal[formal.index("--mini_batch_size") + 1]),
            FORMAL_MINI_BATCH_SIZE,
        )
        self.assertEqual(
            int(formal[formal.index("--max_graphs_per_forward") + 1]),
            FORMAL_STANDARD_GRAPHS,
        )
        self.assertEqual(
            int(formal[formal.index("--grad_accumulation_steps") + 1]),
            3,
        )
        self.assertEqual(
            int(formal[formal.index("--actor_grad_accumulation_steps") + 1]),
            2,
        )
        self.assertEqual(
            int(formal[
                formal.index("--grad_accumulation_target_graphs") + 1
            ]),
            FORMAL_CRITIC_ACCUMULATION_TARGET_GRAPHS,
        )
        self.assertEqual(
            int(formal[
                formal.index(
                    "--actor_grad_accumulation_target_graphs"
                ) + 1
            ]),
            FORMAL_ACTOR_ACCUMULATION_TARGET_GRAPHS,
        )
        self.assertEqual(
            int(formal[formal.index("--device_bc_min_rollouts_per_epoch") + 1]),
            3,
        )

    def test_planning_canary_runs_one_natural_lifecycle_per_worker(self):
        base = [
            "python", "train_hkbz.py", "--rollout_until_done",
            "--episode_length", "150", "--rollout_max_steps", "4000",
        ]
        method_id = "A3_soft_reservation"
        command = configure_command(
            base,
            method_id,
            PLANNING_METHODS[method_id],
            run_tag="test",
            profile="planning_canary",
            teacher_dir=Path("/tmp/unused-teachers"),
            teacher_index=Path("/tmp/unused-index.json"),
            low_memory=False,
        )

        self.assertIn("--rollout_until_done", command)
        self.assertNotIn("--no_rollout_until_done", command)
        self.assertEqual(
            int(command[command.index("--max_train_cases") + 1]),
            PLANNING_ROLLOUT_THREADS,
        )
        self.assertEqual(
            int(command[
                command.index("--device_bc_min_rollouts_per_epoch") + 1
            ]),
            1,
        )
        self.assertEqual(
            int(command[
                command.index("--device_bc_max_rollouts_per_epoch") + 1
            ]),
            1,
        )

    def test_formal_gpu_calibration_projects_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            hardware = suite / "hardware"
            hardware.mkdir()
            memory_log = hardware / "canary_standard_gpu_memory.csv"
            memory_log.write_text(
                "unix_time,memory_used_mib\n1,1000\n2,15245\n",
                encoding="utf-8",
            )
            calibration = formal_gpu_calibration(
                suite, profile="wave1", low_memory=False
            )
            self.assertEqual(calibration["observed_peak_mib"], 15245)
            self.assertEqual(
                calibration["conservative_linear_projection_mib"], 15245
            )
            self.assertGreater(calibration["projected_headroom_mib"], 0)

            memory_log.write_text(
                "unix_time,memory_used_mib\n1,75000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "not supported"):
                formal_gpu_calibration(
                    suite, profile="wave1", low_memory=False
                )

    def test_dagger_executes_whole_teacher_joint_action_per_environment(self):
        runner = HKBZ_Runner.__new__(HKBZ_Runner)
        runner.policy = SimpleNamespace(
            ac=SimpleNamespace(max_plane_agents=2)
        )
        runner.num_agents = 4
        policy = np.arange(16, dtype=np.int64).reshape(2, 4, 2)
        teacher0 = np.zeros((4, 2), dtype=np.int64)
        teacher1 = np.zeros((4, 2), dtype=np.int64)
        teacher0[2:] = [[101, 0], [102, 0]]
        teacher1[2:] = [[201, 0], [202, 0]]
        results = [
            {"actions": teacher0, "info": {"real_dispatches": 2}},
            {"actions": teacher1, "info": {"real_dispatches": 2}},
        ]
        actions, labels, stats = runner._merge_device_bc_actions(
            policy, results, np.asarray([True, False])
        )
        np.testing.assert_array_equal(actions[0, :2], policy[0, :2])
        np.testing.assert_array_equal(actions[0, 2:], teacher0[2:])
        np.testing.assert_array_equal(actions[1], policy[1])
        np.testing.assert_array_equal(labels[0, 2:], teacher0[2:])
        np.testing.assert_array_equal(labels[1, 2:], teacher1[2:])
        self.assertEqual(stats["teacher_executed_envs"], 1)
        self.assertEqual(stats["student_executed_envs"], 1)

    def test_resource_teacher_requires_and_uses_strict_sidecar_binding(self):
        source = json.loads(SOURCE_TEACHER.read_text(encoding="utf-8"))
        metadata = json.loads(
            (CASE_DIR / "metadata.json").read_text(encoding="utf-8")
        )
        case_sha = metadata["fingerprints"]["case_sha256"]
        chromosome = source["search"]["chromosome"]
        compact = {
            "status": "completed",
            "schema_version": 1,
            "teacher_scope": "stage2_resource_policy",
            "teacher_method": "resource_iga_all",
            "resource_policy": "drl",
            "arm": "iga_all",
            "backends": {"ordinary": "iga", "transporter": "iga"},
            "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
            "case": "case_0001",
            "case_sha256": case_sha,
            "frozen_plane_checkpoint_sha256": source[
                "frozen_plane_checkpoint_sha256"
            ],
            "completion_verified": True,
            "completed": True,
            "makespan": float(source["makespan"]),
            "search": {
                "n_var": int(source["search"]["n_var"]),
                "chromosome": chromosome,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            teacher_dir = Path(temporary) / "teachers"
            teacher_dir.mkdir()
            teacher_path = teacher_dir / "case_0001.json"
            teacher_path.write_text(json.dumps(compact), encoding="utf-8")
            index_path = Path(temporary) / "index.json"
            index_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "teacher_scope": "stage2_resource_policy",
                    "teacher_method": "resource_iga_all",
                    "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
                    "teacher_dir": str(teacher_dir.resolve()),
                    "frozen_plane_checkpoint_sha256": source[
                        "frozen_plane_checkpoint_sha256"
                    ],
                    "entries": {
                        "case_0001": {
                            "case_sha256": case_sha,
                            "teacher_sha256": sha256(teacher_path),
                        }
                    },
                }),
                encoding="utf-8",
            )
            env = AircraftScheduleEnv(config(
                resource_iga_teacher_dir=str(teacher_dir),
                resource_iga_teacher_index=str(index_path),
            ))
            try:
                env.reset()
                result = env.resource_iga_teacher_actions(return_info=True)
                self.assertTrue(result["info"]["available"])
                self.assertEqual(result["actions"].shape, (env.n_agents, 2))
                self.assertEqual(
                    result["info"]["teacher_sha256"], sha256(teacher_path)
                )
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
