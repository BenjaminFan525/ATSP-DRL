import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from onpolicy.config.config import get_config
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.HKBZ.core import Job, Resource, Site
from onpolicy.envs.HKBZ.experiment.analyze_iga_teacher_trajectories import (
    case_balanced_row_weights,
)
from onpolicy.scripts.train.run_stage1_research_suite import (
    paired_bootstrap_delta,
    latest_epoch_diagnostic,
    tail_recovery_variants,
    training_command,
    variants,
)
from onpolicy.utils.checkpoint_contract import (
    stage1_observation_metadata,
    stage1_reward_contract,
    validate_stage1_checkpoint_contract,
)


def bare_potential_env():
    env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
    env.trajectory_log = []
    env.device_trajectory_log = []
    env.device_decision_log = []
    env.potential_transition_log = []
    env.hindsight_cmax_coef = 1.0
    env.hindsight_shaping_coef = 0.0
    env.hindsight_terminal_cmax_coef = 0.0
    env.hindsight_reward_mode = 'iga_potential'
    env.iga_potential_weights = {
        name: (1.0 if name == 'remaining_work' else 0.0)
        for name in AircraftScheduleEnv.IGA_POTENTIAL_FEATURES
    }
    env.iga_potential_beta = 0.5
    env.iga_potential_gamma = 1.0
    env.cycle_terminated = False
    env.cycle_agent_ids = set()
    env.plane_cycle_penalty = 20000.0
    return env


def feature_state(remaining_work):
    return {
        name: (float(remaining_work) if name == 'remaining_work' else 0.0)
        for name in AircraftScheduleEnv.IGA_POTENTIAL_FEATURES
    }


class IGAPotentialRewardTests(unittest.TestCase):
    def test_shape_compatible_global_modes_have_distinct_schema_ids(self):
        legacy = AircraftScheduleEnv.global_feature_contract('f1f2')
        departure = AircraftScheduleEnv.global_feature_contract(
            'f1f2_departure'
        )
        self.assertEqual(legacy['dimension'], departure['dimension'])
        self.assertEqual(legacy['feature_names'][:18], departure['feature_names'][:18])
        self.assertNotEqual(legacy['feature_names'][18:], departure['feature_names'][18:])
        self.assertNotEqual(legacy['schema_id'], departure['schema_id'])

    def test_checkpoint_contract_rejects_shape_compatible_semantic_swap(self):
        metadata = stage1_observation_metadata('f1f2')
        checkpoint = {
            **metadata,
            'plane_order_mode': 'fixed',
            'plane_pair_decoder': 'joint_pair',
        }
        validate_stage1_checkpoint_contract(
            checkpoint,
            global_feature_mode='f1f2',
            plane_order_mode='fixed',
            plane_pair_decoder='joint_pair',
            strict_metadata=True,
        )
        with self.assertRaisesRegex(ValueError, 'semantic mismatch'):
            validate_stage1_checkpoint_contract(
                checkpoint,
                global_feature_mode='f1f2_departure',
                plane_order_mode='fixed',
                plane_pair_decoder='joint_pair',
                strict_metadata=True,
            )

    def test_strict_reward_contract_prevents_double_terminal_cmax(self):
        action = stage1_reward_contract(
            reward_mode='cmax_delta',
            reward_coef=0.01,
            hindsight_cmax_coef=1.0,
            terminal_cmax_coef=0.0,
            gamma=1.0,
            potential_gamma=1.0,
        )
        self.assertEqual(action['terminal_objective'], 'negative_cmax')
        with self.assertRaisesRegex(ValueError, 'equivalent scaled -Cmax'):
            stage1_reward_contract(
                reward_mode='cmax_delta',
                reward_coef=0.01,
                hindsight_cmax_coef=1.0,
                terminal_cmax_coef=1.0,
                gamma=1.0,
                potential_gamma=1.0,
            )

    def test_iga_potential_is_allocated_once_across_simultaneous_actions(self):
        env = bare_potential_env()
        env.potential_transition_log = [{
            'before': feature_state(100.0),
            'after': feature_state(80.0),
            'actions': [
                {'step_idx': 0, 'agent_id': 0, 'action': [1, 2]},
                {'step_idx': 0, 'agent_id': 1, 'action': [18, 3]},
            ],
        }]

        rewards = env.calculate_hindsight_rewards()

        self.assertAlmostEqual(
            rewards[(0, 0)]['iga_potential_global_shaping'], 10.0
        )
        self.assertAlmostEqual(rewards[(0, 0)]['iga_potential_shaping'], 5.0)
        self.assertAlmostEqual(rewards[(0, 1)]['iga_potential_shaping'], 5.0)
        self.assertAlmostEqual(sum(item['reward'] for item in rewards.values()), 10.0)

    def test_iga_potential_keeps_exact_cmax_base_credit(self):
        env = bare_potential_env()
        env.trajectory_log = [{
            'step_idx': 2,
            'agent_id': 0,
            'action': [4, 1],
            'end_time': 120.0,
        }]

        rewards = env.calculate_hindsight_rewards()

        self.assertAlmostEqual(rewards[(2, 0)]['makespan_contribution'], 120.0)
        self.assertAlmostEqual(rewards[(2, 0)]['reward'], -120.0)

    def test_config_exposes_iga_potential_arguments(self):
        args = get_config().parse_args([
            '--hindsight_reward_mode', 'iga_potential',
            '--iga_potential_weights_path', '/tmp/weights.json',
            '--iga_potential_beta', '0.25',
            '--iga_potential_gamma', '0.99',
        ])
        self.assertEqual(args.hindsight_reward_mode, 'iga_potential')
        self.assertAlmostEqual(args.iga_potential_beta, 0.25)
        self.assertAlmostEqual(args.iga_potential_gamma, 0.99)

    def test_config_exposes_departure_phase_training_arguments(self):
        args = get_config().parse_args([
            '--hindsight_reward_mode', 'team_time_potential',
            '--global_feature_mode', 'f1f2_departure',
            '--plane_bc_phase_aware',
            '--plane_bc_per_agent_dagger',
            '--plane_bc_staging_move_weight', '2.5',
        ])
        self.assertEqual(args.hindsight_reward_mode, 'team_time_potential')
        self.assertEqual(args.global_feature_mode, 'f1f2_departure')
        self.assertTrue(args.plane_bc_phase_aware)
        self.assertTrue(args.plane_bc_per_agent_dagger)
        self.assertAlmostEqual(args.plane_bc_staging_move_weight, 2.5)

    def test_runtime_potential_beta_setter_validates_value(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        self.assertAlmostEqual(env.set_iga_potential_beta(0.075), 0.075)
        with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
            env.set_iga_potential_beta(-0.1)

    def test_tail_features_track_max_wait_and_release_bound(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        job = SimpleNamespace(code='J1', time=100.0)
        plane = SimpleNamespace(
            left_jobs=['J1'],
            current_jobs=[],
            jobs={'J1': job},
            config={'fuel': 30.0},
            is_busy=False,
            is_waiting=True,
            waiting_time=500.0,
            relocations_since_progress=0,
            no_progress_decisions=0,
            is_completed_all_jobs=lambda: False,
        )
        env.planes = {'plane_0_0': plane}
        env.jobs = {'J1': job}
        env.job_code_list = ['J1']
        env.landing_list = [(1000.0, 0, 1, 30.0)]
        env.total_time = 200.0

        features = env.get_iga_potential_features()

        self.assertEqual(features['remaining_work'], 200.0)
        self.assertEqual(features['max_plane_remaining_work'], 100.0)
        self.assertEqual(features['max_waiting_age'], 500.0)
        self.assertEqual(features['future_release_tail'], 900.0)

    def test_legacy_five_feature_weights_default_new_tail_weights_to_zero(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.hindsight_reward_mode = 'iga_potential'
        with tempfile.TemporaryDirectory() as temporary:
            weights_path = Path(temporary) / 'legacy_weights.json'
            weights_path.write_text(
                json.dumps({
                    'weights': {
                        name: float(name == 'remaining_work')
                        for name in AircraftScheduleEnv.IGA_POTENTIAL_BASE_FEATURES
                    }
                }),
                encoding='utf-8',
            )
            env._load_iga_potential_config({
                'iga_potential_beta': 0.1,
                'iga_potential_gamma': 0.99,
                'iga_potential_weights_path': str(weights_path),
            })

        self.assertEqual(
            {
                name: env.iga_potential_weights[name]
                for name in AircraftScheduleEnv.IGA_POTENTIAL_TAIL_FEATURES
            },
            {
                name: 0.0
                for name in AircraftScheduleEnv.IGA_POTENTIAL_TAIL_FEATURES
            },
        )

    def test_team_time_potential_requires_exact_v2_contract(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.hindsight_reward_mode = 'team_time_potential'
        with tempfile.TemporaryDirectory() as temporary:
            weights_path = Path(temporary) / 'v2_weights.json'
            payload = {
                'potential_schema_version': (
                    AircraftScheduleEnv.IGA_POTENTIAL_SCHEMA_VERSION
                ),
                'environment_semantics_version': (
                    AircraftScheduleEnv.SEMANTICS_VERSION
                ),
                'feature_names': list(
                    AircraftScheduleEnv.IGA_POTENTIAL_FEATURES
                ),
                'weights': {
                    name: float(name == 'remaining_work')
                    for name in AircraftScheduleEnv.IGA_POTENTIAL_FEATURES
                },
            }
            weights_path.write_text(json.dumps(payload), encoding='utf-8')
            env._load_iga_potential_config({
                'iga_potential_beta': 0.1,
                'iga_potential_gamma': 1.0,
                'iga_potential_weights_path': str(weights_path),
            })
            self.assertEqual(env.iga_potential_weights['remaining_work'], 1.0)

            payload['environment_semantics_version'] = 'obsolete-semantics'
            weights_path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'semantics mismatch'):
                env._load_iga_potential_config({
                    'iga_potential_beta': 0.1,
                    'iga_potential_gamma': 1.0,
                    'iga_potential_weights_path': str(weights_path),
                })

    def test_potential_calibration_weights_cases_and_distributions(self):
        rows = [
            {'case': 'iid_a', 'distribution': 'iid'},
            {'case': 'iid_a', 'distribution': 'iid'},
            {'case': 'iid_b', 'distribution': 'iid'},
            {'case': 'stress_a', 'distribution': 'ood_stress'},
            {'case': 'stress_a', 'distribution': 'ood_stress'},
            {'case': 'scale_a', 'distribution': 'ood_scale'},
        ]
        weights = case_balanced_row_weights(
            rows,
            {'iid': 0.50, 'ood_stress': 0.45, 'ood_scale': 0.05},
        )
        normalized = weights / weights.sum()

        self.assertAlmostEqual(normalized[:3].sum(), 0.50)
        self.assertAlmostEqual(normalized[3:5].sum(), 0.45)
        self.assertAlmostEqual(normalized[5:].sum(), 0.05)
        self.assertAlmostEqual(normalized[0] + normalized[1], normalized[2])

    def test_alternative_resource_selection_has_stable_sorted_priority(self):
        job = Job(
            code='J1', time=1, group='保障',
            resources=['R002', 'R001'], predecessor=[], exclusive=[],
        )
        job.resources = {'R002', 'R001'}
        first = Resource('fixed_1', 'R001', ['1'])
        second = Resource('fixed_2', 'R002', ['1'])
        site = Site('1', {
            'position': [0, 0],
            'jobs': {'J1': job},
            'fixed_resources': [second, first],
            'mobile_resources': [],
        })

        site.start_jobs([job])

        self.assertEqual(site.onging_jobs['J1'][1], 'fixed_1')

    def test_representation_suite_uses_fixed_joint_actor_and_dagger(self):
        run_variants, aliases = variants(4)
        self.assertEqual(aliases, {})
        self.assertEqual(
            [item['id'] for item in run_variants],
            [
                'G0_local_teacher',
                'G1_local_dagger',
                'G2_global_f1_dagger',
                'G3_global_f1f2_dagger',
            ],
        )
        self.assertEqual({item['order'] for item in run_variants}, {'fixed'})
        self.assertEqual({item['pair'] for item in run_variants}, {'joint_pair'})
        self.assertEqual(
            {item['plane_bc_order_loss_coef'] for item in run_variants}, {0.0}
        )
        self.assertEqual(
            [item['global_features'] for item in run_variants],
            ['none', 'none', 'f1', 'f1f2'],
        )

        args = SimpleNamespace(
            rollout_threads=80,
            eval_threads=60,
            mini_batch_size=7,
            data_chunk_length=50,
            max_graphs_per_forward=350,
            ipc_timeout_seconds=300.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            weights = directory / 'weights.json'
            weights.write_text(json.dumps({'weights': {
                name: 1.0 for name in AircraftScheduleEnv.IGA_POTENTIAL_FEATURES
            }}), encoding='utf-8')
            checkpoint = directory / 'checkpoint_PrePPO.pt'
            command = training_command(
                args,
                variant=run_variants[3],
                experiment_name='unit_test',
                seed=1,
                epochs=3,
                formal=False,
                teacher_dir=directory,
                potential_weights_path=weights,
                resume_checkpoint=checkpoint,
            )
        self.assertEqual(
            command[command.index('--plane_bc_pretrain_epochs') + 1], '0'
        )
        self.assertEqual(
            command[command.index('--checkpoint_dir') + 1], str(checkpoint)
        )
        self.assertIn('--resume_stage1', command)
        self.assertEqual(
            command[command.index('--global_feature_mode') + 1], 'f1f2'
        )
        self.assertEqual(
            command[command.index('--selection_metric') + 1], 'composite'
        )
        self.assertIn('--adaptive_actor_kl', command)
        self.assertNotIn('--bc_reference_hard_gate', command)
        self.assertEqual(
            command[command.index('--bc_reference_target_kl') + 1], '0.0'
        )
        self.assertEqual(
            command[command.index('--adaptive_actor_lr_max_scale') + 1],
            '4.0',
        )
        self.assertEqual(
            command[
                command.index('--adaptive_actor_min_step_completion') + 1
            ],
            '0.9',
        )

    def test_tail_recovery_uses_one_shared_bc_and_factorial_ppo(self):
        run_variants, aliases = tail_recovery_variants(4)

        self.assertFalse(aliases)
        self.assertEqual(
            [item['id'] for item in run_variants],
            [
                'R0_balanced_dagger_global',
                'R1_tail_potential',
                'R2_team_ratio',
                'R3_tail_team_ratio',
                'R4_tail_team_critic',
            ],
        )
        self.assertEqual(
            [item['plane_bc_epochs'] for item in run_variants],
            [4, 0, 0, 0, 0],
        )
        self.assertEqual(
            {item['global_features'] for item in run_variants}, {'f1f2'}
        )
        self.assertEqual(
            {item['train_sampling_mode'] for item in run_variants},
            {'distribution_balanced'},
        )
        self.assertEqual(
            [item['reward'] for item in run_variants],
            [
                'potential_cmax',
                'iga_potential',
                'potential_cmax',
                'iga_potential',
                'iga_potential',
            ],
        )
        self.assertEqual(
            [item['joint_team_ppo'] for item in run_variants],
            [False, False, True, True, True],
        )
        self.assertEqual(
            [item['central_team_critic'] for item in run_variants],
            [False, False, False, False, True],
        )

    def test_paired_bootstrap_uses_case_identity_and_candidate_minus_baseline(self):
        baseline = {
            'cases': [
                {'case_key': 'a', 'makespan': 10.0, 'completed': True},
                {'case_key': 'b', 'makespan': 20.0, 'completed': True},
                {'case_key': 'c', 'makespan': 30.0, 'completed': True},
            ]
        }
        candidate = {
            'cases': [
                {'case_key': 'c', 'makespan': 27.0, 'completed': True},
                {'case_key': 'a', 'makespan': 9.0, 'completed': True},
                {'case_key': 'b', 'makespan': 22.0, 'completed': True},
            ]
        }

        result = paired_bootstrap_delta(
            baseline,
            candidate,
            bootstrap_samples=200,
            seed=7,
        )

        self.assertEqual(result['case_count'], 3)
        self.assertEqual(result['wins'], 2)
        self.assertEqual(result['losses'], 1)
        self.assertAlmostEqual(
            result['mean_delta_candidate_minus_pre_ppo'],
            (-1.0 + 2.0 - 3.0) / 3.0,
        )

    def test_latest_epoch_diagnostic_enforces_health_and_ood_gates(self):
        variant = variants(1)[0][0]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run1'
            models = run_dir / 'models'
            evaluations = run_dir / 'evaluations'
            models.mkdir(parents=True)
            evaluations.mkdir()
            torch.save(
                {
                    'episodes': 2,
                    'actor_update_health': {
                        'step_completion_rate': 1.0,
                        'zero_update_shards': 0,
                        'post_update_old_policy_kl_max': 0.004,
                    },
                },
                models / 'checkpoint_Epoch2.pt',
            )
            cases = [
                {'case_key': 'a', 'makespan': 50.0, 'completed': True},
                {'case_key': 'b', 'makespan': 50.0, 'completed': True},
            ]
            baseline = {
                'evaluation_label': 'pre_ppo',
                'summary': {
                    'eval_makespan': 100.0,
                    'eval_distribution_ood_stress_makespan': 90.0,
                },
                'cases': cases,
            }
            candidate = {
                'evaluation_label': 'epoch_2',
                'summary': {
                    'eval_makespan': 98.5,
                    'eval_distribution_ood_stress_makespan': 89.0,
                    'eval_completion_rate': 1.0,
                    'eval_cycle_count': 0,
                },
                'cases': [
                    {'case_key': 'a', 'makespan': 49.0, 'completed': True},
                    {'case_key': 'b', 'makespan': 49.5, 'completed': True},
                ],
            }
            (evaluations / 'pre_ppo.json').write_text(
                json.dumps(baseline), encoding='utf-8'
            )
            (evaluations / 'epoch_2.json').write_text(
                json.dumps(candidate), encoding='utf-8'
            )

            diagnostic = latest_epoch_diagnostic(
                models / 'checkpoint_Best.pt',
                variant=variant,
                min_step_completion=0.9,
                min_relative_improvement=0.01,
                bootstrap_samples=100,
            )

        self.assertTrue(diagnostic['passed'])
        self.assertTrue(all(diagnostic['checks'].values()))
        self.assertEqual(diagnostic['candidate_cmax'], 98.5)

    def test_formal_training_command_uses_variant_and_enables_canary(self):
        run_variants, _ = variants(4)
        args = SimpleNamespace(
            rollout_threads=80,
            eval_threads=60,
            mini_batch_size=7,
            data_chunk_length=50,
            max_graphs_per_forward=350,
            ipc_timeout_seconds=300.0,
        )
        teacher_dir = Path('/tmp/iga_teacher')
        weights = Path('/tmp/iga_potential_weights.json')

        command = training_command(
            args,
            variant=run_variants[3],
            experiment_name='formal_unit_test',
            seed=3,
            epochs=8,
            formal=True,
            teacher_dir=teacher_dir,
            potential_weights_path=weights,
        )

        self.assertEqual(
            command[command.index('--experiment_name') + 1], 'formal_unit_test'
        )
        self.assertEqual(command[command.index('--seed') + 1], '3')
        self.assertEqual(command[command.index('--num_episodes') + 1], '8')
        self.assertEqual(
            command[command.index('--hindsight_reward_mode') + 1],
            'potential_cmax',
        )
        self.assertEqual(
            command[command.index('--plane_bc_teacher_dir') + 1],
            str(teacher_dir),
        )
        self.assertEqual(
            command[command.index('--plane_bc_dagger_schedule') + 1],
            '1.00,0.70,0.40,0.10',
        )
        self.assertIn('--canary_stop_on_regression', command)
        self.assertEqual(
            command[command.index('--canary_eval_interval_shards') + 1], '3'
        )


if __name__ == '__main__':
    unittest.main()
