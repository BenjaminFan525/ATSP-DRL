import json
import tempfile
import unittest
from pathlib import Path

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.scripts.train.prepare_stage1_departure_research import (
    PPO_VARIANTS,
    PPO_WAVES,
    bc_command,
    ppo_command,
    validate_potential_contract,
    validate_teacher_contract,
)


def option(command, flag):
    return command[command.index(flag) + 1]


class Stage1DepartureResearchTest(unittest.TestCase):
    def test_phase_bc_and_team_time_commands_are_causally_separated(self):
        source = [
            'python',
            'train_hkbz.py',
            '--experiment_name',
            'old',
            '--no_eval',
            '--canary_eval_interval_shards',
            '2',
            '--canary_stop_on_regression',
            '--selection_checkpoint_dir',
            '/tmp/old-best.pt',
        ]
        root = Path('/tmp/research-contract')
        common = {
            'run_tag': 'unit',
            'seed': 1,
            'epochs': 2,
            'teacher_dir': root / 'teachers',
            'potential_path': root / 'potential.json',
        }
        b1 = bc_command(
            source,
            'B1_phase_dagger_warm',
            warm_start_checkpoint=root / 'm2.pt',
            **common,
        )
        b2 = bc_command(
            source,
            'B2_phase_dagger_cold',
            warm_start_checkpoint=root / 'm2.pt',
            **common,
        )
        self.assertIn('--plane_bc_phase_aware', b1)
        self.assertIn('--plane_bc_per_agent_dagger', b1)
        self.assertIn('--plane_bc_only', b1)
        self.assertEqual(option(b1, '--global_feature_mode'), 'f1f2_departure')
        self.assertIn('--checkpoint_dir', b1)
        self.assertNotIn('--checkpoint_dir', b2)
        self.assertNotIn('--resume_stage1', b2)

        p0 = ppo_command(
            source,
            'P0_action_cmax',
            phase='ppo_screen',
            bc_checkpoint=root / 'plane_bc.pt',
            **common,
        )
        p4 = ppo_command(
            source,
            'P4_team_time_potential_ramp',
            phase='ppo_screen',
            bc_checkpoint=root / 'plane_bc.pt',
            **common,
        )
        p5 = ppo_command(
            source,
            'P5_team_time_potential_fixed',
            phase='ppo_screen',
            bc_checkpoint=root / 'plane_bc.pt',
            **common,
        )
        self.assertEqual(
            option(p0, '--hindsight_reward_mode'), 'cmax_delta'
        )
        self.assertEqual(option(p0, '--hindsight_terminal_cmax_coef'), '0.0')
        self.assertEqual(
            option(p4, '--iga_potential_beta_schedule'),
            '0.10,0.10,0.05,0.00',
        )
        self.assertEqual(
            option(p5, '--iga_potential_beta_schedule'), '0.10'
        )
        self.assertEqual({
            variant['global_feature_mode'] for variant in PPO_VARIANTS.values()
        }, {'f1f2'})
        self.assertEqual(option(p4, '--global_feature_mode'), 'f1f2')
        self.assertEqual(option(p4, '--reward_coef'), '0.01')
        self.assertEqual(option(p4, '--hindsight_terminal_cmax_coef'), '1.0')
        self.assertIn('--reset_value_normalizer_on_resume', p4)
        self.assertIn('--strict_checkpoint_contract', p4)
        self.assertIn('--strict_stage1_reward_contract', p4)
        self.assertIn('--use_eval', p4)
        self.assertNotIn('--no_eval', p4)
        self.assertEqual(option(p4, '--canary_eval_interval_shards'), '0')
        self.assertNotIn('--canary_stop_on_regression', p4)
        self.assertNotIn('--selection_checkpoint_dir', p4)
        self.assertNotIn('--shared_eval_socket', p4)
        self.assertNotIn('--shared_eval_cpu_set', p4)
        self.assertEqual(
            PPO_WAVES,
            {
                1: (
                    'P0_action_cmax',
                    'P1_action_potential',
                    'P2_team_cmax',
                ),
                2: (
                    'P3_team_time',
                    'P4_team_time_potential_ramp',
                    'P5_team_time_potential_fixed',
                ),
            },
        )

    def test_preflight_requires_new_semantics_and_zero_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset'
            teacher_dir = root / 'teachers'
            case = dataset / 'case_0001'
            case.mkdir(parents=True)
            teacher_dir.mkdir()
            (teacher_dir / 'case_0001.json').write_text(json.dumps({
                'schema_version': 2,
                'teacher_scope': 'stage1_plane_policy',
                'resource_policy': 'heuristic',
                'completion_verified': True,
                'environment_semantics_version': (
                    AircraftScheduleEnv.SEMANTICS_VERSION
                ),
                'case': 'case_0001',
                'case_sha256': 'a' * 64,
                'makespan': 100.0,
            }), encoding='utf-8')
            potential = root / 'potential.json'
            potential.write_text(json.dumps({
                'status': 'completed',
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
                'replay': {
                    'expected_cases': 1,
                    'reverified_cases': 1,
                    'stored_cmax_match_cases': 1,
                    'max_abs_cmax_drift': 0.0,
                    'transition_count': 3,
                },
                'calibration': {
                    'case_level_cross_validation': {
                        'case_disjoint': True,
                        'metrics': {'rmse': 1.0},
                    },
                },
            }), encoding='utf-8')

            self.assertEqual(
                validate_teacher_contract(dataset, teacher_dir, 1)[
                    'teacher_count'
                ],
                1,
            )
            self.assertEqual(
                validate_potential_contract(potential, 1)[
                    'transition_count'
                ],
                3,
            )

            payload = json.loads(potential.read_text(encoding='utf-8'))
            payload['replay']['max_abs_cmax_drift'] = 1.0
            potential.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'zero drift'):
                validate_potential_contract(potential, 1)


if __name__ == '__main__':
    unittest.main()
