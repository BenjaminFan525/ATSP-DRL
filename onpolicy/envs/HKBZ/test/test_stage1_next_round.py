import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from onpolicy.scripts.train import run_stage1_next_round_controller as controller
from onpolicy.scripts.train import run_stage1_next_round_trial as trial


class Stage1NextRoundTests(unittest.TestCase):
    @staticmethod
    def parse_command(command):
        options = {}
        switches = set()
        index = 2
        while index < len(command):
            flag = command[index]
            if not flag.startswith('--'):
                raise AssertionError(f'Unexpected token {flag!r}.')
            if index + 1 < len(command) and not command[index + 1].startswith('--'):
                if flag in options:
                    raise AssertionError(f'Duplicate option {flag}.')
                options[flag] = command[index + 1]
                index += 2
            else:
                if flag in switches:
                    raise AssertionError(f'Duplicate switch {flag}.')
                switches.add(flag)
                index += 1
        return options, switches

    @staticmethod
    def args(source, variant):
        return Namespace(
            phase='screen',
            variant=variant,
            seed=1,
            run_tag='next_round_test',
            source_command_json=source,
            epochs=4,
            train_sampling_size=480,
            validation_dir=Path('/validation'),
            partition_seed=20260803,
            potential_path=Path('/potential.json'),
            initial_checkpoint=Path('/checkpoint.pt'),
            shared_eval_socket='/tmp/next-round-test.sock',
            cpu_set='0-11,72-83',
        )

    def test_registered_methods_only_change_preregistered_fields(self):
        source_payload = {
            'command': [
                'python', 'train.py', '--env_name', 'HKBZ',
                '--experiment_name', 'old', '--iga_potential_beta', '0.1',
                '--bc_reference_kl_coef', '0.2', '--use_eval',
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source.json'
            source.write_text(json.dumps(source_payload), encoding='utf-8')
            commands = {
                name: self.parse_command(trial.training_command(
                    self.args(source, name)
                ))
                for name in trial.VARIANTS
            }

        control_options, control_switches = commands['M0_n2_control']
        allowed = {
            '--experiment_name', '--iga_potential_beta_schedule',
            '--bc_reference_kl_coef_schedule', '--train_sampling_mode',
            '--train_sampling_weights', '--tail_policy_start_fraction',
            '--tail_policy_weight',
        }
        for name, (options, switches) in commands.items():
            self.assertEqual(switches, control_switches, name)
            differences = {
                flag for flag in options.keys() | control_options.keys()
                if options.get(flag) != control_options.get(flag)
            }
            self.assertLessEqual(differences, allowed, name)
            self.assertEqual(options['--anneal_original'], '0.3')
            self.assertEqual(options['--anneal_final'], '0.3')
            self.assertEqual(options['--tau_anneal_epochs'], '0')
            self.assertEqual(options['--evaluation_tau'], '0.3')
            self.assertEqual(options['--checkpoint_dir'], '/checkpoint.pt')

    def test_method_schedules_are_explicit_and_hold_final_value(self):
        ramp = trial.method_manifest('M1_potential_ramp', 8)
        self.assertEqual(
            ramp['potential_beta_by_epoch'],
            [0.0, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        )
        anneal = trial.method_manifest('M2_bc_kl_anneal', 8)
        self.assertEqual(
            anneal['bc_reference_kl_by_epoch'],
            [0.4, 0.25, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05],
        )

    def test_pre_ppo_best_checkpoint_is_a_valid_no_improvement_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            evaluations = run_dir / 'evaluations'
            evaluations.mkdir()
            evaluation_path = evaluations / 'pre_ppo.json'
            evaluation_path.write_text(json.dumps({
                'summary': {'eval_selection_score': 8500.0},
            }), encoding='utf-8')
            checkpoint = run_dir / 'checkpoint_Best.pt'
            torch.save({
                'episodes': 0,
                'stage': 'pre_ppo_baseline',
                'selection_score': 8500.0,
                'eval_raw_makespan': 8500.0,
                'eval_iid_makespan': 8500.0,
                'eval_composite_makespan': 8500.0,
                'actor_update_health': {
                    'step_completion_rate': 1.0,
                    'zero_update_shards': 0,
                },
            }, checkpoint)

            result, observed_path = trial.checkpoint_evaluation(
                run_dir, checkpoint
            )

        self.assertEqual(observed_path, evaluation_path)
        self.assertEqual(result['checkpoint_episode'], 0)
        self.assertEqual(result['selection_score'], 8500.0)

    def test_formal_methods_are_crossed_across_gpus(self):
        assignments = controller.formal_assignments(['M1', 'M4'])
        observed = {
            method: sorted(seed for name, seed, _, _ in assignments if name == method)
            for method in ('M1', 'M4')
        }
        self.assertEqual(observed, {'M1': [1, 2, 3], 'M4': [1, 2, 3]})
        self.assertEqual(sum(gpu == 0 for _, _, gpu, _ in assignments), 3)
        self.assertEqual(sum(gpu == 1 for _, _, gpu, _ in assignments), 3)
        self.assertEqual(
            {gpu for name, _, gpu, _ in assignments if name == 'M1'},
            {0, 1},
        )
        self.assertEqual(
            {gpu for name, _, gpu, _ in assignments if name == 'M4'},
            {0, 1},
        )

    def test_cpu_slices_are_disjoint_whole_core_pairs(self):
        def expand(spec):
            cpus = set()
            for part in spec.split(','):
                ends = [int(value) for value in part.split('-', 1)]
                cpus.update(range(ends[0], ends[-1] + 1))
            return cpus

        slices = [
            expand(spec)
            for gpu in (0, 1)
            for spec in controller.GPU_CPU_SLICES[gpu]
        ]
        self.assertEqual(set.union(*slices), set(range(144)))
        for left in range(len(slices)):
            self.assertEqual(len(slices[left]), 24)
            physical = {cpu if cpu < 72 else cpu - 72 for cpu in slices[left]}
            self.assertEqual(len(physical), 12)
            for right in range(left + 1, len(slices)):
                self.assertFalse(slices[left] & slices[right])


if __name__ == '__main__':
    unittest.main()
