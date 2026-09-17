import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from onpolicy.scripts.train import run_stage1_tau_range_trial as trial


class TestTauRangeTrial(unittest.TestCase):

    def _args(self, source: Path, variant: str) -> Namespace:
        return Namespace(
            variant=variant,
            seed=1,
            run_tag='tau_test',
            source_command_json=source,
            epochs=4,
            train_sampling_size=480,
            validation_dir=Path('/validation'),
            partition_seed=20260803,
            potential_path=Path('/potential.json'),
            initial_checkpoint=Path('/checkpoint.pt'),
            shared_eval_socket='/tmp/tau-test.sock',
            cpu_set='0-11,72-83',
        )

    @staticmethod
    def _parse_options(command):
        options = {}
        switches = set()
        index = 2
        while index < len(command):
            flag = command[index]
            if not flag.startswith('--'):
                raise AssertionError(f'Unexpected positional token: {flag}')
            if index + 1 < len(command) and not command[index + 1].startswith('--'):
                if flag in options:
                    raise AssertionError(f'Duplicate option: {flag}')
                options[flag] = command[index + 1]
                index += 2
            else:
                if flag in switches:
                    raise AssertionError(f'Duplicate switch: {flag}')
                switches.add(flag)
                index += 1
        return options, switches

    def test_arms_only_change_tau_schedule_and_experiment_name(self):
        source_payload = {
            'command': [
                'python', 'train.py', '--env_name', 'HKBZ',
                '--experiment_name', 'base', '--anneal_original', '0.3',
                '--anneal_final', '0.3', '--use_eval',
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source.json'
            source.write_text(json.dumps(source_payload), encoding='utf-8')
            parsed = {
                variant: self._parse_options(
                    trial.training_command(self._args(source, variant))
                )
                for variant in trial.VARIANTS
            }

        fixed_options, fixed_switches = parsed['fixed_030']
        allowed_differences = {
            '--experiment_name', '--anneal_original',
            '--anneal_final', '--tau_anneal_epochs',
        }
        for variant, (options, switches) in parsed.items():
            self.assertEqual(switches, fixed_switches, variant)
            differences = {
                flag for flag in options.keys() | fixed_options.keys()
                if options.get(flag) != fixed_options.get(flag)
            }
            self.assertLessEqual(differences, allowed_differences, variant)
            self.assertEqual(options['--seed'], '1')
            self.assertEqual(options['--n_rollout_threads'], '60')
            self.assertEqual(options['--n_eval_rollout_threads'], '60')
            self.assertEqual(
                options['--shared_eval_socket'], '/tmp/tau-test.sock'
            )
            self.assertEqual(options['--evaluation_tau'], '0.3')

    def test_registered_schedules_have_a_final_stability_epoch(self):
        self.assertEqual(
            trial.VARIANTS['fixed_030']['expected_tau'],
            [0.3, 0.3, 0.3, 0.3],
        )
        self.assertEqual(
            trial.VARIANTS['range_050_030']['expected_tau'],
            [0.5, 0.4, 0.3, 0.3],
        )
        self.assertEqual(
            trial.VARIANTS['range_080_030']['expected_tau'],
            [0.8, 0.55, 0.3, 0.3],
        )


if __name__ == '__main__':
    unittest.main()
