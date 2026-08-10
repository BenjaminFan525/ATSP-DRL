import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from onpolicy.envs.HKBZ.experiment.analyze_iga_teacher_trajectories import (
    case_balanced_row_weights,
    case_level_cross_validation,
    temporal_tail_weights,
)
from onpolicy.envs.HKBZ.experiment.eval_common import list_case_folders
from onpolicy.scripts.train import run_stage1_tail_robustness_v2 as v2_scheduler
from onpolicy.scripts.train.shared_hkbz_evaluator import source_training_args
from onpolicy.scripts.train.run_stage1_tail_robustness_v2 import (
    FORMAL_SAFE_ASYNC_GRAPH_CLONE_WORKERS,
    completed_training_checkpoint,
    configure_safe_pipeline,
    configure_exact_recovery_resume,
    interrupted_training_checkpoint,
    parse_seed_list,
)
from onpolicy.scripts.train.train_hkbz import (
    apply_formal_safe_pipeline_manifest,
)
from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner
from onpolicy.utils.shared_eval import format_cpu_set, parse_cpu_set


class TailRobustnessV2Test(unittest.TestCase):
    def test_formal_seed_list_is_positive_unique_and_ordered(self):
        self.assertEqual(parse_seed_list('1,3,2'), (1, 3, 2))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_seed_list('1,1')
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_seed_list('0')

    def test_formal_only_mode_skips_screening_and_runs_requested_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                suite_dir=Path(temporary),
                lane=0,
                gpu=0,
                formal_only_variant='N1_tail_cv_potential',
                formal_seeds=(1, 2, 3),
                start_delay_seconds=0.0,
            )
            statuses = []

            def capture_status(_args, **updates):
                statuses.append(updates)

            with (
                mock.patch.object(v2_scheduler, 'parse_args', return_value=args),
                mock.patch.object(v2_scheduler, 'update_status', side_effect=capture_status),
                mock.patch.object(v2_scheduler, 'training_record') as training,
                mock.patch.object(v2_scheduler.signal, 'signal'),
            ):
                self.assertEqual(v2_scheduler.main(), 0)

            self.assertEqual(
                [call.args[2] for call in training.call_args_list],
                [1, 2, 3],
            )
            self.assertTrue(all(call.kwargs['formal'] for call in training.call_args_list))
            self.assertEqual(statuses[-1]['phase'], 'formal_extension_complete')

    def test_shared_evaluator_cpu_slices_are_exact_and_physical(self):
        pool = parse_cpu_set('0-35,72-107')
        slices = [
            parse_cpu_set('0-11,72-83'),
            parse_cpu_set('12-23,84-95'),
            parse_cpu_set('24-35,96-107'),
        ]
        self.assertEqual(set.union(*(set(value) for value in slices)), set(pool))
        self.assertFalse(any(
            slices[left] & slices[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ))
        self.assertTrue(all(
            len({cpu if cpu < 72 else cpu - 72 for cpu in cpu_set}) == 12
            for cpu_set in slices
        ))
        self.assertEqual(format_cpu_set(slices[0]), '0-11,72-83')

    def test_evaluator_source_command_cannot_restore_training_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            command_path = Path(temporary) / 'command.json'
            command_path.write_text(json.dumps({'command': [
                'python', 'train_hkbz.py', '--seed', '2',
                '--checkpoint_dir', 'recovery.pt', '--resume_stage1',
                '--reset_optimizers_on_resume',
                '--shared_eval_socket', '/tmp/old.sock',
                '--shared_eval_cpu_set', '0-3',
            ]}), encoding='utf-8')
            args = source_training_args(command_path)
            self.assertEqual(args[args.index('--seed') + 1], '2')
            for removed in (
                '--checkpoint_dir', '--resume_stage1',
                '--reset_optimizers_on_resume', '--shared_eval_socket',
                '--shared_eval_cpu_set',
            ):
                self.assertNotIn(removed, args)

    def test_remote_raw_evaluation_round_trip_preserves_selection_inputs(self):
        source = object.__new__(HKBZ_Runner)
        source.last_eval_case_count = 2
        source.last_eval_case_ids = ['case_a', 'case_b']
        source.last_eval_completed_count = 2
        source.last_eval_completion_rate = 1.0
        source.last_eval_timeout_count = 0
        source.last_eval_cycle_count = 0
        source.last_eval_mean_steps = 12.5
        source.last_eval_max_no_progress = 3
        source.last_eval_mean_relocations = 1.5
        source.last_eval_records = [
            {'case_key': 'case_a', 'makespan': 100.0},
            {'case_key': 'case_b', 'makespan': 120.0},
        ]
        payload = source._raw_evaluation_payload(110.0)

        sink = object.__new__(HKBZ_Runner)
        sink.pre_ppo_case_makespan = {'case_a': 105.0, 'case_b': 125.0}
        sink._selection_metrics_from_records = mock.Mock(return_value=111.0)
        sink._write_evaluation_records = mock.Mock()
        score = sink._consume_raw_evaluation(payload, evaluation_label='epoch_1')
        self.assertEqual(score, 111.0)
        self.assertEqual(sink.last_eval_case_ids, ['case_a', 'case_b'])
        self.assertEqual(
            [record['delta_vs_pre_ppo'] for record in sink.last_eval_records],
            [-5.0, -5.0],
        )
        sink._selection_metrics_from_records.assert_called_once_with(
            110.0, evaluation_label='epoch_1'
        )

    def test_remote_eval_snapshots_model_and_cleans_request_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = object.__new__(HKBZ_Runner)
            runner.run_dir = Path(temporary)
            runner.policy = SimpleNamespace(ac=SimpleNamespace(
                tau=0.3,
                plane_order_mode='fixed',
                plane_pair_decoder='joint_pair',
                state_dict=lambda: {'weight': torch.ones(2)},
            ))
            runner.evaluation_tau = 0.3
            runner.shared_eval_cpu_set = '0-3'
            runner.all_args = SimpleNamespace(seed=2)
            runner.n_eval_rollout_threads = 60
            runner.progress_callback = None
            runner._consume_raw_evaluation = mock.Mock(return_value=321.0)

            def response(request):
                checkpoint_path = Path(request['checkpoint_path'])
                self.assertTrue(checkpoint_path.is_file())
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.assertEqual(checkpoint['request_id'], request['request_id'])
                self.assertTrue(torch.equal(
                    checkpoint['model']['weight'], torch.ones(2)
                ))
                return {'evaluation': {}, 'evaluation_seconds': 1.5}

            runner.shared_eval_client = SimpleNamespace(request=response)
            self.assertEqual(
                runner._eval_via_shared_service('pre_ppo'),
                321.0,
            )
            self.assertEqual(
                list((Path(temporary) / 'shared_eval_requests').glob('*.pt')),
                [],
            )

    def test_safe_pipeline_can_be_enabled_for_the_current_phase(self):
        screen = [
            'python', 'train.py', '--safe_graph_batch_pipeline',
            '--safe_dagger_teacher_overlap',
            '--safe_async_graph_clone_workers', '9',
        ]
        configure_safe_pipeline(screen, enabled=False)
        self.assertEqual(
            screen[screen.index('--safe_async_graph_clone_workers') + 1],
            '0',
        )
        self.assertNotIn('--safe_graph_batch_pipeline', screen)
        self.assertNotIn('--safe_dagger_teacher_overlap', screen)

        formal = ['python', 'train.py']
        configure_safe_pipeline(formal, enabled=True)
        self.assertEqual(
            formal[formal.index('--safe_async_graph_clone_workers') + 1],
            str(FORMAL_SAFE_ASYNC_GRAPH_CLONE_WORKERS),
        )
        self.assertIn('--safe_graph_batch_pipeline', formal)
        self.assertIn('--safe_dagger_teacher_overlap', formal)

    def test_exact_recovery_resume_replaces_fresh_optimizer_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'checkpoint_Recovery.pt'
            torch.save({
                'stage': 'post_shard_recovery',
                'episodes': 2,
                'completed_shard': 15,
                'total_shards': 16,
                'total_num_steps': 650760,
            }, checkpoint)
            command = [
                'python', 'train.py', '--n_rollout_threads', '16',
                '--plane_bc_pretrain_epochs', '4', '--checkpoint_dir', 'old.pt',
                '--resume_stage1', '--reset_optimizers_on_resume',
            ]
            cursor = configure_exact_recovery_resume(command, checkpoint)
            self.assertEqual(cursor['completed_shards'], 15)
            self.assertEqual(
                command[command.index('--checkpoint_dir') + 1],
                str(checkpoint.resolve()),
            )
            self.assertEqual(
                command[command.index('--plane_bc_pretrain_epochs') + 1], '0'
            )
            self.assertIn('--resume_stage1', command)
            self.assertNotIn('--reset_optimizers_on_resume', command)

    def test_unfinished_best_is_not_mistaken_for_completed_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary)
            run_dir = result_root / 'experiment' / 'run1'
            model_dir = run_dir / 'models'
            model_dir.mkdir(parents=True)
            torch.save({'model': {'weight': torch.ones(1)}}, model_dir / 'checkpoint_Best.pt')
            torch.save({
                'model': {'weight': torch.ones(1)},
                'stage': 'post_shard_recovery',
                'completed_shard': 15,
                'total_shards': 16,
            }, model_dir / 'checkpoint_Recovery.pt')
            (run_dir / 'run_status.json').write_text(
                json.dumps({'status': 'running'}), encoding='utf-8'
            )
            with mock.patch.object(v2_scheduler, 'RESULT_ROOT', result_root):
                self.assertIsNone(completed_training_checkpoint('experiment'))
                self.assertEqual(
                    interrupted_training_checkpoint('experiment'),
                    model_dir / 'checkpoint_Recovery.pt',
                )
                (run_dir / 'run_status.json').write_text(
                    json.dumps({'status': 'completed'}), encoding='utf-8'
                )
                self.assertEqual(
                    completed_training_checkpoint('experiment'),
                    model_dir / 'checkpoint_Best.pt',
                )

    def test_formal_manifest_cannot_activate_screening(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_tag = 'stage1_tail_test_r1'
            manifest_dir = root / 'result/hkbz_train_logs' / run_tag
            manifest_dir.mkdir(parents=True)
            manifest = {
                'enabled': True,
                'formal_only': True,
                'run_tag': run_tag,
                'safe_async_graph_clone_workers': 4,
                'safe_graph_batch_pipeline': True,
                'safe_dagger_teacher_overlap': True,
            }
            (manifest_dir / 'formal_safe_pipeline.json').write_text(
                json.dumps(manifest), encoding='utf-8'
            )
            screen_args = SimpleNamespace(
                experiment_name=f'{run_tag}_screen_N0_seed1',
                safe_async_graph_clone_workers=0,
                safe_graph_batch_pipeline=False,
                safe_dagger_teacher_overlap=False,
            )
            self.assertIsNone(
                apply_formal_safe_pipeline_manifest(screen_args, root)
            )
            self.assertEqual(screen_args.safe_async_graph_clone_workers, 0)
            self.assertFalse(screen_args.safe_graph_batch_pipeline)

            formal_args = SimpleNamespace(
                experiment_name=f'{run_tag}_formal_N3_seed1',
                safe_async_graph_clone_workers=0,
                safe_graph_batch_pipeline=False,
                safe_dagger_teacher_overlap=False,
            )
            activation = apply_formal_safe_pipeline_manifest(
                formal_args, root
            )
            self.assertIsNotNone(activation)
            self.assertEqual(formal_args.safe_async_graph_clone_workers, 4)
            self.assertTrue(formal_args.safe_graph_batch_pipeline)
            self.assertTrue(formal_args.safe_dagger_teacher_overlap)

    def test_tail_potential_uses_two_monotone_weight_ramps(self):
        weights = temporal_tail_weights(
            np.asarray([0.0, 0.75, 0.825, 0.90, 0.95, 1.0]),
            tail_start=0.75,
            tail_weight=4.0,
            final_start=0.90,
            final_weight=8.0,
        )
        np.testing.assert_allclose(weights, [1.0, 1.0, 2.5, 4.0, 6.0, 8.0])

    def test_case_level_cv_is_finite_and_case_disjoint(self):
        rows = []
        x_rows = []
        targets = []
        for case_index in range(10):
            distribution = 'iid' if case_index < 5 else 'ood_stress'
            profile = 'balanced' if case_index < 5 else 'stress_arrival'
            for step in range(3):
                value = float(case_index + step + 1)
                rows.append({
                    'case': f'case_{case_index}',
                    'distribution': distribution,
                    'profile': profile,
                    'time_before': float(step),
                    'time_to_go': 3.0 - step,
                })
                x_rows.append([value, value * 0.5])
                targets.append(value * 2.0)
        x = np.asarray(x_rows)
        y = np.asarray(targets)
        row_weights = case_balanced_row_weights(
            rows,
            {'iid': 0.5, 'ood_stress': 0.5},
            profile_balance=True,
        )
        result = case_level_cross_validation(
            rows,
            x,
            y,
            row_weights,
            ridge=1e-3,
            folds=5,
            seed=7,
        )
        self.assertTrue(result['case_disjoint'])
        self.assertEqual(result['folds'], 5)
        self.assertTrue(np.isfinite(result['metrics']['r2']))

    def test_evaluation_partition_is_fixed_across_callers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(12):
                case = root / f'case_{index:04d}'
                case.mkdir()
                (case / 'metadata.json').write_text(
                    json.dumps({
                        'profile': 'balanced' if index < 6 else 'stress_arrival',
                    }),
                    encoding='utf-8',
                )
            tune = list_case_folders(
                str(root), 6, case_offset=0, partition_seed=19,
                stratify_by='profile',
            )
            select = list_case_folders(
                str(root), 6, case_offset=6, partition_seed=19,
                stratify_by='profile',
            )
            repeated = list_case_folders(
                str(root), 6, case_offset=0, partition_seed=19,
                stratify_by='profile',
            )
            self.assertEqual(tune, repeated)
            self.assertFalse(set(tune) & set(select))


if __name__ == '__main__':
    unittest.main()
