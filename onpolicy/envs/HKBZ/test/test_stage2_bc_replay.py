"""Fail-closed tests for explicit, evidence-bound A3 replay rebaselining."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from onpolicy.envs.HKBZ.data_generator import _sha256
from onpolicy.utils.stage2_bc_replay import (
    build_local_reference, evaluation_rows, file_sha256,
    validate_case_content, validate_repeat_evidence, verify_local_reference_runtime,
)


def rows():
    return [dict(case_sha256=str(index), completed=True, makespan=1000.0 + index,
                 cycle_terminated=False, timeout=False) for index in range(60)]


def probe():
    return dict(status='completed', strict_repeat_exact=True, checkpoint_sha256='model',
        observations=[dict(variant='strict20', active_action_differences=2)])


class ReplayAmendmentTest(unittest.TestCase):
    def test_read_both_evaluation_formats_and_reject_missing_cases(self):
        self.assertEqual(evaluation_rows({'cases': rows()}), rows())
        self.assertEqual(evaluation_rows({'evaluation': {'records': rows()}}), rows())
        with self.assertRaises(ValueError):
            evaluation_rows({'summary': {'makespan': 1000}})

    def test_exact_repeat_is_required_not_old_tolerance(self):
        first = {'cases': rows()}
        second = {'evaluation': {'records': rows()}}
        self.assertTrue(validate_repeat_evidence(first, second, probe(), 'model')['passed'])
        second['evaluation']['records'][2]['makespan'] += 1e-6
        with self.assertRaisesRegex(ValueError, 'not exactly repeatable'):
            validate_repeat_evidence(first, second, probe(), 'model')

    def test_all_sixty_complete_unique_cases_required(self):
        for mutation in ('missing', 'duplicate', 'incomplete', 'nan'):
            changed = rows()
            if mutation == 'missing':
                changed.pop()
            elif mutation == 'duplicate':
                changed[1]['case_sha256'] = changed[0]['case_sha256']
            elif mutation == 'incomplete':
                changed[1]['completed'] = False
            else:
                changed[1]['makespan'] = float('nan')
            with self.assertRaises(ValueError):
                validate_repeat_evidence({'cases': rows()}, {'cases': changed}, probe(), 'model')

    def test_action_sensitivity_and_exact_control_are_both_required(self):
        for field, value in [('status', 'running'), ('strict_repeat_exact', False),
                             ('checkpoint_sha256', 'wrong'), ('observations', [])]:
            bad = probe()
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_repeat_evidence({'cases': rows()}, {'cases': rows()}, bad, 'model')

    def test_full_provenance_gate_and_content_fingerprints(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostic, source, numeric, dataset = [root / x for x in ('diagnostic', 'failed', 'numeric', 'data')]
            for path in (diagnostic, source, numeric, dataset):
                path.mkdir()

            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            cases = rows()
            for index, row in enumerate(cases):
                row['case_dir'] = f'case_{index:04d}'
                values = {name: {'index': index, 'file': name} for name in (
                    'job.json', 'fixed_resources.json', 'mobile_resources.json', 'sites.json', 'flights.json')}
                for name, value in values.items():
                    write(dataset / row['case_dir'] / name, value)
                row['case_sha256'] = _sha256(values)
                row['fingerprints'] = {'files': {name: _sha256(value) for name, value in values.items()}}
            checkpoint, historical = root / 'checkpoint.pt', root / 'historical.json'
            checkpoint.write_bytes(b'fixture checkpoint')
            old_cases = copy.deepcopy(cases)
            old_cases[0]['makespan'] += 20
            write(historical, {'cases': old_cases})
            reference = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                expected_evaluation=str(historical), expected_evaluation_sha256=file_sha256(historical),
                case_tolerance_seconds=1e-5, cases=60, seed=1, evaluation_tau=0.3)
            model_source = root / 'onpolicy/envs/HKBZ/environment.py'
            model_source.parent.mkdir(parents=True)
            model_source.write_text('fixture source')
            fingerprints = {str(model_source.relative_to(root)): file_sha256(model_source)}
            write(source / 'manifest.json', dict(reference_validation=reference, code_fingerprint=fingerprints,
                evaluation_contract={'dataset': str(dataset), 'workers': 12}))
            result = dict(status='completed', seed=1, evaluation_tau=0.3, checkpoint=str(checkpoint),
                          model_sha256='model-tensor-digest', evaluation={'records': cases})
            write(source / 'reference_a3.json', result)
            write(diagnostic / '12_strict/evaluation.json', result)
            write(diagnostic / '20_legacy/evaluation.json', result)
            state = dict(status='completed', reference=reference, failed_suite=str(source),
                code_fingerprint=fingerprints, fresh_service_per_variant=True,
                runtime={'python': 'fixture', 'torch': 'fixture', 'cuda': 'fixture'},
                hardware='0, GPU-fixture, Fixture GPU, fixture-driver\n',
                variants={'12_strict': {'status': 'completed', 'strict': True, 'workers': 12},
                          '20_legacy': {'status': 'completed', 'strict': False, 'workers': 20}})
            write(diagnostic / 'diagnostic_status.json', state)
            numeric_result = probe()
            numeric_result['checkpoint_sha256'] = reference['checkpoint_sha256']
            write(numeric / 'probe.json', numeric_result)

            def build():
                return build_local_reference(root, reference, diagnosis_dir=diagnostic,
                    numerical_probe_dir=numeric, dataset=dataset, workers=12)

            amended = build()
            self.assertEqual(amended['case_tolerance_seconds'], 0.0)
            self.assertEqual(amended['historical_evaluation'], str(historical))
            self.assertEqual(amended['data_integrity']['input_file_count'], 300)
            self.assertFalse(amended['historical_comparisons']['local_strict12']['passed'])
            self.assertEqual(json.loads(historical.read_text())['cases'], old_cases)
            with patch('onpolicy.utils.stage2_bc_replay.current_runtime', return_value=amended['runtime']):
                verify_local_reference_runtime(amended)
                numeric_result['status'] = 'running'
                write(numeric / 'probe.json', numeric_result)
                with self.assertRaisesRegex(ValueError, 'evidence changed'):
                    verify_local_reference_runtime(amended)
            with patch('onpolicy.utils.stage2_bc_replay.current_runtime', return_value={}):
                with self.assertRaisesRegex(ValueError, 'runtime/GPU'):
                    verify_local_reference_runtime(amended)
            numeric_result['status'] = 'completed'
            write(numeric / 'probe.json', numeric_result)
            wrong_model = copy.deepcopy(result)
            wrong_model['model_sha256'] = 'different-model'
            write(diagnostic / '20_legacy/evaluation.json', wrong_model)
            with self.assertRaisesRegex(ValueError, 'model-tensor digest'):
                build()
            write(diagnostic / '20_legacy/evaluation.json', result)
            state['status'] = 'running'
            write(diagnostic / 'diagnostic_status.json', state)
            with self.assertRaisesRegex(ValueError, 'completed strict-12'):
                build()
            state['status'] = 'completed'
            write(diagnostic / 'diagnostic_status.json', state)
            model_source.write_text('changed semantics')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                build()
            model_source.write_text('fixture source')
            write(dataset / 'case_0000/job.json', {'changed': True})
            with self.assertRaisesRegex(ValueError, 'content changed'):
                validate_case_content(cases, dataset)


if __name__ == '__main__':
    unittest.main()
