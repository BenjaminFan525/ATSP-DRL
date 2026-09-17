"""Cost-only continuation cannot weaken scientific sources or reset budget."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from onpolicy.utils.stage2_cost_continuation import (
    CONTROL_CONTRACTS, digest, source_change_audit, validate_baseline,
)


class CostContinuationTest(unittest.TestCase):
    def test_only_explicit_cost_files_or_tests_may_change(self):
        original = {'onpolicy/utils/stage2_cost_actor_pool.py': 'old',
                    'onpolicy/algorithms/gnn_mappo/algorithm/gnn_actor_critic.py': 'protected'}
        amended = dict(original, **{'onpolicy/utils/stage2_cost_actor_pool.py': 'new',
                                   'onpolicy/utils/stage2_cost_inference.py': 'added'})
        self.assertEqual(len(source_change_audit(original, amended)), 2)
        for name in ('onpolicy/algorithms/gnn_mappo/algorithm/gnn_actor_critic.py',
                     'onpolicy/envs/HKBZ/environment.py',
                     'onpolicy/utils/stage2_cost_improvement.py',
                     'onpolicy/utils/stage2_matching.py'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                source_change_audit(original, dict(amended, **{name: 'changed'}))
        with self.assertRaises(ValueError):
            source_change_audit(original, {})

    def test_bound_complete_controls_only_and_no_unannounced_contract_change(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = dict(execution_stage='bc_pilot', code_fingerprint={'same.py': 'old'},
                            **{key: {'fixture': key} for key in CONTROL_CONTRACTS})
            path = root / 'manifest.json'
            path.write_text(json.dumps(baseline))
            report_path = root / 'comparison.json'
            report = dict(manifest=str(path), complete=True, cost_integrity_passed=True,
                          methods={name: {'integrity_passed': True} for name in
                                   ('N0_BC_teacher_seed11', 'N1_BC_dagger_seed11')})
            report_path.write_text(json.dumps(report))
            binding = dict(baseline_report=str(report_path), source_changes={},
                           artifacts={str(p): digest(p) for p in (path, report_path)})
            validate_baseline(binding, baseline)
            for key in CONTROL_CONTRACTS:
                amended = copy.deepcopy(baseline)
                amended[key] = 'changed'
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_baseline(binding, amended)
            report_path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                validate_baseline(binding, baseline)


if __name__ == '__main__':
    unittest.main()
