"""Rule coverage must not depend on DAgger revisiting a teacher-only state."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from onpolicy.utils.stage2_blocking_audit import (
    BLOCKING_COVERAGE_CONTRACT, CANARY_PAIRS, audit_blocking_coverage,
    fixed_blocking_regressions, uses_fixed_blocking_coverage, verify_blocking_coverage_proof,
)
from onpolicy.utils.stage2_matching import TEACHER_PROJECTION_CONTRACT


def make_coverage_fixture(root, stage='bc_canary'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    teacher = root / 'teacher.json'
    teacher.write_text('{}')
    event = dict(teacher_path=str(teacher), teacher_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
                 teacher_decoder_contract=TEACHER_PROJECTION_CONTRACT,
                 teacher_projection_case='dataset/train/case_0343', derived_teacher_cmax=None,
                 teacher_projection_events=1, teacher_projection_changed_rows=1)
    manifest = dict(research_family='stage2_cost_accelerated', profile='canary', execution_stage=stage,
                    code_fingerprint={'rule.py': 'same-source'}, commands={},
                    analysis_contract={'blocking_repair_coverage': BLOCKING_COVERAGE_CONTRACT})
    methods = {}
    for arm, schedule, events in zip(CANARY_PAIRS[stage], ([1., 1.], [.75, .5]), (1, 0)):
        key = arm + '_seed11'
        run = root / arm
        (run / 'logs').mkdir(parents=True)
        if events:
            (run / 'logs/teacher_projection_events.jsonl').write_text((json.dumps(event) + '\n') * 2)
        manifest['commands'][key] = dict(arm=arm, teacher_execution_schedule=schedule)
        methods[key] = dict(run_dir=str(run), matching_training_metrics=[
            dict(device_bc_epoch=epoch, teacher_projection_checked=20,
                 teacher_projection_events=events, device_bc_matching_audit_passed=True)
            for epoch in (1, 2)])
    return manifest, methods


class BlockingCoverageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='stage2-blocking-audit-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest, self.methods = make_coverage_fixture(self.root)

    def test_fixed_counterexamples_repair_validate_and_are_idempotent(self):
        rows = fixed_blocking_regressions()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['projected'], [2, 1, 0, 0, 0])
        self.assertEqual(rows[1]['projected'], [7, 1, 3, 5])
        self.assertTrue(all(row['invalid_rejected'] and row['repaired_valid'] and row['idempotent'] for row in rows))

    def test_broken_rule_cannot_be_hidden_by_live_counts(self):
        with patch('onpolicy.utils.stage2_blocking_audit.project_teacher_matching',
                   return_value=(np.array([20, 1, 0, 0, 0]), {})):
            with self.assertRaisesRegex(ValueError, 'expected result'):
                audit_blocking_coverage(self.manifest, self.methods)

    def test_dagger_zero_repairs_pass_with_actual_checks_and_teacher_witness(self):
        report = audit_blocking_coverage(self.manifest, self.methods)
        self.assertTrue(report['passed'])
        self.assertEqual([r['events'] for r in report['per_arm_epoch_counts']['N1_BC_dagger_seed11']], [0, 0])
        self.assertEqual(report['live_teacher_witnesses'][0]['events'], 2)

    def test_same_contract_covers_cost_canary_pair(self):
        manifest, methods = make_coverage_fixture(self.root / 'cost', 'cost_canary')
        self.assertTrue(audit_blocking_coverage(manifest, methods)['passed'])

    def test_teacher_requires_repair_in_each_epoch(self):
        self.methods['N0_BC_teacher_seed11']['matching_training_metrics'][1]['teacher_projection_events'] = 0
        with self.assertRaisesRegex(ValueError, 'live Blocking witness'):
            audit_blocking_coverage(self.manifest, self.methods)

    def test_dagger_missing_checks_or_failed_deployment_audit_is_rejected(self):
        for field, value in [('teacher_projection_checked', 0), ('device_bc_matching_audit_passed', False),
                             ('teacher_projection_events', 21)]:
            with self.subTest(field=field):
                methods = copy.deepcopy(self.methods)
                methods['N1_BC_dagger_seed11']['matching_training_metrics'][0][field] = value
                with self.assertRaises(ValueError):
                    audit_blocking_coverage(self.manifest, methods)

    def test_missing_noninteger_negative_and_nonfinite_counts_rejected(self):
        for value in [None, -1, .5, float('nan'), float('inf'), True]:
            with self.subTest(value=value):
                methods = copy.deepcopy(self.methods)
                methods['N1_BC_dagger_seed11']['matching_training_metrics'][0]['teacher_projection_events'] = value
                with self.assertRaisesRegex(ValueError, 'invalid Blocking audit count'):
                    audit_blocking_coverage(self.manifest, methods)

    def test_missing_epoch_or_arm_cannot_supply_coverage(self):
        methods = copy.deepcopy(self.methods)
        methods['N1_BC_dagger_seed11']['matching_training_metrics'].pop()
        with self.assertRaises(ValueError):
            audit_blocking_coverage(self.manifest, methods)
        methods.pop('N1_BC_dagger_seed11')
        with self.assertRaises(ValueError):
            audit_blocking_coverage(self.manifest, methods)

    def test_missing_or_extra_event_rows_are_rejected(self):
        path = Path(self.methods['N0_BC_teacher_seed11']['run_dir']) / 'logs/teacher_projection_events.jsonl'
        path.write_text(path.read_text().splitlines()[0] + '\n')
        with self.assertRaisesRegex(ValueError, 'ledger differs'):
            audit_blocking_coverage(self.manifest, self.methods)

    def test_changed_teacher_cannot_supply_live_witness(self):
        (self.root / 'teacher.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            audit_blocking_coverage(self.manifest, self.methods)

    def test_bad_event_contract_case_or_derived_cost_is_rejected(self):
        path = Path(self.methods['N0_BC_teacher_seed11']['run_dir']) / 'logs/teacher_projection_events.jsonl'
        original = path.read_text()
        for field, value in [('teacher_decoder_contract', 'legacy'), ('derived_teacher_cmax', 100.),
                             ('teacher_projection_case', 'case_9999')]:
            with self.subTest(field=field):
                events = [json.loads(line) for line in original.splitlines()]
                events[0][field] = value
                path.write_text(''.join(json.dumps(row) + '\n' for row in events))
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    audit_blocking_coverage(self.manifest, self.methods)

    def test_legacy_contract_not_silently_changed_and_unknown_contract_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['analysis_contract'].clear()
        self.assertFalse(uses_fixed_blocking_coverage(manifest))
        manifest['analysis_contract']['blocking_repair_coverage'] = 'skip'
        with self.assertRaises(ValueError):
            uses_fixed_blocking_coverage(manifest)
        manifest['analysis_contract']['blocking_repair_coverage'] = BLOCKING_COVERAGE_CONTRACT
        manifest['research_family'] = 'stage2_matching_gap'
        with self.assertRaises(ValueError):
            uses_fixed_blocking_coverage(manifest)

    def test_different_teacher_schedule_is_not_a_control_witness(self):
        self.manifest['commands']['N0_BC_teacher_seed11']['teacher_execution_schedule'] = [.75, .5]
        with self.assertRaisesRegex(ValueError, 'behavior schedule'):
            audit_blocking_coverage(self.manifest, self.methods)

    def test_proof_is_recomputed_and_source_bound_not_a_passed_flag(self):
        report = dict(methods=self.methods, blocking_repair_coverage=audit_blocking_coverage(self.manifest, self.methods))
        verify_blocking_coverage_proof(self.manifest, report)
        with self.assertRaises(ValueError):
            verify_blocking_coverage_proof(self.manifest, dict(methods=self.methods))
        with self.assertRaises(ValueError):
            verify_blocking_coverage_proof(self.manifest, dict(report, blocking_repair_coverage={'passed': True}))
        self.manifest['code_fingerprint']['rule.py'] = 'changed-source'
        with self.assertRaises(ValueError):
            verify_blocking_coverage_proof(self.manifest, report)


if __name__ == '__main__':
    unittest.main()
