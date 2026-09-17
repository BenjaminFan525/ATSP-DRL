"""Do not regress the cost-independent controls or weaken scientific gates."""
import copy
import tempfile
import unittest

from onpolicy.envs.HKBZ.test.test_stage2_blocking_audit import make_coverage_fixture
from onpolicy.utils.stage2_blocking_audit import audit_blocking_coverage

from onpolicy.utils.stage2_cost_schedule import (
    BASELINE_ARMS, COST_ARMS, STAGES, arms_for_stage, capacity_decision, check_proof, prerequisites, workload_estimate,
)
from onpolicy.utils.stage2_cost_timeout import TIMEOUT_CONTRACT


class CostScheduleTest(unittest.TestCase):
    def test_bc_has_no_cost_dependency(self):
        self.assertEqual(prerequisites('bc_canary'), ())
        self.assertEqual(prerequisites('bc_pilot'), ('bc_canary',))
        self.assertEqual(arms_for_stage('bc_pilot'), BASELINE_ARMS)
        self.assertLess(STAGES.index('bc_pilot'), STAGES.index('cost_diagnostic'))

    def test_cost_pilot_requires_all_cost_proofs(self):
        self.assertEqual(prerequisites('cost_pilot'), ('acceleration', 'cost_diagnostic', 'cost_canary'))
        self.assertEqual(arms_for_stage('cost_pilot'), COST_ARMS)

    def test_timeout_amendment_needs_real_load_regression_before_any_cost_stage(self):
        for stage in ('cost_diagnostic', 'cost_canary', 'cost_pilot'):
            self.assertIn('timeout_regression', prerequisites(stage, TIMEOUT_CONTRACT))
        self.assertEqual(prerequisites('bc_pilot', TIMEOUT_CONTRACT), ('bc_canary',))

    def test_small_or_warm_or_partial_load_cannot_certify_timeout_fix(self):
        fingerprints = {'example.py': 'abc'}
        manifest = dict(research_family='stage2_cost_accelerated', code_fingerprint=fingerprints,
                        cost_executor={'timeout_contract': TIMEOUT_CONTRACT, 'inference_mode': 'host_matching'})
        report = dict(status='completed', timeout_contract=TIMEOUT_CONTRACT, timeout_gate_passed=True,
            completed=True, outcomes_exact=True, cache_exact=True, source_weights_unchanged=True,
            logical_queries=29, batch_width=24, fanout_width=16, actor_lanes=4, branch_budget_seconds=600.,
            execution={'physical_jobs': 20, 'peak_live_jobs': 16, 'cache_hits': 0})
        check_proof('timeout_regression', report, manifest, fingerprints, 16, 4, 'host_matching', TIMEOUT_CONTRACT)
        for key, value in (('peak_live_jobs', 4), ('cache_hits', 29), ('physical_jobs', 3)):
            changed = copy.deepcopy(report)
            changed['execution'][key] = value
            with self.assertRaises(ValueError):
                check_proof('timeout_regression', changed, manifest, fingerprints, 16, 4, 'host_matching', TIMEOUT_CONTRACT)
        for key, value in (('completed', False), ('smoke_prefix_steps', 8), ('branch_budget_seconds', 1200.),
                           ('outcomes_exact', False), ('logical_queries', 4)):
            with self.assertRaises(ValueError):
                check_proof('timeout_regression', dict(report, **{key: value}), manifest,
                            fingerprints, 16, 4, 'host_matching', TIMEOUT_CONTRACT)

    def test_estimate_is_actual_work_not_parallel_training_eta(self):
        report = workload_estimate(600., 80)
        self.assertEqual(report['cost_work_per_arm_two_epochs_seconds'], 12000.)
        self.assertEqual(report['two_cost_arms_sequential_work_seconds'], 24000.)
        self.assertIsNone(report['full_training_eta_seconds'])
        self.assertIsNone(report['parallel_training_eta_seconds'])
        self.assertFalse(report['budget_is_scientific_gate'])

    def test_huge_estimate_does_not_change_scientific_result(self):
        result = workload_estimate(40000., 192)
        self.assertGreater(result['two_cost_arms_sequential_work_seconds'], 172800)
        self.assertFalse(result['budget_is_scientific_gate'])

    def test_capacity_hold_is_separate_and_does_not_block_baselines(self):
        estimate = workload_estimate(10000., 192)
        result = capacity_decision(estimate, 172800.)
        self.assertFalse(result['cost_pilot_launch_allowed'])
        self.assertFalse(result['scientific_gate_affected'])
        self.assertFalse(result['baseline_affected'])
        self.assertIsNone(result['complete_training_eta_seconds'])

    def test_serial_sum_is_not_used_as_a_parallel_wall_gate(self):
        estimate = workload_estimate(100., 100)
        self.assertGreater(estimate['two_cost_arms_sequential_work_seconds'], 3000.)
        self.assertTrue(capacity_decision(estimate, 3000.)['cost_pilot_launch_allowed'])

    def test_prefix_or_partial_replay_cannot_pass_full_acceleration_gate(self):
        fingerprints = {'example.py': 'abc'}
        manifest = dict(research_family='stage2_cost_accelerated', code_fingerprint=fingerprints)
        fixture = dict(completed=True, trace_exact=True, outcomes_exact=True, cache_exact=True, speedup=2.)
        report = dict(acceleration_gate_passed=True, fanout_width=8,
                      fixtures=[dict(fixture, batch_width=width) for width in (2, 24)])
        check_proof('acceleration', report, manifest, fingerprints, 8)
        for key, value in [('completed', False), ('trace_exact', False), ('outcomes_exact', False),
                           ('cache_exact', False), ('speedup', 1.)]:
            broken = copy.deepcopy(report)
            broken['fixtures'][1][key] = value
            with self.assertRaises(ValueError):
                check_proof('acceleration', broken, manifest, fingerprints, 8)
        for fixtures in (report['fixtures'][:1], report['fixtures'][1:], []):
            with self.assertRaises(ValueError):
                check_proof('acceleration', dict(report, fixtures=fixtures), manifest, fingerprints, 8)
        with self.assertRaises(ValueError):
            check_proof('acceleration', report, manifest, fingerprints, 4)
        with self.assertRaises(ValueError):
            check_proof('acceleration', report, manifest, fingerprints, 8, actor_lanes=2)

    def test_correctness_and_learnability_are_still_mandatory(self):
        fingerprints = {'example.py': 'abc'}
        manifest = dict(research_family='stage2_cost_accelerated', code_fingerprint=fingerprints)
        report = dict(diagnostic_gate_passed=True, correctness_gate_passed=True, learnability_gate_passed=True)
        check_proof('cost_diagnostic', report, manifest, fingerprints, 8)
        for field in report:
            with self.assertRaises(ValueError):
                check_proof('cost_diagnostic', dict(report, **{field: False}), manifest, fingerprints, 8)

    def test_new_inference_is_bound_and_must_beat_previous_executor(self):
        fingerprints = {'example.py': 'abc'}
        manifest = dict(research_family='stage2_cost_accelerated', code_fingerprint=fingerprints,
                        cost_executor={'inference_mode': 'host_matching'})
        fixture = dict(completed=True, trace_exact=True, outcomes_exact=True, cache_exact=True, speedup=2.,
                       previous_executor=dict(trace_exact=True, outcomes_exact=True),
                       previous_executor_speedup=1.2)
        report = dict(acceleration_gate_passed=True, fanout_width=16, actor_lanes=4,
                      inference_mode='host_matching',
                      fixtures=[dict(fixture, batch_width=width) for width in (2, 24)])
        check_proof('acceleration', report, manifest, fingerprints, 16, 4, 'host_matching')
        with self.assertRaises(ValueError):
            check_proof('acceleration', report, manifest, fingerprints, 16, 4, 'legacy')
        for field in ('previous_executor_speedup', 'previous_executor'):
            broken = copy.deepcopy(report)
            broken['fixtures'][0].pop(field)
            with self.assertRaises(ValueError):
                check_proof('acceleration', broken, manifest, fingerprints, 16, 4, 'host_matching')

    def test_baseline_canary_not_a_cost_canary(self):
        with tempfile.TemporaryDirectory() as root:
            manifest, methods = make_coverage_fixture(root)
            fingerprints = manifest['code_fingerprint']
            report = dict(canary_contract_passed=True, cost_integrity_passed=True, methods=methods,
                          blocking_repair_coverage=audit_blocking_coverage(manifest, methods))
            check_proof('bc_canary', report, manifest, fingerprints, 8)
            with self.assertRaises(ValueError):
                check_proof('cost_canary', report, manifest, fingerprints, 8)
            with self.assertRaises(ValueError):
                check_proof('bc_canary', report, manifest, {'example.py': 'changed'}, 8)
            report.pop('blocking_repair_coverage')
            with self.assertRaises(ValueError):
                check_proof('bc_canary', report, manifest, fingerprints, 8)


if __name__ == '__main__':
    unittest.main()
