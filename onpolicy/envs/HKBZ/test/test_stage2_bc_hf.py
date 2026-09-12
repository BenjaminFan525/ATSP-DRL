"""H/F sensitivity must never silently become training or relabel a source."""
import copy
from types import SimpleNamespace
import unittest

from onpolicy.utils.stage2_bc_hf import (
    ALLOWED_CPUS, CAPACITY, GRID, GPU_UUID, PROTOCOL, SLOTS, SOURCE_SHA, TRAIN_ARMS,
    check_replay, configure_args, current_cases, evaluation_jobs, parse_arm,
    planning_for, validate_manifest,
)


def planning():
    return dict(device_future_intent_horizon=2, device_frontier_max_requests=4,
        device_future_intent_mode='bounded_frontier', device_lookahead_reservation_mode='soft',
        device_lookahead_dispatch=True, device_lookahead_safety_margin=60.,
        device_reservation_grace_seconds=300., device_departure_lookahead=True)


def rows(n, prefix):
    counts = (30, 27, 3) if n == 60 else (120, 108, 12)
    groups = [g for g, size in zip(('iid', 'ood_stress', 'ood_scale'), counts) for _ in range(size)]
    return [dict(case_dir=f'case_{prefix}_{i}', case_sha256=f'{prefix}_{i}', distribution=g,
                 makespan=100. + i, steps=5, completed=True, cycle_terminated=False, timeout=False)
            for i, g in enumerate(groups)]


def manifest():
    return dict(protocol=PROTOCOL, jobs=evaluation_jobs(), train_anchors=list(TRAIN_ARMS),
        source=dict(sha256=SOURCE_SHA), source_planning_contract=planning(),
        evaluation=dict(cases=rows(60, 'tune'), independent=False), train=dict(cases=rows(240, 'train')),
        execution=dict(evaluation_workers=12, fixed_request_capacity=5,
            max_parallel_evaluators=2, total_case_episode_limit=840, evaluation_jobs=14,
            actor_updates=0, critic_updates=0, teacher_queries=0, search_calls=0,
            automatic_training=False, automatic_retry=False, automatic_stage3=False,
            rollout_max_steps=4000, runtime_max_seconds=None),
        resource_contract=dict(gpu=0, gpu_uuid=GPU_UUID, allowed_cpus=ALLOWED_CPUS,
            slots=list(SLOTS), monitor_cpus='30-31,94-95', cuda_memory_fraction=.40,
            blas_threads=1, heartbeat_seconds=30))


class HFTests(unittest.TestCase):
    def test_grid_and_budget_are_explicit(self):
        self.assertEqual(len(set(GRID)), 12)
        jobs = evaluation_jobs()
        self.assertEqual(len(jobs) * 60, 840)
        self.assertTrue(jobs[0]['native'])
        self.assertEqual(jobs[-1]['label'], 'H2_F4_repeat')
        self.assertTrue(all(not j['native'] for j in jobs[1:]))
        validate_manifest(manifest())

    def test_only_hf_changes_semantic_contract(self):
        for arm in GRID:
            with self.subTest(arm=arm):
                source = planning(); saved = copy.deepcopy(source)
                changed = planning_for(source, arm)
                h, f = parse_arm(arm)
                self.assertEqual(source, saved)
                self.assertEqual(changed['device_future_intent_horizon'], h)
                self.assertEqual(changed['device_frontier_max_requests'], f)
                for key in source.keys() - {'device_future_intent_horizon', 'device_frontier_max_requests'}:
                    self.assertEqual(source[key], changed[key])

    def test_invalid_grid_rejected(self):
        for arm in ['H0_F2', 'H4_F4', 'H2_F0', 'H2_F5', '../H2_F4']:
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                parse_arm(arm)

    def test_teacher_disabled_and_padding_fixed(self):
        args = SimpleNamespace(resource_iga_teacher_dir='teacher', resource_iga_teacher_index='index',
            device_lookahead_safety_margin=60., request_ready_policy_injection='none')
        configure_args(args, 'H1_F1')
        self.assertEqual(args.device_request_capacity_per_plane, CAPACITY)
        self.assertEqual(args.device_lookahead_safety_margin, 60.)
        self.assertEqual(args.resource_iga_teacher_dir, '')
        self.assertEqual(args.resource_iga_teacher_index, '')
        self.assertTrue(args.device_global_matching)
        self.assertFalse(args.stage2_resource_v6_observations)
        with self.assertRaises(ValueError):
            configure_args(args, 'H1_F1', native=True)

    def test_no_silent_scope_expansion(self):
        for key, value in [('automatic_training', True), ('actor_updates', 1),
                ('teacher_queries', 1), ('search_calls', 1), ('total_case_episode_limit', 900),
                ('evaluation_workers', 24), ('automatic_retry', True), ('runtime_max_seconds', 1800)]:
            m = manifest(); m['execution'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(m)

    def test_source_resources_data_and_anchor_changes_rejected(self):
        for key, mutate in [
            ('gpu', lambda m: m['resource_contract'].update(gpu=1)),
            ('cpu', lambda m: m['resource_contract'].update(allowed_cpus='0-127')),
            ('source', lambda m: m['source'].update(sha256='wrong')),
            ('anchors', lambda m: m['train_anchors'].pop()),
            ('data', lambda m: m['evaluation']['cases'].pop()),
            ('duplicate', lambda m: m['train']['cases'].__setitem__(1, m['train']['cases'][0])),
            ('blind', lambda m: m['evaluation'].update(independent=True))]:
            m = manifest(); mutate(m)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(m)

    def test_current_metrics_never_inherit_old_diagnostics(self):
        original = dict(case_dir='x', case_sha256='sha', distribution='iid', profile='p',
            makespan=900., resource_policy_defer_seconds=12345., finish_steps=400)
        runtime = dict(case_dir='x', makespan=800., steps=380, completed=True,
                       timeout=False, cycle_terminated=False)
        result = current_cases([runtime], [original])[0]
        self.assertEqual(result['makespan'], 800.)
        self.assertEqual(result['case_sha256'], 'sha')
        self.assertNotIn('resource_policy_defer_seconds', result)
        self.assertNotIn('finish_steps', result)
        with self.assertRaises(ValueError):
            current_cases([{**runtime, 'case_sha256': 'different'}], [original])

    def test_strict_replay_checks_cases_steps_and_not_only_mean(self):
        a = rows(60, 'tune'); b = copy.deepcopy(a)
        self.assertTrue(check_replay(a, b)['passed'])
        b[0]['makespan'] += 1.; b[1]['makespan'] -= 1.
        self.assertFalse(check_replay(a, b)['passed'])
        b = copy.deepcopy(a); b[0]['steps'] += 1
        self.assertFalse(check_replay(a, b)['passed'])
        b = copy.deepcopy(a); b[0]['makespan'] = float('nan')
        with self.assertRaises(ValueError):
            check_replay(a, b)
        b = copy.deepcopy(a); b[0]['timeout'] = True
        with self.assertRaises(ValueError):
            check_replay(a, b)

    def test_old_finish_steps_supported_but_missing_steps_rejected(self):
        a = rows(60, 'tune'); b = copy.deepcopy(a)
        for row in a:
            row['finish_steps'] = row.pop('steps')
        self.assertTrue(check_replay(a, b)['passed'])
        del a[0]['finish_steps']
        with self.assertRaises(ValueError):
            check_replay(a, b)

    def test_collector_refuses_training(self):
        from onpolicy.scripts.train.run_stage2_bc_hf import FrozenEvaluator
        evaluator = object.__new__(FrozenEvaluator)
        with self.assertRaises(ValueError):
            evaluator.collect(None, training=True, hungarian=True)
        with self.assertRaises(ValueError):
            evaluator.collect(None, training=False, hungarian=False)

    def test_forward_refuses_autograd(self):
        import torch
        from onpolicy.scripts.train.run_stage2_bc_hf import FrozenEvaluator
        evaluator = object.__new__(FrozenEvaluator)
        with torch.enable_grad(), self.assertRaises(RuntimeError):
            evaluator.collect_forward(None, None, None, None, None, None)


if __name__ == '__main__':
    unittest.main()
