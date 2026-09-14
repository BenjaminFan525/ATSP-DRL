"""Dependencies for baseline-first Stage2 research; cost failures stay local."""

from onpolicy.utils.stage2_blocking_audit import verify_blocking_coverage_proof
from onpolicy.utils.stage2_cost_timeout import LEGACY_TIMEOUT_CONTRACT, TIMEOUT_CONTRACT

BASELINE_ARMS = ('N0_BC_teacher', 'N1_BC_dagger')
COST_ARMS = ('N2_cost_teacher', 'N3_cost_dagger')
STAGES = ('bc_canary', 'bc_pilot', 'acceleration', 'cost_diagnostic', 'cost_canary', 'cost_pilot')


def arms_for_stage(stage):
    if stage in ('bc_canary', 'bc_pilot'):
        return BASELINE_ARMS
    if stage in ('cost_canary', 'cost_pilot', 'cost_diagnostic'):
        return COST_ARMS
    if stage in ('pipeline', 'acceleration'):
        return BASELINE_ARMS + COST_ARMS
    raise ValueError(f'Unknown Stage2 stage: {stage}')


def prerequisites(stage, timeout_contract=LEGACY_TIMEOUT_CONTRACT):
    """N0/N1 never require cost microfit, cost evidence, or cost throughput."""
    required = {
        'pipeline': (), 'bc_canary': (),
        'bc_pilot': ('bc_canary',), 'acceleration': (),
        'cost_diagnostic': ('acceleration',),
        'cost_canary': ('acceleration', 'cost_diagnostic'),
        'cost_pilot': ('acceleration', 'cost_diagnostic', 'cost_canary'),
    }[stage]
    if timeout_contract == TIMEOUT_CONTRACT and stage in ('cost_diagnostic', 'cost_canary', 'cost_pilot'):
        return ('timeout_regression',) + required
    return required


def check_proof(stage, report, manifest, fingerprints, fanout_width, actor_lanes=1,
                inference_mode='legacy', timeout_contract=LEGACY_TIMEOUT_CONTRACT):
    if (manifest.get('research_family') != 'stage2_cost_accelerated'
            or manifest['code_fingerprint'] != fingerprints):
        raise ValueError('Stage proof has different source/family.')
    if manifest.get('cost_executor', {}).get('timeout_contract', LEGACY_TIMEOUT_CONTRACT) != timeout_contract:
        raise ValueError('Stage proof has different timeout contract.')
    if manifest.get('cost_executor', {}).get('inference_mode', 'legacy') != inference_mode:
        raise ValueError('Stage proof has different inference mode.')
    if stage == 'timeout_regression':
        execution = report.get('execution', {})
        if (timeout_contract != TIMEOUT_CONTRACT or report.get('timeout_contract') != timeout_contract
                or report.get('status') != 'completed' or report.get('smoke_prefix_steps', 0) != 0
                or report.get('timeout_gate_passed') is not True or report.get('completed') is not True
                or report.get('outcomes_exact') is not True or report.get('cache_exact') is not True
                or report.get('source_weights_unchanged') is not True
                or report.get('logical_queries') != 29 or report.get('batch_width') != 24
                or report.get('fanout_width') != fanout_width or report.get('actor_lanes') != actor_lanes
                or execution.get('physical_jobs') != 20 or execution.get('peak_live_jobs') != 16
                or execution.get('cache_hits') != 0 or report.get('branch_budget_seconds') != 600.):
            raise ValueError('Need complete exact cold-cache 29-query/20-job/16-live timeout regression.')
        return
    if stage == 'acceleration':
        fixtures = report.get('fixtures', [])
        if (report.get('acceleration_gate_passed') is not True
                or {x.get('batch_width') for x in fixtures} != {2, 24}
                or report.get('fanout_width') != fanout_width
                or report.get('actor_lanes', 1) != actor_lanes
                or report.get('inference_mode', 'legacy') != inference_mode
                or report.get('timeout_contract', LEGACY_TIMEOUT_CONTRACT) != timeout_contract
                or any(x.get('completed') is not True or x.get('trace_exact') is not True
                       or x.get('outcomes_exact') is not True or x.get('cache_exact') is not True
                       or not x.get('speedup', 0) > 1. for x in fixtures)):
            raise ValueError('Need completed exact 2/24-batch replay and measured speedup, not a prefix smoke.')
        if inference_mode != 'legacy' and any(
                row.get('previous_executor', {}).get('trace_exact') is not True
                or row.get('previous_executor', {}).get('outcomes_exact') is not True
                or not row.get('previous_executor_speedup', 0) > 1. for row in fixtures):
            raise ValueError('New inference must also beat the previous 8-branch/2-lane executor exactly.')
    if stage == 'cost_diagnostic':
        if (report.get('diagnostic_gate_passed') is not True
                or report.get('correctness_gate_passed') is not True
                or report.get('learnability_gate_passed') is not True):
            raise ValueError('Cost correctness/learnability proof failed.')
    elif stage != 'acceleration':
        expected = set(arms_for_stage(stage))
        actual = {entry['arm'] for entry in manifest['commands'].values()}
        if (manifest.get('execution_stage') != stage or actual != expected
                or report.get('canary_contract_passed') is not True
                or report.get('cost_integrity_passed') is not True):
            raise ValueError('Canary proof has the wrong arms/stage or failed integrity.')
        verify_blocking_coverage_proof(manifest, report)


def workload_estimate(actual_collection_seconds, logical_queries, *, cases=24):
    """A workload estimate is not a scientific gate or a measured training ETA."""
    if actual_collection_seconds < 0 or logical_queries < 0 or cases <= 0:
        raise ValueError('Invalid measured cost work.')
    per_arm = actual_collection_seconds * (240 / cases) * 2
    return dict(cost_work_per_arm_two_epochs_seconds=per_arm,
        two_cost_arms_sequential_work_seconds=per_arm * 2,
        actual_diagnostic_collection_seconds=actual_collection_seconds,
        logical_queries=logical_queries, parallel_training_eta_seconds=None,
        full_training_eta_seconds=None, budget_is_scientific_gate=False,
        note='Measured non-overlapping fanout wall time; excludes BC/evaluation, '
             'does not assume two trainers give 2x throughput. Global hard timeout still applies.')


def capacity_decision(estimate, remaining_seconds):
    """Separate engineering hold: never start a clearly over-budget cost pilot.

    This optimistic per-arm projection is NOT proof that parallel training fits.
    It only rejects runs whose label work for even one arm already exceeds the
    remaining global limit. It cannot undo completed N0/N1 or scientific gates.
    """
    per_arm = estimate['cost_work_per_arm_two_epochs_seconds']
    if per_arm < 0 or remaining_seconds < 0:
        raise ValueError('Invalid capacity input.')
    return dict(cost_pilot_launch_allowed=per_arm <= remaining_seconds,
        estimated_cost_only_per_arm_seconds=per_arm, remaining_hard_budget_seconds=remaining_seconds,
        scientific_gate_affected=False, baseline_affected=False, complete_training_eta_seconds=None,
        reason='Optimistic per-arm label-work estimate exceeds remaining hard budget.'
               if per_arm > remaining_seconds else
               'No obvious label-work budget overrun; this is not a measured parallel training ETA.')
