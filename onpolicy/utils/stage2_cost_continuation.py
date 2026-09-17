"""Explicit, source-audited reuse of completed BC controls after cost-only edits.

This is not checkpoint/epoch recovery and cannot reuse a cost gate from an
older source. It carries the original global deadline, never a new 48 hours.
"""
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]

# No model, learner, loss, environment, teacher, evaluator or numerical
# configuration file can change when reusing the original N0/N1 controls.
COST_ONLY_FILES = frozenset({
    'onpolicy/utils/stage2_cost_actor_pool.py',
    'onpolicy/utils/stage2_cost_actor_process.py',
    'onpolicy/utils/stage2_cost_inference.py',
    'onpolicy/utils/stage2_cost_continuation.py',
    'onpolicy/utils/stage2_cost_accelerated.py',
    'onpolicy/utils/stage2_cost_schedule.py',
    'onpolicy/utils/stage2_cost_execution.py',
    'onpolicy/utils/stage2_cost_timeout.py',
    'onpolicy/utils/stage2_cost_timeout_reference.py',
    'onpolicy/scripts/train/verify_stage2_cost_timeout.py',
    'onpolicy/scripts/train/diagnose_stage2_cost_accelerated.py',
    'onpolicy/scripts/train/benchmark_stage2_cost_inference.py',
    'onpolicy/scripts/train/benchmark_stage2_cost_execution.py',
    'onpolicy/scripts/train/prepare_stage2_cost_accelerated.py',
    'onpolicy/scripts/train/run_stage2_cost_accelerated_pipeline.py',
    'onpolicy/scripts/train/analyze_stage2_cost_accelerated.py',
    'onpolicy/scripts/train/launch_stage2_cost_accelerated_gpu0.sh',
    'STAGE2_COST_ACCELERATION_IMPLEMENTATION_20260907.md',
})
CONTROL_CONTRACTS = ('source', 'ready_source', 'warmstart_sources', 'teacher_index',
    'teacher_index_sha256', 'sampling_audits', 'training_contract', 'evaluation_contract')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_change_audit(old, new):
    changes = {}
    for name in sorted(set(old) | set(new)):
        if old.get(name) == new.get(name):
            continue
        cost_test = (name.startswith('onpolicy/envs/HKBZ/test/test_stage2_cost_') and name.endswith('.py'))
        if name not in COST_ONLY_FILES and not cost_test:
            raise ValueError(f'Cannot reuse BC after changing scientific/control code: {name}')
        if name not in new:
            raise ValueError(f'Cannot reuse BC after deleting source: {name}')
        changes[name] = dict(before=old.get(name), after=new[name])
    return changes


def verify_archive(manifest):
    archive = manifest['source_archive']
    if digest(archive['path']) != archive['sha256']:
        raise ValueError('Archived BC source bundle changed.')
    with tarfile.open(archive['path'], 'r:gz') as handle:
        for name, expected in manifest['code_fingerprint'].items():
            stream = handle.extractfile(name)  # Read only: never extract paths to disk.
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected:
                raise ValueError(f'Archived BC source differs: {name}')


def validate_baseline(binding, manifest):
    for path, expected in binding['artifacts'].items():
        if digest(path) != expected:
            raise ValueError(f'Completed BC continuation artifact changed: {path}')
    report = json.loads(Path(binding['baseline_report']).read_text())
    baseline = json.loads(Path(report['manifest']).read_text())
    if (baseline.get('execution_stage') != 'bc_pilot' or report.get('complete') is not True
            or report.get('cost_integrity_passed') is not True
            or set(report['methods']) != {'N0_BC_teacher_seed11', 'N1_BC_dagger_seed11'}
            or any(method.get('integrity_passed') is not True for method in report['methods'].values())):
        raise ValueError('Only complete, integrity-checked N0/N1 controls can be reused.')
    for key in CONTROL_CONTRACTS:
        if baseline[key] != manifest[key]:
            raise ValueError(f'Completed BC has a different {key}.')
    changes = source_change_audit(baseline['code_fingerprint'], manifest['code_fingerprint'])
    if changes != binding['source_changes']:
        raise ValueError('Cost-only source amendment changed after continuation preparation.')
    return report, baseline


def build_continuation(parent_path, manifest):
    parent_path = Path(parent_path).resolve()
    parent = json.loads(parent_path.read_text())
    state_path = parent_path.parent / 'pipeline_state.json'
    state = json.loads(state_path.read_text())
    if (parent.get('execution_stage') != 'pipeline' or not state.get('ended_unix_time')
            or not state.get('baseline_completed') or state.get('cost_completed')
            or (state['phases'].get('bc_pilot', {}).get('status') != 'completed'
                and not parent.get('continuation'))):
        raise ValueError('Continuation requires a stopped pipeline with completed BC and unfinished cost work.')
    if parent.get('continuation'):
        validate_baseline(parent['continuation'], parent)
        report_path = Path(parent['continuation']['baseline_report'])
        baseline_path = Path(json.loads(report_path.read_text())['manifest']).resolve()
    else:
        command = state['phases']['bc_pilot']['command']
        baseline_path = Path(command[command.index('--manifest') + 1]).resolve()
        report_path = baseline_path.parent / 'analysis/comparison.json'
    baseline = json.loads(baseline_path.read_text())
    report = json.loads(report_path.read_text())
    status_path = baseline_path.parent / 'suite_status.json'
    if json.loads(status_path.read_text()).get('status') != 'completed':
        raise ValueError('Baseline suite is not completed.')
    verify_archive(parent)
    verify_archive(baseline)
    deadline = parent.get('continuation', {}).get('hard_deadline_unix_time',
        state['started_unix_time'] + parent['resource_contract']['hard_timeout_seconds'])
    if deadline <= time.time():
        raise ValueError('Original experiment deadline already expired; no automatic budget expansion.')
    paths = {parent_path, state_path, baseline_path, report_path, status_path,
             Path(parent['source_archive']['path']), Path(baseline['source_archive']['path'])}
    for method in report['methods'].values():
        run = Path(method['run_dir'])
        paths.add(Path(method['effective_checkpoint']))
        paths.update((run / 'models').glob('*.pt'))
        paths.update((run / 'evaluations').glob('*.json'))
        paths.update((run / 'logs').glob('cost_*epoch*.json'))
        paths.add(run / 'logs/request_ready_epoch_metrics.jsonl')
    binding = dict(parent_manifest=str(parent_path), baseline_report=str(report_path),
        hard_deadline_unix_time=deadline, baseline_retrained=False,
        old_cost_gates_reused=False, old_cost_cache_imported=False,
        artifacts={str(path): digest(path) for path in sorted(paths)},
        source_changes=source_change_audit(baseline['code_fingerprint'], manifest['code_fingerprint']))
    validate_baseline(binding, manifest)
    return binding
