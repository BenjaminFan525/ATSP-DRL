"""Evidence-bound amendment for an environment-sensitive historical replay.

Changing the replay baseline is explicit, never a relaxed numeric tolerance.
The historical result remains immutable and is always reported separately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from onpolicy.envs.HKBZ.data_generator import _sha256 as canonical_sha256
from onpolicy.utils.stage2_bc_contract import compare_reference_cases


ORCHESTRATION_FILES = {
    'onpolicy/scripts/train/prepare_stage2_bc_injection.py',
    'onpolicy/scripts/train/run_stage2_bc_injection_suite.py',
    'onpolicy/scripts/train/launch_stage2_bc_injection_gpu0.sh',
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_rows(payload):
    rows = payload.get('cases')
    if rows is None:
        rows = payload.get('evaluation', {}).get('records')
    if not isinstance(rows, list):
        raise ValueError('Reference artifact lacks case-level evaluation records.')
    return rows


def validate_repeat_evidence(first, second, numerical_probe, checkpoint_sha256):
    comparison = compare_reference_cases(evaluation_rows(first), evaluation_rows(second),
        case_count=60, tolerance_seconds=0.0)
    if not comparison['passed']:
        raise ValueError('Current-environment replay is not exactly repeatable; no rebaseline allowed.')
    if (numerical_probe.get('status') != 'completed'
            or numerical_probe.get('strict_repeat_exact') is not True
            or numerical_probe.get('checkpoint_sha256') != checkpoint_sha256):
        raise ValueError('A completed exact-repeat numerical probe on the same checkpoint is required.')
    flips = [row for row in numerical_probe.get('observations', [])
             if row.get('variant') in {'strict20', 'nonstrict12'}
             and row.get('active_action_differences', 0) > 0]
    if not flips:
        raise ValueError('Numerical probe has not demonstrated an active-action difference.')
    return comparison


def validate_case_content(rows, dataset):
    """Validate actual inputs, not only possibly stale metadata fingerprints."""
    dataset = Path(dataset).resolve()
    file_count = 0
    for row in rows:
        name = row.get('case_dir', '')
        if not name.startswith('case_') or Path(name).name != name:
            raise ValueError('Invalid reference case directory.')
        values = {}
        fingerprints = row.get('fingerprints', {})
        files = fingerprints.get('files', {})
        required = {'job.json', 'fixed_resources.json', 'mobile_resources.json',
                    'sites.json', 'flights.json'}
        if set(files) != required:
            raise ValueError('Reference case does not fingerprint all five inputs.')
        for filename, expected in files.items():
            values[filename] = json.loads((dataset / name / filename).read_text())
            if canonical_sha256(values[filename]) != expected:
                raise ValueError(f'Reference case content changed: {name}/{filename}')
            file_count += 1
        if canonical_sha256(values) != row['case_sha256']:
            raise ValueError(f'Reference case fingerprint mismatch: {name}')
    return {'case_count': len(rows), 'input_file_count': file_count, 'passed': True}


def current_runtime():
    gpu = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'], text=True).strip()
    return {'python': sys.version, 'torch': str(torch.__version__),
            'cuda': torch.version.cuda, 'gpu0': gpu}


def build_local_reference(root, reference, *, diagnosis_dir, numerical_probe_dir,
                          dataset, workers):
    """Only accept completed independent probes; never bless an unknown drift."""
    root, diagnosis_dir = Path(root), Path(diagnosis_dir).resolve()
    probe_path = Path(numerical_probe_dir).resolve() / 'probe.json'
    status_path = diagnosis_dir / 'diagnostic_status.json'
    diagnostic = json.loads(status_path.read_text())
    if diagnostic.get('status') != 'completed' or workers != 12:
        raise ValueError('Local reference requires completed strict-12 diagnostic evidence.')
    if diagnostic.get('reference') != reference:
        raise ValueError('Diagnostic historical checkpoint/evaluation contract differs.')
    variants = diagnostic.get('variants', {})
    if any(variants.get(name, {}).get('status') != 'completed'
           for name in ('12_strict', '20_legacy')):
        raise ValueError('Both current-strict and historical-setting controls must complete.')
    if (not diagnostic.get('fresh_service_per_variant')
            or variants['12_strict'].get('strict') is not True
            or variants['12_strict'].get('workers') != 12
            or variants['20_legacy'].get('strict') is not False
            or variants['20_legacy'].get('workers') != 20):
        raise ValueError('Diagnostic services do not match the declared numerical controls.')
    source = Path(diagnostic['failed_suite'])
    first_path, second_path = source / 'reference_a3.json', diagnosis_dir / '12_strict/evaluation.json'
    legacy_path = diagnosis_dir / '20_legacy/evaluation.json'
    first, second, legacy = [json.loads(path.read_text())
                             for path in (first_path, second_path, legacy_path)]
    source_manifest = json.loads((source / 'manifest.json').read_text())
    if (source_manifest.get('reference_validation') != reference
            or source_manifest['evaluation_contract']['workers'] != workers
            or Path(source_manifest['evaluation_contract']['dataset']).resolve() != Path(dataset).resolve()):
        raise ValueError('The two replay sources do not use the same evaluation contract.')
    for payload in (first, second, legacy):
        if (payload.get('status') != 'completed' or payload.get('seed') != 1
                or payload.get('evaluation_tau') != 0.3
                or Path(payload.get('checkpoint', '')).resolve() != Path(reference['checkpoint']).resolve()):
            raise ValueError('Reference replay checkpoint, seed, tau or completion mismatch.')
    if (not first.get('model_sha256')
            or any(payload.get('model_sha256') != first['model_sha256'] for payload in (second, legacy))):
        raise ValueError('Reference controls do not share the same model-tensor digest.')
    for field in ('checkpoint', 'expected_evaluation'):
        if file_sha256(reference[field]) != reference[f'{field}_sha256']:
            raise ValueError(f'Historical reference {field} changed.')
    # Changes to launch/preflight orchestration are intentional; no numerical,
    # environment, model, evaluator, runner or existing utility edits are exempt.
    fingerprints = diagnostic['code_fingerprint']
    for relative, expected in fingerprints.items():
        if relative not in ORCHESTRATION_FILES and file_sha256(root / relative) != expected:
            raise ValueError(f'Numerical source changed since diagnostic: {relative}')
        if (relative not in ORCHESTRATION_FILES and relative in source_manifest['code_fingerprint']
                and source_manifest['code_fingerprint'][relative] != expected):
            raise ValueError(f'Independent replay sources differ: {relative}')
    numeric = json.loads(probe_path.read_text())
    repeat = validate_repeat_evidence(first, second, numeric, reference['checkpoint_sha256'])
    content = validate_case_content(evaluation_rows(second), dataset)
    historical = json.loads(Path(reference['expected_evaluation']).read_text())
    comparisons = {name: compare_reference_cases(evaluation_rows(historical), evaluation_rows(payload),
                      case_count=60, tolerance_seconds=reference['case_tolerance_seconds'])
                   for name, payload in [('local_strict12', second), ('local_legacy20', legacy)]}
    runtime = dict(diagnostic['runtime'])
    runtime['gpu0'] = next(row.strip() for row in diagnostic['hardware'].splitlines()
                          if row.strip().startswith('0,'))
    return {
        **reference,
        'mode': 'local_exact_after_environment_diagnosis',
        'historical_evaluation': reference['expected_evaluation'],
        'historical_evaluation_sha256': reference['expected_evaluation_sha256'],
        'expected_evaluation': str(second_path), 'expected_evaluation_sha256': file_sha256(second_path),
        'case_tolerance_seconds': 0.0,
        'runtime': runtime,
        'evaluation_workers': 12, 'strict_algorithms': True,
        'cublas_workspace_config': ':4096:8',
        'amendment': 'Cross-configuration historical equality is diagnostic; exact local replay remains mandatory.',
        'evidence_files': {str(path): file_sha256(path) for path in
            (status_path, first_path, second_path, legacy_path, probe_path, source / 'manifest.json',
             Path(reference['expected_evaluation']))},
        'repeat_comparison': repeat, 'data_integrity': content,
        'historical_comparisons': comparisons,
    }


def verify_local_reference_runtime(reference):
    if reference.get('mode') != 'local_exact_after_environment_diagnosis':
        return
    if current_runtime() != reference['runtime']:
        raise ValueError('Local replay baseline runtime/GPU differs from this launch.')
    for filename, expected in reference['evidence_files'].items():
        if file_sha256(filename) != expected:
            raise ValueError(f'Replay diagnostic evidence changed: {filename}')
