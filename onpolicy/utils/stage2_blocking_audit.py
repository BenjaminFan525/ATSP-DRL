"""Separate deterministic rule coverage from a DAgger policy's visited states."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from onpolicy.utils.stage2_matching import (
    TEACHER_PROJECTION_CONTRACT, project_teacher_matching, validate_teacher_matching,
)


BLOCKING_COVERAGE_CONTRACT = 'fixed_regressions_and_live_teacher_witness_v1'
CANARY_PAIRS = {
    'bc_canary': ('N0_BC_teacher', 'N1_BC_dagger'),
    'cost_canary': ('N2_cost_teacher', 'N3_cost_dagger'),
}


def fingerprint_digest(fingerprints):
    return hashlib.sha256(json.dumps(fingerprints, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def fixed_blocking_regressions():
    """Known counterexamples, not sampled or chosen by learned policy quality."""
    legal = np.zeros((5, 24), dtype=bool)
    for row, requests in enumerate(([0, 1, 2, 11, 13, 17, 20, 21],
                                   [0, 1, 2, 5, 11, 13, 17], [0, 23], [0, 23], [0, 19])):
        legal[row, requests] = True
    lookahead = np.ones(24, dtype=bool)
    lookahead[[0, 1, 2]] = False
    fixtures = [
        ('case_0343_blocking_priority', legal, [20, 1, 0, 0, 0], lookahead, [2, 1, 0, 0, 0]),
        ('serial_last_chance', np.asarray([
            [1, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1, 1, 0]], dtype=bool),
         [0, 7, 0, 5], np.asarray([0, 0, 0, 0, 1, 0, 0, 0], dtype=bool), [7, 1, 3, 5]),
    ]
    results = []
    for name, legal, original, lookahead, expected in fixtures:
        scores = np.where(legal, 0., -np.inf)
        try:
            validate_teacher_matching(scores, original, lookahead)
        except ValueError:
            pass
        else:
            raise ValueError(f'{name}: the known invalid teacher was not rejected.')
        original = np.asarray(original, dtype=np.int64)
        before = original.copy()
        actual, stats = project_teacher_matching(legal, original, lookahead)
        if not np.array_equal(actual, expected) or not np.array_equal(before, original):
            raise ValueError(f'{name}: repair changed its expected result or input.')
        validate_teacher_matching(scores, actual, lookahead)
        repeated, second = project_teacher_matching(legal, actual, lookahead)
        if (stats['teacher_projection_events'] != 1 or stats['teacher_projection_added_blocking'] <= 0
                or not np.array_equal(actual, repeated) or second['teacher_projection_events'] != 0):
            raise ValueError(f'{name}: missing repair or non-idempotent projection.')
        results.append(dict(name=name, original=original.tolist(), projected=actual.tolist(),
                            stats=stats, invalid_rejected=True, repaired_valid=True, idempotent=True))
    return results


def _count(row, name):
    value = row.get(name)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or int(value) != value):
        raise ValueError(f'Missing or invalid Blocking audit count: {name}.')
    return int(value)


def uses_fixed_blocking_coverage(manifest):
    contract = manifest.get('analysis_contract', {}).get('blocking_repair_coverage')
    if contract is None:
        return False  # Legacy manifests retain the original per-arm requirement.
    if contract != BLOCKING_COVERAGE_CONTRACT or manifest.get('research_family') != 'stage2_cost_accelerated':
        raise ValueError('Unknown or out-of-scope Blocking coverage contract.')
    return True


def audit_blocking_coverage(manifest, methods):
    """Require fixed counterexamples AND a same-suite all-teacher live witness.

    Every arm/epoch must still execute projected labels and pass its deployment
    audit. Zero *changes* on a DAgger trajectory is valid; zero *checks* is not.
    The all-teacher arm still must exercise a repair in EACH epoch, including
    provenance-checked raw event records. No event is fabricated for DAgger.
    """
    if not uses_fixed_blocking_coverage(manifest) or manifest.get('profile') != 'canary':
        raise ValueError('Blocking coverage requires an explicitly bound accelerated canary.')
    pair = CANARY_PAIRS.get(manifest.get('execution_stage'))
    if pair is None or set(manifest['commands']) != {arm + '_seed11' for arm in pair}:
        raise ValueError('Blocking coverage requires the matching teacher/DAgger pair.')
    if set(methods) != set(manifest['commands']):
        raise ValueError('Blocking coverage has missing methods.')
    fixtures = fixed_blocking_regressions()
    per_arm, witnesses = {}, []
    for arm, schedule in zip(pair, ([1., 1.], [.75, .5])):
        key = arm + '_seed11'
        entry, method = manifest['commands'][key], methods[key]
        if entry.get('teacher_execution_schedule') != schedule:
            raise ValueError('Blocking witness arm has a different behavior schedule.')
        rows = method['matching_training_metrics']
        if [row.get('device_bc_epoch') for row in rows] != [1, 2]:
            raise ValueError('Blocking coverage requires both completed epochs.')
        counts = []
        for row in rows:
            checked, events = (_count(row, field) for field in
                               ('teacher_projection_checked', 'teacher_projection_events'))
            if checked <= 0 or events > checked or row.get('device_bc_matching_audit_passed') is not True:
                raise ValueError(f'{key}: projected-label/deployment checks are missing or invalid.')
            if schedule == [1., 1.] and events <= 0:
                raise ValueError(f'{key}: all-teacher live Blocking witness is missing.')
            counts.append(dict(epoch=row['device_bc_epoch'], checked=checked, events=events))
        run = Path(method['run_dir'])
        path = run / 'logs/teacher_projection_events.jsonl'
        raw = path.read_bytes() if path.exists() else b''
        events = [json.loads(line) for line in raw.splitlines()]
        if len(events) != sum(row['events'] for row in counts):
            raise ValueError(f'{key}: Blocking event ledger differs from epoch counts.')
        for event in events:
            source = Path(event['teacher_path'])
            if (event.get('teacher_decoder_contract') != TEACHER_PROJECTION_CONTRACT
                    or event.get('derived_teacher_cmax') is not None
                    or _count(event, 'teacher_projection_events') != 1
                    or _count(event, 'teacher_projection_changed_rows') <= 0
                    or not source.is_file()
                    or hashlib.sha256(source.read_bytes()).hexdigest() != event['teacher_sha256']
                    or Path(event['teacher_projection_case']).name not in ('case_0343', 'case_0444')):
                raise ValueError(f'{key}: invalid live Blocking event provenance.')
        per_arm[key] = counts
        if schedule == [1., 1.]:
            witnesses.append(dict(arm=key, path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                  events=len(events)))
    return dict(contract=BLOCKING_COVERAGE_CONTRACT, passed=True, fixtures=fixtures,
                source_fingerprint_sha256=fingerprint_digest(manifest['code_fingerprint']),
                per_arm_epoch_counts=per_arm, live_teacher_witnesses=witnesses,
                note='Rule coverage is deterministic and shared; DAgger zero changes are not zero checks.')


def verify_blocking_coverage_proof(manifest, report):
    """Recompute, do not trust a stand-alone `passed` flag from a partial report."""
    recorded = report.get('blocking_repair_coverage')
    if recorded is None or recorded != audit_blocking_coverage(manifest, report['methods']):
        raise ValueError('Missing, changed or incomplete Blocking coverage proof.')
