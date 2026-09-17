"""Fixed full-load regression: exactly the r3 batch that timed out under r4."""
import json
from pathlib import Path

from onpolicy.utils.stage2_cost_continuation import digest, source_change_audit, verify_archive
from onpolicy.utils.stage2_cost_execution import json_digest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MANIFEST = ROOT / ('result/hkbz_train_logs/'
    'stage2_cost_accelerated_20260907_gpu0_half_r3_cost_diagnostic/manifest.json')


def bind_timeout_reference(current_manifest):
    manifest = json.loads(REFERENCE_MANIFEST.read_text())
    verify_archive(manifest)
    source_change_audit(manifest['code_fingerprint'], current_manifest['code_fingerprint'])
    for key in ('source', 'ready_source', 'warmstart_sources', 'teacher_index_sha256'):
        if manifest[key] != current_manifest[key]:
            raise ValueError(f'Timeout regression has different scientific input: {key}')
    evidence = REFERENCE_MANIFEST.parent / 'diagnostics/cost_evidence_epoch1.jsonl'
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    if (len(rows) != 10 or {row['step'] for row in rows} != {5}
            or sum(len(row['outcomes']) for row in rows) != 29
            or any(row['batch_width'] != 24 or not all(x['completed'] for x in row['outcomes'])
                   for row in rows)):
        raise ValueError('Expected the fixed, fully completed 29-query/24-case r3 regression.')
    first = rows[0]
    for row in rows:
        for key in ('state_path', 'state_sha256', 'batch_snapshot_sha256', 'policy_sha256',
                    'post_forward_recurrent_sha256', 'history_sha256', 'cache_context'):
            if row[key] != first[key]:
                raise ValueError('Regression rows do not belong to one identical full state.')
    paths = {str(REFERENCE_MANIFEST): digest(REFERENCE_MANIFEST), str(evidence): digest(evidence),
        manifest['source_archive']['path']: manifest['source_archive']['sha256'],
        first['state_path']: first['state_sha256'],
        first['target_checkpoint']: first['target_file_sha256']}
    paths.update({snapshot['path']: snapshot['file_sha256'] for snapshot in first['environment_snapshots']})
    for path, sha in paths.items():
        if digest(path) != sha:
            raise ValueError(f'Regression artifact changed: {path}')
    return dict(manifest=str(REFERENCE_MANIFEST), evidence=str(evidence), artifacts=paths,
                logical_queries=29, physical_jobs=20, batch_width=24, selected_states=10,
                diagnostic_reuse_is_exact_cache_only=True, scientific_result=False)


def intervention_signature(rows):
    return json_digest([(sha, row['env']) for row in rows for sha in row['first_actions_sha256']])
