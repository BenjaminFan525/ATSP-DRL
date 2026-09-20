"""Verify the portable Stage2 checkpoint archive without loading pickle or training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify(manifest: Path) -> dict:
    root = manifest.resolve().parent
    data = json.loads(manifest.read_text())
    if data['protocol'] != 'hkbz_stage2_checkpoint_archive_v1':
        raise ValueError('Unknown Stage2 archive protocol')
    records = {}
    for record in data['files']:
        relative = record['path']
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or relative in records:
            raise ValueError('Invalid or duplicate archive path: ' + relative)
        if path.stat().st_size != record['size_bytes'] or digest(path) != record['sha256']:
            raise ValueError('Archived file changed: ' + relative)
        records[relative] = record

    def read(relative: str) -> dict:
        if relative not in records:
            raise ValueError('Evidence is not hash-bound: ' + relative)
        return json.loads((root / relative).read_text())

    source = read('evidence/run/manifest.json')
    source_path = 'evidence/run/source_bundle.tar.gz'
    if records[source_path]['sha256'] != source['source_archive']['sha256']:
        raise ValueError('Source archive does not match the original manifest')
    with tarfile.open(root / source_path) as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        expected = source['code_fingerprint']
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('Source archive file inventory differs')
        for member in members:
            with archive.extractfile(member) as stream:
                actual = hashlib.sha256(stream.read()).hexdigest()
            if actual != expected[member.name]:
                raise ValueError('Source file changed: ' + member.name)

    status = read('evidence/run/suite_status.json')
    comparison = read('evidence/run/analysis/comparison.json')
    if (status['status'] != 'completed' or status['pending']
            or not comparison['complete'] or comparison['formal_promotion'] is not False
            or data['formal_promotion'] is not False or data['original_frozen_b0_replaced'] is not False):
        raise ValueError('Completion or evidence boundary changed')
    if len(status['jobs']) != 3 or any(
            j['status'] != 'completed' or j['exit_code'] != 0 for j in status['jobs'].values()):
        raise ValueError('All three trainers must have completed successfully')
    if sorted(s['stage1_seed'] for s in data['selected']) != [1, 2, 3]:
        raise ValueError('Expected exactly three Stage1 seeds')
    for selection in data['selected']:
        seed = selection['stage1_seed']
        method = comparison['methods'][f'H3F4_stage1seed{seed}_bcseed11']
        epoch = selection['selected_epoch']
        eligible = [e for e in method['epoch_metrics'] if e['gate']['passed']]
        winner = min(eligible, key=lambda e: (e['metrics']['eval_raw_makespan'], e['epoch']))
        checkpoint = records[selection['checkpoint']]
        last = records[selection['last_checkpoint']]
        if (not method['integrity_passed'] or not method['eligible']
                or epoch != method['selected_epoch'] or epoch != winner['epoch']
                or checkpoint['selected_bc_epoch'] != epoch or checkpoint['epoch'] != epoch
                or last['epoch'] != 2 or last['completed_epochs'] != 2
                or not math.isclose(selection['validation120_mean_seconds'],
                                    winner['metrics']['eval_raw_makespan'], abs_tol=1e-8)):
            raise ValueError(f'Selection differs for seed {seed}')
    checkpoints = sum(name.endswith('.pt') for name in records)
    if checkpoints != 12:
        raise ValueError('Expected twelve preserved checkpoints')
    return {'passed': True, 'bound_files_verified': len(records),
            'checkpoints_verified': checkpoints, 'source_files_verified': len(expected),
            'selected_epochs': [s['selected_epoch'] for s in data['selected']],
            'formal_promotion': False,
            'scope': 'file hashes, source archive and historical selection; no model execution'}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=root /
                        'artifacts/stage2_checkpoints/20260921_h3f4_validation120/manifest.json')
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest), indent=2))


if __name__ == '__main__':
    main()
