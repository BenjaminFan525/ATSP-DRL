"""Verify the retained frozen route without loading models or starting experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify(root: Path, manifest: Path) -> dict:
    data = json.loads(manifest.read_text())
    if data['protocol'] != 'hkbz_frozen_route_v1':
        raise ValueError('Unknown frozen route protocol')
    checked = 0
    for record in data['files']:
        path = root / record['path']
        if path.stat().st_size != record['size_bytes'] or digest(path) != record['sha256']:
            raise ValueError('Frozen file changed: ' + record['path'])
        checked += 1
    source_count = 0
    for source in data['source_manifests']:
        registration = json.loads((root / source['path']).read_text())
        for root_key, files_key in source['fields']:
            source_root = Path(registration[root_key])
            for relative, expected in registration[files_key].items():
                if digest(source_root / relative) != expected:
                    raise ValueError('Frozen source changed: ' + str(source_root / relative))
                source_count += 1
    handoff = json.loads((root / data['stage1']['handoff']).read_text())
    if handoff['winner'] != 'P5_team_time_potential_fixed':
        raise ValueError('Stage1 handoff no longer selects P5')
    selected = json.loads((root / data['stage3']['selection']).read_text())
    final = json.loads((root / data['stage3']['final_result']).read_text())
    if (selected['epoch'] != data['stage3']['selected_epoch']
            or selected['checkpoint']['sha256'] != data['stage3']['checkpoint_sha256']
            or not final['completed']
            or final['scientific_target_confirmed'] is not False):
        raise ValueError('Stage3 final selection or evidence boundary changed')
    return {'protocol': data['protocol'], 'passed': True, 'bound_files_verified': checked,
            'frozen_source_files_verified': source_count,
            'selected_epoch': selected['epoch'], 'scientific_target_confirmed': False,
            'scope': 'file identity and frozen selection; no training, rollout, or IGA search'}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=root / 'artifacts/frozen_route/20260920/manifest.json')
    args = parser.parse_args()
    print(json.dumps(verify(root, args.manifest), indent=2))


if __name__ == '__main__':
    main()
