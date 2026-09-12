"""Resolve checksum-pinned, test-only cases without an external dataset tree."""

import hashlib
import json
from pathlib import Path


FIXTURE_ROOT = Path(__file__).resolve().parent / 'fixtures'


def case_dir(name):
    """Return an indexed synthetic TRAIN fixture, rejecting missing/changed data."""
    index = json.loads((FIXTURE_ROOT / 'provenance.json').read_text(encoding='utf-8'))
    if name not in index['cases']:
        raise ValueError(f'Unknown regression fixture: {name!r}')
    case = index['cases'][name]
    path = (FIXTURE_ROOT / case['directory']).resolve()
    if path.parent != FIXTURE_ROOT.resolve() or case['original_split'] != 'train':
        raise ValueError('Regression fixture must be a direct, indexed TRAIN case.')
    for record in case['files']:
        filename = record['filename']
        if Path(filename).name != filename:
            raise ValueError('Regression fixture file must not traverse directories.')
        source = path / filename
        content = source.read_bytes()
        if len(content) != record['size_bytes'] or hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError(f'Regression fixture checksum differs: {source}')
    return path
