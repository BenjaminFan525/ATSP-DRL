#!/usr/bin/env python3
"""Standalone input adapter: imports ONLY the admitted frozen evaluation source."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def checked(record):
    if sha(record['path']) != record['sha256']:
        raise ValueError('Frozen input changed: '+record['path'])
    return Path(record['path'])


def run(manifest, request_path):
    m, q = read(manifest), read(request_path)
    if (identity({k:v for k,v in m.items() if k != 'manifest_sha256'}) != m['manifest_sha256']
            or q['manifest_sha256'] != m['manifest_sha256']
            or identity({k:v for k,v in q.items() if k != 'request_sha256'}) != q['request_sha256']):
        raise ValueError('Evaluation request/suite identity changed')
    source = Path(m['evaluation_source_root'])
    for name, digest in m['evaluation_source_files'].items():
        if sha(source/name) != digest:
            raise ValueError('Frozen evaluation source changed: '+name)
    sys.path.insert(0, str(source))
    import torch
    from onpolicy.runner.shared.stage3_h3_frozen_engine import H3FrozenEngine, model_digest
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import Progress
    from onpolicy.utils.stage3_b0_single_model import CpuEnvironmentPool
    from onpolicy.utils.stage3_b_shared_b0 import cpus, record_summary
    from onpolicy.utils.stage3_canonical_h import enable_canonical_h
    from onpolicy.utils.stage3_numerics import configure_runtime
    from onpolicy.utils.stage3_research import atomic_json
    # Prove Python did not pick up the mutable training implementation.
    import onpolicy.runner.shared.stage3_h3_frozen_engine as module
    if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError('Evaluation imported the training workspace')
    allocation = m['resources']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != allocation['gpu_uuid']:
        raise ValueError('Evaluation escaped the registered GPU')
    if not set(os.sched_getaffinity(0)).issubset(cpus(allocation['cpus'])):
        raise ValueError('Evaluation escaped the CPU half')
    for package, version in m['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Evaluation dependency changed: '+package)
    configure_runtime()
    if (torch.cuda.device_count() != 1 or
            str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') != allocation['gpu_uuid'].removeprefix('GPU-')):
        raise ValueError('Unexpected visible CUDA identity')
    cases = q['cases']
    if not cases or len(cases) % 12 or len({c['path'] for c in cases}) != len(cases):
        raise ValueError('Canonical evaluations require complete unique fixed-12 groups')
    for case in cases:
        for filename, digest in case['files'].items():
            if sha(Path(case['path'])/filename) != digest:
                raise ValueError('Case content changed: '+case['path'])
    parent = read(checked(m['parent_manifest']))
    recipe = copy.deepcopy(parent['recipe'])
    recipe['gpu_headroom_mib'] = torch.cuda.get_device_properties(0).total_memory/2**20 - 4096 - 1024
    root = Path(q['output']); root.mkdir(parents=True, exist_ok=True)
    expected = {r['case_id']:r for r in q.get('expected', [])}
    checkpoint = checked(q['checkpoint'])
    started = time.monotonic()
    with Progress(root/'status.json', phase='evaluation', label=q['label'],
                  manifest_sha256=m['manifest_sha256']) as hb:
        runner = H3FrozenEngine(parent['frozen_manifest']['path'], config=recipe, training=False,
            runtime=dict(sampling_cpus=sorted(cpus(allocation['validator_cpus'])),
                         sampling_output=str(root/'unused_sampling')))
        try:
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            runner.policy.ac.load_state_dict(payload['model'], strict=True)
            runner.policy_updates = payload.get('policy_updates', 0)
            del payload
            before = model_digest(runner.policy.ac)
            if q['decoder'] == 'canonical':
                atomic_json(root/'canonical_contract.json', enable_canonical_h(runner.policy))
            elif q['decoder'] != 'legacy':
                raise ValueError('Unregistered evaluator')
            runner.pool = CpuEnvironmentPool(12, timeout=recipe['ipc_timeout_seconds'],
                                           affinity=cpus(allocation['validator_cpus']))
            records = []
            for start in range(0, len(cases), 12):
                path = root/'groups'/f'batch_{start//12:02d}.json'
                group_cases = cases[start:start+12]
                if path.exists():
                    group = read(path)
                    if group['request_sha256'] != q['request_sha256'] or not group['weights_unchanged']:
                        raise ValueError('Foreign cached evaluation group')
                else:
                    hb.update(batch=start//12, completed_cases=len(records), total_cases=len(cases))
                    runner.cache.last_actor = None
                    began = time.monotonic()
                    rows = runner.rollout(group_cases, [1]*12, deterministic=True,
                                          native=False, decoder='H', heartbeat=hb)
                    group = dict(rows=[record_summary(r) for r in rows],
                        request_sha256=q['request_sha256'], seconds=time.monotonic()-began,
                        weights_unchanged=model_digest(runner.policy.ac) == before,
                        batch=start//12, gpu_uuid=allocation['gpu_uuid'])
                    if not group['weights_unchanged']:
                        raise ValueError('Evaluation mutated model weights')
                    for row in rows:
                        trace = root/'trajectories'/f'{Path(row["case_id"]).name}.json.gz'
                        trace.parent.mkdir(exist_ok=True)
                        with gzip.open(trace, 'wt') as f:
                            json.dump(dict(**record_summary(row), actions=row['actions'],
                                           request_sha256=q['request_sha256']), f)
                    atomic_json(path, group, overwrite=False)
                if [r['case_id'] for r in group['rows']] != [c['path'] for c in group_cases]:
                    raise ValueError('Evaluation ordering changed')
                for row in group['rows']:
                    if (not row['completed'] or row.get('cycle_terminated') or row['tau'] != .3
                            or not row['behavior_deterministic'] or row.get('forced_replay')):
                        raise ValueError('Invalid complete trajectory')
                    if expected:
                        for key in ('makespan', 'steps', 'actions_sha256', 'history_sha256'):
                            if row[key] != expected[row['case_id']][key]:
                                atomic_json(root/'reproduction_failure.json', dict(case=row['case_id'],
                                    field=key, actual=row[key], expected=expected[row['case_id']][key]))
                                raise ValueError('Full-trajectory admission mismatch: '+key)
                records.extend(group['rows'])
            atomic_json(root/'result.json', dict(completed=True, rows=records,
                checkpoint=q['checkpoint'], decoder=q['decoder'], label=q['label'],
                request_sha256=q['request_sha256'], manifest_sha256=m['manifest_sha256'],
                evaluation_source_sha256=identity(m['evaluation_source_files']),
                seconds=time.monotonic()-started, weights_unchanged=True,
                reproduction_passed=bool(expected), solver_queries=0), overwrite=False)
            hb.update(phase='completed', completed_cases=len(records))
        finally:
            runner.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path); parser.add_argument('request', type=Path)
    args = parser.parse_args()
    run(args.manifest, args.request)
