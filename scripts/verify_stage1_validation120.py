"""Verify archived Stage1 evidence and recompute scores without running models."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    check(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7),
          f"Metric mismatch: {actual} != {expected}")


def verify(manifest):
    root = manifest.resolve().parent
    data = json.loads(manifest.read_text())
    check(data['schema_version'] == 1, 'Unknown archive version')
    bound = {}
    for record in data['files'] + data['source_archives']:
        name = record['path']
        path = (root / name).resolve()
        check(path.is_relative_to(root) and name not in bound, 'Invalid archive path')
        payload = path.read_bytes()
        check(len(payload) == record['size'] and
              hashlib.sha256(payload).hexdigest() == record['sha256'], name)
        bound[name] = record

    def read(name):
        check(name in bound, 'Unbound evidence: ' + name)
        return json.loads((root / name).read_text())

    source_count = 0
    for record in data['source_archives']:
        expected = {m['path']: m for m in record['members']}
        snapshot = read(record['path'].split('/')[0] + '/snapshot_manifest.json')['files']
        check({n for n in expected if n.startswith('code/')} ==
              {n for n in snapshot if n.startswith('code/')}, 'Incomplete frozen source')
        with tarfile.open(root / record['path']) as archive:
            members = archive.getmembers()
            check(len(members) == len(expected) and
                  {m.name for m in members} == set(expected), 'Source inventory mismatch')
            for member in members:
                check(member.isfile(), 'Unexpected non-file archive member')
                with archive.extractfile(member) as stream:
                    payload = stream.read()
                digest = hashlib.sha256(payload).hexdigest()
                check(len(payload) == expected[member.name]['size'] and
                      digest == expected[member.name]['sha256'], member.name)
                if member.name in snapshot:
                    check(digest == snapshot[member.name], 'Frozen source changed')
                source_count += 1

    learning = read('learning/manifest.json')
    observed = read('learning/analysis_observed/summary.json')
    state = read('learning/state.json')
    check(state['status'] == data['learning_status'] == 'needs_review', 'Review status changed')
    cases = {c['case']: c for c in learning['cases']}
    check(len(cases) == 120 and Counter(c['distribution'] for c in cases.values()) ==
          {'iid': 60, 'ood_stress': 54, 'ood_scale': 6}, 'Case panel changed')
    jobs = learning['jobs']
    check(len(jobs) == 15 and len({j['id'] for j in jobs}) == 15, 'Checkpoint count changed')
    method_means, method_diffs = defaultdict(list), Counter()
    all_rows, differences = {}, {}
    aligned = None
    for job in jobs:
        name = job['id']
        output = read(f'learning/results/{name}.json')
        audit = read(f'learning/audits/{name}.json')
        reference = read(f'learning/references/{name}_validation60.json')
        check(output['status'] == 'completed' and output['seed'] == job['seed'] and
              output['checkpoint_episode'] == job['best_epoch'] and
              output['model_sha256'] == job['model_sha256'], 'Checkpoint identity: ' + name)
        check(audit['result_sha256'] == bound[f'learning/results/{name}.json']['sha256'],
              'Audit result digest: ' + name)
        if job.get('reuse_result'):
            check(job['reuse_result_sha256'] == bound[f'learning/results/{name}.json']['sha256'] ==
                  bound[f'prior_attempts/r2/results/{name}.json']['sha256'], 'P5 reuse changed')
        evaluation = output['evaluation']
        check(evaluation['policy_history_mode'] == job['policy_history_mode'], 'History contract')
        rows = {r['case_dir']: r for r in evaluation['records']}
        check(len(evaluation['records']) == 120 and set(rows) == set(cases), 'Result panel')
        for case, row in rows.items():
            expected = cases[case]
            check(row['fingerprints']['case_sha256'] == expected['case_sha256'] and
                  row['fingerprints']['files'] == expected['files'] and
                  row['distribution'] == expected['distribution'] and
                  row['profile'] == expected['profile'], 'Case identity: ' + case)
            check(row['completed'] and not row['timeout'] and not row['cycle_terminated'],
                  'Incomplete case: ' + case)
            all_rows[(job['method'], str(job['seed']), case)] = row
        if not job.get('reuse_result'):
            order = learning['case_order']
            check(evaluation['worker_count'] == learning['workers'] and
                  evaluation['case_order_sha256'] == hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                  'Batch contract changed')
            width = len(order) // learning['workers']
            for i, case in enumerate(order):
                check((rows[case]['rank_index'], rows[case]['round_index']) == divmod(i, width),
                      'Case worker layout changed')
        old_cases = {r['case_dir'] for r in reference['cases']}
        check(len(reference['cases']) == len(old_cases) == 60, 'Reference case count')
        if aligned is None:
            aligned = old_cases
        check(aligned == old_cases, 'Different original validation60 panel')
        changed = []
        for old in reference['cases']:
            case = old['case_dir']
            check(old['fingerprints']['case_sha256'] == cases[case]['case_sha256'], 'Reference hash')
            if abs(old['makespan'] - rows[case]['makespan']) > 1e-6:
                changed.append({'case': case, 'old': old['makespan'], 'new': rows[case]['makespan']})
                differences[(name, case)] = (old['makespan'], rows[case]['makespan'])
        check(changed == audit['overlap60_differences'] and
              60 - len(changed) == audit['overlap60_exact_match_count'], 'Overlap audit: ' + name)
        mean = statistics.mean(r['makespan'] for r in rows.values())
        close(mean, audit['metrics']['mean'])
        close(mean, evaluation['raw_makespan'])
        method_means[job['method']].append(mean)
        method_diffs[job['method']] += len(changed)
    for method, means in method_means.items():
        summary = observed['methods'][method]
        check(len(means) == 3 and {j['seed'] for j in jobs if j['method'] == method} == {1, 2, 3},
              'Seed inventory')
        close(statistics.mean(means), summary['mean'])
        close(statistics.stdev(means), summary['seed_sd'])
        check(summary['old60_exact_match'] == 180 - method_diffs[method], 'Match count')
        check(summary['status'] == ('provisional_needs_review' if method_diffs[method] else 'verified'),
              'Method review status')
        for distribution in ('iid', 'ood_stress', 'ood_scale'):
            close(statistics.mean(r['makespan'] for (m, _, _), r in all_rows.items()
                                  if m == method and r['distribution'] == distribution),
                  summary['groups'][distribution])
    for name in ('per_case.csv', 'overlap_differences.csv'):
        with (root / 'learning/analysis_observed' / name).open(newline='') as stream:
            exported = list(csv.DictReader(stream))
        if name == 'per_case.csv':
            keys = {(r['method'], r['seed'], r['case']) for r in exported}
            check(len(exported) == len(keys) == 1800 and keys == set(all_rows), 'CSV case inventory')
            for row in exported:
                original = all_rows[(row['method'], row['seed'], row['case'])]
                close(float(row['makespan']), original['makespan'])
                check(row['case_sha256'] == original['fingerprints']['case_sha256'], 'CSV identity')
        else:
            actual = {(r['job'], r['case']): (float(r['old']), float(r['new'])) for r in exported}
            check(len(exported) == len(actual) == 34 and actual == differences, 'CSV drift inventory')
    check(len(all_rows) == data['learning_case_episodes'] == 1800 and
          len(differences) == data['baseline_changed_overlap_case_seeds'] == 34, 'Archive counts')
    iga_manifest = read('iga/full/manifest.json')
    check(set(iga_manifest['aligned60_cases']) == aligned, 'IGA validation60 alignment')
    phases = []
    for phase in ('iga180', 'iga1800'):
        summary = read(f'iga/full/{phase}/summary.json')
        check(summary['status'] == 'completed' and not summary['failures'], 'IGA completion')
        rows = {r['case']: r for r in summary['cases']}
        check(len(summary['cases']) == 120 and set(rows) == set(cases), 'IGA panel')
        for case, row in rows.items():
            check(row['case_sha256'] == cases[case]['case_sha256'] and
                  row['distribution'] == cases[case]['distribution'], 'IGA case identity')
        close(statistics.mean(r['makespan'] for r in rows.values()),
              summary['validation120']['mean_makespan'])
        close(statistics.mean(rows[c]['makespan'] for c in aligned),
              summary['aligned_validation60']['mean_makespan'])
        phases.append(rows)
    wins = sum(phases[1][c]['makespan'] < phases[0][c]['makespan'] for c in cases)
    ties = sum(phases[1][c]['makespan'] == phases[0][c]['makespan'] for c in cases)
    check((wins, ties) == (111, 9) and data['iga_case_results'] == 240, 'Nested IGA changed')
    return dict(passed=True, bound_files_verified=len(bound), source_files_verified=source_count,
                learning_case_episodes=len(all_rows), iga_case_results=240,
                changed_overlap_case_seeds=len(differences), learning_status='needs_review',
                scope='Archived hashes, case identities and score recomputation; no model execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] /
                        'artifacts/stage1_validation120/20260920/manifest.json')
    print(json.dumps(verify(parser.parse_args().manifest), indent=2))
