#!/usr/bin/env python3
"""GPU0 workers for baseline, admission, training and one persistent validator."""
from __future__ import annotations

import argparse
import copy
import gc
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from onpolicy.runner.shared.stage3_b_shared_b0_engine import BSharedB0Engine
from onpolicy.runner.shared.stage3_research_engine import seed_all
from onpolicy.utils.stage3_b_shared_b0 import (
    PROTOCOL, SOURCE_SHA, cpus, verify_manifest, check_resources, normalization_moments,
    record_summary, submit_evaluation, pending_evaluations, validate_request,
    read_commit, schedule, training_schedule, reconcile_epoch_requests, verify_admission, budget, episodes_at_cursor,
)
from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file, digest_json, Heartbeat
from onpolicy.utils.stage3_distributed import state_digest


class Progress(Heartbeat):
    def update(self, **values):
        if values:
            values['last_progress_unix'] = time.time()
        super().update(**values)
        if values and time.time() - getattr(self, 'last_print', 0) >= 30:
            self.last_print = time.time()
            print(f'[B_SHARED] {values}', flush=True)


def engine(m, *, training=True, width=None, pool=True):
    return BSharedB0Engine(m['frozen_manifest']['path'], config=m['recipe'], training=training,
        width=width or m['recipe']['rollout_workers'], create_pool=pool)


def read_baseline(m):
    data = read_json(Path(m['root']) / 'baselines.json')
    coverage = m['splits']
    if m.get('training_subset'):
        from onpolicy.utils.stage3_b0_subset import bound
        selection = bound(m['training_subset']['selection'])
        coverage = bound(selection['parent_manifest'])['splits']
    if (data['manifest_sha256'] != m['manifest_sha256'] or data['source_sha256'] != SOURCE_SHA
            or digest_json(data['costs']) != data['costs_sha256']
            or set(data['costs']) != {c['path'] for s in ('train', 'validation', 'tune') for c in coverage[s]}):
        raise ValueError('Baseline identity/coverage changed')
    return data


def evaluate_cases(m, runner, cases, output, hb, *, decoder='H', native=False, times=False,
                   binding=None, timeout=None):
    output = Path(output)
    bind = binding or dict(manifest_sha256=m['manifest_sha256'], source_sha256=SOURCE_SHA,
                           decoder=decoder, native=native)
    bind = {**bind, 'batch_contract':m['recipe']['evaluation_batch_contract'],
            'evaluation_cudnn_allow_tf32':m['recipe']['evaluation_cudnn_allow_tf32']}
    rows = {}
    for i, c in enumerate(cases):
        path = output / f'{i:04d}.json'
        if path.exists():
            row = read_json(path)
            if row['binding'] != bind or row['case_sha256'] != c['content_sha256'] or row['result']['case_id'] != c['path']:
                raise ValueError('Cached completed evaluation case changed identity')
            rows[i] = row
    started = time.monotonic()
    width = m['recipe']['validation_workers']
    if runner.width < width:
        raise ValueError('Native evaluation requires all twelve fixed slots')
    for start in range(0, len(cases), width):
        if timeout and time.monotonic() - started > timeout:
            raise TimeoutError('Evaluation request exceeded its fixed timeout')
        indices = list(range(start, min(start + width, len(cases))))
        if all(i in rows for i in indices):
            continue
        trajectories = runner.rollout([cases[i] for i in indices], [1] * len(indices), deterministic=True,
            decoder=decoder, native=native, record_times=times, heartbeat=hb)
        for i, t in zip(indices, trajectories):
            row = dict(binding=bind, case_sha256=cases[i]['content_sha256'], result=record_summary(t))
            if times:
                row['normalization_moments'] = normalization_moments([t])
            if i in rows:
                if rows[i] != row:
                    raise ValueError('Replaying an interrupted fixed batch changed a completed case')
            else:
                atomic_json(output / f'{i:04d}.json', row, overwrite=False)
            rows[i] = row
        hb.update(event='evaluation_progress', completed_cases=len(rows), total_cases=len(cases), decoder=decoder,
                  evaluation_seconds=time.monotonic() - started)
    return [rows[i] for i in range(len(cases))]


def baseline(m, hb):
    root = Path(m['root'])
    if (root / 'baselines.json').exists():
        read_baseline(m)
        return
    runner = engine(m, training=False)
    try:
        records, moments = {}, np.zeros(3, np.float64)
        for split in ('tune', 'validation', 'train'):
            hb.update(event='source_baseline', split=split)
            rows = evaluate_cases(m, runner, m['splits'][split], root / 'baseline_cases' / split, hb,
                                  native=True, times=split == 'train')
            records[split] = [r['result'] for r in rows]
            if split == 'train':
                moments += np.asarray([r['normalization_moments'] for r in rows]).sum(axis=0)
            if split == 'tune':
                historical = read_json(runner.bundle.path('hf_baseline'))
                old = {r['case_dir']: r for r in historical['cases']}
                for c, row in zip(m['splits']['tune'], records['tune']):
                    if (old[c['name']]['case_sha256'] != c['case_sha256']
                            or abs(row['makespan'] - old[c['name']]['makespan']) > 1e-6):
                        raise ValueError(f'Native frozen Tune60 replay differs: {c["name"]}')
        costs = {r['case_id']: r['makespan'] for rows in records.values() for r in rows}
        payload = dict(manifest_sha256=m['manifest_sha256'], source_sha256=SOURCE_SHA,
            source_decoder='native_hungarian', costs=costs, costs_sha256=digest_json(costs), records=records,
            normalization_moments=moments.tolist(), normalization_sha256=digest_json(moments.tolist()),
            completed_case_episodes=780, actor_updates=0, critic_updates=0, teacher_queries=0)
        atomic_json(root / 'baselines.json', payload, overwrite=False)
    finally:
        runner.close()


def zero_check(m, hb):
    root, source = Path(m['root']), read_baseline(m)
    runner = engine(m, training=False)
    try:
        checked = 0
        for split in ('tune', 'validation'):
            reference = {r['case_id']: r for r in source['records'][split]}
            rows = evaluate_cases(m, runner, m['splits'][split], root / 'zero_check_cases' / split, hb)
            for entry in rows:
                r, old = entry['result'], reference[entry['result']['case_id']]
                if (abs(r['makespan'] - old['makespan']) > 1e-6
                        or r['actions_sha256'] != old['actions_sha256']
                        or r['history_sha256'] != old['history_sha256']):
                    raise ValueError(f'B0 zero-update action/history/cost mismatch: {r["case_id"]}')
                checked += 1
        atomic_json(root / 'zero_update_verified.json', dict(manifest_sha256=m['manifest_sha256'],
            passed=True, complete_cases=checked, tensors=508, actor_updates=0, baseline_sha256=digest_file(root / 'baselines.json')),
            overwrite=False)
    finally:
        runner.close()


def collect(runner, cases, seeds, hb):
    group = []
    for start in range(0, len(cases), runner.width):
        group.extend(runner.rollout(cases[start:start + runner.width], seeds[start:start + runner.width],
                                    retain=True, heartbeat=hb))
    return group


def compare_states(expected, actual, *, exact=False, rtol=2e-5):
    """Compare real nested Adam tensors, counters and RNG; never empty fixtures."""
    if isinstance(expected, torch.Tensor):
        actual = actual.detach().cpu()
        expected = expected.detach().cpu()
        ok = torch.equal(expected, actual) if exact or not expected.is_floating_point() else torch.allclose(
            expected, actual, atol=2e-6, rtol=rtol)
        if not ok:
            raise AssertionError(f'Tensor differs: shape={expected.shape}, max={float((expected.double()-actual.double()).abs().max())}')
    elif isinstance(expected, np.ndarray):
        if not np.array_equal(expected, actual):
            raise AssertionError('NumPy RNG/state differs')
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError('State keys differ')
        for key in expected:
            compare_states(expected[key], actual[key], exact=exact, rtol=rtol)
    elif isinstance(expected, (tuple, list)):
        if len(expected) != len(actual):
            raise AssertionError('State lengths differ')
        for a, b in zip(expected, actual):
            compare_states(a, b, exact=exact, rtol=rtol)
    elif expected != actual:
        raise AssertionError(f'State scalar differs: {expected!r} != {actual!r}')


def canary(m, output, hb):
    """Real global32, micro8/micro4 equivalence and nonempty-Adam resume fixture."""
    base = read_baseline(m)
    runner = engine(m)
    try:
        runner.set_normalization(base['normalization_moments'], expected_sha=base['normalization_sha256'])
        cases = m['canary_cases'] * 4
        seeds = [m['recipe']['seed'] + 10000 + i for i in range(32)]
        initial = output / 'initial.pt'
        runner.save(initial, manifest_sha256=m['manifest_sha256'], next_batch=0, diagnostic=True)
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        group = collect(runner, cases, seeds, hb)
        rollout_seconds = time.monotonic() - begin
        atomic_json(output / 'rollouts.json', [record_summary(t) for t in group], overwrite=False)
        diagnostics = runner.gradient_diagnostic(group[:4], base['costs'], heartbeat=hb)
        start = time.monotonic()
        update8 = runner.update(group, base['costs'], microbatch=8, heartbeat=hb)
        update_seconds = time.monotonic() - start
        expected = output / 'micro8.pt'
        runner.save(expected, manifest_sha256=m['manifest_sha256'], next_batch=1, diagnostic=True)
        runner.resume(initial, manifest_sha256=m['manifest_sha256'], allow_diagnostic=True)
        update4 = runner.update(group, base['costs'], microbatch=4, heartbeat=hb)
        actual = output / 'micro4.pt'
        runner.save(actual, manifest_sha256=m['manifest_sha256'], next_batch=1, diagnostic=True)
        a = torch.load(expected, map_location='cpu', weights_only=False)
        b = torch.load(actual, map_location='cpu', weights_only=False)
        for key in ('model', 'actor_optim', 'critic_optim', 'value_normalizer', 'policy_updates'):
            compare_states(a[key], b[key], rtol=0 if key == 'model' else 2e-5)
        if not a['actor_optim']['state'] or not a['critic_optim']['state']:
            raise AssertionError('Resume canary must contain real Adam moments')
        # Check a complete bottom-to-top actor update, not just requires_grad flags.
        original = torch.load(initial, map_location='cpu', weights_only=False)['model']
        changed = [name for name in a['model'] if not torch.equal(original[name], a['model'][name])]
        for prefix in ('encoder.op_embedding.', 'encoder.convs.0.', 'encoder.convs.3.',
                       'actor.', 'device_actor.', 'transporter_actor.'):
            if not any(n.startswith(prefix) for n in changed):
                raise AssertionError(f'No demonstrated update for {prefix}')
        del a, b, original, group
        gc.collect()
        # Commit before the next collection; a separate process redoes this
        # exact collection/update from nonempty Adam and compares all RNG.
        runner.resume(expected, manifest_sha256=m['manifest_sha256'], allow_diagnostic=True)
        next_cases, next_seeds = m['canary_cases'][:4], [m['recipe']['seed'] + 20000 + i for i in range(4)]
        next_group = collect(runner, next_cases, next_seeds, hb)
        next_rows = [record_summary(t) for t in next_group]
        runner.update(next_group, base['costs'], heartbeat=hb)
        runner.save(output / 'expected_next.pt', manifest_sha256=m['manifest_sha256'], next_batch=2, diagnostic=True)
        atomic_json(output / 'result.json', dict(passed=True, manifest_sha256=m['manifest_sha256'],
            complete_global_visits=32, microbatch_comparison=[8, 4], update8=update8, update4=update4,
            gradient_diagnostic=diagnostics, changed_model_tensors=len(changed),
            source_checkpoint=str(expected), source_checkpoint_sha256=digest_file(expected),
            next_cases=next_cases, next_seeds=next_seeds, expected_next_rows=next_rows,
            expected_next_checkpoint=str(output / 'expected_next.pt'),
            rollout_seconds=rollout_seconds, update_seconds=update_seconds,
            batch_seconds=rollout_seconds + update_seconds,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()), overwrite=False)
    finally:
        runner.close()


def resume_check(m, output, hb, canary_dir):
    canary_dir = Path(canary_dir)
    proof = read_json(canary_dir / 'result.json')
    if proof['manifest_sha256'] != m['manifest_sha256'] or digest_file(proof['source_checkpoint']) != proof['source_checkpoint_sha256']:
        raise ValueError('Canary resume source changed')
    base = read_baseline(m)
    runner = engine(m)
    try:
        runner.resume(proof['source_checkpoint'], manifest_sha256=m['manifest_sha256'], allow_diagnostic=True)
        group = collect(runner, proof['next_cases'], proof['next_seeds'], hb)
        if [record_summary(t) for t in group] != proof['expected_next_rows']:
            raise AssertionError('Fresh-process resume changed next collection')
        runner.update(group, base['costs'], heartbeat=hb)
        path = output / 'actual_next.pt'
        runner.save(path, manifest_sha256=m['manifest_sha256'], next_batch=2, diagnostic=True)
        expected = torch.load(proof['expected_next_checkpoint'], map_location='cpu', weights_only=False)
        actual = torch.load(path, map_location='cpu', weights_only=False)
        for key in ('model', 'actor_optim', 'critic_optim', 'value_normalizer', 'policy_updates',
                    'rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda'):
            compare_states(expected[key], actual[key], exact=True)
        atomic_json(output / 'result.json', dict(passed=True, fresh_process=True,
            manifest_sha256=m['manifest_sha256'], next_collection_equal=True, all_training_state_exact=True), overwrite=False)
    finally:
        runner.close()


def train(m, output, hb, resume_commit=None):
    root, base = Path(m['root']), read_baseline(m)
    limits=budget(m)
    epoch_ends=limits['epoch_end_batches']
    verify_admission(m)
    runner = engine(m)
    cursor = 0
    try:
        if resume_commit:
            commit = read_commit(m, resume_commit)
            cursor = runner.resume(commit['checkpoint'], manifest_sha256=m['manifest_sha256'])
            if cursor != commit['next_batch'] or runner.policy_updates != commit['actor_updates']:
                raise ValueError('Resume cursor differs from global commit')
            if m.get('resume_amendment'):
                from onpolicy.utils.stage3_b0_restart import verify_restored_state
                restored = verify_restored_state(runner, commit['checkpoint'])
                atomic_json(output / 'resume_verified.json', dict(restored,
                    manifest_sha256=m['manifest_sha256'], checkpoint_sha256=commit['checkpoint_sha256'],
                    next_batch=cursor, training_episodes=commit['training_episodes'],
                    actor_updates=runner.policy_updates), overwrite=False)
                hb.update(event='checkpoint_restored', next_batch=cursor,
                    training_episodes=commit['training_episodes'], actor_updates=runner.policy_updates)
        else:
            if list((root / 'commits').glob('*.json')):
                raise ValueError('Existing commits require explicit resume')
            runner.set_normalization(base['normalization_moments'], expected_sha=base['normalization_sha256'])
        if runner.normalization_sha256 != base['normalization_sha256']:
            raise ValueError('Restored normalization differs from frozen Train600 statistics')
        plan = training_schedule(m)
        # Reconcile complete epoch commits even if publication was interrupted.
        reconcile_epoch_requests(m, cursor)
        for index in range(cursor, len(plan)):
            while len(pending_evaluations(root)) >= m['recipe']['queue_max_pending']:
                hb.update(event='validator_backpressure', next_batch=index, pending=len(pending_evaluations(root)))
                time.sleep(5)
            batch = plan[index]
            hb.update(event='group_start', batch=index + 1, training_episodes=episodes_at_cursor(m,index), actor_updates=runner.policy_updates)
            started = time.monotonic()
            group = collect(runner, batch['cases'], batch['seeds'], hb)
            if len(group) != len(batch['cases']):
                raise ValueError('Formal update must contain every scheduled complete visit')
            for t, visit in zip(group, batch['visit_ids']):
                t['visit_id'] = visit
            rollout_seconds = time.monotonic() - started
            atomic_json(output / 'rollouts' / f'batch_{index+1:04d}.json', [record_summary(t) for t in group], overwrite=False)
            start = time.monotonic()
            update = runner.update(group, base['costs'], heartbeat=hb)
            update_seconds = time.monotonic() - start
            if len(update['epochs']) != 2 or runner.policy_updates != 2 * (index + 1):
                raise ValueError('The complete two-update budget was not executed')
            checkpoint_start = time.monotonic()
            path = output / 'models' / f'batch_{index+1:04d}.pt'
            sha = runner.save(path, manifest_sha256=m['manifest_sha256'], next_batch=index + 1)
            update_path = output / 'updates' / f'batch_{index+1:04d}.json'
            atomic_json(update_path, dict(batch=index + 1, training_episodes=batch['training_episodes'],
                actor_updates=runner.policy_updates, data_epoch=batch['data_epoch'], update=update,
                rollout_seconds=rollout_seconds, update_seconds=update_seconds,
                checkpoint_seconds=time.monotonic() - checkpoint_start, seconds=time.monotonic() - started), overwrite=False)
            commit = dict(manifest_sha256=m['manifest_sha256'], schedule_sha256=m['schedule_sha256'],
                checkpoint=str(path), checkpoint_sha256=sha, update=str(update_path), update_sha256=digest_file(update_path),
                next_batch=index + 1, training_episodes=batch['training_episodes'], actor_updates=runner.policy_updates,
                committed_unix=time.time(), attempt=str(output))
            atomic_json(root / 'commits' / f'batch_{index+1:04d}.json', commit, overwrite=False)
            cursor = index + 1
            completed_epochs = sum(boundary <= cursor for boundary in epoch_ends)
            hb.update(event='group_committed', batch=cursor, training_episodes=batch['training_episodes'],
                      actor_updates=runner.policy_updates, completed_data_epochs=completed_epochs)
            if cursor in epoch_ends:
                for split in ('validation', 'tune'):
                    submit_evaluation(m, path, batch['training_episodes'], split)
                if completed_epochs in (4, 8):
                    submit_evaluation(m, path, batch['training_episodes'], 'validation', decoder='AR')
                    diagnostic_group = collect(runner, m['canary_cases'][:4],
                        [m['recipe']['seed'] + 30000 + i for i in range(4)], hb)
                    atomic_json(root / 'diagnostics' / f'epoch_{completed_epochs}.json',
                        dict(manifest_sha256=m['manifest_sha256'], checkpoint_sha256=sha,
                             epoch=completed_epochs, rows=[record_summary(t) for t in diagnostic_group],
                             **runner.gradient_diagnostic(diagnostic_group, base['costs'], heartbeat=hb)), overwrite=False)
                    del diagnostic_group
            del group
            gc.collect()
        atomic_json(root / 'training_completed.json', dict(completed=True, manifest_sha256=m['manifest_sha256'],
            training_episodes=limits['training_episodes'], actor_updates=limits['actor_updates'], data_epochs=8,
            last_commit=str(root / 'commits' / f'batch_{limits["global_batches"]:04d}.json')),
            overwrite=False)
    except BaseException:
        runner.save(output / 'uncommitted_failure.pt', manifest_sha256=m['manifest_sha256'], next_batch=cursor, diagnostic=True)
        raise
    finally:
        runner.close()


def validator(m, output, hb):
    root = Path(m['root'])
    runner = engine(m, training=False, width=m['recipe']['validation_workers'])
    loaded = None
    try:
        while not (root / 'validator/stop.json').exists():
            requests = pending_evaluations(root)
            if not requests:
                hb.update(event='validator_idle')
                time.sleep(5)
                continue
            # Creation order ensures a steady epoch queue; IDs are not priorities.
            path = min(requests, key=lambda p: p.stat().st_mtime_ns)
            request = read_json(path)
            validate_request(m, request)
            if request['checkpoint_sha256'] != loaded:
                runner.load_weights(request['checkpoint'], manifest_sha256=m['manifest_sha256'])
                loaded = request['checkpoint_sha256']
            hb.update(event='validation_request', request_id=request['request_id'])
            rows = evaluate_cases(m, runner, m['splits'][request['split']],
                root / 'validator/cases' / request['request_id'], hb, decoder=request['decoder'],
                binding={'request_sha256': request['request_sha256']}, timeout=m['recipe']['evaluation_timeout_seconds'])
            result = dict(request=request, rows=[r['result'] for r in rows], completed=True, finished_unix=time.time())
            atomic_json(root / 'validator/results' / path.name, result, overwrite=False)
    finally:
        runner.close()


def finish_diagnostics(m, hb):
    """Reconcile an interruption between an epoch commit and its diagnostics."""
    root, base = Path(m['root']), read_baseline(m)
    for epoch in (4, 8):
        commit = read_commit(m, root / 'commits' / f'batch_{budget(m)["epoch_end_batches"][epoch-1]:04d}.json')
        path = root / 'diagnostics' / f'epoch_{epoch}.json'
        if path.exists():
            proof = read_json(path)
            if (proof['manifest_sha256'] != m['manifest_sha256']
                    or proof['checkpoint_sha256'] != commit['checkpoint_sha256']):
                raise ValueError('Epoch diagnostic identity changed')
            continue
        runner = engine(m)
        try:
            runner.load_weights(commit['checkpoint'], manifest_sha256=m['manifest_sha256'])
            group = collect(runner, m['canary_cases'][:4], [m['recipe']['seed']+30000+i for i in range(4)], hb)
            atomic_json(path, dict(manifest_sha256=m['manifest_sha256'],
                checkpoint_sha256=commit['checkpoint_sha256'], epoch=epoch,
                rows=[record_summary(t) for t in group],
                **runner.gradient_diagnostic(group, base['costs'], heartbeat=hb)), overwrite=False)
        finally:
            runner.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('--phase', required=True, choices=('baseline', 'zero', 'canary', 'resume-check', 'train', 'validator', 'diagnostics'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume-commit', type=Path)
    p.add_argument('--canary-dir', type=Path)
    args = p.parse_args()
    m = read_json(args.manifest)
    role = 'validator' if args.phase == 'validator' else 'trainer'
    os.sched_setaffinity(0, cpus(m['recipe'][role + '_cpus']))
    verify_manifest(m)
    check_resources(m, role)
    if (args.output / 'status.json').exists():
        raise FileExistsError('Each worker launch needs a new attempt output directory')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    seed_all(m['recipe']['seed'])
    with Progress(args.output / 'status.json', phase=args.phase, protocol=PROTOCOL,
                  visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), cpus=sorted(os.sched_getaffinity(0))) as hb:
        if args.phase == 'baseline': baseline(m, hb)
        elif args.phase == 'zero': zero_check(m, hb)
        elif args.phase == 'canary': canary(m, args.output, hb)
        elif args.phase == 'resume-check': resume_check(m, args.output, hb, args.canary_dir)
        elif args.phase == 'train': train(m, args.output, hb, args.resume_commit)
        elif args.phase == 'diagnostics': finish_diagnostics(m, hb)
        else: validator(m, args.output, hb)


if __name__ == '__main__':
    main()
