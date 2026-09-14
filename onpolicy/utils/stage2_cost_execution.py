"""Exact, bounded fanout for frozen-student counterfactual continuations.

GPU calls retain the original batch width, order, graphs and both GRUs. Only
environment RPCs are batched. Within an environment worker, branches sharing
the *same object history and exact action bytes* share a transition; different
actions fork before mutation. This is not a teacher/CPU policy approximation.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from onpolicy.utils.stage2_cost_timeout import (
    HardProgressDeadline, LEGACY_TIMEOUT_CONTRACT, TIMEOUT_CONTRACT, shared_wall_charges,
)

EXECUTION_CONTRACT = 'exact_full_batch_fanout_shared_environment_prefix_v1'
CACHE_CONTRACT = 'source_policy_batch_both_grus_history_first_action_target_v1'


def array_digest(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class ShadowFanout:
    """Private worker-side states; no process pool or live-environment mutation."""

    def __init__(self, snapshot, job_ids):
        import cloudpickle
        if not job_ids or len(set(job_ids)) != len(job_ids):
            raise ValueError('Fanout requires distinct nonempty job IDs.')
        root = cloudpickle.loads(snapshot)
        self.heads = dict.fromkeys(job_ids, root)

    def step(self, requests):
        ids = [request['job'] for request in requests]
        if len(set(ids)) != len(ids) or not set(ids).issubset(self.heads):
            raise ValueError('Duplicate or unknown fanout job.')
        # Finished branches release their private states on the next RPC.
        self.heads = {key: self.heads[key] for key in ids}
        grouped = {}
        for request in requests:
            action = np.asarray(request['action'])
            if action.dtype != np.int64:
                raise ValueError('Do not silently cast candidate action arithmetic.')
            key = (id(self.heads[request['job']]), array_digest(action))
            grouped.setdefault(key, []).append(request)
        by_head = {}
        for (identity, action_key), items in grouped.items():
            by_head.setdefault(identity, []).append((action_key, items))
        outputs, refs, forks = [], {}, 0
        for groups in by_head.values():
            original = self.heads[groups[0][1][0]['job']]
            # Make all divergent copies BEFORE mutating the original.
            states = [original] + [copy.deepcopy(original) for _ in groups[1:]]
            forks += len(states) - 1
            for (_, items), state in zip(groups, states):
                obs, reward, dones, info = state.step(items[0]['action'])
                status = {'completed': bool(np.all(dones) and state._is_schedule_complete()
                          and not getattr(state, 'cycle_terminated', False)),
                          'makespan': float(state.total_time)}
                index = len(outputs)
                outputs.append((obs, reward, dones, info, status))
                for request in items:
                    self.heads[request['job']] = state
                    refs[request['job']] = index
        return {'outputs': outputs, 'refs': refs, 'requested_steps': len(requests),
                'executed_steps': len(outputs), 'forks': forks}


def semantic_outcome(outcome):
    """Only actual schedule results; runtime/cache fields are not scientific data."""
    return {key: outcome[key] for key in ('completed', 'makespan', 'steps', 'reason')}


class ExactCostCache:
    """Immutable, content-checked completed outcomes. Incompletes are never cached.

    Keys include all 24 companion states and the complete first action, not just
    the target case. No unversioned or teacher-generated evidence is accepted.
    Concurrent producers may compute a miss twice, but cannot overwrite a
    different answer; publication is atomic and disagreement fails closed.
    """

    def __init__(self, directory, source_namespace):
        if len(source_namespace) != 64 or any(c not in '0123456789abcdef' for c in source_namespace):
            raise ValueError('Cache needs a SHA256 source/runtime namespace.')
        self.directory = Path(directory) / source_namespace
        self.directory.mkdir(parents=True, exist_ok=True)
        self.namespace = source_namespace

    def binding(self, context, first_actions, target):
        required = {'policy_sha256', 'batch_snapshot_sha256', 'post_forward_recurrent_sha256',
                    'history_sha256', 'batch_width', 'max_steps', 'arithmetic'}
        if (not required.issubset(context)
                or len(context['batch_snapshot_sha256']) != context['batch_width']
                or np.asarray(first_actions).shape[0] != context['batch_width']
                or not 0 <= int(target) < context['batch_width']):
            raise ValueError('Incomplete full-state counterfactual cache binding.')
        return dict(contract=CACHE_CONTRACT, source_namespace=self.namespace,
                    context=context, first_actions_sha256=array_digest(first_actions), target=int(target))

    def get(self, binding):
        path = self.directory / (json_digest(binding) + '.json')
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        body = {'binding': payload['binding'], 'outcome': payload['outcome']}
        if (body['binding'] != binding or payload['sha256'] != json_digest(body)
                or not body['outcome']['completed'] or body['outcome']['reason'] != 'completed'
                or not math.isfinite(body['outcome']['makespan']) or body['outcome']['makespan'] <= 0
                or not 1 <= body['outcome']['steps'] <= binding['context']['max_steps']):
            raise ValueError(f'Invalid or changed cost cache evidence: {path}')
        return dict(body['outcome'], wall_seconds=0., cache_hit=True,
                    cache_path=str(path), cache_sha256=payload['sha256'])

    def put(self, binding, outcome):
        if not outcome['completed']:
            return
        if (outcome['reason'] != 'completed' or not math.isfinite(outcome['makespan'])
                or outcome['makespan'] <= 0
                or not 1 <= outcome['steps'] <= binding['context']['max_steps']):
            raise ValueError('Cannot publish invalid completed cost evidence.')
        body = {'binding': binding, 'outcome': semantic_outcome(outcome)}
        payload = dict(body, sha256=json_digest(body))
        path = self.directory / (json_digest(binding) + '.json')
        descriptor, temporary = tempfile.mkstemp(prefix='.publish-', dir=self.directory)
        try:
            with os.fdopen(descriptor, 'w') as handle:
                json.dump(payload, handle, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)  # Never replace a concurrent writer.
            except FileExistsError:
                existing = self.get(binding)
                if semantic_outcome(existing) != semantic_outcome(outcome):
                    raise ValueError('Exact cost cache producers disagree.')
            self.get(binding)  # Apply the same validity checks to new writes.
        finally:
            os.unlink(temporary)


def coalesce_queries(queries):
    """Identical full-batch interventions can serve multiple target cases."""
    jobs = []
    positions = {}
    for index, (actions, target) in enumerate(queries):
        key = array_digest(actions)
        if key not in positions:
            positions[key] = len(jobs)
            jobs.append({'first_actions': np.asarray(actions).copy(), 'consumers': []})
        jobs[positions[key]]['consumers'].append((index, int(target)))
    return jobs


def run_fanout(iterator, queries, *, width, context, cache=None, observer=None):
    """Run bounded independent continuations with unchanged full-batch GPU calls.

    Logical query count includes cache hits and duplicate consumers. Actual
    elapsed time is measured once around the executor, never summed across
    lockstep branches. Legacy timing remains available for explicit reference
    replay. The opt-in v2 clock allocates shared wall work without multiplying
    it by fanout/actor concurrency; a separate hard watchdog detects no progress.
    """
    arithmetic = context.get('arithmetic')
    contract = (arithmetic.get('timeout_contract', LEGACY_TIMEOUT_CONTRACT)
                if isinstance(arithmetic, dict) else LEGACY_TIMEOUT_CONTRACT)
    if contract not in (LEGACY_TIMEOUT_CONTRACT, TIMEOUT_CONTRACT):
        raise ValueError('Unknown cost timeout contract.')
    shared = contract == TIMEOUT_CONTRACT
    guard = (HardProgressDeadline(iterator.args.stage2_cost_branch_timeout_seconds)
             if shared else nullcontext())
    with guard as watchdog:
        return _run_fanout(iterator, queries, width=width, context=context, cache=cache,
                           observer=observer, shared=shared, watchdog=watchdog)


def _run_fanout(iterator, queries, *, width, context, cache, observer, shared, watchdog):
    if not 1 <= width <= 16:
        raise ValueError('Fanout width must be in [1, 16].')
    from onpolicy.utils.stage2_cost_improvement import actor_forward
    started = time.monotonic()
    last_progress = started
    results = [None] * len(queries)
    misses, miss_indices, bindings = [], [], {}
    stats = dict(logical_queries=len(queries), cache_hits=0, physical_jobs=0,
                 coalesced_queries=0, vector_steps=0, requested_environment_steps=0,
                 executed_environment_steps=0, environment_forks=0,
                 environment_rpc_seconds=0., actor_seconds=0., wall_seconds=0.)
    if shared:
        stats.update(actor_pool_wall_seconds=0., charged_work_seconds=0.,
                     peak_live_jobs=0, legacy_timeout_avoided_queries=0)
    for index, (actions, target) in enumerate(queries):
        if cache is not None:
            bindings[index] = cache.binding(context, actions, target)
            results[index] = cache.get(bindings[index])
        if results[index] is None:
            misses.append((actions, target))
            miss_indices.append(index)
        else:
            stats['cache_hits'] += 1
    jobs = coalesce_queries(misses)
    stats['physical_jobs'] = len(jobs)
    stats['coalesced_queries'] = len(misses) - len(jobs)
    envs, runner = iterator.runner.envs, iterator.runner
    maximum = runner.episode_length
    timeout = iterator.args.stage2_cost_branch_timeout_seconds
    for offset in range(0, len(jobs), width):
        chunk = jobs[offset:offset + width]
        ids = list(range(len(chunk)))
        envs.call('stage2_cost_fanout_restore', ids)
        if watchdog:
            watchdog.touch()
        live = {}
        for job_id, job in enumerate(chunk):
            live[job_id] = dict(job, actions=job['first_actions'].copy(),
                               hidden=iterator.next_rnn.copy(), compute_seconds=0.,
                               legacy_seconds=0., started=time.monotonic(), steps=0)
        for step in range(1, maximum + 1):
            if not live:
                break
            active_count = len(live)
            if shared:
                stats['peak_live_jobs'] = max(stats['peak_live_jobs'], active_count)
            worker_args = [([{'job': key, 'action': job['actions'][env]}
                             for key, job in live.items()],) for env in range(context['batch_width'])]
            rpc_started = time.monotonic()
            workers = envs.call_each('stage2_cost_fanout_step', worker_args)
            rpc_seconds = time.monotonic() - rpc_started
            if watchdog:
                watchdog.touch()
            stats['environment_rpc_seconds'] += rpc_seconds
            if shared:
                stats['charged_work_seconds'] += rpc_seconds
            for payload in workers:
                stats['requested_environment_steps'] += payload['requested_steps']
                stats['executed_environment_steps'] += payload['executed_steps']
                stats['environment_forks'] += payload['forks']
            forward_requests = []
            for key, job in list(live.items()):
                job['legacy_seconds'] += rpc_seconds
                job['compute_seconds'] += rpc_seconds / active_count if shared else rpc_seconds
                job['steps'] = step
                rows = [payload['outputs'][payload['refs'][key]] for payload in workers]
                graphs, _, done_rows, info_rows, statuses = zip(*rows)
                dones = np.stack(done_rows).astype(bool)
                job['hidden'][dones] = 0.
                pending = []
                for local_index, target in job['consumers']:
                    done = bool(np.all(dones[target]))
                    if done or step == maximum or job['compute_seconds'] > timeout:
                        status = statuses[target]
                        completed = bool(done and status['completed'])
                        outcome = dict(completed=completed,
                            makespan=status['makespan'] if completed else None,
                            steps=step, reason=('completed' if completed else 'incomplete_terminal')
                            if done else ('step_limit' if step == maximum else 'wall_timeout'),
                            wall_seconds=job['compute_seconds'], cache_hit=False,
                            execution=EXECUTION_CONTRACT)
                        if shared:
                            outcome.update(timeout_contract=TIMEOUT_CONTRACT,
                                charged_work_seconds=job['compute_seconds'],
                                legacy_charged_seconds=job['legacy_seconds'],
                                elapsed_wall_seconds=time.monotonic() - job['started'])
                            stats['legacy_timeout_avoided_queries'] += int(
                                completed and job['legacy_seconds'] > timeout)
                        original_index = miss_indices[local_index]
                        results[original_index] = outcome
                        if cache is not None:
                            cache.put(bindings[original_index], outcome)
                    else:
                        pending.append((local_index, target))
                job['consumers'] = pending
                stats['vector_steps'] += 1
                if observer is not None:
                    observer(offset + key, step, job['actions'], job['hidden'], rows)
                if not pending:
                    del live[key]
                    continue
                infos = envs.stack_infos(info_rows)
                active = runner._active_masks_from_info(infos)
                history = runner._authoritative_policy_history(infos, job['actions'],
                                                               runner.policy.ac.max_plane_agents)
                forward_requests.append((key, graphs, job['hidden'], active, history, infos.get('agent_types')))
            actor_pool = getattr(iterator, 'actor_pool', None)
            actor_started = time.monotonic()
            if actor_pool is not None:
                forwarded = actor_pool.run(forward_requests)
            else:
                forwarded = []
                for key, graphs, hidden, active, history, kinds in forward_requests:
                    forward_started = time.monotonic()
                    actions, hidden = actor_forward(iterator.policy, graphs, hidden, active, history, kinds)
                    forwarded.append((key, actions, hidden, time.monotonic() - forward_started))
            actor_wall = time.monotonic() - actor_started
            if watchdog:
                watchdog.touch()
            if shared:
                expected_keys = {request[0] for request in forward_requests}
                if (len(forwarded) != len(expected_keys)
                        or {row[0] for row in forwarded} != expected_keys):
                    raise ValueError('Actor pool returned missing or duplicate jobs.')
                charges = shared_wall_charges(actor_wall, [row[3] for row in forwarded])
                # Empty actor batches have no job to charge; their negligible
                # dispatch overhead remains in total executor wall_seconds.
                if forwarded:
                    stats['actor_pool_wall_seconds'] += actor_wall
                    stats['charged_work_seconds'] += actor_wall
            else:
                charges = [row[3] for row in forwarded]
            for (key, actions, hidden, seconds), charge in zip(forwarded, charges):
                job = live[key]
                job['actions'], job['hidden'] = actions, hidden
                stats['actor_seconds'] += seconds
                job['legacy_seconds'] += seconds
                job['compute_seconds'] += charge
            if step % 100 == 0 or time.monotonic() - last_progress >= 20:
                last_progress = time.monotonic()
                print(f'[StudentFanout] chunk={offset // width + 1} active={len(live)} '
                      f'step={step} elapsed={time.monotonic()-started:.1f}s', flush=True)
                callback = getattr(iterator, 'report_execution_progress', None)
                if callback is not None:
                    callback(dict(stats, current_chunk=offset // width + 1, current_step=step,
                        finished_queries=sum(row is not None for row in results),
                        elapsed_seconds=time.monotonic()-started))
        if live:
            raise RuntimeError('Fanout ended without a terminal result for every query.')
    if any(result is None for result in results):
        raise RuntimeError('Fanout lost a logical query.')
    stats['wall_seconds'] = time.monotonic() - started
    return results, stats
