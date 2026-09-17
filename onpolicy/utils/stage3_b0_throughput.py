"""Opt-in B0 execution adapter with measured independent sampling lanes.

The immutable study source supplies all policy/environment/optimizer math.
Every visit still makes batch-one stochastic forwards with its own seed.
Only scheduling, immutable input caching and scalar-transfer timing change.
"""
from contextlib import contextmanager
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import time
import traceback

import numpy as np
import torch

from onpolicy.runner.shared.stage3_b_shared_b0_engine import BSharedB0Engine, PPOContractError
from onpolicy.utils.stage3_performance import GroupExecutionCache, DeferredScalars
from onpolicy.utils.stage3_research import atomic_json, digest_json

VERSION = 'b0_throughput_v2'


def lane_cpus(allowed, lanes):
    """Keep SMT siblings together and never leave the caller's CPU allocation."""
    allowed = set(allowed)
    physical = sorted(c for c in allowed if c < 64)
    if not 1 <= lanes <= len(physical) or allowed != set(physical) | {c + 64 for c in physical}:
        raise ValueError('Sampling lanes need complete local SMT core pairs')
    return [set(map(int, row)) | {int(c) + 64 for c in row}
            for row in np.array_split(physical, lanes)]


def capture_rng():
    return dict(rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(0))


def restore_rng(state):
    random.setstate(state['rng_python'])
    np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch'])
    torch.cuda.set_rng_state(state['rng_cuda'], 0)


class CapacityWindowComplete(Exception):
    pass


def terminal_rng_index(rows, original_width=16):
    """Original collector ends on the last longest visit of its last group."""
    start = ((len(rows) - 1) // original_width) * original_width
    return max(range(start, len(rows)), key=lambda i: (rows[i]['steps'], i))


class LaneProgress:
    def __init__(self, path, *, stop_step=None):
        self.path, self.last = Path(path), 0.
        self.stop_step=stop_step
        self.started=time.monotonic()
        self.measured_start=None
        self.events=0
        self.step=0

    def update(self, **values):
        if 'rollout_step' in values:
            current=int(values['rollout_step'])
            live=int(values['live_trajectories'])
            if self.measured_start is None:
                self.measured_start=time.monotonic()
            else:
                self.events+=(current-self.step)*live
            self.step=current
            if self.stop_step and current>=self.stop_step:
                raise CapacityWindowComplete()
        if time.monotonic() - self.last >= 10:
            self.last = time.monotonic()
            atomic_json(self.path, dict(unix=time.time(), pid=os.getpid(), **values))


def collect_with_rng(runner, cases, seeds, heartbeat):
    """Observe each original batch-one forward without changing its math/RNG."""
    live, cursor, final_cuda = [], 0, {}
    original_call, original_get = runner.pool.call, runner.policy.get_actions
    def observed_call(commands):
        nonlocal live, cursor
        result = original_call(commands)
        if commands and commands[0][1] == 'reset':
            live, cursor = [row[0] for row in commands], 0
        elif commands and commands[0][1] == 'step':
            live = [i for i, _, _ in commands if not np.asarray(result[i][2]).all()]
            cursor = 0
        return result
    def observed_get(*args, **kwargs):
        nonlocal cursor
        result = original_get(*args, **kwargs)
        final_cuda[live[cursor]] = torch.cuda.get_rng_state(0)
        cursor += 1
        return result
    runner.pool.call, runner.policy.get_actions = observed_call, observed_get
    try:
        rows = runner.rollout(cases, seeds, retain=True, heartbeat=heartbeat)
        ending = capture_rng()
        return rows, [dict(ending, rng_cuda=final_cuda[i]) for i in range(len(rows))]
    finally:
        runner.pool.call, runner.policy.get_actions = original_call, original_get


def _sampling_lane(connection, frozen_manifest, config, affinity, width):
    """Each lane processes visits independently; no model updates occur here."""
    os.sched_setaffinity(0, affinity)
    runner = None
    try:
        runner = BSharedB0Engine(frozen_manifest, config=config, width=width, training=True)
        connection.send_bytes(pickle.dumps(dict(ready=True, pid=os.getpid()), protocol=5))
        while True:
            request = pickle.loads(connection.recv_bytes())
            if request['command'] == 'close':
                break
            started = time.monotonic()
            runner.policy.ac.load_state_dict(request['model'], strict=True)
            runner.policy_updates = request['policy_updates']
            runner.assert_frozen()
            output = Path(request['output'])
            progress = LaneProgress(output.with_suffix('.progress.json'),stop_step=request.get('probe_steps'))
            # All assigned environments are active; every policy forward keeps
            # the original batch-one shape and the visit's own RNG state.
            try:
                rows, rng_states = collect_with_rng(runner, request['cases'], request['seeds'], progress)
            except CapacityWindowComplete:
                connection.send_bytes(pickle.dumps(dict(capacity_window=True,
                    indices=request['indices'],policy_updates=runner.policy_updates,
                    measured_seconds=time.monotonic()-progress.measured_start,
                    events=progress.events,steps=progress.step,seconds=time.monotonic()-started,
                    peak_allocated=torch.cuda.max_memory_allocated(),
                    peak_reserved=torch.cuda.max_memory_reserved()),protocol=5))
                torch.cuda.empty_cache()
                del request
                continue
            temporary = output.with_suffix('.tmp')
            with temporary.open('wb') as file:
                pickle.dump(dict(rows=rows, rng_states=rng_states), file, protocol=5)
            temporary.replace(output)
            torch.cuda.empty_cache()
            connection.send_bytes(pickle.dumps(dict(completed=True, output=str(output),
                indices=request['indices'], seconds=time.monotonic()-started, pid=os.getpid(),
                policy_updates=runner.policy_updates), protocol=5))
            del rows, rng_states, request
    except BaseException:
        try:
            connection.send_bytes(pickle.dumps(dict(error=traceback.format_exc()), protocol=5))
        except (OSError, BrokenPipeError):
            pass
    finally:
        if runner is not None:
            runner.close()
        connection.close()


class SamplingLanes:
    def __init__(self, frozen_manifest, config, output, *, lanes=16, affinity=None, environment_count=32):
        if lanes not in (1, 2, 4, 8, 16,32):
            raise ValueError('Use a measured power-of-two sampling lane count up to 32')
        if environment_count not in (1,2,4,8,16,32,64,128,256,512) or lanes>environment_count:
            raise ValueError('Use a power-of-two environment capacity up to 512, at least one per lane')
        self.output, self.config = Path(output), config
        self.output.mkdir(parents=True, exist_ok=True)
        self.connections, self.processes, self.generation = [], [], 0
        self.lanes = lanes
        self.environment_count = environment_count
        groups = lane_cpus(affinity or os.sched_getaffinity(0), lanes)
        context = mp.get_context('spawn')
        try:
            for group in groups:
                parent, child = context.Pipe()
                process = context.Process(target=_sampling_lane,
                    args=(child, str(frozen_manifest), config, group, environment_count//lanes))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for connection in self.connections:
                if not connection.poll(config['ipc_timeout_seconds']):
                    raise TimeoutError('Sampling lane initialization timed out')
                result = pickle.loads(connection.recv_bytes())
                if not result.get('ready'):
                    raise RuntimeError(result)
        except BaseException:
            self.close()
            raise

    def collect(self, runner, cases, seeds, heartbeat=None, *, probe_steps=None):
        if not cases or len(cases) != len(seeds) or len(cases) > self.environment_count:
            raise ValueError('Complete global group exceeds admitted environment capacity')
        self.generation += 1
        directory = self.output / f'collection_{self.generation:05d}'
        directory.mkdir()
        model = {k: v.detach().cpu() for k, v in runner.policy.ac.state_dict().items()}
        partitions = [list(map(int, a)) for a in np.array_split(np.arange(len(cases)), self.lanes)]
        active = set()
        started = time.monotonic()
        for lane, indices in enumerate(partitions):
            if not indices:
                continue
            self.connections[lane].send_bytes(pickle.dumps(dict(command='collect', model=model,
                policy_updates=runner.policy_updates, indices=indices,
                cases=[cases[i] for i in indices], seeds=[seeds[i] for i in indices],
                output=str(directory / f'lane_{lane}.pkl'),probe_steps=probe_steps), protocol=5))
            active.add(lane)
        rows, rng_states, receipts = [None]*len(cases), [None]*len(cases), []
        last_heartbeat = 0.
        while active:
            for lane in sorted(active.copy()):
                connection = self.connections[lane]
                if not connection.poll():
                    if not self.processes[lane].is_alive():
                        raise RuntimeError(f'Sampling lane {lane} exited')
                    progress = directory / f'lane_{lane}.progress.json'
                    last = progress.stat().st_mtime if progress.exists() else directory.stat().st_mtime
                    if time.time()-last > self.config['ipc_timeout_seconds']:
                        raise TimeoutError(f'Sampling lane {lane} stopped reporting progress')
                    continue
                result = pickle.loads(connection.recv_bytes())
                if probe_steps and result.get('capacity_window') and result.get('policy_updates')==runner.policy_updates:
                    if result['indices']!=partitions[lane]:
                        raise ValueError('Capacity probe changed assigned visits')
                    receipts.append(result)
                    active.remove(lane)
                    continue
                if not result.get('completed') or result.get('policy_updates') != runner.policy_updates:
                    raise RuntimeError(result)
                with Path(result['output']).open('rb') as file:
                    payload = pickle.load(file)
                if result['indices'] != partitions[lane] or len(payload['rows']) != len(partitions[lane]):
                    raise ValueError('Sampling lane changed visit coverage')
                for index, row, rng in zip(result['indices'], payload['rows'], payload['rng_states']):
                    if row['case_id'] != cases[index]['path'] or row['seed'] != seeds[index]:
                        raise ValueError('Sampling lane changed case/seed order')
                    rows[index], rng_states[index] = row, rng
                receipts.append(result)
                active.remove(lane)
                Path(result['output']).unlink()
            if heartbeat and time.monotonic()-last_heartbeat >= 10:
                heartbeat.update(event='parallel_collection', sampling_lanes=self.lanes,
                    completed_visits=sum(r is not None for r in rows), total_visits=len(cases),
                    active_lanes=sorted(active))
                last_heartbeat = time.monotonic()
            if active:
                time.sleep(.1)
        if probe_steps:
            return dict(capacity_window=True,lanes=receipts,
                seconds=time.monotonic()-started,environment_count=len(cases),
                events=sum(r['events'] for r in receipts),
                measured_seconds=max(r['measured_seconds'] for r in receipts))
        # Preserve the original main-process RNG boundary, including a shorter
        # last visit: CPU/Python/NumPy end at its seed, CUDA at the last longest
        # trajectory in the last original group of sixteen.
        ending = dict(rng_states[-1])
        ending['rng_cuda'] = rng_states[terminal_rng_index(rows)]['rng_cuda']
        restore_rng(ending)
        atomic_json(directory / 'receipt.json', dict(version=VERSION, lanes=receipts,
            seconds=time.monotonic()-started, policy_updates=runner.policy_updates,
            visits=len(rows), case_seed_sha256=digest_json(list(zip([c['path'] for c in cases], seeds)))))
        return rows

    def close(self):
        for connection in self.connections:
            try:
                connection.send_bytes(pickle.dumps(dict(command='close'), protocol=5))
            except (OSError, BrokenPipeError):
                pass
        for process in self.processes:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        for connection in self.connections:
            connection.close()
        self.connections, self.processes = [], []


class WholeUpdateCache(GroupExecutionCache):
    """Reuse immutable graph inputs across every pass of the same PPO update."""
    def __init__(self, policy, budget_mib=4096, *, frozen_features=False):
        if budget_mib not in (4096,8192,12288,16384,24576,32768):
            raise ValueError('Use a measured input-cache capacity')
        super().__init__(policy,min(budget_mib,4096),frozen_features=frozen_features)
        self.limit=int(budget_mib*2**20)

    @contextmanager
    def group(self):
        if self.active:
            try:
                yield self
            finally:
                self.last_report = dict(self.stats)
        else:
            with super().group():
                yield self


class ThroughputEngine(BSharedB0Engine):
    def __init__(self, *args, runtime=None, **kwargs):
        self.runtime = dict(runtime or {})
        headroom=self.runtime.get('cuda_memory_headroom_mib')
        if headroom:
            total=torch.cuda.get_device_properties(0).total_memory/2**20
            torch.cuda.set_per_process_memory_fraction((total-headroom)/total,0)
        super().__init__(*args, **kwargs)
        self.sampling_lanes = None
        self.collection_generation = 0
        self.cache = WholeUpdateCache(self.policy, self.runtime.get('input_cache_mib', 4096), frozen_features=False)
        self.policy.stage3_execution_cache = self.cache
        self.policy.ac.stage3_execution_cache = self.cache

    def parallel_collect(self, cases, seeds, heartbeat=None):
        if not self.training:
            raise ValueError('Native fixed12 evaluation cannot enter parallel training collection')
        if self.sampling_lanes is None:
            if self.runtime.get('release_sampling_before_update',False):
                torch.cuda.empty_cache()
            self.collection_generation += 1
            capacity=min(self.runtime.get('sampling_environments',32),1<<(len(cases)-1).bit_length())
            lanes=min(self.runtime.get('sampling_lanes',16),capacity)
            self.sampling_lanes = SamplingLanes(self.bundle.root / 'manifest.json', self.config,
                Path(self.runtime['sampling_output']) / f'pool_{self.collection_generation:05d}',
                lanes=lanes,
                affinity=self.runtime.get('sampling_cpus'),
                environment_count=capacity)
        try:
            return self.sampling_lanes.collect(self, cases, seeds, heartbeat)
        finally:
            # Large backward passes use the GPU after sampling contexts exit.
            # Pool startup/teardown is included in every measured rollout.
            if self.runtime.get('release_sampling_before_update', False):
                self.sampling_lanes.close()
                self.sampling_lanes = None

    @torch.no_grad()
    def replay_metrics(self, trajectories, *, microbatch=None, heartbeat=None):
        self.policy.ac.eval()
        rows = {str(i): dict(decisions=0, kl_sum=0., max_logp_error=0., clip_count=0.) for i in range(3)}
        health = dict(nonfinite=0, mask_mismatch=0)
        deferred = DeferredScalars()
        width = microbatch or self.config['microbatch']
        for start in range(0, len(trajectories), width):
            for step, _, states, _, lp, mask in self._replay(trajectories[start:start+width]):
                old = torch.as_tensor(np.stack([s['old_logp'] for s in states]), device=self.device)
                expected = torch.as_tensor(np.stack([s['mask'] for s in states]), device=self.device)
                deferred.add(health, 'nonfinite', (~torch.isfinite(lp)).sum())
                deferred.add(health, 'mask_mismatch', (mask != expected).sum())
                roles, diff = np.stack([s['roles'] for s in states]), lp-old
                for role, row in rows.items():
                    select = expected.bool() & torch.as_tensor(roles == int(role), device=self.device)
                    d = diff[select]
                    row['decisions'] += d.numel()
                    if d.numel():
                        deferred.add(row, 'kl_sum', (d.clamp(-40, 40).exp()-1-d).sum())
                        deferred.add(row, 'max_logp_error', d.abs().max(), maximum=True)
                        deferred.add(row, 'clip_count', ((d.exp()-1).abs() > .2).sum())
                if (step+1) % self.config['tbptt'] == 0:
                    deferred.flush()
                    if any(health.values()):
                        raise PPOContractError('Nonfinite likelihood or legal decision-mask drift')
                if heartbeat and step % 50 == 0:
                    heartbeat.update(event='likelihood_replay', replay_step=step, replay_microbatch=start//width)
        deferred.flush()
        if any(health.values()):
            raise PPOContractError('Nonfinite likelihood or legal decision-mask drift')
        n = sum(r['decisions'] for r in rows.values())
        for row in rows.values():
            row['kl'] = row['kl_sum']/max(row['decisions'], 1)
        return dict(kl=sum(r['kl_sum'] for r in rows.values())/max(n, 1), roles=rows,
            decisions=n, max_logp_error=max(r['max_logp_error'] for r in rows.values()),
            clip_fraction=sum(r['clip_count'] for r in rows.values())/max(n, 1))

    def update(self, trajectories, source_costs, *, microbatch=None, heartbeat=None, epochs=None):
        width = microbatch or self.runtime.get('microbatch', self.config['microbatch'])
        with self.cache.group():
            result = super().update(trajectories, source_costs, microbatch=width,
                                    heartbeat=heartbeat, epochs=epochs)
        result.update(execution_cache=self.cache.last_report, runtime_version=VERSION)
        return result

    def close(self):
        if self.sampling_lanes is not None:
            self.sampling_lanes.close()
            self.sampling_lanes = None
        super().close()
