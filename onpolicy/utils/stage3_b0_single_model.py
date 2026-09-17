"""One trainable GPU policy, batched inference, and CPU-only environment IPC.

PPO computation is inherited unchanged from the admitted throughput adapter.
Stochastic actions consume the restored global Torch RNG in stable live-slot
order. This is a new sampling stream, not batch-one numerical equivalence.
"""
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import time
import traceback
from functools import wraps

from onpolicy.runner.shared.stage3_research_engine import EnvironmentPool, _environment_worker
from onpolicy.utils.stage3_b0_throughput import ThroughputEngine as UpdateEngine, lane_cpus
from onpolicy.utils.stage3_research import atomic_json

VERSION = 'single_model_batched_v1'
RNG_CONTRACT = 'checkpoint_global_torch_live_slot_order_v1'


def checkpoint_encoder(encoder):
    """Recompute trainable GNN activations; no-grad sampling stays unchanged."""
    import torch
    from torch.utils.checkpoint import checkpoint
    original = encoder.forward
    @wraps(original)
    def forward(*args, **kwargs):
        if torch.is_grad_enabled() and any(p.requires_grad for p in encoder.parameters()):
            return checkpoint(original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
        return original(*args, **kwargs)
    encoder.forward = forward


class EnvironmentSlot:
    """Independent training environment state inside a CPU worker."""
    def __init__(self):
        self.env, self.steps = None, 0
        self.rng = None

    def execute(self, command, payload):
        import numpy as np
        import torch
        if self.rng is not None:
            random.setstate(self.rng[0])
            np.random.set_state(self.rng[1])
            torch.set_rng_state(self.rng[2])
        try:
            return self._execute(command, payload)
        finally:
            self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())

    def _execute(self, command, payload):
        if command == 'reset':
            from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
            self.close()
            config = dict(payload)
            if config.pop('_stage3_diagnostics', False) or '_stage3_native_reset_seed' in config:
                raise ValueError('Multiplexed environment workers are for stochastic training only')
            self.env = AircraftScheduleEnv(config)
            self.steps = 0
            return self.env.reset(seed=config['seed'])
        if command == 'step':
            result = self.env.step(payload)
            self.steps += 1
            return result
        if command == 'summary':
            from onpolicy.envs.HKBZ.experiment.eval_common import completion_details
            result = completion_details(self.env, self.steps, 4000)
            result['makespan'] = float(self.env.total_time)
            return result
        raise ValueError(command)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def _multiplexed_cpu_environment(connection, affinity):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.sched_setaffinity(0, affinity)
    import torch
    torch.set_num_threads(1)
    slots = {}
    try:
        while True:
            commands = pickle.loads(connection.recv_bytes())
            if commands is None:
                break
            try:
                results = {}
                for index, command, payload in commands:
                    if index not in slots:
                        slots[index] = EnvironmentSlot()
                    results[index] = slots[index].execute(command, payload)
                connection.send_bytes(pickle.dumps((True, results), protocol=5))
            except Exception:
                connection.send_bytes(pickle.dumps((False, traceback.format_exc()), protocol=5))
                break
    except EOFError:
        pass
    finally:
        for slot in slots.values():
            slot.close()
        connection.close()


def _cpu_environment(connection, affinity):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.sched_setaffinity(0, affinity)
    _environment_worker(connection)


class CpuEnvironmentPool(EnvironmentPool):
    def __init__(self, width, *, timeout, affinity, worker_count=None):
        self.timeout = timeout
        self.connections, self.processes = [], []
        self.width = width
        self.worker_count = min(width, worker_count or width)
        if not 1 <= self.worker_count <= width:
            raise ValueError('Invalid CPU worker count')
        self.multiplexed = self.worker_count < width
        cores = lane_cpus(affinity, len([c for c in affinity if c < 64]))
        context = mp.get_context('spawn')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        # Spawned interpreters also import their entrypoint before the target.
        # Hide CUDA throughout that startup, not just inside the worker loop.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        try:
            for i in range(self.worker_count):
                parent, child = context.Pipe()
                process = context.Process(target=_multiplexed_cpu_environment if self.multiplexed else _cpu_environment,
                    args=(child, cores[i % len(cores)]), daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.connections.append(parent)
                self.processes.append(process)
        except BaseException:
            self.close()
            raise
        finally:
            if visible is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = visible

    def call(self, commands):
        if not self.multiplexed:
            return super().call(commands)
        grouped = {}
        for index, command, payload in commands:
            if not 0 <= index < self.width:
                raise ValueError('Environment slot outside the configured pool')
            grouped.setdefault(index % self.worker_count, []).append((index, command, payload))
        for worker, items in grouped.items():
            self.connections[worker].send_bytes(pickle.dumps(items, protocol=5))
        result = {}
        for worker in grouped:
            connection = self.connections[worker]
            if not connection.poll(self.timeout):
                raise TimeoutError(f'CPU environment worker {worker} stalled')
            ok, payload = pickle.loads(connection.recv_bytes())
            if not ok:
                raise RuntimeError(f'CPU environment worker {worker} failed:\n{payload}')
            result.update(payload)
        if set(result) != {index for index, _, _ in commands}:
            raise RuntimeError('CPU environment worker returned incorrect logical slots')
        return result

    def close(self):
        # Broadcast close before joining: all environment processes can release
        # their interpreters and resources concurrently.
        for connection in self.connections:
            try:
                connection.send_bytes(pickle.dumps(None if self.multiplexed else ('close', None)))
            except (BrokenPipeError, OSError):
                pass
        for connection, process in zip(self.connections, self.processes):
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            connection.close()
        self.connections, self.processes = [], []


class ThroughputEngine(UpdateEngine):
    batched_sampling = True

    def __init__(self, *args, runtime=None, **kwargs):
        requested = dict(runtime or {})
        cache_mib = requested.get('input_cache_mib', 4096)
        if cache_mib not in (1024, 4096):
            raise ValueError('Single-model cache budget must have capacity evidence')
        # Reuse the admitted cache implementation; a lower admission ceiling
        # changes retention only, never the tensors or the PPO computation.
        super().__init__(*args, runtime=dict(requested, input_cache_mib=4096), **kwargs)
        self.runtime = requested
        self.cache.limit = cache_mib*2**20
        if requested.get('encoder_activation_checkpoint', False):
            checkpoint_encoder(self.policy.ac.encoder)

    def parallel_collect(self, cases, seeds, heartbeat=None):
        if not self.training or self.runtime.get('sampling_lanes') != 1:
            raise ValueError('Single-model training requires exactly one inference owner')
        if not 0 < len(cases) <= self.width or len(seeds) != len(cases):
            raise ValueError('Invalid single-model collection batch')
        started = time.monotonic()
        self.collection_generation += 1
        output = Path(self.runtime['sampling_output']) / f'pool_{self.collection_generation:05d}'
        if self.pool is None:
            self.pool = CpuEnvironmentPool(len(cases), timeout=self.config['ipc_timeout_seconds'],
                affinity=self.runtime.get('sampling_cpus', sorted(os.sched_getaffinity(0))),
                worker_count=self.runtime.get('environment_processes', 64))
        metadata = dict(architecture=VERSION, rng_contract=RNG_CONTRACT,
            model_pid=os.getpid(), policy_updates=self.policy_updates, model_copies=1,
            environment_count=len(cases), environment_pids=[p.pid for p in self.pool.processes],
            environment_processes=len(self.pool.processes),
            environment_cuda_visible_devices='', sampling_started_unix=time.time())
        atomic_json(output / 'workers.json', metadata)
        if heartbeat:
            heartbeat.update(event='single_model_collection', **metadata)
        try:
            rows = self.rollout(cases, seeds, retain=True, heartbeat=heartbeat)
            for row in rows:
                row.update(sampling_architecture=VERSION, sampling_rng_contract=RNG_CONTRACT)
            metadata.update(completed=True, visits=len(rows),
                environment_steps=sum(row['steps'] for row in rows))
            return rows
        finally:
            # Only the CPU environment pool exits. The same policy and CUDA
            # context remain in this process for the subsequent PPO updates.
            self.pool.close()
            self.pool = None
            metadata['seconds'] = time.monotonic()-started
            atomic_json(output / 'result.json', metadata)
