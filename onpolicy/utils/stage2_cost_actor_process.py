"""Spawn-only frozen CUDA actor workers with bounded ownership and no retry.

Each request is still one original full heterogeneous batch. Bytes transport
avoids hundreds of torch shared-memory handles per PyG graph collection.
"""
import copy
from concurrent.futures import ProcessPoolExecutor
import ctypes
import multiprocessing
import os
import signal
import time

import cloudpickle
import torch

_POLICY = None


def _initialize(model_bytes, cpus, memory_fraction, parent_pid):
    global _POLICY
    # Linux process ownership: a killed trainer must not leave GPU workers.
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0:
        raise RuntimeError('Cannot install actor parent-death signal.')
    if os.getppid() != parent_pid:
        raise RuntimeError('Actor parent exited during startup.')
    os.sched_setaffinity(0, cpus)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Frozen actor processes are restricted to physical GPU0.')
    from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
    from onpolicy.utils.stage2_bc_contract import configure_bc_determinism
    from onpolicy.utils.stage2_cost_inference import install_frozen_inference
    configure_bc_determinism(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
    policy = GNN_MAPPOPolicy.__new__(GNN_MAPPOPolicy)
    policy.device = torch.device('cuda:0')
    policy.ac = cloudpickle.loads(model_bytes).to(policy.device)
    policy.ac.eval().requires_grad_(False)
    for module in policy.ac.modules():
        if isinstance(module, torch.nn.RNNBase):
            module.flatten_parameters()
    install_frozen_inference(policy)
    _POLICY = policy


def _forward(payload):
    from onpolicy.utils.stage2_cost_improvement import actor_forward
    started = time.monotonic()
    key, graphs, hidden, active, history, kinds = cloudpickle.loads(payload)
    actions, next_hidden = actor_forward(_POLICY, graphs, hidden, active, history, kinds)
    return key, actions, next_hidden, time.monotonic() - started


class ProcessActorPool:
    def __init__(self, policy, lanes, *, memory_budget=.35):
        if not 1 <= lanes <= 4 or not 0 < memory_budget <= .70:
            raise ValueError('Process actor lanes/budget exceed the reserved experiment capacity.')
        if not torch.are_deterministic_algorithms_enabled():
            raise ValueError('Actor pool requires deterministic numerical configuration.')
        cpus = os.sched_getaffinity(0)
        if not cpus or not cpus.issubset(set(range(32)) | set(range(64, 96))):
            raise ValueError('Actor workers must inherit only the reserved CPU half/lane.')
        worker_fraction = min(.03, memory_budget / (lanes + 2))
        self.parent_fraction = memory_budget - lanes * worker_fraction
        # These are PyTorch allocator caps; CUDA contexts are monitored in the
        # physical GPU preflight/screen and are not falsely counted as tensors.
        torch.cuda.set_per_process_memory_fraction(self.parent_fraction, device=0)
        cpu_model = copy.deepcopy(policy.ac).cpu()
        model_bytes = cloudpickle.dumps(cpu_model, protocol=5)
        self.executor = ProcessPoolExecutor(max_workers=lanes,
            mp_context=multiprocessing.get_context('spawn'), initializer=_initialize,
            initargs=(model_bytes, cpus, worker_fraction, os.getpid()))
        self.closed = False

    def run(self, requests):
        if self.closed:
            raise RuntimeError('Actor pool is closed.')
        futures = [self.executor.submit(_forward, cloudpickle.dumps(request, protocol=5))
                   for request in requests]
        # BrokenProcessPool is propagated, not respawned/retried. The owning
        # service has a hard global deadline even during an unresponsive RPC.
        return [future.result(timeout=600.) for future in futures]

    def close(self):
        if not self.closed:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.closed = True
