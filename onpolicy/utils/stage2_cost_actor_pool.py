"""Bounded private-model CUDA streams; never fuse or reshape inference batches."""
import copy
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import time

import torch


class PrivateActorPool:
    def __init__(self, policy, lanes, *, fast_matching=False):
        if not (1 if fast_matching else 2) <= lanes <= 4:
            raise ValueError('Private CUDA actor lanes must be in [2, 4], or [1, 4] with fast matching.')
        if not torch.are_deterministic_algorithms_enabled():
            raise ValueError('Actor pool requires deterministic numerical configuration.')
        self.contexts = []
        self.local = threading.local()
        self.slots = Queue()
        for _ in range(lanes):
            private = copy.copy(policy)
            private.ac = copy.deepcopy(policy.ac)
            private.ac.eval()
            for parameter in private.ac.parameters():
                parameter.requires_grad_(False)
            if fast_matching:
                from onpolicy.utils.stage2_cost_inference import install_frozen_inference
                install_frozen_inference(private)
            for module in private.ac.modules():
                if isinstance(module, torch.nn.RNNBase):
                    module.flatten_parameters()
            stream = torch.cuda.Stream(device=0)
            self.contexts.append((private, stream))
            self.slots.put((private, stream))
        # Parameter copies/GRU flattening were enqueued on the creating stream.
        # All private streams must wait for them before the first inference.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(0))
        for _, stream in self.contexts:
            stream.wait_event(ready)
        self.ready = ready
        self.executor = ThreadPoolExecutor(max_workers=lanes, initializer=self._initialize,
                                           thread_name_prefix='stage2-frozen-actor')
        self.closed = False

    def _initialize(self):
        self.local.policy, self.local.stream = self.slots.get_nowait()

    def _forward(self, request):
        from onpolicy.utils.stage2_cost_improvement import actor_forward
        key, graphs, hidden, active, history, kinds = request
        started = time.monotonic()
        with torch.cuda.stream(self.local.stream):
            actions, next_hidden = actor_forward(self.local.policy, graphs, hidden, active, history, kinds)
        # actor_forward copies outputs to CPU, completing this stream's result.
        return key, actions, next_hidden, time.monotonic() - started

    def run(self, requests):
        if self.closed:
            raise RuntimeError('Actor pool is closed.')
        futures = [self.executor.submit(self._forward, request) for request in requests]
        try:
            return [future.result() for future in futures]
        except BaseException:
            # Drain owned work before unwinding environment state. No retry.
            for future in futures:
                try:
                    future.result()
                except BaseException:
                    pass
            raise

    def close(self):
        if not self.closed:
            self.executor.shutdown(wait=True)
            self.contexts.clear()
            self.closed = True
