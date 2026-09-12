"""Opt-in, group-local execution optimizations; no optimizer/RNG changes."""
from contextlib import contextmanager

import torch
from torch_geometric.data import Batch


def detach_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_tree(v) for v in value)
    return value


def tensor_bytes(value):
    """Conservative logical bytes (aliases counted twice, never under-budget)."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Batch):
        return tensor_bytes(value.to_dict())
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class GroupExecutionCache:
    """Cache only immutable retained observations, never trainable activations.

    Admission rather than LRU prevents cyclic replay from evicting the whole
    cache when a long group exceeds the budget. An uncached adjacent actor /
    critic pair can still share its input and a detached encoding. References
    are held alongside identity keys, so Python object-ID reuse cannot hit.
    """
    def __init__(self, policy, budget_mib=1024, *, frozen_features=True):
        if not 0 < budget_mib <= 4096:
            raise ValueError("Group cache must be bounded in (0, 4096] MiB")
        self.policy = policy
        self.frozen_features = bool(frozen_features)
        self.limit = int(budget_mib * 2**20)
        self.active = False
        self.clear()

    def clear(self):
        self.entries = {}
        self.graph_entries = {}
        self.last_input = None
        self.last_actor = None
        self.bytes = 0
        self.stats = dict(input_hits=0, frozen_hits=0, critic_reuses=0,
                          cache_peak_bytes=0, cache_entries=0)

    @contextmanager
    def group(self):
        if self.active:
            raise RuntimeError("Nested group cache is not supported")
        self.clear()
        self.active = True
        try:
            yield self
        finally:
            report = dict(self.stats)
            self.active = False
            self.clear()
            self.last_report = report

    def prepare_graph(self, observations):
        if isinstance(observations, Batch):
            return observations.to(self.policy.device)
        observations = tuple(observations)
        key = tuple(id(g) for g in observations)
        entry = self.entries.get(key)
        if entry is None and self.last_input and self.last_input[0] == key:
            entry = self.last_input[1]
        if entry is not None:
            self.stats["input_hits"] += 1
            self.last_input = (key, entry)
            return entry["graph"]
        graph = Batch.from_data_list(list(observations)).to(self.policy.device)
        entry = {"observations": observations, "graph": graph, "frozen": None}
        self.last_input = (key, entry)
        # Admit features and graph together in encode(), where size is known.
        return graph

    def encode(self, graph, *, actor_grad):
        encoder = self.policy.ac.encoder
        if encoder.training or (self.frozen_features and encoder.prefix.training):
            raise RuntimeError("Feature caching requires deterministic eval-mode graph modules")
        if not actor_grad and self.last_actor is not None and self.last_actor[0] is graph:
            result = self.last_actor[1]
            self.last_actor = None  # Never reuse across an optimizer step.
            self.stats["critic_reuses"] += 1
            return result
        self.last_actor = None
        entry = self.graph_entries.get(id(graph))
        if entry is None and self.last_input and self.last_input[1]["graph"] is graph:
            entry = self.last_input[1]
        if not self.frozen_features:
            # Full-depth policies may reuse immutable inputs and the immediately
            # preceding actor's detached output, NEVER an old trainable output.
            if entry is not None and id(graph) not in self.graph_entries:
                size = tensor_bytes(graph)
                if self.bytes + size <= self.limit:
                    key = tuple(id(g) for g in entry["observations"])
                    self.entries[key] = entry
                    self.graph_entries[id(graph)] = entry
                    self.bytes += size
                    self.stats.update(cache_peak_bytes=self.bytes, cache_entries=len(self.entries))
            if actor_grad and torch.is_grad_enabled():
                result = encoder(graph)
                self.last_actor = (graph, detach_tree(result))
            else:
                with torch.no_grad():
                    result = encoder(graph)
            return result
        frozen = entry["frozen"] if entry else None
        if frozen is None:
            with torch.no_grad():
                frozen = (encoder(graph) if encoder.variant == "E0"
                          else encoder.encode_prefix(graph))
            if entry is not None:
                size = tensor_bytes(graph) + tensor_bytes(frozen)
                if self.bytes + size <= self.limit:
                    entry["frozen"] = frozen
                    key = tuple(id(g) for g in entry["observations"])
                    self.entries[key] = entry
                    self.graph_entries[id(graph)] = entry
                    self.bytes += size
                    self.stats.update(cache_peak_bytes=self.bytes, cache_entries=len(self.entries))
        else:
            self.stats["frozen_hits"] += 1
        if encoder.variant == "E0":
            result = frozen
        elif actor_grad and torch.is_grad_enabled():
            result = encoder.encode_tail(graph, frozen)
        else:
            with torch.no_grad():
                result = encoder.encode_tail(graph, frozen)
        if actor_grad and torch.is_grad_enabled():
            self.last_actor = (graph, detach_tree(result))
        return result


class DeferredScalars:
    """One transfer per TBPTT window; retain original per-step FP32 reductions
    and sequential Python-float summation, including KL threshold decisions.
    """
    def __init__(self):
        self.pending = []

    def add(self, destination, key, value, *, maximum=False):
        self.pending.append((destination, key, value.detach().reshape(()), maximum))

    def flush(self):
        if not self.pending:
            return
        # Float64 transfer preserves both FP32 scalars and integer counts.
        values = torch.stack([row[2].to(torch.float64) for row in self.pending]).cpu().tolist()
        for (destination, key, _, maximum), value in zip(self.pending, values):
            if isinstance(destination[key], int):
                value = int(value)
            destination[key] = max(destination[key], value) if maximum else destination[key] + value
        self.pending.clear()
