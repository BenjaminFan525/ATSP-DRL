"""Synchronous trajectory-parallel PPO; CPU/Gloo gradient transport.

Models remain ordinary modules: recurrent ragged histories are replayed locally
and gradients are averaged exactly once per PPO epoch, before clipping/Adam.
This avoids assumptions about GPU peer access and DDP's per-backward hooks.
No rollout/history objects are transferred between GPU ranks during training.
"""
from collections import Counter
import hashlib
import json

import numpy as np
import torch
import torch.distributed as dist


def state_digest(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str((str(array.dtype), array.shape)).encode())
            digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for part in item:
                visit(part)
        else:
            digest.update(repr((type(item).__name__, item)).encode())

    visit(value)
    return digest.hexdigest()


def global_case_weights(shards, rank):
    counts = Counter(case for shard in shards for case in shard)
    if not counts or not shards[rank]:
        raise ValueError("Every replica needs a nonempty trajectory shard")
    return np.asarray([len(shards) / (len(counts) * counts[case]) for case in shards[rank]])


class TrajectoryParallel:
    def __init__(self):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.world_size > 1 and dist.get_backend() != "gloo":
            raise ValueError("This auditable implementation requires CPU/Gloo collectives")

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def sum_tensor(self, value):
        value = value.clone()
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def objective_weights(self, case_ids, local_critic_decisions, *, visit_balanced=False):
        shards = self.gather(list(case_ids))
        total = sum(self.gather(float(local_critic_decisions)))
        if total <= 0:
            raise ValueError("No global critic decisions")
        # Subsequent gradient MEAN cancels these world-size multipliers.
        weights = (np.full(len(case_ids), self.world_size / sum(map(len, shards)))
                   if visit_balanced else global_case_weights(shards, self.rank))
        return weights, total / self.world_size

    def _gradients(self, parameters):
        parameters = list(parameters)
        if not parameters:
            return
        present = self.sum_tensor(torch.tensor([p.grad is not None for p in parameters], dtype=torch.int64))
        active = [p for p, n in zip(parameters, present.tolist()) if n]
        if not active:
            return
        flattened = torch.cat([(p.grad.detach().cpu().reshape(-1) if p.grad is not None
                                else torch.zeros(p.numel(), dtype=p.dtype)) for p in active])
        flattened = self.sum_tensor(flattened).div_(self.world_size)
        offset = 0
        for p in active:
            gradient = flattened[offset:offset + p.numel()].view_as(p).to(p.device)
            if p.grad is None:
                p.grad = gradient.clone()
            else:
                p.grad.copy_(gradient)
            offset += p.numel()

    def synchronize_update(self, policy, stats):
        for optimizer in (policy.actor_optimizer, policy.critic_optimizer):
            self._gradients(p for group in optimizer.param_groups for p in group["params"] if p.requires_grad)
        keys = ("kl", "clip", "logp", "decisions", "mask_mismatch")
        values = self.sum_tensor(torch.tensor([stats[k] for k in keys], dtype=torch.float64)).tolist()
        stats.update(zip(keys, values))
        stats["mask_mismatch"] = int(stats["mask_mismatch"])
        stats["distributed_world_size"] = self.world_size

    def replay_sums(self, local):
        rows = self.gather(local)
        result = {}
        for role in local:
            result[role] = {key: sum(row[role][key] for row in rows)
                            for key in ("decisions", "kl_sum", "nll_sum")}
            result[role]["max_logp_error"] = max(row[role]["max_logp_error"] for row in rows)
        return result

    def assert_replicas(self, runner):
        checksum = state_digest({"model": runner.policy.ac.state_dict(),
            "actor": runner.policy.actor_optimizer.state_dict(),
            "critic": runner.policy.critic_optimizer.state_dict(),
            "norms": {str(k): v.state_dict() for k, v in runner.norms.items()},
            "policy_updates": runner.policy_updates})
        checksums = self.gather(checksum)
        if len(set(checksums)) != 1:
            raise RuntimeError(f"Replica model/optimizer/normalizer divergence: {checksums}")
        return checksum
