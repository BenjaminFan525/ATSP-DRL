"""Independent resource-only V6 adapters; opt-in after loading legacy B0."""
from __future__ import annotations

import torch
from torch import nn

from onpolicy.utils.stage2_resource_v6_observation import (
    DEVICE_DIM, GLOBAL_DIM, HISTORY_DIM, PAIR_DIM, REQUEST_DIM,
)


def masked_pool(values, present):
    present = present.bool()
    mean = (values * present.unsqueeze(-1)).sum(1) / present.sum(1, keepdim=True).clamp_min(1)
    maximum = values.masked_fill(~present.unsqueeze(-1), -torch.inf).amax(1)
    maximum = torch.where(present.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), -1)


def resource_context(graph):
    if not hasattr(graph, 'v6_schema') or not bool((graph.v6_schema == 1).all()):
        raise ValueError('V6 actor requires its versioned resource observation.')
    requests, devices = graph.v6_requests, graph.v6_devices
    if requests.shape[-1] != REQUEST_DIM or devices.shape[-1] != DEVICE_DIM:
        raise ValueError('Incorrect V6 feature dimensions.')
    global_values = torch.cat((masked_pool(requests, requests[..., 0] > 0),
        masked_pool(devices, devices[..., -1] > 0), graph.v6_counts), -1)
    n = requests.shape[0]
    masks = graph.request_mask_matrix.reshape(n, -1, requests.shape[1])[:, -devices.shape[1]:]
    capacity = masks[..., 1:].sum(1).to(requests.dtype) / devices[..., -1].sum(1, keepdim=True).clamp_min(1)
    capacity = torch.cat((torch.zeros_like(capacity[:, :1]), capacity), -1)
    return dict(requests=requests, devices=devices, pairs=graph.v6_pairs,
        history=graph.v6_history, global_values=global_values, masks=masks, capacity=capacity)


def pair_inputs(context, rows, device_idx, legal):
    requests = context['requests'][rows]
    count = requests.shape[1]
    original = context['masks'][rows, device_idx]
    filtered = original & ~legal  # Includes claims AND the last-chance legality restriction.
    prefix = torch.stack((legal.float(), filtered.float()), -1)
    capacity = context['capacity'][rows].unsqueeze(-1)
    available = original.sum(-1).to(requests.dtype).div(max(1, count - 1))[:, None, None].expand(-1, count, 1)
    return torch.cat((context['devices'][rows, device_idx, None, :].expand(-1, count, -1),
        requests, context['global_values'][rows, None, :].expand(-1, count, -1),
        context['pairs'][rows, device_idx], capacity, available, prefix), -1)


class ResourceActorV6(nn.Module):
    def __init__(self, arm, embed_dim=64):
        super().__init__()
        if arm not in ('S1', 'S2'):
            raise ValueError('Only preregistered S1/S2 are enabled.')
        self.arm = arm
        if arm == 'S1':
            self.history_encoder = nn.Sequential(nn.Linear(HISTORY_DIM, embed_dim), nn.Tanh())
        else:
            self.heads = nn.ModuleDict({role: nn.Sequential(nn.Linear(PAIR_DIM, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
                for role in ('ordinary', 'transporter')})

    def history_embedding(self, context, rows, device_idx):
        raw = context['history'][rows, device_idx]
        return self.history_encoder(raw) * raw[:, -2:-1]

    def distribution(self, features, legal, role, tau):
        logits = self.heads[role](features).squeeze(-1).float() / float(tau)
        if not bool(torch.isfinite(logits).all()) or not bool(legal.any(-1).all()):
            raise RuntimeError('Invalid V6 scores or empty action support.')
        return torch.log_softmax(logits.masked_fill(~legal, -torch.inf), -1)

    def act(self, context, rows, device_idx, role, legal, deterministic, tau, chosen=None):
        logits = self.distribution(pair_inputs(context, rows, device_idx, legal), legal, role, tau)
        if chosen is None:
            chosen = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        if not bool(legal.gather(1, chosen.long().unsqueeze(-1)).all()):
            raise RuntimeError('V6 replay selected an illegal request.')
        return chosen, logits.gather(1, chosen.long().unsqueeze(-1)).squeeze(-1), logits


def install_resource_v6(ac, arm):
    if hasattr(ac, 'resource_v6') or getattr(ac, 'resource_encoder', None) is not None:
        raise ValueError('V6 must start from a legacy B0, not an already extended model.')
    if ac.request_ready_policy_injection != 'none' or ac.device_policy_head_mode != 'shared':
        raise ValueError('V6 initial screen requires B0-none with shared role heads.')
    device = next(ac.parameters()).device
    ac.resource_v6 = ResourceActorV6(arm).to(device)
    ac.resource_v6_critic = nn.Sequential(nn.Linear(GLOBAL_DIM, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
    ac.device_global_matching = False
