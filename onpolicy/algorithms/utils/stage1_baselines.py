"""Stage-1 learning baselines adapted to the HKBZ action contract.

The implementations in this module deliberately share the environment,
legality masks, PPO rollout code and heuristic resource dispatcher used by the
proposed policy.  Only the scheduling representation and operation/site
decoder change:

* L2D-AT: operation-only GIN and an earliest-feasible-completion site rule.
* Multi-PPO-AT: operation GIN followed by operation and site policy heads.
* FJSP-DRL-AT: operation/site heterogeneous message passing and pair scoring.
* DANIEL-AT: operation and site attention blocks with pair-feature scoring.

These are faithful *environment adapters*, not byte-for-byte copies of the
upstream repositories.  HKBZ sites replace machines, while mobile resources
remain outside every baseline encoder and are assigned by the common
Hungarian backend.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, TransformerConv
from torch_geometric.utils import scatter


STAGE1_BASELINES = frozenset({
    "proposed", "l2d", "multi_ppo", "fjsp_drl", "daniel",
})
PAIR_FEATURE_DIM = 10


def normalize_stage1_baseline(value: object) -> str:
    """Return the checkpoint/CLI canonical baseline name."""

    name = str(value or "proposed").strip().lower().replace("-", "_")
    aliases = {
        "ours": "proposed",
        "hgnn": "fjsp_drl",
        "fjspdrl": "fjsp_drl",
        "multippo": "multi_ppo",
    }
    name = aliases.get(name, name)
    if name not in STAGE1_BASELINES:
        raise ValueError(
            f"Unsupported stage1_baseline={value!r}; expected one of "
            f"{sorted(STAGE1_BASELINES)}."
        )
    return name


def _activation_layer(name: object):
    if name in {"tanh", torch.tanh, F.tanh}:
        return nn.Tanh
    if name in {"gelu", F.gelu}:
        return nn.GELU
    return nn.ReLU


def _mlp(in_dim, hidden_dim, out_dim, activation, *, device, dtype):
    layer = _activation_layer(activation)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, device=device, dtype=dtype),
        layer(),
        nn.Linear(hidden_dim, out_dim, device=device, dtype=dtype),
    )


def _finite(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0)


def _node_batch(data, node_type: str, node_count: int, device) -> torch.Tensor:
    storage = data[node_type]
    batch = getattr(storage, "batch", None)
    if batch is None:
        return torch.zeros(node_count, dtype=torch.long, device=device)
    return batch.to(device=device, dtype=torch.long)


def _batch_size(data) -> int:
    return int(getattr(data, "num_graphs", 1))


def _dense_batch(
    values: torch.Tensor,
    batch: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A small zero-node-safe equivalent of PyG ``to_dense_batch``.

    PyG Batch stores nodes graph by graph, so local positions are recovered
    from per-graph prefix counts without allocating a dense adjacency matrix.
    """

    width = values.shape[-1]
    if values.shape[0] == 0:
        return (
            values.new_zeros((batch_size, 0, width)),
            torch.zeros((batch_size, 0), dtype=torch.bool, device=values.device),
        )
    counts = torch.bincount(batch, minlength=batch_size)
    max_count = int(counts.max().item())
    dense = values.new_zeros((batch_size, max_count, width))
    valid = torch.zeros(
        (batch_size, max_count), dtype=torch.bool, device=values.device
    )
    starts = torch.cumsum(counts, dim=0) - counts
    local = torch.arange(values.shape[0], device=values.device)
    local = local - torch.repeat_interleave(starts, counts)
    dense[batch, local] = values
    valid[batch, local] = True
    return dense, valid


def _mean_pool(
    values: torch.Tensor,
    batch: torch.Tensor,
    batch_size: int,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid is not None:
        valid = valid.to(device=values.device, dtype=torch.bool)
        values = values[valid]
        batch = batch[valid]
    if values.shape[0] == 0:
        return values.new_zeros((batch_size, values.shape[-1]))
    total = scatter(values, batch, dim=0, dim_size=batch_size, reduce="sum")
    count = scatter(
        values.new_ones((values.shape[0], 1)),
        batch,
        dim=0,
        dim_size=batch_size,
        reduce="sum",
    )
    return total / count.clamp_min(1.0)


def _relation(data, relation, device) -> tuple[torch.Tensor, torch.Tensor]:
    if relation not in data.edge_types:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0, 1), dtype=torch.float32, device=device),
        )
    edge_index = data[relation].edge_index.to(device=device, dtype=torch.long)
    edge_attr = getattr(data[relation], "edge_attr", None)
    if edge_attr is None:
        edge_attr = torch.zeros(
            (edge_index.shape[1], 1), device=device, dtype=torch.float32
        )
    else:
        edge_attr = edge_attr.to(device=device)
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(-1)
    elif edge_attr.dim() > 2:
        edge_attr = edge_attr.flatten(start_dim=1)
    if edge_attr.dim() != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"Edge attributes for {relation} do not align with edge_index."
        )
    return edge_index, _finite(edge_attr)


def _pair_edge_features(
    edge_index: torch.Tensor,
    raw_edge_attr: torch.Tensor,
    raw_operations: torch.Tensor,
) -> torch.Tensor:
    if edge_index.shape[1] == 0:
        return raw_operations.new_zeros((0, 2))
    travel = raw_edge_attr[:, :1].to(
        device=raw_operations.device, dtype=raw_operations.dtype
    )
    processing = raw_operations[edge_index[0], 1:2]
    return _finite(torch.cat((travel, processing), dim=-1))


class _Stage1Encoder(nn.Module):
    """Shared packing helpers; no baseline consumes device/request features."""

    def __init__(self, common_cfg, gnn_cfg, *, device, dtype):
        super().__init__()
        self.embed_dim = int(common_cfg["embed_dim"])
        self.activation = common_cfg.get("activation", "relu")
        self.op_dim = int(gnn_cfg["op_dim"])
        self.site_dim = int(gnn_cfg["site_dim"])
        self.device = device
        self.dtype = dtype

    def _raw(self, data):
        op = _finite(data["operation"].x.to(device=self.device, dtype=self.dtype))
        site = _finite(data["site"].x.to(device=self.device, dtype=self.dtype))
        batch_size = _batch_size(data)
        op_batch = _node_batch(data, "operation", op.shape[0], op.device)
        site_batch = _node_batch(data, "site", site.shape[0], site.device)
        return op, site, op_batch, site_batch, batch_size

    def _stub_nodes(self, data, node_type, batch_size, reference):
        if node_type not in data.node_types:
            return reference.new_zeros((batch_size, 0, self.embed_dim))
        count = int(data[node_type].x.shape[0])
        batch = _node_batch(data, node_type, count, reference.device)
        zeros = reference.new_zeros((count, self.embed_dim))
        return _dense_batch(zeros, batch, batch_size)[0]

    def _pack(self, data, op, site, global_embedding, op_batch, site_batch):
        batch_size = _batch_size(data)
        op_dense, op_valid = _dense_batch(op, op_batch, batch_size)
        site_dense, site_valid = _dense_batch(site, site_batch, batch_size)
        return {
            "global_emb": _finite(global_embedding),
            "op_nodes": _finite(op_dense),
            "op_padding_mask": ~op_valid,
            "site_nodes": _finite(site_dense),
            "site_padding_mask": ~site_valid,
            "device_nodes": self._stub_nodes(
                data, "device", batch_size, global_embedding
            ),
            "request_nodes": self._stub_nodes(
                data, "request", batch_size, global_embedding
            ),
        }

    @staticmethod
    def _reset(module):
        for child in module.modules():
            if isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            elif isinstance(child, nn.LayerNorm):
                if child.elementwise_affine:
                    nn.init.ones_(child.weight)
                    nn.init.zeros_(child.bias)


class OperationGINEncoder(_Stage1Encoder):
    """Operation-only GIN used by L2D-AT and Multi-PPO-AT."""

    def __init__(
        self,
        common_cfg,
        gnn_cfg,
        *,
        variant,
        layer_num,
        lb_iterations=32,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__(common_cfg, gnn_cfg, device=device, dtype=dtype)
        self.variant = normalize_stage1_baseline(variant)
        self.lb_iterations = max(1, int(lb_iterations))
        op_input_dim = 2 if self.variant == "l2d" else min(5, self.op_dim)
        self.op_embedding = _mlp(
            op_input_dim,
            self.embed_dim,
            self.embed_dim,
            self.activation,
            device=device,
            dtype=dtype,
        )
        self.gin_layers = nn.ModuleList()
        self.op_norms = nn.ModuleList()
        for _ in range(int(layer_num)):
            network = _mlp(
                self.embed_dim,
                self.embed_dim,
                self.embed_dim,
                self.activation,
                device=device,
                dtype=dtype,
            )
            self.gin_layers.append(GINConv(network, train_eps=True))
            self.op_norms.append(
                nn.LayerNorm(self.embed_dim, device=device, dtype=dtype)
            )
        self.site_embedding = None
        self.global_fuse = None
        if self.variant == "multi_ppo":
            self.site_embedding = _mlp(
                min(5, self.site_dim),
                self.embed_dim,
                self.embed_dim,
                self.activation,
                device=device,
                dtype=dtype,
            )
            self.global_fuse = _mlp(
                2 * self.embed_dim,
                self.embed_dim,
                self.embed_dim,
                self.activation,
                device=device,
                dtype=dtype,
            )
        self._reset(self)

    def _l2d_features(self, raw_op, precedence):
        # HKBZ does not expose historical completion timestamps in the policy
        # observation.  The admissible lower bound is therefore recomputed
        # from normalized processing times and the precedence DAG.
        scheduled = (raw_op[:, :1] >= (2.0 / 3.0 - 1e-6)).to(raw_op.dtype)
        processing = raw_op[:, 1].clamp_min(0.0)
        lower_bound = processing
        if precedence.shape[1] > 0:
            source, target = precedence
            for _ in range(self.lb_iterations):
                incoming = scatter(
                    lower_bound[source] + processing[target],
                    target,
                    dim=0,
                    dim_size=raw_op.shape[0],
                    reduce="max",
                )
                lower_bound = torch.maximum(lower_bound, incoming)
        return torch.cat((scheduled, lower_bound.unsqueeze(-1)), dim=-1)

    def forward(self, data):
        raw_op, raw_site, op_batch, site_batch, batch_size = self._raw(data)
        precedence, _ = _relation(
            data, ("operation", "precedes", "operation"), raw_op.device
        )
        if self.variant == "l2d":
            op_input = self._l2d_features(raw_op, precedence)
        else:
            op_input = raw_op[:, : min(5, raw_op.shape[1])]
        op = self.op_embedding(op_input)
        for layer, norm in zip(self.gin_layers, self.op_norms):
            update = layer(op, precedence)
            op = norm(op + F.relu(_finite(update)))
        presence = raw_op[:, -1] > 0.5
        op_pool = _mean_pool(op, op_batch, batch_size, presence)

        if self.variant == "l2d":
            site = raw_site.new_zeros((raw_site.shape[0], self.embed_dim))
            global_embedding = op_pool
        else:
            site = self.site_embedding(
                raw_site[:, : min(5, raw_site.shape[1])]
            )
            site_pool = _mean_pool(site, site_batch, batch_size)
            global_embedding = self.global_fuse(
                torch.cat((op_pool, site_pool), dim=-1)
            )
        return self._pack(
            data, op, site, global_embedding, op_batch, site_batch
        )


class FJSPDRLEncoder(_Stage1Encoder):
    """Operation-site HGNN adapter for Song et al.'s FJSP-DRL."""

    def __init__(
        self,
        common_cfg,
        gnn_cfg,
        *,
        layer_num,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__(common_cfg, gnn_cfg, device=device, dtype=dtype)
        core_op = min(5, self.op_dim)
        core_site = min(5, self.site_dim)
        self.op_embedding = _mlp(
            core_op, self.embed_dim, self.embed_dim, self.activation,
            device=device, dtype=dtype,
        )
        self.site_embedding = _mlp(
            core_site, self.embed_dim, self.embed_dim, self.activation,
            device=device, dtype=dtype,
        )
        self.site_messages = nn.ModuleList()
        self.op_site_messages = nn.ModuleList()
        self.op_updates = nn.ModuleList()
        self.site_updates = nn.ModuleList()
        self.op_norms = nn.ModuleList()
        self.site_norms = nn.ModuleList()
        for _ in range(int(layer_num)):
            self.site_messages.append(_mlp(
                self.embed_dim + 2, self.embed_dim, self.embed_dim,
                self.activation, device=device, dtype=dtype,
            ))
            self.op_site_messages.append(_mlp(
                self.embed_dim + 2, self.embed_dim, self.embed_dim,
                self.activation, device=device, dtype=dtype,
            ))
            self.op_updates.append(_mlp(
                4 * self.embed_dim, 2 * self.embed_dim, self.embed_dim,
                self.activation, device=device, dtype=dtype,
            ))
            self.site_updates.append(_mlp(
                2 * self.embed_dim, self.embed_dim, self.embed_dim,
                self.activation, device=device, dtype=dtype,
            ))
            self.op_norms.append(nn.LayerNorm(
                self.embed_dim, device=device, dtype=dtype
            ))
            self.site_norms.append(nn.LayerNorm(
                self.embed_dim, device=device, dtype=dtype
            ))
        self.global_fuse = _mlp(
            2 * self.embed_dim,
            self.embed_dim,
            self.embed_dim,
            self.activation,
            device=device,
            dtype=dtype,
        )
        self._reset(self)

    @staticmethod
    def _aggregate(messages, index, count, width, reference):
        if messages.shape[0] == 0:
            return reference.new_zeros((count, width))
        return scatter(messages, index, dim=0, dim_size=count, reduce="mean")

    def forward(self, data):
        raw_op, raw_site, op_batch, site_batch, batch_size = self._raw(data)
        precedence, _ = _relation(
            data, ("operation", "precedes", "operation"), raw_op.device
        )
        assignable, raw_pair = _relation(
            data, ("operation", "assignable_to", "site"), raw_op.device
        )
        pair_edge = _pair_edge_features(assignable, raw_pair, raw_op)
        op = self.op_embedding(raw_op[:, : min(5, raw_op.shape[1])])
        site = self.site_embedding(raw_site[:, : min(5, raw_site.shape[1])])

        for index in range(len(self.op_updates)):
            if precedence.shape[1] > 0:
                source, target = precedence
                predecessor = self._aggregate(
                    op[source], target, op.shape[0], self.embed_dim, op
                )
                successor = self._aggregate(
                    op[target], source, op.shape[0], self.embed_dim, op
                )
            else:
                predecessor = torch.zeros_like(op)
                successor = torch.zeros_like(op)
            if assignable.shape[1] > 0:
                op_index, site_index = assignable
                to_site = self.site_messages[index](
                    torch.cat((op[op_index], pair_edge), dim=-1)
                )
                site_context = self._aggregate(
                    to_site, site_index, site.shape[0], self.embed_dim, site
                )
                to_op = self.op_site_messages[index](
                    torch.cat((site[site_index], pair_edge), dim=-1)
                )
                op_site_context = self._aggregate(
                    to_op, op_index, op.shape[0], self.embed_dim, op
                )
            else:
                site_context = torch.zeros_like(site)
                op_site_context = torch.zeros_like(op)
            op_update = self.op_updates[index](torch.cat(
                (op, predecessor, successor, op_site_context), dim=-1
            ))
            site_update = self.site_updates[index](torch.cat(
                (site, site_context), dim=-1
            ))
            op = self.op_norms[index](op + F.relu(_finite(op_update)))
            site = self.site_norms[index](site + F.relu(_finite(site_update)))

        presence = raw_op[:, -1] > 0.5
        op_pool = _mean_pool(op, op_batch, batch_size, presence)
        site_pool = _mean_pool(site, site_batch, batch_size)
        global_embedding = self.global_fuse(
            torch.cat((op_pool, site_pool), dim=-1)
        )
        return self._pack(
            data, op, site, global_embedding, op_batch, site_batch
        )


class DANIELEncoder(_Stage1Encoder):
    """Dual operation/site attention adapter for DANIEL."""

    def __init__(
        self,
        common_cfg,
        gnn_cfg,
        *,
        layer_num,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__(common_cfg, gnn_cfg, device=device, dtype=dtype)
        heads = int(gnn_cfg.get("nhead", 4))
        if self.embed_dim % heads != 0:
            raise ValueError("DANIEL embed_dim must be divisible by nhead.")
        dropout = float(gnn_cfg.get("dropout", 0.1))
        self.op_embedding = _mlp(
            min(5, self.op_dim), self.embed_dim, self.embed_dim,
            self.activation, device=device, dtype=dtype,
        )
        self.site_embedding = _mlp(
            min(5, self.site_dim), self.embed_dim, self.embed_dim,
            self.activation, device=device, dtype=dtype,
        )
        out_channels = self.embed_dim // heads
        self.forward_attention = nn.ModuleList()
        self.backward_attention = nn.ModuleList()
        self.site_attention = nn.ModuleList()
        self.operation_site_attention = nn.ModuleList()
        self.op_merges = nn.ModuleList()
        self.site_merges = nn.ModuleList()
        self.op_norms = nn.ModuleList()
        self.site_norms = nn.ModuleList()
        for _ in range(int(layer_num)):
            self.forward_attention.append(TransformerConv(
                self.embed_dim, out_channels, heads=heads, dropout=dropout
            ))
            self.backward_attention.append(TransformerConv(
                self.embed_dim, out_channels, heads=heads, dropout=dropout
            ))
            self.site_attention.append(TransformerConv(
                (self.embed_dim, self.embed_dim),
                out_channels,
                heads=heads,
                edge_dim=2,
                dropout=dropout,
            ))
            self.operation_site_attention.append(TransformerConv(
                (self.embed_dim, self.embed_dim),
                out_channels,
                heads=heads,
                edge_dim=2,
                dropout=dropout,
            ))
            self.op_merges.append(_mlp(
                4 * self.embed_dim, 2 * self.embed_dim, self.embed_dim,
                self.activation, device=device, dtype=dtype,
            ))
            self.site_merges.append(_mlp(
                2 * self.embed_dim, self.embed_dim, self.embed_dim,
                self.activation, device=device, dtype=dtype,
            ))
            self.op_norms.append(nn.LayerNorm(
                self.embed_dim, device=device, dtype=dtype
            ))
            self.site_norms.append(nn.LayerNorm(
                self.embed_dim, device=device, dtype=dtype
            ))
        self.global_fuse = _mlp(
            2 * self.embed_dim, self.embed_dim, self.embed_dim,
            self.activation, device=device, dtype=dtype,
        )
        self._reset(self)

    def forward(self, data):
        raw_op, raw_site, op_batch, site_batch, batch_size = self._raw(data)
        precedence, _ = _relation(
            data, ("operation", "precedes", "operation"), raw_op.device
        )
        assignable, raw_pair = _relation(
            data, ("operation", "assignable_to", "site"), raw_op.device
        )
        pair_edge = _pair_edge_features(assignable, raw_pair, raw_op)
        op = self.op_embedding(raw_op[:, : min(5, raw_op.shape[1])])
        site = self.site_embedding(raw_site[:, : min(5, raw_site.shape[1])])
        reverse_precedence = precedence.flip(0)
        reverse_assignable = assignable.flip(0)

        for index in range(len(self.op_merges)):
            forward = self.forward_attention[index](op, precedence)
            backward = self.backward_attention[index](op, reverse_precedence)
            from_sites = self.operation_site_attention[index](
                (site, op), reverse_assignable, pair_edge
            )
            from_operations = self.site_attention[index](
                (op, site), assignable, pair_edge
            )
            op_update = self.op_merges[index](torch.cat(
                (op, _finite(forward), _finite(backward), _finite(from_sites)),
                dim=-1,
            ))
            site_update = self.site_merges[index](torch.cat(
                (site, _finite(from_operations)), dim=-1
            ))
            op = self.op_norms[index](op + F.relu(_finite(op_update)))
            site = self.site_norms[index](site + F.relu(_finite(site_update)))

        presence = raw_op[:, -1] > 0.5
        op_pool = _mean_pool(op, op_batch, batch_size, presence)
        site_pool = _mean_pool(site, site_batch, batch_size)
        global_embedding = self.global_fuse(
            torch.cat((op_pool, site_pool), dim=-1)
        )
        return self._pack(
            data, op, site, global_embedding, op_batch, site_batch
        )


def _joint_valid_mask(op_valid_mask, site_valid_mask, site_count):
    op_valid_mask = op_valid_mask.bool()
    if site_valid_mask.dim() == 2:
        site_valid_mask = site_valid_mask.unsqueeze(1).expand(
            -1, op_valid_mask.shape[1], -1
        )
    expected = (
        op_valid_mask.shape[0], op_valid_mask.shape[1], site_count
    )
    if tuple(site_valid_mask.shape) != expected:
        raise ValueError(
            f"site_valid_mask must have shape {expected}, got "
            f"{tuple(site_valid_mask.shape)}."
        )
    joint = op_valid_mask.unsqueeze(-1) & site_valid_mask.bool()
    if (~joint.reshape(joint.shape[0], -1).any(dim=-1)).any():
        bad = torch.nonzero(
            ~joint.reshape(joint.shape[0], -1).any(dim=-1), as_tuple=False
        ).flatten().tolist()
        raise RuntimeError(f"Baseline actor has no legal action for rows {bad}.")
    return joint


def _masked_log_softmax(logits, valid):
    logits = torch.nan_to_num(
        logits, nan=float("-inf"), posinf=20.0, neginf=float("-inf")
    )
    logits = logits.masked_fill(~valid, float("-inf"))
    if (~torch.isfinite(logits).any(dim=-1)).any():
        bad = torch.nonzero(
            ~torch.isfinite(logits).any(dim=-1), as_tuple=False
        ).flatten().tolist()
        raise RuntimeError(f"Baseline logits have no legal value for rows {bad}.")
    return F.log_softmax(logits, dim=-1)


def _select_joint(
    log_probs,
    joint_valid,
    deterministic,
    chosen_op,
    chosen_site,
):
    batch_size, op_count, site_count = joint_valid.shape
    flat_valid = joint_valid.reshape(batch_size, -1)
    if (chosen_op is None) != (chosen_site is None):
        raise ValueError("chosen_op and chosen_site must be supplied together.")
    if chosen_op is None:
        if deterministic:
            flat_index = torch.argmax(log_probs, dim=-1)
        else:
            flat_index = torch.distributions.Categorical(
                logits=log_probs
            ).sample()
        op_index = flat_index // site_count
        site_index = flat_index % site_count
    else:
        op_index = chosen_op.long().reshape(-1)
        site_index = chosen_site.long().reshape(-1)
        in_bounds = (
            (op_index >= 0) & (op_index < op_count)
            & (site_index >= 0) & (site_index < site_count)
        )
        if not in_bounds.all():
            raise RuntimeError("PPO replay contains an out-of-range baseline action.")
        rows = torch.arange(batch_size, device=log_probs.device)
        if not joint_valid[rows, op_index, site_index].all():
            raise RuntimeError("PPO replay contains a masked baseline action.")
        flat_index = op_index * site_count + site_index
    if not flat_valid.gather(1, flat_index.unsqueeze(-1)).all():
        raise RuntimeError("Baseline selector produced an illegal action.")
    selected = log_probs.gather(1, flat_index.unsqueeze(-1)).squeeze(-1)
    return op_index, site_index, selected, log_probs


def _materialize_pair_features(
    *,
    batch_size,
    op_count,
    site_count,
    feature_dim,
    joint_valid,
    reference,
    pair_features=None,
    pair_feature_values=None,
    pair_feature_batch_indices=None,
    pair_feature_flat_ids=None,
):
    expected = (batch_size, op_count, site_count, feature_dim)
    sparse = (
        pair_feature_values,
        pair_feature_batch_indices,
        pair_feature_flat_ids,
    )
    has_sparse = any(value is not None for value in sparse)
    if pair_features is not None and has_sparse:
        raise ValueError("Dense and sparse pair features are mutually exclusive.")
    if pair_features is not None:
        if tuple(pair_features.shape) != expected:
            raise ValueError(
                f"pair_features must have shape {expected}, got "
                f"{tuple(pair_features.shape)}."
            )
        return _finite(pair_features)
    if not all(value is not None for value in sparse):
        raise ValueError("This Stage-1 baseline requires operation-site features.")

    values = pair_feature_values.reshape(-1, feature_dim)
    batches = pair_feature_batch_indices.long().reshape(-1)
    flat_ids = pair_feature_flat_ids.long().reshape(-1)
    if values.shape[0] != batches.numel() or values.shape[0] != flat_ids.numel():
        raise ValueError("Sparse pair feature tensors have inconsistent lengths.")
    pair_count = op_count * site_count
    in_bounds = (
        (batches >= 0) & (batches < batch_size)
        & (flat_ids >= 0) & (flat_ids < pair_count)
    )
    if not in_bounds.all():
        raise ValueError("Sparse pair feature coordinates are out of range.")
    still_legal = joint_valid.reshape(batch_size, -1)[batches, flat_ids]
    values = values[still_legal]
    batches = batches[still_legal]
    flat_ids = flat_ids[still_legal]
    coverage = torch.zeros(
        (batch_size, pair_count), dtype=torch.bool, device=reference.device
    )
    coverage[batches, flat_ids] = True
    missing = joint_valid.reshape(batch_size, -1) & ~coverage
    if missing.any():
        bad = torch.nonzero(missing.any(dim=-1), as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"Sparse pair features do not cover legal actions for rows {bad}."
        )
    dense = reference.new_zeros(expected)
    dense.reshape(batch_size, pair_count, feature_dim)[batches, flat_ids] = values
    return _finite(dense)


class _PairFeatureActor(nn.Module):
    requires_pair_features = True

    def __init__(
        self,
        query_dim,
        embed_dim=64,
        nhead=4,
        activation="relu",
        pair_feature_dim=PAIR_FEATURE_DIM,
        device=None,
        dtype=None,
    ):
        super().__init__()
        del nhead
        self.query_dim = int(query_dim)
        self.embed_dim = int(embed_dim)
        self.pair_feature_dim = int(pair_feature_dim)
        if self.pair_feature_dim != PAIR_FEATURE_DIM:
            raise ValueError(
                f"Stage-1 adapters require {PAIR_FEATURE_DIM} pair features, "
                f"got {self.pair_feature_dim}."
            )
        self.activation = activation
        self.factory_kwargs = {"device": device, "dtype": dtype}

    def _inputs(
        self,
        query,
        op_nodes,
        site_nodes,
        op_valid_mask,
        site_valid_mask,
        pair_features,
        pair_feature_values,
        pair_feature_batch_indices,
        pair_feature_flat_ids,
    ):
        batch_size, op_count = op_valid_mask.shape
        site_count = site_nodes.shape[1]
        joint_valid = _joint_valid_mask(
            op_valid_mask, site_valid_mask, site_count
        )
        pairs = _materialize_pair_features(
            batch_size=batch_size,
            op_count=op_count,
            site_count=site_count,
            feature_dim=self.pair_feature_dim,
            joint_valid=joint_valid,
            reference=op_nodes,
            pair_features=pair_features,
            pair_feature_values=pair_feature_values,
            pair_feature_batch_indices=pair_feature_batch_indices,
            pair_feature_flat_ids=pair_feature_flat_ids,
        )
        # The common runner prepends the graph embedding to its recurrent
        # query.  Baselines intentionally ignore the proposed policy's GRU
        # state and consume only this first block.
        global_embedding = query.squeeze(1)[:, : self.embed_dim]
        return global_embedding, joint_valid, pairs


class L2DActor(_PairFeatureActor):
    """PPO operation policy plus deterministic earliest-finish site choice."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation_score = _mlp(
            2 * self.embed_dim,
            self.embed_dim,
            1,
            self.activation,
            **self.factory_kwargs,
        )
        _Stage1Encoder._reset(self)

    def forward(
        self,
        query,
        op_nodes,
        site_nodes,
        op_valid_mask,
        site_valid_mask,
        deterministic=False,
        chosen_op=None,
        chosen_site=None,
        tau=1.0,
        pair_features=None,
        pair_feature_values=None,
        pair_feature_batch_indices=None,
        pair_feature_flat_ids=None,
    ):
        global_embedding, joint_valid, pairs = self._inputs(
            query, op_nodes, site_nodes, op_valid_mask, site_valid_mask,
            pair_features, pair_feature_values, pair_feature_batch_indices,
            pair_feature_flat_ids,
        )
        batch_size, op_count, site_count = joint_valid.shape
        op_global = global_embedding.unsqueeze(1).expand(-1, op_count, -1)
        op_logits = self.operation_score(
            torch.cat((op_nodes, op_global), dim=-1)
        ).squeeze(-1)
        effective_ops = joint_valid.any(dim=-1)
        op_log_probs = _masked_log_softmax(
            op_logits / max(float(tau), 1e-6), effective_ops
        )

        # pair layout: stay, aircraft travel, processing, site ready,
        # resource ETA, ... .  All values are normalized hours.
        earliest_completion = (
            torch.maximum(
                torch.maximum(pairs[..., 1], pairs[..., 3]),
                pairs[..., 4],
            )
            + pairs[..., 2]
        )
        earliest_completion = earliest_completion.masked_fill(
            ~joint_valid, float("inf")
        )
        rule_site = torch.argmin(earliest_completion, dim=-1)
        rule_valid = joint_valid.gather(2, rule_site.unsqueeze(-1)).squeeze(-1)
        if not rule_valid[effective_ops].all():
            raise RuntimeError("L2D earliest-finish rule failed to select a site.")

        # Supervised teacher replay trains only L2D's operation policy.  The
        # teacher's legal site is used for the common sequential mask replay,
        # not learned as an extra site policy.  This opt-in is scoped by the
        # Stage-1 BC runner and must never relax ordinary PPO action replay.
        operation_teacher_replay = bool(getattr(
            self, "operation_teacher_replay", False
        ))
        if operation_teacher_replay:
            if chosen_op is None or chosen_site is None:
                raise ValueError("L2D teacher replay requires both action labels.")
            teacher_ops = chosen_op.long().reshape(-1)
            teacher_sites = chosen_site.long().reshape(-1)
            if not (
                (teacher_ops >= 0) & (teacher_ops < op_count)
                & (teacher_sites >= 0) & (teacher_sites < site_count)
            ).all():
                raise ValueError("L2D teacher action is out of bounds.")
            teacher_rows = torch.arange(batch_size, device=op_logits.device)
            if not joint_valid[teacher_rows, teacher_ops, teacher_sites].all():
                raise ValueError("L2D teacher action violates the environment mask.")
            self.teacher_site_override_count = int(getattr(
                self, "teacher_site_override_count", 0
            )) + int((
                rule_site[teacher_rows, teacher_ops] != teacher_sites
            ).sum().item())
            rule_site = rule_site.clone()
            rule_site[teacher_rows, teacher_ops] = teacher_sites

        flat_log_probs = op_logits.new_full(
            (batch_size, op_count * site_count), float("-inf")
        )
        rows = torch.arange(batch_size, device=op_nodes.device).unsqueeze(1)
        operations = torch.arange(op_count, device=op_nodes.device).unsqueeze(0)
        flat_ids = operations * site_count + rule_site
        flat_log_probs[rows.expand_as(flat_ids), flat_ids] = op_log_probs
        if (
            chosen_op is not None and chosen_site is not None
            and not operation_teacher_replay
        ):
            replay_site = rule_site.gather(1, chosen_op.long().unsqueeze(-1)).squeeze(-1)
            if not torch.equal(replay_site, chosen_site.long().reshape(-1)):
                raise RuntimeError(
                    "L2D PPO replay site differs from the earliest-finish rule."
                )
        return _select_joint(
            flat_log_probs,
            joint_valid & F.one_hot(
                rule_site, num_classes=site_count
            ).bool(),
            deterministic,
            chosen_op,
            chosen_site,
        )


class MultiPPOActor(_PairFeatureActor):
    """Two-policy operation-then-site decoder with exact joint log-probs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation_score = _mlp(
            2 * self.embed_dim, self.embed_dim, 1, self.activation,
            **self.factory_kwargs,
        )
        self.site_score = _mlp(
            3 * self.embed_dim + 5,
            2 * self.embed_dim,
            1,
            self.activation,
            **self.factory_kwargs,
        )
        _Stage1Encoder._reset(self)

    def forward(
        self,
        query,
        op_nodes,
        site_nodes,
        op_valid_mask,
        site_valid_mask,
        deterministic=False,
        chosen_op=None,
        chosen_site=None,
        tau=1.0,
        pair_features=None,
        pair_feature_values=None,
        pair_feature_batch_indices=None,
        pair_feature_flat_ids=None,
    ):
        global_embedding, joint_valid, pairs = self._inputs(
            query, op_nodes, site_nodes, op_valid_mask, site_valid_mask,
            pair_features, pair_feature_values, pair_feature_batch_indices,
            pair_feature_flat_ids,
        )
        batch_size, op_count, site_count = joint_valid.shape
        op_global = global_embedding[:, None, :].expand(-1, op_count, -1)
        op_logits = self.operation_score(
            torch.cat((op_nodes, op_global), dim=-1)
        ).squeeze(-1)
        op_log_probs = _masked_log_softmax(
            op_logits / max(float(tau), 1e-6), joint_valid.any(dim=-1)
        )
        op_grid = op_nodes[:, :, None, :].expand(-1, -1, site_count, -1)
        site_grid = site_nodes[:, None, :, :].expand(-1, op_count, -1, -1)
        global_grid = global_embedding[:, None, None, :].expand(
            -1, op_count, site_count, -1
        )
        site_logits = self.site_score(torch.cat(
            (op_grid, site_grid, global_grid, pairs[..., :5]), dim=-1
        )).squeeze(-1)
        conditional_mask = joint_valid.reshape(
            batch_size * op_count, site_count
        )
        # Operations rejected by the first policy have no conditional site
        # distribution.  Give those dead rows one temporary finite entry;
        # the operation log-probability and final joint mask remove it again.
        effective_rows = conditional_mask.any(dim=-1)
        safe_conditional_mask = conditional_mask.clone()
        safe_conditional_mask[~effective_rows, 0] = True
        site_log_probs = _masked_log_softmax(
            (site_logits / max(float(tau), 1e-6)).reshape(
                batch_size * op_count, site_count
            ),
            safe_conditional_mask,
        ).reshape(batch_size, op_count, site_count)
        joint_log_probs = (
            op_log_probs.unsqueeze(-1) + site_log_probs
        ).masked_fill(~joint_valid, float("-inf"))
        joint_log_probs = joint_log_probs.reshape(batch_size, -1)
        return _select_joint(
            joint_log_probs, joint_valid, deterministic, chosen_op, chosen_site
        )


class FJSPDRLActor(nn.Module):
    """HGNN operation-site candidate-pair actor."""

    requires_pair_features = False
    pair_feature_dim = PAIR_FEATURE_DIM

    def __init__(
        self,
        query_dim,
        embed_dim=64,
        nhead=4,
        activation="relu",
        pair_feature_dim=PAIR_FEATURE_DIM,
        device=None,
        dtype=None,
    ):
        super().__init__()
        del query_dim, nhead
        self.embed_dim = int(embed_dim)
        self.pair_feature_dim = int(pair_feature_dim)
        self.score = _mlp(
            4 * self.embed_dim,
            2 * self.embed_dim,
            1,
            activation,
            device=device,
            dtype=dtype,
        )
        _Stage1Encoder._reset(self)

    def forward(
        self,
        query,
        op_nodes,
        site_nodes,
        op_valid_mask,
        site_valid_mask,
        deterministic=False,
        chosen_op=None,
        chosen_site=None,
        tau=1.0,
        **unused_pair_features,
    ):
        del unused_pair_features
        batch_size, op_count = op_valid_mask.shape
        site_count = site_nodes.shape[1]
        joint_valid = _joint_valid_mask(
            op_valid_mask, site_valid_mask, site_count
        )
        global_embedding = query.squeeze(1)[:, : self.embed_dim]
        op_grid = op_nodes[:, :, None, :].expand(-1, -1, site_count, -1)
        site_grid = site_nodes[:, None, :, :].expand(-1, op_count, -1, -1)
        global_grid = global_embedding[:, None, None, :].expand(
            -1, op_count, site_count, -1
        )
        logits = self.score(torch.cat(
            (op_grid, site_grid, global_grid, op_grid * site_grid), dim=-1
        )).squeeze(-1).reshape(batch_size, -1)
        log_probs = _masked_log_softmax(
            logits / max(float(tau), 1e-6), joint_valid.reshape(batch_size, -1)
        )
        return _select_joint(
            log_probs, joint_valid, deterministic, chosen_op, chosen_site
        )


class DANIELActor(_PairFeatureActor):
    """DANIEL operation-site actor using the common eight pair attributes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score = _mlp(
            4 * self.embed_dim + 8,
            2 * self.embed_dim,
            1,
            self.activation,
            **self.factory_kwargs,
        )
        _Stage1Encoder._reset(self)

    def forward(
        self,
        query,
        op_nodes,
        site_nodes,
        op_valid_mask,
        site_valid_mask,
        deterministic=False,
        chosen_op=None,
        chosen_site=None,
        tau=1.0,
        pair_features=None,
        pair_feature_values=None,
        pair_feature_batch_indices=None,
        pair_feature_flat_ids=None,
    ):
        global_embedding, joint_valid, pairs = self._inputs(
            query, op_nodes, site_nodes, op_valid_mask, site_valid_mask,
            pair_features, pair_feature_values, pair_feature_batch_indices,
            pair_feature_flat_ids,
        )
        batch_size, op_count, site_count = joint_valid.shape
        op_grid = op_nodes[:, :, None, :].expand(-1, -1, site_count, -1)
        site_grid = site_nodes[:, None, :, :].expand(-1, op_count, -1, -1)
        global_grid = global_embedding[:, None, None, :].expand(
            -1, op_count, site_count, -1
        )
        logits = self.score(torch.cat(
            (
                op_grid,
                site_grid,
                global_grid,
                op_grid * site_grid,
                pairs[..., :8],
            ),
            dim=-1,
        )).squeeze(-1).reshape(batch_size, -1)
        log_probs = _masked_log_softmax(
            logits / max(float(tau), 1e-6), joint_valid.reshape(batch_size, -1)
        )
        return _select_joint(
            log_probs, joint_valid, deterministic, chosen_op, chosen_site
        )


def build_stage1_baseline_encoder(
    name,
    common_cfg,
    encoder_cfg,
    *,
    device="cpu",
    dtype=torch.float32,
):
    """Build the canonical encoder for a non-proposed Stage-1 baseline."""

    name = normalize_stage1_baseline(name)
    if name == "proposed":
        raise ValueError("The proposed encoder is built by GNN_Actor_Critic.")
    gnn_cfg = encoder_cfg["gnn_cfg"]
    baseline_cfg = encoder_cfg.get("stage1_baseline_cfg", {})
    if not isinstance(baseline_cfg, Mapping):
        raise ValueError("encoder_cfg.stage1_baseline_cfg must be a mapping.")
    if name == "l2d":
        return OperationGINEncoder(
            common_cfg,
            gnn_cfg,
            variant=name,
            layer_num=int(baseline_cfg.get("l2d_layers", 2)),
            lb_iterations=int(baseline_cfg.get("l2d_lb_iterations", 32)),
            device=device,
            dtype=dtype,
        )
    if name == "multi_ppo":
        return OperationGINEncoder(
            common_cfg,
            gnn_cfg,
            variant=name,
            layer_num=int(baseline_cfg.get("multi_ppo_layers", 3)),
            device=device,
            dtype=dtype,
        )
    if name == "fjsp_drl":
        return FJSPDRLEncoder(
            common_cfg,
            gnn_cfg,
            layer_num=int(baseline_cfg.get("fjsp_drl_layers", 2)),
            device=device,
            dtype=dtype,
        )
    return DANIELEncoder(
        common_cfg,
        gnn_cfg,
        layer_num=int(baseline_cfg.get("daniel_layers", 3)),
        device=device,
        dtype=dtype,
    )


def build_stage1_baseline_actor(
    name,
    common_cfg,
    actor_cfg,
    *,
    device="cpu",
    dtype=torch.float32,
):
    """Build an actor that obeys the common flattened joint-action API."""

    name = normalize_stage1_baseline(name)
    actor_types = {
        "l2d": L2DActor,
        "multi_ppo": MultiPPOActor,
        "fjsp_drl": FJSPDRLActor,
        "daniel": DANIELActor,
    }
    if name == "proposed":
        raise ValueError("The proposed actor is built by GNN_Actor_Critic.")
    config = dict(actor_cfg)
    config.setdefault("pair_feature_dim", PAIR_FEATURE_DIM)
    return actor_types[name](
        **common_cfg, **config, device=device, dtype=dtype
    )
