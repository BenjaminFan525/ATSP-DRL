import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from typing import Optional, Tuple
from onpolicy.algorithms.utils.util import activations, Embedding_layer

def _safe_normalize_probs(prob: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
    prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    if valid_mask is not None:
        if (~valid_mask.any(dim=-1)).any():
            bad_rows = torch.nonzero(~valid_mask.any(dim=-1), as_tuple=False).flatten().tolist()
            raise RuntimeError(f"Pointer mask contains no legal action for rows {bad_rows}.")
        prob = prob.masked_fill(~valid_mask, 0.0)

    denom = prob.sum(dim=-1, keepdim=True)
    if (denom <= 0.0).any():
        bad_rows = torch.nonzero((denom <= 0.0).squeeze(-1), as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Pointer probabilities are empty for rows {bad_rows}.")
    return prob / denom

def _safe_logits(logits: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
    logits = torch.nan_to_num(logits, nan=float('-inf'), posinf=0.0, neginf=float('-inf'))
    if valid_mask is not None:
        if (~valid_mask.any(dim=-1)).any():
            bad_rows = torch.nonzero(~valid_mask.any(dim=-1), as_tuple=False).flatten().tolist()
            raise RuntimeError(f"Action mask contains no legal action for rows {bad_rows}.")
        logits = logits.masked_fill(~valid_mask, float('-inf'))
    dead_ends = ~torch.isfinite(logits).any(dim=-1)
    if dead_ends.any():
        bad_rows = torch.nonzero(dead_ends, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Action logits contain no finite legal action for rows {bad_rows}.")
    return logits

def _masked_log_probs(prob: torch.Tensor, valid_mask: torch.Tensor):
    if prob.shape != valid_mask.shape:
        raise ValueError(
            f"Probability shape {tuple(prob.shape)} does not match "
            f"mask shape {tuple(valid_mask.shape)}."
        )
    if (~valid_mask.any(dim=-1)).any():
        bad_rows = torch.nonzero(~valid_mask.any(dim=-1), as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Action mask contains no legal action for rows {bad_rows}.")

    log_prob = torch.full_like(prob, float('-inf'))
    legal_prob = prob.masked_select(valid_mask)
    if not torch.isfinite(legal_prob).all() or (legal_prob < 0.0).any():
        raise RuntimeError("Legal action probabilities must be finite and non-negative.")
    min_prob = torch.finfo(prob.dtype).tiny
    log_prob.masked_scatter_(valid_mask, torch.log(legal_prob.clamp_min(min_prob)))
    return log_prob

def _select_masked_index(logits: torch.Tensor, valid_mask: torch.Tensor, deterministic: bool):
    logits = _safe_logits(logits, valid_mask=valid_mask)
    if deterministic:
        selected = torch.argmax(logits, dim=-1)
    else:
        selected = torch.distributions.Categorical(logits=logits).sample()

    selected_is_valid = valid_mask.gather(1, selected.unsqueeze(-1)).squeeze(-1)
    if not selected_is_valid.all():
        bad_rows = torch.nonzero(~selected_is_valid, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Masked selector produced an illegal action for rows {bad_rows}.")
    selected_logits = logits.gather(1, selected.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(selected_logits).all():
        raise RuntimeError("Masked selector produced an action with non-finite log probability.")
    return selected

class MaPtrNet(Module):
    def __init__(self, query_dim, embed_dim, bias=True, device=None, dtype=None) -> None:
        """
        标准指针网络核心层。
        计算 Query 对一组 Key 的 Attention 概率分布。
        """
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.bias = bias

        self.q_proj_weight = nn.Linear(query_dim, embed_dim, bias, **self.factory_kwargs)
        self.k_proj_weight = nn.Linear(embed_dim, embed_dim, bias, **self.factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight.weight)
        nn.init.xavier_uniform_(self.k_proj_weight.weight)
        if self.bias:
            nn.init.constant_(self.q_proj_weight.bias, 0.)
            nn.init.constant_(self.k_proj_weight.bias, 0.)

    def dist(self, query: torch.Tensor, key: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, tau: float = 1.0, return_raw: bool = False):
        """
        输出经过 Mask 屏蔽后的概率分布。
        注意：PyTorch 标准中，key_padding_mask 为 True 代表是被屏蔽的非法位置 (Padding)。
        """
        q = torch.nan_to_num(self.q_proj_weight(query), nan=0.0, posinf=1e4, neginf=-1e4)
        k = torch.nan_to_num(self.k_proj_weight(key), nan=0.0, posinf=1e4, neginf=-1e4)

        # 缩放点积注意力 (Scaled Dot-Product Attention)
        ptr = torch.tanh(torch.bmm(q, k.transpose(2, 1)) / (self.embed_dim ** 0.5))
        ptr = torch.nan_to_num(ptr, nan=0.0, posinf=20.0, neginf=-20.0)
        
        # 掩码处理：将非法节点的 Logits 设为负无穷
        valid_mask = None
        if key_padding_mask is not None:
            # 确保 mask 的维度是 [Batch, 1, Seq_len] 以便进行 broadcast
            if key_padding_mask.dim() == 2:
                key_padding_mask = key_padding_mask.unsqueeze(1)
            if key_padding_mask.all(dim=-1).any():
                raise RuntimeError("Pointer attention received an all-masked action row.")
            valid_mask = ~key_padding_mask
            ptr = ptr.masked_fill(key_padding_mask, float('-inf'))
            
        if return_raw:
            return ptr
        prob = F.softmax(ptr / tau, dim=-1)
        return _safe_normalize_probs(prob, valid_mask=valid_mask)


class PlaneOrderPointer(Module):
    """Choose active-plane order and expose exact per-plane PPO log-probs."""

    def __init__(self, embed_dim=64, activation=F.relu, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(activation, str):
            activation = activations[activation]
        self.embed_dim = int(embed_dim)
        self.global_proj = nn.Linear(
            self.embed_dim, self.embed_dim, **factory_kwargs
        )
        self.selected_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=False, **factory_kwargs
        )
        self.candidate_proj = nn.Linear(
            self.embed_dim, self.embed_dim, **factory_kwargs
        )
        activation_layer = nn.ReLU if activation is F.relu else nn.Tanh
        self.score = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim, **factory_kwargs),
            activation_layer(),
            nn.Linear(self.embed_dim, 1, **factory_kwargs),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)

    def forward(
        self,
        global_emb,
        plane_nodes,
        active_mask,
        deterministic=False,
        chosen_ranks=None,
        tau=1.0,
    ):
        if active_mask.shape != plane_nodes.shape[:2]:
            raise ValueError(
                "active_mask and plane_nodes must agree on [batch, planes]."
            )
        batch_size, plane_count, _ = plane_nodes.shape
        device = plane_nodes.device
        dtype = plane_nodes.dtype
        remaining = active_mask.bool().clone()
        order_by_rank = torch.full(
            (batch_size, plane_count), -1, dtype=torch.long, device=device
        )
        rank_by_plane = torch.full_like(order_by_rank, -1)
        log_prob_by_plane = torch.zeros(
            (batch_size, plane_count), dtype=dtype, device=device
        )
        entropy_by_plane = torch.zeros_like(log_prob_by_plane)
        trainable_by_plane = torch.zeros(
            (batch_size, plane_count), dtype=torch.bool, device=device
        )
        selected_sum = torch.zeros(
            (batch_size, self.embed_dim), dtype=dtype, device=device
        )
        selected_count = torch.zeros(
            (batch_size, 1), dtype=dtype, device=device
        )

        if chosen_ranks is not None:
            chosen_ranks = chosen_ranks.long()
            if chosen_ranks.shape != (batch_size, plane_count):
                raise ValueError(
                    "chosen_ranks must have shape "
                    f"[{batch_size}, {plane_count}], got {tuple(chosen_ranks.shape)}."
                )
            if ((~active_mask.bool()) & (chosen_ranks >= 0)).any():
                raise RuntimeError(
                    "PPO replay assigns an order rank to an inactive plane."
                )

        global_context = self.global_proj(global_emb)
        candidate_context = self.candidate_proj(plane_nodes)
        for rank in range(plane_count):
            rows = remaining.any(dim=-1)
            if not rows.any():
                break
            row_indices = torch.nonzero(rows, as_tuple=False).flatten()
            row_remaining = remaining[rows]
            row_candidates = candidate_context[rows]
            selected_mean = (
                selected_sum[rows] / selected_count[rows].clamp_min(1.0)
            )
            context = global_context[rows] + self.selected_proj(selected_mean)
            expanded_context = context.unsqueeze(1).expand(
                -1, plane_count, -1
            )
            score_input = torch.cat(
                (
                    row_candidates,
                    expanded_context,
                    row_candidates * expanded_context,
                ),
                dim=-1,
            )
            logits = self.score(score_input).squeeze(-1)
            logits = logits / max(float(tau), 1e-6)
            logits = _safe_logits(logits, valid_mask=row_remaining)
            log_probs = F.log_softmax(logits, dim=-1)

            if chosen_ranks is None:
                selected = _select_masked_index(
                    log_probs, row_remaining, deterministic
                )
            else:
                rank_match = chosen_ranks[rows] == rank
                match_count = rank_match.sum(dim=-1)
                if not (match_count == 1).all():
                    bad_rows = row_indices[match_count != 1].tolist()
                    raise RuntimeError(
                        "PPO replay plane-order ranks are not contiguous; "
                        f"rank={rank}, rows={bad_rows}."
                    )
                selected = torch.argmax(rank_match.long(), dim=-1)
                selected_valid = row_remaining.gather(
                    1, selected.unsqueeze(-1)
                ).squeeze(-1)
                if not selected_valid.all():
                    bad_rows = row_indices[~selected_valid].tolist()
                    raise RuntimeError(
                        "PPO replay selects an unavailable plane; "
                        f"rank={rank}, rows={bad_rows}."
                    )

            selected_lp = log_probs.gather(
                1, selected.unsqueeze(-1)
            ).squeeze(-1)
            entropy = torch.distributions.Categorical(
                logits=logits
            ).entropy()
            trainable = row_remaining.sum(dim=-1) > 1
            order_by_rank[row_indices, rank] = selected
            rank_by_plane[row_indices, selected] = rank
            log_prob_by_plane[row_indices, selected] = selected_lp
            entropy_by_plane[row_indices, selected] = entropy
            trainable_by_plane[row_indices, selected] = trainable
            selected_sum[row_indices] += plane_nodes[rows, selected]
            selected_count[row_indices] += 1.0
            remaining[row_indices, selected] = False

        if ((rank_by_plane >= 0) != active_mask.bool()).any():
            raise RuntimeError(
                "Plane order pointer did not rank every active plane exactly once."
            )
        return (
            order_by_rank,
            rank_by_plane,
            log_prob_by_plane,
            entropy_by_plane,
            trainable_by_plane,
        )


class JointPairPtrActor(Module):
    """Score legal operation-site pairs with explicit cross features."""

    def __init__(self, query_dim, embed_dim=64, nhead=4, activation=F.relu,
                 pair_feature_dim=0, device=None, dtype=None):
        del nhead
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(activation, str):
            activation = activations[activation]
        self.embed_dim = int(embed_dim)
        self.pair_feature_dim = max(0, int(pair_feature_dim))
        self.query_proj = Embedding_layer(
            query_dim, self.embed_dim, 2,
            activation=activation, **factory_kwargs,
        )
        activation_layer = nn.ReLU if activation is F.relu else nn.Tanh
        self.pair_feature_proj = (
            Embedding_layer(
                self.pair_feature_dim,
                self.embed_dim,
                2,
                activation=activation,
                **factory_kwargs,
            )
            if self.pair_feature_dim > 0 else None
        )
        score_blocks = 6 + int(self.pair_feature_proj is not None)
        self.pair_score = nn.Sequential(
            nn.Linear(
                self.embed_dim * score_blocks,
                self.embed_dim * 2,
                **factory_kwargs,
            ),
            activation_layer(),
            nn.Linear(
                self.embed_dim * 2, self.embed_dim, **factory_kwargs,
            ),
            activation_layer(),
            nn.Linear(self.embed_dim, 1, **factory_kwargs),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)

    def forward(self, query, op_nodes, site_nodes, op_valid_mask,
                site_valid_mask, deterministic=False, chosen_op=None,
                chosen_site=None, tau=1.0, pair_features=None,
                pair_feature_values=None,
                pair_feature_batch_indices=None,
                pair_feature_flat_ids=None):
        batch_size = query.shape[0]
        op_count = op_nodes.shape[1]
        site_count = site_nodes.shape[1]
        if site_valid_mask.dim() == 2:
            site_valid_mask = site_valid_mask.unsqueeze(1).expand(
                -1, op_count, -1
            )
        expected = (batch_size, op_count, site_count)
        if site_valid_mask.shape != expected:
            raise ValueError(
                f"site_valid_mask must have shape {expected}, "
                f"got {tuple(site_valid_mask.shape)}."
            )
        joint_valid = op_valid_mask.unsqueeze(-1) & site_valid_mask
        flat_valid = joint_valid.reshape(batch_size, -1)
        if (~flat_valid.any(dim=-1)).any():
            bad_rows = torch.nonzero(
                ~flat_valid.any(dim=-1), as_tuple=False
            ).flatten().tolist()
            raise RuntimeError(
                f"Joint pair actor has no legal action for rows {bad_rows}."
            )

        sparse_inputs = (
            pair_feature_values,
            pair_feature_batch_indices,
            pair_feature_flat_ids,
        )
        sparse_mode = any(value is not None for value in sparse_inputs)
        if sparse_mode and not all(value is not None for value in sparse_inputs):
            raise ValueError(
                'Sparse pair features require values, batch indices and flat ids.'
            )
        if sparse_mode and pair_features is not None:
            raise ValueError(
                'Dense and sparse pair features cannot be supplied together.'
            )

        q = self.query_proj(query).squeeze(1)
        if sparse_mode:
            if self.pair_feature_proj is None:
                raise ValueError(
                    'Sparse pair features require pair_feature_dim > 0.'
                )
            sparse_values = pair_feature_values
            sparse_batches = pair_feature_batch_indices.long().view(-1)
            sparse_flat_ids = pair_feature_flat_ids.long().view(-1)
            if (
                sparse_values.dim() != 2
                or sparse_values.shape[1] != self.pair_feature_dim
                or sparse_values.shape[0] != sparse_batches.numel()
                or sparse_values.shape[0] != sparse_flat_ids.numel()
            ):
                raise ValueError(
                    'Sparse pair feature tensors have inconsistent shapes: '
                    f'values={tuple(sparse_values.shape)}, '
                    f'batches={tuple(sparse_batches.shape)}, '
                    f'flat_ids={tuple(sparse_flat_ids.shape)}.'
                )
            pair_count = op_count * site_count
            in_bounds = (
                (sparse_batches >= 0)
                & (sparse_batches < batch_size)
                & (sparse_flat_ids >= 0)
                & (sparse_flat_ids < pair_count)
            )
            if not in_bounds.all():
                raise ValueError('Sparse pair feature coordinates are out of range.')

            # Autoregressive plane decisions can only remove choices from the
            # environment mask.  Drop those newly masked entries, then require
            # exact coverage of every pair that remains legal for this call.
            still_legal = flat_valid[sparse_batches, sparse_flat_ids]
            sparse_values = sparse_values[still_legal]
            sparse_batches = sparse_batches[still_legal]
            sparse_flat_ids = sparse_flat_ids[still_legal]
            coverage = torch.zeros_like(flat_valid)
            coverage[sparse_batches, sparse_flat_ids] = True
            missing = flat_valid & ~coverage
            if missing.any():
                missing_rows = torch.nonzero(
                    missing.any(dim=-1), as_tuple=False
                ).flatten().tolist()
                raise RuntimeError(
                    'Sparse pair features do not cover every legal action for '
                    f'rows {missing_rows}.'
                )

            sparse_ops = sparse_flat_ids // site_count
            sparse_sites = sparse_flat_ids % site_count
            q_sparse = q[sparse_batches]
            op_sparse = op_nodes[sparse_batches, sparse_ops]
            site_sparse = site_nodes[sparse_batches, sparse_sites]
            sparse_latent = torch.cat(
                (
                    q_sparse,
                    op_sparse,
                    site_sparse,
                    op_sparse * site_sparse,
                    q_sparse * op_sparse,
                    q_sparse * site_sparse,
                    self.pair_feature_proj(torch.nan_to_num(
                        sparse_values,
                        nan=0.0,
                        posinf=10.0,
                        neginf=-10.0,
                    )),
                ),
                dim=-1,
            )
            sparse_logits = self.pair_score(sparse_latent).squeeze(-1)
            logits = torch.full(
                (batch_size, pair_count),
                float('-inf'),
                dtype=sparse_logits.dtype,
                device=sparse_logits.device,
            )
            logits[sparse_batches, sparse_flat_ids] = sparse_logits
        else:
            q_grid = q[:, None, None, :].expand(
                -1, op_count, site_count, -1
            )
            op_grid = op_nodes[:, :, None, :].expand(
                -1, -1, site_count, -1
            )
            site_grid = site_nodes[:, None, :, :].expand(
                -1, op_count, -1, -1
            )
            pair_latent = torch.cat(
                (
                    q_grid,
                    op_grid,
                    site_grid,
                    op_grid * site_grid,
                    q_grid * op_grid,
                    q_grid * site_grid,
                ),
                dim=-1,
            )
            if self.pair_feature_proj is not None:
                expected_pair_shape = (
                    batch_size, op_count, site_count, self.pair_feature_dim
                )
                if (
                    pair_features is None
                    or pair_features.shape != expected_pair_shape
                ):
                    actual = (
                        None if pair_features is None
                        else tuple(pair_features.shape)
                    )
                    raise ValueError(
                        f"pair_features must have shape {expected_pair_shape}, "
                        f"got {actual}."
                    )
                pair_features = torch.nan_to_num(
                    pair_features, nan=0.0, posinf=10.0, neginf=-10.0
                )
                pair_latent = torch.cat(
                    (pair_latent, self.pair_feature_proj(pair_features)),
                    dim=-1,
                )
            logits = self.pair_score(pair_latent).squeeze(-1)
            logits = logits.reshape(batch_size, -1)
        canonical_raw = None
        if getattr(self, 'canonical_h_decode', False):
            if not deterministic or chosen_op is not None or chosen_site is not None or not (0 < float(tau) < float('inf')):
                raise ValueError('Canonical H is restricted to positive-temperature deterministic evaluation')
            canonical_raw = _safe_logits(logits, valid_mask=flat_valid)
            logits = logits.to(torch.float64)
        logits = logits / (float(tau) if canonical_raw is not None else max(float(tau), 1e-6))
        logits = _safe_logits(logits, valid_mask=flat_valid)
        log_probs = F.log_softmax(logits, dim=-1)

        if chosen_op is not None and chosen_site is not None:
            op_idx = chosen_op.long()
            site_idx = chosen_site.long()
            in_bounds = (
                (op_idx >= 0) & (op_idx < op_count)
                & (site_idx >= 0) & (site_idx < site_count)
            )
            if not in_bounds.all():
                raise RuntimeError(
                    "PPO replay contains an out-of-range joint pair."
                )
            batch_indices = torch.arange(
                batch_size, device=query.device
            )
            if not joint_valid[
                batch_indices, op_idx, site_idx
            ].all():
                raise RuntimeError(
                    "PPO replay contains a masked joint pair."
                )
            flat_idx = op_idx * site_count + site_idx
        else:
            flat_idx = _select_masked_index(
                canonical_raw if canonical_raw is not None else log_probs, flat_valid, deterministic
            )
            op_idx = flat_idx // site_count
            site_idx = flat_idx % site_count
        selected_log_prob = log_probs.gather(
            1, flat_idx.unsqueeze(-1)
        ).squeeze(-1)
        if canonical_raw is not None:
            selected_log_prob = selected_log_prob.to(query.dtype)
            log_probs = log_probs.to(query.dtype)
        return op_idx, site_idx, selected_log_prob, log_probs


class CascadePtrActor(Module):
    def __init__(self, query_dim, embed_dim=64, nhead=4, activation=F.relu, device=None, dtype=None) -> None:
        """
        两级级联指针网络 (Actor)。
        Level 1: 选工序 (Operation)
        Level 2: 结合选中的工序特征，选机位 (Site)
        """
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(activation, str):
            activation = activations[activation]
        self.activation = activation
        self.embed_dim = embed_dim
        
        # Query 预处理层 (融合全局和局部特征)
        self.op_query_ff = Embedding_layer(query_dim, self.embed_dim, 2, activation=self.activation, **self.factory_kwargs)
        self.op_query_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0., batch_first=True, **self.factory_kwargs)
        self.op_query_norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)

        # 级联的第一级：工序选择指针网络
        self.op_ptr_net = MaPtrNet(query_dim=embed_dim, embed_dim=self.embed_dim, **self.factory_kwargs)

        # 级联过渡层：融合 Query 和候选工序 Embedding。
        self.site_query_ff = Embedding_layer(query_dim, self.embed_dim, 2, activation=self.activation, **self.factory_kwargs)
        self.site_query_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0., batch_first=True, **self.factory_kwargs)
        self.site_query_norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)

        # 级联的第二级：机位选择指针网络
        self.site_ptr_net = MaPtrNet(query_dim=embed_dim, embed_dim=self.embed_dim, **self.factory_kwargs)

        self._reset_parameters()

        # 新条件分支必须既保持零输出，又不能推进全局 RNG；否则即使权重清零，
        # 后续旧模块的随机初始化也会整体偏移，破坏可比较的 pre-PPO baseline。
        with torch.random.fork_rng(devices=[], enabled=True):
            self.site_op_condition = nn.Linear(
                self.embed_dim,
                self.embed_dim,
                bias=False,
                dtype=dtype,
            )
            nn.init.zeros_(self.site_op_condition.weight)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, query, op_nodes, site_nodes, op_valid_mask,
                site_valid_mask, deterministic: bool = False, chosen_op=None,
                chosen_site=None, tau=1.0, pair_features=None):
        """
        条件联合架构 (Operation-conditioned Joint Architecture)。
        工序头计算 P(Op|s)，机位头用每个候选工序条件化的 query 打分机位，
        最后在所有合法 operation-site 对上构建唯一、完全归一化的联合分布。
        """
        del pair_features
        B = query.shape[0]
        N_ops = op_nodes.shape[1]
        N_sites = site_nodes.shape[1]
        batch_indices = torch.arange(B, device=query.device)

        # =======================================================
        # Head 1: 独立评估所有工序 P(Op)
        # =======================================================
        if site_valid_mask.dim() == 2:
            site_valid_mask = site_valid_mask.unsqueeze(1).expand(-1, N_ops, -1)
        if site_valid_mask.shape != (B, N_ops, N_sites):
            raise ValueError(
                "site_valid_mask must have shape "
                f"[{B}, {N_ops}, {N_sites}], got {tuple(site_valid_mask.shape)}."
            )

        joint_valid_mask = op_valid_mask.unsqueeze(-1) & site_valid_mask
        if (~joint_valid_mask.view(B, -1).any(dim=-1)).any():
            bad_rows = torch.nonzero(
                ~joint_valid_mask.view(B, -1).any(dim=-1),
                as_tuple=False,
            ).flatten().tolist()
            raise RuntimeError(f"Plane actor has no legal operation-site pair for rows {bad_rows}.")

        effective_op_mask = joint_valid_mask.any(dim=-1)
        effective_site_mask = joint_valid_mask.any(dim=1)
        op_pad_mask = ~effective_op_mask

        op_q = self.op_query_ff(query)
        op_attn_out, _ = self.op_query_attn(op_q, op_nodes, op_nodes, key_padding_mask=op_pad_mask)
        op_q = self.op_query_norm(op_q + op_attn_out)

        # 得到工序的 1D 概率分布 [B, N_ops]
        op_prob_v = self.op_ptr_net.dist(op_q, op_nodes, key_padding_mask=op_pad_mask, tau=tau).squeeze(1)
        op_log_prob = _masked_log_probs(op_prob_v, effective_op_mask)

        # =======================================================
        # Head 2: 为每个候选工序构造条件化机位 query。
        # site_op_condition 零初始化，因此新模块在首次更新前不改变旧策略排序。
        # =======================================================
        site_pad_mask = ~effective_site_mask

        site_q = self.site_query_ff(query).expand(-1, N_ops, -1)
        site_q = site_q + self.site_op_condition(op_nodes)
        
        site_attn_out, _ = self.site_query_attn(site_q, site_nodes, site_nodes, key_padding_mask=site_pad_mask)
        site_q = self.site_query_norm(site_q + site_attn_out)

        # 得到操作条件化的机位分数 [B, N_ops, N_sites]。此处仅使用全局
        # site padding mask；具体 operation-site 可行性在联合网格中统一屏蔽和归一化。
        site_prob_v = self.site_ptr_net.dist(
            site_q,
            site_nodes,
            key_padding_mask=site_pad_mask,
            tau=tau,
        )
        site_score_mask = effective_site_mask.unsqueeze(1).expand(-1, N_ops, -1)
        site_log_prob = _masked_log_probs(site_prob_v, site_score_mask)

        # =======================================================
        # 联合网格构建与展平采样 (Joint Grid & Sampling)
        # =======================================================
        # 利用 PyTorch 的广播机制 (Broadcasting) 直接生成正交矩阵
        # op_log_prob:  [B, N_ops]   -> 变形成 [B, N_ops, 1]
        # site_log_prob: [B, N_ops, N_sites]
        joint_log_prob = op_log_prob.unsqueeze(-1) + site_log_prob
        
        # ++++++++++ 【核心修复区：彻底抹杀幽灵概率】 ++++++++++
        # 构建二维联合掩码：只要 op 或 site 任意一个是非法的，联合动作即为非法 (True 代表非法)
        joint_pad_mask = ~joint_valid_mask
        
        # 强制将所有非法联合动作的 Logit 设为绝对的负无穷
        joint_log_prob = joint_log_prob.masked_fill(joint_pad_mask, float('-inf'))
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 将二维联合空间拍平
        joint_logits_flat = joint_log_prob.view(B, -1) # Shape: [B, N_ops * N_sites]
        joint_valid_mask_flat = (~joint_pad_mask).view(B, -1)
        joint_logits_flat = _safe_logits(joint_logits_flat, valid_mask=joint_valid_mask_flat)

        # PPO 必须保存与 Categorical 实际采样相同、在合法动作对上归一化的 log-prob。
        joint_log_prob_flat = F.log_softmax(joint_logits_flat, dim=-1)

        if chosen_op is not None and chosen_site is not None:
            # RL 算 Loss 阶段：直接查表
            op_idx = chosen_op.long()
            site_idx = chosen_site.long()
            in_bounds = (
                (op_idx >= 0) & (op_idx < N_ops)
                & (site_idx >= 0) & (site_idx < N_sites)
            )
            if not in_bounds.all():
                raise RuntimeError("PPO replay contains an out-of-range plane action.")
            if not joint_valid_mask[batch_indices, op_idx, site_idx].all():
                raise RuntimeError("PPO replay contains a plane action rejected by the current mask.")
            flat_idx = op_idx * N_sites + site_idx
        else:
            # RL 采样阶段
            flat_idx = _select_masked_index(
                joint_log_prob_flat,
                joint_valid_mask_flat,
                deterministic,
            )
            
            # 从 1D 索引中解码出工序和机位
            op_idx = flat_idx // N_sites
            site_idx = flat_idx % N_sites

        # 直接返回真实采样分布的 log-prob，避免 exp -> clamp -> log 丢失精度。
        joint_log_prob_selected = joint_log_prob_flat[batch_indices, flat_idx]

        # 返回完全归一化的联合 log-prob，供 entropy/KL/测试共用。
        normalized_joint_log_probs = joint_log_prob_flat

        return op_idx, site_idx, joint_log_prob_selected, normalized_joint_log_probs


class DeviceRequestPtrActor(Module):
    def __init__(self, query_dim, embed_dim=64, nhead=4, activation=F.relu,
                 timing_head=False, device=None, dtype=None) -> None:
        """
        设备派遣指针网络。
        对每个空闲移动设备，从 request 节点中选择一个服务请求；request 0 固定为 no-op。
        """
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(activation, str):
            activation = activations[activation]
        self.activation = activation
        self.embed_dim = embed_dim

        self.req_query_ff = Embedding_layer(query_dim, self.embed_dim, 2, activation=self.activation, **self.factory_kwargs)
        self.req_query_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0., batch_first=True, **self.factory_kwargs)
        self.req_query_norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)
        self.req_ptr_net = MaPtrNet(query_dim=embed_dim, embed_dim=self.embed_dim, **self.factory_kwargs)
        # The pointer score answers "which request".  Stage2 additionally
        # needs an explicit, low-dimensional answer to "dispatch now or
        # defer".  Keeping this gate optional preserves exact compatibility
        # with every historical checkpoint and lets the structural screen
        # isolate timing factorisation from the ranking loss.
        self.timing_head = (
            nn.Linear(self.embed_dim, 1, **self.factory_kwargs)
            if bool(timing_head) else None
        )

        self._reset_parameters()
        if self.timing_head is not None:
            nn.init.zeros_(self.timing_head.weight)
            nn.init.zeros_(self.timing_head.bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, request_nodes, request_valid_mask, deterministic: bool = False,
                chosen_request=None, tau=1.0):
        B = query.shape[0]
        batch_indices = torch.arange(B, device=query.device)

        if (~request_valid_mask.any(dim=-1)).any():
            bad_rows = torch.nonzero(
                ~request_valid_mask.any(dim=-1),
                as_tuple=False,
            ).flatten().tolist()
            raise RuntimeError(f"Device actor has no legal request for rows {bad_rows}.")
        req_pad_mask = ~request_valid_mask

        req_q = self.req_query_ff(query)
        req_attn_out, _ = self.req_query_attn(req_q, request_nodes, request_nodes, key_padding_mask=req_pad_mask)
        req_q = self.req_query_norm(req_q + req_attn_out)

        canonical = bool(getattr(self, 'canonical_h_decode', False))
        if canonical:
            if not deterministic or chosen_request is not None or self.timing_head is not None or not (0 < float(tau) < float('inf')):
                raise ValueError('Canonical H excludes training, replay and timing gates')
            raw = self.req_ptr_net.dist(req_q, request_nodes, key_padding_mask=req_pad_mask,
                                        tau=tau, return_raw=True).squeeze(1)
            self._canonical_h_raw = raw
            req_logits = F.log_softmax(raw.to(torch.float64)/float(tau), dim=-1).to(query.dtype)
        else:
            req_prob_v = self.req_ptr_net.dist(req_q, request_nodes, key_padding_mask=req_pad_mask, tau=tau).squeeze(1)
            req_logits = _masked_log_probs(req_prob_v, request_valid_mask)
        if self.timing_head is not None:
            dispatch_gate = self.timing_head(req_q.squeeze(1)).squeeze(-1)
            # A positive gate shifts mass from no-op to every legal real
            # request without changing their relative pointer ranking.  The
            # final log-softmax makes the returned tensor a normalized action
            # distribution, exactly like the legacy path.
            adjusted = req_logits.clone()
            adjusted[:, 0] = adjusted[:, 0] - 0.5 * dispatch_gate
            adjusted[:, 1:] = adjusted[:, 1:] + 0.5 * dispatch_gate.unsqueeze(-1)
            adjusted = adjusted.masked_fill(
                ~request_valid_mask, float('-inf')
            )
            req_logits = torch.log_softmax(adjusted, dim=-1)
        req_logits = _safe_logits(req_logits, valid_mask=~req_pad_mask)

        if chosen_request is not None:
            req_idx = chosen_request.long()
            in_bounds = (req_idx >= 0) & (req_idx < request_valid_mask.shape[-1])
            if not in_bounds.all():
                raise RuntimeError("PPO replay contains an out-of-range device request.")
            if not request_valid_mask[batch_indices, req_idx].all():
                raise RuntimeError("PPO replay contains a device request rejected by the current mask.")
        else:
            req_idx = _select_masked_index(
                raw if canonical else req_logits,
                request_valid_mask,
                deterministic,
            )

        req_log_prob_selected = req_logits[batch_indices, req_idx]
        return req_idx, req_log_prob_selected, req_logits
