import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from onpolicy.algorithms.utils.util import activations, Embedding_layer


class GlobalStepCritic(Module):
    """One centralized value head for a simultaneous scheduling state."""

    def __init__(self, embed_dim=64, device=None, dtype=None, **_kwargs):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.value = nn.Sequential(
            nn.LayerNorm(embed_dim, **factory_kwargs),
            nn.Linear(embed_dim, embed_dim, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(embed_dim, 32, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(32, 1, **factory_kwargs),
        )
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, global_embedding):
        return torch.nan_to_num(
            self.value(global_embedding),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )


def _sanitize_attn_mask(mask, seq_len):
    if mask is None:
        return None
    mask = mask.bool()
    dead_ends = mask.all(dim=-1)
    if dead_ends.any() and seq_len > 0:
        mask = mask.clone()
        mask[dead_ends, 0] = False
    return mask

class StepCritic(Module):
    def __init__(self, query_dim=192, embed_dim=64, nhead=4, activation=F.relu, device=None, dtype=None) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(activation, str):
            activation = activations[activation]
        self.activation = activation
        self.embed_dim = embed_dim
        
        # 1. Query 投影层 (融合大盘面与GRU记忆)
        self.query_ff = Embedding_layer(query_dim, self.embed_dim, 2, activation=self.activation, **self.factory_kwargs)
        
        # 2. 任务感知头 (Operation Attention)：看向工序节点
        self.op_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0., batch_first=True, **self.factory_kwargs)
        self.op_norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)

        # 3. 资源感知头 (Site Attention)：看向机位节点
        self.site_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0., batch_first=True, **self.factory_kwargs)
        self.site_norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)

        # 4. 价值评估网络 (Value Head)
        # 接收 op 和 site 两个头的拼接特征，所以输入维度是 2 * embed_dim
        self.critic_ff = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim, **self.factory_kwargs),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 32, **self.factory_kwargs),
            nn.ReLU(),
            nn.Linear(32, 1, **self.factory_kwargs)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, query, op_nodes, site_nodes, op_pad_mask=None, site_pad_mask=None) -> torch.Tensor:
        """
        计算状态价值 V(s)
        
        Inputs:
            query: 融合了全局特征与局部记忆的初始张量 [B, 1, query_dim]
            op_nodes: 工序节点特征库 [B, N_ops, embed_dim]
            site_nodes: 机位节点特征库 [B, N_sites, embed_dim]
            op_pad_mask: (可选) 屏蔽无关工序的掩码 [B, N_ops]
            site_pad_mask: (可选) 屏蔽无效机位的掩码 [B, N_sites]
        """
        # 将高维 Query 投影到统一维度
        query = torch.nan_to_num(query, nan=0.0, posinf=1e4, neginf=-1e4)
        op_nodes = torch.nan_to_num(op_nodes, nan=0.0, posinf=1e4, neginf=-1e4)
        site_nodes = torch.nan_to_num(site_nodes, nan=0.0, posinf=1e4, neginf=-1e4)
        op_pad_mask = _sanitize_attn_mask(op_pad_mask, op_nodes.shape[1])
        site_pad_mask = _sanitize_attn_mask(site_pad_mask, site_nodes.shape[1])

        query = torch.nan_to_num(self.query_ff(query), nan=0.0, posinf=1e4, neginf=-1e4)
        
        # --- 分支 1：评估任务处境 (我还有多重的任务负担？) ---
        # 利用 op_pad_mask，只关注当前飞机需要做的工序
        op_attn_out, _ = self.op_attn(query, op_nodes, op_nodes, key_padding_mask=op_pad_mask)
        op_attn_out = torch.nan_to_num(op_attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
        q_op = self.op_norm(query + op_attn_out) # [B, 1, embed_dim]
        q_op = torch.nan_to_num(q_op, nan=0.0, posinf=1e4, neginf=-1e4)

        # --- 分支 2：评估资源拥挤度 (机场现在的空间资源够不够？) ---
        site_attn_out, _ = self.site_attn(query, site_nodes, site_nodes, key_padding_mask=site_pad_mask)
        site_attn_out = torch.nan_to_num(site_attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
        q_site = self.site_norm(query + site_attn_out) # [B, 1, embed_dim]
        q_site = torch.nan_to_num(q_site, nan=0.0, posinf=1e4, neginf=-1e4)

        # --- 融合评估 ---
        # 将“任务压力”和“资源充裕度”拼接在一起打分
        v_input = torch.cat([q_op, q_site], dim=-1) # [B, 1, 2 * embed_dim]
        v_input = torch.nan_to_num(v_input, nan=0.0, posinf=1e4, neginf=-1e4)
        
        value = self.critic_ff(v_input)
        return torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)
