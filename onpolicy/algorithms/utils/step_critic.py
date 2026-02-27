import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from onpolicy.algorithms.utils.util import activations, Embedding_layer

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
        query = self.query_ff(query)
        
        # --- 分支 1：评估任务处境 (我还有多重的任务负担？) ---
        # 利用 op_pad_mask，只关注当前飞机需要做的工序
        op_attn_out, _ = self.op_attn(query, op_nodes, op_nodes, key_padding_mask=op_pad_mask)
        q_op = self.op_norm(query + op_attn_out) # [B, 1, embed_dim]

        # --- 分支 2：评估资源拥挤度 (机场现在的空间资源够不够？) ---
        site_attn_out, _ = self.site_attn(query, site_nodes, site_nodes, key_padding_mask=site_pad_mask)
        q_site = self.site_norm(query + site_attn_out) # [B, 1, embed_dim]

        # --- 融合评估 ---
        # 将“任务压力”和“资源充裕度”拼接在一起打分
        v_input = torch.cat([q_op, q_site], dim=-1) # [B, 1, 2 * embed_dim]
        
        return self.critic_ff(v_input)