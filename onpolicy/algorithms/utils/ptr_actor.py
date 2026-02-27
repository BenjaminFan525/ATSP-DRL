import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from typing import Optional, Tuple
from onpolicy.algorithms.utils.util import activations, Embedding_layer

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

    def dist(self, query: torch.Tensor, key: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, tau: float = 1.0):
        """
        输出经过 Mask 屏蔽后的概率分布。
        注意：PyTorch 标准中，key_padding_mask 为 True 代表是被屏蔽的非法位置 (Padding)。
        """
        q = self.q_proj_weight(query)
        k = self.k_proj_weight(key)

        # 缩放点积注意力 (Scaled Dot-Product Attention)
        ptr = torch.tanh(torch.bmm(q, k.transpose(2, 1)) / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32)))
        
        # 掩码处理：将非法节点的 Logits 设为负无穷
        if key_padding_mask is not None:
            # 确保 mask 的维度是 [Batch, 1, Seq_len] 以便进行 broadcast
            if key_padding_mask.dim() == 2:
                key_padding_mask = key_padding_mask.unsqueeze(1)
            ptr = ptr.masked_fill(key_padding_mask, float('-inf'))
            
        return F.softmax(ptr / tau, dim=-1)


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

        # 级联过渡层：融合 Query 和 第一级选中的工序 Embedding
        self.site_query_ff = Embedding_layer(query_dim + self.embed_dim, self.embed_dim, 2, activation=self.activation, **self.factory_kwargs)
        self.site_query_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0., batch_first=True, **self.factory_kwargs)
        self.site_query_norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)

        # 级联的第二级：机位选择指针网络
        self.site_ptr_net = MaPtrNet(query_dim=embed_dim, embed_dim=self.embed_dim, **self.factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, query, op_nodes, site_nodes, op_valid_mask, site_mask_matrix, 
                deterministic: bool = False, chosen_op=None, chosen_site=None, tau=1.0):
        """
        两级级联前向传播
        """
        B = query.shape[0]

        # =======================================================
        # Level 1: 预处理与工序选择 (Operation)
        # =======================================================
        # PyTorch 的 Padding Mask 习惯：True 代表要被屏蔽(屏蔽掉非法工序)
        op_pad_mask = ~op_valid_mask 

        # 1. 第一级 Query 特征抽取与交叉注意力
        op_q = self.op_query_ff(query)
        op_attn_out, _ = self.op_query_attn(op_q, op_nodes, op_nodes, key_padding_mask=op_pad_mask)
        op_q = self.op_query_norm(op_q + op_attn_out)

        # 2. 第一级指针网络打分
        op_prob_v = self.op_ptr_net.dist(op_q, op_nodes, key_padding_mask=op_pad_mask, tau=tau).squeeze(1)

        # 3. 确定工序动作
        if chosen_op is not None:
            op_idx = chosen_op
        else:
            if deterministic:
                op_idx = torch.argmax(op_prob_v, dim=-1)
            else:
                op_dist = torch.distributions.Categorical(op_prob_v)
                op_idx = op_dist.sample()

        # 提取选中工序的概率
        batch_indices = torch.arange(B, device=query.device)
        op_prob = op_prob_v[batch_indices, op_idx]

        # =======================================================
        # Level 2: 预处理与机位选择 (Site)
        # =======================================================
        # 1. 提取选中工序的高阶特征，并与原始 Query 拼接
        chosen_op_emb = op_nodes[batch_indices, op_idx, :].unsqueeze(1) # [B, 1, embed_dim]
        # 注意：这里的拼接维度是 query_dim + embed_dim
        site_q_input = torch.cat([query, chosen_op_emb], dim=-1) 

        # 2. 初始机位查询向量生成
        site_q = self.site_query_ff(site_q_input)

        # 3. 提取“特定于该工序”的机位合法掩码
        cur_site_valid_mask = site_mask_matrix[batch_indices, op_idx, :] # [B, N_sites]
        site_pad_mask = ~cur_site_valid_mask

        # 4. 新增的亮点：机位意图交叉注意力 (Site Cross-Attention)
        # 让 site_q 提前关注那些合法机位的状态（拥挤度、距离等）
        site_attn_out, _ = self.site_query_attn(site_q, site_nodes, site_nodes, key_padding_mask=site_pad_mask)
        site_q = self.site_query_norm(site_q + site_attn_out)

        # 5. 第二级指针网络打分
        site_prob_v = self.site_ptr_net.dist(site_q, site_nodes, key_padding_mask=site_pad_mask, tau=tau).squeeze(1)

        # 6. 确定机位动作
        if chosen_site is not None:
            site_idx = chosen_site
        else:
            if deterministic:
                site_idx = torch.argmax(site_prob_v, dim=-1)
            else:
                site_dist = torch.distributions.Categorical(site_prob_v)
                site_idx = site_dist.sample()

        # 提取选中机位的概率
        site_prob = site_prob_v[batch_indices, site_idx]

        # =======================================================
        # 联合输出 (Joint Output)
        # =======================================================
        # 联合概率 P = P(工序) * P(机位 | 工序)
        joint_prob = op_prob * site_prob 
        
        # 将两个分布打包返回，便于 PPO 算 Entropy 和 Loss
        joint_dist = (op_prob_v, site_prob_v)

        return op_idx, site_idx, joint_prob, joint_dist