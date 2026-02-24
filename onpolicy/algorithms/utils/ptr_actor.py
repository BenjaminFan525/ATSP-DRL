import torch
import torch.nn as nn
from torch.nn import Module
from copy import deepcopy
import torch.nn.functional as F
from onpolicy.algorithms.utils.util import FeatureBlock, activations, Embedding_layer, PositionalEncoding
from typing import Optional, Tuple
from torch.nn.utils.rnn import pad_sequence

class MaPtrNet(Module):
    def __init__(self, query_dim, embed_dim, bias=True, device=None, dtype=None) -> None:
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

    def dist(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            tau: float = 1,
            ):
        key_padding_mask = torch.zeros((key.shape[0], key.shape[1] + 1), dtype=torch.bool, device=key.device) if key_padding_mask is None else key_padding_mask
        masked_ptr = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        
        q = self.q_proj_weight(query)
        k = self.k_proj_weight(key)

        ptr = torch.tanh(torch.bmm(q, k.transpose(2, 1)) / torch.sqrt(torch.tensor(self.embed_dim)))
        
        masked_ptr = masked_ptr.unsqueeze(1)
        key_padding_mask = key_padding_mask.unsqueeze(1)
        masked_ptr[~key_padding_mask] = ptr[~key_padding_mask]
        return F.softmax(masked_ptr / tau, dim=-1)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            deterministic: bool = False,
            tau = 1,
            idx = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        prob_v = self.dist(query, key, key_padding_mask)
        if idx is not None:
            index = idx.unsqueeze(1)
            prob = torch.gather(prob_v, dim=-1, index=index)
        elif deterministic:
            prob, index = prob_v.max(-1, keepdim=True)
        else:
            choice_soft = F.gumbel_softmax(torch.log(prob_v), tau=tau)
            index = choice_soft.max(-1, keepdim=True)[1]
            prob = torch.gather(prob_v, dim=-1, index=index)
        choice_hard = torch.zeros_like(prob_v, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        choice = choice_hard # - choice_soft.detach() + choice_soft

        return choice, index.squeeze(-1), prob.squeeze(-1)

class PtrEntryActor(Module):
    def __init__(self, query_dim=192, embed_dim=64, nhead=4, activation=F.relu, device=None, dtype=None) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(activation, str):
            activation = activations[activation]
        self.activation = activation
        self.embed_dim = embed_dim
        
        self.query_ff = Embedding_layer(query_dim, self.embed_dim, 2, activation=self.activation, **self.factory_kwargs)
        self.query_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0., batch_first=True, **self.factory_kwargs)
        self.query_norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)

        self.ptr_net = MaPtrNet(query_dim=embed_dim, embed_dim=self.embed_dim, **self.factory_kwargs)

        self.entry_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0., batch_first=True, **self.factory_kwargs)
        self.entry_norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)
        self.entry_ff = nn.Sequential(
            nn.Linear(2 * embed_dim, 32, **self.factory_kwargs),
            nn.ReLU(),
            nn.Linear(32, 2, **self.factory_kwargs)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, query, agt, key_padding_mask=None, deterministic: bool = False, tau=1, chosen_idx=None, chosen_entry=None, force_chosen_mask=None):
        """
        Choosing the next node and its corresponding entrance.

        Input:
            agt: encoder output agent embedding, B x N x d
            query: the query, B x 1 x D
            key_padding_mask: specify which agts have been arranged for each batch, B x N
            deterministic: use the most probably choice or sample from the distrbution.
            tau: the temperature factor in gumbel soft max.
            chosen_idx: the chosen index, to get the new prob.
            chosen_entry: the chosen entry, to get the new prob.
            force_chosen_mask: if an element is True, it's corresponding chosen_idx and chosen_entry will be chosen with p(a|s) = 1, B. 
        Output:
            chosen_agt: chosen agts' encoding sampled from agt, concated with the 2d 1-hot vector of entrance, B x 1 x (d + 2).
            choice: the one-hot choice.
            index: chosen index, B x 1.
            chosen_entry: chosen entrance, B x 1.
            prob: probability of the choice, B x 1.
        """
        query = self.query_ff(query)
        query = self.query_norm(query + self.query_attn(query, agt, agt, key_padding_mask)[0])
        
        # 1. Choose the node and corresponding probability p(n|s). 
        node_prob_v = self.ptr_net.dist(query, agt, key_padding_mask, tau)
        agt_embed = self.entry_norm(agt + self.entry_attn(agt, agt, agt, key_padding_mask)[0])
        entry_feature = torch.cat([query.repeat(1, agt.shape[1], 1), agt_embed], dim=-1)
        
        # 2. Choose the corresponding entrance.
        entry_prob_v = F.softmax(self.entry_ff(entry_feature) / tau, dim=-1).squeeze(1)
        prob_v = node_prob_v.transpose(1, 2).repeat(1, 1, 2) * entry_prob_v
        prob_v = prob_v.view((query.shape[0], -1))
        if force_chosen_mask is not None:
            assert chosen_idx is not None and chosen_entry is not None, "chosen_idx and chosen_entry should be given if force_chosen_mask is not None."
            prob_v[force_chosen_mask] = torch.zeros_like(prob_v).scatter_(-1, chosen_idx * 2 + chosen_entry, 1.0)[force_chosen_mask]
        dist = torch.distributions.Categorical(prob_v)
        if chosen_idx is not None and force_chosen_mask is None:
            prob = torch.gather(prob_v, -1, chosen_idx * 2 + chosen_entry)
            index = chosen_idx
        else:
            if deterministic:
                prob, idx = torch.max(prob_v, dim=-1)
                prob = prob.view(-1, 1)
                idx = idx.view(-1, 1)
            else:
                idx = dist.sample([1]).view(-1, 1)
                prob = torch.gather(prob_v, dim=-1, index=idx)
            index = idx // 2
            chosen_entry = idx % 2
        choice = torch.zeros_like(node_prob_v, memory_format=torch.legacy_contiguous_format).scatter_(-1, index.unsqueeze(-1), 1.0)

        # 3. Concate the chosen nodes' encoding and the corredponding entrance one-hot vector.
        entry_choice = torch.zeros((query.shape[0], 2), device=query.device).scatter_(-1, chosen_entry, 1.0).unsqueeze(1)

        return choice, index, chosen_entry, prob, prob_v
