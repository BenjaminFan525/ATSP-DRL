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
        
        self.query_ff = Embedding_layer(query_dim, self.embed_dim, 2, activation=self.activation, **self.factory_kwargs)
        self.query_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0., batch_first=True, **self.factory_kwargs)
        self.query_norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)

        self.critic_ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, **self.factory_kwargs),
            nn.ReLU(),
            nn.Linear(embed_dim, 32, **self.factory_kwargs),
            nn.ReLU(),
            nn.Linear(32, 1, **self.factory_kwargs)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, query, node, key_padding_mask=None) -> torch.Tensor:
        """
        node: encoder output node embedding, B x N x d
        query: the query, B x 1 x D
        key_padding_mask: specify which agts have been arranged for each batch, B x N
        """
        query = self.query_ff(query)
        query = self.query_norm(query + self.query_attn(query, node, node, key_padding_mask)[0])

        return self.critic_ff(query)