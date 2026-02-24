import torch
import torch.nn as nn
from torch.nn import Module
from copy import deepcopy
import torch.nn.functional as F
from onpolicy.algorithms.utils.util import FeatureBlock, activations, Embedding_layer, PositionalEncoding
from typing import Optional
from torch.nn.utils.rnn import pad_sequence

class SelectionEncoder(Module):
    def __init__(self, input_dim=66, embed_dim=64, activation=F.relu, device=None, dtype=None) -> None:
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        if isinstance(activation, str):
            activation = activations[activation]

        self.embedding = Embedding_layer(input_dim, embed_dim, 2, 
                                        activation=activation, **self.factory_kwargs)
        self.seq_encoder = nn.GRU(embed_dim, embed_dim, 1, batch_first=True, **self.factory_kwargs)

        self.seq_ff = Embedding_layer(embed_dim, embed_dim, 2, activation=activation, **self.factory_kwargs)
        self.seq_norm1 = nn.LayerNorm(embed_dim, **self.factory_kwargs)
        self.seq_norm2 = nn.LayerNorm(embed_dim, **self.factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reset_buffer(self, batch_size: int, zero_node: torch.Tensor, veh: torch.Tensor):
        '''
        zero_node: the node No. 0, which usually represents the garbage for vehicles 
                    or the simbol of the end of a nodes' slice, B x 1 x D. 
        veh: the first vehicle. B x 1 x veh_dim.
        batch_size: the number of sequences in the batch.

        output:
        zero_embedding: the embedding of the zero node, B x 1 x d.
        veh: the first vehicle, B x 1 x veh_dim.
        '''
        self.zero_node = self.embedding(zero_node)
        _, self.zero_node_hidden = self.seq_encoder(self.zero_node)
        self.seq_hidden = torch.zeros_like(self.zero_node_hidden)
        self.zero_embedding = self.seq_block(self.zero_node, 
                                             torch.ones(batch_size, device=zero_node.device, dtype=torch.bool))
        return self.zero_embedding, veh

    def seq_block(self, seq: torch.Tensor, hidden_state: torch.Tensor):
        '''
        Embedding the current sequence.

        seq: current sequence, B x 1 x D
        hidden_state: current hidden state, N x B x D

        output:
        seq_embed: current sequence embedding, B x 1 x d
        '''
        seq_embed, new_hidden_state = self.seq_encoder(seq, hidden_state)
        seq_embed = self.seq_norm1(seq + seq_embed)
        seq_embed = self.seq_norm2(seq_embed + self.seq_ff(seq_embed))
        return seq_embed, new_hidden_state

    def forward(self, selection: torch.Tensor, veh: torch.Tensor, hidden_state: torch.Tensor):
        '''
        selection: new node selected for current sequences, B x 1 x input_dim
        veh: current vehicle, B x 1 x veh_dim
        end_of_seq: bool tensor to note which sequnces have ended, B x 1
        new_veh: new vehicles for ended sequnce, b x 1 x veh_dim, where b is the number of 'True' in end_of_seq
        key_mask: noting which tasks haven't finished yet, B
        veh_all: all veh embeddings, B x M x d.
        veh_key_padding_mask: mask of veh_all B x M.
        '''
        # Get the current sequence's embedding.
        seq = self.embedding(selection) + veh

        # Get the sequence's feature, where the last nodes appended are used as query.
        seq_embed, new_hidden_state = self.seq_block(seq, hidden_state)

        # Concat the sequences' feature with their corresponding vehicles' feature, and then embedding them. 
        slice_embedding = veh
        return seq_embed, slice_embedding, new_hidden_state