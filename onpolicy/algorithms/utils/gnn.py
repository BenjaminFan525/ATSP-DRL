import torch
import torch.nn as nn
from torch.nn import Module
from copy import deepcopy
import torch.nn.functional as F
from onpolicy.algorithms.utils.util import FeatureBlock, activations, Embedding_layer
from typing import Optional
from torch_geometric.data import Data
from torch_geometric.utils import unbatch, from_networkx, to_dense_adj, scatter
import torch_geometric.nn as gnn
from torch.nn.utils.rnn import pad_sequence

activations = {
    'relu': F.relu,
    'gelu': F.gelu,
    'tanh': F.tanh,
    'sigmoid': F.sigmoid
}

class GNNWithEdge(Module):
    def __init__(self, node_dim, edge_dim, embedding_layer=1, embed_dim=64, nhead=4, 
                 activation=F.relu, layer_num = 2, dropout: float = 0.1,
                 device='cpu', dtype=torch.float32) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(activation, str):
            activation = activations[activation]
        super().__init__()
        self.activation = activation
        self.embed_dim = embed_dim

        self.node_embedding = Embedding_layer(node_dim, self.embed_dim, embedding_layer,
                                                        activation=self.activation, **self.factory_kwargs)
        self.edge_embedding = Embedding_layer(edge_dim, self.embed_dim, embedding_layer,
                                                        activation=self.activation, **self.factory_kwargs)
        # edge feature
        self.edge_attn = nn.MultiheadAttention(self.embed_dim, nhead, **self.factory_kwargs)
        self.edge_linear1 = nn.Linear(self.embed_dim, self.embed_dim, **self.factory_kwargs)
        self.edge_linear2 = nn.Linear(self.embed_dim, self.embed_dim, **self.factory_kwargs)
        self.edge_norm1 = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)
        self.edge_norm2 = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)
        self.edge_dropout1 = nn.Dropout(dropout)
        self.edge_dropout2 = nn.Dropout(dropout)

        # node feature (main GNN part)
        self.attn_head_dim = self.embed_dim // nhead
        assert self.attn_head_dim * nhead == self.embed_dim, "embed_dim must be divisible by num_heads"
        GNN_layer = gnn.TransformerConv(self.embed_dim, self.attn_head_dim, nhead, 
                                    edge_dim=self.embed_dim, dropout=dropout).to(device)
        self.mods = nn.ModuleList([deepcopy(GNN_layer) for _ in range(layer_num)])
        norm_layer = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)
        self.norms = nn.ModuleList([deepcopy(norm_layer) for _ in range(layer_num)])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def edge_feature(self, e: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        e = self.edge_norm1(e + self.edge_attn_block(e, batch))
        e = self.edge_norm2(e + self.edge_ff(e))
        return e

    def edge_ff(self, e: torch.Tensor) -> torch.Tensor:
        return self.edge_dropout2(self.edge_linear2(self.activation(self.edge_linear1(e))))

    def edge_attn_block(self, e: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            return self.edge_dropout1(self.edge_attn(e, e, e, need_weights=False)[0])
        
        attn_output = [self.edge_attn(x, x, x, need_weights=False)[0] for x in unbatch(e, batch)]
        return self.edge_dropout1(torch.concat(attn_output))
        
    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if data.batch is None:
            edge_to_graph = None
        else:
            edge_to_graph = data.batch[edge_index[0]]

        x = self.node_embedding(x)
        e = self.edge_embedding(edge_attr)

        e = self.edge_feature(e, edge_to_graph)

        for mod, norm in zip(self.mods, self.norms):
            x = norm(x + mod(x, edge_index, e))

        return x

class GNNEncoder(Module):
    def __init__(self, common_cfg=None, gnn_cfg=None, gff_cfg=None, frozen_gnn=False, device='cpu', dtype=torch.float32) -> None:

        super().__init__()
        self.factory_kwargs = {'dtype': dtype}

        self.common_cfg = common_cfg
        self.gnn_cfg = gnn_cfg
        self.gff_cfg = gff_cfg

        if isinstance(self.common_cfg['activation'], str):
            self.activation = activations[self.common_cfg['activation']]
        else:
            self.activation = self.common_cfg['activation']
        
        self.frozen_gnn = frozen_gnn
        if 'activation' in self.gnn_cfg:
            self.gnn_cfg.update({'embed_dim': self.common_cfg['embed_dim']})
        else:
            self.gnn_cfg.update(self.common_cfg)
        self.gnn = GNNWithEdge(**self.gnn_cfg, **self.factory_kwargs)
        self.global_embedding = Embedding_layer(2 * self.common_cfg['embed_dim'], 
                                                          self.common_cfg['embed_dim'], 
                                                          self.gff_cfg['layer'], activation=self.activation, 
                                                          **self.factory_kwargs)
    
    def freeze_gnn(self):
        self.frozen_gnn = True

    def unfreeze_gnn(self):
        self.frozen_gnn = False

    def forward(self, data: Data):
        if self.frozen_gnn:
            with torch.no_grad():
                x = self.gnn(data)
        else:
            x = self.gnn(data)
        if data.batch is None:
            g = torch.unsqueeze(torch.cat([torch.mean(x, dim=0), torch.max(x, dim=0)[0]]), 0)
            x = torch.unsqueeze(x, 0)
            key_padding_mask = torch.zeros((x.shape[0], x.shape[1]), device=x.device).bool()
        else:
            g = torch.cat([
                scatter(x, data.batch, dim=0, reduce='mean'),
                scatter(x, data.batch, dim=0, reduce='max')
            ], dim=-1)
            x = unbatch(x, data.batch)
            key_padding_mask = [torch.zeros(len(d), device=d.device) for d in x]
            x = pad_sequence(x, batch_first=True, padding_value=0.)
            key_padding_mask = pad_sequence(key_padding_mask, batch_first=True, padding_value=1).bool()
        
        return self.global_embedding(g), x, key_padding_mask

class MaGNNEncoder(Module):
    def __init__(self, common_cfg=None, gnn_cfg=None, veh_dim=5, embedding_layer=1, enc_type='enc', layer_num=1, nhead=4, device=None, dtype=None) -> None:
        '''
        embedding_layer: number of layers of car embedding
        layer_num: number of layers of car FeatureBlock
        '''
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.common_cfg = common_cfg
        self.gnn_cfg = gnn_cfg

        if isinstance(self.common_cfg['activation'], str):
            self.activation = activations[self.common_cfg['activation']]
        else:
            self.activation = self.common_cfg['activation']

        self.embed_dim = self.common_cfg['embed_dim']

        self.veh_embedding = Embedding_layer(veh_dim, self.embed_dim, embedding_layer, 
                                             activation=self.activation, **self.factory_kwargs)
        self.nodes_encoder = GNNEncoder(common_cfg=self.common_cfg, **self.gnn_cfg, **self.factory_kwargs)
             
        self.veh_attn = FeatureBlock(enc_type, layer_num, self.embed_dim, nhead, self.activation, **self.factory_kwargs)

        self.global_veh_ff = Embedding_layer(2 * self.embed_dim, self.embed_dim, 2, 
                                             activation=self.activation, **self.factory_kwargs)
        
        
        self.own_modules = [self.veh_embedding, self.nodes_encoder.global_embedding,
                            self.veh_attn, self.global_veh_ff]

        self.ff = nn.Linear(self.embed_dim, self.embed_dim, **self.factory_kwargs)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, nhead, batch_first=True, **self.factory_kwargs)
        self.own_modules += [self.ff, self.cross_attn]
        self.own_modules = nn.ModuleList(self.own_modules)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def freeze_gnn(self):
        self.nodes_encoder.freeze_gnn()
    
    def unfreeze_gnn(self):
        self.nodes_encoder.unfreeze_gnn()

    def forward(self, data, vehicles: torch.Tensor, veh_key_padding_mask: Optional[torch.Tensor], veh_num: torch.Tensor, graph_embd=False):
        """
        Forward pass for the Vehicle Encoder.
        Handles embedding, self-attention, cross-attention with nodes, and global pooling.
        """
        # 1. Graph Node Encoding
        if graph_embd:
            # Readout of the graph: global features, individual node features, and padding mask
            global_nodes, nodes, node_key_padding_mask = self.nodes_encoder(data)
        else:
            global_nodes, nodes, node_key_padding_mask = data
            
        # 2. Vehicle Embedding
        # Project raw vehicle features to embedding dimension
        veh = self.veh_embedding(vehicles) 
        
        global_nodes = global_nodes.unsqueeze(1)

        # Prepare expanded mask (B, N_v, 1) for subsequent `masked_fill` operations
        # This is used to zero out features of padding vehicles to prevent gradient pollution.
        if veh_key_padding_mask is not None:
             mask_expanded = veh_key_padding_mask.unsqueeze(-1)
        else:
             mask_expanded = None

        # =======================================================
        # [Gradient Safety 1] Construct Safe Mask for Self-Attention
        # Problem: If a batch index has ALL vehicles masked (all True), Softmax causes NaN gradients.
        # Solution: Temporarily unmask the first vehicle (index 0) to allow valid Softmax flow.
        #           The output for this dummy vehicle will be zeroed out later.
        # =======================================================
        if veh_key_padding_mask is not None:
            all_masked = veh_key_padding_mask.all(dim=1)
            safe_veh_mask = veh_key_padding_mask.clone()
            if all_masked.any():
                # Force unmask the first element to prevent NaN in gradients
                safe_veh_mask[all_masked, 0] = False
        else:
            safe_veh_mask = None

        # 3. Vehicle Self-Attention
        # Use safe_veh_mask to avoid NaN during backward pass
        veh = self.veh_attn(veh, safe_veh_mask)
        
        # [Gradient Safety 2] Non-inplace Cleaning
        # Crucial: Use `masked_fill` (creates new tensor) instead of in-place assignment `veh[mask]=0`.
        # This prevents "RuntimeError: one of the variables needed for gradient computation has been modified".
        if mask_expanded is not None:
            veh = veh.masked_fill(mask_expanded, 0.0)

        # 4. Cross-Attention (Vehicle -> Nodes)
        B, N_v, _ = veh.shape
        _, N_n, _ = nodes.shape
        device = veh.device

        # Determine the number of vehicles per batch for dynamic masking
        if veh_num is not None:
             M_tensor = veh_num.view(B, 1, 1).long()
        else:
             # Fallback: infer from mask (less reliable if layout isn't compact)
             M_tensor = torch.sum(~veh_key_padding_mask, dim=1).view(B, 1, 1).long()

        # --- Construct Cross-Attention Mask ---
        # 4.1 Base Mask: Start with node padding mask
        attn_mask = node_key_padding_mask.unsqueeze(1).expand(B, N_v, N_n).clone()
        
        # 4.2 Coordinate Grids for broadcasting
        col_idx = torch.arange(N_n, device=device).view(1, 1, N_n) # Node indices
        row_idx = torch.arange(N_v, device=device).view(1, N_v, 1) # Vehicle indices

        # 4.3 Logic: Restrict Depots Visibility
        # Rule: Vehicles should NOT see other vehicles' depots.
        
        # Step A: Mask ALL depot nodes (assuming depots are at the beginning: indices < 2 * M)
        is_depot_zone = col_idx < (2 * M_tensor)
        attn_mask = attn_mask | is_depot_zone 
        
        # Step B: Unmask ONLY the specific Start and End depot for the current vehicle
        is_my_start = (col_idx == row_idx)                   # Start node index matches vehicle index
        is_my_end   = (col_idx == (row_idx + M_tensor))      # End node is offset by M
        is_my_depot = is_my_start | is_my_end
        
        # Apply unmasking (False in mask means visible)
        attn_mask = attn_mask & (~is_my_depot)
        
        # 4.4 Adapt for Multi-Head Attention
        num_heads = self.cross_attn.num_heads
        attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)

        # Execute Cross Attention
        veh_cross = self.cross_attn(veh, nodes, nodes, attn_mask=attn_mask)[0]
        
        # 5. Residual Connection & FFN
        veh = veh + veh_cross
        
        # [Gradient Safety] Clean padding again before FFN
        if mask_expanded is not None:
            veh = veh.masked_fill(mask_expanded, 0.0)
            
        ff_out = self.ff(veh)
        veh = veh + self.activation(ff_out)

        # [Gradient Safety] Final clean after FFN
        if mask_expanded is not None:
            veh = veh.masked_fill(mask_expanded, 0.0)

        # 6. Global Pooling (Vectorized)
        # Aggregates individual vehicle features into a global fleet representation.
        B, M, D = veh.shape
        
        # Prepare valid masks for calculation (1.0 for valid, 0.0 for padding)
        valid_mask_float = (~veh_key_padding_mask).float().unsqueeze(-1)
        
        # --------------------
        # A. Mean Pooling
        # --------------------
        # Sum valid features (padding is already 0.0, but valid_mask_float ensures safety)
        sum_veh = (veh * valid_mask_float).sum(dim=1) # Shape: (B, D)
        
        # Count valid vehicles
        counts = valid_mask_float.sum(dim=1) # Shape: (B, 1)
        
        # Safe Division: Avoid division by zero for empty batches (all padding)
        # Trick: Replace 0 counts with 1. The numerator is 0 anyway, so result 0/1 = 0 is correct.
        safe_counts = counts.clone()
        safe_counts[counts == 0] = 1.0 
        mean_veh = sum_veh / safe_counts # Shape: (B, D)

        # --------------------
        # B. Max Pooling
        # --------------------
        # Fill padding with -inf so max() ignores them
        # Note: Must use masked_fill (non-inplace) for gradient safety
        veh_for_max = veh.masked_fill(mask_expanded, float('-inf'))
        
        # Take max along the vehicle dimension
        max_veh_val = veh_for_max.max(dim=1)[0] # Shape: (B, D)
        
        # Fix empty samples: If a sample was all padding, max result is -inf. Reset to 0.0.
        is_empty_sample = (counts == 0) # Shape: (B, 1)
        max_veh_val = max_veh_val.masked_fill(is_empty_sample, 0.0)

        # --------------------
        # C. Output Formatting
        # --------------------
        # Restore dimension (B, D) -> (B, 1, D) for concatenation
        max_veh = max_veh_val.unsqueeze(1)
        mean_veh = mean_veh.unsqueeze(1)

        global_veh = torch.cat([max_veh, mean_veh], dim=-1)
                
        return self.global_veh_ff(global_veh), veh, global_nodes, nodes