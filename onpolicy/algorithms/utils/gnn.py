import torch
import torch.nn as nn
from torch.nn import Module
from copy import deepcopy
import torch.nn.functional as F
from onpolicy.algorithms.utils.util import FeatureBlock, activations, Embedding_layer
from typing import Optional
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import unbatch, from_networkx, to_dense_adj, scatter
import torch_geometric.nn as gnn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool, global_max_pool

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

class HeteroFJSPEncoder(Module):
    def __init__(self, common_cfg, gnn_cfg, gff_cfg, device='cpu', dtype=torch.float32) -> None:
        """
        专为航空 FJSP 设计的异构图编码器。
        接收环境输出的 HeteroData，提取出工序、机位、设备的高阶特征，并输出给 Actor (Ptr-Net)。
        """
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.common_cfg = common_cfg
        
        # 激活函数解析
        if isinstance(self.common_cfg['activation'], str):
            self.activation = activations[self.common_cfg['activation']]
        else:
            self.activation = self.common_cfg['activation']

        self.embed_dim = self.common_cfg['embed_dim']
        layer_num = gnn_cfg.get('layer_num', 2)
        nhead = gnn_cfg.get('nhead', 4)
        dropout = gnn_cfg.get('dropout', 0.1)

        # ==========================================
        # 1. 节点特征独立投影层 (Node Encoders)
        # 将不同维度的节点原始特征统一映射到 embed_dim
        # ==========================================
        # 工序节点 (Dim: 6) -> Status, Proc_time, Rem_ops, Req_res, Wait_time, PID
        self.op_embedding = Embedding_layer(gnn_cfg['op_dim'], self.embed_dim, gnn_cfg['embedding_layer'], activation=self.activation, **self.factory_kwargs)
        
        # 机位节点 (Dim: 20) -> Occ, Interf, Rem_time + 17维 One-hot
        self.site_embedding = Embedding_layer(gnn_cfg['site_dim'], self.embed_dim, gnn_cfg['embedding_layer'], activation=self.activation, **self.factory_kwargs)
        
        # 设备节点 (Dim: 5) -> Type, Status, Rem_time, Pos_X, Pos_Y
        self.dev_embedding = Embedding_layer(gnn_cfg['dev_dim'], self.embed_dim, gnn_cfg['embedding_layer'], activation=self.activation, **self.factory_kwargs)

        # ==========================================
        # 2. 异构图卷积网络 (HeteroConv)
        # 替代了原先复杂的 Cross-Attention，让信息顺着拓扑结构流动
        # ==========================================
        self.convs = nn.ModuleList()
        for _ in range(layer_num):
            # 定义所有类型的边如何传递信息
            conv = gnn.HeteroConv({
                # 1. 工序之间的拓扑先后约束 (无边缘特征)
                ('operation', 'precedes', 'operation'): gnn.TransformerConv(
                    self.embed_dim, self.embed_dim // nhead, heads=nhead, dropout=dropout, **self.factory_kwargs),
                
                # 2. 工序 -> 机位 (前向传播：工序把需求传给机位) [带有边缘特征: 距离/时间]
                ('operation', 'assignable_to', 'site'): gnn.TransformerConv(
                    self.embed_dim, self.embed_dim // nhead, heads=nhead, edge_dim=1, dropout=dropout, **self.factory_kwargs),
                
                # 3. 机位 -> 工序 (反向传播：机位把排队拥挤情况反馈给工序)
                ('site', 'rev_assignable_to', 'operation'): gnn.TransformerConv(
                    self.embed_dim, self.embed_dim // nhead, heads=nhead, edge_dim=1, dropout=dropout, **self.factory_kwargs),
                
                # 4. 工序 -> 设备 (前向传播)
                ('operation', 'needs', 'device'): gnn.TransformerConv(
                    self.embed_dim, self.embed_dim // nhead, heads=nhead, edge_dim=1, dropout=dropout, **self.factory_kwargs),
                
                # 5. 设备 -> 工序 (反向传播：设备把自身的空闲状态反馈给需求方)
                ('device', 'rev_needs', 'operation'): gnn.TransformerConv(
                    self.embed_dim, self.embed_dim // nhead, heads=nhead, edge_dim=1, dropout=dropout, **self.factory_kwargs),
            }, aggr='sum')
            self.convs.append(conv)

        # 每个卷积层后的 LayerNorm 和残差连接
        self.norms_op = nn.ModuleList([nn.LayerNorm(self.embed_dim, **self.factory_kwargs) for _ in range(layer_num)])
        self.norms_site = nn.ModuleList([nn.LayerNorm(self.embed_dim, **self.factory_kwargs) for _ in range(layer_num)])
        self.norms_dev = nn.ModuleList([nn.LayerNorm(self.embed_dim, **self.factory_kwargs) for _ in range(layer_num)])

        # ==========================================
        # 3. 全局 Readout (Global Pooling & FFN)
        # ==========================================
        # 全局上下文包含: 3种节点的(mean + max) = 6个向量拼接
        self.global_embedding = Embedding_layer(6 * self.embed_dim, self.embed_dim, gff_cfg['layer'], 
                                                activation=self.activation, **self.factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data: HeteroData):
        """
        前向传播
        """
        # 1. 获取并映射基础节点特征
        x_dict = {
            'operation': self.op_embedding(data['operation'].x),
            'site': self.site_embedding(data['site'].x),
            'device': self.dev_embedding(data['device'].x)
        }

        # 2. 动态构建反向边 (Bidirectional Message Passing)
        edge_index_dict = data.edge_index_dict.copy()
        edge_attr_dict = data.edge_attr_dict.copy()

        # 添加 工序<->机位 的反向边
        if ('operation', 'assignable_to', 'site') in edge_index_dict:
            edge_index_dict[('site', 'rev_assignable_to', 'operation')] = edge_index_dict[('operation', 'assignable_to', 'site')].flip([0])
            edge_attr_dict[('site', 'rev_assignable_to', 'operation')] = edge_attr_dict[('operation', 'assignable_to', 'site')]
            
        # 添加 工序<->设备 的反向边
        if ('operation', 'needs', 'device') in edge_index_dict:
            edge_index_dict[('device', 'rev_needs', 'operation')] = edge_index_dict[('operation', 'needs', 'device')].flip([0])
            edge_attr_dict[('device', 'rev_needs', 'operation')] = edge_attr_dict[('operation', 'needs', 'device')]

        # 3. 通过 HeteroConv 层传递消息
        for conv, norm_op, norm_site, norm_dev in zip(self.convs, self.norms_op, self.norms_site, self.norms_dev):
            # out_dict 将包含本次卷积更新后的各类节点特征
            out_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            
            # 加上残差连接 (Residual) 并做 LayerNorm
            if 'operation' in out_dict:
                x_dict['operation'] = norm_op(x_dict['operation'] + self.activation(out_dict['operation']))
            if 'site' in out_dict:
                x_dict['site'] = norm_site(x_dict['site'] + self.activation(out_dict['site']))
            if 'device' in out_dict:
                x_dict['device'] = norm_dev(x_dict['device'] + self.activation(out_dict['device']))

        # 4. 全局特征提取 (Global Readout)
        # 获取 Batch index (处理单独输入 1 张图 和 PPO Batch 输入 n 张图兼容)
        op_batch = data['operation'].batch if hasattr(data['operation'], 'batch') else None
        site_batch = data['site'].batch if hasattr(data['site'], 'batch') else None
        dev_batch = data['device'].batch if hasattr(data['device'], 'batch') else None

        def get_pool(x, batch):
            if batch is None:
                return x.mean(dim=0, keepdim=True), x.max(dim=0, keepdim=True)[0]
            else:
                return global_mean_pool(x, batch), global_max_pool(x, batch)

        g_ops_mean, g_ops_max = get_pool(x_dict['operation'], op_batch)
        g_sites_mean, g_sites_max = get_pool(x_dict['site'], site_batch)
        
        # 容错：有可能某些时刻图中没有任何设备
        if x_dict['device'].shape[0] > 0:
            g_devs_mean, g_devs_max = get_pool(x_dict['device'], dev_batch)
        else:
            g_devs_mean = torch.zeros_like(g_ops_mean)
            g_devs_max = torch.zeros_like(g_ops_max)

        # 拼接并映射全局上下文
        g_concat = torch.cat([g_ops_mean, g_ops_max, g_sites_mean, g_sites_max, g_devs_mean, g_devs_max], dim=-1)
        g_global = self.global_embedding(g_concat)

        # 5. 格式化输出 (Dense 转换，供 Actor 的 Pointer-Net 使用)
        # to_dense_batch 会将散点图节点自动补齐(Pad)成 [Batch_size, Max_nodes, Dim] 的张量
        # 且自动返回 padding mask (True 表示是真实的节点，False 表示是补齐的空节点)
        if op_batch is not None:
            op_dense, op_mask = to_dense_batch(x_dict['operation'], op_batch)
            site_dense, site_mask = to_dense_batch(x_dict['site'], site_batch)
            # PyTorch 标准的 key_padding_mask 习惯是 True 代表是 Padding (需要屏蔽的)
            op_key_padding_mask = ~op_mask
            site_key_padding_mask = ~site_mask
        else:
            op_dense = x_dict['operation'].unsqueeze(0)
            site_dense = x_dict['site'].unsqueeze(0)
            op_key_padding_mask = torch.zeros((1, op_dense.size(1)), device=op_dense.device).bool()
            site_key_padding_mask = torch.zeros((1, site_dense.size(1)), device=site_dense.device).bool()

        # 返回 Actor 需要的全部物料
        return {
            "global_emb": g_global,                       # [B, embed_dim] - 用于初始化 Actor 查询向量
            "op_nodes": op_dense,                         # [B, max_ops, embed_dim] - 工序节点备选库
            "op_padding_mask": op_key_padding_mask,       # [B, max_ops]
            "site_nodes": site_dense,                     # [B, max_sites, embed_dim] - 机位节点备选库
            "site_padding_mask": site_key_padding_mask    # [B, max_sites]
        }