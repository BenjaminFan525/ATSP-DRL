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

class HeteroGraphEncoder(Module):
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
                # 【修改】：删除了结尾的 **self.factory_kwargs
                ('operation', 'precedes', 'operation'): gnn.TransformerConv(
                    self.embed_dim, self.embed_dim // nhead, heads=nhead, dropout=dropout),
                
                # 2. 工序 -> 机位 (前向传播：工序把需求传给机位) [带有边缘特征: 距离/时间]
                ('operation', 'assignable_to', 'site'): gnn.TransformerConv(
                    self.embed_dim, self.embed_dim // nhead, heads=nhead, edge_dim=1, dropout=dropout),
                
                # 3. 机位 -> 工序 (反向传播：机位把排队拥挤情况反馈给工序)
                ('site', 'rev_assignable_to', 'operation'): gnn.TransformerConv(
                    self.embed_dim, self.embed_dim // nhead, heads=nhead, edge_dim=1, dropout=dropout),
                
                # 4. 工序 -> 设备 (前向传播)
                ('operation', 'needs', 'device'): gnn.TransformerConv(
                    self.embed_dim, self.embed_dim // nhead, heads=nhead, edge_dim=1, dropout=dropout),
                
                # 5. 设备 -> 工序 (反向传播：设备把自身的空闲状态反馈给需求方)
                ('device', 'rev_needs', 'operation'): gnn.TransformerConv(
                    self.embed_dim, self.embed_dim // nhead, heads=nhead, edge_dim=1, dropout=dropout),
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