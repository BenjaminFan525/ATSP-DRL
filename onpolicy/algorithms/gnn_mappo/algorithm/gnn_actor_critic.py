import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from onpolicy.algorithms.utils.gnn import HeteroFJSPEncoder
from onpolicy.algorithms.utils.gru import SelectionEncoder
from onpolicy.algorithms.utils.ptr_actor import CascadePtrActor
from onpolicy.algorithms.utils.step_critic import StepCritic

class GNN_Actor_Critic(nn.Module):
    def __init__(self, common_cfg=None, encoder_cfg=None, sel_encoder_cfg=None, actor_cfg=None, critic_cfg=None, critic_list=['default'], device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.common = common_cfg
        self.encoder_cfg = encoder_cfg
        self.selection_enc = sel_encoder_cfg
        self.actor_cfg = actor_cfg
        self.critic_cfg = critic_cfg

        # 1. 异构图编码器
        self.encoder = HeteroFJSPEncoder(self.common, self.encoder_cfg, self.common, **self.factory_kwargs)        
        
        # 2. GRU 时序记忆编码器
        self.sel_enc = SelectionEncoder(**self.common, **self.selection_enc, **self.factory_kwargs)         
        
        # 3. Actor 动作网络
        self.actor = CascadePtrActor(**self.common, **self.actor_cfg, **self.factory_kwargs)
        
        # 4. Critic 价值网络
        self.critic = nn.ModuleDict({})
        for name in critic_list:
            self.critic.update({name: StepCritic(**self.common, **self.critic_cfg, **self.factory_kwargs)})
        
        self.tau = 1.0

        self.cfg = {
            'common_cfg': self.common,
            'encoder_cfg': self.encoder_cfg,
            'sel_encoder_cfg': self.selection_enc,
            'actor_cfg': self.actor_cfg,
            'critic_cfg': self.critic_cfg,
            'critic_list': critic_list
        }

        # 优化器参数分组
        self.actor_param = nn.ModuleList([self.encoder, self.sel_enc, self.actor])
        # 注意：这里如果 encoder 没有 own_modules 属性，可以直接用 self.encoder
        self.actor_param_without_gnn = nn.ModuleList([self.sel_enc, self.actor]) 
        self.critic_param = nn.ModuleList([self.encoder, self.sel_enc, self.critic])

        self.to(device)

    def forward(self, data, info, deterministic: bool = False, chosen_op=None, chosen_site=None,
                actor_grad: bool = True, criticize: bool = True, eval_action: bool = False, 
                dist_only: bool = False, criticize_only: bool = False):
        '''
        FJSP 异构图 Actor-Critic 前向传播核心逻辑
        
        参数映射说明：
        chosen_idx -> 代表选中的工序 (chosen_op)
        chosen_entry -> 代表选中的机位 (chosen_site)
        '''
        assert not (dist_only and criticize_only)
        if dist_only: criticize = False
        if criticize_only: criticize = True; actor_grad = False
        if eval_action: criticize = False; actor_grad = True

        # ========================================================
        # 1. 编码异构图 (Hetero Encoder)
        # ========================================================
        enc_out = self.encoder(data['graph'])
        global_emb = enc_out['global_emb']      # [B, Embed_Dim]
        op_nodes = enc_out['op_nodes']          # [B, N_ops, Embed_Dim]
        site_nodes = enc_out['site_nodes']      # [B, N_sites, Embed_Dim]
        
        # 获取环境生成的合法动作掩码 (True 代表合法可用)
        op_valid_mask = data['graph'].op_mask.clone()             # [B, N_agents, N_ops]
        site_mask_matrix = data['graph'].site_mask_matrix.clone() # [B, N_ops, N_sites]

        # ========================================================
        # 2. 初始化张量与隐状态
        # ========================================================
        bsz = global_emb.shape[0]
        # PPO 传进来的 info 中的 tensor 均需要挪到相应 device
        active_agents = info['active_agents']
        last_op_indices = info['last_op_indices']
        
        M = active_agents.shape[1] # 最大飞机数 n_agents
        
        op_choice = -torch.ones((bsz, M), device=global_emb.device, dtype=torch.long)
        site_choice = -torch.ones((bsz, M), device=global_emb.device, dtype=torch.long)
        prob = torch.ones((bsz, M), device=global_emb.device)
        value_mask = torch.zeros((bsz, M), dtype=torch.bool, device=global_emb.device)
        value = {name: torch.zeros((bsz, M), device=global_emb.device) for name in self.critic.keys()}
        
        # 继承并克隆 GRU 的历史记忆
        new_hidden_state = data['hidden_states'].clone()
        
        # 用于累加信息熵
        total_entropy = torch.tensor(0.0, device=global_emb.device)
        entropy_counts = 0

        # ========================================================
        # 3. 遍历智能体 (飞机) 进行串行自回归决策
        # ========================================================
        for agent_idx in range(M):
            active_mask_i = active_agents[:, agent_idx]
            if not active_mask_i.any():
                continue
                
            value_mask[:, agent_idx] = active_mask_i
            
            # ---------------------------------------------------------
            # A. 提取 GRU 的输入 1：上一次的选择 (刚做完的工序)
            # ---------------------------------------------------------
            last_op_idx = last_op_indices[active_mask_i, agent_idx]
            valid_last_op = (last_op_idx >= 0)
            safe_last_op_idx = torch.clamp(last_op_idx, min=0)
            
            # 抽取高阶特征并处理冷启动
            last_op_emb = op_nodes[active_mask_i, safe_last_op_idx, :]
            last_op_emb = last_op_emb * valid_last_op.unsqueeze(-1).float() 
            last_selection_emb = last_op_emb.unsqueeze(1) # [B_active, 1, Embed_Dim]

            # ---------------------------------------------------------
            # B. 提取 GRU 的输入 2：当前飞机的诉求 (当前可用工序上下文)
            # ---------------------------------------------------------
            cur_agent_mask = op_valid_mask[:, agent_idx, :] # [B, N_ops]
            valid_ops_emb = op_nodes[active_mask_i] * cur_agent_mask[active_mask_i].unsqueeze(-1).float()
            
            counts = cur_agent_mask[active_mask_i].sum(dim=1, keepdim=True).clamp(min=1e-5)
            agent_context = valid_ops_emb.sum(dim=1) / counts
            veh_emb = agent_context.unsqueeze(1) # [B_active, 1, Embed_Dim]

            # ---------------------------------------------------------
            # C. 过 GRU 记忆更新并拼接 Query
            # ---------------------------------------------------------
            if new_hidden_state is not None:
                hidden_state_i = data['hidden_states'][active_mask_i, agent_idx:agent_idx+1, :].squeeze(1).transpose(0, 1)
                
                seq_embed, slice_embed, new_hidden_state_i = self.sel_enc(
                    last_selection=last_selection_emb, 
                    veh=veh_emb, 
                    hidden_state=hidden_state_i
                )
                
                new_hidden_state[active_mask_i, agent_idx:agent_idx+1, :] = new_hidden_state_i.transpose(0, 1).unsqueeze(1)
                
                query = torch.cat([global_emb[active_mask_i], seq_embed.squeeze(1), slice_embed.squeeze(1)], dim=-1)
            else:
                query = torch.cat([global_emb[active_mask_i], veh_emb.squeeze(1)], dim=-1) # fallback

            # ---------------------------------------------------------
            # D. PPO 强制动作注入 (算 Loss 用)
            # ---------------------------------------------------------
            cur_chosen_op = chosen_op[active_mask_i, agent_idx].squeeze(-1) if chosen_op is not None else None
            cur_chosen_site = chosen_site[active_mask_i, agent_idx].squeeze(-1) if chosen_site is not None else None
            # ---------------------------------------------------------
            # E. 级联 Actor 决策
            # ---------------------------------------------------------
            if actor_grad:
                cur_op, cur_site, cur_prob, cur_dist = self.actor(
                    query=query.unsqueeze(1), 
                    op_nodes=op_nodes[active_mask_i], 
                    site_nodes=site_nodes[active_mask_i], 
                    op_valid_mask=cur_agent_mask[active_mask_i], 
                    site_mask_matrix=site_mask_matrix[active_mask_i], 
                    deterministic=deterministic, 
                    tau=self.tau,
                    chosen_op=cur_chosen_op, 
                    chosen_site=cur_chosen_site
                )
            else:
                with torch.no_grad():
                    cur_op, cur_site, cur_prob, cur_dist = self.actor(
                        query.unsqueeze(1), op_nodes[active_mask_i], site_nodes[active_mask_i], 
                        cur_agent_mask[active_mask_i], site_mask_matrix[active_mask_i], 
                        deterministic, cur_chosen_op, cur_chosen_site, self.tau
                    )

            # 记录决策动作与概率
            op_choice[active_mask_i, agent_idx] = cur_op
            site_choice[active_mask_i, agent_idx] = cur_site
            prob[active_mask_i, agent_idx] = cur_prob

            # 累加信息熵 (评估阶段专用)
            if eval_action or dist_only:
                op_prob_v, site_prob_v = cur_dist
                # PPO 算法需要真实的分布对象来算 loss 或者熵
                dist_op = torch.distributions.Categorical(op_prob_v)
                dist_site = torch.distributions.Categorical(site_prob_v)
                total_entropy += dist_op.entropy().sum() + dist_site.entropy().sum()
                entropy_counts += active_mask_i.sum().item() * 2

            # ---------------------------------------------------------
            # F. Critic 价值评估
            # ---------------------------------------------------------
            if criticize:
                cur_op_pad_mask = ~cur_agent_mask[active_mask_i] 
                cur_site_pad_mask = None # 全局资源视角
                
                for key, value_head in self.critic.items():
                    value[key][active_mask_i, agent_idx] = value_head(
                        query=query.unsqueeze(1), 
                        op_nodes=op_nodes[active_mask_i], 
                        site_nodes=site_nodes[active_mask_i], 
                        op_pad_mask=cur_op_pad_mask,
                        site_pad_mask=cur_site_pad_mask
                    ).squeeze(-1)

            # ---------------------------------------------------------
            # G. 自回归动态掩码刷新 (避免同一个 Batch 互相抢资源)
            # ---------------------------------------------------------
            batch_indices = torch.nonzero(active_mask_i).squeeze(1)
            # 刚才被选走的工序不能再选
            op_valid_mask[batch_indices, agent_idx, cur_op] = False
            # 刚才被分配的机位不能再选
            site_mask_matrix[batch_indices, :, cur_site] = False


        # ========================================================
        # 4. 返回值格式化
        # ========================================================
        # 将联合概率转换为 Log Prob
        log_prob = torch.log(prob + 1e-10)
        
        # 兼容旧代码，将两个动作打在一个 Tensor 的最后一个维度里
        action = torch.stack([op_choice, site_choice], dim=2)
        
        # 计算平均熵
        ent = total_entropy / max(1, entropy_counts)

        if criticize_only:
            return value
        if dist_only:
            # 级联网络没有单一的 Categorical 分布，通常返回 Entropy 即可供 PPO 更新
            return None # 或者自定义联合分布返回
        if criticize:
            return value, action, log_prob, new_hidden_state
        if eval_action:
            return log_prob, ent
            
        return action, new_hidden_state