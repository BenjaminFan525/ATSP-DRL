import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from onpolicy.algorithms.utils.gnn import MaGNNEncoder
from onpolicy.algorithms.utils.gru import SelectionEncoder
from onpolicy.algorithms.utils.ptr_actor import PtrEntryActor
from onpolicy.algorithms.utils.step_critic import StepCritic

class GNN_Actor_Critic(nn.Module):
    def __init__(self, common_cfg=None, encoder_cfg=None, sel_encoder_cfg=None, actor_cfg=None, critic_cfg=None, critic_list=['default'], device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        # Initialize common parameters
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.common = common_cfg
        self.encoder_cfg = encoder_cfg
        self.selection_enc = sel_encoder_cfg
        self.actor_cfg = actor_cfg
        self.critic_cfg = critic_cfg

        # Initialize encoder, selection encoder, actor, and critic
        self.encoder = MaGNNEncoder(self.common, **self.encoder_cfg, **self.factory_kwargs)        
        self.sel_enc = SelectionEncoder(**self.common, **self.selection_enc, **self.factory_kwargs)         
        self.actor = PtrEntryActor(**self.common, **self.actor_cfg, **self.factory_kwargs)
        self.critic = nn.ModuleDict({})
        for name in critic_list:
            self.critic.update({name: StepCritic(**self.common, **self.critic_cfg, **self.factory_kwargs)})
        
        self.tau = 1

        self.cfg = {
            'common_cfg': self.common,
            'encoder_cfg': self.encoder_cfg,
            'sel_encoder_cfg': self.selection_enc,
            'actor_cfg': self.actor_cfg,
            'critic_cfg': self.critic_cfg,
            'critic_list': critic_list
        }

        self.actor_param = nn.ModuleList([self.encoder, self.sel_enc, self.actor])
        self.actor_param_without_gnn = nn.ModuleList([self.encoder.own_modules, self.sel_enc, self.actor])
        self.critic_param = nn.ModuleList([self.encoder, self.sel_enc, self.critic])

        self.to(device)

    def forward(self, data, info, deterministic: bool = False, chosen_idx=None, chosen_entry=None,
                actor_grad: bool = True, criticize: bool = True, eval_action: bool = False, dist_only: bool = False, criticize_only: bool = False, force_chosen: bool=False):
        '''
        Forward pass of the GNN Actor-Critic model.

        Input:
        data: {'graph': PyG Graph Data, 'vehicles': B x M x Dim Tensor, 'hidden_states': B x M x 1 x Dim Tensor}
        info: {'veh_key_padding_mask': B x M Tensor, 'node_key_padding_mask': B x N tensor, 'key_mask': B x 1 Tensor, 'veh_num': B x 1 Tensor}
        deterministic: if True, output the choice with max probability.
        chosen_idx: chosen indexs. B x M x 1 Tensor
        chosen_entry: chosen entry. B x M x 1 Tensor
        actor_grad: whether need the actor's gradiant.
        criticize: whether calculate V(s).
        dist_only: if True, only out put the distribution.
        criticize only: if True, only calculate V(s).
        force_chosen: if True, the choices in chosen_idx(index != -1) will be chosen with probability 1.0. 

        Output:
        seq_enc: sequence encoding information.
        '''
        assert not (dist_only and criticize_only)
        if dist_only:
            criticize = False
        if criticize_only:
            criticize = True
            actor_grad = False
        if eval_action:
            criticize = False
            actor_grad = True

        # Encode
        global_veh, veh, global_nodes, nodes = self.encoder(data['graph'], data['vehicles'][:, :, :-2], info['veh_key_padding_mask'], info['veh_num'])
        
        # Initialize variables
        veh_key_padding_mask = info['veh_key_padding_mask'].clone()
        key_mask = info['key_mask'].clone()
        N = nodes.shape[1]
        bsz, M = veh.shape[0], veh.shape[1]
        veh_num = info['veh_num']
        node_key_padding_mask = info['node_key_padding_mask'].clone()[:, :N]
        value = {}

        choice = torch.zeros((bsz, M, N), device=veh.device)
        index = - torch.ones((bsz, M), device=veh.device, dtype=torch.long)
        entry = torch.zeros((bsz, M), device=veh.device, dtype=torch.long)
        prob = torch.ones((bsz, M), device=veh.device)
        dists = torch.ones((bsz, M, 2 * N), device=veh.device)
        value_mask = torch.zeros((bsz, M), dtype=torch.bool, device=veh.device)

        for name in self.critic.keys():
            value.update({name: torch.zeros((bsz, M), device=veh.device)})
            
        # mask chosen nodes for force_chosen setting
        if force_chosen: 
            node_key_padding_mask.scatter_(dim=1, index=chosen_idx.squeeze(-1).long(), value=True)

        selected_nodes = torch.cat([nodes[torch.arange(bsz, device=veh.device).unsqueeze(1), 
                                          data['vehicles'][:, :, -2].long()], 
                                          data['vehicles'][:, :, -1:],
                                          1-data['vehicles'][:, :, -1:]], dim=-1)

        new_hidden_state = data['hidden_states'].clone()
        
        for veh_idx in range(M):
            veh_mask_i = (~veh_key_padding_mask[:, veh_idx])
            if not veh_mask_i.any():
                continue
            veh_i = veh[veh_mask_i, veh_idx:veh_idx+1, :]
            selected_nodes_i = selected_nodes[veh_mask_i, veh_idx:veh_idx+1, :]
            hidden_state_i = data['hidden_states'][veh_mask_i, veh_idx:veh_idx+1, :].squeeze(1).transpose(0, 1)
            
            node_key_padding_mask_i = node_key_padding_mask[veh_mask_i].clone()
            start_end_region_mask = torch.arange(node_key_padding_mask_i.shape[1], device=veh.device).unsqueeze(0) < (2 * veh_num[veh_mask_i])
            node_key_padding_mask_i.masked_fill_(start_end_region_mask, True)
            task_region_mask = ~start_end_region_mask
            has_unfinished_tasks = ((~node_key_padding_mask_i) & task_region_mask).any(dim=1)
            is_last_vehicle = ((key_mask[veh_mask_i]).sum(dim=1) == 1)
            can_go_home = (~is_last_vehicle) | (is_last_vehicle & ~has_unfinished_tasks)
            node_key_padding_mask_i.scatter_(1, (veh_num[veh_mask_i] + veh_idx).long(), (~can_go_home).unsqueeze(1))

            seq_embed, slice_embed, new_hidden_state_i  = self.sel_enc(selected_nodes_i, veh_i, hidden_state_i)
            new_hidden_state[veh_mask_i, veh_idx:veh_idx+1, :] = new_hidden_state_i.transpose(0, 1).unsqueeze(1)
            
            query = torch.cat([global_veh[veh_mask_i],
                               global_nodes[veh_mask_i],
                               seq_embed, 
                               slice_embed], dim=-1)

            if not force_chosen:
                force_chosen_mask = None
                cur_chosen_idx = chosen_idx[veh_mask_i, veh_idx:veh_idx+1] if chosen_idx is not None else None
                cur_chosen_entry = chosen_entry[veh_mask_i, veh_idx:veh_idx+1] if chosen_entry is not None else None
            else:
                cur_chosen_idx = chosen_idx[:, veh_idx:veh_idx+1, :].squeeze(-1) 
                cur_chosen_entry = chosen_entry[:, veh_idx:veh_idx+1, :].squeeze(-1)

                force_chosen_mask = (veh_mask_i.unsqueeze(1)) & (cur_chosen_idx != -1)

                cur_chosen_idx = torch.where(force_chosen_mask, cur_chosen_idx, torch.tensor(0, device=veh.device))
                cur_chosen_entry = torch.where(force_chosen_mask, cur_chosen_entry, torch.tensor(0, device=veh.device))

                cur_chosen_idx = cur_chosen_idx[veh_mask_i]
                cur_chosen_entry = cur_chosen_entry[veh_mask_i]
                force_chosen_mask = force_chosen_mask[veh_mask_i]

            if actor_grad:
                cur_choice, cur_index, cur_entry, cur_prob, cur_dist = self.actor(query, nodes[veh_mask_i], 
                                                                        node_key_padding_mask_i, 
                                                                        deterministic, self.tau, 
                                                                        cur_chosen_idx, 
                                                                        cur_chosen_entry,
                                                                        force_chosen_mask)
            else:
                with torch.no_grad():
                    cur_choice, cur_index, cur_entry, cur_prob, cur_dist = self.actor(query, nodes[veh_mask_i], 
                                                                        node_key_padding_mask_i, 
                                                                        deterministic, self.tau, 
                                                                        cur_chosen_idx, 
                                                                        cur_chosen_entry,
                                                                        force_chosen_mask)
            # assert action_mask_i.shape[0] == cur_choice.shape[0]
            choice[veh_mask_i, veh_idx:veh_idx+1] = cur_choice
            index[veh_mask_i, veh_idx:veh_idx+1] = cur_index
            prob[veh_mask_i, veh_idx:veh_idx+1] = cur_prob
            entry[veh_mask_i, veh_idx:veh_idx+1] = cur_entry
            dists[veh_mask_i, veh_idx] = cur_dist
            value_mask[:, veh_idx] = veh_mask_i

            is_return_depot = (cur_index == (veh_num[veh_mask_i] + veh_idx).long()).squeeze() 
            global_indices = torch.nonzero(veh_mask_i).squeeze(1)[is_return_depot]
            key_mask = key_mask.clone()
            key_mask[global_indices, veh_idx] = False 
            
            if criticize:
                for key, value_head in self.critic.items():
                    value[key][veh_mask_i, veh_idx:veh_idx+1] = value_head(query, nodes[veh_mask_i], node_key_padding_mask_i).squeeze(1)

            node_key_padding_mask[choice[:, veh_idx].detach().bool()] = True

        prob = torch.log(prob)
        dist = torch.distributions.Categorical(dists[value_mask])
        ent = dist.entropy().mean()

        action = torch.stack([index, entry], dim=2)
        
        if criticize_only:
            return value
        if dist_only:
            return dist
        if criticize:
            return value, action, prob, new_hidden_state
        if eval_action:
            return prob, ent
        return action, new_hidden_state