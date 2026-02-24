import numpy as np
import math
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss, expand_slice
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
from torch_geometric.loader.dataloader import Batch
from collections import defaultdict

class MAPPO_Trainer():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.mini_batch_size = args.mini_batch_size
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.obj = args.obj
        self.n_rollout_threads = args.n_rollout_threads

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            # self.value_normalizer = ValueNorm(1).to(self.device)
            self.value_normalizer = ValueNorm(input_shape=1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        # if self._use_popart or self._use_valuenorm:
        #     self.value_normalizer.update(return_batch.reshape(-1, 1))
        #     error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
        #     error_original = self.value_normalizer.normalize(return_batch) - values
        # else:
        #     error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

        # if self._use_huber_loss:
        #     value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        #     value_loss_original = huber_loss(error_original, self.huber_delta)
        # else:
        #     value_loss_clipped = mse_loss(error_clipped)
        value_loss_original = mse_loss(error_original)

        # if self._use_clipped_value_loss:
        #     value_loss = torch.max(value_loss_original, value_loss_clipped)
        # else:
        value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def update_policy_net(self, graph_batch, sample, update_actor=True):
        obs_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, veh_nums_batch, rewards_batch = sample

        global_nodes_batch, nodes_batch, mask_batch = self.policy.ac.encoder.nodes_encoder(graph_batch)
        
        total_batch_size = obs_batch.shape[0]
        num_envs = global_nodes_batch.shape[0]
        episode_length = total_batch_size // num_envs

        global_nodes = expand_slice(global_nodes_batch, episode_length, total_batch_size)
        nodes = expand_slice(nodes_batch, episode_length, total_batch_size)
        node_key_padding_mask = expand_slice(mask_batch, episode_length, total_batch_size)
        
        # Reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy = self.policy.evaluate_actions(global_nodes, nodes, node_key_padding_mask,
                                                                        obs_batch, 
                                                                        rnn_states_batch,
                                                                        masks_batch, 
                                                                        active_masks_batch,
                                                                        available_actions_batch,
                                                                        veh_nums_batch,
                                                                        actions_batch)
        
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        # active_masks_batch = check(1-active_masks_batch.copy()).to(**self.tpdv)
        # masks_batch = check(masks_batch.copy()).to(**self.tpdv)
        loss_masks_batch = check(1-active_masks_batch).to(**self.tpdv)
        rewards_batch = check(rewards_batch).to(**self.tpdv)
        rewards = (rewards_batch*loss_masks_batch).sum()
        # actor update
        imp_weights = torch.exp(action_log_probs.unsqueeze(-1) - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        if self._use_policy_active_masks:
            # maximaize the minimum of the surrogate losses
            policy_action_loss = (torch.sum(torch.max(surr1, surr2),
                                            dim=-1,
                                            keepdim=True) * loss_masks_batch).sum() / loss_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.ac.actor_param.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.ac.actor_param.parameters())

        self.policy.actor_optimizer.step()
        return {
            "policy_loss": policy_loss.item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "dist_entropy": dist_entropy.item(),
            "advantages": adv_targ.mean().item(),
            "rewards": rewards.item() / num_envs
        }

    def update_value_net(self, graph_batch, sample):
        obs_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, veh_nums_batch, rewards_batch = sample
        
        # critic update
        global_nodes_batch, nodes_batch, mask_batch = self.policy.ac.encoder.nodes_encoder(graph_batch)

        total_batch_size = obs_batch.shape[0]
        num_envs = global_nodes_batch.shape[0]
        episode_length = total_batch_size // num_envs

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        loss_masks_batch = check(1-active_masks_batch).to(**self.tpdv)

        global_nodes = expand_slice(global_nodes_batch, episode_length, total_batch_size)
        nodes = expand_slice(nodes_batch, episode_length, total_batch_size)
        node_key_padding_mask = expand_slice(mask_batch, episode_length, total_batch_size)

        # Reshape to do in a single forward pass for all steps
        values = self.policy.evaluate_values(global_nodes, nodes, node_key_padding_mask,
                                                                              obs_batch, 
                                                                              rnn_states_batch,
                                                                              masks_batch, 
                                                                              active_masks_batch,
                                                                              available_actions_batch,
                                                                              veh_nums_batch,
                                                                              actions_batch)

        value_loss = self.cal_value_loss(values[self.obj].unsqueeze(-1), value_preds_batch, return_batch, loss_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.ac.critic_param.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.ac.critic_param.parameters())

        self.policy.critic_optimizer.step()

        return {
            "value_loss": value_loss.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "value_mean": values[self.obj].mean().item()
        }

    def ppo_update(self, graph_batch, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        train_info = defaultdict(list)
        if update_actor:
            self.policy.ac.eval()
            self.policy.ac.sel_enc.train()
            for _ in range(self.ppo_epoch):
                policy_results = self.update_policy_net(graph_batch, sample, update_actor)
                for k, v in policy_results.items():
                    train_info[k].append(v)
            self.policy.ac.train()

        for _ in range(self.ppo_epoch):
            value_results = self.update_value_net(graph_batch, sample)
            for k, v in value_results.items():
                train_info[k].append(v)
        
        return {k: np.mean(v) if v else 0.0 for k, v in train_info.items()}

    def train(self, graph_obs, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        # if self._use_popart or self._use_valuenorm:
        #     advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        # else:
        advantages = buffer.returns - buffer.value_preds[:-1]
        # advantages_copy = advantages.copy()
        # advantages_copy[(1-buffer.active_masks[:-1]) == 0.0] = 0
        # advantages_copy[(1-buffer.active_masks[:-1]) == 0.0] = np.nan
        # mean_advantages = np.nanmean(advantages_copy)
        # std_advantages = np.nanstd(advantages_copy)
        # advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
    
        train_info = defaultdict(float)
    
        data_generator = buffer.graph_recurrent_generator(advantages, self.mini_batch_size)
        for sample in data_generator:
            update_results = self.ppo_update(
                Batch.from_data_list([graph_obs[i] for i in sample[-1]]).to(self.device), 
                sample[:-1], 
                update_actor
            )
            
            for k, v in update_results.items():
                train_info[k] += v

        num_mini_batch = math.ceil(self.n_rollout_threads / self.mini_batch_size)
        for k in train_info.keys():
            train_info[k] /= num_mini_batch
 
        return train_info
    
    def prep_training(self):
        self.policy.ac.train()

    def prep_rollout(self):
        self.policy.ac.eval()
