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
        
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
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
        error_original = return_batch - values
        value_loss_original = mse_loss(error_original)
        value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def update_policy_net(self, sample, update_actor=True):
        graph_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, last_op_batch, last_site_batch, rewards_batch = sample
        
        # Reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy = self.policy.evaluate_actions(graph_batch,
                                                                      rnn_states_batch,
                                                                      active_masks_batch,
                                                                      last_op_batch,
                                                                      last_site_batch,
                                                                      actions_batch)
        
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        rewards_batch = check(rewards_batch).to(**self.tpdv)
        rewards = (rewards_batch*active_masks_batch).sum() / active_masks_batch.sum()
        # actor update
        imp_weights = torch.exp(action_log_probs.unsqueeze(-1) - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        
        if self._use_policy_active_masks:
            # maximaize the minimum of the surrogate losses
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
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
            "rewards": rewards.item()
        }

    def update_value_net(self, sample):
        graph_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, last_op_batch, last_site_batch, rewards_batch = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values = self.policy.evaluate_values(graph_batch,
                                             rnn_states_batch,
                                             active_masks_batch,
                                             last_op_batch,
                                             last_site_batch,
                                             actions_batch)

        value_loss = self.cal_value_loss(
            values.view(-1, 1), 
            value_preds_batch.view(-1, 1), 
            return_batch.view(-1, 1), 
            active_masks_batch.view(-1, 1)
        )

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
            "value_mean": values.mean().item()
        }

    def ppo_update(self, sample, update_actor=True):
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
                policy_results = self.update_policy_net(sample, update_actor)
                for k, v in policy_results.items():
                    train_info[k].append(v)
            self.policy.ac.train()

        for _ in range(self.ppo_epoch):
            value_results = self.update_value_net(sample)
            for k, v in value_results.items():
                train_info[k].append(v)
        
        return {k: np.mean(v) if v else 0.0 for k, v in train_info.items()}

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        # advantages_copy = advantages.copy()
        # advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        # mean_advantages = np.nanmean(advantages_copy)
        # std_advantages = np.nanstd(advantages_copy)
        # advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
    
        train_info = defaultdict(float)
    
        data_generator = buffer.graph_recurrent_generator(advantages, self.mini_batch_size)
        for sample in data_generator:
            update_results = self.ppo_update(
                sample, 
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
