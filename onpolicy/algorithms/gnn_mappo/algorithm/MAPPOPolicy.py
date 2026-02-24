import torch
from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import GNN_Actor_Critic
from onpolicy.utils.util import update_linear_schedule, update_linear_anneal


class GNN_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, ac_cfg, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.anneal_final = args.anneal_final
        self.anneal_original = args.anneal_original

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.ac = GNN_Actor_Critic(**ac_cfg, device=device)
        self.actor_optimizer = torch.optim.Adam(self.ac.actor_param.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.ac.critic_param.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def _build_inputs(self, global_nodes, nodes, node_key_padding_mask, obs, rnn_states, masks, active_masks, available_actions, veh_nums):
        return {
            'graph': (global_nodes, nodes, node_key_padding_mask),
            'vehicles': torch.from_numpy(obs).to(self.device, dtype=torch.float32),
            'hidden_states': torch.from_numpy(rnn_states).to(self.device, dtype=torch.float32),
        },\
        {
            'key_mask': torch.from_numpy(masks).squeeze(-1).to(self.device) > 0.5,
            'node_key_padding_mask': torch.from_numpy(available_actions).squeeze(-1).to(self.device) > 0.5,
            'veh_key_padding_mask': torch.from_numpy(active_masks).squeeze(-1).to(self.device) > 0.5,
            'veh_num': torch.from_numpy(veh_nums).to(self.device)
        }

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def hyperparams_anneal(self, episode, episodes):
        """
        Anneal the temperature parameter tau to balance the exploration-exploitation tradeoff.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_anneal(self.ac, self.anneal_original, self.anneal_final, episode, episodes)

    def get_actions(self, global_nodes, nodes, node_key_padding_mask, obs, rnn_states, masks, active_masks, available_actions, veh_nums, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param graph_obs(PyG Data): graph input to the encoder.
        :param obs (np.ndarray): global agent inputs to the encoder.
        :param rnn_states: (np.ndarray) GRU states for sel_encoder.
        :param masks: (np.ndarray) denotes whether episode is terminated (1.0) or not (0.0).
        :param available_actions: (np.ndarray) denotes which actions are available to agent
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        
        data, info = self._build_inputs(global_nodes, nodes, node_key_padding_mask, obs, rnn_states, masks, active_masks, available_actions, veh_nums)
        values, actions, action_log_probs, rnn_states = self.ac(data, info, deterministic)

        return values, actions, action_log_probs, rnn_states

    def get_values(self, global_nodes, nodes, node_key_padding_mask, obs, rnn_states, masks, active_masks, available_actions, veh_nums):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """

        data, info = self._build_inputs(global_nodes, nodes, node_key_padding_mask, obs, rnn_states, masks, active_masks, available_actions, veh_nums)

        values = self.ac(data, info, criticize_only=True)

        return values

    def evaluate_actions(self, global_nodes, nodes, node_key_padding_mask, obs, rnn_states, masks, active_masks, available_actions, veh_nums, actions):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param graph_obs(PyG Data): graph input to the encoder.
        :param obs (np.ndarray): global agent inputs to the encoder.
        :param rnn_states: (np.ndarray) GRU states for sel_encoder.
        :param masks: (np.ndarray) denotes whether episode is terminated (1.0) or not (0.0).
        :param available_actions: (np.ndarray) denotes which actions are available to agent
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        
        data, info = self._build_inputs(global_nodes, nodes, node_key_padding_mask, obs, rnn_states, masks, active_masks, available_actions, veh_nums)

        action_log_probs, dist_entropy = self.ac(data, info, chosen_idx=torch.from_numpy(actions[:, :, 0]).long().to(self.device), chosen_entry=torch.from_numpy(actions[:, :, 1]).long().to(self.device), eval_action=True)

        return action_log_probs, dist_entropy
    
    def evaluate_values(self, global_nodes, nodes, node_key_padding_mask, obs, rnn_states, masks, active_masks, available_actions, veh_nums, actions):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param graph_obs(PyG Data): graph input to the encoder.
        :param obs (np.ndarray): global agent inputs to the encoder.
        :param rnn_states: (np.ndarray) GRU states for sel_encoder.
        :param masks: (np.ndarray) denotes whether episode is terminated (1.0) or not (0.0).
        :param available_actions: (np.ndarray) denotes which actions are available to agent
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        
        data, info = self._build_inputs(global_nodes, nodes, node_key_padding_mask, obs, rnn_states, masks, active_masks, available_actions, veh_nums)
        
        values = self.ac(data, info, chosen_idx=torch.from_numpy(actions[:, :, 0]).long().to(self.device), chosen_entry=torch.from_numpy(actions[:, :, 1]).long().to(self.device), actor_grad=False, criticize_only=True)

        return values
    def act(self, global_nodes, nodes, node_key_padding_mask, obs, rnn_states, masks, active_masks, available_actions, veh_nums, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        
        data, info = self._build_inputs(global_nodes, nodes, node_key_padding_mask, obs, rnn_states, masks, active_masks, available_actions, veh_nums)

        actions, rnn_states = self.ac(data, info, deterministic, criticize=False)
        return actions, rnn_states
