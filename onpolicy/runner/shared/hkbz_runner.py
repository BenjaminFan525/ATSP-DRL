import time
import os
import json
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
from torch_geometric.loader.dataloader import Batch
import wandb
import imageio
import copy
from tqdm import tqdm
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer
import cProfile
import pstats

def _t2n(x):
    return x.detach().cpu().numpy()

class HKBZ_Runner(Runner):
    """Runner class to perform training, evaluation. and data collection for the IAs. See parent class for details."""
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.ac_config = config['ac_config']
        self.num_envs = config['num_envs']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.num_episodes = self.all_args.num_episodes
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.use_anneal = self.all_args.use_anneal
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.obj = self.all_args.obj
        self.reward_coef = self.all_args.reward_coef
        self.fuse_s = self.all_args.fuse_s
        self.fuse_epoch = self.all_args.fuse_epoch
        self.start_epoch = self.all_args.start_epoch
        self.auto_fuse = self.all_args.auto_fuse

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.checkpoint_dir = self.all_args.checkpoint_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.render_dir = str(self.run_dir / 'renders')
            if not os.path.exists(self.render_dir):
                os.makedirs(self.render_dir)
        from onpolicy.algorithms.gnn_mappo.gnn_mappo import MAPPO_Trainer as TrainAlgo
        from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy as Policy

        # share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        
        # policy network
        self.policy = Policy(self.all_args, self.ac_config,
                            device = self.device)

        if self.checkpoint_dir is not None:
            self.restore(self.checkpoint_dir)

        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        None, None, None)

    def run(self):   
        
        start = time.time()
        episodes = self.num_episodes
        self.total_num_steps = 0

        pbar = tqdm(range(episodes), 
              desc="Training",    
              unit="episode",     
              total=episodes,       
              ncols=160)
        for episode in pbar:
            # profiler = cProfile.Profile()
            # profiler.enable()
            # self.envs.shuffer_data()
            self.episode = episode

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            if self.use_anneal:
                self.trainer.policy.hyperparams_anneal(episode, episodes)

            for _ in range(self.num_envs):
                self.warmup()
                training_rewards = []
                for step in range(self.episode_length):
                    # Sample actions
                    values, actions, action_log_probs, rnn_states = self.collect(step)
                        
                    # Obser reward and next obs
                    obs, rewards, dones, infos = self.envs.step(actions)

                    data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states

                    # insert data into buffer
                    self.insert(data)

                # compute return and update network
                self.compute()

                train_infos = self.train()

                # eval
                if (self.total_num_steps == 0 or self.total_num_steps % self.eval_interval == 0) and self.use_eval:
                    train_infos['makespan'] = self.eval(render=True)
                
                self.total_num_steps += self.n_rollout_threads
                self.log_train(train_infos, self.total_num_steps)
                training_rewards.append(train_infos["rewards"] / self.reward_coef)
                pbar.set_description(f"[Episode {episode+1}]")
                pbar.set_postfix(
                    average_episode_rewards=np.mean(training_rewards),
                    total_num_steps=self.total_num_steps,
                    fps=int(self.total_num_steps / (time.time() - start)))

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # profiler.disable()
            # stats = pstats.Stats(profiler).sort_stats('cumtime')
            # stats.print_stats(30) # 打印耗时前20的函数

            # log information
            # if episode % self.log_interval == 0:
                # env_infos = {}
                # self.log_train(train_infos, total_num_steps * self.envs.num_fields)
                # self.log_env(env_infos, total_num_steps)

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
                            Batch.from_data_list(self.buffer.graph_obs[-1]),
                            self.buffer.rnn_states[-1],
                            self.buffer.active_masks[-1],
                            self.buffer.actions[-2, ..., 0],
                            self.buffer.actions[-2, ..., 1],
                            )
        next_values = _t2n(next_values).reshape(self.n_rollout_threads, self.num_agents, 1)

        hindsight_rewards = self.envs.get_rewards()
        self.buffer.rewards.fill(0.0)
        for env_idx, env_rewards in enumerate(hindsight_rewards):
            for (step_idx, agent_id), data in env_rewards.items():
                self.buffer.rewards[step_idx, env_idx, agent_id, 0] = data['reward'] * self.reward_coef

        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        return train_infos

    def warmup(self):
        self.trainer.prep_rollout()
        # reset env
        obs, dones, infos = self.envs.reset()

        for thread_idx in range(self.n_rollout_threads):
            self.buffer.graph_obs[0][thread_idx] = obs[thread_idx].clone()

        self.buffer.rnn_states[0] = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        
        self.buffer.masks[0] = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32).reshape(self.n_rollout_threads, self.num_agents, 1)
        self.buffer.masks[0][dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        self.buffer.active_masks[0] = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32).reshape(self.n_rollout_threads, self.num_agents, 1)
        self.buffer.active_masks[0][infos['active_agents'] == True] = np.ones(((infos['active_agents'] == True).sum(), 1), dtype=np.float32)

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states \
            = self.trainer.policy.get_actions(
                            Batch.from_data_list(self.buffer.graph_obs[step]),
                            self.buffer.rnn_states[step],
                            self.buffer.active_masks[step],
                            self.buffer.actions[step - 1, ..., 0] if step > 0 else -np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32),
                            self.buffer.actions[step - 1, ..., 1] if step > 0 else -np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
                            )
        # [self.envs, agents, dim]
        values = _t2n(value)
        actions = _t2n(action)
        action_log_probs = _t2n(action_log_prob)
        rnn_states = _t2n(rnn_states)

        return values, actions, action_log_probs, rnn_states

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum()), dtype=np.float32)

        active_masks = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        active_masks[infos['active_agents'] == True] = np.ones(((infos['active_agents'] == True).sum()), dtype=np.float32)
        
        self.buffer.graph_insert(obs, rnn_states, actions, action_log_probs, values, rewards, masks, active_masks)

    @torch.no_grad()
    def eval(self, render=False):
        eval_obs, eval_dones, eval_infos = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()

            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

            eval_active_masks = np.zeros_like(eval_masks)
            eval_active_masks[eval_infos['active_agents'] == True] = np.ones(((eval_infos['active_agents'] == True).sum(), 1), dtype=np.float32)

            eval_action, eval_rnn_states = self.trainer.policy.act(
                        Batch.from_data_list(eval_obs),
                        eval_rnn_states,
                        eval_active_masks,
                        eval_actions[..., 0] if eval_step > 0 else -np.ones((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32),
                        eval_actions[..., 1] if eval_step > 0 else -np.ones((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32),
                        deterministic=True)
            eval_actions = _t2n(eval_action)
            eval_rnn_states = _t2n(eval_rnn_states)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)

        return np.mean(self.eval_envs.get_episode_rewards())

    # TODO: add render function for FarmEnv
    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            active_masks = np.ones_like(masks)
            available_actions = np.ones((self.n_rollout_threads, self.num_agents, self.envs.action_space[0].n),dtype=np.float32)


            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                graph_obs, vector_obs = obs['graph'], obs['vector']
                action, rnn_states = self.trainer.policy.act(
                                                Batch.from_data_list(graph_obs),
                                                np.concatenate(vector_obs),
                                                np.concatenate(rnn_states),
                                                np.concatenate(masks),
                                                np.concatenate(active_masks),
                                                np.concatenate(available_actions),
                                                deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        model = self.trainer.policy
        save_path = os.path.join(self.save_dir, 'checkpoint_Epoch' + str(episode+1) + '.pt')
        checkpoint = {
            'episodes': episode + 1,
            'tau': model.ac.tau,
            'model': model.ac.state_dict(),
            'actor_optim': model.actor_optimizer.state_dict(),
            'critic_optim': model.critic_optimizer.state_dict(),
            # 'lagrangmdvrpn_multiplier': lagrangmdvrpn_multiplier
        }
        if hasattr(model, 'lagrangmdvrpn_multipliers'):
            checkpoint.update({
                'lagrangmdvrpn_multiplier': model.lagrangmdvrpn_multipliers,
                'lagrangmdvrpn_optimizer': model.lambda_optimizer.state_dict()
            })
        torch.save(checkpoint, save_path)

    def restore(self, checkpoint):
        """Restore policy's networks from a saved model."""
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.policy.ac.load_state_dict(checkpoint['model'])  
        self.policy.ac.tau = checkpoint['tau']
        self.all_args.anneal_original = checkpoint['tau']
        self.policy.actor_optimizer.load_state_dict(checkpoint['actor_optim'])
        self.policy.critic_optimizer.load_state_dict(checkpoint['critic_optim'])
        # self.episode = checkpoint['episodes']
        # self.num_episodes = self.num_episodes - checkpoint['episodes']
