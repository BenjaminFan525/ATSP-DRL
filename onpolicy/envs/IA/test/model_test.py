#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import random_split
from torch_geometric.loader.dataloader import Batch
import yaml
import random
import cProfile
import pstats
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../')) # 指向 MAIA
sys.path.insert(0, root_dir)
env_dir = os.path.abspath(os.path.join(current_dir, '../')) # 指向 IA
sys.path.insert(0, env_dir)

from onpolicy.config.config import get_config
from onpolicy.envs.IA.environment import FarmScheduleEnv
from onpolicy.envs.IA.simulator import EventDrivenSimulator

def parse_args(args, parser):
    parser.add_argument('--ac_config', type=str, default='/home/fanyx/MAIA/onpolicy/config/ac.yaml', help="Path to the ac config file")
    all_args = parser.parse_known_args(args)[0]  

    return all_args

def _t2n(x):
    return x.detach().cpu().numpy()

def main(args):
    profiler = cProfile.Profile()
    profiler.enable()

    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(str('cuda:3'))
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # config
    ac_config = all_args.ac_config
    if ac_config is not None:
        with open(ac_config, 'r') as f:
            ac_config = yaml.safe_load(f)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    DATA_DIR = "/home/fanyx/MAIA/onpolicy/dataset/Task_md/Train" 
    DATA_LIST = [] 

    for subdir in os.listdir(DATA_DIR):
        subdir = os.path.join(DATA_DIR, subdir)
        try:
            DATA_LIST += [os.path.join(subdir, x) for x in os.listdir(subdir) if x.split('.')[-1] != 'txt' and x.split('.')[-1] != 'pkl']
        except NotADirectoryError as e:
                print(f"Not a directory: {subdir}. Error: {e}")

    # DATA_LIST = random.sample(DATA_LIST, 100)
    # DATA_LIST = [DATA_LIST[0]]
    DATA_LIST = ['/home/fanyx/MAIA/onpolicy/envs/IA/test/18_52_12']
    # DATA_LIST = ['/home/fanyx/MAIA/onpolicy/dataset/Task_md/Test/1_3/3_46_12']
    env = FarmScheduleEnv(DATA_LIST, 
                          all_args.max_agent_num, 
                          all_args.episode_length)

    # run experiments
    from onpolicy.algorithms.gnn_mappo.gnn_mappo import MAPPO_Trainer as TrainAlgo
    from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy as Policy
    from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import GNN_Actor_Critic
    import json

    policy = Policy(all_args, ac_config,
                env.observation_space[0],
                env.share_observation_space[0],
                env.action_space[0],
                device = all_args.device)

    checkpoint_dir = '/home/fanyx/MAIA/onpolicy/scripts/results/IA/simple/gnn_mappo/train-new-gae-ppo3/run29/models/checkpoint_Epoch51.pt'
    checkpoint = torch.load(checkpoint_dir, map_location=all_args.device)
    policy.ac.load_state_dict(checkpoint['model'])  
    episodes = all_args.num_episodes
    policy.ac.eval()
    
    trainer = TrainAlgo(all_args, policy, device = device)

    for idx in range(len(DATA_LIST)):
        obs, done, info = env.reset()
        num_nodes = len(env.field.working_line_list) + env.veh_num
        num_actions = 0
        total_time = 0.0
        total_dist = 0.0
        total_cost = 0.0
        rnn_states = np.zeros(
                (all_args.max_agent_num, all_args.recurrent_N, all_args.hidden_size),
                dtype=np.float32)
        rnn_states = np.expand_dims(rnn_states, axis=0)

        for step in range(all_args.episode_length):
            # Sample actions
            trainer.prep_rollout()

            rnn_states[0][done == True] = np.zeros(((done == True).sum(), all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
            mask = np.ones((all_args.max_agent_num, 1), dtype=np.float32)
            mask[done == True] = np.zeros(((done == True).sum(), 1), dtype=np.float32)

            active_mask = np.zeros((all_args.max_agent_num, 1), dtype=np.float32)
            active_mask[info['active_agents'] == True] = np.ones(((info['active_agents'] == True).sum(), 1), dtype=np.float32)

            available_action = np.zeros((all_args.episode_length, 1), dtype=np.float32)
            available_action[info['available_actions'] == True] = np.ones(((info['available_actions'] == True).sum(), 1), dtype=np.float32)

            veh_num = info['veh_nums']

            global_nodes, nodes, node_padding_mask = policy.ac.encoder.nodes_encoder(Batch.from_data_list([env.working_graph]).to(device))
            
            action, rnn_states \
                = trainer.policy.act(
                                global_nodes, nodes, node_padding_mask,
                                np.expand_dims(obs, axis=0),
                                rnn_states,
                                np.expand_dims(mask, axis=0),
                                np.expand_dims(active_mask, axis=0),
                                np.expand_dims(available_action, axis=0),
                                np.expand_dims(np.array([veh_num]), axis=0),
                                deterministic=True)            
            # Obser reward and next obs
            # value = np.array(_t2n(value[all_args.obj]))
            action = np.array(_t2n(action))
            # action_log_prob = np.array(_t2n(action_log_prob))
            rnn_states = np.array(_t2n(rnn_states))
            obs, reward, done, info = env.step(action[0].tolist())

            num_actions += np.sum(action[0][:, 0] != -1)
            total_time += np.sum(env._get_reward()['t']*(1-active_mask.squeeze()))
            total_dist += np.sum(env._get_reward()['s']*(1-active_mask.squeeze()))
            total_cost += np.sum(env._get_reward()['c']*(1-active_mask.squeeze()))
            # print(f"Step: {step}\n Reward: {reward[all_args.obj].tolist()}\n Action: {action[0].tolist()}\n Dones: {mask.tolist()}\n Active_mask: {(1-active_mask).tolist()}\n")

            # if done.all():
            #     print(f"Episode {idx} finished.")
            #     break
        assert num_actions == num_nodes, f"Number of actions {num_actions} does not match number of nodes {num_nodes}."
        assert np.isclose(env.simulator.global_time, total_time), f"Total time {env.simulator.global_time} does not match sum of rewards {total_time}."
        assert np.isclose(sum([v.total_dist for v in env.simulator.vehicles]), total_dist), f"Total distance {sum([v.total_dist for v in env.simulator.vehicles])} does not match sum of rewards {total_dist}."
        assert np.isclose(sum([v.total_cost for v in env.simulator.vehicles]), total_cost), f"Total cost {sum([v.total_cost for v in env.simulator.vehicles])} does not match sum of rewards {total_cost}."
        print(f"Episode {idx} finished. Total time: {total_time}, Total distance: {total_dist}, Total cost: {total_cost}, Number of actions: {num_actions}")
        print(env.arrangements)

        env.render_all(save_path='/home/fanyx/MAIA/onpolicy/envs/IA/test/trajectory_result.png')
    
    # simulator = EventDrivenSimulator(env.field, env.car_cfg)
    # simulator.reset(env.arrangements, shuffle=True)
    # simulator.run()
    
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(20) # 打印耗时前20的函数

if __name__ == "__main__":
    main(sys.argv[1:])
