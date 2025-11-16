import numpy as np
import torch
from torch.distributions import one_hot_categorical
import os
from arrangement import arrange_devices

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        if self.args.result_dir != '':
            self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        # 开始收集与环境交互的情况
        # 收集了8个东西：obs, action, reward, state, avail_action, action_onehot, 结束标志, padding
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        # win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        plane_num_per_batch = self.n_agents       
        landing_list = [item for item in list(range(0, 120*plane_num_per_batch, 120))]
        step_time = 0
        
        while not terminated and step < self.episode_limit:
            if step in landing_list or step_time == 0:
                obs = self.env.get_obs(self.n_agents)  # [[],[],...]
                state = self.env.get_state(self.n_agents) # []
                actions = [0 for _ in range(self.n_agents)]
                avail_actions = [self.env.get_avail_agent_actions() for _ in range(self.n_agents)]
                actions_onehot = [np.zeros(self.args.n_actions) for _ in range(self.n_agents)]
                
                pidx = (step % 3600) // 120
                if step in landing_list:
                    plane_cfg = {
                        'velocity': 5,
                        'site': self.env.sites['Z'],
                        'fuel': np.random.randint(0, 30),
                        'jobs': self.env.jobs.values()
                    }
                    self.env.add_planes([{'batch': 0, 'idx': pidx, **plane_cfg}]) 

                action = {}
                action["planes"] = [[[] for _ in range(self.n_agents)] for _ in range(plane_num_per_batch)]
                chosen_actions = []
                # 选择动作
                for plane_id, plane in self.env.planes.items():
                    bidx_, pidx_ = int(plane_id.split('_')[1]), int(plane_id.split('_')[2])
                    plane_action = [None, None]

                    avail_action = self.env.get_avail_agent_actions(plane_id)  # 获取该agent可用动作列表

                    if len(self.env.planes) and all([plane.is_completed_all_jobs() for plane_id, plane in self.env.planes.items() if int(plane_id.split('_')[1]) == 0]):
                        avail_action[-4:-1] = [1 if not site.is_occupied else 0 for site in self.env.sites.values() if site.code in ['29', '30', '31']]
                    for a in chosen_actions:
                        avail_action[a] = 0
                    # 全0时保证等待
                    if all(x == 0 for x in avail_action):    
                        avail_action[-1] = 1
                    agent_action = self.agents.choose_action(obs[pidx_], last_action[pidx_],pidx_,                                                    avail_action, epsilon, evaluate)
                    # 除去等待的情况
                    if agent_action != len(avail_action) -1:
                        chosen_actions.append(agent_action)
                        plane_action[0] = list(self.env.sites.keys())[agent_action]
                    jobs = plane.get_avail_jobs(self.env.sites[plane_action[0]]) if plane_action[0] else []
                    if jobs and plane.is_idle(): 
                        plane_action[1] = sorted(jobs, key=lambda x: plane.jobs[x].time, reverse=True)[0]
                    if len(self.env.planes) == 1 and plane.is_completed_all_jobs():
                        if plane_action[0] == None:
                            if plane.is_idle():
                                print(6666)
                    action["planes"][bidx_][pidx_] = plane_action
  
                    # action的one-hot向量
                    action_onehot = np.zeros(self.args.n_actions)
                    action_onehot[agent_action] = 1
                    actions[pidx_] = agent_action
                    actions_onehot[pidx_] = action_onehot
                    avail_actions[pidx_] = avail_action
                    last_action[pidx_] = action_onehot             
                
                action["devices"] = {}
                for device_type, devices in self.env.mobile_devices.items():
                    action["devices"][device_type] = [device.site.code for device in devices]
                
                for job_code, waiting_sites in self.env.waiting_sites.items():
                    if len(waiting_sites) > 0:
                        devices = self.env.get_idle_devices(self.env.jobs[job_code].resources)
                        if len(devices) > 0:
                            planes = [plane for plane in self.env.planes.values() if plane.site.code in waiting_sites]
                            assignments = arrange_devices(devices, planes)
                            for device_type, device_code, target_site in assignments:
                                for idx, device in enumerate(self.env.mobile_devices[device_type]):
                                    if device.resource.code == device_code:
                                        action["devices"][device_type][idx] = target_site
                                        waiting_sites.remove(target_site)
                                        break
                
                
                # 进行一个step，获取总reward，是否结束，交互信息
                reward, terminated = self.env.step(action, self.env.step_time - step_time, self.n_agents)
                step_time = self.env.step_time

                o.append(obs)
                s.append(state)
                u.append(np.reshape(actions, [self.n_agents, 1]))
                u_onehot.append(actions_onehot)
                avail_u.append(avail_actions)
                r.append([reward])
                terminate.append([terminated])
                padded.append([0.])
                episode_reward += reward
                # 更新epsilon
                if self.args.epsilon_anneal_scale == 'step':
                    epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

            if terminated:
                print(f"All planes have taken off. Total time: {step}, Reward: {reward}")
                break
            elif step_time == 0:
                continue
            step += 1
            step_time -= 1

        # last obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = [self.env.get_avail_agent_actions() for _ in range(self.n_agents)]
        for plane_id, plane in self.env.planes.items():
            avail_action = self.env.get_avail_agent_actions(plane_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]


        if step < self.episode_limit:
            for i in range(step, self.episode_limit):
                o.append(np.zeros((self.n_agents, self.obs_shape)))
                u.append(np.zeros([self.n_agents, 1]))
                s.append(np.zeros(self.state_shape))
                r.append([0.])
                o_next.append(np.zeros((self.n_agents, self.obs_shape)))
                s_next.append(np.zeros(self.state_shape))
                u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
                padded.append([1.])
                terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        return episode, episode_reward, step, 0, 0 , 0


# RolloutWorker for communication
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset(self.n_agents)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        
        for_gantt = []
        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            # get the weights of all actions for all agents
            weights = self.agents.get_action_weights(np.array(obs), last_action)

            # choose action for each agent
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(weights[agent_id], avail_action, epsilon, evaluate)
                
                if action < len(self.env.sites):  # 更新环境状态
                    self.env.has_chosen_action(action, agent_id)

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
            if terminated:
                for_gantt = info["episodes_situation"]
        win_tag = terminated
        print("step:", step)
        move_time = sum(job_trans[5] for job_trans in info["episodes_situation"])
        move_time = move_time / self.n_agents
        # last obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        if step < self.episode_limit:
            for i in range(step, self.episode_limit):
                o.append(np.zeros((self.n_agents, self.obs_shape)))
                u.append(np.zeros([self.n_agents, 1]))
                s.append(np.zeros(self.state_shape))
                r.append([0.])
                o_next.append(np.zeros((self.n_agents, self.obs_shape)))
                s_next.append(np.zeros(self.state_shape))
                u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
                padded.append([1.])
                terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        return episode, episode_reward, info["time"], win_tag, for_gantt , 0
