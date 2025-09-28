import numpy as np
import os
from MARL.common.rollout import RolloutWorker, CommRolloutWorker
from MARL.agent.agent import Agents, CommAgents
from MARL.common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import sys
import json
from datetime import datetime


class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []
        self.results = {
            "evaluate_reward": [],
            "average_reward": [],
            "evaluate_makespan": [],
            "average_makespan": [],
            "evaluate_move_time": [],
            "average_move_time": [],
            "schedule_results": [],
            "win_rates": [],
            "train_reward": [],
            "train_makespan": [],
            "train_move_time": [],
            "loss": []
        }

        self.save_path = self.args.result_dir + '/' + args.alg + '/' + str(args.n_agents)+'_agents' + '/' + args.result_name
        print(self.save_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, alg):
        start_time = datetime.now()
        file_name = f"info.json"
        file_path = os.path.join(self.save_path, file_name)
        
        train_steps = 0
        for_gantt_data =[]
        r_s = [0]
        evaluate_times = 1
        for epoch in range(self.args.n_epoch):

            if epoch % self.args.evaluate_cycle == 0 and epoch != 0:
                print('\nevaluate times:', evaluate_times, end=' ')
                win_rate, reward, time, move_time = self.evaluate()
                print('Evaluate win_rate: {}, reward: {}, makespan: {}, move_times: {}'.format(win_rate, reward, time, move_time))
                self.win_rates.append(win_rate)
                self.episode_rewards.append(reward)
                evaluate_times += 1

            episodes = []
            r_s = []
            t_s = []

            for episode_idx in range(self.args.n_episodes):
                episode, train_reward, train_time, _, _, train_move_time = self.rolloutWorker.generate_episode(episode_idx)
                self.results['train_reward'].append(train_reward)
                self.results['train_makespan'].append(train_time)
                self.results['train_move_time'].append(train_move_time)
                episodes.append(episode)
                r_s.append(sum(episode['r'][0])[0])
                t_s.append(train_time)
            

            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                loss = self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                self.results['loss'].append(loss)
                train_steps += 1
            else:
                # 这几个类型的算法需要进行buffer的存储
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    loss = self.agents.train(mini_batch, train_steps)
                    self.results['loss'].append(loss)
                    train_steps += 1

            # 显示输出
            text = '\rRun {}, train epoch {}, ave_rewards {}, ave_makespan {}'
            sys.stdout.write(text.format(alg, epoch+1, sum(r_s)/len(r_s), sum(t_s)/len(t_s)))
            sys.stdout.flush()

        # 保存结果文件
        end_time = datetime.now()
        running_time = end_time - start_time
        self.results["running_time"] = str(running_time)
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=4)


    def evaluate(self):
        win_number = 0
        reward = 0
        time = 0
        move_time = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, episode_time, win_tag, for_gant, episode_move_time = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            self.results['evaluate_reward'].append(episode_reward)
            self.results['evaluate_makespan'].append(episode_time)
            self.results['evaluate_move_time'].append(episode_move_time)
            self.results['schedule_results'].append(for_gant)
            reward += episode_reward
            time += episode_time
            move_time += episode_move_time
            if win_tag:
                win_number += 1
        win_rate = win_number / self.args.evaluate_epoch
        reward = reward / self.args.evaluate_epoch
        time = time / self.args.evaluate_epoch
        move_time = move_time / self.args.evaluate_epoch
        self.results['average_reward'].append(reward)
        self.results['average_makespan'].append(time)
        self.results['average_move_time'].append(move_time)
        self.results['win_rates'].append(win_rate)
        if self.args.load_model and not self.args.learn:
            file_name = f"evaluat.json"
            file_path = os.path.join(self.save_path, file_name)
            with open(file_path, 'w') as f:
                json.dump(self.results, f, indent=4)

        return win_rate, reward, time , move_time

