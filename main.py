
import numpy as np
import pickle
from environment import ScheduleEnv
import sys
import os
# from os.path import dirname, abspath
# sys.path.append(dirname(dirname(abspath(__file__))))
from MARL.runner import Runner
from MARL.common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, \
    get_reinforce_args, \
    get_commnet_args, get_g2anet_args
from utils.PDRs.shortestDistence import SDrules
import json
from datetime import datetime

np.random.seed(2)



# 强化学习决策函数，带入来自DRL的强化学习agent
def marl_agent_wrapper():


    args = get_common_args()

    if args.alg.find('coma') > -1:  # 判断模型的参数
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    # 加载调度环境
    env = ScheduleEnv()

    env.reset(args.n_agents)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    print("是否加载模型（测试必须）：", args.load_model, "是否训练：",args.learn)
    # args.replay_dir
    runner = Runner(env, args)
    
    if args.learn:
        runner.run(args.alg)  
    else:
        start_time = datetime.now()
        win_rate, reward, time , move_time = runner.evaluate()
        end_time = datetime.now()
        running_time = end_time - start_time
        print('Evaluate win_rate: {}, reward: {}, makespan: {}, move_times: {}, running_time: {}'.format(win_rate, reward, time, move_time, running_time))
    print("Exiting Main")


# 随机决策函数，用于测试环境
def random_agent_wrapper():

    results = {
            "makespan": [],
            "schedule_results": [],
            "win_tag":[],
            "win_rates": []
        }
    episodes = 100
    env = ScheduleEnv()
    n_agents = 8
    env.reset(n_agents)
    env_info = env.get_env_info()
    episode_limit = env_info["episode_limit"]
    win_num = 0
    start_time = datetime.now()
    for episode in range(episodes):

        env.reset(n_agents)
        is_terminal = False
        step = 0
        while not is_terminal and step < episode_limit:
            actions = []
            # 每次只调运空闲的agent
            for i in range(len(env.planes)):
                avail_actions = env.get_avail_agent_actions(i)
                tem_choose = []
                for k, eve in enumerate(avail_actions):
                    if eve == 1:
                        tem_choose.append(k)
                if len(env.sites)+1 in tem_choose or len(env.sites)+2 in tem_choose:  # i正忙
                    actions.append(len(env.sites))
                else:
                    if len(env.sites) in tem_choose:
                        tem_choose.remove(len(env.sites))
                    if tem_choose == []:
                        action = len(env.sites)
                    else:
                        action = np.random.choice(tem_choose, 1, False)[0]
                    if action < len(env.sites):
                        env.has_chosen_action(action, i)
                    actions.append(action)
            reward, is_terminal, info = env.step(actions)
            step += 1
        if is_terminal:
            win_num += 1
            results["schedule_results"].append(info["episodes_situation"])
        results["win_tag"].append(is_terminal)
        results["makespan"].append(info["time"])
        print("episode ", episode+1, " makespan: ", info["time"])
    win_rate = win_num/episodes
    average_makespan = sum(results["makespan"])/episodes
    end_time = datetime.now()
    running_time = end_time - start_time
    results["running_time"] = str(running_time)
    results["win_rates"].append(win_rate)
    print('Evaluate win_rate: {}, makespan: {}, running_time: {}, step: {}'.format(win_num/episodes, average_makespan, running_time, step))
    # 存储中间结果
    with open("result/random/test.json", 'w') as f:
        json.dump(results, f, indent=4)
    


def SDrules_agent_wrapper():
    n_agents = 8
    episodes = 20
    results = {
            "makespan": [],
            "schedule_results": [],
            "win_tag":[],
            "win_rates": []
        }
    sd_rules = SDrules()
    env = ScheduleEnv()
    env.reset(n_agents)
    env_info = env.get_env_info()
    episode_limit = env_info["episode_limit"]
    sites_locations = [env.sites[i].absolute_position for i in range(len(env.sites))]
    win_num = 0
    actions = []
    start_time = datetime.now()
    for episode in range(episodes):
        done = False
        env.reset(n_agents)
        speed = env.planes_obj.plane_speed
        step = 0
        while not done and step < episode_limit:
            actions = []
            agents_id_sequence = sd_rules.FIFO_generate_agents_sequence(len(env.planes))  # 智能体选择顺序
            # agents_id_sequence = sd_rules.MLF_generate_agents_sequence(env.planes)
            # agents_id_sequence = sd_rules.LLF_generate_agents_sequence(env.planes)

            for agent_id in agents_id_sequence:
                avail_actions = env.get_avail_agent_actions(agent_id)
                current_plane_location = env.planes[agent_id].position
                action = sd_rules.choose_action(agent_id, avail_actions, current_plane_location, sites_locations, speed)
                actions.append(action)
                if action < len(env.sites):
                    env.has_chosen_action(action, agent_id)
            # 按照action的顺序进行重新整理，因为环境需要按照顺序接受actions
            reorder_actions = [-1 for i in range(len(env.planes))]
            for i in range(len(env.planes)):
                reorder_actions[agents_id_sequence[i]] = actions[i]
            
            _, done, info = env.step(reorder_actions)
            step += 1
        if done:
            win_num += 1
            results["schedule_results"].append(info["episodes_situation"])
        results["win_tag"].append(done)
        results["makespan"].append(info["time"])
        print("episode ", episode+1, " makespan: ", info["time"])
    win_rate = win_num/episodes
    average_makespan = sum(results["makespan"])/episodes
    end_time = datetime.now()
    running_time = end_time - start_time
    results["running_time"] = str(running_time)
    results["win_rates"].append(win_rate)
    print('Evaluate win_rate: {}, makespan: {}, running_time: {}'.format(win_num/episodes, average_makespan, running_time))
    # 存储中间结果
    save_path = "result/SDrules/" + str(n_agents) +"/info.json"
    with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)


def Genetic_agent_wrapper():
    from GA import GeneticAlgorithm
    episodes = 20
    results = {
            "makespan": [],
            "average_makespan": [],
            "move_time":[],
            "average_move_time":[],
            "schedule_results": [],
            "win_tag":[],
            "win_rate": []
        }
    n_agents = 12
    env = ScheduleEnv()
    env.reset(n_agents)
    env_info = env.get_env_info()
    episode_limit = env_info["episode_limit"]
    win_num = 0
    start_time = datetime.now()
    for episode in range(episodes):
        done = False
        env.reset(n_agents)
        step = 0
        while not done and step < episode_limit:
            ga = GeneticAlgorithm(env, population_size=10, generations=5, mutation_rate=0.1)
            actions = ga.run()
            for i, action in enumerate(actions):
                if action < len(env.sites):
                        env.has_chosen_action(action, i)
            reward, done, info = env.step(actions)
            step += 1
        if done:
            win_num += 1
            results["schedule_results"].append(info["episodes_situation"])
            move_time = sum(job_trans[5] for job_trans in info["episodes_situation"])
            move_time = move_time / n_agents
        results["win_tag"].append(done)
        results["makespan"].append(info["time"])
        results["move_time"].append(move_time)
        print("episode ", episode+1, " makespan: ", info["time"])
    
    win_rate = win_num/episodes
    average_makespan = sum(results["makespan"])/episodes
    results["average_makespan"].append(average_makespan)
    average_move_time = sum(results["move_time"])/episodes
    results["average_move_time"].append(average_move_time)
    results["win_rate"].append(win_num/episodes)
    end_time = datetime.now()
    running_time = end_time - start_time
    results["running_time"] = str(running_time)
    print('Evaluate win_rate: {}, makespan: {}, move_times: {}, running_time: {}'.format(win_rate, average_makespan, average_move_time, running_time))
    # 存储中间结果
    save_path = "result/GA/" + str(n_agents) +"/info.json"
    with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
    

def Diff_Evolution_agent_wrapper():
    from DE import DifferentialEvolution
    episodes = 300
    results = {
            "makespan": [],
            "average_makespan": [],
            "move_time":[],
            "average_move_time":[],
            "schedule_results": [],
            "win_tag":[],
            "win_rate": []
        }
    n_agents = 8
    env = ScheduleEnv()
    env.reset(n_agents)
    env_info = env.get_env_info()
    episode_limit = env_info["episode_limit"]
    win_num = 0
    start_time = datetime.now()
    for episode in range(episodes):
        done = False
        env.reset(n_agents)
        step = 0
        # episode_density = 0
        while not done and step < episode_limit:
            de = DifferentialEvolution(env, population_size=10, generations=5)
            actions = de.run()
            for i, action in enumerate(actions):
                if action < len(env.sites):
                        env.has_chosen_action(action, i)
            reward, done, info = env.step(actions)
            step += 1
        if done:
            win_num += 1
            results["schedule_results"].append(info["episodes_situation"])
            move_time = sum(job_trans[5] for job_trans in info["episodes_situation"])
            move_time = move_time / n_agents
        results["win_tag"].append(done)
        results["makespan"].append(info["time"])
        results["move_time"].append(move_time)
        print("episode ", episode+1, " makespan: ", info["time"])
    
    win_rate = win_num/episodes
    average_makespan = sum(results["makespan"])/episodes
    results["average_makespan"].append(average_makespan)
    average_move_time = sum(results["move_time"])/episodes
    results["average_move_time"].append(average_move_time)
    results["win_rate"].append(win_num/episodes)
    end_time = datetime.now()
    running_time = end_time - start_time
    results["running_time"] = str(running_time)
    print('Evaluate win_rate: {}, makespan: {}, move_times: {}, running_time: {}'.format(win_rate, average_makespan, average_move_time, running_time))
    # 存储中间结果
    save_path = "result/DE/" + str(n_agents) +"/info.json"
    with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)




if __name__ == "__main__":
    marl_agent_wrapper()
    # random_agent_wrapper()
    # SDrules_agent_wrapper()
    # Genetic_agent_wrapper()
    # Diff_Evolution_agent_wrapper()
