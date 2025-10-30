import utils
print(utils.__file__)
from utils.env_utils import Sites
from utils.job import Jobs
from utils.task import Task
from utils.plane import Planes
from utils import util
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 整个环境类
class ScheduleEnv(gym.Env):
    environment_name = "Boat Schedule"

    def __init__(self):
        # 类变量的声明
        self.sites = []
        self.jobs = []
        self.task = []
        self.planes = []
        self.state = [[]]
        self.done = False
        self.state_left_time = []
        self.episode_time_slice = []  # 每个step消耗时间组成的episode的时间列表
        self.plane_speed = 0  # 运行速度
        self.id = "Boat Schedule"
        self.job_record_for_gant = []  # 用于存储调度中间过程四元组
        self.sites_state_global = None
        # 全局状态，观测
        self.state4marl = None
        self.obs4marl = None


    def initialize(self, n_agents):
        sites_obj = Sites(n_agents)
        self.sites_obj = sites_obj
        jobs_obj = Jobs()
        task_obj = Task()
        self.planes_obj = Planes(n_agents)
        self.sites = sites_obj.sites_object_list
        self.jobs = jobs_obj.jobs_object_list
        self.task = task_obj.simple_task_object
        self.planes = self.planes_obj.planes_object_list
        self.state = [[len(self.planes)+1, [1 if j in self.sites[i].resource_ids_list else 0 for j in range(len(self.jobs))]] for i in range(len(self.sites))]
        self.sites_state_global = [-1 for i in range(len(self.sites))]  # 站位状态（处理作业id，-1代表该站位没有被安排保障任务）
        self.job_record_for_gant = []  # 用于存储调度中间过程四元组
        self.done = False
        self.state_left_time = np.array([0 for i in range(len(self.sites))])    # 每个站位剩余处理时间
        self.episode_time_slice = []    # 当前episode所有step消耗的时间
        self.plane_speed = self.planes_obj.plane_speed  # 飞机速度
        self.obs4marl = [[] for i in range(len(self.planes))]   # 每个飞机的观测列表
        self.current_finishing_jobs = 0   # 已完成作业个数
        self.step_count = 0   # 记录step


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, n_agents):
        self.initialize(n_agents)  # 初始化参数
        info = {
            "sites": [[self.sites[i].absolute_position,          # 站位位置
                       self.state[i][0],                         # 占用该站位的飞机id，初始为空闲
                       self.state[i][1]                          # 站位可以执行作业的资源one-hot向量
                       ] for i in range(len(self.sites))],
            "planes": [[self.planes[i].left_job[0].index_id,     # 飞机下一个作业的id
                        self.jobs[self.planes[i].left_job[0].index_id].time_span,   # 飞机下一个作业的时间
                        len(self.planes[i].left_job)             # 飞机的剩余作业数量
                        ] if len(self.planes[i].left_job) != 0
                       else [
                len(self.planes[i].static_job_list),
                0,
                len(self.planes[i].left_job)
            ] for i in range(len(self.planes))],
            "planes_obj": self.planes
        }
        state = self.conduct_state(info)

        return state

    def conduct_state(self, info):
        res = []    # 全局状态
        temp = []   # 站位占用状态

        for eve in info["sites"]:
            res.append(eve[1])     # 占用该站位的飞机id，空闲为飞机数+1
            temp.append(eve[1])

        res.extend(self.state_left_time)    # 剩余工作时间

        for eve in info["sites"]:
            res += eve[2]           # 将每个站位可以执行的作业资源one-hot向量拼接起来
        
        for i, eve in enumerate(info["planes"]):
            res += [eve[0], eve[1], eve[2]]         # 飞机剩余作业数量，下一个作业id，下一个作业时间
            temp_obs = [eve[0], eve[1], eve[2]]   # 存储每个飞机的观测
            for l, eve_1 in enumerate(temp):
                if self.planes[i].left_job == []:   # 若飞机的剩余作业列表为空，观测为0
                    temp_obs.extend([0,0])
                else:
                    # 观测为到每个站位的时间（无该作业资源的为0），剩余工作时间（t or 0）
                    if self.planes[i].left_job[0].index_id in self.sites[l].resource_ids_list:  # 站位是否可以执行该飞机下一个作业
                        temp_obs.append(util.count_path_on_road(self.planes[i].position, self.sites[l].absolute_position, self.plane_speed))
                    else:
                        temp_obs.append(0)
                    if eve_1 == len(self.planes)+1:  # 站位是否空闲
                        temp_obs.append(0)
                    else:
                        temp_obs.append(self.state_left_time[l])
            self.obs4marl[i] = temp_obs
        
        # 飞机是否正忙也加到观测里
        current_working_plane_ids = []  # 正忙的飞机
        for eve in self.state:
            if eve[0] != len(self.planes)+1:   # 若站位不空闲
                current_working_plane_ids.append(eve[0])    # 存储该站位占用飞机id
        for k in range(len(self.planes)):
            if k in current_working_plane_ids:  # 若该飞机正在忙
                # 正忙状态下的观测为除了自身的观测都是0
                self.obs4marl[k] = [0 if kk >= 3 else self.obs4marl[k][kk] for kk in range(len(self.obs4marl[k]))]
                self.obs4marl[k].append(1)  # 最后一位为1
            else:
                self.obs4marl[k].append(0)  # 飞机空闲，最后一位为0
        
        obslen = len(self.obs4marl[0])
        zero_obs = [0 for i in range(obslen)]

        # 若飞机已完成所有作业，观测为0
        for i, plane in enumerate(self.planes):
            if len(plane.left_job) == 0:
                self.obs4marl[i] = zero_obs

        self.state4marl = np.array(res)
        return np.array(res)
    
    # 检查动作的合法性
    def check_inflict_action(self, action):
        res = []

        for eve in action:
            if eve == len(self.sites) or eve == len(self.sites)+1 or eve == len(self.sites)+2:
                res.append(eve)
            else:
                if eve not in res:
                    res.append(eve)
                else:
                    raise Exception("sloppy error in actions", action)  # 若有选择同一个站位的动作则报错
        return res

    # 将action中的正忙和完成换成等待，并计算当前等待动作数量
    def action_replace(self, action):
        res = []
        real_conflict_num = 0
        for eve in action:
            if eve == len(self.sites)+1 or eve == len(self.sites)+2:
                res.append(len(self.sites))
            else:
                if eve == len(self.sites):
                    real_conflict_num += 1  # 等待动作数量，用于计算reward
                res.append(eve)
        return res, real_conflict_num

    # 交互函数
    def step(self, action):
        self.step_count += 1
        action, real_conflict_num = self.action_replace(action) # 将action中的正忙和完成换成等待
        assert len(action) == len(self.planes)
        rewards = [0 for eve in action]
        max_time_on_roads = [0 for eve in action]
        count_for_reward = 0    # 成功执行作业的个数
        action = self.check_inflict_action(action)
        time_span_increase = np.array([0 for eve in self.sites])    # 每个站位当前step的工作时间列表
        move_times = 0    # 总移动次数
        
        # 执行动作
        for i, site_id in enumerate(action):
            if site_id == len(self.sites):  # 动作为等待
                pass

            else:  # 安排保障任务    
                # 飞机在两个站位之间的调运时间(距离/速度)，为0代表其留在了原地加工
                time_on_road = util.count_path_on_road(self.planes[i].position,
                                                        self.sites[site_id].absolute_position.tolist(), self.plane_speed)
                if time_on_road != 0:
                    move_times += 1
                    
                # 保存调度六元组（到当前step选择动作之前的总消耗时间，飞机当前step进行的作业id，飞机当前step选择的站位id，飞机id，作业完成时间，运输时间）
                if type(site_id) == int:
                    self.save_env_info(
                        (sum(self.episode_time_slice), self.planes[i].left_job[0].index_id, site_id, i, self.planes[i].left_job[0].time_span, int(time_on_road)))
                else:
                    self.save_env_info((sum(self.episode_time_slice), self.planes[i].left_job[0].index_id, site_id.item(), i, self.planes[i].left_job[0].time_span, int(time_on_road)))
                # 飞机执行该作业并获取执行时间
                temp_time = self.planes[i].execute_task(self.planes[i].left_job[0], self.sites[site_id])
                # 每个站位当前step的工作时间=作业执行时间+飞机在路上的时间
                time_span_increase[site_id] = temp_time + time_on_road
                count_for_reward += 1
                # 存储每个飞机在路上的时间，为构造每个飞机的reward存储maxtime，方便归一化
                max_time_on_roads[i] = time_on_road

        real_did = 0    # 实际执行作业的动作个数
        for eve in action:
            if eve < len(self.sites):
                real_did += 1
        # 计算个体奖励
        for i, site_id in enumerate(action):
            if site_id == len(self.sites):  # 代表这个飞机不安排保障任务
                rewards[i] = - 30  # 不执行作业的飞机奖励为-30
            else:  # 安排保障任务
                if rewards[i] == 0:
                    rewards[i] = -(max_time_on_roads[i]+0.1)/(max(max_time_on_roads)+0.1)
                else:
                    pass

        # 更新当前的站位剩余工作时间
        self.state_left_time = self.state_left_time + time_span_increase

        min_time = util.min_but_zero(self.state_left_time)  # 若当前站位剩余工作时间不全为0则返回非0的最小时间，否则返回0
        self.episode_time_slice.append(min_time)  # 这个step消耗的时间为最小的站位工作时间
        self.state_left_time = util.advance_by_min_time(min_time, self.state_left_time)  # step推进，将剩余工作时间中非0的都减去min_time

        # 更新站位状态,主要是检查哪些站位工作完了
        # state transition 2：当前step时间推进完再进行一次状态转移
        for i, eve_time in enumerate(self.state_left_time):
            if eve_time == 0:   # 若站位剩余工作时间为0
                self.sites_state_global[i] = -1  # 站位状态空闲
                self.state[i][0] = len(self.planes)+1
                self.state[i][1] = [1 if j in self.sites[i].resource_ids_list else 0 for j in range(len(self.jobs))]
            else:
                assert self.state[i][0] != len(self.planes)+1

        # 判断当前episode是否完成了
        is_all_done = [-1 for eve in self.planes]
        for i, plane in enumerate(self.planes):
            if len(plane.left_job) == 0:
                is_all_done[i] = 0
        if sum(is_all_done) == 0:
            self.done = True
        else:
            self.done = False

        # 计算总奖励
        if self.done:   # 所有飞机的作业都完成了
            reward = 6000 / (sum(self.episode_time_slice) + max(self.state_left_time))
            # print(11, reward)
        else:   # 还没完成
            reward = real_did - self.step_count/60 - real_conflict_num*2
            
        info = {
            "sites": [[self.sites[i].absolute_position,
                       self.state[i][0],
                       self.state[i][1]
                       ] for i in range(len(self.sites))],
            "planes": [[self.planes[i].left_job[0].index_id,
                        self.jobs[self.planes[i].left_job[0].index_id].time_span,
                        len(self.planes[i].left_job)
                        ] if len(self.planes[i].left_job) != 0  
                       else [
                        10,
                        0,
                        len(self.planes[i].left_job)            # 飞机已完成作业状态[[10,0,0]...]
                    ]for i in range(len(self.planes))],
            "planes_obj": self.planes
        }
        state = self.conduct_state(info)
        return reward, self.done, {"time": sum(self.episode_time_slice)+max(self.state_left_time),
                                    "left": self.state_left_time,
                                    "original_state": self.state,
                                    "planes_obj": self.planes,
                                    "rewards": rewards,
                                    "sites_state_global": self.sites_state_global,
                                   "episodes_situation": self.job_record_for_gant
                                   }
    # 返回飞机可用的动作
    def get_avail_agent_actions(self, agent_id):
        # 检查飞机是否处于正忙状态
        for eve in self.state:
            if agent_id == eve[0]:  # 代表此飞机还在处于加工状态
                return [0 for i in range(len(self.sites))] + [0, 1, 0]
        # 如果飞机准备进行下一步操作则执行下部分程序
        res = [0 for i in range(len(self.sites))]    # 存储该飞机可以选择的站位
        for i, eve in enumerate(self.sites_state_global):
            if eve == -1:   # 该站位没有安排作业
                if len(self.planes[agent_id].left_job) != 0:    # 该飞机有剩余作业
                    # 判断该飞机下一个要完成的作业是否被包含在了站位可执行作业列表中
                    if self.planes[agent_id].left_job[0].index_id in self.sites[i].resource_ids_list:
                        res[i] = 1  # 该站位可用
                else:  # 证明此时的这个飞机已经完成了所有的作业
                    return [0 for i in range(len(self.sites))] + [0, 0, 1]
        return res + [1, 0, 0]

    # state transition 1：选择动作之后先进行一次状态转移
    def has_chosen_action(self, action_id, agent_id):
        if self.planes[agent_id].left_job[0].index_id in self.sites[action_id].resource_ids_list:  # 如果飞机选择的战位可以进行他的第一个剩余作业
            self.sites_state_global[action_id] = self.planes[agent_id].left_job[0].index_id  # 更新战位处理的作业id
            # 注意这里是在选择合理的动作而不是已经做了动作
            self.sites_obj.update_site_resources(action_id, self.planes[agent_id].left_job[0].index_id)
            self.state[action_id][0] = agent_id  # 更新站位状态，被某个飞机占用了
            self.state[action_id][1] = [1 if j in self.sites[action_id].resource_ids_list else 0 for j in range(len(self.jobs))]  # 更新站位资源抢占状态
        else:   # 飞机选择了不能进行该作业的站位报错
            raise Exception("不合理的动作没有mask", self.sites_state_global, action_id, self.planes[agent_id].left_job[0].index_id, self.sites[action_id].resource_ids_list)


    def save_env_info(self, job_transition):
        self.job_record_for_gant.append(job_transition)

    def get_state(self):
        assert self.state4marl is not None
        return self.state4marl

    def get_obs(self):
        # assert self.obs4marl is not None and self.obs4marl != []
        agents_obs = [self.get_obs_agent(i) for i in range(len(self.planes))]
        return agents_obs

    def get_obs_agent(self, agent_id):
        return self.obs4marl[agent_id]

    def get_env_info(self):
        return {
            "n_actions": len(self.sites) + 3,
            "n_agents": len(self.planes),
            "state_shape": len(self.get_state()),
            "obs_shape": len(self.get_obs()[0]),
            "episode_limit": len(self.planes)*10
        }

        
