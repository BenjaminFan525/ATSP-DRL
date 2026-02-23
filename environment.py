# grid_world_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.agent import Plane, Device
from utils.env_utils import Job, Resource, Site
import json
import matplotlib.pyplot as plt
from gymnasium.utils import seeding
import math


# 调度环境类：基于Gymnasium的多智能体强化学习环境，用于模拟飞机在机场站点的调度过程
class ScheduleEnv(gym.Env):
    environment_name = "Plane Schedule"
    
    def __init__(self, config, render_mode: str = None):
        super().__init__()
        self.config = config  # 环境配置字典
        self.render_mode = render_mode  # 渲染模式
        
        # MARL核心属性
        self.n_agents = 0  # 智能体数量（设备+飞机）
        self.n_actions = 0  # 动作空间维度
        self.obs_shape = 0  # 观测空间维度
        self.state_shape = 0  # 全局状态维度
        
        # 训练监控
        self.steps = 0  # 当前步数
        self.step_time = 0  # 单步耗时
        self.total_time = 0  # 总耗时
        
        # 可视化
        self.fig = None  # Matplotlib图形对象
        self.ax = None  # Matplotlib坐标轴对象
        
        # 初始化环境数据
        self.initialize(config)
        self.seed(config.get('seed', None))
        
        # MARL状态缓存
        self.obs4marl = None  # 多智能体观测缓存
        self.state4marl = None  # 全局状态缓存

    def initialize(self, config):
        '''加载配置文件并初始化环境'''
        # 加载作业数据
        with open(config['jobs_path'], 'r') as f:
                data = json.load(f)
        self.jobs = {item["作业编号"] : Job(code=item["作业编号"], 
                    time=item["作业时间"], 
                    group=item["分组"], 
                    resources=item["需要设备类型"] if isinstance(item["需要设备类型"], list) else [], 
                    predecessor=item["前置作业"] if isinstance(item["前置作业"], list) else [], 
                    exclusive=item["互斥作业"] if isinstance(item["互斥作业"], list) else [])
                for item in data}
        
        # 等待队列：按作业类型分类存储等待站点
        self.waiting_sites = {job_type: [] for job_type in self.jobs.keys()}
        
        # 加载固定资源
        with open(config['fixed_res_path'], 'r') as f:
            data = json.load(f)
        self.fixed_resources = {
            item["设备编号"]: Resource(
                item["设备编号"],
                item["类型"],
                [str(idx) for idx in range(int(item["支持停机位"].split("-")[0]), int(item["支持停机位"].split("-")[1]) + 1)],
                max_service=5
            ) for item in data
        }
        
        # 加载移动资源
        with open(config['mobile_res_path'], 'r') as f:
            data = json.load(f)
        self.mobile_resources = {
            item["设备编号"]: Resource(
                item["设备编号"],
                item["类型"],
                [item["初始停机位"]],
                max_service=1
            ) for item in data
        }
        
        # 加载站点
        with open(config['sites_path'], 'r') as f:
            data = json.load(f)
        self.sites = {
            code: Site(code, {
                'position': pos,
                'jobs': self.jobs,
                'fixed_resources': [res for res in self.fixed_resources.values() if code in res.sites],
                'mobile_resources': [res for res in self.mobile_resources.values() if code in res.sites]
            }) for code, pos in zip(data['sites_codes'], data['sites_positions'])
        }
        
        # 初始化移动设备
        self.mobile_devices = {}
        for res in self.mobile_resources.values():
            device_cfg = {
                'resource': res,
                'velocity': 5 if res.code == 'R014' else 3,# 转运车速度和飞机速度一致
                'site': self.sites[res.sites[0]]
            }
            if res.type not in self.mobile_devices:
                self.mobile_devices[res.type] = [Device(res.code, device_cfg)]
            else:
                self.mobile_devices[res.type].append(Device(res.code, device_cfg))
        
        # 飞机字典
        self.planes = {}
        self.num_planes = 0
        self.n_agents = len(self.mobile_devices) + 1  # 移动设备集群 + 飞机集群
        
        # 强制转运飞机列表：因干涉需要强制移动
        self.force_transfer_planes = []
        
        # 构建MARL动作空间：选择机位（所有站点+等待动作）
        self.n_actions = len(self.sites) + 1
        
        # 状态相关
        self.sites_state_global = [-1] * len(self.sites)  # 站位全局状态
        self.state_left_time = np.zeros(len(self.sites))  # 站位剩余时间
        
    def seed(self, seed=None):
        '''设置随机种子'''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def add_planes(self, new_planes_cfg):
        '''添加新飞机到环境'''
        for plane_cfg in new_planes_cfg:
            plane_id = f'Plane_{plane_cfg["batch"]}_{plane_cfg["idx"]}'
            if plane_id in self.planes:
                # print(f"Warning: Plane {plane_id} already exists!")
                continue
            plane = Plane(plane_id, plane_cfg)
            self.planes[plane_id] = plane
            self.sites[plane.site.code].add_plane(plane)
        self.num_planes += len(new_planes_cfg)

    def remove_planes(self, plane_ids):
        '''从环境移除飞机'''
        for plane_id in plane_ids:
            if plane_id in self.planes:
                self.sites[self.planes[plane_id].site.code].remove_plane()
                del self.planes[plane_id]
                self.num_planes -= 1

    def add_interfere_sites(self, site_codes, rec_time):
        '''添加站点干涉'''
        for site_code in site_codes:
            if site_code in self.sites:
                plane_id = self.sites[site_code].start_interfere(rec_time)
                if plane_id:
                    self.force_transfer_planes.append(plane_id)
        
    def get_idle_devices(self, res_types):
        '''获取指定资源类型的空闲设备
        
        输入:
            res_types: list - 资源类型列表
        返回:
            list - 空闲设备对象列表
        示例:
            idle_devices = env.get_idle_devices(['R001', 'R002'])
        '''
        ret = []
        for res_type in res_types:
            ret += [device for device in self.mobile_devices[res_type] if device.is_idle()]
        return ret
    
    def get_idle_transporters(self):
        '''获取所有空闲转运车
        
        返回:
            list - 空闲运输车设备对象列表
        示例:
            transporters = env.get_idle_transporters()
        '''
        return [device for device in self.mobile_devices["R014"] if not device.is_busy and not device.is_transporting]
    
    def get_waiting_planes(self):
        '''
        获取所有等待中的飞机
        
        返回:
            list - 等待状态的飞机对象列表
        示例:
            waiting = env.get_waiting_planes()
        '''
        return [plane for plane in self.planes.values() if plane.is_waiting]

    def get_avail_sites(self, plane=None):
        '''获取飞机可用的站点列表
        
        输入:
            plane: Plane对象或None - 指定飞机，默认为None
        返回:
            list - 可用站点代码列表
        逻辑:
            排除被占用、被干涉的站点，排除特殊站点(Z,29-31)
            如果指定了飞机且不在特殊站点，包含当前站点
        示例:
            sites = env.get_avail_sites(plane_instance)
        '''
        ret = [site.code for site in self.sites.values() if not site.is_occupied and not site.is_interfered and site.code not in ['Z', '29', '30', '31']]
        if plane:
            if plane.site.code not in ['Z', '29', '30', '31']:
                ret.append(plane.site.code)  # 包括当前所在位置
            # if not plane.is_idle():
            #     # 如果飞机正在忙碌，排除当前站位
            #     ret = [site for site in ret if site != plane.site.code]
        return ret
    
    def get_avail_takeoff_sites(self):
        '''获取可用起飞跑道
        
        返回:
            list - 空闲起飞跑道代码列表
        示例:
            runways = env.get_avail_takeoff_sites()
        '''
        return [site.code for site in self.sites.values() if not site.is_occupied and site.code in ['29', '30', '31']]

    def get_obs_agent(self, agent_id):
        '''获取单个智能体的观测（飞机作为主要智能体）'''
        return self.obs4marl[agent_id]
    
    def get_obs(self, n_agents=None):
        '''获取所有智能体的观测'''
        self.update_state(n_agents)
        if n_agents:
            return [self.get_obs_agent(i) for i in range(n_agents)]
        else:
            return [self.get_obs_agent(i) for i in range(len(self.planes))]
    
    def get_state(self, n_agents=None):
        '''获取全局状态'''
        if n_agents or self.state4marl is None:
            self.update_state(n_agents)
        return self.state4marl
    
    def get_avail_agent_actions(self, plane_id=None):
        '''获取智能体可用动作掩码
        
        输入:
            plane_id: str或None - 飞机ID
        返回:
            list - 动作掩码列表（1表示可用，0表示不可用）
        结构:
            [站点1可用性, 站点2可用性, ..., 等待动作可用性]
        示例:
            mask = env.get_avail_agent_actions('Plane_1_1')
        '''
        '''获取智能体可用的动作掩码'''
        # 检查飞机是否处于正忙状态
        if plane_id:
            plane = self.planes[plane_id]
            avail_sites = self.get_avail_sites(plane)
        else:
            avail_sites = self.get_avail_sites()
        # 为每个站点生成可用性掩码
        avail_actions = [1 if site.code in avail_sites else 0 for site in self.sites.values()]
        # 返回 [可用机位..., 1, 0, 0] 表示可以"等待"
        if plane_id:
            if not plane.is_idle() or plane.is_completed_all_jobs():
                avail_actions = [0] * len(avail_actions) + [0]
            else:
                avail_actions += [0]
        else:
            avail_actions += [0]
        return avail_actions
    
    def get_env_info(self):
        '''返回环境基本信息
        
        返回:
            dict - 包含动作数、智能体数、状态维度等信息
        示例:
            info = env.get_env_info()
        '''
        '''返回环境基本信息供Runner使用'''
        return {
            "n_actions": self.n_actions,
            "n_agents": len(self.planes),  # 实际智能体数量是飞机数量
            "state_shape": len(self.get_state()),
            "obs_shape": len(self.get_obs()[0]) if self.obs4marl else 0,
            "episode_limit": len(self.planes) * 10
        }
    
    def update_state(self, n_agents=None):
        '''更新MARL全局状态和观测
        
        输入:
            n_agents: int或None - 智能体数量
        作用:
            构建全局状态向量和每个智能体的观测向量
            状态包含站点占用、剩余时间、资源可用性、飞机状态
        示例:
            env.update_state()  # 手动刷新状态
        '''
        '''更新MARL全局状态和观测'''
        # 构建全局状态
        state = []
        avail_sites = self.get_avail_sites()
        for site in self.sites.values():
            site_occupied, site_left_time, site_resource = [], [], []
            if site.code not in ['Z', '29', '30', '31']:
                # 1. 站位占用状态
                site_occupied.append(len(self.planes) + 1 if site.code in avail_sites or site.is_interfered else int(site.plane.code.split('_')[2]))
                # 2. 站位剩余工作时间
                if site.is_interfered:
                    site_left_time.append(site.left_rec_time)
                elif site.is_occupied:
                    site_left_time.append(site.left_job_time)
                else:
                    site_left_time.append(0)
                # 3. 站位资源向量（one-hot）
                site_resource += [1 if site.is_avail_job(job) else 0 for job in self.jobs.values() if job.group == '保障' and job.code not in ['ZY01', 'ZY-L']]
            state += site_occupied + site_left_time + site_resource
        # 4. 飞机状态
        if n_agents:
            plane_state = [[0, 0, 0] for _ in range(n_agents)]
        else:
            plane_state = [[0, 0, 0] for _ in range(len(self.planes))]
        for agent_id, (plane_id, plane) in enumerate(self.planes.items()):
            pid = int(plane_id.split('_')[2])
            if len(plane.left_jobs) > 0:
                next_job = sorted(plane.get_avail_jobs(), key=lambda x: plane.jobs[x].time, reverse=True)[0]
                if n_agents:
                    plane_state[pid] = [list(plane.jobs.keys()).index(next_job), plane.jobs[next_job].time, len(plane.left_jobs)]
                else:
                    plane_state[agent_id] = [list(plane.jobs.keys()).index(next_job), plane.jobs[next_job].time, len(plane.left_jobs)]
            else:
                if n_agents:
                    plane_state[pid] = [len(plane.jobs), 0, 0]  # 已完成所有作业
                else:
                    plane_state[agent_id] = [len(plane.jobs), 0, 0]
        
        for s in plane_state:
            state += s
            
        self.state4marl = np.array(state, dtype=np.float32)
        
        self.obs_dim = 3 + 2 * len(self.sites) + 1    # 3 个飞机自身信息 + 每站位 2 维 + 忙碌 flag
        zero_obs = [0.0] * self.obs_dim

        if n_agents:
            self.obs4marl = [zero_obs for _ in range(n_agents)]
        else:
            self.obs4marl = [zero_obs for _ in range(len(self.planes))]
        for agent_id, (plane_id, plane) in enumerate(self.planes.items()):
            pid = int(plane_id.split('_')[2])
            # 飞机已完成所有作业 -> 全零观测
            if plane.is_completed_all_jobs():
                if n_agents:
                    self.obs4marl[pid] = zero_obs
                else:
                    self.obs4marl[agent_id] = zero_obs
                continue

            next_job = sorted(plane.get_avail_jobs(), key=lambda x: plane.jobs[x].time, reverse=True)[0]
            self_obs = [float(list(plane.jobs.keys()).index(next_job)), 
                        float(plane.jobs[next_job].time), 
                        float(len(plane.left_jobs))]

            # 对每一个站位：运输时间 + 剩余工时
            per_site_obs = []
            for site in self.sites.values():
                # 运输时间（若无资源则为 0）
                if site.is_avail_job(self.jobs[next_job]):
                    distance = abs(plane.site.pos[0]-site.pos[0]) + abs(plane.site.pos[1]-site.pos[1]) # 曼哈顿距离
                    travel_t = distance / plane.velocity
                else:
                    travel_t = 0.0

                # 站位剩余工时（空闲为 0）
                left_t = site.left_job_time
                per_site_obs.extend([travel_t, left_t])

            # 忙碌标记
            if not plane.is_idle():
                # 正忙：除自身前 3 维外全部置 0
                per_site_obs = [0.0] * len(per_site_obs)

            if n_agents:
                self.obs4marl[pid] = self_obs + per_site_obs + [float(not plane.is_idle())]
            else:
                self.obs4marl[agent_id] = self_obs + per_site_obs + [float(not plane.is_idle())]
    
    def reset(self, seed=None, options=None):
        '''重置环境
        
        输入:
            seed: int或None - 随机种子
            options: dict或None - 额外选项
        返回:
            tuple - (初始状态, 信息)
        作用:
            重置所有状态，清空飞机和设备，重新初始化
        示例:
            state, info = env.reset()
        '''
        '''重置环境，返回初始全局状态'''
        self.steps = 0
        self.total_time = 0
        self.step_time = 0
        self.done = False
        
        # 清空飞机和设备状态
        self.planes.clear()
        for devices in self.mobile_devices.values():
            for device in devices:
                device.reset()
        
        # 重新初始化（保持配置不变）
        self.initialize(self.config)
        
        return np.array([]), {}
    
    def step(self, action, dt, n_agents=None):
        '''执行一步动作
        
        输入:
            action: dict - 包含设备和飞机的动作决策
                结构: {"devices": {类型: [目标站点代码列表]}, "planes": {批次: {编号: [目标站点, 目标作业]}}}
            dt: int - 时间步长（秒）
            n_agents: int或None - 智能体数量
        返回:
            tuple - (全局奖励, 是否结束)
        作用:
            处理动作，更新环境状态，计算奖励
        示例:
            reward, done = env.step(action_dict, 60)
        '''
        '''
        执行一步动作
        action: dict 包含飞机和设备的动作
        dt: int 时间差
        '''
        self.steps += 1
        self.total_time += dt 
        
        # 动作处理逻辑（参考您的原始实现）
        step_time, real_did = self.process_action(action, dt)    
        self.step_time = step_time              
        self.done = len(self.planes) == 0    
        
        # 计算全局奖励
        reward = self.calculate_global_reward(real_did)

        # 更新状态
        self.update_state(n_agents)
        
        return reward, self.done
    
    def process_action(self, action, dt):
        '''处理动作并计算即时奖励
        
        输入:
            action: dict - 动作字典
            dt: int - 时间步长
        返回:
            tuple - (step_time: int, real_did: int)
                step_time: 本次动作的实际推进时间
                real_did: 实际执行的动作数量
        作用:
            解析并执行设备和飞机的动作，处理等待、运输、作业逻辑
        示例:
            step_time, actions = env.process_action(action_dict, 60)
        '''
        '''处理动作并计算即时奖励'''
        step_time = np.inf
        real_did = 0
        remove_planes = []
        
        # 处理设备动作
        for device_type, devices in self.mobile_devices.items():
            for idx, device in enumerate(devices):
                if not device.is_idle():
                    # 更新非空闲设备状态
                    step_time = min(step_time, device.update(dt)) 
                elif action["devices"][device_type][idx] == device.site.code:
                    continue  # 目标位置与当前位置相同，无需移动
                else:
                    # 启动设备运输
                    step_time = min(step_time, device.start_transport(self.sites[action["devices"][device_type][idx]]))
        
        # 处理飞机动作
        for plane_id, plane in self.planes.items():
            if not plane.is_idle():
                # 更新非空闲飞机状态
                step_time = min(step_time, plane.update(dt))
            else:
                # 解析飞机目标站点和作业
                target_site_code, target_job = action["planes"][int(plane_id.split('_')[1])][int(plane_id.split('_')[2])]
                
                # 先考虑转运，再考虑作业
                if target_site_code and target_site_code != plane.site.code:
                    # 无可用转运车的情况
                    if plane.site.get_avail_transporter() is None:
                        if plane.site.code == 'Z':  # 起始位置可以直接滑行
                            step_time = min(step_time, plane.start_transport(self.sites[target_site_code], None))
                            plane.choosed_job = target_job
                            real_did += 1
                        else:
                            # 启动等待状态
                            plane.start_waiting()
                            self.waiting_sites['ZY-T'].append(plane.site.code)
                            plane.destination = self.sites[target_site_code]
                            plane.destination.add_plane(plane)
                            # print(f"[Time: {self.total_time}] Plane {plane_id} is waiting for a transporter at site {plane.site.code}.")
                            plane.choosed_job = target_job
                            step_time = 0
                    elif target_site_code:
                        # 有转运车，启动运输
                        step_time = min(step_time, plane.start_transport(self.sites[target_site_code], plane.site.get_avail_transporter()))
                        plane.choosed_job = target_job
                        real_did += 1
                
                else:
                    # 不转运，直接作业
                    if target_job not in plane.get_avail_jobs():
                        if not plane.is_completed_all_jobs(): # 资源不足
                            plane.start_waiting()
                            self.waiting_sites[target_job.code].append(plane.site.code)
                            self.choosed_job = target_job
                    else:
                        step_time = min(step_time, plane.choose_job(target_job))
                        real_did += 1

            # 检查是否需要移除已完成所有作业的飞机
            if plane.is_completed_all_jobs() and plane.site.code in ['29', '30', '31'] and plane.is_idle():
                remove_planes.append(plane_id)
        
        # 更新干涉站点
        for site in self.sites.values():
            if site.is_interfered:
                step_time = min(step_time, site.update(dt))

        self.remove_planes(remove_planes)
        # for plane_id in remove_planes:
            # print(f"[Time: {self.total_time}] Plane {plane_id} has taken off and removed from the airport.")
        assert step_time != np.inf, "No planes or devices in the environment!"
        return step_time, real_did
    
    def calculate_global_reward(self, real_did):
        '''计算全局奖励
        
        输入:
            real_did: int - 实际执行的动作数量
        返回:
            float - 全局奖励值
        奖励构成:
            - 等待惩罚：每个等待飞机-2分
            - makespan奖励：完成时按总时间倒数，否则按动作效率
        示例:
            reward = env.calculate_global_reward(5)
        '''
        '''计算全局奖励'''
        # 等待惩罚
        waiting_penalty = sum(1 for p in self.planes.values() if p.is_waiting) * 2
        # makespan相关奖励
        if self.done:
            makespan_reward = 60000 / self.total_time
        else:
            max_steps = len(self.planes) * 10
            makespan_reward = real_did - self.steps / 120
        # 综合奖励
        global_reward = makespan_reward  - waiting_penalty
        
        return global_reward

    def render(self):
        """渲染环境可视化（模仿 path_test_3 画风，并自动截屏）

        功能：
        - 画停机坪 / 跑道（彩色矩形）
        - 画飞机（彩色圆点 + 编号）
        - 画移动设备（紫色小方块 + 编号）
        - 每次调用 render 时，自动按“5 个波次 × 每波 5 张”保存前 25 张图片，
          然后额外保存一张“空白底图”（只有停机坪，没有飞机和设备）。

        说明：
        - 是否保存、保存目录、波次数量、每波张数都可以在下面的默认参数里改。
        - 假设你在外部控制“隔多久调用一次 render”，环境只负责
          在调用的前 25 次里自动保存图片。
        """
        if self.render_mode is None:
            return

        import os
        import matplotlib.pyplot as plt

        # ========= 字体设置（解决中文变方框问题，只做一次） =========
        if not hasattr(self, "_font_inited"):
            # 依次尝试这些中文字体，环境里有哪个就用哪个
            plt.rcParams["font.sans-serif"] = [
                "SimHei",               # Windows 常见
                "Microsoft YaHei",      # Windows 常见
                "WenQuanYi Micro Hei",  # Ubuntu 常见
                "Noto Sans CJK SC",     # 常用 CJK 字体
                "DejaVu Sans",          # 兜底（不一定全支持中文）
            ]
            plt.rcParams["axes.unicode_minus"] = False
            self._font_inited = True

        # ========= 截图参数初始化（只做一次） =========
        if not hasattr(self, "_render_save_inited"):
            self._render_save_inited = True
            # 你可以按需修改这些默认参数（也可以在外部手动改属性）
            self.render_num_waves = getattr(self, "render_num_waves", 5)           # 波次数
            self.render_frames_per_wave = getattr(self, "render_frames_per_wave", 20)  # 每波张数
            self.render_save_dir = getattr(self, "render_save_dir", "render_output")  # 保存目录
            self.render_auto_save = getattr(self, "render_auto_save", True)        # 是否自动保存
            # ✅ 新增：截图间隔（例如 20 表示每 20 次 render 保存一张）
            self.render_save_stride = getattr(self, "render_save_stride", 30)

            self._render_saved_count = 0
            self._render_blank_saved = False
            self._render_call_count = 0
            if self.render_auto_save:
                os.makedirs(self.render_save_dir, exist_ok=True)

        # ========= 创建 / 清空 画布 =========
        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(16, 11), dpi=120)
            self.ax.set_title("机场调度可视化", fontsize=16, fontweight="bold")

        ax = self.ax
        ax.clear()

        # ========= 1. 计算视野范围 =========
        xs = [site.pos[0] for site in self.sites.values()]
        ys = [site.pos[1] for site in self.sites.values()]
        if xs and ys:
            margin = 10
            xmin, xmax = min(xs) - margin, max(xs) + margin
            ymin, ymax = min(ys) - margin, max(ys) + margin
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        ax.set_facecolor("#f7f7f7")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

        # 小工具：根据站位 code 决定区域类型
        def _zone_type(code: str) -> str:
            if code == "Z":
                return "landing"   # 降落区
            if code in ["29", "30", "31"]:
                return "takeoff"   # 起飞区
            return "parking"       # 普通停机坪

        # ========= 2. 画所有机位（停机坪矩形，模仿 path_test_3） =========
        for code, site in self.sites.items():
            x, y = site.pos
            zone = _zone_type(code)

            if zone == "landing":
                facecolor = "lightgreen"
                edgecolor = "darkgreen"
                lw = 2.0
                alpha = 0.7
            elif zone == "takeoff":
                facecolor = "lightyellow"
                edgecolor = "orange"
                lw = 2.0
                alpha = 0.7
            else:
                facecolor = "lightblue"
                edgecolor = "blue"
                lw = 1.5
                alpha = 0.6

            rect = plt.Rectangle(
                (x - 4, y - 4),
                8,
                8,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=lw,
                alpha=alpha,
                zorder=1,
            )
            ax.add_patch(rect)
            ax.text(
                x,
                y,
                code,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                zorder=2,
            )

        # ========= 3. 画飞机（彩色圆点 + 编号） =========
        for idx, (plane_id, plane) in enumerate(self.planes.items()):
            x, y = plane.site.pos
            bidx, pidx = int(plane_id.split('_')[1]), int(plane_id.split('_')[2])  # 提取飞机编号
            color = plt.cm.tab10(pidx % 12)

            ax.scatter(
                x,
                y,
                s=150,
                marker="o",
                c=[color],
                edgecolors="black",
                zorder=5,
            )
            ax.text(
                x,
                y + 5,
                plane_id,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                zorder=6,
            )

        # ========= 4. 画移动设备（紫色小方块 + 编号） =========
        for device_type, devices in self.mobile_devices.items():
            if device_type == "R014":
                for d in devices:
                    x, y = d.site.pos
                    ax.scatter(
                        x,
                        y - 3,
                        s=100,
                        marker="s",
                        c=["purple"],
                        edgecolors="black",
                        zorder=4,
                    )
                    ax.text(
                        x,
                        y - 7,
                        d.code,
                        ha="center",
                        va="top",
                        fontsize=7,
                        color="purple",
                        zorder=6,
                    )

        # ========= 5. 坐标轴 & 图例（中文不再方框） =========
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor="lightgreen", edgecolor="darkgreen",
                          alpha=0.7, label="Landing Zone (Z)"),
            plt.Rectangle((0, 0), 1, 1, facecolor="lightyellow", edgecolor="orange",
                          alpha=0.7, label="Takeoff Area (29/30/31)"),
            plt.Rectangle((0, 0), 1, 1, facecolor="lightblue", edgecolor="blue",
                          alpha=0.6, label="Aircraft parking area"),
            plt.Line2D([0], [0], marker="o", color="w", label="Plane",
                       markerfacecolor="red", markeredgecolor="black", markersize=10),
            plt.Line2D([0], [0], marker="s", color="w", label="Device",
                       markerfacecolor="purple", markeredgecolor="black", markersize=8),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

        # ========= 6. 刷新画面 =========
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # ========= 7. 截图保存逻辑（按间隔保存） =========
        if not self.render_auto_save:
            return

        # 统计 render 被调用了多少次
        self._render_call_count += 1

        max_images = self.render_num_waves * self.render_frames_per_wave

        # 7.1 只在“到达间隔点”时尝试保存：
        #     比如 stride=20，就只有第 1, 21, 41, ... 次 render 会保存。
        if self._render_saved_count < max_images:
            if (self._render_call_count - 1) % self.render_save_stride != 0:
                # 还没到保存间隔，直接返回
                return

            wave_idx = self._render_saved_count // self.render_frames_per_wave
            frame_idx = self._render_saved_count % self.render_frames_per_wave
            filename = f"wave{wave_idx + 1}_frame{frame_idx + 1}_step{self.steps:05d}.png"
            filepath = os.path.join(self.render_save_dir, filename)
            # self.fig.savefig(filepath, dpi=150, bbox_inches="tight")
            self._render_saved_count += 1

        # 7.2 保存一张“空白底图”（只有停机坪）
        elif not self._render_blank_saved:
            blank_fig, blank_ax = plt.subplots(figsize=(16, 11), dpi=120)
            blank_ax.set_facecolor("#f7f7f7")
            blank_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

            if xs and ys:
                blank_ax.set_xlim(min(xs) - 10, max(xs) + 10)
                blank_ax.set_ylim(min(ys) - 10, max(ys) + 10)

            # 只画停机坪 / 跑道，不画飞机和设备
            for code, site in self.sites.items():
                x, y = site.pos
                zone = _zone_type(code)

                if zone == "landing":
                    facecolor = "lightgreen"
                    edgecolor = "darkgreen"
                    lw = 2.0
                    alpha = 0.7
                elif zone == "takeoff":
                    facecolor = "lightyellow"
                    edgecolor = "orange"
                    lw = 2.0
                    alpha = 0.7
                else:
                    facecolor = "lightblue"
                    edgecolor = "blue"
                    lw = 1.5
                    alpha = 0.6

                rect = plt.Rectangle(
                    (x - 4, y - 4),
                    8,
                    8,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=lw,
                    alpha=alpha,
                    zorder=1,
                )
                blank_ax.add_patch(rect)
                blank_ax.text(
                    x,
                    y,
                    code,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                    zorder=2,
                )

            blank_ax.set_xlabel("X")
            blank_ax.set_ylabel("Y")
            blank_ax.set_aspect("equal", adjustable="box")

            blank_path = os.path.join(self.render_save_dir, "blank.png")
            # blank_fig.savefig(blank_path, dpi=150, bbox_inches="tight")
            # plt.close(blank_fig)

            self._render_blank_saved = True


    def close(self):
        '''关闭环境，释放资源'''
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None