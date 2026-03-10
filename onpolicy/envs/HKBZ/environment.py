# grid_world_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Tuple
from onpolicy.envs.HKBZ.core import Plane, Device, Job, Resource, Site
from onpolicy.envs.HKBZ.utils import arrange_devices
import json
import matplotlib.pyplot as plt
from gymnasium.utils import seeding
import math
import torch
from torch_geometric.data import HeteroData

# 调度环境类：基于Gymnasium的多智能体强化学习环境，用于模拟飞机在机场站点的调度过程
class AircraftScheduleEnv(gym.Env):
    environment_name = "Plane Schedule"
    
    def __init__(self, config, render_mode: str = None):
        super().__init__()
        self.config = config  # 环境配置字典
        self.render_mode = render_mode  # 渲染模式
        
        # MARL核心属性
        self.n_agents = 0
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
        
        # 加载作业数据
        with open(config['jobs_path'], 'r', encoding='utf-8') as f:
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
        with open(config['fixed_res_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.fixed_resources = {
            item["设备编号"]: Resource(
                item["设备编号"],
                item["类型"],
                [str(idx) for idx in range(int(item["支持停机位"].split("-")[0]), int(item["支持停机位"].split("-")[1]) + 1)],
                max_service=3
            ) for item in data
        }
        
        # 加载移动资源
        with open(config['mobile_res_path'], 'r', encoding='utf-8') as f:
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
        with open(config['sites_path'], 'r', encoding='utf-8') as f:
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
        
        # [修改点 1]：读取航班配置文件
        with open(config['flights_path'], 'r', encoding='utf-8') as f:
            self.flights_data = json.load(f)
            
        self.num_planes = len(self.flights_data)
        self.n_agents = config.get('n_agents', self.num_planes)  # 根据配置或实际飞机数量动态调整
        
        # 强制转运飞机列表：因干涉需要强制移动
        self.force_transfer_planes = []
        
        # 构建MARL动作空间
        self.action_space = spaces.MultiDiscrete([len(self.sites) + 1, len(self.jobs) + 1])
        
        # 状态相关
        self.sites_state_global = [-1] * len(self.sites)  # 站位全局状态
        self.state_left_time = np.zeros(len(self.sites))  # 站位剩余时间

        self.site_code_list = list(self.sites.keys())
        self.job_code_list = [job.code for job in self.jobs.values() if job.group == '保障' and job.code not in ['ZY01', 'ZY-L']]
        self.runway_code_list = [self.site_code_list[0]] + self.site_code_list[-3:]
        
        # [修改点 2]：一次性生成环境的基础（Base）飞机降落时间表（不包含随机扰动）
        self.base_landing_list = []
        for idx, item in enumerate(self.flights_data):
            # 将 "34%" 转换为整数 34
            fuel_percentage = int(item["初始燃油状态"].replace('%', ''))
            self.base_landing_list.append({
                'land_time': item["到达时间"],
                'fuel': fuel_percentage,
                'bidx': 0,        # 简化为全属批次0
                'pidx': idx,      # 原始索引
                'plane_id': item["飞机编号"]
            })
        # 确保时间轴排序
        self.base_landing_list.sort(key=lambda x: x['land_time'])
        
        # 环境设置
        self.seed(config.get('seed', None))
        self.trajectory_log = []
        self.pending_actions = {}
        self.use_domain_rand = config.get('use_domain_rand', True)
        
        # 初始加载时调用一次复位（替代原有硬编码的加载循环）
        # self.reset() 通常由外部调用，此处无需重复
        
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
            ret += [device for device in self.mobile_devices[res_type] if device.is_idle() and device.resource.is_available()]
        return ret

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
        ret = [site.code for site in self.sites.values() if not site.is_occupied and not site.is_interfered and site.code not in self.runway_code_list]
        if plane:
            if plane.site.code not in self.runway_code_list:
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

    def _get_obs(self):
        """
        更新环境状态并构建异构图数据对象 (HeteroData)。 
        【采用静态拓扑】：无论飞机是否在场，恒定生成 n_agents * n_ops 个工序节点，
        确保网络在不同 Step、不同 Episode 获得的张量形状绝对一致。
        """
        data = HeteroData()
        
        # ==========================================
        # 1. 解析全局常量与映射字典
        # ==========================================
        site_list = list(self.sites.values())
        site2idx = {site.code: idx for idx, site in enumerate(site_list)}
        n_sites = len(site_list)
        
        device_list = [dev for devs in self.mobile_devices.values() for dev in devs]
        device2idx = {dev.code: idx for idx, dev in enumerate(device_list)}
        dev_types = list(self.mobile_devices.keys())

        # n_ops 直接由初始化好的 job_code_list 决定
        n_ops = len(self.job_code_list)
        n_agents = self.n_agents
        
        # 提取当前场上活跃的飞机，映射为全局唯一 PID
        active_planes = {}
        for plane in self.planes.values():
            parts = plane.code.split('_')
            bidx, pidx = int(parts[1]), int(parts[2])
            global_pid = bidx * self.plane_num_per_batch + pidx
            active_planes[global_pid] = plane

        # ==========================================
        # 2. 构建停机位与设备节点 (Site & Device)
        # ==========================================
        site_features = []
        global_site_valid = [] 
        for site in site_list:
            occ = 1.0 if site.is_occupied else 0.0
            interf = 1.0 if site.is_interfered else 0.0
            rem_time = float(max(site.left_job_time, site.left_rec_time))
            job_onehot = site.avail_job_onehot if hasattr(site, 'avail_job_onehot') else [0] * n_ops
            
            site_features.append([occ, interf, rem_time] + job_onehot)
            
            # 物理限制
            if site.code in self.runway_code_list or site.is_interfered or site.is_occupied:
                global_site_valid.append(False)
            else:
                global_site_valid.append(True)
                
        data['site'].x = torch.tensor(site_features, dtype=torch.float32)
        
        device_features = []
        for dev in device_list:
            dtype_enc = float(dev_types.index(dev.resource.type))
            status_enc = 0.0 if dev.is_idle() else 1.0
            rem_time = float(max(dev.left_trans_time, dev.left_rec_time))
            pos_x, pos_y = float(dev.site.pos[0]), float(dev.site.pos[1])
            device_features.append([dtype_enc, status_enc, rem_time, pos_x, pos_y])
            
        if len(device_features) > 0:
            data['device'].x = torch.tensor(device_features, dtype=torch.float32)
        else:
            data['device'].x = torch.empty((0, 5), dtype=torch.float32)

        # ==========================================
        # 3. 构建工序节点 (Operation) & 机位掩码 (Site Mask)
        # ==========================================
        op_features = np.zeros((n_agents * n_ops, 6), dtype=np.float32)
        agent_op_mask = np.zeros((n_agents, n_agents * n_ops), dtype=bool) 
        
        # 维度缩减为 (n_agents, n_sites)
        ptr_site_mask_matrix = np.zeros((n_agents, n_sites), dtype=bool)
        
        for global_pid in range(n_agents):
            plane = active_planes.get(global_pid, None)
            
            if plane is not None:
                # ---------------------------------------------------
                # A. 计算该飞机对所有机位的合法性 (不再依赖具体的 job)
                # ---------------------------------------------------
                for s_idx, site in enumerate(site_list):
                    if global_site_valid[s_idx]:
                        # 1. 全局合法（无人占用且无干涉），此机位可用
                        ptr_site_mask_matrix[global_pid, s_idx] = True
                    else:
                        # 2. 全局不合法，但如果占用它的正是当前这架飞机本身，且无干涉，则可用（允许飞机原地干活）
                        if site.code != 'Z' and site == plane.site and not site.is_interfered:
                            ptr_site_mask_matrix[global_pid, s_idx] = True
                        else:
                            ptr_site_mask_matrix[global_pid, s_idx] = False

                # ---------------------------------------------------
                # B. 计算工序特征与掩码
                # ---------------------------------------------------
                current_avail_jobs = plane.get_avail_jobs(site=None)
                for j_idx, job_code in enumerate(self.job_code_list):
                    u_idx = global_pid * n_ops + j_idx 
                    job_obj = self.jobs[job_code]
                    
                    if job_code in plane.finished_jobs:
                        status, is_ready = 3.0, False
                    elif job_code in plane.current_jobs:
                        status, is_ready = 2.0, False
                    elif job_code in current_avail_jobs:
                        status, is_ready = 1.0, True
                    else:
                        status, is_ready = 0.0, False
                        
                    can_schedule = plane.is_idle() and is_ready
                    agent_op_mask[global_pid, u_idx] = can_schedule
                    
                    proc_time = float(job_obj.time) if job_obj.time else 0.0
                    rem_ops = float(len(plane.left_jobs))
                    req_res = 1.0 if len(set(job_obj.resources).intersection(set(dev_types))) > 0 else 0.0
                    wait_time = float(plane.waiting_time) if status == 1.0 else 0.0
                    
                    op_features[u_idx] = [status, proc_time, rem_ops, req_res, wait_time, float(global_pid)]
                    
            else:
                # 飞机不在场上：幽灵节点
                for j_idx in range(n_ops):
                    u_idx = global_pid * n_ops + j_idx
                    op_features[u_idx] = [3.0, 0.0, 0.0, 0.0, 0.0, float(global_pid)]
                    agent_op_mask[global_pid, u_idx] = True 
                
                # 不在场的飞机，机位掩码全为 True 防止计算 NaN
                ptr_site_mask_matrix[global_pid, :] = True

            # 安全保险 - 如果一架在场的飞机当前没任何活能干，强制把它自己的节点置为 True 占位
            if not agent_op_mask[global_pid].any():
                agent_op_mask[global_pid, global_pid * n_ops : (global_pid + 1) * n_ops] = True
                
        data['operation'].x = torch.tensor(op_features, dtype=torch.float32)

        # ==========================================
        # 4. 构建边索引与边特征 (Edges)
        # ==========================================
        edge_precedes = [[], []]   
        edge_os = [[], []]         
        attr_os = []
        edge_or = [[], []]         
        attr_or = []
        
        # 只为在场的活跃飞机连边，Dummy 节点作为孤岛存在即可
        for global_pid, plane in active_planes.items():
            for j_idx, j_code in enumerate(self.job_code_list):
                u_idx = global_pid * n_ops + j_idx
                job_obj = self.jobs[j_code]
                
                # --- A. Precedes ---
                for pred_code in job_obj.predecessor:
                    if pred_code in self.job_code_list:
                        pred_j_idx = self.job_code_list.index(pred_code)
                        pred_u_idx = global_pid * n_ops + pred_j_idx
                        edge_precedes[0].append(pred_u_idx)
                        edge_precedes[1].append(u_idx)
                        
                # --- B. O-S Edge ---
                # 注意：图网络建边只需要看 ptr_site_mask_matrix 就行了
                for s_idx, is_valid in enumerate(ptr_site_mask_matrix[global_pid]):
                    if is_valid:
                        site = site_list[s_idx]
                        edge_os[0].append(u_idx)
                        edge_os[1].append(s_idx)
                        dist = abs(plane.site.pos[0] - site.pos[0]) + abs(plane.site.pos[1] - site.pos[1])
                        attr_os.append([dist / plane.velocity])
                
                # --- C. O-R Edge ---
                needed_dev_types = set(job_obj.resources).intersection(set(dev_types))
                if j_code not in self.job_code_list and getattr(plane, 'destination', None) is not None:
                    needed_dev_types = {'R014'}
                    
                for dev_type in needed_dev_types:
                    for dev in self.mobile_devices.get(dev_type, []):
                        if dev.is_idle():
                            v_idx = device2idx[dev.code]
                            edge_or[0].append(u_idx)
                            edge_or[1].append(v_idx)
                            dist = abs(dev.site.pos[0] - plane.site.pos[0]) + abs(dev.site.pos[1] - plane.site.pos[1])
                            attr_or.append([dist / dev.velocity])

        # ==========================================
        # 5. 赋值到 PyG 数据对象 (携带空边防崩保护)
        # ==========================================
        if len(edge_precedes[0]) > 0:
            data['operation', 'precedes', 'operation'].edge_index = torch.tensor(edge_precedes, dtype=torch.long)
        else:
            data['operation', 'precedes', 'operation'].edge_index = torch.empty((2, 0), dtype=torch.long)
            
        if len(edge_os[0]) > 0:
            data['operation', 'assignable_to', 'site'].edge_index = torch.tensor(edge_os, dtype=torch.long)
            data['operation', 'assignable_to', 'site'].edge_attr = torch.tensor(attr_os, dtype=torch.float32)
        else:
            data['operation', 'assignable_to', 'site'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['operation', 'assignable_to', 'site'].edge_attr = torch.empty((0, 1), dtype=torch.float32)
            
        if len(edge_or[0]) > 0:
            data['operation', 'needs', 'device'].edge_index = torch.tensor(edge_or, dtype=torch.long)
            data['operation', 'needs', 'device'].edge_attr = torch.tensor(attr_or, dtype=torch.float32)
        else:
            data['operation', 'needs', 'device'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['operation', 'needs', 'device'].edge_attr = torch.empty((0, 1), dtype=torch.float32)
            
        # 直接挂载 Numpy 生成的绝对固定形状的 Tensor
        data.op_mask = torch.tensor(agent_op_mask, dtype=torch.bool)             # Shape: [n_agents, n_ops]
        data.site_mask_matrix = torch.tensor(ptr_site_mask_matrix, dtype=torch.bool) # Shape: [n_agents, n_sites]
            
        return data

    def _get_reward(self):
        """
        计算每个智能体的即时奖励。
        【重构核心】：
        1. 引入 Reward Shaping（阶段性奖励与离场奖励），打破稀疏惩罚，引导 Actor 走出随机探索的深渊。
        2. 修复 SMDP 奖励分配漏洞：每一步流逝的 dt 产生的奖惩，必须分发给所有在场飞机。
        """
        dt = self.step_time 
        
        # 如果没有时间流逝，直接返回 0
        if dt <= 0:
            return np.zeros((self.n_agents, 1), dtype=np.float32)

        # 1. 计算全局时间惩罚 (Global Penalty)
        # 仍在场上的飞机越多，每秒扣分越狠，倒逼模型学会并行调度以缩短 Total Flow Time
        # active_plane_count = sum(1 for p in self.planes.values() if not p.is_completed_all_jobs())
        global_time_penalty = - dt / 100.0

        rewards = np.zeros((self.n_agents, 1), dtype=np.float32)

        for plane_id, plane in self.planes.items():
            pid = int(plane_id.split('_')[-1])
            
            # --- 动态初始化追踪属性 (无需去 Plane 类里改底座代码) ---
            if not hasattr(plane, '_last_rewarded_job_count'):
                plane._last_rewarded_job_count = len(plane.finished_jobs)
            if not hasattr(plane, '_has_received_completion_bonus'):
                plane._has_received_completion_bonus = False
            
            # 基础奖励：无论是否在做决策，所有在场飞机都要承担时间流逝的惩罚
            agent_reward = global_time_penalty

            # --- A. 进度激励 (Dense Reward) ---
            # 检查在刚才流逝的 dt 时间内，该飞机是否完成了新的保障工序
            current_finished_count = len(plane.finished_jobs)
            newly_finished = current_finished_count - plane._last_rewarded_job_count
            
            if newly_finished > 0:
                # 每完成一个保障工序，给予正向反馈，引导模型 "多干活"
                agent_reward += newly_finished * 5.0
                plane._last_rewarded_job_count = current_finished_count

            # --- B. 终局奖励 (Terminal Reward) ---
            # 如果刚刚完成了所有任务（准备起飞或已经起飞离场）
            if plane.is_completed_all_jobs() and not plane._has_received_completion_bonus:
                # 给予巨大的通关奖励，这是策略网络后期收敛的核心动力
                agent_reward += 50.0
                plane._has_received_completion_bonus = True
            
            # --- C. 非法动作瞬时惩罚 (Invalid Action Penalty) ---
            # 用于配合后续的强行拦截逻辑。如果下发了必定失败或不合法的动作，狠狠扣分。
            if hasattr(plane, 'invalid_action_flag') and plane.invalid_action_flag:
                agent_reward -= 10.0
                plane.invalid_action_flag = False  # 重置标志

            # 赋值：不再用 current_active_agents 屏蔽，保障休眠期的连续奖励传递
            rewards[pid, 0] = agent_reward
            
        return rewards
    
    def _get_done(self):
        """
        获取每个智能体在当前回合是否结束 (Episode Termination)。

        Output:
        - agent_dones: List[bool] - 当前仍在场上的每架飞机是否已完成自身所有任务
        """
        agent_dones = [False] * self.n_agents
        
        # 遍历当前仍在环境中的所有飞机
        for plane in self.planes.values():
            pid = int(plane.code.split('_')[-1])
            # 单架飞机结束的标志：剩余待办作业列表为空
            agent_dones[pid] = plane.is_completed_all_jobs()
            
        return np.array(agent_dones)

    def _get_info(self):
        """
        获取环境的额外信息，包括用于 GRU 时序记忆的上一次动作索引。
        """
        active_agents = [False] * self.n_agents
        for plane in self.planes.values():
            # 活跃条件 1：飞机处于空闲状态（不忙碌、不在运输、不在等待）
            # 活跃条件 2：飞机还有未完成的作业
            pid = int(plane.code.split('_')[-1])
            if plane.is_idle() and not plane.is_completed_all_jobs():
                active_agents[pid] = True


        # 准备全局固定顺序的机位列表 (与 _get_obs 中的 site_list 顺序保持绝对一致)
        site_codes = list(self.sites.keys())
        
        # 初始化记录数组，-1 表示没有历史记录 (例如 Episode 刚开始)
        last_site_indices = [-1] * self.n_agents
        last_op_indices = [-1] * self.n_agents
        
        # 遍历当前场上的飞机
        
        for plane in self.planes.values():
            pid = int(plane.code.split('_')[-1])
            
            # ---------------------------------------------------------
            # A. 计算上一次机位的全局索引
            # ---------------------------------------------------------
            last_site_indices[pid] = plane.last_site_idx
                
            # ---------------------------------------------------------
            # B. 计算上一次工序的全局索引
            # ---------------------------------------------------------
            last_op_indices[pid] = plane.last_job_idx
                    
        return {
            'active_agents': np.array(active_agents), 
            'last_site_indices': np.array(last_site_indices, dtype=np.int32),
            'last_op_indices': np.array(last_op_indices, dtype=np.int32),
        }
    
    def step(self, action):
        '''执行一步纯事件驱动动作，并在内部自动快进时间直到出现可决策状态'''

        # =====================================================================
        # Phase 1A: 动作分配 (Agents' Turn) —— 每次 step 只执行一次
        # =====================================================================
        active_agents = [False] * self.n_agents
        for plane in self.planes.values():
            pid = int(plane.code.split('_')[-1])
            if plane.is_idle() and not plane.is_completed_all_jobs():
                active_agents[pid] = True
        self.current_active_agents = active_agents
        
        for plane_id, plane in self.planes.items():
            pid = int(plane_id.split('_')[2])
            if plane.is_idle() and not plane.is_completed_all_jobs():
                # 解析 RL 动作
                if self.current_active_agents[pid]:
                    job_idx = action[pid][0] - pid * len(self.job_code_list)  
                    site_idx = action[pid][1] 
                    target_job_code = self.job_code_list[job_idx]
                    target_site_code = self.site_code_list[site_idx]
                    target_job = self.jobs[target_job_code]
                    plane.last_site_idx = job_idx
                    plane.last_job_idx = site_idx

                    self.pending_actions[plane_id] = {
                        'step_idx': self.steps,
                        'agent_id': pid,
                        'action': [int(action[pid][0]), int(action[pid][1])], # 确保是原生int
                        'start_time': self.total_time,
                        'plane_id': plane_id,
                        'site_id': target_site_code,
                        'device_ids': [] # 初始化为空，如果用到设备会在后续追加
                    }

                else:
                    target_site_code = plane.site.code
                    target_job = None

                # 动作执行逻辑
                if target_site_code != plane.site.code:
                    if plane.site.get_avail_transporter() is None:
                        if plane.site.code == 'Z': 
                            plane.start_transport(self.sites[target_site_code], None)
                            plane.choosed_job = target_job_code
                        else:
                            plane.start_waiting('ZY-T')
                            # self.waiting_sites['ZY-T'].append(plane.site.code)
                            plane.destination = self.sites[target_site_code]
                            plane.destination.add_plane(plane)
                            plane.choosed_job = target_job_code
                    else:
                        transporter = plane.site.get_avail_transporter()
                        plane.start_transport(self.sites[target_site_code], transporter)
                        plane.choosed_job = target_job_code
                        if transporter and plane_id in self.pending_actions:
                            self.pending_actions[plane_id]['device_ids'].append(transporter.code)
                else:
                    if target_job and target_job.code not in plane.get_avail_jobs(plane.site):
                        if not plane.is_completed_all_jobs():
                            plane.start_waiting(target_job.code)
                            # self.waiting_sites[target_job.code].append(plane.site.code)
                            plane.choosed_job = target_job_code
                            plane.trans_time = 0
                    elif target_job:
                        plane.choose_job(target_job.code)
                        plane.trans_time = 0

            if plane.is_waiting:
                self.waiting_sites[plane.pending_job].append(plane.site.code)

        # =====================================================================
        # 核心重构：内部事件推演循环 (Fast-Forward Loop)
        # =====================================================================
        time_prev = self.total_time
        while True:
            internal_step_time = np.inf
            
            # -----------------------------------------------------------
            # Phase 1B: 环境设备自治调度 (NPCs' Turn)
            # -----------------------------------------------------------
            for job_code, waiting_sites_list in self.waiting_sites.items():
                if not waiting_sites_list: continue
                needed_res_types = ['R014'] if job_code == 'ZY-T' else [res for res in self.jobs.get(job_code).resources if res in self.mobile_devices]
                if not needed_res_types: continue

                idle_devices = self.get_idle_devices(needed_res_types)
                if idle_devices:
                    waiting_planes = [p for p in self.planes.values() if p.site.code in waiting_sites_list]
                    assignments = arrange_devices(idle_devices, waiting_planes)
                    for device, target_site in assignments:
                        device.start_transport(target_site)
                        if target_site.code in waiting_sites_list:
                            waiting_sites_list.remove(target_site.code)
                        for wp in waiting_planes:
                            if wp.site.code == target_site.code:
                                if wp.code in self.pending_actions:
                                    self.pending_actions[wp.code]['device_ids'].append(device.code)
                                break

            # -----------------------------------------------------------
            # Phase 1C: 统计所有忙碌实体的剩余预期时间
            # -----------------------------------------------------------
            for plane in self.planes.values():
                if plane.is_transporting:
                    internal_step_time = min(internal_step_time, plane.left_trans_time)
                elif plane.is_busy:
                    internal_step_time = min(internal_step_time, plane.site.left_job_time)
                    
            for devices in self.mobile_devices.values():
                for device in devices:
                    if not device.is_idle():
                        left_t = device.left_trans_time if device.is_transporting else device.left_rec_time
                        internal_step_time = min(internal_step_time, left_t)
                        
            for site in self.sites.values():
                if site.is_interfered:
                    internal_step_time = min(internal_step_time, site.left_rec_time)

            # -----------------------------------------------------------
            # Phase 2: 时间跃迁 dt 计算
            # -----------------------------------------------------------
            future_landings = [item[0] for item in self.landing_list if item[0] > self.total_time]
            next_landing_dt = future_landings[0] - self.total_time if future_landings else np.inf
            
            dt = min(internal_step_time, next_landing_dt)
            if dt == np.inf:
                # print("[Warning] No upcoming events detected! Force break loop.")
                break
                
            self.total_time += dt 

            # -----------------------------------------------------------
            # Phase 3: 物理状态推演 (时间流逝)
            # -----------------------------------------------------------
            if dt >= 0:
                # 第一段：只结算干涉、正在运输（会到达目的地）和正在作业（会结束作业）的实体
                for site in self.sites.values():
                    if site.is_interfered: site.update(dt)
                        
                for plane in list(self.planes.values()):
                    if plane.is_busy or plane.is_transporting:
                        plane.update(dt)

                for devices in self.mobile_devices.values():
                    for device in devices:
                        if not device.is_idle():
                            device.update(dt)

                # 第二段：结算所有“处于等待状态”的飞机
                # 此时，所有该腾空位置的飞机都已经完成了离开动作，绝不会发生旧飞机挡住新飞机的情况
                for plane in list(self.planes.values()):
                    if plane.is_waiting:
                        plane.update(dt)

            # -----------------------------------------------------------
            # Phase 4: 触发外部事件 (处理到达预定时间的飞机降落)
            # -----------------------------------------------------------
            while self.landing_list and self.total_time >= self.landing_list[0][0]:
                # 直接解包获取完美无误的编号
                land_time, bidx, pidx, _ = self.landing_list.pop(0) 
                
                plane_cfg = {
                    'velocity': 5,
                    'site': self.sites['Z'],
                    'fuel': (self.np_random.integers(0, 30) if hasattr(self, 'np_random') else np.random.randint(0, 30)) if getattr(self, 'use_domain_rand', True) else 30,
                    'jobs': [job for job in self.jobs.values() if job.code in self.job_code_list]
                }
                self.add_planes([{'batch': bidx, 'idx': pidx, **plane_cfg}])

            for plane_id, plane in list(self.planes.items()):
                if plane_id in self.pending_actions:
                    # 动作完成的条件：飞机再次需要下发指令，或者已经彻底结束所有流程准备离场
                    if plane.is_idle() or plane.is_completed_all_jobs():
                        record = self.pending_actions.pop(plane_id)
                        record['trans_time'] = plane.trans_time
                        record['job_time'] = plane.job_time
                        record['total_job_time'] = plane.total_job_time
                        record['waiting_time'] = self.total_time - record['start_time'] - record['trans_time'] - record['job_time']
                        record['end_time'] = self.total_time
                        self.trajectory_log.append(record)
                        
            # 【新增】：极端容错 - 拦截并结算可能被从环境中移除 (remove_planes) 的飞机日志
            for p_id in list(self.pending_actions.keys()):
                if p_id not in self.planes:
                    record = self.pending_actions.pop(p_id)
                    record['trans_time'] = plane.trans_time
                    record['job_time'] = plane.job_time
                    record['total_job_time'] = plane.total_job_time
                    record['waiting_time'] = self.total_time - record['start_time'] - record['trans_time'] - record['job_time']
                    record['end_time'] = self.total_time
                    self.trajectory_log.append(record)
            
            # -----------------------------------------------------------
            # Phase 5: 检查是否可以退出快进循环
            # -----------------------------------------------------------
            self.done = (len(self.planes) == 0 and len(self.landing_list) == 0)
            if self.done:
                break # 环境结束，跳出循环
                
            # 检查场上是否出现了活跃的（可以做决策的）飞机
            has_active = False
            for plane in self.planes.values():
                if plane.is_idle() and not plane.is_completed_all_jobs():
                    has_active = True
                    break
                    
            if has_active:
                break # 有飞机空闲了需要下发动作，跳出循环
        
        self.step_time = self.total_time - time_prev
        self.steps += 1
        # 最终返回观测和奖励
        return self._get_obs(), self._get_reward(), self._get_done(), self._get_info()

    def reset(self):
        super().reset(seed=None)
        
        # 解析域随机化开关 (优先级：options传入 > config配置 > 默认开启)
        self.steps = 0
        self.total_time = 0
        self.step_time = 0
        self.done = False
        self.trajectory_log = []
        self.pending_actions = {}
        
        self.planes.clear()
        self.num_planes = 0
        self.force_transfer_planes.clear()
        for key in self.waiting_sites.keys():
            self.waiting_sites[key] = []
            
        # ==========================================================
        # DR 1: 进场时间扰动 (Arrival Jitter)
        # ==========================================================
        self.landing_list = []
        self.plane_num_per_batch = len(self.flights_data)
        
        # 仅在开启随机化时生成预置飞机
        if self.use_domain_rand:
            num_pre_planes = self.np_random.integers(0, 3) if hasattr(self, 'np_random') else np.random.randint(0, 3)
        else:
            num_pre_planes = 0
            
        # [修改点 3]：基于 base_landing_list 恢复环境并加入随机扰动
        for idx, base_flight in enumerate(self.base_landing_list):
            if idx >= len(self.flights_data) - num_pre_planes:
                continue
                
            base_time = base_flight['land_time']
            
            # ====== 【核心修改：强制第一架飞机在 t=0 降落】 ======
            if base_time == 0:
                land_time = 0
            else:
                jitter = (self.np_random.integers(-5, 6) if hasattr(self, 'np_random') else np.random.randint(-5, 6)) if self.use_domain_rand else 0
                land_time = max(0, base_time + jitter)
            # ====================================================

            self.landing_list.append((land_time, base_flight['bidx'], base_flight['pidx'], base_flight['fuel']))
                
        # 按时间进行排序
        self.landing_list.sort(key=lambda x: x[0])
        
        # ==========================================================
        # DR 3: 移动设备初始位置打乱 
        # (这部分代码保持你上一版的原样，不需要动)
        # ==========================================================
        for site in self.sites.values():
            site.reset()
        gate_codes = [str(i) for i in range(1, len(self.site_code_list) - len(self.runway_code_list))] 
        for devices in self.mobile_devices.values():
            for device in devices:
                device.reset() 
                if self.use_domain_rand and device.resource.type != 'R014': 
                    random_site_code = self.np_random.choice(gate_codes) if hasattr(self, 'np_random') else np.random.choice(gate_codes)
                    device.start_transport(self.sites[random_site_code])
                    device.finish_transport()

        # ==========================================================
        # DR 2 & 4: 初始化进场与在场飞机
        # ==========================================================
        optional_jobs = ['ZY05', 'ZY06', 'ZY09']
        
        # [修改点 4]：处理 0 时刻即到达着陆跑道的飞机
        while self.landing_list and self.total_time >= self.landing_list[0][0]:
            land_time, bidx, pidx, fuel = self.landing_list.pop(0) 
            
            actual_jobs = []
            for job in self.jobs.values():
                if job.code in self.job_code_list:
                    # 随机丢弃非必要作业
                    if self.use_domain_rand and job.code in optional_jobs:
                        if (self.np_random.random() if hasattr(self, 'np_random') else np.random.random()) < 0.2:
                            continue
                    actual_jobs.append(job)
            
            # 油量也加入微小扰动以增加样本多样性
            final_fuel = fuel
            if self.use_domain_rand:
                fuel_jitter = self.np_random.integers(-5, 6) if hasattr(self, 'np_random') else np.random.randint(-5, 6)
                final_fuel = max(0, min(100, fuel + fuel_jitter))
                
            plane_cfg = {
                'velocity': 5,
                'site': self.sites['Z'],
                'fuel': final_fuel,
                'jobs': actual_jobs
            }
            self.add_planes([{'batch': bidx, 'idx': pidx, **plane_cfg}])

        # --- 新增：随机生成开局就已经停在机位上的飞机 ---
        if num_pre_planes > 0:
            chosen_gates = self.np_random.choice(gate_codes, num_pre_planes, replace=False) if hasattr(self, 'np_random') else np.random.choice(gate_codes, num_pre_planes, replace=False)
            for i, gate_code in enumerate(chosen_gates):
                # 【修复 1】：使用合法的正数编号。接在第 0 批次的末尾
                pidx = self.plane_num_per_batch - num_pre_planes + i
                pre_cfg = {
                    'velocity': 5,
                    'site': self.sites[gate_code],
                    'fuel': 100, 
                    'jobs': [job for job in self.jobs.values() if job.code in self.job_code_list]
                }
                # 依然当做 batch 0 注册，这样 global_pid 计算出来是完全合法的
                self.add_planes([{'batch': 0, 'idx': pidx, **pre_cfg}])
                plane_obj = self.planes[f'Plane_0_{pidx}']
                plane_obj.finished_jobs.extend(['ZY_Z', 'ZY_M', 'ZY01'])
                
        return self._get_obs(), self._get_done(), self._get_info()
    
    def _get_episode_rewards(self):
        return self.total_time

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

    def calculate_hindsight_rewards(self) -> Dict[Tuple[int, int], dict]:
        """
        通过前向包络计算每个决策对全局最大完成时间 (Makespan) 的增量。
        如果一个动作的结束时间突破了“当前已知的最大结束时间”，则突破量即为该动作的耗时惩罚。
        
        参数:
            trajectory_log: Episode 中记录的所有动作事件流水
            makespan: 当前 Episode 的总耗时 (env.total_time) - 此处仅作为参考，实际以日志推演为准
        """
        
        # 1. 初始化所有决策步的奖励为 0
        step_rewards = {}
        for record in self.trajectory_log:
            step_rewards[(record['step_idx'], record['agent_id'])] = {
                'action': record['action'],
                'makespan_contribution': 0.0,
                'reward': 1.0*(record['total_job_time'] - record['job_time']) - 2.0*record['waiting_time'] - 1.0*record['trans_time']
            }
        return step_rewards