# grid_world_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
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
        self.n_agents = config.get('n_agents', 0)  # 智能体数量（默认为0，实际根据飞机数量动态调整）
        
        # 强制转运飞机列表：因干涉需要强制移动
        self.force_transfer_planes = []
        
        # 构建MARL动作空间
        # 动作维度1：目标站点 (28个停机位 + 3跑道 + 1等待 = 32)
        # 动作维度2：选择的作业 (18个作业 + 1等待 = 19)
        self.action_space = spaces.MultiDiscrete([len(self.sites) + 1, len(self.jobs) + 1])
        
        # 状态相关
        self.sites_state_global = [-1] * len(self.sites)  # 站位全局状态
        self.state_left_time = np.zeros(len(self.sites))  # 站位剩余时间

        self.site_code_list = list(self.sites.keys())
        self.job_code_list = [job.code for job in self.jobs.values() if job.group == '保障' and job.code not in ['ZY01', 'ZY-L']]
        
        # 根据配置，一次性生成未来所有的飞机降落事件时间表
        self.batch_num = config.get('batch_num', 1)
        self.plane_num_per_batch = config.get('plane_num_per_batch', 12)
        self.landing_list = []
        for bidx in range(self.batch_num):
            self.landing_list += [item + bidx * 3600 for item in list(range(0, 120 * self.plane_num_per_batch, 120))]
        self.landing_list.sort() # 确保时间轴是从小到大排列的
        self.seed(config.get('seed', None))
        
        while self.landing_list and self.total_time >= self.landing_list[0]:
            land_time = self.landing_list.pop(0)
            bidx = land_time // 3600
            pidx = (land_time % 3600) // 120
            plane_cfg = {
                'velocity': 5,
                'site': self.sites['Z'],
                'fuel': self.np_random.integers(0, 30) if hasattr(self, 'np_random') else np.random.randint(0, 30),
                'jobs': self.jobs.values()
            }
            self.add_planes([{'batch': bidx, 'idx': pidx, **plane_cfg}])
        
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
            ret += [device for device in self.mobile_devices[res_type] if device.is_idle()]
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

    def _estimate_system_total_time(self):
        """
        计算当前状态下，系统中所有飞机的预估完工时间之和（Total Flow Time）。
        取代单纯的 C_max，为每个智能体提供密集且公平的奖励信号。
        """
        total_sum = 0
        
        # 1. 预估当前在场的飞机
        for plane in self.planes.values():
            # 基础时间为当前环境绝对时间
            t = self.total_time 
            
            # 加上正在进行的动作剩余时间
            if plane.is_transporting:
                t += plane.left_trans_time
            elif plane.is_busy:
                t += plane.site.left_job_time
                
            # 加上剩余未做作业的预估时间
            for job_code in plane.left_jobs:
                job_obj = self.jobs[job_code]
                t += job_obj.time if job_obj.time else 0
                t += 120  # 预估平均转运时间
                
            total_sum += t
            
        # 2. 预估未来尚未降落的飞机
        for land_time in self.landing_list:
            if land_time >= self.total_time:
                t = land_time
                for job in self.jobs.values():
                    if job.group == '保障' and job.code not in ['ZY01', 'ZY-L']:
                        t += job.time if job.time else 0
                        t += 120
                total_sum += t
                
        # 注：已经完成并飞走的飞机，其完工时间已变成常量。
        # 常量在计算前后的差值(old_sum - new_sum)时会被自然抵消，因此无需纳入求和！
        
        return total_sum

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
            if site.code in ['Z', '29', '30', '31'] or site.is_interfered or site.is_occupied or getattr(site, 'is_reserved', False):
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
        # 3. 构建工序节点 (Operation) & 形状绝对安全的 Mask
        # ==========================================
        # 预分配 Numpy 数组，确保形状铁打不动
        op_features = np.zeros((n_agents * n_ops, 6), dtype=np.float32)
        agent_op_mask = np.zeros((n_agents, n_agents * n_ops), dtype=bool) 
        ptr_site_mask_matrix = np.zeros((n_agents * n_ops, n_sites), dtype=bool)
        
        for global_pid in range(n_agents):
            plane = active_planes.get(global_pid, None)
            
            if plane is not None:
                # 飞机在场上，正常提取状态
                current_avail_jobs = plane.get_avail_jobs(site=None)
                for j_idx, job_code in enumerate(self.job_code_list):
                    # u_idx 是这个作业在全局图中的唯一节点索引
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
                    req_res = 1.0 if len(job_obj.resources.intersection(set(dev_types))) > 0 else 0.0
                    wait_time = float(plane.waiting_time) if status == 1.0 else 0.0
                    
                    op_features[u_idx] = [status, proc_time, rem_ops, req_res, wait_time, float(global_pid)]
                    
                    # 站点掩码计算
                    for s_idx, site in enumerate(site_list):
                        if not global_site_valid[s_idx]:
                            if site.code != 'Z' and site == plane.site and not site.is_interfered:
                                ptr_site_mask_matrix[u_idx, s_idx] = True
                            continue
                            
                        job_idx_in_onehot = self.job_code_list.index(job_code)
                        ptr_site_mask_matrix[u_idx, s_idx] = (site.avail_job_onehot[job_idx_in_onehot] == 1)
            else:
                # 飞机不在场上（未降落或已起飞）：填入幽灵节点 (Dummy Nodes)
                for j_idx in range(n_ops):
                    u_idx = global_pid * n_ops + j_idx
                    op_features[u_idx] = [3.0, 0.0, 0.0, 0.0, 0.0, float(global_pid)]
                    # 【修改 3】：不在场的飞机 op_mask 必须为 True，防止 Ptr-Net 出现 NaN
                    agent_op_mask[global_pid, u_idx] = True 
                    ptr_site_mask_matrix[u_idx, :] = True

            # 【修改 4】：安全保险 - 如果一架在场的飞机当前没任何活能干，强制把它自己的节点置为 True 占位
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
                for s_idx, is_valid in enumerate(ptr_site_mask_matrix[u_idx]):
                    if is_valid:
                        site = site_list[s_idx]
                        edge_os[0].append(u_idx)
                        edge_os[1].append(s_idx)
                        dist = abs(plane.site.pos[0] - site.pos[0]) + abs(plane.site.pos[1] - site.pos[1])
                        attr_os.append([dist / plane.velocity])
                
                # --- C. O-R Edge ---
                needed_dev_types = job_obj.resources.intersection(set(dev_types))
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
        data.site_mask_matrix = torch.tensor(ptr_site_mask_matrix, dtype=torch.bool) # Shape: [n_agents*n_ops, n_sites]
            
        return data

    def _get_reward(self):
        """
        计算每个智能体的即时奖励。
        使用 Total Flow Time 差值，提供极度密集的梯度信号。
        """
        # 1. 提取执行动作并经过 dt 时间跃迁后的【新】预估完工时间总和
        new_total_sum = self._estimate_system_total_time()
        
        if not hasattr(self, 'last_total_sum'):
            self.last_total_sum = new_total_sum
            
        # 2. 核心数学转换：差值即为全局奖励
        # 只要任意一架飞机的进度提前，全局奖励就为正
        # 只要有飞机处于空闲/等待导致时间虚耗，全局奖励必为负
        global_reward = self.last_total_sum - new_total_sum
        
        self.last_total_sum = new_total_sum
        
        # 3. 分发奖励
        rewards = np.zeros((self.n_agents, 1), dtype=np.float32)
        
        for plane in self.planes.values():
            pid = int(plane.code.split('_')[-1])
            # 只有当步积极参与决策、或是被堵塞的活跃飞机，才分配该奖励（或惩罚）
            if self.current_active_agents[pid]:
                rewards[pid, 0] = global_reward
            else:
                rewards[pid, 0] = 0.0
                
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
        '''执行一步纯事件驱动动作，并自动进行时间跃迁
        
        输入:
            action: numpy array - RL网络输出的动作索引 [n_agents, 2] (或兼容旧版dict)
        返回:
            tuple - (obs, reward, done, info)
        '''
        self.steps += 1

        # 获取当前 RL 决策前的活跃状态掩码
        active_agents = [False] * self.n_agents
        for plane in self.planes.values():
            # 活跃条件 1：飞机处于空闲状态（不忙碌、不在运输、不在等待）
            # 活跃条件 2：飞机还有未完成的作业
            pid = int(plane.code.split('_')[-1])
            if plane.is_idle() and not plane.is_completed_all_jobs():
                active_agents[pid] = True
        self.current_active_agents = active_agents
        
        # =====================================================================
        # Phase 1: 动作分配与内部事件时间预期计算
        # =====================================================================
        internal_step_time = np.inf
        
        # -----------------------------------------------------------
        # A. 飞机动作分配 (Agents' Turn)
        # -----------------------------------------------------------
        for plane_id, plane in self.planes.items():
            pid = int(plane_id.split('_')[2])
            
            if plane.is_idle() and not plane.is_completed_all_jobs():
                # 解析 RL 动作
                if self.current_active_agents[pid]:
                    job_idx = action[pid][0] - pid * len(self.job_code_list)  # 0号位是作业(Job)
                    site_idx = action[pid][1] # 1号位是机位(Site)
                    
                    target_job_code = self.job_code_list[job_idx]
                    target_site_code = self.site_code_list[site_idx]
                    target_job = self.jobs[target_job_code]
                    # 记录强化学习的状态锚点
                    plane.last_site_idx = job_idx
                    plane.last_job_idx = site_idx
                else:
                    target_site_code = plane.site.code
                    target_job = None

                # 动作执行逻辑
                if target_site_code != plane.site.code:
                    if plane.site.get_avail_transporter() is None:
                        if plane.site.code == 'Z': 
                            time_needed = plane.start_transport(self.sites[target_site_code], None)
                            internal_step_time = min(internal_step_time, time_needed)
                            plane.choosed_job = target_job_code
                        else:
                            # 缺车，进入转运等待队列
                            plane.start_waiting()
                            self.waiting_sites['ZY-T'].append(plane.site.code)
                            plane.destination = self.sites[target_site_code]
                            plane.destination.add_plane(plane)
                            plane.choosed_job = target_job_code
                            internal_step_time = 0  # 状态改变，算作即刻事件
                    else:
                        time_needed = plane.start_transport(self.sites[target_site_code], plane.site.get_avail_transporter())
                        internal_step_time = min(internal_step_time, time_needed)
                        plane.choosed_job = target_job_code
                else:
                    if target_job and target_job.code not in plane.get_avail_jobs(plane.site):
                        if not plane.is_completed_all_jobs():
                            # 缺作业设备，进入作业等待队列
                            plane.start_waiting()
                            self.waiting_sites[target_job.code].append(plane.site.code)
                            plane.choosed_job = target_job_code
                            internal_step_time = 0 
                    elif target_job:
                        time_needed = plane.choose_job(target_job.code)
                        internal_step_time = min(internal_step_time, time_needed)
                    else:
                        print(f"[Warning] Plane {plane_id} took a no-op action. It will remain idle this step.")

        # -----------------------------------------------------------
        # B. 环境设备自治调度 (NPCs' Turn)
        # 此时飞机已经做完了决定，如果有需要等待设备的飞机，它们已经在 waiting_sites 里了
        # -----------------------------------------------------------
        for job_code, waiting_sites_list in self.waiting_sites.items():
            if not waiting_sites_list:
                continue
                
            # 获取这种需求需要的特定设备类型 ('ZY-T' 是转运需求，特殊处理为 'R014')
            if job_code == 'ZY-T':
                needed_res_types = ['R014']
            else:
                job_obj = self.jobs.get(job_code)
                if not job_obj: continue
                # 找出这种作业依赖的移动设备类型
                needed_res_types = [res for res in job_obj.resources if res in self.mobile_devices]
                
            if not needed_res_types:
                continue

            # 获取对应的空闲设备
            idle_devices = self.get_idle_devices(needed_res_types)
            if idle_devices:
                # 找出需要该设备的飞机对象
                waiting_planes = [p for p in self.planes.values() if p.site.code in waiting_sites_list]
                
                # 调用环境内部的匈牙利算法进行指派
                assignments = arrange_devices(idle_devices, waiting_planes)
                
                for device, target_site in assignments:
                    # 设备开始前往目标机位
                    time_needed = device.start_transport(target_site)
                    internal_step_time = min(internal_step_time, time_needed)
                    # 从等待队列中移除已被响应的需求
                    if target_site.code in waiting_sites_list:
                        waiting_sites_list.remove(target_site.code)

        # -----------------------------------------------------------
        # C. 统计所有忙碌实体的剩余预期时间
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

        # =====================================================================
        # Phase 2: 时间跃迁 dt 计算 (统筹内部与外部事件)
        # =====================================================================
        future_landings = [t for t in self.landing_list if t > self.total_time]
        next_landing_dt = future_landings[0] - self.total_time if future_landings else np.inf
        
        dt = min(internal_step_time, next_landing_dt)
        if dt == np.inf:
            print("[Warning] No upcoming events detected! Setting dt to 0 to avoid infinite loop.")
            
        self.total_time += dt
        self.step_time = dt 

        # =====================================================================
        # Phase 3: 物理状态推演 (时间流逝)
        # =====================================================================
        if dt >= 0:
            for devices in self.mobile_devices.values():
                for device in devices:
                    if not device.is_idle():
                        device.update(dt)
                        
            for plane in self.planes.values():
                if not plane.is_idle():
                    plane.update(dt)
                    
            for site in self.sites.values():
                if site.is_interfered:
                    site.update(dt)

        # =====================================================================
        # Phase 4: 触发外部事件 (处理到达预定时间的飞机降落)
        # =====================================================================
        while self.landing_list and self.total_time >= self.landing_list[0]:
            land_time = self.landing_list.pop(0)
            bidx = land_time // 3600
            pidx = (land_time % 3600) // 120
            plane_cfg = {
                'velocity': 5,
                'site': self.sites['Z'],
                # 兼容不同 numpy 版本的随机数生成
                'fuel': self.np_random.integers(0, 30) if hasattr(self, 'np_random') else np.random.randint(0, 30),
                'jobs': self.jobs.values()
            }
            self.add_planes([{'batch': bidx, 'idx': pidx, **plane_cfg}])

        # =====================================================================
        # Phase 5: 清理已完成的飞机
        # =====================================================================
        # remove_planes = []
        # for plane_id, plane in self.planes.items():
        #     if plane.is_completed_all_jobs() and plane.is_idle():
        #         remove_planes.append(plane_id)
        #         print(f"Plane {plane_id} has completed all jobs and will be removed from the environment.")
        # self.remove_planes(remove_planes)

        # =====================================================================
        # Phase 6: 判断回合结束与数据返回
        # =====================================================================
        # 当场上没飞机了，且未来降落时刻表也空了，宣告 Episode 结束
        self.done = (len(self.planes) == 0 and len(self.landing_list) == 0)
        
        return self._get_obs(), self._get_reward(), self._get_done(), self._get_info()

    def reset(self, seed=None, options=None):
        '''重置环境
        
        输入:
            seed: int或None - 随机种子
            options: dict或None - 额外选项
        返回:
            tuple - (初始观测obs, 是否结束done, 额外信息info)
        作用:
            通过调用各个组件的内部 reset 方法，实现毫秒级的状态复原，
            彻底避免重复读取 JSON 文件带来的极高 I/O 延迟和内存开销。
        '''
        super().reset(seed=seed)
        
        # 1. 基础时间与计数器复位
        self.steps = 0
        self.total_time = 0
        self.step_time = 0
        self.done = False
        
        # 2. 清空动态生成的对象和事件队列
        self.planes.clear()
        self.num_planes = 0
        self.force_transfer_planes.clear()
        for key in self.waiting_sites.keys():
            self.waiting_sites[key] = []
            
        self.landing_list = []
        for bidx in range(self.batch_num):
            self.landing_list += [item + bidx * 3600 for item in list(range(0, 120 * self.plane_num_per_batch, 120))]
        self.landing_list.sort()
        
        # 3. 极其关键的复位顺序：先重置站点，再重置设备！
        # 原因：Site.reset() 会清空机位上的设备列表 (self.devices = [])
        # 随后 Device.reset() 会将设备强制挪回初始机位，并重新挂载到机位的 devices 列表中。
        # 如果顺序反了，设备会丢失拓扑绑定。
        for site in self.sites.values():
            site.reset()
            
        for devices in self.mobile_devices.values():
            for device in devices:
                device.reset()
        
        while self.landing_list and self.total_time >= self.landing_list[0]:
            land_time = self.landing_list.pop(0)
            bidx = land_time // 3600
            pidx = (land_time % 3600) // 120
            plane_cfg = {
                'velocity': 5,
                'site': self.sites['Z'],
                'fuel': self.np_random.integers(0, 30) if hasattr(self, 'np_random') else np.random.randint(0, 30),
                'jobs': self.jobs.values()
            }
            self.add_planes([{'batch': bidx, 'idx': pidx, **plane_cfg}])

        self.last_total_sum = self._estimate_system_total_time()
        
        # 4. 返回 RL Runner 期望的三个初始状态值
        return self._get_obs(), self._get_done(), self._get_info()
    
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