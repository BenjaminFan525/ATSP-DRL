from .multi_field import multiField
from .simulator import EventDrivenSimulator
from .dubins import Dubins
from .utils import COLORS
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import pickle
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import types
import importlib
from concurrent.futures import ThreadPoolExecutor

class FarmScheduleEnv(gym.Env):
    def __init__(self, data_list, max_agent_num, max_node_num):
        super(FarmScheduleEnv, self).__init__()
        
        self.max_agent_num = max_agent_num
        self.max_node_num = max_node_num
        
        self.data_list = data_list 
        self.data_idx = 0
        self.num_fields = len(data_list)
        
        init_data = self._load_data_from_disk(self.data_list[0])
        self._apply_data(init_data, clone=False) # 刚读出来的数据，不需要 clone
        
        # 初始化模拟器
        self.simulator = EventDrivenSimulator(self.field, self.car_cfg, record_traj=False)
        self.veh_num = len(self.car_cfg)
        self.action = [[] for _ in self.car_cfg]
        self.simulator.reset(self.action)
        self.observation = self._get_obs()
        
        # 状态初始化
        self.arrangements = [[] for _ in self.car_cfg]
        self.remain_lines = self.field.working_line_list
        self.complete = False

        # 初始化路径生成器
        self._init_path_generators()

        # 定义空间
        self._init_spaces()

    def _init_spaces(self):
        """初始化 Action 和 Observation Space"""
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for _ in range(self.max_agent_num):
            self.action_space.append(spaces.Discrete(self.max_node_num))
            # 假设 car_cfg 结构一致
            obs_dim = len(self.car_cfg[0]) - 1 + 2
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)) 
        
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.max_agent_num)]

    def _patch_env_modules(self):
        if 'env' not in sys.modules:
            fake_env = types.ModuleType('env')
            fake_env.__path__ = [] 
            sys.modules['env'] = fake_env
        else:
            fake_env = sys.modules['env']
            if not hasattr(fake_env, '__path__'):
                fake_env.__path__ = []

        for mod_name in ['multi_field', 'field']:
            real_mod = None

            if mod_name == 'multi_field' and 'multiField' in globals():
                try:
                    real_mod_name = globals()['multiField'].__module__
                    if real_mod_name in sys.modules:
                        real_mod = sys.modules[real_mod_name]
                except Exception:
                    pass

            if real_mod is None:
                try:
                    if __package__: 
                        real_mod = importlib.import_module(f'.{mod_name}', package=__package__)
                except (ImportError, AttributeError, ValueError):
                    pass

            if real_mod:
                sys.modules[f'env.{mod_name}'] = real_mod
                setattr(fake_env, mod_name, real_mod)

    def _load_data_from_disk(self, data_dir):
        """
        多线程并行读取数据
        """
        # 1. 先在主线程完成环境 Patch (避免多线程竞争修改 sys.modules)
        self._patch_env_modules()

        # 2. 定义单纯的读取函数 (线程安全)
        def load_pickle(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        # 3. 准备文件路径
        paths = {
            'graph': os.path.join(data_dir, 'pygdata.pkl'),
            'car_cfg': os.path.join(data_dir, 'car_cfg.pkl'),
            'field': os.path.join(data_dir, 'field.pkl')
        }

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_key = {executor.submit(load_pickle, path): key for key, path in paths.items()}

            for future in future_to_key:
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    print(f"Error loading {key} from {data_dir}: {e}")
                    raise e 

        return results
    
    def _apply_data(self, data_dict, clone=False):
        data = deepcopy(data_dict) if clone else data_dict
        self.working_graph = data['graph']
        self.car_cfg = data['car_cfg']
        self.field = data['field']

    def _init_path_generators(self):
        """初始化 Dubins 路径生成器"""
        self.path_generators = [Dubins(cfg['min_R'], np.clip(cfg['min_R'] / 4, 1, 5)) for cfg in self.car_cfg]

    def step(self, action, remain_depot=True):
        if remain_depot:
            self.action = [[a[0] - 2*self.veh_num, a[1]] if len(a) > 1 else [] for a in action]
        else:
            self.action = action
            
        for veh in self.simulator.free_vehicles:
            veh_action = self.action[veh.id]
            if len(veh_action) and veh_action[0] >= 0:
                veh.task_queue = [veh_action]
                self.arrangements[veh.id].append(veh_action)
                self.remain_lines.remove(self.field.working_line_list[veh_action[0]])
            else:
                veh.task_queue = []
                
        self.simulator.step()
        
        return self._get_obs(), self._get_reward(), self._get_done(), self._get_info()
    
    def reset(self):
        current_data_dir = self.data_list[self.data_idx]
        
        # 实时 I/O 读取
        new_data = self._load_data_from_disk(current_data_dir)
        
        # 应用数据 (无需 clone，因为 new_data 是刚从文件读出来的全新对象)
        self._apply_data(new_data, clone=False)
        
        # 循环索引
        self.data_idx = (self.data_idx + 1) % len(self.data_list)

        # 重置模拟器
        self.simulator = EventDrivenSimulator(self.field, self.car_cfg, record_traj=False)
        
        self.veh_num = len(self.car_cfg)
        self.action = [[] for _ in self.car_cfg]
        self.simulator.reset(self.action)
        
        self.arrangements = [[] for _ in self.car_cfg]
        self.remain_lines = self.field.working_line_list.copy()
        self.complete = False
        
        # 重置生成器
        self._init_path_generators()
        
        # 重置渲染状态
        self.render_progress = [0] * self.max_agent_num
        self.render_finished_flags = [False] * self.max_agent_num
        self.background_rendered = False
        
        return self._get_obs(), self._get_done(), self._get_info()

    def get_graph(self):
        return self.working_graph

    def shuffer_data(self):
        """
        打乱数据顺序，适用于训练阶段。
        """
        np.random.shuffle(self.data_cache)
        self.data_idx = 0  # 重置索引以从新顺序开始

    def render(self, ax=None):
        """
        增量式渲染：每次调用只绘制上次 render 之后新增的路径。
        包含：行间转移路径(虚线) + 作业行内路径(粗实线)
        """
        # 1. 处理背景绘制
        if ax is None:
            if not self.background_rendered:
                 ax = self.field.render(working_lines=False, show=False)
                 self.background_rendered = True
                 self.last_ax = ax 
            else:
                 ax = getattr(self, 'last_ax', self.field.render(working_lines=False, show=False))
        else:
            if not self.background_rendered:
                self.field.render(ax, working_lines=False, show=False)
                self.background_rendered = True
            self.last_ax = ax

        # 2. 遍历所有车辆，绘制新增路径
        for idx in range(len(self.car_cfg)):
            history = self.arrangements[idx]
            current_len = len(history)
            rendered_len = self.render_progress[idx]
            generator = self.path_generators[idx]
            color = COLORS[idx % len(COLORS)]

            # A. 绘制新增的工作路径片段
            while rendered_len < current_len:
                targets = []
                target_line_name = None # 初始化变量
                
                # --- 情况 1: 这是第一步 (Start -> Line 1) ---
                if rendered_len == 0:
                    start_node_name = f'start-{idx}'
                    start_dir = 1 
                    
                    target_line_idx = history[0][0]
                    target_line_name = self.field.working_line_list[target_line_idx]
                    target_exit_idx = history[0][1]
                    
                    # 路径：Depot -> 第一个行
                    targets = self.field.get_path(start_node_name, target_line_name, start_dir, target_exit_idx, True)

                # --- 情况 2: 后续步骤 (Line A -> Line B) ---
                else:
                    prev_line_idx = history[rendered_len - 1][0]
                    prev_line_name = self.field.working_line_list[prev_line_idx]
                    prev_exit_dir = history[rendered_len - 1][1]
                    
                    target_line_idx = history[rendered_len][0]
                    target_line_name = self.field.working_line_list[target_line_idx]
                    target_exit_idx = history[rendered_len][1]

                    # 路径：上一行终点 -> 这一行终点
                    targets = self.field.get_path(prev_line_name, target_line_name, 1 - prev_exit_dir, target_exit_idx, True)

                # --- 绘制 Dubins 转移路径 (虚线) ---
                if targets:
                    path, _ = generator.dubins_multi(targets)
                    # 只有第一段路径加 label，避免图例重复
                    ax.plot(path[:, 0], path[:, 1], '--', color=color, label=f'Veh{idx+1}' if rendered_len==0 else "")
                
                # =========================================================
                # [新增代码] 绘制作业行内的直线路径 (粗实线)
                # =========================================================
                if target_line_name is not None:
                    field_idx, line_idx = target_line_name.split('-')[1], target_line_name.split('-')[2]
                    # 获取作业行两端节点的坐标 (假设命名规则为 line-0 和 line-1)
                    p0 = self.field.Graph.nodes[f'line_{target_exit_idx}-{field_idx}-{line_idx}']['coord']
                    p1 = self.field.Graph.nodes[f'line_{1-target_exit_idx}-{field_idx}-{line_idx}']['coord']
                    
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 
                            '-',           # 实线
                            color=color,   # 与车辆颜色一致
                            linewidth=2.5, # 线宽加粗 (默认通常是1.5)
                            alpha=0.8)     # 轻微透明度，防止完全遮盖底层细节
                # =========================================================

                # 更新进度
                rendered_len += 1
            
            self.render_progress[idx] = rendered_len

            # B. 绘制回车库路径 (逻辑保持不变)
            is_veh_finished = self.simulator.vehicles[idx].finished
            
            if is_veh_finished and not self.render_finished_flags[idx]:
                targets = []
                end_node_name = f'end-{idx}' if self.field.ends is not None else f'start-{idx}'
                
                if current_len == 0:
                    start_node_name = f'start-{idx}'
                    if np.allclose(self.field.Graph.nodes[f'start-{idx}']['coord'], self.field.Graph.nodes[end_node_name]['coord']):
                        pass
                    else:
                        targets = self.field.get_path(start_node_name, end_node_name, 1, 1, True)
                else:
                    last_line_idx = history[-1][0]
                    last_line_name = self.field.working_line_list[last_line_idx]
                    last_exit_dir = history[-1][1]
                    targets = self.field.get_path(last_line_name, end_node_name, 1 - last_exit_dir, 1, True)
                
                if targets:
                    path, _ = generator.dubins_multi(targets)
                    ax.plot(path[:, 0], path[:, 1], '--', color=color) # 回库路径依然用虚线
                
                self.render_finished_flags[idx] = True

        # 处理图例
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='lower left')

        plt.pause(0.01) 
        return ax
    
    def render_all(self, ax=None, show=False, save_path=None):
        """
        一次性渲染当前所有的车辆路径（忽略增量进度）。
        适用于：Episode 结束后的总结绘图、或者不需要动态显示的场景。
        """
        
        # 1. 准备画布和背景
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            # 绘制农田背景 (边界、障碍物等)
            self.field.render(ax, working_lines=False, show=False)
        else:
            self.field.render(ax, working_lines=False, show=False)

        # 2. 遍历所有车辆
        for idx in range(len(self.car_cfg)):
            history = self.arrangements[idx]
            if not history:
                continue # 该车辆没有分配任务，跳过

            generator = self.path_generators[idx]
            # 尝试获取全局颜色，如果没有则定义一个默认列表
            colors_list = globals().get('COLORS', ['r', 'g', 'b', 'c', 'm', 'y', 'k'])
            color = colors_list[idx % len(colors_list)]
            
            # ==========================================
            # A. 绘制 Start -> 第一个 Task 的路径
            # ==========================================
            start_node_name = f'start-{idx}'
            first_line_idx = history[0][0]
            first_exit_dir = history[0][1] # 注意：这是离开方向
            first_line_name = self.field.working_line_list[first_line_idx]

            # 获取从起点到第一行入口的 Dubins 路径
            # 参数说明: (start_node, target_node, start_dir, target_dir, is_dubins)
            targets = self.field.get_path(start_node_name, first_line_name, 1, first_exit_dir, True)
            
            if targets:
                path, _ = generator.dubins_multi(targets)
                ax.plot(path[:, 0], path[:, 1], '--', color=color, label=f'Veh{idx+1}', alpha=0.7)

            # ==========================================
            # B. 遍历任务历史，绘制 作业行(实线) 和 转移路径(虚线)
            # ==========================================
            for i in range(len(history)):
                curr_line_idx = history[i][0]
                curr_exit_dir = history[i][1]
                curr_line_name = self.field.working_line_list[curr_line_idx]

                # --- 1. 绘制行内作业路径 (粗实线) ---
                # 解析行名称获取 graph 中的节点 key (例如: field-0-line-1)
                # 假设 name 格式为: "type-field_idx-line_idx" 或类似结构，这里参考原代码 split 逻辑
                parts = curr_line_name.split('-')
                field_idx, line_idx = parts[1], parts[2]
                
                # 获取行两端的坐标
                # line_{dir} 代表该行的端点。curr_exit_dir 是出口，1-curr_exit_dir 是入口
                p_end = self.field.Graph.nodes[f'line_{curr_exit_dir}-{field_idx}-{line_idx}']['coord']
                p_start = self.field.Graph.nodes[f'line_{1-curr_exit_dir}-{field_idx}-{line_idx}']['coord']

                ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 
                        '-', color=color, linewidth=2.5, alpha=0.9)

                # --- 2. 绘制从当前行到下一行的转移路径 (虚线) ---
                if i < len(history) - 1:
                    next_line_idx = history[i+1][0]
                    next_exit_dir = history[i+1][1]
                    next_line_name = self.field.working_line_list[next_line_idx]

                    # 路径：当前行出口 -> 下一行入口
                    # 入口方向通常由 target_exit_dir 决定 (具体取决于 get_path 实现，参考原代码逻辑)
                    targets = self.field.get_path(curr_line_name, next_line_name, 
                                                  1 - curr_exit_dir, next_exit_dir, True)
                    if targets:
                        path, _ = generator.dubins_multi(targets)
                        ax.plot(path[:, 0], path[:, 1], '--', color=color, alpha=0.7)

            # ==========================================
            # C. 绘制 回库路径 (如果车辆已完成)
            # ==========================================
            # 检查车辆是否标记为完成，或者我们可以假设 render_all 是在结束时调用，画出最后一段
            if self.simulator.vehicles[idx].finished:
                end_node_name = f'end-{idx}' if self.field.ends is not None else f'start-{idx}'
                
                last_line_idx = history[-1][0]
                last_line_name = self.field.working_line_list[last_line_idx]
                last_exit_dir = history[-1][1]

                # 从最后一行出口回到终点
                targets = self.field.get_path(last_line_name, end_node_name, 1 - last_exit_dir, 1, True)
                if targets:
                    path, _ = generator.dubins_multi(targets)
                    ax.plot(path[:, 0], path[:, 1], '--', color=color, alpha=0.7)

        # 3. 设置图例和标题
        ax.set_title(f'$s_P$={np.round(sum([v.total_dist for v in self.simulator.vehicles]), 2)}m, $t_P$={np.round(self.simulator.global_time, 2)}s, $c_P$={np.round(sum([v.total_cost for v in self.simulator.vehicles]), 2)}L')
            
        # 去重图例
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best')

        ax.set_aspect('equal')
        
        if show:
            plt.show()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)

    def _get_obs(self):
        """
        Get the observation of the environment.

        Output:
        - obs: {'graph': PyG Graph Data, 'vector': (max_agent_num, obs_dim) np.array}
        Each vehicle's observation is a concatenation of its configuration, last_line_idx(the index of the last line the vehicle was on), and last_exit_dir (the type of the last exit (0 or 1)).
        If there are fewer vehicles than max_agent_num, the remaining entries are padded with zeros.
        """
        veh_state = []
        for veh in self.simulator.vehicles:
            veh_state.append(
                np.concatenate([list(veh.cfg.values())[:-1], 
                                [veh.last_line_idx + 2*self.veh_num if veh.last_line_idx else veh.id], 
                                [1 - veh.last_exit_dir if veh.last_exit_dir else 0]])
            )
        for _ in range(self.max_agent_num - len(veh_state)):
            veh_state.append(np.zeros(len(self.car_cfg[0]) - 1 + 2))  # +2 for last_line_idx and last_exit_dir
        return np.stack(veh_state)

    def _get_reward(self):
        return {
            's': np.array([v.reward['s'] for v in self.simulator.vehicles] + [0]*(self.max_agent_num - self.veh_num), dtype=np.float32),
            't': np.array([v.reward['t'] for v in self.simulator.vehicles] + [0]*(self.max_agent_num - self.veh_num), dtype=np.float32),
            'c': np.array([v.reward['c'] for v in self.simulator.vehicles] + [0]*(self.max_agent_num - self.veh_num), dtype=np.float32)}

    def _get_done(self):
        return np.array([veh.finished for veh in self.simulator.vehicles] + [True] * (self.max_agent_num - self.veh_num))

    def _get_info(self):
        return {
            'active_agents': self._get_active_agent(),
            'available_actions': self._get_avail_action(),
            'veh_nums': self.veh_num,
        }

    def _get_active_agent(self):
        """
        Get the active agents in the environment.

        Output:
        - active_agent: List[bool] 
        """
        return np.array([veh not in self.simulator.free_vehicles for veh in self.simulator.vehicles] + [True] * (self.max_agent_num - self.veh_num))
    
    def _get_avail_action(self):
        """
        Get the available actions for each agent in the environment.

        Output:
        - avail_action: List[bool] indicating whether each agent can take an action.
        """
        return np.array([True]*self.veh_num*2 + [line not in self.remain_lines for line in self.field.working_line_list] + [True]*(self.max_node_num - len(self.field.working_line_list) - self.veh_num*2))
    
    def _get_episode_rewards(self):
        return {
            's': sum([v.total_dist for v in self.simulator.vehicles]),
            't': self.simulator.global_time,
            'c': sum([v.total_cost for v in self.simulator.vehicles])
        }