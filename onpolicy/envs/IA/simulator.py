import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .multi_field import multiField
from .dubins import Dubins
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from .car import Robot, DRIVE_MODE
import random

COLORS = (
    [
        # deepmind style
        '#d61600',
        '#002c7c',
        '#18754a',
        '#8700aa',
        'yellow',
        'blue',
        'orange',
        # '#0072B2',
        '#009E73',
        '#D55E00',
        '#CC79A7',
        # '#F0E442',
        '#d73027',  # RED
        # built-in color
        'blue',
        
        'pink',
        'cyan',
        'magenta',
        
        'purple',
        'brown',
        'orange',
        'teal',
        'lightblue',
        'lime',
        'lavender',
        'turquoise',
        'darkgreen',
        'tan',
        'salmon',
        'gold',
        'darkred',
        'darkblue',
        'green',
        # personal color
        '#313695',  # DARK BLUE
        '#74add1',  # LIGHT BLUE
        '#f46d43',  # ORANGE
        '#4daf4a',  # GREEN
        '#984ea3',  # PURPLE
        '#f781bf',  # PINK
        '#ffc832',  # YELLOW
        '#000000',  # BLACK
    ]
)

# ==========================================
# Part 1: Event-Driven Simulator (Fast Mode)
# ==========================================

class VehicleState:
    def __init__(self, idx: int, cfg: dict, tasks: list, record_traj: bool = False):
        self.id = idx
        self.cfg = cfg
        self.record_traj = record_traj
        
        # --- 任务队列 ---
        self.task_queue = tasks 
        
        # --- 状态追踪 ---
        self.next_free_time = 0.0  
        self.finished = False      
        self.last_line_idx = -1 
        self.last_exit_dir = 0 
        
        # --- 统计指标 ---
        self.total_time = 0.0
        self.total_dist = 0.0      # Total distance (Transfer)
        self.total_cost = 0.0
        
        # 细分统计 (为了匹配物理仿真的输出格式)
        self.time_transfer = 0.0
        self.time_work = 0.0
        self.dist_transfer = 0.0
        self.dist_work = 0.0

        self.reward = {'s': 0.0, 't': 0.0, "c": 0.0}
        
        # --- 轨迹记录 ---
        self.trajectory_segments = []

    def update_stats(self, duration: float, dist_to_add: float, mode: str):
        if mode == 'work':
            c_rate = self.cfg['cw']
            self.time_work += duration
            self.dist_work += dist_to_add
            # 在物理仿真定义中，total_dist 通常只计入行驶/转移距离，作业距离不计入
            final_dist_add = 0.0 
        else: # transfer
            c_rate = self.cfg['cv']
            self.time_transfer += duration
            self.dist_transfer += dist_to_add
            final_dist_add = dist_to_add
            
        d_cost = duration * c_rate
        
        self.total_time += duration 
        self.total_dist += final_dist_add
        self.total_cost += d_cost
        
        self.next_free_time += duration

    def update_reward(self, s, t, c):
        self.reward['s'] = s
        self.reward['t'] = t
        self.reward['c'] = c


class EventDrivenSimulator:
    def __init__(self, field, car_cfgs: list, record_traj: bool = False):
        self.field = field
        self.car_cfgs = car_cfgs
        self.vehicles = []
        self.global_time = 0.0
        self.record_traj = record_traj

    def reset(self, arrangements: list):
        self.global_time = 0.0
        self.vehicles = []
        for idx, (arrange, cfg) in enumerate(zip(arrangements, self.car_cfgs)):
            # 这里需注意：物理仿真传入的 arrange 是 list of tuples，直接给 VehicleState
            veh = VehicleState(idx, cfg, list(arrange), record_traj=self.record_traj)
            self.vehicles.append(veh)
        self.free_vehicles = self.vehicles

    def _get_path_points(self, start_desc, end_desc):
        """仅用于渲染时的路径点获取"""
        p_start, p_start_entry = None, 1
        p_end, p_end_entry = None, 1
        
        if isinstance(start_desc, str): p_start = start_desc
        else: p_start, p_start_entry = start_desc
            
        if isinstance(end_desc, str): p_end = end_desc
        else: p_end, p_end_entry = end_desc
            
        return self.field.get_path(p_start, p_end, p_start_entry, p_end_entry, True)

    def step(self):
        if all(v.finished for v in self.vehicles):
            for veh in self.vehicles:
                veh.update_reward(0, 0, 0)
            return True

        for veh in self.free_vehicles:
            t_prev = max(v.next_free_time for v in self.vehicles)
            s_prev = veh.total_dist
            c_prev = veh.total_cost
            # Case A: Has tasks
            if veh.task_queue:
                next_line_idx, next_entry = veh.task_queue.pop(0)
                
                # === Phase 1: Transfer ===
                transfer_dist = 0.0
                
                if veh.last_line_idx == -1:
                    transfer_dist = self.field.ori[veh.id, next_line_idx, next_entry, 0]
                else:
                    transfer_dist = self.field.D_matrix[veh.last_line_idx, next_line_idx, veh.last_exit_dir, next_entry]

                t_duration = transfer_dist / veh.cfg['vv'] if veh.cfg['vv'] > 0 else 0
                veh.update_stats(t_duration, transfer_dist, mode='transfer')
                
                # === Phase 2: Work ===
                work_dist = self.field.line_length[next_line_idx]
                w_duration = work_dist / veh.cfg['vw'] if veh.cfg['vw'] > 0 else 0
                veh.update_stats(w_duration, work_dist, mode='work')
                
                veh.last_line_idx = next_line_idx
                veh.last_exit_dir = 1 - next_entry 
                
            # Case B: Go Home
            elif not veh.finished:
                if veh.last_line_idx != -1:
                    transfer_dist = self.field.des[veh.id, veh.last_line_idx, veh.last_exit_dir, 0]
                else:
                    transfer_dist = 0
                t_duration = transfer_dist / veh.cfg['vv'] if veh.cfg['vv'] > 0 else 0
                veh.update_stats(t_duration, transfer_dist, mode='transfer')
                veh.finished = True
            
            veh.update_reward(veh.total_dist - s_prev, 
                                max(max(v.next_free_time for v in self.vehicles) - t_prev, 0), 
                                veh.total_cost - c_prev)

        active_vehicles = [v for v in self.vehicles if not v.finished]
        if not active_vehicles: 
            self.global_time = max([v.total_time for v in self.vehicles])
            self.free_vehicles = []
            return True
            
        earliest_free_time = min(v.next_free_time for v in active_vehicles)
        if self.global_time < earliest_free_time:
            self.global_time = earliest_free_time

        epsilon = 1e-5
        self.free_vehicles = [v for v in active_vehicles if v.next_free_time <= self.global_time + epsilon]

        return False

    def run(self):
        while not self.step():
            pass
        return self.vehicles

    def render(self, ax=None, show=True, label=False):
        if ax is None:
            _, ax = plt.subplots()
            ax.axis('equal')
        
        # self.field.render(ax, working_lines=False, show=False)
        colors = ['#d61600', '#002c7c', '#18754a', '#8700aa', 'orange', 'blue']
        
        for i, veh in enumerate(self.vehicles):
            c = colors[i % len(colors)]
            label_text = f'Veh{i+1}' if label else None
            # Plot dummy for legend
            if label:
                ax.plot([], [], '-', color=c, linewidth=2, label=label_text)

            for path_points, mode in veh.trajectory_segments:
                if path_points is None or len(path_points) < 2: continue
                pts = np.array(path_points)
                if pts.shape[1] > 2: pts = pts[:, :2]
                
                style = '-' if mode == 'work' else '--'
                alpha = 0.9 if mode == 'work' else 0.6
                lw = 2 if mode == 'work' else 1
                ax.plot(pts[:, 0], pts[:, 1], style, color=c, linewidth=lw, alpha=alpha)
                    
        if label:
            ax.legend(fontsize=16, loc='lower left')
        if show:
            plt.show()
        return ax

# ==========================================
# Part 2: Physical Simulator (Existing Code)
# ==========================================

class PhysicalSimulator:
    def __init__(self, path: list, car_list: list, line_nodes: list, lines_id: list, field: multiField, 
                 working_list: list = None, car_tracking = None, ctrl_period = 0.1, max_step = 5000,
                 time_teriminate=None) -> None:
        self.path_list = path # 路径列表
        self.car_list = car_list # 车辆列表
        self.working_list = working_list if working_list is not None else [[True for _ in range(len(p))] for p in self.path_list]
        self.line_nodes = line_nodes
        self.lines_id = lines_id
        self.field = field
        self.time_teriminate = time_teriminate
        self.dynamic_terminate = False
        self.veh_terminate_idx = []
        # self.car_status = ['home', [0, 0]]*len(car_list)
        self.car_status = [{'line': self.field.nodes_list[0+idx], 'entry': None, 'pos': path[0][0], 'inline': False, 'teriminate': False, 'traveled': []} for idx, _ in enumerate(range(len(car_list)))]
        self.traveled_line = [[] for _ in range(len(car_list))]
        if car_tracking:
            self.car_tracking = car_tracking # 车辆是否跟踪路径
        else:
            self.car_tracking = [True] * len(car_list)
        for car, tracking, working in zip(self.car_list, self.car_tracking, self.working_list):
            if tracking and hasattr(car, 'working'):
                car.working = working[0]
        self.tracking_vector_list = []
        for path in self.path_list:
            if len(path) < 2:
                self.tracking_vector_list.append(None)
            else:
                while np.allclose(path[0], path[1]):
                    path = path[1:, :]
                t = path[1] - path[0]
                t = t / np.linalg.norm(t)
                self.tracking_vector_list.append(t)

        self.drive_mode = "physical"
        self.drive_mode_ = DRIVE_MODE.AUTO_VEL_PSI

        self.ctrl_period = ctrl_period # 上位机控制周期
        self.inner_period = self.car_list[0].ctrl_period # 内环PID控制周期
        self.inner_loop_times = int(self.ctrl_period / self.inner_period) # 每次上位机控制后的内环PID控制次数
        self.ctrl_period = self.inner_loop_times * self.inner_period

        self.time = 0.
        self.steps: int = 0
        self.max_step = max_step
        self.terminate = False
        # self.max_step = len(self.path_list[0])//2
    
    # One contrl step. 
    # 'a' means 'action'. 
    # Action should be a ndarray with shape (number of cars)x2 or 2x(number of cars). 
    # An action of a car is a tuple (v, psi), which defines the desired velosity vector in the car's body axis. 
    # If a car is in tracking mode, the action tuple represents (working velosity, driving velosity). 
    # Return True if all cars in tracking mode finished their job and the other cars have stopped.
    # Else return False.
    def step(self, a: np.ndarray):
        
        # assert a.shape == (len(self.car_list), 2) or a.shape == (2, len(self.car_list))
        # if a.shape == (2, len(self.car_list)):
        #     a = a.T

        self.terminate = True
        for idx, car in enumerate(self.car_list):
            if self.car_tracking[idx]:
                # 没有要跟踪的路径， 停车
                if self.tracking_vector_list[idx] is None:
                    car.stop()
                # 经过了终点，换下一个路径
                elif (car.position() - self.path_list[idx][1]) @ self.tracking_vector_list[idx] >= 0:
                    if len(self.line_nodes[idx]) and np.allclose(self.path_list[idx][0], self.line_nodes[idx][0]):
                        self.line_nodes[idx] = np.delete(self.line_nodes[idx], 0, axis=0)
                        if len(self.line_nodes[idx]) % 2 == 0:
                            line, entry = self.lines_id[idx].pop(0)
                            self.traveled_line[idx].append([line, entry])
                            self.car_status[idx]['traveled'].append(self.field.working_line_list[line])
                    
                    self.car_status[idx]['line'] = self.field.working_line_list[self.lines_id[idx][0][0]] if len(self.lines_id[idx]) else self.field.nodes_list[idx+len(self.car_list)]
                    self.car_status[idx]['entry'] = self.lines_id[idx][0][1] if len(self.lines_id[idx]) else None
                    self.car_status[idx]['pos'] = self.path_list[idx][0]
                    self.car_status[idx]['inline'] = len(self.line_nodes[idx]) % 2 != 0
                    
                    self.path_list[idx] = np.delete(self.path_list[idx], 0, axis=0) # 删除路径中的第一个点(原路径起点)
                    self.working_list[idx].pop(0)
                    if len(self.path_list[idx]) < 2:
                        self.tracking_vector_list[idx] = None
                        car.stop()
                    elif self.time_teriminate and self.time > self.time_teriminate:
                        self.car_status[idx]['teriminate'] = True
                        car.stop()
                    elif self.dynamic_terminate and 'end' in self.car_status[idx]['line']:
                        self.car_status[idx]['teriminate'] = True
                        self.tracking_vector_list[idx] = None
                        car.stop()  
                        if idx in self.veh_terminate_idx:
                            for status in self.car_status:
                                status['teriminate'] = True                 
                    elif self.car_status[idx]['teriminate']:
                        car.stop()
                    else:
                        self.terminate = False
                        self.car_status[idx]['teriminate'] = False
                        if hasattr(car, 'working'):
                            car.working = self.working_list[idx][0]
                        t = self.path_list[idx][1] - self.path_list[idx][0]
                        self.tracking_vector_list[idx] = t / np.linalg.norm(t)
                
                if self.tracking_vector_list[idx] is not None and self.car_status[idx]['teriminate'] == False:
                    self.terminate = False
                    car.follow_trail(self.tracking_vector_list[idx], 
                                     self.path_list[idx][0], 
                                     a[idx][0] if self.working_list[idx][0] else a[idx][1], 
                                     follow = False)
            else:
                car.v_des, car.psi_des = a[idx]
            
            for _ in range(self.inner_loop_times):
                car.update_state(self.drive_mode_)
                # if idx == 0:
                #     print(car.state.v)

            if car.v() > 0:
                self.terminate = False

        self.steps += 1
        self.time += self.ctrl_period

        if self.steps >= self.max_step:
            self.terminate = True
        return self.terminate 

    # Finish one trajectory
    def rollout(self, a, render = False, ax = None, update = True, show = True, output_figure = False, label=False):
        # ori_ax = deepcopy(ax)
        
        if render:
            if ax is None:
                _, ax = plt.subplots()
                ax.axis('equal')
            ori_ax = len(ax.lines)
            if output_figure:
                figures = []
            # text = ax.text(0.01, 0.99, 'Initializing', 
            #             horizontalalignment='left', 
            #             verticalalignment='top',
            #             transform=ax.transAxes)
        
        while not self.step(a):
            if render:
                if self.steps % 15 == 0:
                    if update:
                        while len(ax.lines) > ori_ax:
                            ax.lines.pop()
                    self.render(ax, show = False, label=label)
                    # disp_str = ''.join([f"Car {idx+1} | line {status['line']} | pos ({status['pos'][0]:.2f}, {status['pos'][1]:.2f}) | inline {status['inline']}\n" for idx, status in enumerate(self.car_status)])
                    # disp_str = str(self.time)
                    # text.set_text(disp_str)
                    if output_figure:
                        canvas = FigureCanvasAgg(plt.gcf())
                        w, h = canvas.get_width_height()
                        canvas.draw()
                        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                        buf.shape = (w, h, 3)
                        buf = buf[:, :, [2, 1, 0]]
                        # buf = np.roll(buf, 3, axis=2)
                        image = Image.frombytes("RGB", (w, h), buf.tobytes())
                        figures.append(np.asarray(image)[:, :, :3])
                    if show:
                        plt.pause(0.01)

        return figures if render and output_figure else None
    
    def render(self, ax = None, show = True, label=False):
        if ax is None:
            _, ax = plt.subplots()
            ax.axis('equal')

        for idx, car in enumerate(self.car_list):
            if label:
                car.plot(color = COLORS[idx % len(COLORS)], mode = 1, ax = ax, label=f'Veh{idx+1}')
                ax.legend(fontsize=16, loc='lower left')
            else:
                car.plot(color = '#646464', mode = 1, ax = ax)

        # print(self.working_direction)   
        if show:
            plt.show()
            
        return ax 
    
    def set_drive_mode(self, mode):
        if mode == "physical":
            self.drive_mode_ = DRIVE_MODE.AUTO_VEL_PSI
        else:
            assert mode == "direct"
            self.drive_mode_ = DRIVE_MODE.DIRECT_PSI
        self.drive_mode = mode

    # 设置车辆工作模式 
    # tracking=True表示跟踪路径，action=(v, _)
    # tracking=False表示直接控制，action=(v, psi)
    def set_tracking_mode(self, idx, tracking):
        if isinstance(idx, int):
            self.car_tracking[idx] = tracking
        else:
            for i in idx:
                self.car_tracking[i] = tracking[i]

    def set_path(self, idx, path):
        if isinstance(idx, int):
            self.path_list[idx] = path
        else:
            for i in idx:
                self.path_list[i] = path[i]
    
    def add_path(self, idx, path):
        if isinstance(idx, int):
            np.vstack(self.path_list[idx], path)
        else:
            for i in idx:
                np.vstack(self.path_list[i], path[i])
    
    def pop(self, idx):
        if isinstance(idx, int):
            self.path_list.pop(idx)
            self.car_list.pop(idx)
        else:
            idx.sort()
            for i in idx[::-1]:
                self.path_list.pop(i)
                self.car_list.pop(i)
    
    def add(self, path, car):
        if isinstance(path, list):
            assert len(path) == len(car)
            self.path_list += path
            self.car_list += car
            for path in self.path_list:
                if len(path) < 2:
                    self.tracking_vector_list.append(None)
                else:
                    t = path[1] - path[0]
                    t = t / np.linalg.norm(t)
                    self.tracking_vector_list.append(t)
        else:
            self.path_list.append(path)
            self.car_list.append(car)
            if len(path) < 2:
                self.tracking_vector_list.append(None)
            else:
                t = path[1] - path[0]
                t = t / np.linalg.norm(t)
                self.tracking_vector_list.append(t)
            

def get_working_path(path_g, targets, ori_working_list):
    # 辅助函数，保持不变
    if hasattr(path_g, 'dubins_multi'): # Handle single generator case
         path_g = [path_g] * len(targets)
    paths = []
    working_lists = []
    info_list = []
    for target, working_list, generator in zip(targets, ori_working_list, path_g):
        if len(target) > 1:
            path, info = generator.dubins_multi(target)
        else:
            path = np.array(target)
            info = {'straight': [0], 'sp':[0], 'length': 0.}
        paths.append(path)
        info_list.append(info)
        working_ = []
        for idx, working in enumerate(working_list[:-1]):
            working_ += [working for _ in range(info['sp'][idx + 1] - info['sp'][idx])]
        if len(working_) == 0:
            working_ = [False]
        working_ += [working_[-1]]
        working_lists.append(working_[1:])
    return paths, working_lists, info_list

# ==========================================
# Part 3: Integrated Wrapper
# ==========================================
 
class arrangeSimulator():
    def __init__(self, field, car_cfg) -> None:
        self.field = field
        self.car_cfg = car_cfg
        self.sim_type = 'physical'
        self.simulator = None     # For physical
        self.event_sim = None     # For event-driven
        
    def init_simulation(self, arrangements, debug = False, drive_mode = 'direct', 
                        init_simulator = True, dense_straight = True, path_split = None,
                        sim_type = 'physical'): # 新增 sim_type 参数
        
        self.sim_type = sim_type
        
        if self.sim_type == 'event':
            # Event Mode: 初始化 EventDrivenSimulator
            # 这里的 arrangements 结构 [(line_idx, entry), ...] 正好是 Event Sim 需要的
            self.event_sim = EventDrivenSimulator(self.field, self.car_cfg, record_traj=False)
            self.event_sim.reset(arrangements)
            
            # 返回空路径或占位符，因为不需要 Dubins
            return [], [], []

        else:
            targets = []
            ori_working_list = []
            lines_id = [[] for _ in arrangements]
            line_nodes = [[] for _ in arrangements]
            for idx, arrange in enumerate(arrangements):
                if len(arrange) == 0:
                    if np.allclose(self.field.Graph.nodes[self.field.starts[idx]]['coord'], self.field.Graph.nodes[self.field.ends[idx]]['coord']):
                        targets.append([self.field.Graph.nodes[self.field.starts[idx]]['coord']])
                        ori_working_list.append([False])
                    else:
                        targets.append(self.field.get_path(f'start-{idx}', f'end-{idx}', 1, 1, True))
                        ori_working_list.append([False, False])
                    continue
                lines_id[idx] = [line for line in arrange]
                targets.append(self.field.get_path(f'start-{idx}', self.field.working_line_list[arrange[0][0]], 1, arrange[0][1], True))
                # if len(targets[idx]) > 1:
                ori_working_list.append([False, True])
                # else:
                #     ori_working_list.append([False])
                for working_line in arrange:
                    line_name = self.field.working_line_list[working_line[0]]
                    point_name = [
                        self.field.working_graph.nodes[line_name][f'end{working_line[1]}'],
                        self.field.working_graph.nodes[line_name][f'end{1-working_line[1]}']
                    ]
                    line_nodes[idx] += [self.field.Graph.nodes[name]['coord'] for name in point_name]
                
                for line1, line2 in zip(arrange[:-1], arrange[1:]):
                    delta_target = self.field.get_path(self.field.working_line_list[line1[0]], self.field.working_line_list[line2[0]], 1 - line1[1], line2[1], True)
                    targets[idx] += delta_target
                    ori_working_list[idx] += [False, True]
                if self.field.ends is None:
                    delta_target = self.field.get_path(self.field.working_line_list[arrange[-1][0]], f'start-{idx}', 1 - arrange[-1][1], 1, True)
                else:
                    delta_target = self.field.get_path(self.field.working_line_list[arrange[-1][0]], f'end-{idx}', 1 - arrange[-1][1], 1, True)
                targets[idx] += delta_target
                ori_working_list[idx] += [False, False]      
            
            path_g = [Dubins(cfg['min_R'], 
                                    np.clip(cfg['min_R'] / 4, 1, 5) if path_split is None else path_split, 
                                    dense_straight) 
                        for cfg in self.car_cfg]
            self.paths, working_list, info_list = get_working_path(path_g, targets, ori_working_list)
            
            if not init_simulator:
                return targets, ori_working_list
            
            first_dir = [0 for _ in range(len(self.car_cfg))]
            for idx, path in enumerate(self.paths):
                if len(path) < 2:
                    continue
                while np.allclose(path[0], path[1]):
                    path = path[1:, :]
                first_dir_tmp = path[1] - path[0]
                first_dir[idx] = np.arctan2(first_dir_tmp[1], first_dir_tmp[0])

            init_pos = [[self.field.Graph.nodes[self.field.starts[idx]]['coord'][0], self.field.Graph.nodes[self.field.starts[idx]]['coord'][1]] for idx in range(len(self.car_cfg))]
            
            cars = [Robot({'car_model': "car3", 'working_width': self.field.working_width, **cfg}, 
                    state_ini=[pos[0], pos[1], 0, 0, f_dir, 0], debug=debug) 
                for path, f_dir, cfg, pos in zip(self.paths, first_dir, self.car_cfg, init_pos)]
                
            self.simulator = PhysicalSimulator(self.paths, cars, line_nodes, lines_id, self.field, working_list, max_step=1000000)
            self.simulator.set_drive_mode(drive_mode)
            return self.paths, working_list, info_list
        
    def render_arrange(self, ax=None):
        if ax is None:
            ax = self.field.render(working_lines=False, show=False)
        else:
            self.field.render(ax, working_lines=False, show=False)
        
        # 只有在 physical 模式下才有 self.paths
        if self.sim_type == 'physical' and hasattr(self, 'paths'):
            for idx, path in enumerate(self.paths):
                ax.plot(path[:, 0], path[:, 1], '--', color = COLORS[idx % len(COLORS)], label=f'Veh{idx+1}')
            ax.legend(fontsize=16, loc='lower left')
        return ax
    
    def simulate(self, ax = None, render = False, show = True, output_figure = False, label = False):
        
        # === Case A: Event Driven ===
        if self.sim_type == 'event':
            # 如果需要渲染，开启轨迹记录
            if render:
                self.event_sim.record_traj = True
                if ax is None:
                    ax = self.render_arrange() # 虽然没有 dubins paths，但可以画地图

            # 运行仿真
            vehicles = self.event_sim.run()
            
            # 渲染
            figs = None
            if render:
                self.event_sim.render(ax, show=show, label=label)
                # Event Sim 是一次性计算，不支持生成帧动画列表 (figures)，除非重写渲染逻辑
                # 此处 figs 返回 None
                self.event_sim.record_traj = False # 重置

            # 收集数据以匹配 Physical Sim 的输出格式
            # Physical returns: t, c, s, car_time(list of [tv, tw]), car_dis(list of [dv, dw]), ax, figs
            
            # 计算 Makespan (Max Time)
            t = max([v.next_free_time for v in vehicles]) if vehicles else 0
            c = sum([v.total_cost for v in vehicles])
            s = sum([v.total_dist for v in vehicles]) # 只包含 Transfer dist
            
            car_time = [[v.time_transfer, v.time_work] for v in vehicles]
            car_dis = [[v.dist_transfer, v.dist_work] for v in vehicles]
            
            return t, c, s, car_time, car_dis, ax, figs

        # === Case B: Physical ===
        else:
            if ax == None and render:
                ax = self.render_arrange()
            
            velosity = np.array([[cfg['vw'], cfg['vv']] for cfg in self.car_cfg])
            figs = self.simulator.rollout(velosity, render=render, ax=ax, show=show, output_figure=output_figure, label=label)
            
            s = 0
            c = 0
            car_time = []
            car_dis = []
            for robo in self.simulator.car_list:
                s += robo.total_distance
                c += robo.total_cost
                if robo.debug:
                    car_time.append([robo.tv, robo.tw])
                    car_dis.append([robo.driving_distance, robo.working_distance])
                else:
                    car_time.append(None)
                    car_dis.append(None)
            t = self.simulator.time
            return t, c, s, car_time, car_dis, ax, figs