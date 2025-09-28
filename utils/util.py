"""
一些通用的工具类的封装
"""
import math

# 这个函数没用
def left_planes(chosen_plane, current_idle_planes):
    res = []
    for eve in current_idle_planes:
        if eve not in chosen_plane:
            res.append(eve)
    return res


def min_but_zero(state_left_time):
    non_zero_list = []
    for eve in state_left_time:
        if eve != 0:
            non_zero_list.append(eve)   # 各个站位非0工作时间列表
    if len(non_zero_list) != 0:
        return min(non_zero_list)   # 返回最小的工作时间
    else:
        return 0    # 若每个站位的工作时间都为0则返回0


# 将state_left_time中非0的都减去min_time
def advance_by_min_time(min_time, state_left_time):
    res = []
    for eve in state_left_time:
        if eve != 0:
            assert eve >= min_time
            res.append(eve - min_time)
        else:
            res.append(0)
    return res


# 返回飞机在两个战位之间调运的时间
def count_path_on_road(initial_pos, end_pos, speed):
    return math.sqrt((end_pos[0]-initial_pos[0]) ** 2 + (end_pos[1]-initial_pos[1]) ** 2) / speed


# 计算站位周围占用密度
def count_occupied_density(station, state, row, col, n_agents):
    count = 0
    sum = 0
    directions = [(-1, -1), (-1, 0), (-1, 1),  # 左上，上，右上
                  (0, -1),          (0, 1),    # 左，右
                  (1, -1), (1, 0), (1, 1)]     # 左下，下，右下
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 6:
            sum += 1
            if state[station[nr][nc]][0] != n_agents+1:
                count += 1
    
    return count/sum
