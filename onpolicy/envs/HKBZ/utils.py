import random
import numpy as np
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def seconds_to_datetime(start: str, delta_seconds: int) -> str:
    '''start: "HH:MM:SS"
    return: "HH:MM:SS"  24h制，不跨天
    '''
    t = datetime.strptime(start, "%H:%M:%S") + timedelta(seconds=delta_seconds)
    # 折回同一天
    t = t.replace(day=1)          # 把日期固定到同一天
    return t.strftime("%H:%M:%S")

def datetime_to_seconds(now: str, start: str) -> int:
    '''计算时间差（秒数）'''
    def to_sec(t: str) -> int:
        h, m, s = map(int, t.split(":"))
        return h * 3600 + m * 60 + s
    return (to_sec(now) - to_sec(start)) % (24 * 3600)

def site_disable(env, disable_time: int):
    """
    禁用指定站点
    :param env: 调度环境
    :param site_code: 站点代码
    :param disable_time: 禁用时间
    """
    site_code = random.choice([site.code for site in env.sites if site.code != 'Z'])
    if site_code in env.sites:
        env.add_interfere_sites([site_code], disable_time)
        print(f"Site {site_code} disabled for {disable_time} seconds.")
    else:
        print(f"Site {site_code} does not exist in the environment.")

def device_disable(env, res_type: str, disable_time: int):
    """
    禁用指定设备
    :param env: 调度环境
    :param res_type: 资源类型
    :param device_code: 设备代码
    :param disable_time: 禁用时间
    """
    res_type = random.choice(list(env.mobile_devices.keys()))
    device_code = random.choice([device.resource.code for device in env.mobile_devices.get(res_type, [])])
    if res_type in env.mobile_devices and device_code in [device.resource.code for device in env.mobile_devices[res_type]]:
        for device in env.mobile_devices[res_type]:
            if device.resource.code == device_code:
                for site_code in device.resource.on_service:
                    # 将正在服务的飞机调离
                    env.add_interfere_sites([site_code], 0)
                    env.sites[site_code].is_interfered = False
                device.disable(disable_time)
                print(f"Device {device_code} of type {res_type} disabled for {disable_time} seconds.")
                return
    print(f"Device {device_code} of type {res_type} does not exist in the environment.")

def arrange_devices(devices, planes):
    '''使用匈牙利算法安排转运车
    
    通过最小化总运输距离，将空闲转运车分配给等待的飞机
    支持车多任务少的情况，确保每个任务点被服务一次
    '''
    if not devices or not planes:
        return []

    # 按等待时间排序，优先响应等得最久的飞机
    if len(planes) > len(devices):
        planes = sorted(planes, key=lambda x: x.waiting_time, reverse=True)[:len(devices)]

    device_positions = np.array([device.site.pos for device in devices])
    plane_positions = np.array([plane.site.pos for plane in planes])
    
    # 计算曼哈顿距离矩阵
    cost_matrix = cdist(device_positions, plane_positions, metric='cityblock')
    
    # 匈牙利算法求最小代价匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = []
    for d_idx, p_idx in zip(row_ind, col_ind):
        device, plane = devices[d_idx], planes[p_idx]
        assignments.append((device, plane.site))
        
    return assignments
