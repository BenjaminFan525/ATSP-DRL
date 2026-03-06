import math
import numpy as np

# 设备类：定义移动设备（如转运车）的属性和行为，用于资源在站点间的动态调度
class Device:
    def __init__(self, code, config):
        self.code = code  # 设备唯一标识码
        self.config = config  # 设备配置字典
        self.resource = config['resource']  # 设备携带的资源对象
        self.velocity = config['velocity']  # 设备移动速度
        self.site = config['site']  # 设备当前所在站点
        self.is_transporting = False  # 设备是否处于运输状态
        self.left_trans_time = 0  # 当前运输任务剩余时间（秒）
        self.is_busy = False  # 设备是否忙碌（资源被占用）
        self.left_rec_time = 0
        self.is_disable = False
        # 将设备注册到站点设备列表中
        self.site.devices.append(self)

    def start_transport(self, destination):
        '''启动设备运输任务
        
        输入:
            destination: Site对象 - 目的地站点
        返回:
            time: int - 预计运输时间（秒）
        作用:
            将设备及其携带资源从当前站点移动到目标站点
            从原站点移除资源，更新运输状态和时间
        异常:
            AssertionError - 当设备或资源不空闲时触发
        示例:
            trans_time = device.start_transport(site_B)  # 设备开始运输到站点B
        '''
        assert self.is_transporting is False, "Device must be idle to start transporting!"
        assert self.resource.is_available(), f"Resource must be idle to start transporting!"
        # 从当前站点移除资源
        self.site.remove_resource(self.resource)
        self.site.devices.remove(self)
        # 计算曼哈顿距离和运输时间
        distance = abs(self.site.pos[0]-destination.pos[0]) + abs(self.site.pos[1]-destination.pos[1])
        time = math.ceil(distance / self.velocity)
        self.is_transporting = True
        self.left_trans_time = time
        self.site = destination
        return time

    def finish_transport(self):
        '''完成设备运输任务
        
        作用:
            将设备状态恢复为空闲，资源注册到目标站点
        示例:
            device.finish_transport()  # 设备完成运输
        '''
        # assert self.is_transporting is True and self.destination is not None, "Device must be transporting to finish transport."
        self.is_transporting = False
        self.left_trans_time = 0
        # 更新资源可服务站点列表
        self.resource.sites = [self.site.code]
        self.site.add_resource(self.resource)
        self.site.devices.append(self)
    
    def is_idle(self):
        '''检查设备是否处于空闲状态
        
        返回:
            bool - 如果设备不忙碌且不在运输中返回True，否则返回False
        示例:
            if device.is_idle():  # 检查设备是否可用
                device.start_transport(target_site)
        '''
        return not self.is_transporting and not self.is_disable
    
    def start_disable(self, time):
        self.is_disable = True
        self.left_rec_time = time

    def finish_disable(self):
        self.is_disable = False

    def update(self, time):
        '''更新设备状态（时间推进）
        
        输入:
            time: int - 推进的时间（秒）
        返回:
            int/float - 剩余时间（运输剩余时间或inf）
        作用:
            处理运输倒计时，更新忙碌状态
        异常:
            AssertionError - 当设备处于空闲状态时触发
        示例:
            left_time = device.update(30)  # 推进30秒
        '''
        assert not self.is_idle(), "Device must be either busy or transporting."
        ret = np.inf
        if self.is_disable:
            self.left_rec_time -= time
            if self.left_rec_time == 0:
                self.finish_disable()
            return self.left_rec_time  
        # 处理运输状态
        elif self.is_transporting:
            self.left_trans_time -= time
            # if self.left_trans_time <= 0 and self.resource.type != 'R014':
            assert self.left_trans_time >= 0, "Device transport time cannot be negative."
            if self.left_trans_time == 0:
                self.finish_transport()
            return self.left_trans_time    
        # 更新忙碌状态（与资源可用性同步）
        self.is_busy = not self.resource.is_available()
        return ret

    def reset(self):
        '''重置设备状态
        
        作用:
            将设备恢复到初始配置状态，包括初始站点、空闲状态，并清除所有倒计时。
        示例:
            device.reset()  # 重置设备
        '''
        # 1. 如果设备当前记录的站点不是初始站点，将其从当前站点的设备列表中移除
        if self.site != self.config['site']:
            if self in self.site.devices:
                self.site.devices.remove(self)
                
        # 2. 恢复初始站点
        self.site = self.config['site']
        
        # 3. 重新注册到初始站点（去重保护）
        if self not in self.site.devices:
            self.site.devices.append(self)
            
        # 4. 重置绑定资源的站点位置
        self.resource.sites = [self.site.code]
        
        # 5. 清除所有的状态标志位和倒计时
        self.is_transporting = False
        self.left_trans_time = 0
        self.is_busy = False
        self.is_disable = False
        self.left_rec_time = 0