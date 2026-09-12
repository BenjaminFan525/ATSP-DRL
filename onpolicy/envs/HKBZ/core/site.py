class Site:
    # 站点类：定义一个工作站的属性和行为，用于管理飞机停靠站点的状态、资源分配和作业执行
    def __init__(self, code, config):
        self.code = code  # 站点唯一标识码
        self.config = config  # 站点配置字典，包含位置、资源、作业等信息
        self.pos = config['position']  # 站点物理位置坐标
        self.plane = None  # 当前停靠的飞机对象
        self.devices = []  # 站点部署的设备对象列表
        self.is_occupied = False  # 站点是否被飞机占用
        self.is_interfered = False  # 站点是否受到干涉（如天气、冲突等）
        self.left_rec_time = 0  # 剩余干涉恢复时间（秒）
        # 当前正在进行的作业字典，key为作业代码，value为[剩余时间, 资源代码]
        self.onging_jobs = {}
        self.left_job_time = 0  # 当前所有作业的剩余时间（取最大值）
        # 初始化站点可用资源字典
        self.resources = {}
        # 遍历固定和移动资源，将支持本站的资源加入资源列表
        for res in self.config['fixed_resources'] + self.config['mobile_resources']:
            if self.code in res.sites:
                self.resources[res.code] = res
        
        # The network contract keeps site nodes checkpoint-compatible at
        # 5 base features + 17 protection-resource flags. Transfer/departure
        # compatibility is represented by operation-site masks and pair
        # features, not by widening the site node.
        self.target_jobs = [
            job for job in self.config['jobs'].values()
            if job.group == '保障' and job.code not in ['ZY01', 'ZY-L']
        ]
        self.avail_job_onehot = [0] * len(self.target_jobs)
        
        self.update_resources()

    def add_plane(self, plane):
        '''将飞机分配到当前站点
        
        输入:
            plane: Plane对象 - 需要停靠的飞机
        作用:
            设置飞机占用状态，记录飞机对象
        异常:
            AssertionError - 当站点已被其他飞机占用时触发
        示例:
            site.add_plane(plane_instance)  # 飞机停靠到站点
        '''
        assert self.is_occupied is False or self.plane == plane, f"Site {self.code} is already occupied with plane {self.plane.code} cannot add plane {plane.code}!"
        self.plane = plane
        self.is_occupied = True
        # print(f"Plane {plane.code} has reached Site {self.code}.")

    def remove_plane(self):
        '''将飞机从当前站点移除
        
        作用:
            清空飞机对象，释放站点占用状态
        异常:
            AssertionError - 当站点没有飞机时触发
        示例:
            site.remove_plane()  # 飞机离开站点
        '''
        assert self.is_occupied is True, f"Site {self.code} is already empty!"
        # print(f"Plane {self.plane.code} has left Site {self.code}.")
        self.plane = None
        self.is_occupied = False

    def add_resource(self, resource):
        '''向站点添加资源
        
        输入:
            resource: Resource对象 - 需要添加的资源
        作用:
            将资源加入站点资源列表，并更新可用资源字典
        示例:
            site.add_resource(resource_instance)  # 添加资源到站点
        '''
        self.resources[resource.code] = resource
        self.update_resources()

    def remove_resource(self, resource):
        '''从站点移除资源
        
        输入:
            resource: Resource对象 - 需要移除的资源
        作用:
            从资源列表中删除资源，并更新可用资源字典
        示例:
            site.remove_resource(resource_instance)  # 从站点移除资源
        '''
        if resource.code in self.resources:
            del self.resources[resource.code]
        self.update_resources()

    def start_jobs(self, jobs):
        '''在站点启动一组作业
        
        输入:
            jobs: Job对象列表 - 需要启动的作业
        作用:
            为每个作业分配可用资源，并开始执行
            记录作业剩余时间和使用的资源（可能为None）
        TODO: 优化资源使用逻辑
        示例:
            site.start_jobs([job1, job2])  # 启动多个作业
        '''
        # TODO: 优化资源使用逻辑
        for job in jobs:
            assert job.code not in self.onging_jobs, f"Job {job.code} is already being serviced at Site {self.code}!"
            res = None
            # 遍历作业所需资源类型，寻找可用资源
            for res_code in sorted(job.resources):
                if res_code in self.res_avail:
                    # 分配资源并记录
                    self.resources[self.res_avail[res_code][-1]].add_service(self.code)
                    res = self.res_avail[res_code][-1]
                    break
            # 记录作业状态：[剩余时间, 资源代码]
            self.onging_jobs[job.code] = [job.time, res]
        self.update_resources()
    
    def finish_jobs(self, jobs):
        '''完成站点的一组作业
        
        输入:
            jobs: Job对象列表 - 需要完成的作业
        作用:
            释放作业占用的资源，从正在作业列表中移除
        示例:
            site.finish_jobs([job1, job2])  # 完成多个作业
        '''
        for job in jobs:
            assert job.code in self.onging_jobs, f"Job {job.code} is not being serviced at Site {self.code}!"
            # 如果有占用资源，释放资源服务
            if self.onging_jobs[job.code][1] is not None:
                self.resources[self.onging_jobs[job.code][1]].remove_service(self.code)
            # 从正在作业列表中移除
            self.onging_jobs.pop(job.code)
        self.update_resources()

    def start_interfere(self, rec_time):
        '''启动站点干涉状态
        
        输入:
            rec_time: int - 干涉恢复时间（秒）
        返回:
            plane_code: str - 当前占用站点的飞机代码（如果有），否则不返回任何值
        作用:
            设置干涉状态，暂停所有作业，处理飞机状态
        异常:
            AssertionError - 当站点已被干涉时触发
        示例:
            plane_code = site.start_interfere(300)  # 启动5分钟干涉
        '''
        assert self.is_interfered is False, f"Site {self.code} is already interfered!"
        self.is_interfered = True
        self.left_rec_time = rec_time
        if self.is_occupied:
            # 暂停当前正在进行的作业
            suspended_jobs = []
            for job_code in self.onging_jobs:
                suspended_jobs.append(job_code)
            self.finish_jobs([self.config['jobs'][job] for job in suspended_jobs])
            # 处理当前飞机的作业
            self.plane.current_jobs = [job for job in self.plane.current_jobs if job not in suspended_jobs]
            self.plane.finished_jobs += self.plane.current_jobs
            self.plane.left_jobs = [job for job in self.plane.left_jobs if job not in self.plane.current_jobs]
            self.plane.current_jobs = []
            self.plane.is_busy = False
            # self.plane.site = None
            return self.plane.code

    def finish_interfere(self):
        '''结束站点干涉状态
        
        作用:
            清除干涉状态，重置站点到初始状态
        异常:
            AssertionError - 当站点未被干涉时触发
        示例:
            site.finish_interfere()  # 结束干涉
        '''
        assert self.is_interfered is True, f"Site {self.code} is not being interfered!"
        self.is_interfered = False
        self.reset()

    def get_avail_transporter(self):
        '''获取当前站点可用的运输车设备
        
        返回:
            transporter: Device对象或None - 可用的运输车对象，如果没有则返回None
        作用:
            检查并返回空闲的运输车资源（资源代码R014）
        注意:
            如果找到R014资源但未找到匹配的设备，可能返回未定义的变量
        示例:
            transporter = site.get_avail_transporter()  # 获取可用运输车
        '''
        # self.update_resources()
        # 检查是否有运输车资源R014可用
        transporter = None
        if "R014" in self.res_avail:
            res = self.resources[self.res_avail["R014"][-1]]
            # 在设备列表中查找对应的空闲运输车
            for device in self.devices:
                if (
                    device.resource.code == res.code
                    and not device.is_busy
                    and not device.is_transporting
                    and getattr(device, 'reserved_for_plane', None) is None
                ):
                    transporter = device
                    break
        return transporter

    def is_avail_job(self, job):
        '''检查作业是否可以在当前站点执行
        
        输入:
            job: Job对象 - 需要检查的作业
        返回:
            bool - 如果站点有所需资源返回True，否则返回False
        示例:
            if site.is_avail_job(job_instance):  # 检查作业可行性
                site.start_jobs([job_instance])
        '''
        # self.update_resources()
        # 遍历作业所需资源类型，检查是否有可用资源
        for res_code in sorted(job.resources):
            if res_code in self.res_avail:
                return True
        return False
    
    def is_all_finished(self):
        '''检查站点所有作业是否已完成
        
        返回:
            bool - 如果没有正在进行的作业返回True，否则返回False
        示例:
            if site.is_all_finished():  # 检查作业完成情况
                site.remove_plane()
        '''
        return len(self.onging_jobs) == 0
    
    def update(self, time):
        '''更新站点状态（时间推进）
        
        输入:
            time: int - 推进的时间（秒）
        返回:
            int - 剩余作业时间（秒），如果无作业返回0；如果处于干涉状态，返回剩余恢复时间
        作用:
            处理干涉恢复、作业进度更新、完成作业判断
        示例:
            left_time = site.update(60)  # 推进1分钟，获取剩余时间
        '''
        ret = 0
        # 如果处于干涉状态，处理恢复倒计时
        if self.is_interfered:
            self.left_rec_time -= time
            assert self.left_rec_time >= 0, "Site recovery time cannot be negative."
            if self.left_rec_time == 0:
                self.finish_interfere()
            return self.left_rec_time 

        # 更新所有进行中的作业剩余时间
        finished_jobs = []
        for job_code in self.onging_jobs:
            self.onging_jobs[job_code][0] -= time
            if self.onging_jobs[job_code][0] <= 0:
                finished_jobs.append(job_code)
            else:
                ret = max(ret, self.onging_jobs[job_code][0])
        # 完成已到期的作业
        if len(finished_jobs) > 0:
            self.finish_jobs([self.config['jobs'][job] for job in finished_jobs])
        # 更新站点总体剩余作业时间
        if len(self.onging_jobs) == 0:
            self.left_job_time = 0
        else:
            self.left_job_time = ret
        return ret
    
    def update_resources(self):
        '''更新该站位的可用资源类型字典
        
        作用:
            检查所有资源状态，按类型分类可用资源
            结果存储在self.res_avail中，格式：{资源类型: [资源代码列表]}
        示例:
            site.update_resources()  # 手动刷新可用资源列表
        '''
        self.res_avail = {}
        for code, res in sorted(self.resources.items()):
            if res.is_available():
                if res.type in self.res_avail:
                    self.res_avail[res.type].append(code)
                else:
                    self.res_avail[res.type] = [code]

        new_onehot = []
        for job in self.target_jobs:
            is_avail = 0
            for res_code in sorted(job.resources):
                if res_code in self.res_avail:
                    is_avail = 1
                    break
            new_onehot.append(is_avail)
        self.avail_job_onehot = new_onehot

    def reset(self):
        '''重置站点到初始状态
        
        作用:
            清空所有运行状态，包括飞机、作业、干涉等
            清空外来设备，恢复初始的资源绑定关系，并更新可用资源字典
        示例:
            site.reset()  # 重置站点
        '''
        # 1. 基础状态清空
        self.plane = None
        self.is_occupied = False
        self.is_interfered = False
        self.left_rec_time = 0
        self.onging_jobs = {}
        self.left_job_time = 0
        
        # 2. 清空当前机位上的设备列表
        # (因为外部环境会调用 Device.reset()，那些原本就属于本站点的设备会自动重新 append 进来)
        self.devices = []
        
        # 3. 恢复初始的资源绑定关系
        # 必须重新初始化 dict，把上个回合跑过来的移动资源剔除，找回自己原本的资源
        self.resources = {}
        for res in self.config['fixed_resources'] + self.config['mobile_resources']:
            # 注意：环境在外部通常会先 reset 设备和资源，所以此时 res.sites 已经是初始状态了
            if self.code in res.sites:
                self.resources[res.code] = res
                
        # 4. 刷新可用资源状态与 one-hot 编码
        self.update_resources()
