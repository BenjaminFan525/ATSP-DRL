import math
import numpy as np

# 飞机类：定义飞机的各项属性和行为，用于管理飞机的作业流程、移动和状态转换
class Plane:
    LONG_OCCUPANCY_JOBS = frozenset({'ZY02', 'ZY03'})
    TRANSFER_JOB_CODE = 'ZY-T'
    DEPARTURE_JOB_GROUP = '出场'

    def __init__(self, code, config):
        self.code = code  # 飞机唯一标识码
        self.config = config  # 飞机配置字典
        self.velocity = config['velocity']  # 飞机移动速度
        self.site = config['site']  # 飞机当前所在站点
        self.choosed_job = None  # 飞机选择的下一个作业（未执行）
        self.current_jobs = []  # 当前正在执行的作业列表
        self.finished_jobs = []  # 已完成的作业列表
        self.job_time = 0
        self.total_job_time = 0
        self.is_busy = False  # 飞机是否处于作业状态
        self.destination = None  # 飞机移动目的地
        self.transporter = None  # 协助运输的转运车设备
        self.is_transporting = False  # 飞机是否处于运输状态
        self.left_trans_time = 0  # 当前运输剩余时间（秒）
        self.trans_time = 0
        self.is_waiting = False  # 飞机是否处于等待状态
        self.waiting_time = 0  # 已等待时间
        self.last_site_idx = -1  # 记录刚才选的机位
        self.last_job_idx = -1    # 记录刚才选的工序 (如果是对象，取其 code)
        self.pending_job = None
        # ZY02/ZY03 can be reset by relocation, so they are not irreversible
        # progress. These counters make repeated relocation observable.
        self.ever_finished_jobs = set()
        self.relocations_since_progress = 0
        self.total_relocations = 0
        self.no_progress_decisions = 0
        self.last_departed_site_code = None
        self.last_transport_reset_jobs = ()
        self.transport_purpose = None
        self.pending_transport_purpose = None
        self.departure_staging_decided = False
        self.takeoff_site_codes = frozenset(
            str(code) for code in config.get('takeoff_site_codes', ())
        )
        self.decision_signature_counts = {}

        # 初始化飞机作业字典，过滤和配置作业参数
        self.jobs = {}
        for job in config['jobs']:
            # Plane jobs contain persistent service/departure operations. ZY-T
            # is a repeatable scheduling action managed by the environment and
            # therefore deliberately does not enter ``left_jobs``.
            if (
                (job.group == '保障' and job.code not in ['ZY01', 'ZY-L'])
                or job.group == self.DEPARTURE_JOB_GROUP
            ):
                self.jobs[job.code] = job
            # 为特定作业设置计算时间
            if job.time is None:
                if job.code == 'ZY10':
                    job.time = math.ceil((100-self.config['fuel'])/5*60)
            # 将前驱和资源转换为集合，便于集合运算
            job.predecessor = set(job.predecessor)
            job.resources = set(job.resources)
        self.service_job_codes = frozenset(
            code for code, job in self.jobs.items() if job.group == '保障'
        )
        self.departure_job_codes = tuple(
            code for code, job in self.jobs.items()
            if job.group == self.DEPARTURE_JOB_GROUP
        )
        self.left_jobs = list(self.jobs.keys())  # 初始剩余作业列表
        self.current_avail_jobs = self.get_avail_jobs(self.site)

    def get_avail_jobs(self, site = None):
        '''获取当前站点可用的作业列表
        
        输入:
            site: Site对象或None - 指定站点，默认为当前站点
        返回:
            list - 可用作业代码列表
        逻辑:
            1. 前驱作业必须全部完成
            2. 站点必须有所需资源类型（如果提供站点参数）
        示例:
            avail_jobs = plane.get_avail_jobs(current_site)  # 获取可用作业
        '''
        # assert not self.is_busy and not self.is_transporting, \
        #     "Current plane must be idle to get available jobs."
        avail = []                               # 存放所有可选作业编码
        for job_code in self.left_jobs:          # 遍历剩余作业
            job = self.jobs[job_code]            # 当前作业对象
            # 1. 前序作业必须全部完工
            if job.predecessor.issubset(self.finished_jobs):
                # 2. 资源需求检查
                if site:
                    site.update_resources()
                    if len(job.resources) == 0 or not job.resources.isdisjoint(site.res_avail.keys()):
                        avail.append(job_code)
                else:
                    avail.append(job_code)

        return avail

    def get_parallel_jobs(self, job_code):
        '''获取与指定作业可并行执行的作业列表
        
        输入:
            job_code: str - 目标作业代码
        返回:
            list - 可并行执行的作业代码列表
        逻辑:
            包含所选作业本身，以及不互斥且作业时间不大于所选作业的作业
        示例:
            parallel = plane.get_parallel_jobs('ZY05')  # 获取可并行作业
        '''
        parallel_jobs = []
        # Reserve the selected anchor before admitting other jobs. Keep the
        # original candidate/output order, including the anchor's position.
        admitted = [job_code]
        for job in self.get_avail_jobs(self.site):
            if job == job_code:
                parallel_jobs.append(job)
            elif self.jobs[job_code].time >= self.jobs[job].time and all(
                other not in self.jobs[job].exclusive
                and job not in self.jobs[other].exclusive
                for other in admitted
            ):
                parallel_jobs.append(job)
                admitted.append(job)
        return parallel_jobs
    
    def choose_job(self, job_code):
        '''选择并启动作业
        
        输入:
            job_code: str - 要启动的作业代码
        返回:
            int - 作业执行时间（秒）
        作用:
            验证作业可行性，启动并行作业组，更新飞机和站点状态
        异常:
            AssertionError - 当作业不可用时触发
        示例:
            exec_time = plane.choose_job('ZY05')  # 选择并启动作业
        '''
        assert job_code in self.get_avail_jobs(self.site), f"Job {job_code} is not available."
        self.current_jobs = self.get_parallel_jobs(job_code)
        self.left_jobs = [job for job in self.left_jobs if job not in self.current_jobs]
        self.is_busy = True
        self.site.start_jobs([self.jobs[job] for job in self.current_jobs])
        # print(f"Plane {self.code} starts job {job_code} at site {self.site.code}.")
        self.total_job_time = sum([self.jobs[job].time for job in self.current_jobs])
        self.job_time = self.jobs[job_code].time
        return self.jobs[job_code].time

    def start_transport(self, destination, transporter, purpose=None):
        '''启动飞机运输任务
        
        输入:
            destination: Site对象 - 目的地站点
            transporter: Device对象或None - 协助运输的转运车（可为空）
        返回:
            int - 运输所需时间（秒，包含准备时间）
        作用:
            计算运输时间，更新运输状态，协调转运车任务
        注意:
            根据站点类型添加不同的准备时间（起飞前准备、固定/解固定等）
        示例:
            trans_time = plane.start_transport(target_site, transporter_device)  # 启动运输
        '''
        # assert self.is_busy is False and self.is_transporting is False, "Plane must be idle to start transporting."
        if destination.code in self.takeoff_site_codes and transporter is None:
            raise RuntimeError(
                f"Plane {self.code} cannot enter takeoff site "
                f"{destination.code} without an R014 transporter."
            )
        self.last_departed_site_code = self.site.code
        self.transport_purpose = purpose
        self.pending_transport_purpose = None
        self.last_transport_reset_jobs = tuple(
            sorted(self.LONG_OCCUPANCY_JOBS.intersection(self.finished_jobs))
        )
        self.relocations_since_progress += 1
        self.total_relocations += 1
        self.site.remove_plane()
        # 计算曼哈顿距离和基础运输时间
        distance = abs(self.site.pos[0]-destination.pos[0]) + abs(self.site.pos[1]-destination.pos[1])
        time = math.ceil(distance / self.velocity)
        # 根据起止站点类型添加固定准备时间
        if self.site.code == 'Z':
            time += 1*60  # 加上固定的时间
        elif destination.code in self.takeoff_site_codes:
            # Only release/unfix is part of the transport. ZY-S and ZY-F are
            # explicit operations and account for the remaining four minutes.
            time += 1*60
        else:
            time += 2*60  # 解固+固定
        self.is_transporting = True
        self.left_trans_time = time
        # 协调转运车任务
        if transporter is not None:
            self.transporter = transporter
            transporter.start_transport(destination)
            # 转运车准备时间比飞机短
            if self.site.code == 'Z':
                transporter.left_trans_time = time - 1*60
            elif destination.code in self.takeoff_site_codes:
                transporter.left_trans_time = time
            else:
                transporter.left_trans_time = time - 2*60
        
        self.site = destination
        if self.site.plane != self:
            self.site.add_plane(self)

        # print(f"Plane {self.code} starts transporting to {destination.code} with transporter {transporter.code if transporter else 'None'}.")

        if transporter is not None:
            self.trans_time = transporter.left_trans_time
            return transporter.left_trans_time
        else:
            self.trans_time = time
            return time
    
    def finish_transport(self):
        '''完成飞机运输任务
        
        作用:
            重置运输状态，处理转运车关系，重置长占作业
        异常:
            AssertionError - 当飞机不在运输状态时触发
        示例:
            plane.finish_transport()  # 完成运输
        '''
        assert self.is_transporting is True, "Plane must be transporting to finish transport."
        if self.transporter:
            # self.transporter.finish_transport()
            self.transporter = None
        preserve_service = self.transport_purpose in {
            'post_service_relocation',
            'departure',
        }
        # Mid-service relocation invalidates long-occupancy work. Once all
        # service work is complete, staging/departure transport must not undo
        # operations that have already been accepted as complete.
        if not preserve_service and not self.is_completed_all_jobs():
            self.finished_jobs = [
                item for item in self.finished_jobs
                if item not in self.LONG_OCCUPANCY_JOBS
            ]
            for job_code in sorted(self.LONG_OCCUPANCY_JOBS):
                if job_code not in self.left_jobs:
                    self.left_jobs.append(job_code)
        self.is_transporting = False
        self.transport_purpose = None

    @property
    def irreversible_progress_count(self):
        """Number of first-time completions that relocation cannot reset."""
        return len(self.ever_finished_jobs.difference(self.LONG_OCCUPANCY_JOBS))

    def start_waiting(self, pending_job=None, transport_purpose=None):
        '''启动飞机等待状态
        
        作用:
            设置等待标志，重置等待时间
        异常:
            AssertionError - 当飞机不在空闲状态时触发
        示例:
            plane.start_waiting()  # 开始等待
        '''
        assert self.is_busy is False and self.is_transporting is False, "Plane must be idle to start waiting."
        self.is_waiting = True
        self.pending_job = pending_job
        self.pending_transport_purpose = transport_purpose
        self.waiting_time = 0

    def finish_waiting(self):
        '''结束飞机等待状态
        
        作用:
            清除等待标志和时间
        异常:
            AssertionError - 当飞机不在等待状态时触发
        示例:
            plane.finish_waiting()  # 结束等待
        '''
        assert self.is_waiting is True, "Plane must be waiting to finish waiting."
        self.is_waiting = False
        self.pending_job = None
        self.pending_transport_purpose = None
        self.waiting_time = 0

    def has_completed_service_jobs(self):
        """Return whether every selected protection operation is complete."""
        unfinished = set(self.left_jobs).union(self.current_jobs)
        return self.service_job_codes.isdisjoint(unfinished)

    def has_started_departure(self):
        """Return whether at least one explicit departure job has started."""
        return any(
            code in self.finished_jobs or code in self.current_jobs
            for code in self.departure_job_codes
        )

    def has_completed_departure(self):
        """Return whether the complete explicit departure sequence finished."""
        return bool(self.departure_job_codes) and set(
            self.departure_job_codes
        ).issubset(self.finished_jobs)

    def mark_departure_staging_decided(self):
        """Allow at most one post-service stand-selection decision."""
        self.departure_staging_decided = True

    def is_completed_all_jobs(self):
        '''检查飞机是否完成所有作业
        
        返回:
            bool - 如果没有剩余作业返回True，否则返回False
        示例:
            if plane.is_completed_all_jobs():  # 检查作业完成状态
                print("所有作业完成")
        '''
        return (len(self.left_jobs) == 0) and (len(self.current_jobs) == 0)

    def is_idle(self):
        '''检查飞机是否处于空闲状态
        
        返回:
            bool - 如果飞机不忙、不在运输中、不在等待中返回True
        示例:
            if plane.is_idle():  # 检查飞机状态
                plane.start_waiting()
        '''
        return not self.is_busy and not self.is_transporting and not self.is_waiting

    def _next_unfinished_predecessor(self, job_code):
        """Return a ready-frontier predecessor that must run before ``job_code``.

        Relocation resets the long-occupancy jobs ZY02/ZY03. A previously
        selected downstream job can therefore become unavailable on arrival.
        Walking the dependency graph prevents the plane from waiting forever
        on that downstream job while one of its predecessors is still undone.
        """
        visiting = set()

        def visit(code):
            if code in visiting or code not in self.jobs:
                return None
            visiting.add(code)
            for predecessor in sorted(self.jobs[code].predecessor):
                if predecessor in self.finished_jobs:
                    continue
                nested = visit(predecessor)
                if nested is not None:
                    return nested
                if predecessor in self.left_jobs:
                    return predecessor
            return None

        return visit(job_code)

    def _resume_selected_job(self):
        """Resume the selected job, automatically restoring prerequisites."""
        if not self.choosed_job:
            return np.inf

        avail_jobs = self.get_avail_jobs(self.site)
        if self.choosed_job in avail_jobs:
            ret = self.choose_job(self.choosed_job)
            if self.choosed_job in self.current_jobs:
                self.choosed_job = None
            return ret

        prerequisite = self._next_unfinished_predecessor(self.choosed_job)
        pending_job = prerequisite or self.choosed_job
        if pending_job in avail_jobs:
            return self.choose_job(pending_job)

        self.start_waiting(pending_job)
        return np.inf
    
    def update(self, time):
        '''更新飞机状态（时间推进）
        
        输入:
            time: int - 推进的时间（秒）
        返回:
            int/float - 剩余时间（作业、运输或等待相关时间）
        作用:
            处理等待、运输、作业三种状态的推进和状态转换
            自动处理等待结束后的作业启动或运输启动
        异常:
            AssertionError - 当飞机处于无效状态（同时忙碌和运输）时触发
        示例:
            left_time = plane.update(60)  # 推进1分钟
        '''
        assert not(self.is_busy and self.is_transporting), "Plane must be either busy or transporting."
        ret = np.inf
        # 等待状态更新
        if self.is_waiting:
            self.waiting_time += time
            chosen_job = None
            # 检查是否可以结束等待（有目的地且有可用运输车）
            if self.destination and self.destination != self.site:
                if self.site.get_avail_transporter() is not None:
                    transport_purpose = self.pending_transport_purpose
                    self.finish_waiting()
                    self.transporter = self.site.get_avail_transporter()
                    ret = self.start_transport(
                        self.destination,
                        self.transporter,
                        purpose=transport_purpose,
                    )
                    self.destination = None
            # 检查是否可以结束等待（等待的资源已到位，或原选择作业已可用）
            else:
                avail_jobs = self.get_avail_jobs(self.site)
                if self.pending_job and self.pending_job in avail_jobs:
                    chosen_job = self.pending_job
                elif not self.pending_job and self.choosed_job and self.choosed_job in avail_jobs:
                    chosen_job = self.choosed_job

            if self.is_waiting and chosen_job is not None:
                self.finish_waiting()
                ret = self.choose_job(chosen_job)
                if self.choosed_job in self.current_jobs:
                    self.choosed_job = None
                    
        # 运输状态更新
        elif self.is_transporting:
            self.left_trans_time -= time
            assert self.left_trans_time >= 0, "Plane transport time cannot be negative."
            if self.left_trans_time <= 0:
                self.finish_transport()
                self.current_avail_jobs = self.get_avail_jobs(self.site)
                # 运输完成后，如果有预选的作业则立即启动
                if self.choosed_job:
                    ret = self._resume_selected_job()
            else:
                ret = self.left_trans_time

        # 作业状态更新
        elif self.is_busy:
            # 通过站点更新推进作业进度
            ret = self.site.update(time)
            if self.site.is_all_finished():
                completed_jobs = set(self.current_jobs)
                new_first_completions = completed_jobs.difference(self.ever_finished_jobs)
                self.ever_finished_jobs.update(completed_jobs)
                self.finished_jobs += self.current_jobs
                self.current_jobs = []
                self.is_busy = False
                if new_first_completions.difference(self.LONG_OCCUPANCY_JOBS):
                    self.relocations_since_progress = 0
                    self.no_progress_decisions = 0
                    self.decision_signature_counts.clear()
                self.current_avail_jobs = self.get_avail_jobs(self.site)
                # 作业完成后，如果有预选的作业则立即启动
                if self.choosed_job:
                    ret = self._resume_selected_job()
        return ret

    def reset(self):
        '''重置飞机状态到初始配置
        
        作用:
            恢复所有状态参数，重置作业列表，清空临时状态，并修复站点拓扑关系。
        示例:
            plane.reset()  # 重置飞机
        '''
        # 1. 拓扑关系复位：如果飞机当前不在初始站点，将其从当前站点的记录中移除
        if self.site != self.config['site']:
            if getattr(self.site, 'plane', None) == self:
                self.site.remove_plane()
                
        # 2. 恢复初始基础配置
        self.velocity = self.config['velocity']  # 飞机速度
        self.site = self.config['site']  # 位置
        
        # 确保飞机被正确注册到了初始站点上
        if getattr(self.site, 'plane', None) != self:
            self.site.add_plane(self)

        # 3. 清除引用目标与队列
        self.choosed_job = None
        self.destination = None
        self.transporter = None
        self.transport_purpose = None
        self.pending_transport_purpose = None
        self.departure_staging_decided = False
        self.current_jobs = []
        self.finished_jobs = []
        
        # 4. 重置所有的状态标志位和时间倒计时
        self.is_busy = False
        self.is_transporting = False
        self.left_trans_time = 0
        self.is_waiting = False
        self.waiting_time = 0
        self.job_time = 0
        self.trans_time = 0
        self.total_job_time = 0
        
        # 5. 强化学习 (RL) 动作记忆锚点复位 (极其重要，用于处理冷启动)
        self.last_site_idx = -1
        self.last_job_idx = -1
        self.ever_finished_jobs = set()
        self.relocations_since_progress = 0
        self.total_relocations = 0
        self.no_progress_decisions = 0
        self.last_departed_site_code = None
        self.last_transport_reset_jobs = ()
        self.decision_signature_counts = {}
        
        # 6. 重置作业清单
        self.left_jobs = list(self.jobs.keys())
        self.current_avail_jobs = self.get_avail_jobs(self.site)
