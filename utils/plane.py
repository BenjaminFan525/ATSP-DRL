"""
飞机类
包含:飞机位置,飞机id,静态作业对象列表(每个飞机一样),已完成作业对象列表,剩余作业对象列表,已用时间,进行作业的历史站位id列表

"""
from utils.task import Task
import numpy as np


class Planes:

    def __init__(self, numbers):

        initial_position = np.array([10, 70])   # 初始位置
        self.plane_speed = 5   # 飞机速度

        task = Task()
        # 所有飞机对象
        self.planes_object_list = []
        for i in range(numbers):
            temp_object = Plane(i, task.simple_task_object, initial_position)  # 目前所有飞机的作业都是一样的
            self.planes_object_list.append(temp_object)

    # 计算当前所有飞机的剩余作业数以及总作业数
    def count_jobs(self):
        left_jobs = 0
        all_jobs = 0
        for eve in self.planes_object_list:
            left_jobs += len(eve.left_job)
            all_jobs += len(eve.static_job_list)
        return left_jobs, all_jobs


class Plane:
    # 初始化飞机的变量
    def __init__(self, plane_id, job_object_list, initial_position):
        self.position = initial_position
        self.plane_id = plane_id
        self.static_job_list = job_object_list  # 表示不会变的总作业列表
        self.finished_job = []  # 完成的作业列表
        self.left_job = [eve for eve in job_object_list]   # 剩下的作业对象列表
        self.time_spent = 0  # 花费的时间
        # self.is_idle = False  # true代表正在完成任务，false代表空闲
        self.site_history = []  # 存储每一步完成保障作业所在的战位id

    # 执行一个job， 注意这个job应该在战位中有对应位置，返回该作业的执行时间
    def execute_task(self, job_object, site_object):
        assert job_object.index_id == self.left_job[0].index_id
        time = job_object.time_span
        self.time_spent += time
        self.finished_job.append(job_object)
        self.left_job.pop(0)  # 拿走第一位置的job
        self.position = site_object.absolute_position  # 做完这个job后plane所处的位置
        self.site_history.append(site_object.site_id)
        return time
