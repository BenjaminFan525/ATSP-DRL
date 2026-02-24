class Job:
    # 作业类：定义一个作业任务的基本属性，用于描述飞机在站点需要完成的具体工作任务
    def __init__(self, code: str, time, group, resources: list = [], predecessor: list = [], exclusive: list = []):
        self.code = code  # 作业唯一标识码
        # 作业所需时间（分钟），如果输入是数字则转换为分钟，否则设为None
        self.time = time*60 if isinstance(time, (int, float)) else None
        self.group = group  # 作业所属组别
        self.resources = resources  # 完成作业所需的资源类型代码列表
        self.predecessor = predecessor  # 前置作业列表，表示必须先完成的作业
        self.exclusive = exclusive  # 互斥作业列表，表示不能同时进行的作业