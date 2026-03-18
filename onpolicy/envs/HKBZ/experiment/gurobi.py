import gurobipy as gp
from gurobipy import GRB
import json
import math
import os

def solve_atsp_hrc(case_dir, time_limit=1800):
    print(f"🚀 开始使用 Gurobi 求解算例: {case_dir}")
    
    # ==========================================
    # 1. 数据加载与预处理 (对应你的 environment.py 初始化)
    # ==========================================
    with open(os.path.join(case_dir, 'flights.json'), 'r', encoding='utf-8') as f:
        flights = json.load(f)
    with open(os.path.join(case_dir, 'job.json'), 'r', encoding='utf-8') as f:
        jobs_data = json.load(f)
    with open(os.path.join(case_dir, 'sites.json'), 'r', encoding='utf-8') as f:
        sites_data = json.load(f)
    with open(os.path.join(case_dir, 'fixed_resources.json'), 'r', encoding='utf-8') as f:
        fixed_res = json.load(f)
    with open(os.path.join(case_dir, 'mobile_resources.json'), 'r', encoding='utf-8') as f:
        mobile_res = json.load(f)

    # 提取集合
    P = [f["飞机编号"] for f in flights]  # 飞机集合 M
    J = [j["作业编号"] for j in jobs_data if j["分组"] == '保障'] # 工序集合 (简化为保障作业)
    L = sites_data['sites_codes']        # 机位集合 N
    R = [r["设备编号"] for r in fixed_res + mobile_res] # 资源集合
    
    # 提取参数
    Arrival = {f["飞机编号"]: f["到达时间"] for f in flights}
    
    # 整理工序属性
    JobTime = {}
    JobReqs = {}
    JobPred = {}
    JobExcl = {}
    for j in jobs_data:
        j_code = j["作业编号"]
        JobTime[j_code] = j.get("作业时间", 1) * 60 if isinstance(j.get("作业时间"), (int, float)) else 300 # 估算动态时间
        JobReqs[j_code] = j.get("需要设备类型", [])
        JobPred[j_code] = j.get("前置作业", [])
        JobExcl[j_code] = j.get("互斥作业", [])

    # 计算距离矩阵 (曼哈顿距离)
    Dist = {}
    positions = sites_data['sites_positions']
    for i, code1 in enumerate(L):
        for j, code2 in enumerate(L):
            Dist[(code1, code2)] = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])

    BIG_M = 100000  # 极大值，用于打破线性约束的“非此即彼”

    # ==========================================
    # 2. 建立 Gurobi 模型
    # ==========================================
    m = gp.Model("ATSP_HRC_Exact")
    m.setParam('TimeLimit', time_limit) # 设置 1800 秒超时
    m.setParam('MIPGap', 0.0)           # 追求 0% Gap 的绝对最优解

    # ==========================================
    # 3. 定义决策变量
    # ==========================================
    # 连续变量：开始时间和完成时间
    S = m.addVars(P, J, vtype=GRB.CONTINUOUS, name="Start")
    C = m.addVars(P, J, vtype=GRB.CONTINUOUS, name="Completion")
    
    # 0-1 变量：机位分配 (飞机 p 的作业 j 是否在机位 l 执行)
    X = m.addVars(P, J, L, vtype=GRB.BINARY, name="SiteAssign")
    
    # 0-1 变量：资源分配 (飞机 p 的作业 j 是否使用了资源 r)
    Y = m.addVars(P, J, R, vtype=GRB.BINARY, name="ResAssign")
    
    # 0-1 排序变量 (用于防止同一个资源或机位上的时间重叠)
    # Z_site[p1, j1, p2, j2] = 1 表示飞机1的作业1 排在 飞机2的作业2 之前
    Z_site = m.addVars(P, J, P, J, vtype=GRB.BINARY, name="OrderSite")
    Z_res  = m.addVars(P, J, P, J, vtype=GRB.BINARY, name="OrderRes")
    
    # ==========================================
    # 4. 构建约束条件 (Constraints)
    # ==========================================
    
    for p in P:
        for j in J:
            # (1) 基础时间约束：完成时间 = 开始时间 + 处理时间
            m.addConstr(C[p, j] >= S[p, j] + JobTime[j], name=f"Time_{p}_{j}")
            
            # (2) 到达时间约束：所有作业的开始时间必须大于飞机的到达时间
            m.addConstr(S[p, j] >= Arrival[p], name=f"Arr_{p}_{j}")
            
            # (3) 拓扑顺序约束 (Precedence)：前置作业必须先做完
            for pred in JobPred[j]:
                if pred in J:
                    m.addConstr(S[p, j] >= C[p, pred], name=f"Pred_{p}_{pred}_{j}")
            
            # (4) 互斥作业约束 (Mutually Exclusive)：如同你图2里的 ZY04 和 ZY10
            # 引入辅助变量 w 来决定谁先谁后
            for exc in JobExcl[j]:
                if exc in J and j < exc: # 防止重复添加
                    w = m.addVar(vtype=GRB.BINARY, name=f"ExcW_{p}_{j}_{exc}")
                    m.addConstr(S[p, j] >= C[p, exc] - BIG_M * w)
                    m.addConstr(S[p, exc] >= C[p, j] - BIG_M * (1 - w))
            
            # (5) 场地分配约束：每个作业只能在一个场地执行
            m.addConstr(gp.quicksum(X[p, j, l] for l in L) == 1, name=f"SiteOne_{p}_{j}")

    # (6) 场地容量约束 (No-Overlap on Sites)：
    # 同一个机位 l 上，两架不同飞机的作业不能时间重叠
    for p1 in P:
        for p2 in P:
            if p1 < p2:
                for j1 in J:
                    for j2 in J:
                        for l in L:
                            # 如果 p1和p2都在 l 执行，则必须依靠 Z_site 排序
                            m.addConstr(S[p1, j1] >= C[p2, j2] - BIG_M * (3 - X[p1, j1, l] - X[p2, j2, l] - Z_site[p1, j1, p2, j2]))
                            m.addConstr(S[p2, j2] >= C[p1, j1] - BIG_M * (2 - X[p1, j1, l] - X[p2, j2, l] + Z_site[p1, j1, p2, j2]))

    # (7) 资源独占约束 (No-Overlap on Resources)：
    # 同一个资源 r 在同一时间只能服务一个作业
    for p1 in P:
        for p2 in P:
            if p1 < p2:
                for j1 in J:
                    for j2 in J:
                        for r in R:
                            m.addConstr(S[p1, j1] >= C[p2, j2] - BIG_M * (3 - Y[p1, j1, r] - Y[p2, j2, r] - Z_res[p1, j1, p2, j2]))
                            m.addConstr(S[p2, j2] >= C[p1, j1] - BIG_M * (2 - Y[p1, j1, r] - Y[p2, j2, r] + Z_res[p1, j1, p2, j2]))

    # ==========================================
    # 5. 定义目标函数 (Objective)
    # ==========================================
    # 引入辅助变量 C_max 表示全局最晚完工时间 (Makespan)
    C_max = m.addVar(vtype=GRB.CONTINUOUS, name="Makespan")
    for p in P:
        for j in J:
            m.addConstr(C_max >= C[p, j])
            
    # 目标：最小化 Makespan
    m.setObjective(C_max, GRB.MINIMIZE)

    # ==========================================
    # 6. 开始求解
    # ==========================================
    m.optimize()

    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        print(f"\n✅ 求解结束! 状态码: {m.status}")
        print(f"🏅 最优 Makespan: {m.objVal}")
        print(f"⏱️ 求解耗时: {m.Runtime:.2f} 秒")
        print(f"📉 MIP Gap: {m.MIPGap * 100:.2f}%")
        return m.objVal, m.Runtime
    else:
        print("❌ 未能找到可行解。可能约束冲突或问题规模太大。")
        return None, None

if __name__ == "__main__":
    # 指向你的生成算例目录
    test_case_path = "/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/dataset/test/case_01" 
    solve_atsp_hrc(test_case_path)