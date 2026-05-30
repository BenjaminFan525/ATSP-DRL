import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= 1. 设置参照图风格 =================
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] # 删掉后面的 fallback
plt.rcParams['mathtext.fontset'] = 'stix'        # 让坐标轴里的负号和公式也变成 Times 风格
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 0.5

# ================= 2. 最早的平滑逻辑 =================
def smooth_curve(scalars, weight=0.85):
    """最早的 EMA 平滑算法"""
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# ================= 3. 单图绘制函数 =================
def generate_single_styled_pdf(csv_path, save_path, ylabel, color, origin_curve=True):
    if not os.path.exists(csv_path):
        print(f"⚠️ 找不到文件: {csv_path}，跳过生成...")
        return

    # 读取并处理数据 (保留原始 Step 数值，不除以 1000，交给科学计数法处理)
    df = pd.read_csv(csv_path)
    episodes = df['Step']
    values = df['Value'].tolist()
    smoothed_values = smooth_curve(values, weight=0.85) # 可微调 weight 改变平滑度
    
    # 创建适合作为单张小图的画布 (尺寸适当放大保证清晰度)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=600)
    
    # 绘制曲线 (蓝底橙线)
    if origin_curve:
        ax.plot(episodes, values, color=color, alpha=0.2, linewidth=1.0)
        ax.plot(episodes, smoothed_values, color=color, alpha=0.9, linewidth=2.0)
    else:
        ax.plot(episodes, smoothed_values, color=color, alpha=0.9, linewidth=2.0)
    
    # 坐标轴标签 (去除 fontweight='bold')
    ax.set_xlabel('Episodes', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    
    # 强制使用科学计数法 (MathText格式，形如 1 x 10^5)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # 全封闭边框与实线网格
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.grid(True, linestyle='-', color='#d3d3d3', linewidth=0.5)
    
    # 图例设置
    # ax.legend(loc='upper right', frameon=True, fontsize=11)
    
    # 紧凑布局并保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 成功生成: {save_path}")

# ================= 4. 批量执行 =================
if __name__ == "__main__":
    # 任务列表：(输入CSV, 输出PDF, Y轴标签)
    # 不在 Python 里加 Title，Title 由 LaTeX \centerline 提供

    c_reward = '#1f77b4'  # 蓝色系
    c_makespan = '#d62728' # 红色系

    tasks = [
        ('/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/results/rewards_medium.csv', '/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/figures/reward_small.pdf', 'Reward', c_reward, False),
        ('/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/results/rewards_medium.csv', '/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/figures/reward_medium.pdf', 'Reward', c_reward, False),
        ('/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/results/rewards_medium.csv', '/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/figures/reward_large.pdf', 'Reward', c_reward, False),
        
        ('/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/results/makespan_small.csv', '/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/figures/makespan_small.pdf', 'Makespan', c_makespan, True),
        ('/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/results/makespan_medium.csv', '/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/figures/makespan_medium.pdf', 'Makespan', c_makespan, True),
        ('/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/results/makespan_large.csv', '/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/figures/makespan_large.pdf', 'Makespan', c_makespan, True)
    ]
    
    print("🚀 开始批量生成独立学术图表...")
    for csv_in, pdf_out, ylabel, color, origin_curve in tasks:
        generate_single_styled_pdf(csv_in, pdf_out, ylabel, color, origin_curve)
    print("🎉 全部处理完毕！")