import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 1. 设置学术绘图全局格式 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] # 确保通用性
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2

# ================= 2. 数据生成器 =================
def generate_mock_data():
    # 删除了末尾多余的 \n
    algorithms = ['FIFO', 'SPT', 'MWKR', 'IGA\n', 
                  'DRL-G\nsmall', 'DRL-G\nmedium', 'DRL-G\nlarge', 
                  'DRL-S\nsmall', 'DRL-S\nmedium', 'DRL-S\nlarge']
    
    # 修正：数据与注释逻辑对齐 (这里保持你给出的 7000+ 真实值)
    means = [7842.6, 8456.5, 7920.8, 7099.4, 7424.6, 7187.9, 7121.6, 7137.9, 7049.8, 7046.5]
    stds = [371.7, 360.2, 383.6, 274.3, 332.8, 279.9, 272.7, 245.5, 267.4, 262.3]
    
    return algorithms, means, stds

# ================= 3. 核心绘图逻辑 =================
if __name__ == "__main__":
    algorithms, means, stds = generate_mock_data()
    
    # 配色方案
    colors = ['#B0BEC5', '#B0BEC5', '#B0BEC5', '#90CAF9', 
              '#FFB74D', '#FFB74D', '#FFB74D', '#E57373', '#E57373', '#E57373']
    edgecolors = ['#78909C', '#78909C', '#78909C', '#42A5F5', 
                  '#F57C00', '#F57C00', '#F57C00', '#D32F2F', '#D32F2F', '#D32F2F']

    fig, ax = plt.subplots(figsize=(9, 5), dpi=100) # 稍微调宽一点以容纳更多标签

    # 绘制柱状图
    bars = ax.bar(algorithms, means, yerr=stds, 
                  color=colors, edgecolor=edgecolors, linewidth=1.5,
                  capsize=4, ecolor='#424242', alpha=0.9, width=0.7, zorder=3)

    # ================= 4. 美化与排版 =================
    ax.set_ylabel('Makespan ($C_{max}$)', fontsize=12, fontweight='bold')
    
    # --- 重要修正：调整 Y 轴范围 ---
    # 根据数据动态设置，例如最小值减去一点，最大值加上一点
    ymin = 6000 
    ymax = 9500
    ax.set_ylim(ymin, ymax)

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # 在柱子上标注数值
    for bar in bars:
        yval = bar.get_height()
        # 将文字放在柱子顶部稍微靠下的位置
        ax.text(bar.get_x() + bar.get_width()/2, yval - 500, f'{int(yval)}', 
                ha='center', va='bottom', color='white', fontweight='bold', fontsize=9, zorder=4)

    # ================= 5. 保存图片 =================
    save_path = '/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/experiment/figures/generalization_bar.pdf'
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"✅ 图表已成功生成至: {save_path}")