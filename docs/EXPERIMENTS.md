# 实验说明

## 配置

- `onpolicy/config/env.yaml`：数据路径和环境规模；相对路径以仓库根目录解析。
- `onpolicy/config/ac.yaml`：异构图编码器、GRU、actor 和 critic 配置。
- `onpolicy/config/config.py`：MAPPO、并行环境和日志的命令行默认值。

命令行参数优先于 `config.py` 的默认值；环境数据路径当前从 `env.yaml` 读取。

## 输出

训练运行写入：

```text
onpolicy/scripts/results/<env>/<scenario>/<algorithm>/<experiment>/runN/
├── logs/
└── models/checkpoint_EpochN.pt
```

这些文件不应直接提交到 Git。要复现实验，应另行保存完整命令、提交哈希、环境清单、数据清单与随机种子。

## 评估方法

| 脚本 | 方法 | 主要参数 |
|---|---|---|
| `valid_model_greedy.py` | DRL 确定性 greedy | `--checkpoint`, `--dataset-dir` |
| `valid_model_parallel.py` | DRL 多次随机采样取优 | `--checkpoint`, `--dataset-dir`, `--samples`, `--device` |
| `valid_pdrs.py` | FIFO/SPT/MWKR | `--dataset-dir`, `--seed` |
| `valid_iga.py` | 单目标遗传算法 | `--time-limit`, `--population-size`, `--generations` |
| `valid_nsga.py` | NSGA-II | `--population-size`, `--generations` |
| `gurobi.py` | 简化 MIP 基线 | case 目录、`--time-limit` |

IGA 的 `--time-limit` 是每个算例的挂钟预算。为保证公平比较，应同时报告 population、generation、seed、CPU 型号、并行设置和实际耗时。Gurobi 脚本建模的是简化版本，结果不能未经核对就当作完整环境的严格最优解。

## 历史参考值

`valid_iga.py`、`valid_pdrs.py` 和两个 DRL 评估脚本中的 `KNOWN_OPTIMAL_CMAX` 来自该提交历史 `test_large` 的 20 个算例。它不是按 case 名称通用的 oracle：若数据被重新生成，即使仍叫 `case_01`，参考值也已失效。此时应只使用原始 Cmax，或先替换为与新数据集绑定且可追溯的参考文件。

## 绘图

```bash
python onpolicy/envs/HKBZ/experiment/plot_training.py
python onpolicy/envs/HKBZ/experiment/plot_generalization.py
python onpolicy/envs/HKBZ/experiment/plot_gantt.py
```

`experiment/results/` 和 `experiment/figures/` 是提交 `7721cd6` 已包含的示例产物。`reward_large.pdf` 没有随提交提供对应 CSV，因此目前不能由仓库内容完整重建；发布论文复现包时应补充源数据或移除该图。
