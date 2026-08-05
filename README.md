# HKBZ Scheduling Environment

面向飞机地面保障协同调度的研究代码。当前主线使用异构图神经网络编码作业、机位和设备状态，以 GNN-MAPPO 学习联合的“作业—机位”决策，并提供 IGA、NSGA-II、优先派工规则和 Gurobi 基线。

本分支由历史提交 `7721cd69a189593ff712de05accf8f4280738239` 整理而来，仅保留 HKBZ 研究主线。数据集、模型权重和训练日志不纳入 Git，需单独生成或分发。

## 代码结构

```text
onpolicy/
├── algorithms/gnn_mappo/       # MAPPO policy and trainer
├── algorithms/utils/           # heterogeneous GNN, GRU, actor and critic
├── config/                     # model/environment defaults and one example case
├── envs/HKBZ/
│   ├── core/                   # domain entities
│   ├── environment.py          # Gymnasium scheduling environment
│   ├── data_generator.py       # reproducible synthetic case generator
│   └── experiment/             # DRL and baseline evaluation, plotting
├── runner/shared/hkbz_runner.py
└── scripts/train/train_hkbz.py # training entry point
```

## 安装

建议使用 Python 3.10 或 3.11，并根据本机 CUDA 版本先安装匹配的 PyTorch 与 PyTorch Geometric，再安装其余依赖：

```bash
conda create -n hkbz python=3.11 -y
conda activate hkbz

# 按 PyTorch/PyG 官方说明安装与 CUDA 匹配的版本后：
pip install -r requirements.txt
```

Gurobi 精确基线是可选项，需要有效的 Gurobi 安装与许可证：

```bash
pip install -r requirements-optional.txt
```

## 准备数据

每个算例目录包含 `job.json`、`fixed_resources.json`、`mobile_resources.json`、`sites.json` 和 `flights.json`。例如：

```bash
python -m onpolicy.envs.HKBZ.data_generator \
  --output onpolicy/envs/HKBZ/dataset/train_large \
  --num-cases 100 --num-stands 40 --num-planes 24 \
  --seed 42 --no-layouts

python -m onpolicy.envs.HKBZ.data_generator \
  --output onpolicy/envs/HKBZ/dataset/test_large \
  --num-cases 20 --num-stands 40 --num-planes 24 \
  --seed 1042 --no-layouts
```

随后检查 [环境配置](onpolicy/config/env.yaml) 中的数据集路径。相对路径统一以仓库根目录为基准。数据格式和划分建议见 [docs/DATASETS.md](docs/DATASETS.md)。

## 训练

下面的线程数必须分别整除训练集和测试集的算例数：

```bash
python onpolicy/scripts/train/train_hkbz.py \
  --experiment_name hkbz-large \
  --n_rollout_threads 20 \
  --n_eval_rollout_threads 10 \
  --num_episodes 200 \
  --use_eval
```

默认在 `onpolicy/scripts/results/` 下写入 TensorBoard 日志和 checkpoint；该目录已被 Git 忽略。添加 `--use_wandb` 可改用 Weights & Biases。

## 评估

```bash
# DRL greedy policy
python onpolicy/envs/HKBZ/experiment/valid_model_greedy.py \
  --checkpoint /path/to/checkpoint.pt \
  --dataset-dir onpolicy/envs/HKBZ/dataset/test_large

# Stochastic parallel rollouts
python onpolicy/envs/HKBZ/experiment/valid_model_parallel.py \
  --checkpoint /path/to/checkpoint.pt \
  --dataset-dir onpolicy/envs/HKBZ/dataset/test_large \
  --samples 50 --device cuda:0

# Baselines
python onpolicy/envs/HKBZ/experiment/valid_pdrs.py --dataset-dir DATASET
python onpolicy/envs/HKBZ/experiment/valid_iga.py --dataset-dir DATASET --time-limit 180
python onpolicy/envs/HKBZ/experiment/valid_nsga.py --dataset-dir DATASET
```

评估脚本、内置参考值与绘图产物的边界说明见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 复现边界

- 仓库不包含私有/大体积数据集、teacher 标签、checkpoint 和运行日志。
- `experiment/results/` 与 `experiment/figures/` 是该提交已有的示例论文产物，不等同于对任意新数据集的基准保证。
- 当前评估脚本中的 `KNOWN_OPTIMAL_CMAX` 只对应历史 `test_large` 的 20 个同名算例；更换数据集时不要直接解释其中的 Gap。
- 随机性来自环境、策略采样和进化算法；正式报告应固定种子并进行多次独立运行。

## 来源与引用

仓库历史起源于 MASA-QMIX 研究代码。若使用其思想或历史实现，请引用：

> Wang, X., Zhang, L., Lin, T., Zhao, C., Wang, K., & Chen, Z. (2022). Solving job scheduling problems in a resource preemption environment with multi-agent reinforcement learning. *Robotics and Computer-Integrated Manufacturing, 77*, 102324.

当前仓库尚未声明开源许可证。公开发布前必须确认代码、数据、模型和图表的权利边界并选择许可证，详见 [发布检查清单](docs/RELEASE_CHECKLIST.md)。
