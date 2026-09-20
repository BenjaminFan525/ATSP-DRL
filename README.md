# HKBZ 冻结技术路线

当前工作空间只维护这一条技术链：**Stage1 P5 → Stage2 冻结 B0 → 全共享 H3 R0 E8 → C03 续训 R9 E10**。
Stage3 使用 H3/F4/soft、canonical Hungarian 解码；RL/IGA 比较只读取冻结的 Validation120 参考。

最终选中的是 R9 E10。技术路线锁定不等于研究目标全部通过：当前只有一个 Stage3 训练 seed，独立确认集未开启，IGA 尾部风险目标尚未全部达到。

## 入口与证据

| 内容 | 入口 |
| --- | --- |
| 完整路线、最终 checkpoint、运行参数与限制 | [冻结路线说明](docs/FROZEN_ROUTE.md) |
| 机器可读文件清单与 SHA256 | [冻结路线 manifest](artifacts/frozen_route/20260920/manifest.json) |
| Stage1 三个 P5 来源 | [STAGE1_FINAL.md](STAGE1_FINAL.md) |
| Stage2 B0 交接与冻结约束 | [STAGE2_FROZEN.md](STAGE2_FROZEN.md) |
| 新增 Stage2 H3/F4 三种子 checkpoint 与 Validation120 记录 | [2026-09-21 保存包](artifacts/stage2_checkpoints/20260921_h3f4_validation120/README.md) |
| Stage3 唯一 IGA 比较基线 | [IGA 冻结说明](STAGE3_IGA_FROZEN_20260918.md) |
| 清理范围、保留原因与恢复位置 | [工作空间整理记录](docs/WORKSPACE_RETIREMENT.md) |
| 数据准备 | [数据集说明](docs/DATASETS.md) |

## 只读验证

在仓库根目录执行：

```bash
python scripts/verify_frozen_route.py
python scripts/export_stage2_frozen.py --verify-only
python scripts/verify_stage2_checkpoints.py
python -m pytest -q tests
```

第一条核对最终路线的 checkpoint、manifest、评测参考及冻结源码哈希；第二条核对原 Stage2 交接包。
第三条核对新增 Stage2 保存包的文件哈希、源码快照和原始选模记录，可在新 checkout 中执行。
这些命令不启动训练或 IGA，不把归档结果重新解释为独立复现。

## 保留的实现

- `onpolicy/envs/HKBZ/`：调度环境、数据生成、必要的评测与回归。
- `onpolicy/algorithms/`、`onpolicy/runner/shared/`、`onpolicy/utils/`：冻结路线实际依赖的策略、PPO、恢复和解码实现。
- `onpolicy/scripts/train/train_hkbz.py`：共享模型构建和训练入口；Stage2 冻结门禁继续生效。
- `onpolicy/scripts/train/run_stage3_h3_frozen.py`、`run_stage3_h3_continuation.py`：原注册协议实现及恢复依赖。
- `artifacts/stage2_frozen/20260912/`：保持原始字节的可移植 B0/Stage1/R1 交接包。
- `artifacts/stage2_checkpoints/20260921_h3f4_validation120/`：新增 Stage2 三种子的 Best、Last、E1/E2 原始文件及证据，不替换原 B0。
- `result/`：最终结果、必要的来源链、冻结 IGA 和不可变源码快照；大文件仍不纳入 Git。

历史启动协议中依赖已删除 Tune60 IGA 的新建实验流程不能直接重开；具体边界见冻结路线说明。旧计划中出现的其他实验分支不再是活动开发路线。

原始依赖安装仍使用 `requirements.txt`，开发检查使用 `requirements-dev.txt`。模型加载与 PyTorch 回归需要原有兼容的训练环境；纯文件哈希校验仅依赖 Python 标准库。
