# Stage2 冻结与 Stage3 交接（2026-09-12）

Stage2 开发已冻结，不再自动启动 BC、RL、成本标签或 H/F 搜索，也不自动启动 Stage3。
本版本保留可验证的共享运行时、严格交接入口、原始 checkpoint 和历史结果。
冻结是研究工作安排，不是“完全超过 IGA180、达到/超过 IGA1800”的验收通过。

## 唯一冻结交接源

入口：[manifest.json](artifacts/stage2_frozen/20260912/manifest.json)。
协议为 `stage2_frozen_b0_v1`，`scientific_goal_confirmed=false`。

| 文件 | 用途 | SHA256 前缀 |
| --- | --- | --- |
| `checkpoints/b0.pt` | B0 BC epoch 1、seed 11，Stage3 完整模型起点 | `b7740b2b688c83dc` |
| `checkpoints/stage1.pt` | Stage1 P5 seed 3，保护参数与来源核验 | `8ce0244f9b0b3877` |
| `checkpoints/r1.pt` | R1 预测头，严格冻结来源核验 | `c0c4825946e469b6` |

三个文件均保持原始字节，未重存张量或伪造 gate。manifest 记录完整 SHA256、原路径、
架构配置、teacher index 和来源命令。加载器只在内存中重定位经过哈希验证的来源路径，
不回退到原机器的绝对路径。teacher index 用于核验；历史标签轨迹不是加载必需项。

固定语义：H=2、F=4，`bounded_frontier`，每架飞机请求容量 5，soft 预约、
grace=300 秒、safety=60 秒；DRL 资源评分 + Hungarian 匹配，Ready 注入 `none`，
保留原共享编码器。H/F 或网络结构变更属于另一个实验，不是本冻结基线。

## 验证与使用

在仓库根目录运行（Python 3.11、PyTorch/PyG 等依赖需已安装）：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python onpolicy/scripts/train/verify_stage2_frozen.py
python scripts/export_stage2_frozen.py --verify-only
python -m unittest onpolicy.envs.HKBZ.test.test_stage2_frozen \
  onpolicy.envs.HKBZ.test.test_stage2_freeze_guard
```

第一条是 CPU 零更新验证：检查哈希、保护参数、观测/规划合同、模型形状与精确加载；
调用生产环境的 Stage3 恢复函数，并验证优化器及归一化状态重新初始化。
它不是调度评测、训练提速测试或新的 IGA 对比。

新分支发布验证记录见 [release_verification.json](artifacts/stage2_frozen/20260912/verification/release_verification.json)。
发布分支提供完整 CPU 回归和两个校验绑定的合成 TRAIN fixture，无需原机器数据路径；
IPC 测试需允许本地进程间 socket。fixture 仅用于测试，不是完整 benchmark 或性能证据。

Stage3 代码可以在创建 policy 前显式调用：

```python
from onpolicy.utils.stage2_frozen import load_frozen_stage2

checkpoint, args, bundle = load_frozen_stage2(
    "artifacts/stage2_frozen/20260912/manifest.json", args
)
# 创建 policy 后 bundle.validate(policy.ac.state_dict())；生产 runner 完成严格恢复。
```

通用训练入口支持 `--training_stage joint_finetune --stage2_frozen_manifest MANIFEST`。
这只是显式的新 Stage3 初始化接口，不是一套已经验收的 Stage3 训练方案。
需另行提供适配的数据集配置、预算和更新范围；不能同时传 `--checkpoint_dir`，
后续 Stage3 同阶段恢复不应再次传入冻结初始化 manifest。
旧 `prepare_stage3_*` / 自动实验启动脚本不是本冻结版本的正式交接入口。

## 保留结果与目标差距

最近完整 H/F 对照为同一 B0 在 tune60 上的 12 个 H/F 组合及 2 次控制复验，
共 840 case-episodes；没有更新 actor/critic。原生 H2/F4 回放与来源结果一致。

| 同一 tune60、H2/F4 | 平均 Cmax（秒，越低越好） | B0 相对差距 |
| --- | ---: | ---: |
| B0 + Hungarian | 8352.9222 | — |
| 历史 IGA180 | 8200.4389 | +1.86% |
| 历史 IGA1800 | 8094.9889 | +3.19% |

这是同算例历史结果对比；IGA 标签的名义搜索预算与实际耗时不完全一致，不能据此声称
严格等时优势，也不能将 tune60 的选择结果解释为盲测结论。
选择 B0 是因为交接与回放可验证，并非声称它是所有设置下的历史最优。
详情见 [结果说明](artifacts/stage2_frozen/20260912/results/README.md)。

`history/experiment_results.tar.gz` 保留 2,943 份历史 JSON 的原始字节，
`history/results_index.json` 列出来源、大小和 SHA256，未裁剪工程验证参数。
本地 `result/`、`onpolicy/scripts/results/` 的完整产物继续保留并被 Git 忽略；
未把约 148 GiB 原始日志、全部候选模型、数据集和标签轨迹上传。

## 退役与恢复

旧 Stage2 开发脚本及其专用测试已退出活动目录；仍被运行时导入的模块保留。
通用 `resource_joint` 训练和保留控制器的自动启动入口现在明确拒绝新 Stage2 训练；
只读模型构建、评测组件以及 Stage1/Stage3 仍可使用。

`history/retired_development.tar.gz` 保存退役前的原始脚本、测试和历史协议文档，
`history/retirement.json` 记录逐文件哈希。需要查阅时，解压到新建的独立临时目录；
不要覆盖当前代码。恢复历史脚本不等于该旧实验重新获得启动授权。

当前发布快照不包含无关迁移工具、私钥或尚未纳入本次验证的独立 Stage1/Stage3 开发入口。
