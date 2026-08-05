# 数据集说明

## 目录约定

训练与评估入口扫描数据集根目录下所有以 `case_` 开头的子目录：

```text
dataset/
├── train_large/
│   ├── case_01/
│   │   ├── job.json
│   │   ├── fixed_resources.json
│   │   ├── mobile_resources.json
│   │   ├── sites.json
│   │   └── flights.json
│   └── ...
└── test_large/
    └── case_01/...
```

五类 JSON 分别描述作业拓扑与资源需求、固定资源、移动资源、机位坐标和航班到达信息。字段名称沿用环境实现中的中文 schema。`onpolicy/config/` 提供一组最小示例文件。

## 生成数据

生成器接受固定场景规模和随机种子：

```bash
python -m onpolicy.envs.HKBZ.data_generator --help
```

建议为训练、验证、测试集分别指定不同的种子和输出目录。生成器不会自动划分数据，也不会覆盖式混合已有划分。`--no-layouts` 会跳过每个算例的 PNG，可显著减少生成时间和仓库体积。

## 训练约束

当前并行环境会把算例均分给 rollout worker，因此：

- `n_rollout_threads` 不能超过训练样本数，且必须整除训练样本数；
- `n_eval_rollout_threads` 不能超过评估样本数，且必须整除评估样本数；
- 同一批次中的算例应具有一致的节点规模，否则动作空间和批处理张量可能不一致。

## 发布数据时

数据集默认由 `.gitignore` 排除。若计划公开数据，优先使用独立 release、对象存储或数据仓库，并同时提供：

- 生成脚本版本、参数和随机种子；
- train/validation/test 的不可重叠清单或校验和；
- IGA teacher 的预算、随机种子和标注版本；
- 数据许可证及任何第三方来源说明。
