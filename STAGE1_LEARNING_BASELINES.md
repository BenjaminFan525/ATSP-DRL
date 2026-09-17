# Stage 1 学习型基线

仓库现已提供四个面向 HKBZ Stage 1 的同口径适配器：`l2d`、
`multi_ppo`、`fjsp_drl` 和 `daniel`。它们只替换飞机调度策略的表示与
解码器；环境动力学、合法动作掩码、训练/验证/测试实例、PPO 交互预算、
计时口径以及移动资源的 Hungarian 分配后端保持一致。

这些实现目前明确是 **Stage-1-only**。它们不用于 Stage 2 资源策略训练，
也不作为 Stage 3 联合微调的初始化架构。训练器会拒绝把非 `proposed`
基线与 `resource_policy=drl`、`resource_joint` 或 `joint_finetune` 组合。

## 方法映射

| CLI 值 | HKBZ 实验名 | 表示 | 动作解码 |
|---|---|---|---|
| `l2d` | L2D-AT | 两特征工序图 + 2 层 GIN | PPO 选择工序，规则选择最早可行完成场地 |
| `multi_ppo` | Multi-PPO-AT | 工序 GIN + 场地编码 | 工序策略后接条件场地策略 |
| `fjsp_drl` | FJSP-DRL-AT | 工序—场地 HGNN | 在合法工序—场地候选对上评分 |
| `daniel` | DANIEL-AT | 工序注意力块 + 场地注意力块 | 候选对与 8 个公共 pair 属性联合评分 |

这里的 `-AT` 表示面向本文航空保障环境的适配。它们保留原方法的核心
表示和动作分解，但不是上游仓库的逐字节复制：机器被映射为场地，动态
到达和空间约束由 HKBZ 环境统一处理，移动设备节点不进入基线编码器。
为隔离策略表示/解码器变量，四个适配器还共用本仓库的 PPO 优化、价值训练
后端和 rollout 逻辑；因此报告时应使用 `*-AT` 名称，而不能宣称复现了上游
模拟器中的原始数值。
基线 actor/critic 不读取本文解码器的 recurrent selection memory；公共接口中
对应的两个 query 槽固定为零。

## 单次训练

在已有 Stage 1 训练命令上只需增加架构开关，并确保使用飞机预训练阶段和
启发式资源后端：

```bash
$PYTHON onpolicy/scripts/train/train_hkbz.py \
  ...原有且固定的 Stage 1 参数... \
  --training_stage plane_pretrain \
  --resource_policy heuristic \
  --plane_order_mode fixed \
  --plane_pair_decoder joint_pair \
  --plane_bc_pretrain_epochs 0 \
  --stage1_baseline l2d
```

将最后一个值依次替换为 `multi_ppo`、`fjsp_drl`、`daniel`。环境 YAML
同样必须声明 `resource_policy: heuristic`，因为环境配置会覆盖命令行默认值。

## 批量生成公平实验命令

推荐从已经冻结的数据、奖励、PPO 预算和评测参数的 Stage 1 command JSON
生成四方法 × 至少三随机种子的命令：

```bash
$PYTHON onpolicy/scripts/train/prepare_stage1_learning_baselines.py \
  --source-command-json /path/to/reference.command.json \
  --output-dir result/stage1_learning_baselines/commands \
  --run-tag stage1_learning_baselines_v1 \
  --seeds 1,2,3
```

生成器只准备命令，不会启动训练。它会清除 checkpoint resume 和 IGA
behavior-cloning 参数，固定 `plane_pretrain + heuristic + fixed order`，并在
`experiment_manifest.json` 中记录源命令 SHA-256 和共同交互预算。静态方法
定义位于 `onpolicy/config/stage1_learning_baselines.yaml`。

## 检查点与评测

检查点新增 `stage1_baseline` 架构字段；本地恢复和共享 evaluator 都会核对
该字段，并能按请求重建对应编码器/actor。旧检查点没有该字段时只允许解释为
`proposed`，不会被误载入任一外部基线。
`valid_fjsp_v2_comparison.py` 也会从检查点自动恢复架构，并分别使用
`L2D-AT`、`Multi-PPO-AT`、`FJSP-DRL-AT`、`DANIEL-AT` 作为结果键，避免
把外部基线误标为 `DRL-G`。

所有结果仍需实际训练和测试后填写；实现本身不会生成或推断论文数值。应按
实例 ID 配对报告均值、标准差、相对 IGA gap 和置信区间，并至少使用三个种子。

## 回归验证

```bash
$PYTHON -m unittest -v \
  onpolicy.envs.HKBZ.test.test_stage1_learning_baselines
```

测试覆盖模型工厂、合法动作掩码、L2D 场地规则、PPO 动作回放、编码器形状和
设备信息隔离。实现入口是
`onpolicy/algorithms/utils/stage1_baselines.py`。

## 一手来源

- L2D: <https://github.com/zcaicaros/L2D>
- Multi-PPO: <https://github.com/Lei-Kun/End-to-end-DRL-for-FJSP>
- FJSP-DRL: <https://github.com/songwenas12/fjsp-drl>
- DANIEL: <https://github.com/wrqccc/FJSP-DRL>
