# HKBZ 两阶段训练迁移说明

当前入口只有两个阶段：

1. **Stage 1 / M2**：`plane_pretrain` 的飞机策略阶段，使用 heuristic
   移动资源并训练飞机 BC+PPO。由于离场状态机已经改变，旧 M2 只可作为历史对照，
   不能作为新环境的有效交接源；必须先用新语义重新生成 IGA 教师并从头训练 S1。
   新 M2 完成三种子验证后，再更新 `onpolicy/config/stage1_m2_handoff.json` 中的
   路径、大小、SHA256 和模型语义。
2. **Stage 2 / `resource_joint`**：使用 `env_resource_joint.yaml` 的 fjsp_v3 数据语义和 DRL resource policy。Runner 严格校验 M2 的 plane/shared protected tensors、`plane_order_mode`、`plane_pair_decoder` 与 `global_feature_mode`，然后执行正数 resource BC warm-up 和冻结 plane/shared 的 resource PPO。

Stage 2 显式启用 `device_lookahead_dispatch`，并使用 60 秒
`device_lookahead_safety_margin`。飞机一旦提交 operation/site
组合，环境就会为该目标工序缺少的移动资源发布“可延迟”的预请求；设备因此可在
飞机运输、等待 R014 或执行前驱工序期间预布置，而不必等到飞机正式阻塞。预请求
选择 no-op 后会抑制到下一次物理时间推进，避免零时间重复决策；真正阻塞飞机的
请求仍保留最后兼容设备不得 no-op 的防死锁约束。请求特征保持 checkpoint 兼容的
8 维，其中时间维用负 lead-time 表示预请求、非负 waiting-time 表示阻塞请求。

离场采用 `progressive-departure-r014-pipeline-v2` 语义。全局“所有飞机完成保障”
屏障已取消：每架飞机完成自身全部保障工序后，先执行一次 `ZY-T` 留位/腾位决策，
随后生成该飞机专属的 R014 pickup 请求。R014 在飞机原站位完成空驶预定位前，
飞机不能选择 `ZY-S`，起飞跑道也不会被提前占用；车辆到位后，环境按等待时间与
拖运距离动态匹配当前空闲跑道。这样未完工或尚未到达的其他飞机不会阻塞已完工
飞机离场，同时仍允许已完工飞机通过一次 `ZY-T` 为后续飞机腾出机位。

该语义会改变教师轨迹，旧 `departure-barrier-r014-v1` IGA 标注和在其上训练的
S1/M2 都不能复用。新教师契约明确记录
`teacher_scope=stage1_plane_policy`、`resource_policy=heuristic` 和
`environment_semantics_version=progressive-departure-r014-pipeline-v2`；加载器会拒绝
旧环境标签以及误用于 S2 的资源策略标签。

新环境的 train600 标注由
`onpolicy/scripts/train/launch_departure_iga_labels.sh` 串行执行：先用整机 CPU 生成
IGA-180，600/600 案例完成并独立回放验证后，才自动开始 IGA-1800。两套标签用于
S1 飞机策略的基线比较与 BC/DAgger，均不训练移动设备策略。

Stage 2 的 `checkpoint_DeviceBC.pt` 只是同一阶段内部的 resource-BC warm-up 边界，不是第三阶段，也不是新的恢复源。当前不承诺 shard 级精确恢复；中断后应从已验证的 Stage-1 M2 重新执行 warm-up。

## 迁移契约

`run_hkbz_two_stage_pipeline.py` 会先验证 hand-off 中全部 checkpoint；任一文件缺失、大小变化或 SHA256 不一致都会在启动训练前失败。它生成的原子 manifest 记录：

- `source_m2.path` 与 `source_m2.sha256`；执行前会再次检查源文件未被替换。
- canonical `training_stage: resource_joint`、`resource_policy: drl`、正数 BC/PPO epoch。
- `device_bc_train_gnn: false`、`device_bc_reset_optim: true`、`device_bc_save: true`，以及覆盖全部 PPO epoch 的 `gnn_freeze_epochs`/`plane_freeze_epochs`。
- 从 hand-off 中经哈希验证的 M2 命令继承 60 rollout、图批处理、梯度累积、ValueNorm 和自适应 KL 等已验证运行参数；旧 shared-evaluator socket 会被清除。
- PPO 使用 `team_cmax` 回报和 role-balanced resource-action ratio；不继承只聚合 plane action 的 `joint_team_ppo`，否则冻结 plane 后 resource actor 将没有有效 PPO 梯度。
- 完整 Stage 2 命令和 `operation_log`。Stage 1-only 的 plane BC、BC-reference 正则、selection/recovery 开关会被清除。
- warm-up、最终 `checkpoint_Best.pt` 和 `checkpoint_Last.pt` 的源 M2 SHA256/phase lineage。三个文件必须是不同、可验证的 Stage 2 artifacts。
- 训练 run 的 `run_status.json` 路径与 SHA256；只有 `status=completed`、最终 `resource_joint_completed` phase、M2 lineage、protected digest 不变、BC/PPO resource actor 更新证据都齐全时，manifest 才能进入 `completed`。

`device_bc` 和 `frozen_joint` 仍可被旧 Runner 解析，但只是 deprecated alias；新命令统一使用 `resource_joint`。`full_joint` 已退休。

## 完整示例

默认按 `SEED` 从已验证 hand-off 中选择 M2；它不会把 `checkpoint_DeviceBC.pt` 当成新的阶段输入：

```bash
ROOT="$PWD"
MANIFEST="$ROOT/result/hkbz_train_logs/two_stage/m2_to_resource_joint.json"

SEED=1 \
RUN_TAG="m2_to_resource_joint" \
MANIFEST_PATH="$MANIFEST" \
STOP_AFTER_STAGE=2 \
/bin/bash "$ROOT/onpolicy/scripts/train/launch_hkbz_two_stage_service.sh"
```

先登记而不训练：

```bash
SEED=1 \
RUN_TAG="m2_to_resource_joint" \
MANIFEST_PATH="$MANIFEST" \
STOP_AFTER_STAGE=1 DRY_RUN=1 \
/bin/bash "$ROOT/onpolicy/scripts/train/launch_hkbz_two_stage_service.sh"
```

只生成和审计命令（不会创建 systemd service，也不会启动训练）：

```bash
SEED=1 \
RUN_TAG="m2_to_resource_joint_audit" \
MANIFEST_PATH="$ROOT/two_stage_audit.json" \
STOP_AFTER_STAGE=2 DRY_RUN=1 \
/bin/bash "$ROOT/onpolicy/scripts/train/launch_hkbz_two_stage_service.sh"
```

也可以直接调用 controller：

```bash
python -u onpolicy/scripts/train/run_hkbz_two_stage_pipeline.py \
  run \
  --source-seed 1 \
  --run-tag m2_to_resource_joint \
  --manifest "$MANIFEST" \
  --bc-epochs 2 \
  --ppo-epochs 8 \
  --dry-run
```

只有在审计外部 M2 时才显式传入 `--source-m2`；此时还应显式传入与
checkpoint 一致的 `--plane-order-mode`、`--plane-pair-decoder` 和
`--global-feature-mode`。仓库内 M2 的固定语义是
`fixed / joint_pair / f1f2`。

真实执行完成后，controller 会要求并校验 `checkpoint_DeviceBC.pt`、`checkpoint_Best.pt`、`checkpoint_Last.pt` 的 lineage；缺失或源 SHA256 不一致会使 manifest 进入 `failed`，不会伪报完成。

## 旧命令迁移

旧的 `launch_hkbz_four_stage_service.sh` 现在只是兼容 shim，并显示弃用提示：

- `STOP_AFTER_STAGE=1`：转发到新 launcher，登记外部 M2。
- `STOP_AFTER_STAGE=2`：转发到 canonical Stage 2。
- `STOP_AFTER_STAGE=3` 或 `4`：立即拒绝；不存在 S3/full-joint 迁移。

新代码不再引用历史、被忽略且可能缺失的 `result/hkbz_train_logs/run_four_stage_fjsp_v2.sh`，也不复用 fjsp_v2 或旧绝对路径语义。

## S2 评测数据

`env_resource_joint.yaml` 训练仍使用完整的 600 个 fjsp_v3 train cases，在线
选模只读取独立的 `joint/tune`。`joint/gate` 用于正式选择，只有 winner 锁定后
才能读取 `joint/finalblind`。三者由以下命令原子生成：

```bash
python -u onpolicy/envs/HKBZ/experiment/build_resource_joint_eval_datasets.py \
  --output onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811 \
  --seed 20260811 \
  --exclude-manifest onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/manifest.json \
  --exclude-manifest onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/manifest.json
```

生成器会验证 60/120/60 数量、50/45/5 分布、内容哈希、跨 split 重复、派生
seed 重复和与排除数据集的重叠；目标目录非空时拒绝覆盖。
