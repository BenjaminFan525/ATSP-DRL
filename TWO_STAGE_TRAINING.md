# HKBZ 两阶段训练迁移说明

当前入口只有两个阶段：

1. **Stage 1 / M2**：既有 `plane_pretrain` 训练的 M2 产物。它是外部完成、不可变的交接源；`onpolicy/config/stage1_m2_handoff.json` 固定三个 seed 的路径、大小、SHA256 和模型语义，controller 只验证并登记，不重新解释或覆盖 Stage 1。
2. **Stage 2 / `resource_joint`**：使用 `env_resource_joint.yaml` 的 fjsp_v3 数据语义和 DRL resource policy。Runner 严格校验 M2 的 plane/shared protected tensors、`plane_order_mode`、`plane_pair_decoder` 与 `global_feature_mode`，然后执行正数 resource BC warm-up 和冻结 plane/shared 的 resource PPO。

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
