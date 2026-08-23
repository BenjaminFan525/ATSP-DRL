# HKBZ 两阶段训练迁移说明

当前入口只有两个阶段：

1. **Stage 1 / P5**：`plane_pretrain` 的飞机策略阶段，使用 heuristic
   移动资源并训练飞机 BC+PPO。新离场语义下的 P5 已完成三种子 validation 与
   blind test，并作为关闭后的 Stage-1 交接源登记在
   `onpolicy/config/stage1_m2_handoff.json`。文件名中的 `m2` 只是历史接口名，当前
   winner 是 `P5_team_time_potential_fixed`，不是 8 月 8 日的旧 M2。
2. **Stage 2 / `resource_joint`**：使用 `env_resource_joint.yaml` 的 fjsp_v3 数据语义和 DRL resource policy。Runner 严格校验 P5 的 plane/shared protected tensors、`plane_order_mode`、`plane_pair_decoder`、`global_feature_mode`、观测 schema 与环境状态机版本，然后执行正数 resource BC warm-up 和冻结 plane/shared 的 resource PPO。它已经合并原先的 S2/S3；不会再解冻进入 S4。

当前默认交接源是 P5 seed3 的 validation Best（episode 8）。选择依据是三种子中
最低的 formal-validation composite `9116.5293`，不是 blind test。Stage-2 的
`SEED` 只控制资源策略训练随机性；如未显式设置 `SOURCE_SEED`，三个 Stage-2 seed
都从同一份 seed3 飞机策略出发，以隔离资源学习的因果贡献。若要做完整端到端
seed lineage 对照，可显式设置 `SOURCE_SEED=1|2|3`。

HKBZ 没有采用 IA 农机参考实现中的双 encoder。当前仍是一个共享异构图 encoder，
Stage 2 将其与飞机 GRU/actor 全程按位冻结，只训练普通移动设备与 R014 各自的
GRU/actor，以及 critic。这样既保留 P5 的飞机观察表示，也允许两类资源拥有不同
动作后端。

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

Stage 2 的移动资源教师由
`onpolicy/scripts/train/launch_stage2_resource_iga_labels.sh` 生成。飞机策略固定为
hand-off 中按 formal-validation 选出的 P5 seed3 validation-Best，普通移动设备与
R014 均由 `iga_all` 染色体控制；每个案例保存完整的 plane/resource 决策轨迹和
终局独立回放证据。标注器在两张 GPU 上各运行 3 个分片，每个分片绑定 12 个完整
物理核及其 SMT sibling，并把内存优先分配到 GPU 对应 NUMA node。IGA-1800 是
IGA-180 的嵌套细化：先验证 180 秒 incumbent，再追加 1620 秒搜索，因此累计预算
仍为 1800 秒而不是 1980 秒。输出的 `teacher_scope` 为
`stage2_resource_policy`，Stage-1 的教师加载器会拒绝误用这套标签。

当前 Stage2 方法筛选不再把这些轨迹仅作为离线分析材料。训练环境通过
`resource_iga_teacher_actions()` 在每个实际访问状态上用实时 request mask 解码
IGA-1800 染色体；600 份教师先被压缩为只含 provenance、染色体和终局指标的
训练视图，并由 sidecar 将“原始轨迹 SHA256 → 精简教师 SHA256 → 数据集 case
fingerprint”逐案例绑定。任何案例、轨迹或环境语义不一致都会在 BC 前失败。

GPU0 单卡 Wave 1 固定 Stage1 P5 seed3 和 Stage2 seed1，并行比较三种方法：

1. `M0_heuristic_uniform`：Hungarian 实时教师、普通 pooled BC；
2. `M1_iga_uniform`：IGA-1800 实时教师、普通 pooled BC；
3. `M2_iga_role_balanced`：IGA-1800 实时教师，普通移动设备与 R014 分别归一化
   NLL 后等权合并。

三组共同使用 180 个固定比例 screen cases、28 rollout workers、2 BC epochs、
4 PPO epochs 和同一 `team_cmax` PPO 契约。BC 每个 epoch 强制轮转 7 次，从而
覆盖全部 180 个案例，而不是达到 64 个 label 后只看首批 24 个案例。资源 DAgger
支持按 epoch 的教师执行率，但混合单位是“一整个环境的资源联合动作”；教师标签
始终在学生实际访问的状态上查询，禁止逐设备混合造成重复 claim。

正式 Wave 1 会依据三并发 canary 的实测峰值自动采用吞吐配置：28 rollout
workers、`mini_batch_size=24`、`data_chunk_length=50`，即大批次处理 1200
个图、余下 4 个环境处理 200 个图；actor/critic 梯度累积相应改为 2/6，分别
得到精确的 1400 图和 4200 图 effective batch，接近原来的 `4×350` 与
`11×350`。PPO generator 按“时间块优先、环境分片其次”输出，并按每个分片的
有效决策质量加权，因此 `1200+200` 会落在同一个 optimizer step 内。180 个
案例因此每个 BC epoch 轮转 7 次并保证全部
覆盖。正式 manifest 会读取 canary 的逐 5 秒显存记录，按整个 GPU 占用随图数
线性增长这一更保守的假设投影；投影超过 72 GiB 时拒绝启动，而不是冒险 OOM。
r7 中三组 resource actor 同时反向传播时的实测峰值约为 19.34 GiB；1200 图的
保守投影约为 66.31 GiB，在 72 GiB 启动红线下仍保留约 7.42 GiB 余量。

入口为：

```bash
bash onpolicy/scripts/train/launch_stage2_resource_research_gpu0.sh
```

启动器先在 GPU0 同时运行三组 24-case canary，并让它们共用一组 validation
进程。canary 的训练轨迹固定为 64 步：这足以让每个进程实际执行在线教师回放、
BC backward 和 350-graph PPO microbatch，不再为显存预检额外跑完整的
400--600 步调度。canary 还把 `actor_warmup_shards` 设为 0，确保唯一 shard
确实更新 resource actor；它省略只用于选模的 Pre-PPO baseline，但三组
Post-PPO validation 仍始终自然结束。正式 Wave 1 恢复一个 actor warm-up shard、
Pre-PPO baseline，并且 BC/PPO 训练也
始终自然结束。只有三组均正常退出、日志无 CUDA OOM 且整卡峰值不超过 72 GiB，
才自动启动正式 Wave 1。若标准 350-graph forward 超限，会自动改用 250-graph
microbatch，并用更大的梯度累积近似保持 effective batch。这个单 GPU 方法筛选轮
不做实验间 CPU 隔离：三个训练和共享 evaluator 的 systemd cgroup 都允许使用
`0-143`，由系统动态调度；每个进程仍设置 OMP/MKL/OpenBLAS/NumExpr 为 1，避免
环境 worker 内部线程爆炸。allocator 固定继承
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

Stage 2 的 `checkpoint_DeviceBC.pt` 只是同一阶段内部的 resource-BC warm-up 边界，不是第三阶段，也不是新的恢复源。当前不承诺 shard 级精确恢复；中断后应从已验证的 Stage-1 P5 hand-off 重新执行 warm-up。

## Stage2 全量正式训练

Wave 4 的 180-case 结果只用于方法筛选。全量正式入口为：

```bash
bash onpolicy/scripts/train/launch_stage2_full_formal_gpu0.sh
```

该入口从同一份 Stage1 P5 seed3 checkpoint 重新开始，选择五个未发生 OOM 的
稳定配置：soft reservation、hard reservation、IGA-flow BC、wait constraint、
IGA + wait constraint。训练设置恢复 Stage1 的完整 train600 覆盖；50/45/5
平衡采样将 600 个唯一案例展开为每 epoch 960 个槽位，即 20 个 rollout worker
顺序执行 48 个 shard。四轮 DAgger/BC 后执行八轮 PPO，飞机策略和统一共享
encoder 在全部 S2 PPO epoch 中保持冻结。

五个训练进程仅使用 GPU0。每个训练服务独占六个物理核及其两个 SMT 线程；
共享验证独占五个物理核；监控/系统预留一个物理核。CPU 集合合计恰好是 NUMA0
的 72/144 个逻辑 CPU，不跨物理核、NUMA 或训练服务重叠。soft heuristic 与
wait-constraint 复用一份完整 BC，IGA-flow 与 IGA-constraint 复用另一份；hard
reservation 单独生成 BC。复用只消除完全相同的监督阶段，五条 PPO 轨迹、优化器
和 checkpoint 仍相互独立。

## 迁移契约

`run_hkbz_two_stage_pipeline.py` 会先验证 hand-off 中全部 checkpoint；任一文件缺失、大小变化或 SHA256 不一致都会在启动训练前失败。新版 hand-off 还会在 CPU 上加载 checkpoint，逐一验证新离场语义、观测 schema、validation Best 元数据，以及当前 Stage-2 网络中所有 protected tensor 的名称和 shape。P5 的多命令实验 manifest 通过 `source_command_key` 精确定位对应 seed，不会误继承另一条命令。它生成的原子 manifest 记录：

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

默认从 hand-off 选择预登记的 P5 seed3；它不会把
`checkpoint_DeviceBC.pt` 当成新的阶段输入：

```bash
ROOT="$PWD"
MANIFEST="$ROOT/result/hkbz_train_logs/two_stage/m2_to_resource_joint.json"

SEED=1 \
RUN_TAG="m2_to_resource_joint" \
MANIFEST_PATH="$MANIFEST" \
STOP_AFTER_STAGE=2 \
/bin/bash "$ROOT/onpolicy/scripts/train/launch_hkbz_two_stage_service.sh"
```

显式使用同 seed 的端到端 lineage 时设置 `SOURCE_SEED`：

```bash
SEED=1 SOURCE_SEED=1 \
RUN_TAG="p5_seed1_to_resource_joint_seed1" \
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
  --source-seed 3 \
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
