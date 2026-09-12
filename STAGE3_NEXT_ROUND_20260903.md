# Stage 3 下一轮研究计划：T2 起点的共享编码器梯度冲突复现实验

## Material Passport

- Material ID: `stage3-t2-gradconf-4090-20260903-r1`
- Material type: Code Experiment Plan
- Status: `WAVE1_RUNNING`
- Created: 2026-09-03 (Asia/Shanghai)
- Working directory: `/data/fanyx/HKBZ-environment`
- Execution environment: `/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11`
- Hardware: 8 x NVIDIA GeForce RTX 4090 (24 GiB), 2 NUMA nodes, 64 physical / 128 logical CPUs
- Data classification: internal experiment artifacts; no external upload

## 1. 研究目标与可证伪假设

上一轮 G0--G3 共享编码器梯度冲突实验使用的不是最终选定的 Stage2 T2，且两组正式流程均被外部 SIGTERM 中断，不能形成方法结论。本轮只回答一个问题：在相同的 T2 初始化、样本顺序、PPO 配置和验证集下，冲突感知的共享梯度合并是否比直接求和更稳定、并带来可重复的 makespan 改善。

- H0：`norm_balance`、`norm_pcgrad`、`cagrad` 相对 `sum` 没有一致的 raw makespan 改善，或改善伴随 OOD/tail 不可接受退化。
- H1：至少一种冲突感知方法在两个配对种子上均优于 `sum`，两种子平均 raw makespan 至少改善 1%，同时 OOD-stress 退化不超过 2%、tail 退化不超过 3%。
- 本轮只有两个种子，结论只用于筛选；不做显著性声明。若进入下一轮，胜者与 G0 再补种子 3--5。

## 2. Stage2 起点与交接契约

历史 Stage2 评选明确选中：

`onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage2_adaptive_trust_20260830_r4_wave1_T2_backtrack_adaptive_soft_bc_seed1/run1/models/checkpoint_Epoch3.pt`

- 文件 SHA-256：`41cfc782c1ff908bb527c0219b75b73a1421d48a2dafa02d449015abbe5ab030`
- 完整模型摘要 SHA-256：`09854d261ea2fdfb6f79aeaf971c660f7d2b38cad7b4172ca41c4e10aa648148`
- Stage2 Valid60：selection 9328.2667 s，raw 8785.6333 s，OOD-stress 9437.8148 s，tail 10956.1667 s，完成率 100%，cycle/timeout 均为 0。
- 迁移后兼容预检：501/501 张量精确对应，无 missing/unexpected/shape mismatch；加载前后模型摘要一致；role-specific ValueNorm 可恢复。
- Stage3 初始化：精确继承模型权重和 role-specific ValueNorm；重置 actor/critic optimizer，不继承 Stage2 optimizer 动量；不继承 `shared_gradient_state`。
- 兼容性说明：T2 属于历史 `resource_joint PPO` checkpoint。当前严格 `joint_finetune` 分支已改为只接受新式纯监督 Stage2，因此本轮使用现有 `training_stage=auto` 通用恢复路径，并显式启用 `resume_stage1 + reset_optimizers_on_resume` 来实现旧 Stage3 的交接语义。实验记录必须保留上述 checkpoint 文件哈希和模型摘要；不得把 T2 改写或伪装成新式监督 checkpoint。

## 3. 实验变量与八卡布局

唯一自变量是共享编码器的三角色梯度合并方法：

| Arm | 方法 | 作用 |
|---|---|---|
| G0 | `sum` | 直接求和，对照组 |
| G1 | `norm_balance` | EMA 范数平衡，scale 限制在 `[0.5, 2.0]` |
| G2 | `norm_pcgrad` | 范数平衡后，对 cosine < -0.05 的角色梯度做对称 PCGrad |
| G3 | `cagrad` | `c=0.2` 的冲突规避合并 |

每个方法跑 seed 1 和 seed 2；同一 seed 内使用相同数据池和顺序，构成配对比较。每张 GPU 只有一个训练进程；Pre-PPO 与逐 epoch Valid 在该进程内顺序执行，不在同卡常驻第二份模型。

| GPU | NUMA | Arm | Seed | CPU（完整 SMT sibling 对） |
|---:|---:|---|---:|---|
| 0 | 0 | G0 | 1 | `0-7,64-71` |
| 1 | 0 | G1 | 1 | `8-15,72-79` |
| 2 | 0 | G2 | 1 | `16-23,80-87` |
| 3 | 0 | G3 | 1 | `24-31,88-95` |
| 4 | 1 | G0 | 2 | `32-39,96-103` |
| 5 | 1 | G1 | 2 | `40-47,104-111` |
| 6 | 1 | G2 | 2 | `48-55,112-119` |
| 7 | 1 | G3 | 2 | `56-63,120-127` |

隔离规则：systemd `AllowedCPUs`、`CPUAffinity` 与 `taskset` 三者使用同一集合；`NUMAPolicy=bind` 到 GPU 本地 NUMA；所有 BLAS/OpenMP 线程数固定为 1。八个 CPU 集合互不相交、覆盖全部 128 个逻辑 CPU，且不拆分 SMT sibling。

## 4. 固定训练契约

正式 Wave 1 沿用已有 `gradient_conflict_wave1` 协议，仅缩小 GPU microbatch：

- 4 个 epoch；每 epoch 240 个训练 case，来自固定 960-case 旋转池；20 个 rollout worker。
- Epoch 1：12 个 shard 的 critic-only calibration；Epoch 2：角色 actor 更新、共享编码器冻结；Epoch 3--4：共享编码器解冻，LR scale 固定为 0.01，此时才比较 G0--G3。
- `ppo_epoch=1`，`data_chunk_length=50`，`mini_batch_size=5`，`max_graphs_per_forward=250`。
- 有效批量目标保持 `grad_accumulation_target_graphs=5000`、`actor_grad_accumulation_target_graphs=2000`，因此降低的是峰值显存，不改变目标累计图数。
- 共享编码器 activation checkpoint 开启；PPO 后清 CUDA cache；PyTorch allocator 使用 expandable segments。
- 学习率与稳定器：actor `5e-6`、critic `1e-4`、clip 0.05、target KL 0.005、adaptive actor KL 开启。
- 奖励为纯 `team_time`；BC/DAgger、BC-reference KL、potential、wait dual 全部关闭。
- 所有 arm 使用相同 H3/F4、hard reservation 运行契约。T2 本身是 H1/F2 hard，因此每条流程先在目标契约下做相同的 Pre-PPO Valid60；PPO 增益以各自 Pre-PPO 为基线，避免把环境契约切换误算成学习增益。

## 5. 验证与主要终点

- Canary：每卡 20 个训练 case、10 个验证 case、1 个真实解冻更新；验证显存和完整 train->eval 路径。
- Wave 1：启动前 Valid60，随后每个 epoch 独立 Valid60；`evaluation_tau=0.3`，固定 partition seed 20260803，按 profile 分层。
- 主要终点：`best_epoch_raw_makespan - pre_ppo_raw_makespan`（越负越好），先做同 seed 的 G1/G2/G3 对 G0 配对差。
- 次要终点：IID、OOD-stress、OOD-scale、tail10%、完成率、cycle、timeout、actor step completion、zero-update shard、角色 KL、梯度 cosine/scale/projection rate/CAGrad weights。
- 不使用训练集 reward 选择胜者；不在看到 test60 后调参。

## 6. OOM 防护与 Canary 准入门槛

- RTX 4090 单卡约 24 GiB；训练器 allocator fraction 为 0.85，理论上限约 20.4 GiB，保留约 3.6 GiB 物理余量。
- 旧 A800 解冻更新在 graph=1000 时峰值 reserved 约 60--62 GiB；本轮从 graph=250 开始，并用每个方法自己的 canary 覆盖额外梯度操作。
- 训练和验证顺序执行；不使用 shared evaluator，不在同卡加载第二份常驻模型。
- Canary 准入：进程正常结束；`checkpoint_Epoch1.pt`、`pre_ppo.json`、`epoch_1.json` 均存在；无 CUDA OOM/non-finite/cycle/timeout；日志记录到一次解冻更新；峰值 reserved 不超过 20.0 GiB。
- 若峰值在 20.0--20.4 GiB 或发生 OOM，不自动重试；停止对应正式 arm，人工决定是否统一降到 graph=200。严禁只给个别 arm 改 graph cap。

## 7. 选择规则与后续波次

Wave 1 只有在 8/8 流程完成且数据契约一致时才排名。Arm 先满足：100% completion、0 cycle、0 timeout、actor step completion >= 0.95、0 zero-update shard、无 non-finite。然后按以下顺序：

1. 两种子平均 raw 配对改善；
2. 最差种子的 raw 配对改善；
3. OOD-stress 与 tail 安全性；
4. 梯度冲突率降低是否与性能方向一致。

若第一、第二名差距小于 0.25%，视为未决，不凭两个种子选胜者。Wave 2 对 G0 和最多两个候选补 seed 3--5；只有冻结配置后才跑 blind test60。

## 8. 执行阶段、监控与产物

1. Phase 0（立即启动）：8 卡 Canary，硬超时 6 小时，`Restart=no`，不自动重试。
2. Phase 1：8/8 Canary 通过后，人工确认统一 graph cap，再启动 8 卡 Wave 1；硬超时 96 小时。
3. Phase 2：收集 JSON/checkpoint/梯度诊断，做 paired descriptive analysis。
4. Phase 3：按上节门槛决定补种子或接受 H0。

监控至少覆盖：systemd 进程存活、30 秒 `run_status.json` heartbeat、service log 增长、每卡显存、6 小时硬超时。stall/OOM/异常退化只报告，不自动杀进程（硬超时除外）或自动重试。

- Suite 根目录：`result/hkbz_train_logs/stage3_t2_gradconf_4090_20260903_r1/`
- Canary 训练输出：`onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage3_t2_gradconf_4090_20260903_r1_canary_<ARM>_seed<SEED>/run1/`
- Wave 1 训练输出：`onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage3_t2_gradconf_4090_20260903_r1_wave1_<ARM>_seed<SEED>/run1/`
- 既有代码验证：`test_stage3_gradient_conflict.py` 与 `test_stage3_joint_finetune.py`，14 tests passed。

### 8.1 启动记录（2026-09-03 02:16 +08:00）

- 8 个 Phase-0 Canary 已通过 transient user service 启动：`hkbz-s3t2gc-can-r1-g0.service` 至 `hkbz-s3t2gc-can-r1-g7.service`。
- 启动后二次核验为 8/8 `active/running`、8/8 `NRestarts=0`；每张 GPU 恰有一个对应训练进程。
- 8 份 `run_status.json` 均为 `status=running`、`epoch=0`、`event=shard_started`，heartbeat 持续更新；Pre-PPO 产物已开始落盘。
- CPU affinity、AllowedCPUs、taskset 与 NUMA bind 均符合第 3 节冻结映射；GPU 0--3 绑定 NUMA 0，GPU 4--7 绑定 NUMA 1。
- 当前仍处于 Pre-PPO/验证阶段，每卡显存约 634--736 MiB；服务日志未检出 OOM、Traceback、non-finite、fatal 或 RuntimeError。解冻更新后的峰值显存仍须按第 6 节门槛验收。
- 完整解析命令保存在 Suite 根目录的 `commands/resolved_canary_*.json`，标准输出与错误输出保存在 `service_logs/`；禁止在 Canary 完成前启动正式 Wave 1。

### 8.2 Canary 验收与 Wave 1 启动（2026-09-03 07:47 +08:00）

- Canary 8/8 正常完成，服务 exit code 均为 0、`NRestarts=0`；每条流程的 `pre_ppo.json`、`epoch_1.json` 和 `checkpoint_Epoch1.pt` 均存在。
- 8 条 Canary 的 Valid10 均为 completion 100%、cycle 0、timeout 0；解冻更新峰值 allocated 为 15.921--16.722 GiB，峰值 reserved 为 16.504--17.148 GiB，低于统一 20.0 GiB 门槛。
- Pre-PPO 前 `best_eval_iid_makespan=inf` 与 `best_eval_composite_makespan=inf` 警告是尚未初始化的 best-metric 哨兵值；之后已由有限评估值替换，不属于 PPO 梯度或损失 non-finite。
- 正式 Wave 1 已通过 `hkbz-s3t2gc-w1-r1-g0.service` 至 `hkbz-s3t2gc-w1-r1-g7.service` 启动。启动验收为 8/8 `active/running`、8/8 `NRestarts=0`、8/8 heartbeat 正常，每张 GPU 恰有一个对应训练进程。
- 96 小时硬超时、`Restart=no`、CPU/NUMA 隔离和 19.989 GiB allocator 上限均已生效；每条流程已确认 T2 checkpoint、严格 observation contract、role-specific ValueNorm、新 actor/critic optimizer、240 cases/epoch 的 960-sample rotating pool及 Valid60 分区。
- 正式解析命令保存在 `commands/resolved_wave1_g<GPU>_<ARM>_seed<SEED>.json`；服务日志保存在 `service_logs/hkbz-s3t2gc-w1-r1-g<GPU>.log`。

## 9. 当前限制

本轮的 legacy handoff 可精确复现既有 T2 权重与 ValueNorm，但 `run_status.training_stage` 会记录为 `auto`，不是当前新式 `joint_finetune`。因此本轮适合回答旧 Stage3 梯度冲突问题，不可与新式监督 Stage2/matching 分支混称。后者的 2026-09-02 matching wave 在服务器迁移时中断，尚无正式胜者；若未来切换到该分支，应另建实验 ID 并重新做完整 Canary/Wave 1。
