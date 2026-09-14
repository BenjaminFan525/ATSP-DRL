# HKBZ 研究进展恢复总账（2026-09-04）

> Material Passport
>
> - Origin Skill: `academic-research-suite / experiment-agent`
> - Mode: `validate + recovery planning`
> - Origin Date: `2026-09-04 Asia/Shanghai`
> - Verification Status: `ANALYZED; NO EXPERIMENT EXECUTED`
> - Version: `research-state-restore-v1`

## 0. 本文的用途

本文是工作空间从原本机迁往外部服务器、在那里继续研究、再完整回流本机后的统一研究总账。它回答四个问题：

1. 哪些结论已经锁定，哪些只是候选或失败分支；
2. 当前真正的研究前沿停在哪里；
3. 哪些中断任务可以续跑，哪些必须以新标签重启；
4. 如何在本机 `2 × RTX A6000 + 64C/128T` 上继续，而不破坏原有实验合同。

状态词约定：

- **LOCKED**：结论已由正式验证锁定，可以作为下游不可变量；
- **COMPLETE / NO PROMOTION**：实验完整，但没有方法通过晋级门槛；
- **SCREEN LEADER**：当前筛选最优，但尚不足以形成正式结论；
- **INTERRUPTED / RESTART REQUIRED**：中断且不能保证精确续训，必须保留旧证据并新标签重启；
- **PLANNED**：尚未执行。

本次恢复没有启动训练、评测或测试，也没有修改实验代码。

## 1. 工作空间来源与 Git 血缘

### 1.1 迁移闭环

工作空间的实际生命周期是：

```text
本机原始工作空间
  ├─ 本机 GitHub/清理分支（现保存在 before-full-restore 备份）
  └─ 2026-08-02 研究快照迁往外部服务器
       └─ 外部服务器继续完成 Stage 1、Stage 2 和后续研究
            └─ 2026-09-04 完整工作空间归档并回流本机
```

这不是两个无关项目。两个仓库在 2026-08-02 的提交 ID 不同，但树对象完全相同：

| 副本 | 提交 | tree | 含义 |
|---|---|---|---|
| 本机备份 `main` | `3351ec7` | `d4b1f9bdfb518c442b5295163852e564ad8cb4de` | complete stage1 dataset and training pipeline |
| 外部研究线 `master` | `47242cb` | `d4b1f9bdfb518c442b5295163852e564ad8cb4de` | 同一份 2026-08-02 内容 |

提交 ID 不同源于本机分支做过 history rewrite。研究内容血缘相同，但 Git 拓扑已经分叉。

### 1.2 两条分支各自承担的内容

- 当前恢复仓库：`master @ 9330e2f137fc983175be22e59ef6d1123c71ba3b`，承载外部服务器上的研究演进；相对它的旧 bundle origin ahead 5。
- 原本机仓库备份：`main @ dafbbf405342d4cc454f092167ff57c15c1589e8`，主要承载 2026-08-05～06 的 GitHub 发布清理、依赖和文档调整。
- 本机备份路径：`/home/fanyx/HKBZ-environment.before-full-restore-20260904`。
- 完整归档及校验文件仍保存在上述备份目录；验证完成前不要删除。

两条线不能直接 merge 或 force-push：本机清理线删除/重组了大量旧文件，而外部研究线在历史重写前的提交上继续开发。以后若要同步 GitHub，安全做法是先将当前研究状态冻结到独立恢复分支，再逐项移植发布清理；不要把 `dafbbf4` 当作 `9330e2f` 的祖先。

当前恢复仓库的 `origin` 指向外部服务器上的本地 bundle 路径，该路径在本机不存在。因此，在重新设计 remote 之前不要执行 `git fetch` 或 `git push`。备份仓库的 `origin` 才指向 GitHub。

### 1.3 当前 Git 状态的解释

已提交的外部研究线为：

```text
47242cb  2026-08-02  complete stage1 dataset and training pipeline
401e664  2026-08-10  add stage1 robust training and shared evaluation pipeline
a15821d  2026-08-11  Close Stage1 and prepare resource-joint Stage2
9829b87  2026-08-18  Add progressive departure Stage1 research pipeline
958e2dd  2026-08-23  Complete Stage2 resource planning pipeline
9330e2f  2026-08-23  Add full-data Stage2 formal training suite
```

2026-08-23 之后的大部分研究进展仍在未提交工作树中：当前有 61 个 tracked 文件被修改，另有 84 个 untracked 状态项。它们包含 8 月 24 日至 9 月 4 日的实验实现、计划、分析和状态，不应被清理或被旧仓库覆盖。

在继续实验前，应先建立一个新的恢复分支并冻结这批研究代码。提交时必须排除根目录的 `id_rsa`、运行结果、迁移包和用户凭据。

## 2. 当前的规范研究合同

整体问题分为两个正式训练阶段；Stage 3 目前只是探索性联合微调分支：

```text
Stage 1：飞机/工序调度
  progressive-departure-r014-pipeline-v2
  ↓ 锁定共享 encoder + 飞机 actor
Stage 2：移动资源联合决策
  resource_joint，监督学习为当前主线
  ↓ 只有 Stage 2 正式锁定后才允许进入
Stage 3：联合微调/共享表示研究（当前暂停）
```

当前规范说明以 [TWO_STAGE_TRAINING.md](TWO_STAGE_TRAINING.md) 和 [stage1_m2_handoff.json](onpolicy/config/stage1_m2_handoff.json) 为准。

[STAGE1_FINAL.md](STAGE1_FINAL.md) 中的旧 `M2_bc_kl_anneal` 结论属于 progressive-departure 语义引入前的历史阶段，已被后续 P5 正式实验取代。`stage1_m2_handoff.json` 文件名中的 `m2` 只是历史兼容名，不表示当前胜者仍是旧 M2。

## 3. 主研究链条

```text
旧 Stage 1 M2（历史结论，已被取代）
  ↓ 引入 progressive departure、R014 管线和 Potential V2
P5_team_time_potential_fixed（Stage 1 LOCKED）
  ↓ 固定 seed 3 检查点作为 Stage 2 源
Stage 2 资源 BC / planning / full-formal F1–F3
  ↓ matched-information IGA 揭示监督/表示差距
credit、PPO、Stage 3 多分支探索（多数无晋级）
  ↓ 回到高保真监督主线
H2/F4 IGA-1800 标签（600 cases）
  ↓ permutation-invariant matching
A3_matching_margin（当前 Stage 2 SCREEN LEADER）
  ↓ 发现 ready-time 头误差和评测混杂
P0–P5 predictor-only 训练筛选
  ↓ 改成 case-level frozen holdout
R0–R4 reliability Wave 1（2026-09-04 中断）
  ↓ 下一步必须从新标签完整重启
held-out predictor winner
  ↓
q50 / q80 / dynamic-risk 策略注入
  ↓
paired Cmax + 多种子正式验证
  ↓
Stage 2 LOCKED，之后才重新评估 Stage 3
```

## 4. Stage 1：已经锁定

### 4.1 当前正式胜者

- 方法：`P5_team_time_potential_fixed`
- 语义：`progressive-departure-r014-pipeline-v2`
- 飞机顺序：`fixed`
- pair decoder：`joint_pair`
- global feature：`f1f2`
- Stage 1 状态：**LOCKED / closed**
- Stage 2 默认源：seed 3

三个正式检查点：

| seed | 选择 epoch | validation raw Cmax | selection score | SHA-256 |
|---:|---:|---:|---:|---|
| 1 | 8 | 9223.850 | 9214.231 | `3055a1110502f63bf9db3be1c253cb922547e648a0e4cb90227ec0158229629b` |
| 2 | 7 | 9231.883 | 9223.377 | `70498db39c336c48b2144ab2f3ac5777b11a75f55abcb81ef0df519019a477d8` |
| 3 | 8 | 9134.617 | 9116.529 | `8ce0244f9b0b3877e2cc581a2479a6e5edab1854c1ed643749712612d0c5379d` |

默认源检查点：

`onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage1_departure_reward_formal_dual_20260816_r1_formal_P5_team_time_potential_fixed_seed3/run1/models/checkpoint_Best.pt`

本次恢复已重新计算 seed 3 文件哈希，仍为 `8ce024…379d`，与 handoff、teacher index 和可靠性 manifest 的源绑定一致。

### 4.2 blind test60 结论

证据：[summary_vs_iga.json](result/hkbz_train_logs/stage1_departure_reward_formal_dual_20260816_r1/test60_p5_best/summary_vs_iga.json)

- 3 seeds × 60 cases，完成率 100%；cycle、timeout、error 均为 0；
- learned mean = `8985.411 ± 70.288`；
- IGA-180 mean = `9473.167`，learned 优 `487.756` 秒；
- IGA-1800 mean = `8933.133`，learned 尚差 `52.278` 秒，即 `0.585%`；
- 三个 seed 全部达到 103% near-IGA 门槛；
- 整体没有达到“等于或优于 IGA-1800”的更强结论。

P5 在 blind test 前已经由 validation 锁定，test60 没有用于反向挑选方法或检查点。因此 Stage 1 不应重开调参。

### 4.3 Stage 1 学习型论文基线

[STAGE1_LEARNING_BASELINES.md](STAGE1_LEARNING_BASELINES.md) 已实现四个同口径 Stage-1-only 适配器：L2D-AT、Multi-PPO-AT、FJSP-DRL-AT、DANIEL-AT。

2026-09-03 只启动了 seed 1，四组均未完成：

| 方法 | 最后状态 | 最后位置 |
|---|---|---|
| L2D-AT | stale `running`，本机无对应进程 | epoch 0 / shard 7 |
| Multi-PPO-AT | interrupted | epoch 0 / shard 7 |
| FJSP-DRL-AT | interrupted | epoch 0 / shard 6 |
| DANIEL-AT | interrupted | epoch 0 / shard 5 |

seed 2 和 seed 3 尚未开始。现有 pre-PPO、Recovery 或 Emergency 文件不是论文结果；控制器也拒绝复用已有状态目录。该基线套件状态为 **INTERRUPTED / RESTART REQUIRED**，以后应新建 run tag，完整执行 4 methods × 3 seeds。

## 5. Stage 2：从资源规划到当前前沿

### 5.1 早期 full-formal F1–F3

证据：[ppo30_status.json](result/hkbz_train_logs/stage2_full_formal_three_20260823_r2/handoff/ppo30_status.json)

| 方法 | Pre-PPO | 最好 PPO | 最后可用结果 | 状态 |
|---|---:|---:|---:|---|
| F1 hard reservation | 8770.367 | 8737.517（epoch 3） | 8803.250（epoch 8） | completed |
| F2 IGA flow BC | 8792.228 | 8762.878（epoch 1） | 8896.717（epoch 5） | interrupted/failed lane |
| F3 wait constraint | **8747.183（Pre-PPO）** | PPO 未超过 Pre-PPO | 8904.522（epoch 8） | completed but PPO regressed |

该阶段说明 PPO 改善很小且不稳定；suite 总状态因 F2 中断为 failed，不能据此关闭 Stage 2。

### 5.2 matched-information IGA 诊断

证据：[matched_iga_comparison.json](result/hkbz_train_logs/stage2_matched_iga_tune60_20260827_r1/analysis/matched_iga_comparison.json)

| 对照 | mean Cmax | learned − matched IGA |
|---|---:|---:|
| matched IGA-1800 hard / F1 | 8592.850 | F1 `+144.667` |
| matched IGA-1800 soft / F2–F3 | 8562.239 | F2 `+200.639`；F3 `+209.694` |

三个 paired 95% CI 的整体上、下界均在 0 以上。该结果把主要问题定位到监督、表示和策略使用信息的方式，而不是继续无约束增加 PPO epoch。

### 5.3 8 月 24～31 日探索分支的处置

| 分支 | 结果 | 当前处置 |
|---|---|---|
| Stage 3 role credit | B1 在硬门控合格者中最低，但仍比共同 Pre-PPO 差 0.1284% | 不晋级 |
| Stage 3 encoder wave | 无方法 admitted | 停止 |
| Stage 3 critical path | C3 在本轮规则下被选中；C2 虽 raw Cmax 更低但未过门 | 仅保留诊断证据 |
| Stage 3 counterfactual HAPPO | 无方法 admitted | 停止 |
| Stage 2 event credit r1/r2 | pipeline failed | 停止 |
| Stage 2 MC/CF/timing-tail Wave A | 无 eligible method | 停止 |
| Stage 2 structured credit Wave B | 无 eligible method | 停止 |
| Stage 2 adaptive trust r4 | T2 通过该轮 gate，best score 9328.267，raw 仅改善 38.633 秒，距 matched IGA 仍 192.783 秒 | 历史候选，已被后续监督主线取代 |
| Stage 3 gradient conflict G0–G3 | 2026-09-01 集体中断，无最终 selection | 不续跑；Stage 2 锁定后再决定是否重开 |

这些实验不是“没有价值”：它们共同排除了单纯增加联合 PPO、credit shaping 或共享 encoder 更新就能稳定弥合差距的假设。但它们不能覆盖后续 A3 matching 结果，因此目前应暂停，而不是永久宣告 Stage 3 无效。

### 5.4 高保真 H2/F4 标签与完整监督恢复

证据：[hf_search_report.json](result/hkbz_train_logs/stage2_supervised_hf_labels_20260901_r1/hf_search_report.json)

- 独立 gate120 选择 `H=2, F=4`；mean Cmax = `8147.528`；
- 生成 600 个 IGA-1800 teacher cases；
- `intrinsic_ready_time` 标签 schema = 1；
- labeled requests = `2,956,665`，coverage = `0.9894806`；
- teacher 与 Stage 1 seed-3 SHA `8ce024…379d` 绑定。

完整监督训练 r2～r5 因浮点下界或 vector worker 故障失败；r6 在 2026-09-02 完成 2 个 BC epoch、每 epoch 16 rollouts。故障重试是同一研究节点的工程恢复，不应统计为五个独立科学实验。

### 5.5 matching Wave 1：当前资源策略筛选领先者

低 Cmax 更好；每组 60 cases、完成率 100%、cycle 0。

| 方法 | Pre-supervised | Post-supervised | Δ |
|---|---:|---:|---:|
| A0 sequential categorical | 8844.861 | 8667.383 | -177.478 |
| A1 global categorical | 8839.189 | 9218.933 | +379.744 |
| A2 final matching | 8839.189 | 8348.356 | -490.833 |
| A3 matching margin | 8839.189 | **8340.606** | **-498.583** |
| A4 resource adapter | 8859.089 | 8372.606 | -486.483 |

A3 的分布结果为 IID `7954.056`、OOD-stress `9018.136`、OOD-scale `6108.333`、tail `11170.167`。A3 是当前 **SCREEN LEADER**，但只有 seed 1 / 单轮筛选，尚不是 Stage 2 正式锁定胜者。

同日完成的旧损失基线为 L0 `8851.217`、L1 ranking-only `8747.344`、L2 timing-only `9101.050`，均不及 A2/A3/A4，支持 permutation-invariant matching 是关键改进。

### 5.6 ready-head Wave 1：训练集筛选，不是最终选型

P0–P5 都完成了 3 个优化 epoch，且预测没有注入策略、没有执行 PPO。最终训练轨迹主分数：

| 方法 | train selection score（秒） |
|---|---:|
| P0 | 610.168 |
| P1 | 562.174 |
| P2 | 561.629 |
| P3 | **560.293** |
| P4 | 566.105 |
| P5 | 565.501 |

P3 只以很小差距领先。由于这些值来自训练轨迹，且 Blocking 标签和 update 等权汇总存在混杂，P3 只能用于生成下一轮假设，不能被称为可靠性胜者。

## 6. 当前断点：ready-head reliability Wave 1

最新研究计划：[STAGE2_READY_HEAD_RELIABILITY_PLAN_20260904.md](STAGE2_READY_HEAD_RELIABILITY_PLAN_20260904.md)

### 6.1 原定合同

- 方法：R0–R4；
- 固定 Stage 1 seed 3、同一 H2/F4 IGA-1800 teacher、seed 1；
- 240 个分布平衡 cases：IID/OOD-stress/OOD-scale = 120/108/12；
- 3 个训练 epoch × 10 rollouts × 24 envs；
- 最后执行一次冻结、无反向传播的 fold-0 holdout pass；
- holdout 共 51 cases：IID 27、OOD-stress 22、OOD-scale 2；
- 只训练 `request_ready_head`；共享 encoder、飞机策略、资源策略、critic 和 `request_ready_feature` 全部冻结；
- `request_ready_policy_injection=none`，不评价等价的 frozen-policy Cmax；
- 主指标：`0.60 × H2_MAE + 0.30 × H1_MAE + 0.10 × Departure_MAE`；
- 选择器只接受完整 holdout 行、`checkpoint_BestPredictor.pt` 和所有冻结参数哈希不变。

### 6.2 实际中断位置

五组于 2026-09-04 08:17～08:18 启动，并在 19:30 同时收到 SIGTERM：

- 每组只完整产出训练 epoch 1 的一行指标和 `checkpoint_RequestReadyEpoch1.pt`；
- epoch 2 的 rollout 1～7 已完整结束；
- SIGTERM 落在 epoch 2 的 rollout 8 内，各组停在该 rollout 的不同 env step；
- 每组都有 `checkpoint_Emergency.pt`，但没有 holdout pass、没有 `checkpoint_BestPredictor.pt`、没有 `analysis/selection.json`。

训练 epoch 1 的观察值如下，它们不参与正式选型：

| 方法 | train-only score（秒） |
|---|---:|
| R0 block-rule control | 599.761 |
| R1 symmetric hybrid | 597.296 |
| R2 explicit context | 578.873 |
| R3 split/kind-balanced | 582.165 |
| R4 ordered quantiles | **568.287** |

R4 只是当前部分训练观察上最低，**不是 winner**。分析器会因五组缺少 holdout 而拒绝选出胜者。

### 6.3 为什么不能精确续训

底层 Stage 2 runner 支持“emergency checkpoint + 已完成 rollout 游标”的监督恢复，但本次中断发生在 rollout 8 中间：

- 应急检查点已经包含 rollout 8 的部分 optimizer 更新；
- 日志只证明 7 个 rollout 完整结束；
- 检查点没有 step-level 游标；
- 把 cursor 设为 7 会重做部分 rollout 8，把 cursor 设为 8 会跳过其未完成部分；两者都改变训练合同；
- checkpoint 内还序列化了外部服务器绝对路径，二进制迁移时没有改写。

因此原 r1 必须作为 **INTERRUPTED / RESTART REQUIRED** 的只读证据保存。不要覆盖、删除或在原目录内继续；新运行应从锁定的 Stage 1 源检查点完整重启。

## 7. 迁移后的完整性边界

### 7.1 已验证不变

| 项目 | 当前 SHA-256 | 结论 |
|---|---|---|
| Stage 1 handoff | `d53a2bc717ff97c7238eec6f530d1d8bdf1bd875a5b251d5cdd8a6cd19cd460f` | 与 r1 manifest 一致 |
| Stage 1 seed-3 checkpoint | `8ce0244f9b0b3877e2cc581a2479a6e5edab1854c1ed643749712612d0c5379d` | 与 handoff/teacher 一致 |
| Stage 1 P5 source command | `4df1d8b64069fc8ff8d1dd2c256dfdd1c605d3589f8ae125f197a53f6252d8a1` | 与 r1 manifest 一致 |
| teacher case 数 | 600 | 完整 |

### 7.2 teacher index 的预期哈希变化

迁移前 reliability manifest 记录 teacher index SHA：

`3587f54ea030b307d9bf62f2dbdb0a0957dc37f16b0b461ceadc85f963760d6b`

路径重定位后的当前 SHA：

`3b2cbe592d22fb2d4610a8edec3d6f10baec7b5258e3e6ca069e7efa6cfa9d35`

将当前 index 中的新项目根路径纯文本替换回外部服务器旧根路径后，SHA 精确恢复为 `3587f5…0d6b`。因此变化只来自 8 处绝对路径替换，不是 teacher 标签或规划合同变化。

旧 r1 manifest 仍记录迁移前 index SHA，适合作为历史证据，但不能直接作为新运行 manifest。新批次必须重新执行 prepare 阶段，生成绑定当前路径和当前 index SHA 的新 manifest。

历史 JSON 和 PyTorch checkpoint 中仍可能保留外部服务器路径；历史路径只用于溯源时，应按当前仓库相对位置定位。运行时代码和配置中的旧项目根残留已经清零。

## 8. 本机资源差异与启动器适配

| 项目 | 外部服务器原启动合同 | 当前本机 |
|---|---|---|
| GPU | GPU0 = NVIDIA A800 80GB | 2 × NVIDIA RTX A6000，约 48 GiB/卡 |
| CPU | 72 physical / 144 logical | 64 physical / 128 logical |
| NUMA | 双 NUMA 资源规划 | 单 NUMA node |
| RAM | 启动器按每 lane 88/96 GiB 上限 | 251 GiB 总 RAM，当前约 239 GiB available |

旧 [launch_stage2_ready_head_reliability_gpu0.sh](onpolicy/scripts/train/launch_stage2_ready_head_reliability_gpu0.sh) 不能原样执行：它会强制检查 `A800 80GB`，并要求 CPU 0–143、SMT sibling 偏移 72 和五路覆盖整个 144-thread affinity。

建议的本机等价 CPU 划分如下，保留完整 SMT sibling，并留 2 个物理核给 monitor/系统：

| lane | physical cores | logical CPU set |
|---:|---:|---|
| R0 | 13 | `0-12,64-76` |
| R1 | 13 | `13-25,77-89` |
| R2 | 12 | `26-37,90-101` |
| R3 | 12 | `38-49,102-113` |
| R4 | 12 | `50-61,114-125` |
| monitor/system | 2 | `62-63,126-127` |

首选仍让五组共享同一块 GPU0，以保持“所有方法处于同一物理 GPU”这一原设计；每组继续使用 allocator fraction `0.19`。外部服务器记录显示，启动时已有另一个进程占用约 52.9 GiB，而五组运行期间总占用峰值约 60.2 GiB；据此推断五组新增显存约 7.2 GiB，但该差值受外部进程影响，只能作为容量预检依据，不能视为精确单组显存测量。

若 A6000 预检显示显存不足，备选方案是固定 `R0/R1/R2 → GPU0`、`R3/R4 → GPU1`，并把 GPU UUID、映射和 allocator fraction 写入执行 manifest；方法、数据、seed、rollout 数和选择合同不能随之改变。

## 9. 恢复执行队列

### Gate 0：冻结研究代码与 Git 拓扑

状态：**PLANNED，执行前需要确认**。

1. 从当前恢复仓库建立独立 recovery branch；
2. 审阅并提交 8 月 23 日后的研究代码、配置和本文；
3. 明确排除 `id_rsa`、迁移归档、凭据和运行时大文件；
4. 保留 `before-full-restore` 仓库，不做自动 merge；
5. 为当前研究线配置一个不会覆盖 GitHub `main` 的新 remote/branch。

### Gate 1：静态与合同预检

状态：**PLANNED，未执行**。

应至少通过：

```bash
PYTHON=/home/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11

$PYTHON -m unittest -v \
  onpolicy.envs.HKBZ.test.test_stage2_supervised_pipeline \
  onpolicy.envs.HKBZ.test.test_two_stage_orchestration \
  onpolicy.envs.HKBZ.test.test_stage1_learning_baselines
```

还要验证：Stage 1 seed-3 SHA、handoff SHA、当前 teacher index SHA、600 个 teacher cases、240-case sample、51-case holdout、五路命令之间只有预定方法变量不同。

### Gate 2：完整重启 reliability Wave 1

状态：**NEXT / 最高优先级，未执行**。

建议新标识：

```text
RUN_TAG=stage2_ready_reliable_20260904_local_a6000_r2
UNIT_PREFIX=hkbz-s2-ready-rel-local-a6000-r2
```

要求：

- 从 Stage 1 seed-3 `checkpoint_Best.pt` 新鲜初始化；
- 重新生成 manifest，不复用旧 manifest 或 Emergency checkpoint；
- 保留原 r1 目录只读；
- 使用本机 CPU sibling 偏移 64；
- 把实际 GPU name/UUID、CPU set、代码提交和 dirty-state 指纹写入记录；
- 完成五组 3 个训练 epoch 和 frozen holdout 后才运行 analyzer。

### Gate 3：可靠性选型

状态：**BLOCKED BY GATE 2**。

只有同时满足以下条件才允许产生 winner：

- R0–R4 五组全部 `completed`；
- 每组 metrics 中恰有 3 个 `train` pass 和 1 个 `holdout_validation` pass；
- holdout case count = 51，分布 = 27/22/2；
- Blocking MAE 精确为 0；
- 所有冻结参数哈希不变；
- 每组存在 `checkpoint_BestPredictor.pt`；
- analyzer 生成完整 `analysis/selection.json` 且无 failure。

### Gate 4：策略注入

状态：**BLOCKED BY GATE 3**。

固定可靠性 winner 后，单独比较 q50、q80 和动态风险阈值的预测注入方式。该阶段才重新评估 paired Cmax、late dispatch、资源等待和 OOD/tail；不能把 predictor-only 训练分数当成策略收益。

### Gate 5：Stage 2 正式锁定

状态：**BLOCKED BY GATE 4**。

至少执行 3 seeds、固定 gate/test cases、paired bootstrap 和 blind selection discipline。只有此时才把 A3 或其后继方法写入正式 Stage 2 handoff。

### 独立论文基线队列

状态：**待 reliability 主线稳定后重启**。

Stage 1 的 4 × 3 学习型基线必须使用新 run tag 从 seed 1 重新开始，不能把 9 月 3 日的部分 checkpoint 当作完成值。它与 predictor 主线科学上独立，但二者都高度占用 rollout CPU；本机不应在没有重新分配 CPU 合同的情况下同时满载执行。

### Stage 3

状态：**PARKED**。

现阶段不续跑 gradient-conflict 或旧 joint-finetune 分支。Stage 2 正式锁定后，再决定是否用新 Stage 2 source 重开共享表示/联合微调；旧 Stage 3 结论只约束旧 source，不自动约束 A3 或后继模型。

## 10. 恢复后的单一事实源

后续恢复或写论文时按以下优先级取证：

1. terminal `run_status.json`、evaluation JSON、selection JSON 和 checkpoint SHA；
2. [stage1_m2_handoff.json](onpolicy/config/stage1_m2_handoff.json)；
3. 本文和最新日期研究计划；
4. launcher/manifest/command record；
5. service log；
6. 旧总结文档和目录名。

目录名中的 `m2`、`running` 或 `best` 都不能覆盖更高优先级证据。例如：L2D 的 JSON 仍写 `running`，但进程已经不存在，应解释为 stale；R4 的 epoch-1 分数最低，但没有 holdout，不能解释为 winner。

## 11. 当前一句话状态

**Stage 1 已由 P5 正式关闭；Stage 2 的当前单种子筛选领先者是 A3 matching-margin，研究正停在 ready-time predictor 的 case-level holdout 可靠性验证中；R0–R4 因迁移前 SIGTERM 在第二个训练 epoch 中途同时中断，下一步是在本机适配启动器后以新标签从头完整重启该 Wave，而不是续跑旧检查点。**

## 12. 2026-09-05 后续进展（不改写上面的历史快照）

本机 reliability r2 已完整完成；其 `analysis/selection.json` 选择了
`R1_symmetric_hybrid`，绑定 predictor SHA256
`c0c4825946e469b680e764a4c4b9f3bcbcfdaa66ccfb2fee221633b5d3afb1f1`。
这是预测头的选择结果，不等于已证明其能改善调度策略，也不代表正式 Stage2 关闭。

按后续计划新增 frozen-R1 BC 实现：B0 不注入、B1 仅 DAG、B2 DAG+R1，
只更新资源策略及适用的时间特征投影；共享编码器、飞机策略、critics 和 R1
预测头保持冻结。支持逐 epoch 评测及真实 Best 选择、完整 epoch 边界恢复、
严格确定性计算、逐案例配对分析与代码归档。未改动 Stage1 handoff，未启动 PPO。

工程验证：100 项回归测试通过；确定性 canary 和其恢复运行通过全部完整性审计。
连续运行与恢复运行的 508 个模型张量、Adam 状态、全部 RNG 流逐位一致，
4 个工程案例的 Cmax 差异为零。调试过程中暴露的失败及非确定性记录均保留，
具体证据见 [实现与验证记录](STAGE2_BC_INJECTION_IMPLEMENTATION_20260905.md)。

2026-09-05 15:46 CST 已启动
`stage2_bc_injection_20260905_gpu0_half_pilot_r1` 实验流程，使用 GPU0 和
CPU 0-31,64-95（64/128 个逻辑 CPU）。截至 15:47，状态为
`reference_validation`：正在重放历史 A3 的 tune60，三组 BC 尚未开始；
复验通过才自动执行固定 240 例、每组 2 个 epoch 的 B0/B1/B2 pilot。
最多并行两个训练进程，第三组排队。GPU1 的另一实验未受调度修改。
九组正式实验不会自动启动，仍需 pilot 结果审查。

## 13. 2026-09-05 A3 复验排查与 pilot_r2（最新）

pilot_r1 在16:00:51因 A3 历史逐案例门控失败退出，三组训练均未开始。
历史20环境评测与新12环境严格计算的均值为8340.605556 / 8362.016667秒，
16/60案例不同。已核验300输入文件及60案例的规范化指纹、模型及历史评测SHA，
并对照迁移源码确认环境转移、GNN、匹配解码和评测循环未改变。

固定输入探针直接证明：严格计算仅改变batch12→20，约2.3e-6的编码差异
即可使请求3在设备槽位34、57之间改派。恢复历史20环境/非严格计算后的
本机完整评测均值8341.288889秒，但仍有15/60案例不同；不能凭均值接近
声称跨硬件精确复现。另一方面，两次独立新进程的本机严格12环境评测，
全部60案例、Cmax及完整evaluation payload完全一致，最大差为0。

因此显式版本化本机A3基准，保留历史差异，门控改为本机零容差复验，并绑定
运行环境、源码、数据及诊断证据；没有放宽训练选择阈值。105项测试通过。
详见 [复验诊断报告](STAGE2_A3_REPLAY_DIAGNOSIS_20260905.md)。

17:26已启动 `stage2_bc_injection_20260905_gpu0_half_pilot_r2`，
仍限GPU0、CPU0-31,64-95；B0/B1/B2各240固定案例、2epoch、PPO=0，
最多两组并行。启动时状态为新服务自身的 `reference_validation`，
通过后再启动训练。原r1及全部诊断证据保留；正式九组仍未启动。

17:41更新：新服务复验也已通过，60/60逐例差为0，耗时875.660秒。
B0/B1训练进程已于17:40:46启动（PID2376349/2376350），完成Stage1与R1
加载，目前处于`pre_supervised`基线评测，尚未进行梯度更新；B2排队。
已核查整棵进程树64个进程、91个线程，无CPU越界，仅使用GPU0。

19:26 更新：按用户“启动 B2”的明确要求，B2 已于 19:24:58 提前启动
（PID2385273），正在执行训练前基线评测；B0/B1 保留原 PID 和训练进度。
三组 CPU 配额改为各 16 个逻辑 CPU，评测 12、监控 4，总数仍为 64，GPU0
不变；24/12 的训练/评测环境数和所有研究参数不变。原队列采用独占锁和
结果接管，避免重复启动 B2；接管前其旧 pending 列表不代表 B2 的实际状态。
真实状态见 suite 内 `external_jobs/B2_R1_seed11.json`，细节及 53 项通过的
回归测试见 [B2 提前启动记录](STAGE2_B2_EARLY_START_20260905.md)。
