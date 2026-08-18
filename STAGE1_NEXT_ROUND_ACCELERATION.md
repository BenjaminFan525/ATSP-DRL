# Stage-1 下一轮训练加速策略（待启用）

状态：`planned`。记录日期：2026-08-15。

本文只保存下一轮的调度与性能优化方案。当前正在运行的
`stage1_departure_reward_screen_20260813_r1` 不修改、不重启，也不在中途改变 CPU/GPU
亲和性。下一轮必须先完成短跑 A/B 验证，只有通过验收门槛的调度策略才可用于正式训练。

## 1. 当前实测基线

当前 Wave 2 每张 GPU 同时运行三个 trainer，每个 trainer 使用 12 个物理核及其两个
SMT sibling（24 个逻辑 CPU）、60 个 rollout worker，并与同卡其他实验共享一个
validation 服务。

最近八个 shard 的六任务聚合实测为：

- rollout：约 14.4 分钟/shard，占约 61%；
- PPO update：约 9.3 分钟/shard，占约 39%；
- 总计：约 23.8 分钟/shard；
- 16 shards：约 6 小时 20 分钟；
- validation：每组约 11--13 分钟；
- 完整 epoch：约 6 小时 27--30 分钟。

资源采样表明：

- 每个 trainer 的 systemd cgroup 约有 132 个 tasks，但只允许使用 24 个逻辑 CPU；
- 整机仍有约 56%--68% CPU idle，同时 runnable queue 可达到 60--220，说明静态分区造成
  “局部排队、全局空闲”；
- 每个训练进程占用约 24.8--25.1 GiB 显存，每卡合计约 75 GiB；
- GPU SM 利用率在 10%--100% 间变化，短时均值约 70%，显存带宽通常远未占满；
- 系统仍有约 209 GiB available 内存，swap-in/out 和磁盘 wait 均为 0，因此内存和磁盘
  不是当前主瓶颈；
- 新离场环境平均约 358--362 个决策步、44.8 次转运；旧环境约 265--294 个决策步、
  22.4 次转运。环境工作量增加是不可消除的基础成本。

## 2. 不允许通过加速改变的训练合同

以下内容必须保持与研究配置完全一致：

- 数据集、每 epoch 960 个采样案例及其确定性顺序；
- 60 个案例构成一个现有 shard 的语义；
- reward、potential、tau、BC-KL、学习率和所有退火日程；
- `ppo_epoch`、mini-batch、梯度累积、ValueNorm 和 optimizer step 数；
- actor/critic 网络、图输入、随机种子和 checkpoint 合同；
- completion/cycle/deadlock、old-policy KL、zero-update 等安全门槛。

不得为了速度直接减少训练案例、缩短 rollout、把 `ppo_epoch` 从 2 改为 1，或增加会导致
OOM 的 batch。当前每进程约占 25 GiB，三进程同卡时没有足够空间盲目扩大 batch。
`n_rollout_threads=60` 对 12 个物理核已经过量，下一轮不得继续增加 worker 数。

## 3. 第一优先级：低风险相位错峰

当前同卡三个任务只错开 0/60/120 秒，远小于约 23.8 分钟的 shard 周期，因此很容易形成
三个任务同时 rollout、随后同时 PPO update 的相位锁定。

下一轮候选配置：

```text
同卡 lane0 启动偏移：0 秒
同卡 lane1 启动偏移：480 秒
同卡 lane2 启动偏移：960 秒
另一张 GPU 整体再偏移 30 秒
```

该方案不改变任何训练样本和更新，只减少三个 CUDA 进程同时进入 PPO update 的概率。
预期收益为 7%--15%。实际偏移量必须由 A/B 短跑确认；若新的 shard 周期明显变化，可在
6--9 分钟范围内重新选取约三分之一周期的偏移。

## 4. 第二优先级：NUMA 内 CPU 弹性共享

静态 12 核切片保证可复现的时延，但 trainer 在 GPU update 时，其 CPU 核不能借给同卡
仍在 rollout 的任务。下一轮测试一个显式的 `soft_numa` 调度候选：

- 三个同卡 trainer 的 `AllowedCPUs`/`CPUAffinity` 均限制在对应 NUMA 节点内；
- 移除 trainer 内部的 12 核硬 `taskset`，保留相同 `CPUWeight`；
- `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、
  `NUMEXPR_NUM_THREADS=1` 保持不变；
- 不允许跨 NUMA 节点借核，避免远端内存访问；
- shared evaluator 仍为每 GPU 一个，评测请求使用受控 CPU 集合，不能无上限抢占仍在训练
  的两个任务；
- launcher 必须提供 `hard_slice` 回退模式，不能覆盖现有稳定调度。

`soft_numa` 预计可将 rollout 再缩短约 10%--20%，但可能增加单 shard 抖动。因此它不能
直接进入 Formal，必须先通过第 7 节的验收门槛。

## 5. Validation 调度

validation 在独立共享进程中运行，不参与 PPO 梯度。下一轮按实验目的选择：

- 四轮因果 screen：默认仍保留每 epoch validation，防止丢失学习曲线和退化发生点；
- 八轮 Formal 且不使用 validation early-stop：候选改为 E2/E4/E6/E8；
- Pre-PPO 只评测一次并复用同一不可变结果；
- 最终 checkpoint 和 test60 评测不得省略；
- 如果 checkpoint 选择依赖逐 epoch validation，则不得降低评测频率，除非事先把选择规则
  固定为最后一个 epoch。

每减少一次中间 validation，可节约约 11--13 分钟/任务，并降低同卡 evaluator 排队。

## 6. 并发模式按研究目标选择

### `screen_throughput`

- 每 GPU 三个 trainer；
- 优先测试 `hard_slice + 0/480/960 秒错峰`；
- 通过后再测试 `soft_numa + 相位错峰`；
- 目标是缩短六配置全部完成的总时间，而不是让单个配置最早结束。

### `formal_three_seed`

- 三个 seed 只需要在两张 GPU 上运行，不机械沿用“每卡三任务”；
- A/B 比较“一卡两个 trainer、另一卡一个 trainer”和“一次两 seed、随后补第三 seed”；
- 两 trainer 同卡时每个分配 18 个物理核及其 SMT sibling；单 trainer 可使用本 NUMA 的
  36 个物理核；
- 选择 makespan 最短的方案，而不是只比较单 seed 的 epoch 时间。

降低同卡并发预计可把单实验 epoch 缩短约 25%--35%，但六配置 screen 若因此分成多个
wave，总完成时间可能反而增加。因此不能把“单任务更快”等同于“整轮实验更快”。

## 7. 下一轮启动前的短跑 A/B

使用同一 checkpoint、同一训练 seed 和同一 case 顺序，先运行 warm-up shard，再至少测量
两个完整 shard。候选顺序为：

1. A：当前 `hard_slice`，同卡启动偏移 0/60/120 秒；
2. B：`hard_slice`，同卡启动偏移 0/480/960 秒；
3. C：`soft_numa`，同卡启动偏移 0/480/960 秒；
4. D：每卡两个 trainer、每 trainer 18 个物理核，用于 Formal 调度估算。

每个候选必须记录：

- 单任务和整卡 aggregate cases/s；
- rollout/update/shard 的均值、P50、P95；
- 六任务或三 seed 的预计总 makespan；
- GPU SM duty cycle、每进程显存峰值；
- CPU idle、runnable queue、NUMA 远端访问和 evaluator 排队时间；
- actor planned/actual optimizer steps、old-policy KL、BC-reference KL；
- completion、cycle、deadlock、timeout、zero-update、OOM；
- manifest、checkpoint、case-order 和训练合同哈希。

验收条件：

- 训练合同完全一致，所有案例 100% 完成；
- optimizer step completion 不低于基线且不低于 99%；
- zero-update、OOM、cycle、deadlock、timeout 均为 0；
- 显存峰值不超过安全预算；
- 相对 A，整轮预计 makespan 至少降低 10%，否则保留现有 `hard_slice`；
- 如果 P95 shard 时延恶化超过 20%，即使均值略快也不启用 `soft_numa`。

短跑只验证执行等价性和吞吐，不使用两三个 shard 的 validation Cmax 判断研究方法优劣。

## 8. 预期收益与回退

在不改变训练合同的前提下，目标区间为：

- 相位错峰：7%--15%；
- NUMA 内弹性借核：额外 10%--20%；
- Formal 降低中间 validation：约 0.5--1.5 小时/seed，取决于评测频率；
- 组合目标：把约 6.5 小时/epoch 降到约 5--5.5 小时；
- 只有 A/B 实测支持时，才把 4.5--5 小时视为可达目标。

任何候选出现训练合同变化、OOM、worker 卡死、step completion 下降、异常 KL、NUMA 跨节点
抖动或总 makespan 没有至少 10% 改善，立即回退到当前的静态 12 核切片方案。

## 9. 实现约束

后续实现时必须：

- 使用显式环境变量或参数选择调度，例如
  `HKBZ_SCHED_PROFILE=hard_staggered|soft_numa|formal_2plus1`；
- 默认值仍为现有 `hard_slice`，避免旧脚本静默改变行为；
- 将最终 CPU 集合、启动偏移、validation interval 和调度 profile 写入 manifest；
- launcher 启动前验证 NUMA 拓扑、CPU 集合、GPU 空闲和服务名不冲突；
- 不修改已经运行的 systemd 服务；新策略仅对新的 run tag 生效；
- 保存 A/B 原始状态、GPU/CPU 采样和结论，之后再启动正式实验。
