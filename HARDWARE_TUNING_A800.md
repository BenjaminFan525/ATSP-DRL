# 双 A800 训练配置

## 本机拓扑

- GPU：2× NVIDIA A800 80GB PCIe。
- CPU：2×36 物理核，SMT2，共 144 个逻辑 CPU。
- GPU0/NUMA0：`0-35,72-107`。
- GPU1/NUMA1：`36-71,108-143`。
- 两张 GPU 之间以及 GPU 到远端 NUMA 节点均经过 `SYS` 链路，因此独立 seed 应使用本地 NUMA CPU，不应共享全机 CPU 池。

## 2026-08-02 实测

使用同一个 Stage-1 checkpoint 和相同 60 个案例：

| 测试 | 用时 | 结论 |
| --- | ---: | --- |
| 推理 batch 60 | 255.64 秒 | 推荐 |
| 推理 batch 30×2 | 361.46 秒 | batch 60 快 29.3% |
| PPO 350 graphs/forward | update 301.20 秒，峰值 19.49GiB | 推荐 |
| PPO 700 graphs/forward | update 300.16 秒，峰值 38.64GiB | 无有效提速，浪费约 19GiB |

双卡同时跑一个 shard 时，350 和 700 配置的总耗时分别为 629.69 秒和
645.08 秒。两者 rollout 分别为 328.49 秒和 344.93 秒，说明主要瓶颈是
CPU 环境仿真和 IPC，而不是 A800 显存或 PPO 图批量。

### 每卡双任务错峰实验

进一步测试了每张 GPU 同时承载两个完整的 60-worker 训练进程。第二对任务在
第一对进入 PPO update 后启动，使 rollout 与 update 重叠：

| 任务 | rollout | update | shard 总时间 |
| --- | ---: | ---: | ---: |
| GPU0 第一项 | 332.53 秒 | 400.66 秒 | 733.20 秒 |
| GPU1 第一项 | 341.52 秒 | 410.97 秒 | 752.49 秒 |
| GPU0 第二项 | 407.10 秒 | 309.57 秒 | 716.67 秒 |
| GPU1 第二项 | 413.11 秒 | 311.38 秒 | 724.49 秒 |

- 双任务基线：120 cases / 645.08 秒 = `0.1860 cases/s`。
- 四任务错峰冷启动：240 cases / 1142.24 秒 = `0.2101 cases/s`。
- 保守总吞吐增益：`12.95%`。
- 重叠窗口内系统内存使用约 132GiB，仍有约 367GiB available；显存约
  22GiB/卡，无 OOM 或 IPC 超时。

错峰会使第一项 update 慢约 33%，第二项 rollout 慢约 22%，但增加的并行
工作量更大，因此即使把第二对约 6.4 分钟的延迟全部计入，整体吞吐仍提升。
长训练会摊薄启动延迟，但评估阶段可能改变相位关系，所以四任务模式应视为
“最大吞吐模式”，双任务静态隔离仍是“最低抖动模式”。

## 推荐正式配置

每个任务使用：

- 1 张 A800；
- 与该 GPU 同 NUMA 的 36 个物理核及其 72 个 SMT 线程；
- 60 个 rollout worker、60 个 eval worker；
- `mini_batch_size=7`、`data_chunk_length=50`、`max_graphs_per_forward=350`；
- `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`；
- systemd `AllowedCPUs` + `CPUAffinity`，进程内再用 `taskset`；
- 两个服务错开 30 秒启动。

GPU0 和 GPU1 应运行相互独立的 seed/实验。不要让同一服务跨两张 PCIe GPU；
当前代码没有 DDP，跨卡不会加速一个模型，反而会经过跨 NUMA `SYS` 链路。

若实验数量充足且目标是整机总吞吐，可在每张 GPU 上放置两个任务；同卡两项
共享该 GPU 的本地 NUMA CPU 集，并错开约 360–390 秒。实测这比每卡单任务
快约 13%，代价是单任务完成时间增加、性能抖动变大。正式基线比较仍建议使用
每卡单任务模式。

## 启动

启动两个独立的 Stage-1 实验：

```bash
BASE_TAG=my_stage1 \
FORMAL_SEEDS_GPU0=31 \
FORMAL_SEEDS_GPU1=32 \
bash onpolicy/scripts/train/launch_stage1_dual_a800.sh
```

恢复现有的单个 suite 时保留原始训练几何：

```bash
RUN_TAG=tail_recovery_stage1_20260730_r1 \
GPU=0 \
CPU_AFFINITY=0-35,72-107 \
RESUME=1 \
bash onpolicy/scripts/train/launch_stage1_iga_weekly_suite.sh
```

现有 suite 的 seed1 仍应使用 350-graph 配置精确续跑。不要在恢复点中途切换
mini-batch 或 rollout shard 几何；新的独立实验再使用双服务并发调度。

## 2026-08-04 保序流水线

在不改变 worker 数、rollout shard、mini-batch、优化器步数和随机数调用顺序的
前提下，增加了三项默认关闭的加速路径：

- `--safe_dagger_teacher_overlap`：先向全部环境发出只读 IGA teacher RPC，
  同时执行确定性的 student 图组批和前向，最后仍按 worker 原顺序取回标签；
- `--safe_graph_batch_pipeline`：后台构造下一个 PyG `Batch`，同一 minibatch 在
  BC reference、当前策略和 post-update probe 之间复用，不改变 sample 顺序；
- `--safe_async_graph_clone_workers 4`：IPC reply 到达后立即在父进程线程池中
  克隆图张量，并与尚未完成的环境 worker 重叠；最终结果仍按 worker 编号提交。

训练环境的图克隆线程会在延迟创建 eval worker 前停止，eval worker 全部 fork
完成后再恢复，避免多线程父进程直接 fork。所有选项默认值都是关闭状态，因此
正在执行的 screen、audit 和 phase2 不切换实现；Stage-1 v2 仅在新建 formal
训练进程中启用。checkpoint 和 `run_status.json` 会记录三项实际取值，formal
现有的 canary、KL gate、step-completion gate 和 deterministic validation 保持不变。
