# Stage3 双臂 CPU 隔离记录

2026-09-17 18:54 CST：已在运行中的双臂实验落实全线程 CPU 绑定，无需重启训练。

| 实验 | 允许的逻辑 CPU | 物理核 / 逻辑 CPU | 已核验进程 / 线程 |
|---|---|---|---|
| C03 | `0-15,64-79` | 16 / 32 | 34 / 70 |
| T10 | `16-31,80-95` | 16 / 32 | 34 / 70 |

两个集合不重叠，SMT 兄弟线程完整归属同一实验。合计仍为授权的
`0-31,64-95`，GPU0 上的双臂训练继续运行。

## 修复内容

此前主线程和环境进程已经分区，但两臂各有一个在 Python 初始化时创建的
辅助线程继承了整块 CPU 亲和性。本次纠正了这两个线程，并检查全部子进程
和线程，保留环境进程原有的更窄绑核范围。

- 工作区启动器在执行 Python 之前使用 `taskset`，防止库初始化线程先于绑核创建。
- 本轮不可变训练快照由外部守护补充检查，每 2 秒扫描一次，覆盖后续 epoch。
- 守护服务为 `hkbz-stage3-h3-cpu-isolation-20260917.service`，PID 3804506；
  绑定 CPU `31,95`，实测内存约 8.4 MiB，随训练服务退出。
- 实施方式是 Linux 全线程 scheduler affinity。宿主机未向当前用户委派
  cpuset controller，因此没有声称完成 cgroup cpuset 分区或内存隔离。

原训练 controller PID 3746830、两臂 PID 3757026/3757027 保持运行。
486 个不可变训练源文件的哈希全部一致。MemoryHigh 164 GiB、MemoryMax
176 GiB 未更改；核验时内存约 169.4 GiB，内存限速问题仍存在。

## 验证

33 项相关 CPU 测试通过，包括真实子进程启动时的主线程、辅助线程、后代
亲和性继承，以及对已运行辅助线程的纠偏、窄范围保留与 PID 身份检查。
这不是完整仓库测试。

证据位于
`result/hkbz_train_logs/stage3_h3_continuation_tau_20260917_r2_dual_gpu0/`：

- `cpu_isolation_initial.json`：两个越界线程的修正前后记录。
- `cpu_isolation_status.json`：持续更新的全线程核验结果。
- `operational/cpu_isolation_20260917/deployment.json`：守护源码哈希与部署配置。
- `operational/cpu_isolation_20260917/verified.json`：服务状态、进程核验和内存现状。
- `operational/cpu_isolation_20260917/tests.xml`：33 项测试报告。
