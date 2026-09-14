# B_SHARED B0 训练前评测并行化

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent。
- Mode：implementation / runtime verification。
- Date：2026-09-12。
- Scope：r2 的训练前评测调度；GPU0、CPU `0–31,64–95`。
- Verification Status：19 项 CPU 测试通过，实际 GPU 并发对照 24 例逐项完全一致；已于 19:30 停止原 controller，并切换双进程前置评测。

## 改动范围

当前原生评测为一个策略进程，最近 5 分钟 GPU0 平均利用率约 32%，显存峰值 7519 MiB。Tune60 已复现冻结 B0，完整耗时约 799 秒。优化采用两个独立评测进程，共用 GPU0；每个进程仍执行原始固定 12 例批次。

新增入口 `onpolicy/scripts/train/run_stage3_b_shared_b0_preflight.py` 导入 r2 的原始源码快照，核验完整 manifest、源码、依赖和数据 SHA。适配器单独冻结并记录 SHA，不改写原有训练快照、manifest 或科学预算。

- 任务是完整的 `(baseline/zero, split, 原始批次起点)`，不重排或缩减批内案例。
- 环境 seed、槽位编号、已完成环境、cuDNN 设置、模型、Hungarian 解码均使用原始冻结 evaluator。
- 保留 baseline780 和 zero180 全部查询要求、Train600 归一化统计。已落盘结果逐项验证来源、case hash、历史、解码及完成标记后复用。
- 不完整批次按原始完整批次重放；原有案例记录只允许逐字段完全相同，不覆盖不同结果。
- 双进程各用 15 个物理核：`0–14,64–78` 和 `15–29,79–93`；控制器使用 `30–31,94–95`。
- 对原始 controller 使用同一个互斥锁。原服务释放 GPU 和锁后才发布加速器结果，避免并发写同一缓存。
- 两个 shard 结束后自动 `resume` 原控制器。原控制器仍负责合并 Train600 moments、核对历史 Tune60、核对 180 例零更新动作/历史/成本，再执行 global32 micro8/4 与新进程恢复检查。

双进程仅用于评测。微批一致性检查和跨进程恢复存在状态依赖，继续按原顺序执行；AR validator 与 canary 原本已经并行，不重复宣称为新增收益。

## 技术验证

`test_stage3_b0_preflight.py` 与原 B0 测试合计 19 passed。覆盖原始 80 个固定批次/960 次评测访问完整覆盖、CPU shard 不交叉、部分批次保留全局索引、缓存拒绝覆盖、零更新成本/步数/动作/历史完全一致及 Train600 moments 来源约束。

实际 GPU 对照使用已完成 Tune 的前两个固定批次。原服务继续推进 baseline，新进程在预留的 8 个物理核上运行新入口零更新。对照逐例与原生记录比较成本、完成步数、完整动作 SHA 和历史 SHA；检查通过的 24 例结果在原服务停止后导入零更新缓存，避免重复计算。

实测两个并发对照批次分别为 221.57、231.59 秒，24/24 例成本、步数、动作 SHA、历史 SHA 完全一致。原生 Tune60 五批平均 159.80 秒，另一原服务进程同期仍持续完成原生评测；以 `2 × 159.80 / mean(221.57, 231.59)` 作排期近似，整体吞吐约为单进程的 1.41 倍。此为不同角色/CPU 分配下的对照估计，最终双 15 核进程的实际吞吐继续记录，不能当作严格同案例的完整系统 speedup 测量。并发时 GPU 利用率近期均值约 61%，显存峰值约 15 GiB。

## 运行证据

产物保存在 r2 的 `preflight_acceleration/`：适配器源码、benchmark_spec.json、benchmark/ 结果、plan.json 和启动记录。正式并行阶段状态及资源记录保存在 r2 `attempts/`，从原服务停止至恢复之间的预留 GPU 时间独立记账。对照测试与原服务共卡期间不额外重复计算整卡预留时间。

19:31 启动服务 `hkbz-b0-preflight-parallel-r2-1789212718.service`，控制器 PID 866961，两个评测进程 PID 867028/867029。复用的原生基线为 Tune60 + validation84，共 144 例；并发对照产生的零更新 Tune24 也已导入。剩余 66 个原始固定批次，每个 shard 33 批。初次宿主机检查发现两个进程均位于 GPU0，整套服务所有线程均在规定 CPU 半区；GPU0 约 14065 MiB、利用率 76%，服务 MemoryCurrent 约 10.8 GiB。动态状态仍读取 r2 `run_status.json`。

剩余评测排期约 2–3 小时，以实际 shard 吞吐更新；此范围不包括之后尚未实测的 global32 微批及恢复准入时间，也不是整轮训练 ETA。

切换后首批两个 shard 已实际完成并发布 canonical 缓存：validation 起点 84/96 的两个 12 例批次分别耗时 220.42/218.34 秒，服务继续推进 validation 最后一批和 Train600 第一批。折合首批合计约 390 例/小时；与之前 Tune60 约 270 例/小时相比，初步吞吐约 1.45 倍。此时剩余 64 批（768 次评测访问），按该短期速度约 1.95 小时，排期保留 2–3 小时范围。证据为 `preflight_acceleration/first_completions.json` 和两个 worker 的 receipts；未发生正式 PPO 更新。

加速率只按实际 batch/资源日志估算。不能将低 GPU 利用率直接解释为可获得 2 倍加速；共享 GPU、CPU 和 IPC 的竞争会影响收益。
