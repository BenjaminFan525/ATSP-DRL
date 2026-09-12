# Stage3：受控探索到 greedy 改善

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-06
- Verification Status: development tests and live update/resume checks passed; production state in suite/status.json
- Version Label: local_exploration_v1
- 用户授权：实现上一轮计划并启动；只使用半数 GPU、CPU。旧结果、原准入失败记录不改。
- 数据：内部数据，不上传。Probe/Tune/Gate 已有历史暴露；不打开 Finalblind 结果。

## 冻结目标与对照

C0 为 Stage2 T2 Epoch3，SHA256 `41cfc782c1ff908bb527c0219b75b73a1421d48a2dafa02d449015abbe5ab030`。
不改网络维度、环境目标、合法掩码、H3/F4、hard reservation 或资源请求语义。
主指标始终是新 checkpoint 单次 greedy 相对 C0 greedy 的案例等权平均成本改善。
保留 C0 后择优、best-of-N 和训练损失下降均不算主目标达标。

## 源码与数据

执行快照基于已验证的 sampling audit r2，仅覆盖 `OVERLAYS` 中声明的新研究代码。
共享工作区的其他改动只提醒，不会热更新或终止本研究；C0、实际快照、协议和数据仍校验。
沿用 Diag32 / Fit16 / Probe64 / Pilot120 / Tune60；Fit 是 Diag 子集，其余内容去重。
主训练仅使用 Pilot120；探索诊断和 Fit 的权重、教师不作为 RL 初始化或主精英池输入。

## 阶段与准入

1. Contract：R/J 两路径各 K=8 实际更新；复现审计动作/成本；更新前 recurrent logp 误差 <=0.002。
   冻结参数不变；初始概率重放、两档 LR、更新后每角色 KL、BC 路径、完整 optimizer/RNG 恢复都验收。
   GPU 峰值 reserved <18GiB。R 路径完整复现 C0 Tune60，成本误差 <=0.1秒。
   两组共同选择 `[5e-6, 1e-6]` 中第一个数值验收通过的 LR，不用 Tune 收益选 LR。
2. Sampling：R=仅资源随机 tau=.3；J=全角色 tau=.03。补齐 Diag32 各32样本，Probe64 各8样本。
   复用原8例两模式共512条完整、哈希校验的归档轨迹，新增2560条。
   每模式准入：Diag >=8/32 例出现 >1% 胜出；Diag best8 平均改善 >=1%；Probe best8 改善 >0。
   这只证明探索信号，不能证明 RL 改善。某模式失败只禁止对应 pilot。
   新轨迹另记实际长占作业完成状态被撤销的次数、重复转运、无不可逆进展转运及资源等待；
   这些是诊断而非新增奖励。原归档未记录的细项保持缺失，不填零、不伪造可比统计。
3. Fit：固定 Fit16，自身探索每例一个胜出教师，无赢家案例仍保留评估。冻结 shared encoder，按模式冻结 plane。
   最多10遍、BC基础LR=1e-5；测教师质量、初末NLL、闭环Fit和Probe。仅给混合臂准入，不拦纯PPO。
   混合臂要求教师平均潜力 >=0.5%，Fit 恢复 >=50% 潜力，Probe 总体不退步并满足风险门槛。
4. 分叉：原8例（含0239）×4位置×8分支=256。每位置2资源、2飞机、2单步联合、2四步联合。
   同C0前缀，窗口外greedy；保存角色、窗口、种子、动作、转运差和终局成本。无改善不自动删除案例。
   角色变化会影响后续合法掩码，不能把它声称为孤立单agent Q；本诊断不是泛化测试。
5. Pilot：只启动通过相应门槛的 R_PPO/J_PPO/R_PPO_E/J_PPO_E；全部从C0新建，optimizer重置。
   完成pilot后停止审阅，不自动确认，不自动扩大网络或数据，不自动读Finalblind。

## PPO 与辅助回放

纯PPO共同优势为 `0.01*(C0(case)-sample_cost)`，不丢负优势，不做案例内标准化。
K=8；120例×2遍=1920新轨迹/臂；固定配对seed=2026090611，各遍哈希确定的案例顺序。
epoch=2, clip=.2, TBPTT=8, gamma=1, critic LR=1e-4, grad clip=1。
shared encoder冻结；R另冻结飞机分支且飞机不进入actor loss/KL/BC。critic参数独立。
训练角色温度写入checkpoint；采样和forced replay使用相同原始可微head，不改全局selector或Top-k。
当前policy版本、温度和随机角色不匹配，或greedy/forced轨迹，都禁止作为新PPO批次。
每步更新后重建RNN历史测KL；超过.02不追加epoch，任一训练角色KL超过.04使本臂失败留档，不静默回滚。
资源分支冻结保证同输入下飞机输出不变，不保证环境变动后的整条飞机轨迹不变。

E臂每4组至多一次独立BC，LR=1e-6；只收主训练自身得到且可重放的C0胜出轨迹，每例最多一个。
整个精英池按案例使用次数公平轮转，重建当前hidden，绝不当作on-policy PPO。额外重放/更新量单列。
拟合checkpoint禁止用作RL初始化。若只有E臂通过，后续确认须含纯PPO及必要BC-only归因对照。

## 验证、监控与停止

每240新轨迹保存不可覆盖checkpoint；每480提交Tune60 greedy，固定评估点480/960/1440/1920。
共享validator只读Tune，单GPU单模型，身份包含checkpoint/case/code/contract/训练探索/评估协议。
每臂最多2个未完成验证请求，背压等待写明状态。缓存持久化、按身份区分，不跨模型热覆盖。
最终1440和1920两次均改善 >=1%，OOD-stress/stress_joint退步 <=0.5%、tail10%退步 <=1%才过pilot。
两次评估中成本退步超过5%的案例比例也必须 <=5%。
连续两次总体退步 >=2%提前失败；未达1%本身不触发追加训练或提前终止。
30秒心跳，实际进度10分钟无变化告警；无自动重试。环境IPC300秒，轨迹最大4000步。
contract6小时、sampling36小时、fit36小时、分叉12小时、validator单请求6小时。
pilot按canary group时间×240×2.5、下限24小时/上限120小时；整个服务硬上限10天。
错误/预定硬超时只清理本suite拥有的进程组，旧产物不覆盖，其他实验不动。

## 半数资源与操作

GPU0–3；CPU0–31,64–95（32物理核/64逻辑线程），每lane8环境，BLAS/Torch单线程。
原始v1快照：采样4lane；训练GPU0–2，GPU3唯一validator，第四臂排队，不占第五张卡。
2026-09-07资源调度修订：新准备的manifest使用`pilot_validator.mode=colocate_if_four`。
恰好四臂获准时，GPU0–3各一训练臂，GPU3同时运行唯一共享异步validator；无需第四臂排队。
GPU3对应CPU拆分为训练`24-29,88-93`（6物理核）及验证`30-31,94-95`（2物理核）。
同卡trainer/validator的PyTorch allocator上限分别60%/20%，合计仍80%；这是allocator额度，
不是驱动级硬显存分区，预留20%覆盖CUDA上下文等额外开销。总CPU、GPU、主存预算不增加。
K=8、验证batch=8、种子、更新次数、评估点、准入和停止门槛均不改变。
只有1–3臂获准时仍使用专卡validator；旧manifest不含此配置时保持原调度，不静默修改运行快照。
实际放置写入`pilot_resources.json`、各worker的`commands/*.json`及心跳。
进程组MemoryHigh32GiB/MemoryMax40GiB、单GPUallocator80%；canary内存不通过不启动长跑。

```bash
PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
$PYTHON onpolicy/scripts/train/run_stage3_local_exploration.py prepare --output /ABSOLUTE/NEW_SUITE
bash onpolicy/scripts/train/launch_stage3_local_exploration_half.sh /ABSOLUTE/NEW_SUITE
```

完整续跑是显式动作：worker支持 `--phase train --arm ARM --resume-checkpoint CHECKPOINT --attempt-id NEW_ID`。
只接受同协议/同arm/同案例顺序/同温度的正式完整组checkpoint，恢复optimizer、CPU/CUDA/NumPy/Python RNG、
policy版本、group、精英池/使用计数与验证请求；输出到新的attempt，不覆盖旧更新。
需要共享validator服务可用；不自动恢复中断的验证任务，不把failure_state或BC诊断文件当作可续跑训练点。

核心产物：manifest、commands、logs、contract、sampling、exploration_gate、fit、counterfactual、
pilot_admission、pilot四臂、validator、report。报告区分成功启动、实现验收、研究准入和算法达标。
后续三个新seed、每seed4800轨迹及Finalblind授权另行决定，本入口没有自动执行该阶段的命令。

## 开发验收记录

`result/hkbz_train_logs/stage3_local_dev_20260906_r1/` 为独立开发验收，不是正式训练结果。
GPU0/1、各K=2：R/J均精确复现原审计动作和成本；初始logp最大误差分别为
1.79e-6、1.31e-5。两档LR真实反传及辅助BC通过；完整恢复后的下一次更新模型最大误差
均1.49e-8，optimizer状态在声明的FP32容差内一致，CPU/CUDA/NumPy/Python随机状态完全一致。
此证据不代替正式服务的K=8、完整Tune60契约检查，也不表示greedy改善目标已达到。
最终开发回归：81 tests passed（含新角色训练/缓存/调度集成、旧sampling、diagnostic和source-relative检查）。
Python编译、launcher shell语法通过。未修改与本任务无关的基线源码或现有工作区差异。

## 2026-09-07 共卡调度修订验收

### Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-07
- Verification Status: IMPLEMENTED / UNIT_TESTED; live GPU colocation not yet verified; running v1 snapshot unchanged
- Version Label: local_resource_colocation_v1

资源调度、实际进程启动参数、同卡占用保护、CPU分区、显存额度和worker实际资源校验，
连同local_exploration、sampling_audit、diagnostic_protocol、source_relative_rl回归共96 tests passed。
Python编译和launcher shell语法通过。保留研究算法、K=8、验证batch=8与全部准入门槛。
截至本次修改，`stage3_local_exploration_half_20260906_r1`的运行快照未修改，诊断继续执行。
本轮启用须另行确认阶段边界切换；该记录不表示共卡实机验收或当前服务切换已经完成。
