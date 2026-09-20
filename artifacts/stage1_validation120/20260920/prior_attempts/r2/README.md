# Stage1 validation120 补评

正式补评运行目录；启动及实时状态以 state.json 为准。

范围：P5、Multi-PPO-AT、FJSP-DRL-AT、DANIEL-AT、L2D-AT，各3个原有 validation60 Best checkpoint。固定原选点，不重新训练、不在validation120上重新选Best。

运行：一个共享 validator，GPU0，60个环境worker，CPU 0–127，数值库单线程。控制服务为 hkbz-stage1-validation120-best-20260920-r2.service；所有15项完成后自动关闭本次创建的validator。

环境：Stage1 plane_pretrain，heuristic资源后端，fixed飞机顺序，joint_pair解码，tau=0.3，f1f2全局特征，progressive-departure-r014-pipeline-v2语义，关闭域随机化。使用与本轮IGA一致的代码和120例数据快照。

完整面板采用真实标签：60 IID / 54 OOD-stress / 6 OOD-scale。env.yaml 的 dataset_manifest 为空，因此仅使用案例metadata，不会再用旧manifest覆盖分布标签。

| 方法 | seed1 Best | seed2 Best | seed3 Best |
|---|---:|---:|---:|
| P5 | E8 | E7 | E8 |
| multi_ppo | E7 | E1 | E7 |
| fjsp_drl | E6 | E8 | E4 |
| daniel | E5 | E8 | E7 |
| l2d | E1 | E4 | E5 |

每项结果必须包含完整120例、全部完成、无cycle/timeout，且原有60例makespan逐例复现；出现偏差时控制器保存审计并停止队列，防止带着口径偏差继续生成总表。

输出：

- manifest.json：checkpoint选择规则、15个模型与原文件/副本摘要、案例身份。
- snapshot_manifest.json：独立代码、输入、模型与历史参考的SHA256。
- state.json、logs/controller.log、logs/validator.log：当前进度和错误。
- results/*.json：每个checkpoint的120例原始结果。
- audits/*.json：完成性、内容指纹、原60例复现检查与统计。
- summary.json：完成方法的3seed均值/标准差，并列已有IGA-180/1800结果。
- report.md、per_case.csv：全套完成后自动生成的最终总表和逐例记录。

完整validation120仍包含原选模用的60例，因此扩展覆盖结果不能当作独立test集结论。

P5 兼容性说明：首轮发现 60/60 历史分数偏差，保留在 r1 目录。历史代码证据表明 P5 使用前一步策略动作作为循环历史，非活动行输出 -1；后续代码改成环境历史并保留非活动行。r2 通过独立 evaluator 适配器按模型摘要恢复各自原评测接口，所有模型参数、物理环境和选模结果不变。两个案例对照记录见 provenance；仍须通过完整原60例复现检查才验收。

全部通过后，独立汇总任务自动生成 analysis/report.md 和 analysis/summary.json，包含完整120例、原有60例、新增60例、尾部指标及P5逐例胜平负。
