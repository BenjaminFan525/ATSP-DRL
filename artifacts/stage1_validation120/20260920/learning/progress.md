# 四基线 validation120 补评状态

已于2026-09-20 22:48:18 +08:00全部跑完，当前没有评测进程运行。四基线12个checkpoint共1440次案例评测全部完成。

复现验收：L2D三种子全部通过；Multi-PPO、FJSP-DRL、DANIEL仍有待核查项。四基线5/12个checkpoint完全通过，7个待核查；复用P5后状态为8/15通过。原60例×12个模型的720个结果中686个精确复现，34个存在差异。

实际结果、三种子汇总、尾部和逐例比较见 [analysis_observed/report.md](analysis_observed/report.md)，机器可读数据见 [analysis_observed/summary.json](analysis_observed/summary.json)。表中明确区分已验收结果与暂定结果，不以历史分数回填新结果。最新原始状态见 [state.json](state.json)。
