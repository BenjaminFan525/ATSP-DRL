# Stage1 validation120 四基线补评 r3

范围：Multi-PPO-AT、FJSP-DRL-AT、DANIEL-AT、L2D-AT，各3个原validation60 Best checkpoint。共12个checkpoint、1440次案例评测；已完成的P5三种子120例结果经摘要验证后复用。不重新训练、不重新选模。

一个共享validator，GPU0，12个环境worker，CPU 0–127，数值库单线程。案例分组恢复四基线原评测的worker及round：每个worker先运行原有5例，再运行新增5例。完整120例与本轮IGA的数据指纹一致，采用真实标签60 IID/54 OOD-stress/6 OOD-scale。

r2的Multi-PPO seed1/2结果保留作核对，本轮四个基线统一使用原12-worker设置。首先运行曾发生单例偏差的Multi-PPO seed3，然后seed1/2和其余三个基线。

验收要求：120例完整且全部完成、没有超时/循环终止，模型和数据摘要正确，案例worker/round与固定布局一致，原60例makespan逐例复现。个别模型的复现差异单独标记needs_review并保留原始输出，后续独立模型继续；数据/模型契约错误则停止队列。未通过验收的模型不进入正式三种子排名。

状态与产物：state.json和logs/controller.log为实时依据；results/为原始结果，audits/为逐项校验。全部通过后自动生成summary.json、report.md、per_case.csv、analysis/report.md和analysis/summary.json，包含P5与IGA比较以及原有60例/新增60例拆分。

完整validation120包含原选模60例，不是独立test结论。
