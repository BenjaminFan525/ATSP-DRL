#!/usr/bin/env python3
"""Summarize saved temperature-divergence evidence without loading a GPU."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from fractions import Fraction
import gzip
import json
from pathlib import Path
import pickle
import sys

import numpy as np


def numerical_probes(frames, source_root):
    from scipy.optimize import linear_sum_assignment
    from scipy.special import logsumexp
    path=Path(source_root)/'onpolicy/utils/stage2_matching.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='solve_resource_matching')
    namespace=dict(np=np,linear_sum_assignment=linear_sum_assignment)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    solve=namespace['solve_resource_matching']
    result=dict(full_trajectory_replayed=False,float64_transforms=[],row_order_probes=[])
    for label in ['reference','candidate']:
        m=frames[label].get('matching')
        if m is None:continue
        z=m['raw'].astype(float)
        deployed=solve(m['scores'],m['lookahead'])
        if not np.array_equal(deployed,m['assignment']):
            raise ValueError('Saved deployed scores do not reproduce the observed matching')
        raw_assignment=solve(z,m['lookahead'])
        def total(values, assignment):
            return sum((Fraction.from_float(float(values[i,j])) for i,j in enumerate(assignment)),Fraction())
        raw_total=total(z,raw_assignment)
        for tau in [.03,.3,.5]:
            lp=z/tau-logsumexp(z/tau,axis=1,keepdims=True)
            a=solve(lp,m['lookahead'])
            gap=total(z,a)-raw_total
            result['float64_transforms'].append(dict(raw_source=label,tau=tau,assignment=a.tolist(),
                raw_gap_vs_unscaled_assignment=float(gap),raw_gap_exact=str(gap),
                blocking_count=int(((a>0)&~m['lookahead'][a]).sum())))
        result.setdefault('unscaled_raw_assignment',{})[label]=raw_assignment.tolist()
        n=len(z)
        permutations=[np.arange(n),np.arange(n)[::-1]]
        other=frames['candidate' if label=='reference' else 'reference'].get('matching')
        if other is not None and np.array_equal(m['rows'],other['rows']):
            changed=np.flatnonzero(m['assignment']!=other['assignment'])
            if len(changed)>=2:
                p=np.arange(n);p[changed[:2]]=p[changed[:2]][::-1];permutations.append(p)
        for p in permutations:
            a=solve(m['scores'][p],m['lookahead']);restored=np.empty_like(a);restored[p]=a
            result['row_order_probes'].append(dict(score_source=label,physical_row_order=m['rows'][p].tolist(),
                assignment_in_original_order=restored.tolist(),
                changed_physical_assignment=not np.array_equal(restored,m['assignment']),
                logprob_gap_vs_observed=float(total(m['scores'],restored)-total(m['scores'],m['assignment']))))
    return result


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();manifest=read(root/'manifest.json');study=read(manifest['study_path'])
    proofs=[read(p) for p in sorted((root/'reproduction').glob('*.json'))]
    findings=[read(p) for p in sorted((root/'findings').glob('*.json'))]
    completed=read(root/'completed.json') if (root/'completed.json').exists() else None
    enriched=[]
    for finding in findings:
        finding=dict(finding)
        path=root/'findings'/f"{finding['case']}_tau_{str(finding['tau']).replace('.','p')}.pkl.gz"
        with gzip.open(path,'rb') as f:frames=pickle.load(f)
        finding['numerical_probes']=numerical_probes(frames,manifest['source_root'])
        physical=root/('physical_'+path.name.replace('.pkl.gz','.json'))
        if physical.exists():finding['physical']=read(physical)
        detail=[]
        ref=frames['reference'].get('matching');alt=frames['candidate'].get('matching')
        if ref is not None and alt is not None and finding.get('matching',{}).get('comparable'):
            A,B=ref['assignment'],alt['assignment']
            for label,row,tau in [('reference',ref,.3),('candidate',alt,finding['tau'])]:
                from scipy.special import logsumexp
                stats=finding['matching']['matrices'][label]
                if 'raw_score_reference_minus_candidate' not in stats:continue
                if not np.array_equal(np.isfinite(row['raw']),np.isfinite(row['scores'])):
                    raise ValueError('Raw-score observer mask does not match deployed scores')
                z=row['raw'].astype(float)/tau
                expected=z-logsumexp(z,axis=1,keepdims=True)
                finite=np.isfinite(expected)
                residual=float(np.max(np.abs(expected[finite]-row['scores'][finite])))
                if residual>1e-4:raise ValueError('Raw-score observer does not match deployed row probabilities')
                stats['observer_score_alignment_max_abs']=residual
                stats['normalization_gap_residual']=stats['logprob_reference_minus_candidate']-stats['raw_score_reference_minus_candidate']/tau
                stats['blocking_equal']=stats['blocking_reference']==stats['blocking_candidate']
            for j,agent in enumerate(ref['rows']):
                if A[j]==B[j]:continue
                detail.append(dict(agent=int(agent),request_reference=int(A[j]),request_candidate=int(B[j]),
                    raw_in_reference_A=float(ref['raw'][j,A[j]]),raw_in_reference_B=float(ref['raw'][j,B[j]]),
                    raw_in_candidate_A=float(alt['raw'][j,A[j]]),raw_in_candidate_B=float(alt['raw'][j,B[j]]),
                    lp_in_reference_A=float(ref['scores'][j,A[j]]),lp_in_reference_B=float(ref['scores'][j,B[j]]),
                    lp_in_candidate_A=float(alt['scores'][j,A[j]]),lp_in_candidate_B=float(alt['scores'][j,B[j]])))
        finding['changed_edge_scores']=detail
        if finding['plane_changes']:
            finding['interpretation']='plane action divergence; inspect saved raw joint scores'
        elif finding.get('matching',{}).get('comparable'):
            matrix=finding['matching'];stats=list(matrix['matrices'].values())
            raw_tied=all(x.get('raw_score_reference_minus_candidate')==0 and x.get('blocking_equal') for x in stats)
            if matrix['raw_exactly_equal'] and raw_tied:
                finding['interpretation']='identical raw resource scores; the two selected matchings have exactly tied raw objectives'
            elif matrix['raw_exactly_equal'] and all(x.get('blocking_equal') for x in stats):
                finding['interpretation']='identical raw resource scores; temperature-dependent rounding changes the matching preference'
            elif raw_tied:
                finding['interpretation']='two selected matchings have tied raw objectives; upstream raw scores also differ'
            else:
                finding['interpretation']='near-tie or upstream score difference; inspect exact raw and deployed margins'
        else:
            finding['interpretation']='matching supports differ; inspect saved first-divergence frames'
        enriched.append(finding)
    expected_findings={(name,tau) for name in manifest['targets'] for tau in manifest['temperatures'] if tau!=.3}
    expected_groups={(i,tau) for i in range(len(manifest['batches'])) for tau in manifest['temperatures']}
    verified=(completed and completed['weights_unchanged']
        and {(p['batch_index'],p['tau']) for p in proofs}==expected_groups
        and len(proofs)==len(expected_groups) and all(p['passed'] and p['weights_unchanged'] for p in proofs)
        and {(d['case'],d['tau']) for d in enriched}==expected_findings)
    outcomes={(Path(row['case_id']).name,p['tau']):row['makespan'] for p in proofs for row in p['rows']}
    report=dict(verification_status='VERIFIED' if verified else 'IN_PROGRESS',
        manifest=manifest,completion=completed,verified_groups=sum(p['passed'] for p in proofs),
        verified_case_episodes=sum(p['cases'] for p in proofs if p['passed']),findings=enriched,
        limitation='First-divergence localization does not alone establish the causal contribution of that action to final makespan.')
    (root/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: validate / reproducibility',
        '- Origin Date: 2026-09-16',
        '- Verification Status: '+report['verification_status'],
        '- Source: fixed epoch 1 checkpoint; original H3/F4/soft fixed12 GPU0 evaluation',
        '', '## 回放范围与复现', '',
        '目标：case_0020、case_0052、case_0055、case_0114；温度：0.3、0.03、0.5。',
        '保留原案例顺序和固定12槽批次，包括已完成槽；仅添加原始分数观察器，不改变策略计算。',
        f"完整复现组数：{report['verified_groups']}/9；逐案例复现记录：{report['verified_case_episodes']}/108。",
        '每例同时核对完工时间、步数、完整动作哈希和历史哈希；模型权重在每组结束后核对。',
        '', '## 物理动作与原评测结果', '',
        '以下将候选温度相对0.3的首次分歧映射到设备编号；物理映射通过CPU环境逐步回放并逐步核对输入图哈希。',
        '“设备A→设备B”表示同一请求改由B承接，A改为等待。决策序号从1开始；时间为环境双精度仿真时间。',
        '', '| 案例 | 候选τ | 首次决策 | 时间/秒 | 设备A→设备B | 请求/工序/飞机/位置 | 完工时间0.3→候选/秒 |',
        '|---|---:|---:|---:|---|---|---|']
    for d in enriched:
        p=d.get('physical')
        if p is None:continue
        devices=p['devices'];edges=d['changed_edge_scores']
        former=next((e for e in edges if e['request_reference']>0 and e['request_candidate']==0),None)
        latter=next((e for e in edges if e['request_candidate']>0 and e['request_reference']==0),None)
        if former is None or latter is None:continue
        code_a=devices[str(former['agent'])]['code'];code_b=devices[str(latter['agent'])]['code']
        request=p['requests'][str(former['request_reference'])]
        job=f"{request['id']}/{request['job_code']}/{request['plane_id']}/{request['site_code']}"
        before=outcomes[(d['case'],.3)];after=outcomes[(d['case'],d['tau'])]
        lines.append(f"| {d['case']} | {d['tau']} | {d['decision_number']} | {p['env_time']:.6f} | {code_a}→{code_b} | {job} | {before:.6f}→{after:.6f} |")
    lines += [
        '', '## 首次有效动作分歧', '',
        '| 案例 | 候选τ | 第几次决策 | 仿真时刻/秒 | 变化agent | 输入图相同 | 隐状态相同 | 资源原始分数相同 |',
        '|---|---:|---:|---:|---|---|---|---|']
    for d in enriched:
        lines.append(f"| {d['case']} | {d['tau']} | {d['decision_number']} | {d['time_reference']:.6f} | {d['active_changed_agents']} | {d['graph_identical']} | {d['recurrent_input_identical']} | {(d.get('matching') or {}).get('raw_exactly_equal')} |")
    for d in enriched:
        if not d['recurrent_input_identical']:
            lines.append(f"\n{d['case']}、τ={d['tau']}的RNN输入存在最大{d['recurrent_input_max_abs']:.12g}差异；输入图、此前动作及本次实际资源原始分数仍相同。")
    lines+=['','## 匹配分差','',
        'A为τ=0.3实际选择，B为候选温度实际选择。所有分差均为A−B，正值偏向A。',
        'raw为温度缩放之前的指针分数；LP为实际float32对数概率；部署分数包含float64的Blocking奖励。',
        '使用浮点数的精确有理数表示求和，避免汇总大额Blocking奖励时掩盖微小差值。',
        '', '| 案例 | 候选τ | 矩阵来源 | raw差 | LP差 | 部署差 | 两方案Blocking数 | 选中−最佳强制替代 |',
        '|---|---:|---|---:|---:|---:|---|---:|']
    for d in enriched:
        mat=d.get('matching') or {}
        for label,s in mat.get('matrices',{}).items():
            if 'raw_score_reference_minus_candidate' not in s:continue
            gap=s['chosen_minus_best_forced_alternative'];gap='N/A' if gap is None else f'{gap:.12g}'
            lines.append(f"| {d['case']} | {d['tau']} | {label} | {s['raw_score_reference_minus_candidate']:.12g} | {s['logprob_reference_minus_candidate']:.12g} | {s['deployed_reference_minus_candidate']:.12g} | {s['blocking_reference']}/{s['blocking_candidate']} | {gap} |")
    lines+=['', '“选中−最佳强制替代”通过逐一禁止所选边并重新求解获得；0表明找到等分的不同匹配，负值说明浮点求解与精确求和存在细微差异。',
        '此处最佳替代是部署浮点求解器返回的候选，不是高精度求解器提供的全局证书。',
        '', '## 固定原始分数的双精度对照', '',
        '只对已保存矩阵重算，没有重新运行完整策略或环境轨迹。A/B之外可能仍有原始分数并列的匹配。',
        '', '| 案例 | 候选τ | 固定0.3原始分数，FP64重算三档是否同解 | 未缩放原始分数解 |',
        '|---|---:|---|---|']
    for d in enriched:
        probe=d['numerical_probes']
        assignments=[x['assignment'] for x in probe['float64_transforms'] if x['raw_source']=='reference']
        same=bool(assignments) and all(x==assignments[0] for x in assignments)
        lines.append(f"| {d['case']} | {d['tau']} | {same} | {probe.get('unscaled_raw_assignment',{}).get('reference')} |")
    lines += ['', '## 并列与顺序探针', '',
        '用保存的部署分数重算，全部复现原匹配；再仅交换或反转设备行顺序，并将结果还原为原设备编号。',
        '', '| 案例 | 候选τ | 0.3分数受行顺序影响 | 候选分数受行顺序影响 | FP64三档解的原始总分均与未缩放解相等 |',
        '|---|---:|---|---|---|']
    for d in enriched:
        probe=d['numerical_probes']
        changed={label:any(x['changed_physical_assignment'] for x in probe['row_order_probes'] if x['score_source']==label)
                 for label in ['reference','candidate']}
        equal=all(x['raw_gap_exact']=='0' for x in probe['float64_transforms'])
        lines.append(f"| {d['case']} | {d['tau']} | {changed['reference']} | {changed['candidate']} | {equal} |")
    lines+=[
        '', '## 解释边界', '',
        '先检查状态、历史及原始分数是否一致，再归因于温度归一化或并列处理。原始分差为0、LP分差非0可直接证明数值计算打破了原始并列。',
        '匹配函数没有独立于分数表示的显式二级排序目标；当前行列顺序与SciPy实现共同决定并列选择，不能概括为总是选择编号最小的设备。',
        '双精度能减少舍入误差，但原始目标真正并列时仍需明确的二级规则；不能把FP64单独视为已验证的完整修复。',
        '首次动作差异可能只是可交换资源的置换；定位首次分歧不等于已经证明它单独造成最终完工时间差异。',
        '实际代码和模型不变，未修改训练温度、选模规则或并列解码规则，未调用IGA。',
        '', '逐边分数及精确差值见 analysis.json；原始逐步记录见 traces/；首次分歧的成对状态见 findings/*.pkl.gz。']
    (root/'analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['verification_status'],groups=report['verified_groups'],findings=len(enriched),
                         interpretations=dict(Counter(d['interpretation'] for d in enriched))),ensure_ascii=False))


if __name__=='__main__':main()
