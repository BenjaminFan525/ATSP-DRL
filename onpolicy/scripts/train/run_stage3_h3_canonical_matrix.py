#!/usr/bin/env python3
"""Bounded S0/epoch6 canonical-H extension with a frozen epoch1 reference."""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROTOCOL = 'stage3_h3_canonical_checkpoint_matrix_v1'
LEAF = 'onpolicy/scripts/train/run_stage3_h3_canonical_tau.py'


def helper(path):
    spec = importlib.util.spec_from_file_location('canonical_matrix_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_contract(models):
    reference = models['epoch1']
    for name, epoch in [('s0', 0), ('epoch1', 1), ('epoch6', 6)]:
        current = models[name]
        if current['epoch'] != epoch:
            raise ValueError('Checkpoint epoch does not match its matrix label')
        for key in ['parent', 'cases', 'temperatures', 'seed', 'fixed_batch_size',
                    'gpu_uuid', 'cpu_affinity', 'allocator_limit_mib', 'batch_order']:
            if current[key] != reference[key]:
                raise ValueError('Cross-checkpoint evaluation contract differs: '+key)
        # The orchestration script can change; all policy/environment/runtime code
        # must be byte-identical to the previously verified epoch1 implementation.
        policy_files = {k:v for k,v in current['source_files'].items() if k != LEAF}
        expected = {k:v for k,v in reference['source_files'].items() if k != LEAF}
        if policy_files != expected:
            raise ValueError('Canonical policy/environment source differs from epoch1')


def prepare(args):
    h = helper(Path(__file__).with_name('run_stage3_h3_canonical_tau.py'))
    models = {name:h.load_manifest(getattr(args, name)) for name in ['s0','epoch1','epoch6']}
    check_contract(models)
    base = models['epoch1']
    groups, records = h.collect_records(base)
    if (len(groups) != 30 or any(len(v) != 120 for v in records.values())
            or not h.read(Path(base['root'])/'completed.json')['completed']):
        raise ValueError('Epoch1 reuse requires all 360 completed trajectories')
    root = args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        raise FileExistsError('Matrix output is already frozen')
    driver = root/'driver.py'; shutil.copy2(__file__,driver)
    manifest = dict(protocol=PROTOCOL,root=str(root),driver=h.bind(driver),
        helper=h.bind(Path(models['s0']['source_root'])/LEAF),
        models={name:h.bind(getattr(args,name)) for name in models},
        reused_evidence=[h.bind(p) for p in sorted((Path(base['root'])/'groups').glob('*.json'))]
            +[h.bind(Path(base['root'])/'completed.json')],
        execution_order=['s0','epoch6'],new_case_episodes=720,reused_case_episodes=360,
        legacy_admission_case_episodes=24,runtime_max_seconds=46800,
        bootstrap_seed=2026091505,bootstrap_draws=20000,
        gpu_uuid=base['gpu_uuid'],cpu_affinity=base['cpu_affinity'],
        created_unix=time.time(),training_mutation=False,iga_solver_queries=0)
    manifest['manifest_sha256']=h.identity(manifest)
    h.write(root/'manifest.json',manifest)
    h.write(root/'status.json',dict(status='prepared',new_case_episodes=720,
                                   reused_case_episodes=360,heartbeat_unix=time.time()))
    (root/'plan.md').write_text('\n'.join([
        '## Material Passport','','- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run / validate','- Verification Status: PREPARED','',
        'S0、epoch 1、epoch 6：H3/F4/soft、H解码、Validation120、seed 1、原固定12槽位。',
        '新增S0与epoch 6各自三温度0.3/0.03/0.5的独立完整轨迹，共720次；逐案例保存完整动作。',
        '复用已完成且按输入/代码/结果哈希绑定的epoch 1共360条轨迹。',
        '原始分数精确整数总和、Blocking覆盖优先、设备/请求ID字典序并列规则均与已验证版本一致。',
        'S0和epoch 6各先复现原解码的0013–0024共12例，再运行新规则；共24次准入复现。',
        '单个评测模型顺序运行，仅GPU0，CPU22–29/86–93，Torch上限4GiB，服务内存上限16GiB。',
        '不修改主训练快照、参数或评测；不求解IGA。legacy_reference_study仅用于冻结已有参考，不运行旧温度实验。',
        '比较温度间完整动作/历史哈希、步数和makespan；三个checkpoint在同一新规则下比较均值、分布、尾部和案例回退。',
        '最终使用profile分层配对bootstrap 20000次，固定种子2026091505。置信区间只覆盖固定训练种子的案例不确定性。',
        'checkpoint预先固定，不按本次温度结果重新选择；已使用过Validation120，不能声称独立确认或跨种子泛化。',
        '全温度全部报告；数值规则效果与训练收益分开解释。完整轨迹验证确定性H的正温度不变性，不验证训练采样温度。',
        '进程与心跳持续监控，失败保留证据；服务硬超时13小时。', '']))
    print(json.dumps(dict(manifest=str(root/'manifest.json'),new_episodes=720,reused_episodes=360)))


def load(path):
    m=json.loads(Path(path).read_text()); h=helper(m['helper']['path'])
    if m['protocol']!=PROTOCOL or h.identity({k:v for k,v in m.items() if k!='manifest_sha256'})!=m['manifest_sha256']:
        raise ValueError('Matrix manifest changed')
    h.checked(m['helper']);h.checked(m['driver'])
    for record in m['reused_evidence']:h.checked(record)
    models={name:h.load_manifest(h.checked(record)) for name,record in m['models'].items()}
    check_contract(models)
    return m,h,models


def report(path):
    m,h,models=load(path); root=Path(m['root'])
    sys.path.insert(0,models['s0']['source_root'])
    from onpolicy.utils.stage3_h3_frozen import paired_metrics
    from onpolicy.utils.stage3_research import digest_json
    data={};temperature=[]
    fields=['makespan','steps','actions_sha256','history_sha256']
    for name,leaf in models.items():
        groups,records=h.collect_records(leaf)
        complete=h.read(Path(leaf['root'])/'completed.json')
        if not complete['completed'] or len(groups)!=30 or any(len(v)!=120 for v in records.values()):
            raise ValueError('Matrix requires three complete 360-trajectory checkpoints')
        if not complete['weights_unchanged'] or not complete['legacy_reproduction_passed']:
            raise ValueError('Checkpoint lacks completed reproduction/weight checks')
        for tau,rows in records.items():
            mismatches={field:[Path(c).name for c,row in rows.items()
                        if row[field]!=records[.3][c][field]] for field in fields}
            temperature.append(dict(model=name,tau=tau,n=len(rows),
                makespan=sum(r['makespan'] for r in rows.values())/len(rows),mismatches=mismatches))
            if name in m['execution_order']:
                for case,row in rows.items():
                    with gzip.open(Path(leaf['root'])/'trajectories'/f'{Path(case).name}_tau_{h.tag(tau)}.json.gz','rt') as f:
                        trace=json.load(f)
                    if (trace['manifest_sha256']!=leaf['manifest_sha256']
                            or digest_json(trace['actions'])!=row['actions_sha256']
                            or any(trace[field]!=row[field] for field in fields)):
                        raise ValueError('Stored full actions differ from the evaluation record')
        data[name]=records[.3]
    paired={}
    for candidate,baseline in [('epoch1','s0'),('epoch6','s0'),('epoch6','epoch1')]:
        rows=[data[candidate][c['path']] for c in models[candidate]['cases']]
        refs={c:r['makespan'] for c,r in data[baseline].items()}
        paired[f'{candidate}_vs_{baseline}']=paired_metrics(rows,refs,
            seed=m['bootstrap_seed'],draws=m['bootstrap_draws'])
    invariant=all(not any(t['mismatches'].values()) for t in temperature)
    result=dict(completed=True,case_episodes=1080,new_case_episodes=720,reused_case_episodes=360,
        full_actions_verified=720,temperatures=temperature,temperature_invariance_verified=invariant,
        paired=paired,manifest_sha256=m['manifest_sha256'],iga_solver_queries=0)
    h.write(root/'summary.json',result)
    import csv
    with (root/'per_case.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case','profile','distribution','s0','epoch1','epoch6',
                                          'epoch1_gap_s0','epoch6_gap_s0','epoch6_gap_epoch1'])
        writer.writeheader()
        for case in models['s0']['cases']:
            row={name:data[name][case['path']]['makespan'] for name in ['s0','epoch1','epoch6']}
            writer.writerow(dict(case=case['name'],profile=case['profile'],distribution=case['distribution'],
                **row,epoch1_gap_s0=row['epoch1']/row['s0']-1,epoch6_gap_s0=row['epoch6']/row['s0']-1,
                epoch6_gap_epoch1=row['epoch6']/row['epoch1']-1))
    lines=['## Material Passport','','- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run / validate','- Verification Status: '+('VERIFIED' if invariant else 'MISMATCH'),'',
        '完整矩阵1080条；新增720条，复用epoch 1共360条。新轨迹完整动作文件已逐一核对哈希。','',
        '| 模型 | τ | 案例数 | 平均秒 | 与同模型0.3不一致的字段数 |',
        '|---|---:|---:|---:|---:|']
    for t in temperature:
        lines.append(f"| {t['model']} | {t['tau']} | {t['n']} | {t['makespan']:.6f} | {sum(bool(x) for x in t['mismatches'].values())} |")
    lines+=['','| 比较 | gap（负为改善） | 配对95%区间 | 胜/平/负 | 最差10%比值 |',
            '|---|---:|---|---|---:|']
    for name,v in paired.items():
        ci=v['paired_gap_ci95']
        lines.append(f"| {name} | {v['gap_fraction']:+.4%} | [{ci[0]:+.4%}, {ci[1]:+.4%}] | {v['wins']}/{v['ties']}/{v['losses']} | {v['tail_ratio']:.6f} |")
    lines+=['','同一新规则内的温度差异与不同checkpoint的训练差异分别报告；旧主实验仍使用原冻结解码契约。',
        '固定规则下的确定性H动作应与正温度无关；本验证检验完整执行是否符合该性质，不涉及随机采样温度。',
        '区间按profile分层成对bootstrap 20000次，种子2026091505；为探索性、未作多重比较校正，不含训练种子不确定性。',
        '11项解释自检：无显著性夸大、报告效果量、配对案例、分布/尾部可查、无小样本外推、'
        '承认多次比较、固定checkpoint选择限制、全温度报告、未按结果删例、无跨split相减、无因果/跨种子推广。','']
    (root/'report.md').write_text('\n'.join(lines))
    return result


def run(path):
    m,h,models=load(path);root=Path(m['root'])
    if (root/'run_started.json').exists():
        raise FileExistsError('This frozen matrix has already been started; no implicit retry')
    start=time.monotonic()
    h.write(root/'run_started.json',dict(pid=os.getpid(),started_unix=time.time(),manifest_sha256=m['manifest_sha256']))
    for name in m['execution_order']:
        leaf=models[name]
        command=[sys.executable,'-B','-u',str(Path(leaf['source_root'])/LEAF),'run',m['models'][name]['path']]
        with (root/(name+'.log')).open('a') as log:
            child=subprocess.Popen(command,cwd=leaf['source_root'],stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                state=Path(leaf['root'])/'status.json'
                progress=h.read(state) if state.exists() else {}
                h.write(root/'status.json',dict(status='running',current_model=name,child_pid=child.pid,
                    heartbeat_unix=time.time(),elapsed_seconds=time.monotonic()-start,progress=progress,
                    manifest_sha256=m['manifest_sha256']))
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
            if child.returncode:
                h.write(root/'status.json',dict(status='failed',current_model=name,exit_code=child.returncode,
                                               heartbeat_unix=time.time()))
                raise RuntimeError(f'{name} evaluation exited {child.returncode}; see {name}.log')
    result=report(path)
    h.write(root/'completed.json',dict(**result,seconds=time.monotonic()-start))
    h.write(root/'status.json',dict(status='completed',heartbeat_unix=time.time(),
            temperature_invariance_verified=result['temperature_invariance_verified'],case_episodes=1080))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare')
    for name in ['s0','epoch1','epoch6','output']:q.add_argument('--'+name,type=Path,required=True)
    for name in ['run','report']:sub.add_parser(name).add_argument('manifest',type=Path)
    args=p.parse_args()
    if args.mode=='prepare':prepare(args)
    elif args.mode=='run':run(args.manifest)
    else:print(json.dumps(report(args.manifest),ensure_ascii=False))


if __name__=='__main__':main()
