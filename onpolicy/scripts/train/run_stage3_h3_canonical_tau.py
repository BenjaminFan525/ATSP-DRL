#!/usr/bin/env python3
"""Frozen-checkpoint Validation120 full-trajectory canonical H temperature study."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROTOCOL = 'stage3_h3_canonical_h_temperature_v1'
TAUS = [.3, .03, .5]
TARGETS = ['case_0020', 'case_0052', 'case_0055', 'case_0114']
OVERLAYS = ['onpolicy/algorithms/utils/ptr_actor.py',
            'onpolicy/algorithms/gnn_mappo/algorithm/gnn_actor_critic.py',
            'onpolicy/utils/stage3_canonical_h.py',
            'onpolicy/scripts/train/run_stage3_h3_canonical_tau.py']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def checked(record):
    if sha(record['path']) != record['sha256']:
        raise ValueError('Bound input changed: '+record['path'])
    return Path(record['path'])


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)+'\n')
    temporary.replace(path)


def tag(tau):
    return str(tau).replace('.', 'p')


def load_original(study):
    s = read(study)
    spec = importlib.util.spec_from_file_location('original_temperature_driver', checked(s['driver']))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.load_study(study)


def prepare(args):
    original, parent = load_original(args.study)
    model_name = getattr(args, 'model', 'checkpoint')
    selected = original['models'][model_name]
    references = {}
    for tau in TAUS:
        if tau == .3:
            references[tag(tau)] = selected['reference']
            continue
        result = Path(original['root'])/'results'/f'{model_name}_tau_{tag(tau)}.json'
        if result.exists():
            value = read(result)
            if (value['checkpoint'] != selected['checkpoint'] or value['model'] != model_name
                    or value['tau'] != tau or not value['weights_unchanged']):
                raise ValueError('Legacy result belongs to another model or temperature')
            references[tag(tau)] = bind(result)
    # An existing complete .3 evaluation is sufficient to admit a new checkpoint.
    # Missing legacy temperatures are not fabricated or silently solved again.
    for tau, reference in references.items():
        validate_rows(read(checked(reference))['rows'], original['cases'], float(tau.replace('p', '.')))
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    source = root/'source'
    shutil.copytree(original['source_root'], source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    workspace = Path(__file__).resolve().parents[3]
    for name in OVERLAYS:
        target = source/name; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.chmod(target.stat().st_mode | 0o200)
        shutil.copy2(workspace/name, target)
    files = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
    manifest = dict(protocol=PROTOCOL, root=str(root), source_root=str(source), source_files=files,
        original_study=bind(args.study), original_source_root=original['source_root'],
        parent=original['parent'], checkpoint=selected['checkpoint'], model_name=model_name,
        epoch=selected['epoch'], policy_updates=selected['policy_updates'],
        temperatures=TAUS, cases=original['cases'], batch_order=[1, 4, 9, 0, 2, 3, 5, 6, 7, 8],
        target_cases=TARGETS, gpu_uuid=original['gpu_uuid'], cpu_affinity=original['cpu_affinity'],
        allocator_limit_mib=4096, fixed_batch_size=12, total_case_episodes=360,
        legacy_admission_batch=1, runtime_max_seconds=21600, seed=1,
        original_results=references, save_full_trajectories=getattr(args, 'all_trajectories', False),
        training_mutation=False, iga_solver_queries=0, created_unix=time.time())
    manifest['manifest_sha256'] = identity(manifest)
    write(root/'manifest.json', manifest)
    (root/'plan.md').write_text('\n'.join([
        '## Material Passport', '', '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run / validate', '- Verification Status: PLANNED', '',
        f'固定{model_name}（epoch {selected["epoch"]}），H3/F4/soft，Validation120，seed 1，GPU0，原固定12槽批次及完成槽保留规则。',
        '先完整复现原解码的0013–0024共12例，然后独立运行三档温度0.3、0.03、0.5的全部120例。',
        '优先批次0013–0024、0049–0060、0109–0120，包含四个首次分歧目标案例。',
        '新解码按Blocking覆盖数、二进制原始分数精确总和、设备与请求编号顺序依次优化；等待排在真实请求之后。',
        '飞机采用原始联合分数argmax，并列按工序/位置展开索引；概率采用FP64 log-softmax，保留实际温度。',
        '全程独立环境轨迹，不复用其他温度的动作或新解码结果。',
        '成功标准：360次运行完成；比较完工时间、完整动作哈希、历史哈希及步数；逐组确认权重未变。',
        '温度效果和解码规则效果分别报告。与旧结果存在选模及同验证集探索限制，不据此自动更换主实验评测规则。',
        '不训练、不修改正在运行的源快照、不重新求解IGA。', '']))
    print(json.dumps(dict(manifest=str(root/'manifest.json'), source=str(source), groups=30,
                         cases=360, legacy_admission_cases=12), ensure_ascii=False))


def load_manifest(path):
    m = read(path)
    if m['protocol'] != PROTOCOL or identity({k:v for k,v in m.items() if k!='manifest_sha256'}) != m['manifest_sha256']:
        raise ValueError('Canonical study manifest identity changed')
    checked(m['original_study']); checked(m['parent']); checked(m['checkpoint'])
    for record in m['original_results'].values():
        checked(record)
    for name, digest in m['source_files'].items():
        if sha(Path(m['source_root'])/name) != digest:
            raise ValueError('Frozen canonical source changed: '+name)
    return m


def validate_rows(rows, cases, tau):
    if [r['case_id'] for r in rows] != [c['path'] for c in cases]:
        raise ValueError('Full trajectories must follow the frozen case order')
    for row, case in zip(rows, cases):
        if (row['tau'] != tau or row['decoder'] != 'H' or row['seed'] != 1
                or not row['behavior_deterministic'] or not row['completed']
                or row.get('cycle_terminated') or row.get('forced_replay')
                or row['profile'] != case['profile'] or row['distribution'] != case['distribution']):
            raise ValueError('Full-trajectory evaluation contract mismatch: '+case['path'])


def collect_records(m):
    """Reject foreign, duplicated or incomplete groups before summarizing them."""
    groups = [read(p) for p in sorted((Path(m['root'])/'groups').glob('*.json'))]
    records = {t:{} for t in m['temperatures']}
    seen = set()
    for group in groups:
        key = (group['batch_index'], group['tau'])
        if (key in seen or group['batch_index'] not in m['batch_order']
                or group['tau'] not in records or not group['weights_unchanged']
                or group['manifest_sha256'] != m['manifest_sha256']):
            raise ValueError('Foreign or duplicate trajectory group')
        seen.add(key)
        start = group['batch_index']*m['fixed_batch_size']
        validate_rows(group['rows'], m['cases'][start:start+m['fixed_batch_size']], group['tau'])
        for row in group['rows']:
            if row['case_id'] in records[group['tau']]:
                raise ValueError('Duplicate completed trajectory')
            records[group['tau']][row['case_id']] = row
    return groups, records


def report(m):
    import numpy as np
    root = Path(m['root'])
    groups, records = collect_records(m)
    old = {t:{r['case_id']:r for r in read(checked(m['original_results'][tag(t)]))['rows']}
           for t in m['temperatures'] if tag(t) in m['original_results']}
    comparisons = []
    for tau in m['temperatures']:
        paired = [c['path'] for c in m['cases'] if c['path'] in records[tau] and c['path'] in records[.3]]
        if not paired:
            continue
        a = np.array([records[tau][c]['makespan'] for c in paired])
        b = np.array([records[.3][c]['makespan'] for c in paired])
        legacy = np.array([old[.3][c]['makespan'] for c in paired])
        comparisons.append(dict(tau=tau, n=len(paired), mean_makespan=float(a.mean()),
            gap_vs_canonical_tau03=float(a.mean()/b.mean()-1),
            gap_vs_legacy_tau03=float(a.mean()/legacy.mean()-1),
            wins_vs_legacy=int((a < legacy-1e-6).sum()), ties_vs_legacy=int((np.abs(a-legacy)<=1e-6).sum()),
            losses_vs_legacy=int((a > legacy+1e-6).sum()),
            temperature_mismatches={field:[Path(c).name for c in paired if records[tau][c][field] != records[.3][c][field]]
                for field in ['makespan', 'steps', 'actions_sha256', 'history_sha256']}))
    complete = (len(groups)==30 and all(len(records[t])==120 for t in m['temperatures'])
                and all(g['weights_unchanged'] for g in groups))
    outcome = dict(completed=complete, completed_groups=len(groups), planned_groups=30,
        completed_case_episodes=sum(map(len,records.values())), comparisons=comparisons,
        manifest_sha256=m['manifest_sha256'])
    write(root/'summary.json', outcome)
    columns = ['case', 'profile']+[f'{kind}_{tag(t)}' for kind in ['legacy_makespan','canonical_makespan','canonical_actions_sha256'] for t in m['temperatures']]
    with (root/'per_case.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for c in m['cases']:
            row=dict(case=c['name'], profile=c['profile'])
            for t in m['temperatures']:
                if t in old:
                    row[f'legacy_makespan_{tag(t)}']=old[t][c['path']]['makespan']
                if c['path'] in records[t]:
                    row[f'canonical_makespan_{tag(t)}']=records[t][c['path']]['makespan']
                    row[f'canonical_actions_sha256_{tag(t)}']=records[t][c['path']]['actions_sha256']
            writer.writerow(row)
    lines=['## Material Passport', '', '- Origin Skill: academic-research-suite / experiment-agent',
           '- Origin Mode: run / validate', '- Verification Status: '+('COMPLETED' if complete else 'RUNNING'), '',
           f"完整轨迹：{outcome['completed_case_episodes']}/360；批次：{len(groups)}/30。", '',
           '| τ | 已配对案例 | 平均完工秒 | 相对新规则0.3 | 相对旧规则0.3 | 新旧规则胜/平/负 | 温度间动作不同案例 |',
           '|---|---:|---:|---:|---:|---|---:|']
    for r in comparisons:
        lines.append(f"| {r['tau']} | {r['n']} | {r['mean_makespan']:.6f} | {r['gap_vs_canonical_tau03']:+.6%} | {r['gap_vs_legacy_tau03']:+.6%} | {r['wins_vs_legacy']}/{r['ties_vs_legacy']}/{r['losses_vs_legacy']} | {len(r['temperature_mismatches']['actions_sha256'])} |")
    lines += ['', '温度比较在同一新规则内进行；新旧规则比较固定τ=0.3。部分批次结果不能代表Validation120整体。',
        '新规则在精确算术下消除确定性H动作对正温度的依赖；完整轨迹检验检查实现、历史和批次执行是否符合该性质。',
        '概率仍随温度变化；本结果不涉及随机采样或PPO训练温度。',
        '所有轨迹独立运行。旧规则的参考结果按哈希绑定，代码默认路径另做12例完整复现。',
        '固定一个已选checkpoint和seed，不据此推断跨训练种子、其它解码器或未见数据的改善。',
        '统计解释11项覆盖：分布与profile保留、案例配对、固定checkpoint选择限制、无按结果筛样本、'
        '无基准率推断、全部失败保留、无跨种子外推、全温度报告、执行前冻结、'
        '不把解码变化归因为训练进步、无反向因果主张。未进行显著性检验或温度选优。', '']
    (root/'report.md').write_text('\n'.join(lines))
    return outcome


def run(path):
    m=load_manifest(path); original,parent=load_original(checked(m['original_study']))
    sys.path.insert(0,m['source_root'])
    import torch
    import psutil
    from onpolicy.runner.shared.stage3_h3_frozen_engine import H3FrozenEngine, model_digest
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import Progress
    from onpolicy.utils.stage3_b0_single_model import CpuEnvironmentPool
    from onpolicy.utils.stage3_b_shared_b0 import cpus, record_summary
    from onpolicy.utils.stage3_h3_frozen import verify_manifest
    from onpolicy.utils.stage3_numerics import configure_runtime
    from onpolicy.utils.stage3_canonical_h import enable_canonical_h
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=m['gpu_uuid']:
        raise ValueError('Only the original physical GPU0 is authorized')
    if not set(os.sched_getaffinity(0)).issubset(cpus(m['cpu_affinity'])):
        raise ValueError('CPU affinity escaped the original allocation')
    free=int(subprocess.check_output(['nvidia-smi','--id='+m['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<6144 or psutil.virtual_memory().available<12*2**30:
        raise RuntimeError('Insufficient bounded diagnostic headroom')
    verify_manifest(parent); configure_runtime()
    for c in m['cases']:
        for filename,digest in c['files'].items():
            if sha(Path(c['path'])/filename)!=digest:
                raise ValueError('Case content changed: '+c['name'])
    recipe=copy.deepcopy(parent['recipe'])
    recipe['gpu_headroom_mib']=torch.cuda.get_device_properties(0).total_memory/2**20-m['allocator_limit_mib']-1024
    class TemperatureEngine(H3FrozenEngine):
        diagnostic_tau=.3
        @contextmanager
        def evaluation_mode(self,decoder='H'):
            with super().evaluation_mode(decoder):
                self.policy.ac.tau=self.diagnostic_tau
                yield
    root=Path(m['root']); started=time.monotonic()
    with Progress(root/'status.json',protocol=PROTOCOL,phase='initializing',planned_groups=30,
                  epoch=m['epoch'],model_name=m.get('model_name','checkpoint')) as hb:
        runner=TemperatureEngine(parent['frozen_manifest']['path'],config=recipe,
            runtime=dict(sampling_cpus=sorted(cpus(m['cpu_affinity'])),sampling_output=str(root/'sampling')),training=False)
        try:
            runner.pool=CpuEnvironmentPool(12,timeout=recipe['ipc_timeout_seconds'],affinity=cpus(m['cpu_affinity']))
            weights=torch.load(checked(m['checkpoint']),map_location='cpu',weights_only=False)['model']
            runner.policy.ac.load_state_dict(weights,strict=True); runner.policy_updates=m['policy_updates']
            before=model_digest(runner.policy.ac)
            def evaluate(batch,tau,phase):
                if time.monotonic()-started>m['runtime_max_seconds']:
                    raise TimeoutError('Canonical temperature study reached its runtime budget')
                cases=m['cases'][batch*12:batch*12+12]
                runner.cache.last_actor=None; runner.diagnostic_tau=tau
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                hb.update(phase=phase,batch_index=batch,tau=tau,case_names=[c['name'] for c in cases])
                began=time.monotonic()
                rows=runner.rollout(cases,[1]*12,deterministic=True,decoder='H',native=False,heartbeat=hb)
                validate_rows(rows,cases,tau)
                for row in rows:
                    if row['tau']!=tau or not row['completed'] or row.get('cycle_terminated'):
                        raise RuntimeError('Incomplete or incorrectly configured evaluation')
                    if phase=='canonical_evaluation' and Path(row['case_id']).name in TARGETS:
                        trace=root/'target_trajectories'/f'{Path(row["case_id"]).name}_tau_{tag(tau)}.json.gz'
                        trace.parent.mkdir(exist_ok=True)
                        with gzip.open(trace,'wt') as f:
                            json.dump(dict(actions=row['actions'],makespan=row['makespan'],tau=tau),f)
                    if phase=='canonical_evaluation' and m.get('save_full_trajectories'):
                        trace=root/'trajectories'/f'{Path(row["case_id"]).name}_tau_{tag(tau)}.json.gz'
                        trace.parent.mkdir(exist_ok=True)
                        with gzip.open(trace,'wt') as f:
                            json.dump(dict(**record_summary(row),actions=row['actions'],
                                           manifest_sha256=m['manifest_sha256']),f)
                if model_digest(runner.policy.ac)!=before:
                    raise RuntimeError('Evaluation changed model weights')
                return dict(batch_index=batch,tau=tau,rows=[record_summary(r) for r in rows],
                    weights_unchanged=True,seconds=time.monotonic()-began,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    manifest_sha256=m['manifest_sha256'])
            proof=evaluate(m['legacy_admission_batch'],.3,'legacy_reproduction')
            expected={r['case_id']:r for r in read(checked(m['original_results'][tag(.3)]))['rows']}
            proof['mismatches']=[dict(case=r['case_id'],field=k,actual=r[k],expected=expected[r['case_id']][k])
                for r in proof['rows'] for k in ['makespan','steps','actions_sha256','history_sha256']
                if r[k]!=expected[r['case_id']][k]]
            proof['passed']=not proof['mismatches']; write(root/'legacy_reproduction.json',proof)
            if not proof['passed']:
                raise RuntimeError('Default path no longer reproduces the original full trajectories')
            write(root/'canonical_contract.json',enable_canonical_h(runner.policy))
            for batch in m['batch_order']:
                for tau in m['temperatures']:
                    group=evaluate(batch,tau,'canonical_evaluation')
                    write(root/'groups'/f'batch_{batch:02d}_tau_{tag(tau)}.json',group)
                    summary=report(m)
                    hb.update(event='group_completed',completed_groups=summary['completed_groups'],
                              completed_case_episodes=summary['completed_case_episodes'])
            verify_manifest(parent); load_manifest(path)
            summary=report(m)
            if not summary['completed']:
                raise RuntimeError('Canonical temperature matrix is incomplete')
            write(root/'completed.json',dict(**summary,seconds=time.monotonic()-started,
                  weights_unchanged=True,legacy_reproduction_passed=True,iga_solver_queries=0))
            hb.update(phase='completed',total_seconds=time.monotonic()-started)
        finally:
            runner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--study',type=Path,required=True); q.add_argument('--output',type=Path,required=True)
    q.add_argument('--model',choices=['checkpoint','s0'],default='checkpoint')
    q.add_argument('--all-trajectories',action='store_true')
    for mode in ['run','report']:
        q=sub.add_parser(mode); q.add_argument('manifest',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    elif a.mode=='run':run(a.manifest)
    else:report(load_manifest(a.manifest))


if __name__=='__main__':main()
