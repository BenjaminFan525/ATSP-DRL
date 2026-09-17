#!/usr/bin/env python3
"""Isolated, fixed-checkpoint H3 temperature evaluation; never trains a model.

Imports the parent run's immutable source snapshot. The sole policy change is
the temperature inside its existing deterministic evaluation context. Cached
tau=.3 references require a fixed12 action/history/cost reproduction first.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PROTOCOL = 'stage3_h3_fixed_checkpoint_temperature_v1'
TEMPERATURES = (0.03, 0.1, 0.3)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def checked(record):
    if sha(record['path']) != record['sha256']:
        raise ValueError(f"Input identity changed: {record['path']}")
    return Path(record['path'])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def tag(tau):
    return str(tau).replace('.', 'p')


def validate_rows(rows, cases, tau):
    if len(rows) != len(cases) or [r['case_id'] for r in rows] != [c['path'] for c in cases]:
        raise ValueError('Pairing requires the complete original ordered case list')
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate evaluation cases')
    for row, case in zip(rows, cases):
        if (row['tau'] != tau or row['decoder'] != 'H' or row['seed'] != 1
                or not row['behavior_deterministic'] or not row['completed']
                or row.get('cycle_terminated') or row.get('forced_replay')
                or row['distribution'] != case['distribution'] or row['profile'] != case['profile']):
            raise ValueError(f"Evaluation contract mismatch: {row['case_id']}")


def prepare(args):
    root, output = args.source_run.resolve(), args.output.resolve()
    m = read(root / 'manifest.json')
    base = read(root / 'baselines.json')
    epoch = args.epoch
    commit = read(root / 'commits' / f'epoch_{epoch:04d}.json')
    evaluated = read(root / 'validator/results' / f'epoch_{epoch:04d}.json')
    if evaluated['checkpoint'] != commit['checkpoint'] or not evaluated['completed']:
        raise ValueError('Selected checkpoint has no matching complete evaluation')
    if not 0 < args.allocator_mib <= 4096:
        raise ValueError('Concurrent diagnostics are capped at 4096 MiB in the Torch allocator')
    output.mkdir(parents=True, exist_ok=False)
    driver = output / 'driver.py'
    shutil.copy2(__file__, driver)
    cases = m['splits']['validation']
    if len(cases) != 120 or m['recipe']['validation_workers'] != 12:
        raise ValueError('This study requires the original Validation120 / fixed12 contract')
    models = {
        'checkpoint': dict(checkpoint=commit['checkpoint'], epoch=epoch,
                           policy_updates=commit['actual_ppo_steps'],
                           cache_dir=str(root / 'validator/cases' / f'epoch_{epoch:04d}')),
        's0': dict(checkpoint=base['initialization'], epoch=0, policy_updates=0,
                   cache_dir=str(root / 'baseline_cases' / base['initialization_label'] / 'validation')),
    }
    for name, item in models.items():
        checked(item['checkpoint'])
        entries = [read(Path(item['cache_dir']) / f'{i:04d}.json') for i in range(len(cases))]
        rows = [e['result'] for e in entries]
        validate_rows(rows, cases, .3)
        for entry, case in zip(entries, cases):
            b = entry['binding']
            if (entry['case_sha256'] != case['content_sha256']
                    or b['model_sha256'] != item['checkpoint']['sha256']
                    or b['code_sha256'] != m['code_sha256'] or b['planning'] != base['planning']
                    or b['tau'] != .3 or b['decoder'] != 'H' or b['seed'] != 1
                    or b['batch_contract'] != m['recipe']['evaluation_batch_contract']):
                raise ValueError(f'Cached reference has a different identity: {name}')
        reference = output / 'references' / f'{name}_tau_0p3.json'
        write(reference, dict(entries=entries, rows=rows, original_directory=item['cache_dir']))
        item['reference'] = bind(reference)
    study = dict(protocol=PROTOCOL, created_unix=time.time(), root=str(output),
        parent=bind(root / 'manifest.json'), parent_baselines=bind(root / 'baselines.json'),
        parent_code_sha256=m['code_sha256'], source_root=m['source_root'], driver=bind(driver),
        cases=cases, ordered_cases_sha256=identity(cases), models=models,
        temperatures=list(TEMPERATURES), decoder='H', seed=1,
        checkpoint_selection=f'epoch {epoch}, fixed before temperature evaluation; no reselection',
        cache_reproduction_cases=12, new_case_episodes=504, cached_case_episodes=240,
        allocator_limit_mib=args.allocator_mib, cpu_affinity=m['recipe']['validator_cpus'],
        gpu_uuid=m['recipe']['gpu_uuid'], gpu_index=0, per_temperature_timeout_seconds=21600,
        complete_timeout_seconds=43200, bootstrap_draws=20000, bootstrap_seed=2026091601,
        training_mutation=False, solver_queries=0,
        interpretation='Validation-only exploratory decoder sensitivity; fixed training seed; '
                       'cached tau=.3 scores checked on original first12 for each checkpoint')
    study['study_sha256'] = identity(study)
    write(output / 'manifest.json', study)
    (output / 'plan.md').write_text(
        '## Material Passport\n\n- Origin Skill: academic-research-suite / experiment-agent\n'
        '- Origin Mode: run\n- Origin Date: 2026-09-16\n- Verification Status: PREPARED\n'
        f'- Version Label: {PROTOCOL}\n\n'
        f'固定 epoch {epoch} checkpoint 和初始化 S0，各自评测 tau=0.03/0.1/0.3。\n'
        '案例为原顺序 Validation120，H3/F4/soft，H 解码，seed=1，固定12槽位，完成槽位保留。\n'
        'tau=0.3 的两份120例结果已冻结；分别复跑原首12例，动作/历史哈希、步数、成本须一致后才能复用。\n'
        '新增24例复现检查与480例温度评测。对照失败时停止诊断，保留证据。\n'
        '使用 GPU0；Torch 分配上限4 GiB；CPU仅使用原半区内验证子集；不修改或暂停训练，不求解IGA。\n'
        '保存逐案例成本、动作/历史哈希、胜平负、尾部、分布和profile结果；分层配对bootstrap 20000次。\n'
        '报告全部温度，探索性区间不作多重比较校正，不据此宣布独立泛化或最优温度；不自动修改主实验设置。\n')
    print(json.dumps(dict(manifest=str(output / 'manifest.json'), models=models,
                          new_case_episodes=504), ensure_ascii=False))


def prepare_extension(args):
    base_path, output = args.base_study.resolve(), args.output.resolve()
    base, _ = load_study(base_path)
    added = sorted(set(args.temperatures))
    if (not added or len(added) != len(args.temperatures)
            or any(not max(base['temperatures']) < t <= 10 for t in added)):
        raise ValueError('Extension temperatures must be distinct, higher than existing values, and at most 10')
    output.mkdir(parents=True, exist_ok=False)
    driver = output / 'driver.py'
    shutil.copy2(__file__, driver)
    s = copy.deepcopy(base)
    s.pop('study_sha256')
    s.update(created_unix=time.time(), root=str(output), driver=bind(driver),
        temperatures=sorted(base['temperatures'] + added), evaluation_temperatures=added,
        base_study=bind(base_path), base_study_sha256=base['study_sha256'],
        base_service_unit=(Path(base['root']) / 'service_unit.txt').read_text().strip(),
        dependency_wait_timeout_seconds=50400,
        cache_reproduction_cases=0, inherited_reproduction_cases=24,
        new_case_episodes=len(added)*len(base['models'])*len(base['cases']),
        cached_case_episodes=len(base['temperatures'])*len(base['models'])*len(base['cases']),
        interpretation='Sequential extension of a fixed-checkpoint temperature study; '
                       'imports every previous temperature only after successful completion; '
                       'exploratory Validation-only comparison, no training or IGA solving')
    s['study_sha256'] = identity(s)
    write(output / 'manifest.json', s)
    (output / 'plan.md').write_text(
        '## Material Passport\n\n- Origin Skill: academic-research-suite / experiment-agent\n'
        '- Origin Mode: run\n- Origin Date: 2026-09-16\n- Verification Status: QUEUED\n'
        f'- Version Label: {PROTOCOL}\n\n'
        f'接续温度：{added}；最终合并温度：{s["temperatures"]}。\n'
        f'前置研究：{base_path}。前置任务完整成功且进程退出后，才加载GPU模型。\n'
        '固定原 epoch1 checkpoint、S0、Validation120原顺序、H3/F4/soft、H解码和seed=1。\n'
        f'新增{s["new_case_episodes"]}例评测；复用前置任务全部{s["cached_case_episodes"]}个模型/温度/案例结果。\n'
        '沿用GPU0、原CPU半区验证子集、一个评测模型、固定12环境、4 GiB Torch分配上限。\n'
        '前置24例复现证据及每个完整结果绑定哈希；不重跑已完成的温度，不修改原任务、训练或IGA。\n'
        '最终报告全部六档温度的逐案例、分布、profile、胜平负、尾部和配对区间。\n')
    write(output / 'status.json', dict(status='queued', phase='waiting_for_previous_temperatures',
          study_sha256=s['study_sha256'], temperatures=added, dependency=str(base_path),
          heartbeat_unix=time.time()))
    print(json.dumps(dict(manifest=str(output / 'manifest.json'), added=added,
                          new_cases=s['new_case_episodes']), ensure_ascii=False))


def load_study(path):
    s = read(path)
    data = {k: v for k, v in s.items() if k != 'study_sha256'}
    if s['protocol'] != PROTOCOL or identity(data) != s['study_sha256']:
        raise ValueError('Diagnostic manifest changed')
    checked(s['driver'])
    m = read(checked(s['parent']))
    checked(s['parent_baselines'])
    if identity(s['cases']) != s['ordered_cases_sha256'] or s['cases'] != m['splits']['validation']:
        raise ValueError('Validation case order changed')
    for item in s['models'].values():
        checked(item['checkpoint']); checked(item['reference'])
    sys.path.insert(0, s['source_root'])
    return s, m


def wait_for_base(s):
    """Wait without importing Torch or allocating a GPU model."""
    if 'base_study' not in s:
        return
    root = Path(s['root'])
    base = read(checked(s['base_study']))
    if (base['study_sha256'] != s['base_study_sha256']
            or identity({k: v for k, v in base.items() if k != 'study_sha256'}) != base['study_sha256']
            or base['models'] != s['models'] or base['cases'] != s['cases']
            or base['parent_code_sha256'] != s['parent_code_sha256']):
        raise ValueError('Temperature extension changed the fixed checkpoint/case/code identity')
    checked(base['driver'])
    started = time.monotonic()
    while True:
        base_root = Path(base['root'])
        progress = read(base_root / 'status.json') if (base_root / 'status.json').exists() else {}
        completed = read(base_root / 'completed.json') if (base_root / 'completed.json').exists() else {}
        active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', s['base_service_unit']],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        state = dict(status='queued', phase='waiting_for_previous_temperatures', pid=os.getpid(),
            study_sha256=s['study_sha256'], heartbeat_unix=time.time(),
            waiting_seconds=time.monotonic()-started, dependency=base['root'],
            dependency_status=progress.get('status'), dependency_phase=progress.get('phase'),
            dependency_model=progress.get('model'), dependency_tau=progress.get('tau'),
            dependency_completed_cases=progress.get('completed_cases'),
            new_temperatures=s['evaluation_temperatures'])
        write(root / 'status.json', state)
        if completed.get('completed') and progress.get('status') == 'completed' and not active:
            break
        if progress.get('status') == 'failed' or not active:
            write(root / 'status.json', dict(**{k: v for k, v in state.items() if k != 'status'},
                  status='failed', error='Previous temperature study did not finish successfully'))
            raise RuntimeError('Previous temperature study failed or stopped; extension was not started')
        if time.monotonic()-started > s['dependency_wait_timeout_seconds']:
            raise TimeoutError('Temperature extension exceeded its dependency waiting budget')
        time.sleep(30)
    if completed['study_sha256'] != base['study_sha256']:
        raise ValueError('Dependency completion belongs to another study')
    admission = read(base_root / 'admission.json')
    if not admission['passed'] or not admission['weights_unchanged']:
        raise ValueError('Dependency lacks a successful immutable-weight reproduction')
    imported = []
    for name in s['models']:
        for tau in base['temperatures']:
            source = base_root / 'results' / f'{name}_tau_{tag(tau)}.json'
            item = read(source)
            validate_rows(item['rows'], s['cases'], tau)
            if (item['study_sha256'] != base['study_sha256']
                    or item['checkpoint'] != s['models'][name]['checkpoint']):
                raise ValueError('Dependency result has a different checkpoint/study identity')
            result = dict(item, study_sha256=s['study_sha256'], reused_from_base_study=True,
                          dependency_result=bind(source))
            write(root / 'results' / source.name, result)
            imported.append(bind(source))
    write(root / 'dependency_import.json', dict(completed=True, base_study=s['base_study'],
        completion=bind(base_root / 'completed.json'), admission=bind(base_root / 'admission.json'),
        results=imported, cached_case_episodes=s['cached_case_episodes'], imported_unix=time.time()))


def report(s):
    import numpy as np
    from onpolicy.utils.stage3_h3_frozen import paired_metrics
    root = Path(s['root'])
    records = {}
    for name in s['models']:
        for tau in s['temperatures']:
            path = root / 'results' / f'{name}_tau_{tag(tau)}.json'
            if path.exists():
                records[name, tau] = read(path)['rows']
                validate_rows(records[name, tau], s['cases'], tau)
    complete = len(records) == len(s['models']) * len(s['temperatures'])
    comparisons = {}
    summaries = []
    for (name, tau), rows in records.items():
        ref = records.get((name, .3))
        if ref is None:
            continue
        metrics = paired_metrics(rows, {r['case_id']: r['makespan'] for r in ref},
                                 seed=s['bootstrap_seed'], draws=s['bootstrap_draws'])
        metrics['different_action_cases'] = sum(a['actions_sha256'] != b['actions_sha256']
                                                for a, b in zip(rows, ref))
        metrics['different_history_cases'] = sum(a['history_sha256'] != b['history_sha256']
                                                 for a, b in zip(rows, ref))
        summaries.append(dict(model=name, tau=tau, **metrics))
    for tau in s['temperatures']:
        if ('checkpoint', tau) in records and ('s0', tau) in records:
            comparisons[str(tau)] = paired_metrics(records['checkpoint', tau],
                {r['case_id']: r['makespan'] for r in records['s0', tau]},
                seed=s['bootstrap_seed'], draws=s['bootstrap_draws'])
    payload = dict(completed=complete, study_sha256=s['study_sha256'], summaries=summaries,
                   checkpoint_versus_s0=comparisons, solver_queries=0, training_mutation=False,
                   ci_scope='Exploratory pointwise 95% profile-stratified paired intervals; '
                            'conditional on fixed checkpoint and training seed; not multiplicity corrected')
    columns = ['case_id', 'profile', 'distribution']
    for key in records:
        prefix = f'{key[0]}_tau_{tag(key[1])}'
        columns += [prefix + '_makespan', prefix + '_actions_sha256']
    with (root / 'per_case.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for i, case in enumerate(s['cases']):
            row = dict(case_id=case['path'], profile=case['profile'], distribution=case['distribution'])
            for key, rows in records.items():
                prefix = f'{key[0]}_tau_{tag(key[1])}'
                for field in ('makespan', 'actions_sha256'):
                    row[prefix + '_' + field] = rows[i][field]
            writer.writerow(row)
    profiles = []
    for (name, tau), rows in records.items():
        reference = records.get((name, .3))
        if reference is None:
            continue
        for dimension in ('profile', 'distribution'):
            for label in sorted({r[dimension] for r in rows}):
                ids = [i for i, r in enumerate(rows) if r[dimension] == label]
                a = np.array([rows[i]['makespan'] for i in ids]); b = np.array([reference[i]['makespan'] for i in ids])
                profiles.append(dict(model=name, tau=tau, dimension=dimension, group=label, n=len(ids),
                    makespan=float(a.mean()), reference=float(b.mean()),
                    gap_fraction=float(a.mean()/b.mean()-1), wins=int((a < b-1e-6).sum()),
                    ties=int((np.abs(a-b) <= 1e-6).sum()), losses=int((a > b+1e-6).sum())))
    write(root / 'profiles.json', profiles)
    write(root / 'summary.json', payload)
    lines = ['## Material Passport', '', '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run / validate', '- Origin Date: 2026-09-16',
        '- Verification Status: ' + ('COMPLETED' if complete else 'RUNNING'), f'- Version Label: {PROTOCOL}', '',
        f'固定 checkpoint: epoch {s["models"]["checkpoint"]["epoch"]}；对照 S0；Validation120；H3/F4/soft；H 解码；seed=1。',
        ('tau=0.3 使用冻结120例记录，原任务已对每个模型首12例核对动作/历史/成本；'
         '本接续任务按哈希复用原三档结果和复现证据。' if 'base_study' in s else
         'tau=0.3 使用冻结120例记录，每个模型在同一诊断进程复跑原首12例进行动作/历史/成本核对。'), '',
        '| 模型 | tau | 平均耗时秒 | 相对本模型tau=0.3 | 配对95%区间 | 胜/平/负 | 动作变化案例 |',
        '|---|---:|---:|---:|---|---|---:|']
    for row in summaries:
        lo, hi = row['paired_gap_ci95']
        lines.append(f'| {row["model"]} | {row["tau"]} | {row["makespan"]:.6f} | '
            f'{row["gap_fraction"]:+.4%} | [{lo:+.4%}, {hi:+.4%}] | '
            f'{row["wins"]}/{row["ties"]}/{row["losses"]} | {row["different_action_cases"]} |')
    lines += ['', '差距为候选/参考−1，负数表示候选更快。', '',
              '| tau | checkpoint相对同温度S0 | 配对95%区间 |', '|---|---:|---|']
    for tau, row in comparisons.items():
        lo, hi = row['paired_gap_ci95']
        lines.append(f'| {tau} | {row["gap_fraction"]:+.4%} | [{lo:+.4%}, {hi:+.4%}] |')
    lines += ['', '逐案例见 per_case.csv；分布和profile见 profiles.json；完整统计见 summary.json。',
              '本诊断不改变训练温度、主评测或选模流程；不重新求解IGA。', '',
              '统计解释检查11/11：按profile和distribution检查方向；以案例为单位；固定验证集与已选checkpoint存在选择限制；'
              '无按结果筛样本或控制碰撞变量；不作诊断基准率推断；保留S0防止把选模效应当温度收益；'
              '要求全部案例完成，不丢弃失败案例；全部温度完整报告；方案在执行前冻结；'
              '因果解释仅限固定模型解码温度；无反向因果或跨种子泛化主张。',
              '区间是探索性、未经多重比较校正的逐项区间；温度选择仍需独立数据验证。', '']
    (root / 'report.md').write_text('\n'.join(lines))
    return payload


def run(path):
    s, m = load_study(path)
    root = Path(s['root'])
    wait_for_base(s)
    import torch
    import psutil
    from onpolicy.runner.shared.stage3_h3_frozen_engine import H3FrozenEngine, model_digest
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import Progress, evaluate_cases
    from onpolicy.utils.stage3_b0_single_model import CpuEnvironmentPool
    from onpolicy.utils.stage3_b_shared_b0 import cpus
    from onpolicy.utils.stage3_h3_frozen import verify_manifest
    from onpolicy.utils.stage3_research import digest_json

    if os.environ.get('CUDA_VISIBLE_DEVICES') != s['gpu_uuid']:
        raise ValueError('Diagnostic requires the explicitly bound GPU0 UUID')
    if not set(os.sched_getaffinity(0)).issubset(cpus(s['cpu_affinity'])):
        raise ValueError('Diagnostic escaped the existing CPU half allocation')
    if psutil.virtual_memory().available < 12 * 2**30:
        raise RuntimeError('Less than 12 GiB host memory available for bounded diagnostics')
    raw = subprocess.check_output(['nvidia-smi', '--id=' + s['gpu_uuid'],
        '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    if int(raw[0]) != 0 or int(raw[1]) < 6144:
        raise RuntimeError('GPU0 has less than the diagnostic admission headroom')
    verify_manifest(m)
    for case in s['cases']:
        for filename, checksum in case['files'].items():
            if sha(Path(case['path']) / filename) != checksum:
                raise ValueError(f'Case content changed: {case["path"]}/{filename}')

    class TemperatureEngine(H3FrozenEngine):
        diagnostic_tau = .3

        @contextmanager
        def evaluation_mode(self, decoder='H'):
            with super().evaluation_mode(decoder):
                self.policy.ac.tau = self.diagnostic_tau
                yield

    recipe = copy.deepcopy(m['recipe'])
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    # H3FrozenEngine adds 1024 MiB to this value. Cap only this process; the
    # parent trainer's allocator and CUDA state are never modified.
    recipe['gpu_headroom_mib'] = total_mib - s['allocator_limit_mib'] - 1024
    runtime = dict(sampling_cpus=sorted(cpus(s['cpu_affinity'])),
                   sampling_output=str(root / 'sampling'))
    start = time.monotonic()
    with Progress(root / 'status.json', protocol=PROTOCOL, study_sha256=s['study_sha256'],
                  planned_new_cases=s['new_case_episodes'], phase='initializing') as hb:
        runner = TemperatureEngine(m['frozen_manifest']['path'], config=recipe,
                                   runtime=runtime, training=False)
        try:
            runner.pool = CpuEnvironmentPool(12, timeout=recipe['ipc_timeout_seconds'],
                                             affinity=cpus(s['cpu_affinity']))
            def load_model(name):
                item = s['models'][name]
                weights = torch.load(checked(item['checkpoint']), map_location='cpu', weights_only=False)['model']
                runner.policy.ac.load_state_dict(weights, strict=True)
                if not all(torch.equal(v.cpu(), weights[k]) for k, v in runner.policy.ac.state_dict().items()):
                    raise ValueError('Diagnostic model is not identical to its fixed checkpoint')
                runner.policy_updates = item['policy_updates']
                runner.cache.last_actor = None
                return model_digest(runner.policy.ac)

            def evaluate(name, tau, cases, directory, phase):
                if time.monotonic() - start > s['complete_timeout_seconds']:
                    raise TimeoutError('Temperature study reached its 12-hour budget')
                before = load_model(name)
                runner.diagnostic_tau = tau
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                hb.update(phase=phase, model=name, tau=tau, completed_cases=0, total_cases=len(cases))
                began = time.monotonic()
                rows = evaluate_cases(m, runner, cases, directory, hb, decoder='H', native=False,
                    binding=dict(protocol=PROTOCOL, study_sha256=s['study_sha256'],
                        model_sha256=s['models'][name]['checkpoint']['sha256'], planning=recipe['planning'],
                        code_sha256=m['code_sha256'], diagnostic_driver_sha256=s['driver']['sha256'],
                        cases_sha256=digest_json(cases), history=recipe['history'], decoder='H', tau=tau,
                        seed=1, label=f'{name}_tau_{tag(tau)}'), timeout=s['per_temperature_timeout_seconds'])
                rows = [row['result'] for row in rows]
                validate_rows(rows, cases, tau)
                if model_digest(runner.policy.ac) != before:
                    raise ValueError('Evaluation changed model parameters')
                return dict(rows=rows, model=name, tau=tau, seconds=time.monotonic()-began,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    allocator_limit_mib=s['allocator_limit_mib'], weights_unchanged=True,
                    checkpoint=s['models'][name]['checkpoint'], study_sha256=s['study_sha256'])

            if 'base_study' not in s:
                for name, item in s['models'].items():
                    result = evaluate(name, .3, s['cases'][:12], root / 'proof_cases' / name, 'reference_reproduction')
                    reference = read(checked(item['reference']))['rows']
                    mismatches = []
                    for actual, expected in zip(result['rows'], reference[:12]):
                        for field in ('case_id', 'makespan', 'steps', 'actions_sha256', 'history_sha256'):
                            if actual[field] != expected[field]:
                                mismatches.append(dict(case_id=actual['case_id'], field=field,
                                                       actual=actual[field], expected=expected[field]))
                    proof = dict(**result, passed=not mismatches, mismatches=mismatches)
                    write(root / 'proofs' / f'{name}.json', proof)
                    if mismatches:
                        raise ValueError(f'{name} tau=.3 reproduction failed; cached comparisons not admitted')
                    write(root / 'results' / f'{name}_tau_0p3.json',
                          dict(rows=reference, model=name, tau=.3, reused=True,
                               complete_reference=bind(item['reference']['path']),
                               reproduction=bind(root / 'proofs' / f'{name}.json'),
                               checkpoint=item['checkpoint'], study_sha256=s['study_sha256']))
            write(root / 'admission.json', dict(passed=True,
                reproduced_cases=24 if 'base_study' not in s else 0,
                inherited_reproduced_cases=s.get('inherited_reproduction_cases', 0),
                weights_unchanged=True, allocator_limit_mib=s['allocator_limit_mib'],
                study_sha256=s['study_sha256'], solver_queries=0))
            report(s)
            for tau in s.get('evaluation_temperatures', [.03, .1]):
                for name in s['models']:
                    result = evaluate(name, tau, s['cases'], root / 'cases' / f'{name}_tau_{tag(tau)}', 'paired_evaluation')
                    write(root / 'results' / f'{name}_tau_{tag(tau)}.json', result)
                    report(s)
            verify_manifest(m)
            load_study(path)
            summary = report(s)
            if not summary['completed']:
                raise ValueError('Incomplete temperature matrix')
            hb.update(phase='completed', total_seconds=time.monotonic()-start, solver_queries=0)
            write(root / 'completed.json', dict(completed=True, new_cases=s['new_case_episodes'],
                cached_cases=s['cached_case_episodes'],
                total_seconds=time.monotonic()-start, study_sha256=s['study_sha256'], solver_queries=0))
        finally:
            runner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epoch', type=int, default=1)
    p.add_argument('--allocator-mib', type=int, default=4096)
    p = sub.add_parser('extend')
    p.add_argument('--base-study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--temperatures', nargs='+', type=float, required=True)
    for name in ('run', 'report'):
        p = sub.add_parser(name); p.add_argument('manifest', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'extend':
        prepare_extension(args)
    elif args.mode == 'run':
        run(args.manifest)
    else:
        s, _ = load_study(args.manifest); report(s)


if __name__ == '__main__':
    main()
