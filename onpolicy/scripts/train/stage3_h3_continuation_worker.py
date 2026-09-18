#!/usr/bin/env python3
"""One committed epoch or a disposable fresh-process restore canary."""
from __future__ import annotations

import argparse
import copy
import gc
import gzip
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from onpolicy.runner.shared.stage3_h3_continuation_engine import H3ContinuationEngine
from onpolicy.runner.shared.stage3_research_engine import seed_all, gradient_mode
from onpolicy.runner.shared.stage3_h3_frozen_engine import model_digest
from onpolicy.scripts.train.stage3_b_shared_b0_worker import Progress, compare_states
from onpolicy.utils.stage3_b_shared_b0 import cpus, record_summary
from onpolicy.utils.stage3_h3_continuation import active_arms, check_resources, schedule, planned_new_steps
from onpolicy.utils.stage3_h3_frozen import bind, checked
from onpolicy.utils.stage3_research import atomic_json, digest_json, read_json, verify_cases


def load(path):
    m = read_json(path)
    if digest_json({k:v for k,v in m.items() if k != 'manifest_sha256'}) != m['manifest_sha256']:
        raise ValueError('Arm manifest changed')
    suite = read_json(checked(m['suite']))
    if digest_json({k:v for k,v in suite.items() if k != 'manifest_sha256'}) != suite['manifest_sha256']:
        raise ValueError('Suite manifest changed')
    for name, checksum in suite['source_files'].items():
        from onpolicy.utils.stage3_research import digest_file
        if digest_file(Path(suite['source_root'])/name) != checksum:
            raise ValueError('Training source changed: '+name)
    for record in (m['parent_checkpoint'], m['parent_manifest'], m['baselines'], m['frozen_manifest']):
        checked(record)
    verify_cases(suite['splits']['train'], training=True)
    if digest_json(schedule(suite['splits']['train'], m['recipe'])) != m['schedule_sha256']:
        raise ValueError('Arm visit schedule changed')
    return m, suite


def engine(m, output):
    r = m['recipe']
    runner = H3ContinuationEngine(m['frozen_manifest']['path'], config=r,
        runtime=dict(sampling_cpus=sorted(cpus(r['trainer_cpus'] if r.get('execution_mode') == 'dual' else r['cpus'])),
                     sampling_output=str(output/'sampling')))
    receipt = runner.fork_parent(checked(m['parent_checkpoint']),
        expected_sha256=m['parent_checkpoint']['sha256'], parent_manifest=read_json(checked(m['parent_manifest'])),
        parent_epoch=r['parent_epoch'])
    # This is a declared new rollout schedule, common to both arms. Resume
    # subsequently restores the arm's committed RNG rather than reseeding it.
    if r.get('rng_initialization') != 'parent_checkpoint':
        seed_all(r['seed'])
    if m.get('resume_epoch'):
        previous = read_json(Path(m['root'])/'commits'/f'epoch_{m["resume_epoch"]:04d}.json')
        if previous['manifest_sha256'] != m['manifest_sha256']:
            raise ValueError('Imported resume commit belongs to a different arm')
        cursor = runner.resume(checked(previous['checkpoint']), manifest_sha256=m['manifest_sha256'])
        if cursor != m['resume_epoch'] or runner.policy_updates != previous['cumulative_ppo_steps']:
            raise ValueError('Imported resume cursor or optimizer steps changed')
        receipt.update(resume_epoch=cursor, resumed_updates=runner.policy_updates,
                       resume_checkpoint=previous['checkpoint'])
    return runner, receipt


def canary(m, suite, output, *, resumed=False):
    runner, fork = engine(m, output)
    output.mkdir(parents=True, exist_ok=True)
    with Progress(output/'status.json', phase='canary_resume' if resumed else 'canary',
                  arm=m['recipe']['arm'], tau=m['recipe']['train_tau']) as hb:
        try:
            costs = read_json(checked(m['baselines']))['source_costs']
            if not resumed:
                atomic_json(output/'fork.json', fork, overwrite=False)
                cases = [min((c for c in suite['splits']['train'] if c['profile'] == profile),
                             key=lambda c:c['content_sha256']) for profile in ('balanced','resource_ood')]
                rows = runner.collect_logical(cases, [2026091791,2026091792], ['canary-0','canary-1'],
                    logical_id='canary', heartbeat=hb)
                torch.save(rows, output/'trajectories.pt')
                runner.save(output/'before.pt', manifest_sha256=m['manifest_sha256'], next_batch=0, diagnostic=True)
                if m['recipe'].get('deferred_replay_metrics'):
                    rng=torch.cuda.get_rng_state().clone()
                    runner.config['deferred_replay_metrics']=False
                    with runner.cache.group():
                        reference=runner.replay_metrics(rows,microbatch=2,heartbeat=hb)
                    runner.config['deferred_replay_metrics']=True
                    with runner.cache.group():
                        measured=runner.replay_metrics(rows,microbatch=2,heartbeat=hb)
                    if reference!=measured or not torch.equal(rng,torch.cuda.get_rng_state()):
                        raise ValueError('Deferred native replay changed metrics or CUDA RNG')
                    atomic_json(output/'replay_equivalence.json',dict(passed=True,
                        metrics=measured,exact_reductions=True,cuda_rng_unchanged=True,
                        manifest_sha256=m['manifest_sha256']),overwrite=False)
            else:
                runner.resume(output/'before.pt', manifest_sha256=m['manifest_sha256'], allow_diagnostic=True)
                rows = torch.load(output/'trajectories.pt', map_location='cpu', weights_only=False)
            update = runner.update_logical(rows, costs, logical_id='canary', shuffle_seed=2026091796,
                                          diagnostic=True, minibatch=2, microbatch=2, heartbeat=hb)
            if update['actual_ppo_steps'] != 2 or update['completed_passes'] != 2:
                raise ValueError('Canary did not apply exactly two Adam steps')
            name = 'actual.pt' if resumed else 'expected.pt'
            runner.save(output/name, manifest_sha256=m['manifest_sha256'], next_batch=1, diagnostic=True)
            atomic_json(output/('resume_update.json' if resumed else 'update.json'), update, overwrite=False)
            if resumed:
                a = torch.load(output/'expected.pt', map_location='cpu', weights_only=False)
                b = torch.load(output/'actual.pt', map_location='cpu', weights_only=False)
                for key in ('model','actor_optim','critic_optim','value_normalizer','policy_updates',
                            'rng_python','rng_numpy','rng_torch','rng_cuda','train_tau','next_batch'):
                    compare_states(a[key], b[key], exact=True)
                atomic_json(output/'passed.json', dict(passed=True, fresh_process_resume=True,
                    cases=2, diagnostic_steps_per_path=2, train_tau=m['recipe']['train_tau'],
                    full_microbatch_equivalence='waived_not_claimed',
                    manifest_sha256=m['manifest_sha256'], solver_queries=0), overwrite=False)
        finally:
            runner.close()


def capacity(m, suite, output):
    """Disposable native-size windows; never commit these as training visits."""
    import numpy as np
    runner, _ = engine(m, output)
    output.mkdir(parents=True, exist_ok=True)
    before=model_digest(runner.policy.ac)
    proof=read_json(Path(m['root'])/'canary/passed.json')
    if not proof['passed'] or proof['manifest_sha256']!=m['manifest_sha256']:
        raise ValueError('Capacity probe needs a complete restore canary')
    trajectory_path=Path(m['root'])/'canary/trajectories.pt'
    rows=torch.load(trajectory_path,map_location='cpu',weights_only=False)
    costs=read_json(checked(m['baselines']))['source_costs']
    widths=(m['recipe']['rollout_workers'],m['recipe']['microbatch'])
    def states_at(offset,width):
        return [rows[i%len(rows)]['states'][min(offset,len(rows[i%len(rows)]['states'])-1)] for i in range(width)]
    def inference(offset):
        states=states_at(offset,widths[0])
        stack=lambda k:np.stack([s[k] for s in states])
        with torch.no_grad():
            result=runner.policy.get_actions([s['graph'] for s in states],stack('hidden'),stack('active'),
                stack('op'),stack('site'),deterministic=False,agent_types=stack('roles'),return_decision_mask=True)
        del result
    with Progress(output/'status.json',phase='capacity',arm=m['recipe']['arm']) as hb:
        try:
            inference(0)  # Explicitly exclude one warmup from measured compute.
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            began=time.monotonic();started=time.time();metrics=[]
            for offset in (0,128,256):
                inference(offset)
                chunk=[]
                for i in range(widths[1]):
                    t=copy.copy(rows[i%len(rows)])
                    first=min(offset,max(0,len(t['states'])-16))
                    t['states']=t['states'][first:first+16]
                    chunk.append(t)
                gradient_mode(runner.policy.ac)
                runner.policy.actor_optimizer.zero_grad(set_to_none=True)
                runner.policy.critic_optimizer.zero_grad(set_to_none=True)
                with runner.cache.group():
                    metrics.append(runner._minibatch_backward(chunk,costs,widths[1],hb))
                hb.update(window_offset=offset,allocated_mib=torch.cuda.memory_allocated()/2**20,
                          reserved_mib=torch.cuda.memory_reserved()/2**20)
            torch.cuda.synchronize();ended=time.time();seconds=time.monotonic()-began
            allocated=torch.cuda.max_memory_allocated()/2**20
            reserved=torch.cuda.max_memory_reserved()/2**20
            ceiling=m['recipe'].get('cuda_allocator_mib')
            if ceiling is None:
                ceiling=torch.cuda.get_device_properties(0).total_memory/2**20-m['recipe']['gpu_headroom_mib']-1024
            if model_digest(runner.policy.ac)!=before or reserved>ceiling+2:
                raise ValueError('Probe changed weights or exceeded its CUDA allocator ceiling')
            atomic_json(output/'result.json',dict(passed=True,manifest_sha256=m['manifest_sha256'],
                arm=m['recipe']['arm'],started_unix=started,ended_unix=ended,measured_seconds=seconds,
                inference_slots=widths[0],backward_microbatch=widths[1],windows=3,steps_per_window=16,
                peak_allocated_mib=allocated,peak_reserved_mib=reserved,
                cuda_allocator_mib=ceiling,metrics=metrics,
                input_trajectories=bind(trajectory_path),model_unchanged=True,
                scope='Repeated real-case graph windows; zero optimizer steps, not complete training trajectories'),overwrite=False)
        finally:
            runner.close()


def train_epoch(m, suite, epoch, output):
    root = Path(m['root']); r = m['recipe']
    commit_path = root/'commits'/f'epoch_{epoch:04d}.json'
    if commit_path.exists():
        row = read_json(commit_path)
        checked(row['checkpoint']); checked(row['update'])
        if row['manifest_sha256'] != m['manifest_sha256'] or row['epoch'] != epoch:
            raise ValueError('Foreign committed epoch')
        return
    if epoch <= r.get('optimizer_resize_after_epoch', 0):
        raise ValueError('Historical epochs must be imported, never retrained with the new minibatch')
    for arm in active_arms(suite):
        proof = read_json(Path(suite['root'])/'arms'/arm/'canary/passed.json')
        admitted = read_json(Path(suite['root'])/'arms'/arm/'manifest.json')
        if not proof['passed'] or proof['manifest_sha256'] != admitted['manifest_sha256']:
            raise ValueError('Every registered arm requires restore admission before training')
    runner, fork = engine(m, output)
    begin = time.monotonic()
    with Progress(output/'status.json', phase='training', arm=r['arm'], epoch=epoch,
                  manifest_sha256=m['manifest_sha256']) as hb:
        try:
            if epoch > 1:
                previous = read_json(root/'commits'/f'epoch_{epoch-1:04d}.json')
                if previous['manifest_sha256'] != m['manifest_sha256']:
                    raise ValueError('Previous commit belongs to a different arm')
                cursor = runner.resume(checked(previous['checkpoint']), manifest_sha256=m['manifest_sha256'])
                if cursor != epoch-1 or runner.policy_updates != previous['cumulative_ppo_steps']:
                    raise ValueError('Resume cursor/update ledger mismatch')
            else:
                atomic_json(output/'fork.json', fork, overwrite=False)
            baseline = read_json(checked(m['baselines']))
            if runner.normalization_sha256 != baseline['normalization_sha256']:
                raise ValueError('Inherited normalizer differs from frozen training reference')
            row = schedule(suite['splits']['train'], r)[epoch-1]
            logical_id = f'{r["arm"]}-epoch-{epoch:04d}'
            torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            trajectories = runner.collect_logical(row['cases'], row['seeds'], row['visit_ids'],
                logical_id=logical_id, heartbeat=hb)
            rollout_seconds = time.monotonic()-began
            rollout_memory = dict(allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                  reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            atomic_json(output/'rollout_memory.json', rollout_memory, overwrite=False)
            atomic_json(output/'rollouts.json', [record_summary(t) for t in trajectories], overwrite=False)
            atomic_json(output/'decision_stats.json', runner.decision_stats, overwrite=False)
            diagnostics = {c['path'] for c in read_json(checked(suite['diagnostic_train24']))['cases']}
            import json
            for t in trajectories:
                if t['case_id'] in diagnostics:
                    path = output/'traces'/f'{Path(t["case_id"]).name}_{t["visit_id"][:12]}.json.gz'
                    path.parent.mkdir(exist_ok=True)
                    with gzip.open(path, 'wt') as f:
                        json.dump(dict(**record_summary(t), actions=t['actions'],
                            times=[s['time'] for s in t['states']],
                            decision_masks=[s['mask'].tolist() for s in t['states']],
                            behavior_model_sha256=t['behavior_model_sha256'],
                            source_cost=baseline['source_costs'][t['case_id']],
                            event_attribution='Action/time traces only; no inferred idle-cause labels'), f)
            torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            previous_steps = runner.policy_updates
            update = runner.update_logical(trajectories, baseline['source_costs'], logical_id=logical_id,
                shuffle_seed=r['optimizer_shuffle_seed']+epoch, heartbeat=hb)
            seconds = time.monotonic()-began
            if runner.policy_updates-previous_steps != update['actual_ppo_steps']:
                raise ValueError('Adam step counter differs from the update ledger')
            model_path = output/'checkpoint.pt'
            runner.save(model_path, manifest_sha256=m['manifest_sha256'], next_batch=epoch)
            update_path = output/'update.json'
            atomic_json(update_path, dict(epoch=epoch, arm=r['arm'], training_episodes=epoch*384,
                cumulative_ppo_steps=runner.policy_updates, inherited_ppo_steps=fork['inherited_updates'],
                new_ppo_steps=runner.policy_updates-fork['inherited_updates'], update=update,
                rollout_seconds=rollout_seconds, update_seconds=seconds, total_seconds=time.monotonic()-begin,
                rollout_memory=rollout_memory, update_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                update_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                manifest_sha256=m['manifest_sha256']), overwrite=False)
            atomic_json(commit_path, dict(epoch=epoch, checkpoint=bind(model_path), update=bind(update_path),
                training_episodes=epoch*384, cumulative_ppo_steps=runner.policy_updates,
                inherited_ppo_steps=fork['inherited_updates'],
                new_ppo_steps=runner.policy_updates-fork['inherited_updates'],
                ppo_budget_complete=(runner.policy_updates-fork['inherited_updates']==planned_new_steps(r,epoch)),
                manifest_sha256=m['manifest_sha256'], committed_unix=time.time()), overwrite=False)
            hb.update(phase='committed', checkpoint=str(model_path), new_ppo_steps=runner.policy_updates-fork['inherited_updates'])
        finally:
            runner.close()
            gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train','canary','canary-resume','capacity'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); m, suite = load(a.manifest)
    os.sched_setaffinity(0, cpus(m['recipe']['trainer_cpus']))
    check_resources(m['recipe'])
    a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == 'train':
        if a.epoch not in range(1, m['recipe']['epochs'] + 1):
            raise ValueError('Unregistered epoch')
        train_epoch(m, suite, a.epoch, a.output)
    elif a.mode == 'capacity':
        capacity(m, suite, a.output)
    else:
        canary(m, suite, a.output, resumed=a.mode == 'canary-resume')


if __name__ == '__main__':
    main()
