#!/usr/bin/env python3
"""Bounded capacity admission for a user-requested single-model configuration.

Exercise actual environment slots and a dense historical TBPTT window. This is
not a full training group, an optimizer update, or legacy numerical equivalence.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CODE_FILES = ('onpolicy/utils/stage3_b0_single_model.py',
    'onpolicy/runner/shared/stage3_b_shared_b0_engine.py',
    'onpolicy/utils/stage3_b0_throughput.py',
    'onpolicy/scripts/train/autotune_stage3_b_shared_b0.py',
    'onpolicy/scripts/train/probe_stage3_b0_requested_capacity.py')


class WindowComplete(Exception):
    pass


def environment_parity(runner, cases, seeds):
    """Compare real environment reset/step outputs with and without slot reuse."""
    import numpy as np
    import torch
    from onpolicy.utils.stage3_b0_single_model import CpuEnvironmentPool
    from onpolicy.utils.stage3_b0_throughput import capture_rng, restore_rng
    from onpolicy.runner.shared.stage3_research_engine import authoritative_history
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states
    def plain(value):
        if hasattr(value, 'to_dict'):
            value = value.to_dict()
        if isinstance(value, dict):
            return {k:plain(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value
    rng = capture_rng()
    pools = []
    try:
        for workers in (4,2):
            pools.append(CpuEnvironmentPool(4, timeout=600,
                affinity=sorted(os.sched_getaffinity(0)), worker_count=workers))
        commands = [(i,'reset',runner.environment_config(c['path'],s))
                    for i,(c,s) in enumerate(zip(cases[:4],seeds[:4]))]
        left,right = [pool.call(commands) for pool in pools]
        compare_states(plain(left), plain(right), exact=True)
        obs,infos = [left[i][0] for i in range(4)],[left[i][2] for i in range(4)]
        hidden=np.zeros((4,104,1,64),np.float32)
        previous=np.full((4,104,3),-1,np.int64)
        with torch.no_grad():
            for _ in range(3):
                hist=authoritative_history(previous,infos)
                active=np.stack([row['active_agents'] for row in infos]).reshape(4,104)
                roles=np.stack([row['agent_types'] for row in infos]).reshape(4,104)
                _,actions,_,new_h,_=runner.policy.get_actions(obs,hidden,active,hist[...,0],hist[...,1],
                    deterministic=False,agent_types=roles,return_decision_mask=True)
                actions=actions.cpu().numpy().astype(np.int64)
                if actions.shape[-1]==2:
                    actions=np.concatenate((actions,np.full((4,104,1),-1,np.int64)),axis=-1)
                commands=[(i,'step',actions[i]) for i in range(4)]
                left,right=[pool.call(commands) for pool in pools]
                compare_states(plain(left),plain(right),exact=True)
                hidden=new_h.cpu().numpy();previous=actions
                for i in range(4):
                    obs[i],_,done,infos[i]=left[i]
                    hidden[i,np.asarray(done).reshape(104).astype(bool)]=0
        return dict(passed=True,environments=4,reference_processes=4,multiplexed_processes=2,steps_per_environment=3)
    finally:
        for pool in pools:
            pool.close()
        restore_rng(rng)


def main():
    import numpy as np
    import torch
    from onpolicy.utils.stage3_b0_single_model import ThroughputEngine, VERSION, RNG_CONTRACT
    from onpolicy.utils.stage3_b0_throughput import capture_rng, restore_rng
    from onpolicy.utils.stage3_b0_restart import verify_restored_state
    from onpolicy.utils.stage3_b_shared_b0 import training_schedule, read_commit, manifest_identity
    from onpolicy.utils.stage3_research import read_json, atomic_json, digest_file
    from onpolicy.utils.stage3_performance import tensor_bytes
    from onpolicy.runner.shared.stage3_research_engine import gradient_mode
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import Progress, compare_states, read_baseline
    from onpolicy.scripts.train.autotune_stage3_b_shared_b0 import dense_window, memory_monitor
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trajectories', type=Path, required=True)
    p.add_argument('--environments', type=int, default=240)
    p.add_argument('--microbatch', type=int, default=80)
    p.add_argument('--environment-processes', type=int, default=64)
    p.add_argument('--input-cache-mib', type=int, default=4096)
    p.add_argument('--encoder-checkpoint', action='store_true')
    p.add_argument('--memory-limit-gib', type=int, default=114)
    p.add_argument('--training-selection', type=Path)
    args = p.parse_args()
    m = read_json(args.manifest)
    if manifest_identity(m) != m['manifest_sha256']:
        raise ValueError('Parent manifest changed')
    out = args.output.resolve(); out.mkdir(parents=True)
    commits = sorted((Path(m['root'])/'commits').glob('batch_*.json'))
    commit = read_commit(m, commits[-1])
    plan = training_schedule(m)[commit['next_batch']:]
    cases = [c for row in plan for c in row['cases']][:args.environments]
    seeds = [s for row in plan for s in row['seeds']][:args.environments]
    subset_proof = None
    if args.training_selection:
        from onpolicy.utils.stage3_b0_subset import binding, verify_selection, epoch_rows
        selected = verify_selection(m, binding(args.training_selection))
        initial = epoch_rows(selected, 0)[:args.environments]
        cases, seeds = [r[0] for r in initial], [r[1] for r in initial]
        subset_proof = dict(selection=binding(args.training_selection),
            cases_sha256=selected['cases_sha256'], sampling_cases=[c['path'] for c in cases],
            sampling_seeds=seeds, formal_training_visits=0)
    if len(cases) != args.environments:
        raise ValueError('Insufficient remaining visits for the capacity window')
    parameters = dict(environments=args.environments, microbatch=args.microbatch,
        lanes=1, cache_mib=args.input_cache_mib, environment_processes=args.environment_processes,
        encoder_activation_checkpoint=args.encoder_checkpoint)
    rows, stop = [], threading.Event()
    monitor = threading.Thread(target=memory_monitor, args=(out, stop, rows), daemon=True)
    monitor.start()
    runner = None
    try:
        runtime = dict(global_batch=args.environments, sampling_environments=args.environments,
            sampling_lanes=1, microbatch=args.microbatch, input_cache_mib=args.input_cache_mib,
            cuda_memory_headroom_mib=8192, environment_processes=args.environment_processes,
            encoder_activation_checkpoint=args.encoder_checkpoint,
            sampling_output=str(out/'sampling'), sampling_cpus=sorted(os.sched_getaffinity(0)))
        runner = ThroughputEngine(m['frozen_manifest']['path'], config=m['recipe'],
            width=args.environments, create_pool=False, runtime=runtime)
        runner.resume(commit['checkpoint'], manifest_sha256=m['manifest_sha256'])
        restored = verify_restored_state(runner, commit['checkpoint'])
        atomic_json(out/'resume_verified.json', restored)
        parity = environment_parity(runner,cases,seeds)
        verify_restored_state(runner,commit['checkpoint'])
        atomic_json(out/'environment_parity.json',parity)
        original = runner.policy.get_actions
        repeated = False
        retained = [dict(states=[]) for _ in range(args.environments)]
        observed_pool = False
        def checked(*a, **kw):
            nonlocal repeated, observed_pool
            if repeated:
                first = original(*a, **kw)
            else:
                before = capture_rng(); first = original(*a, **kw); after = capture_rng()
                restore_rng(before); second = original(*a, **kw)
                compare_states(first, second, exact=True)
                compare_states(after, capture_rng(), exact=True)
                repeated = True
            if len(a[0]) != args.environments:
                raise ValueError('Sampling window did not batch all requested environments')
            _,actions,logp,_,mask = first
            actions=actions.cpu().numpy().astype(np.int64)
            if actions.shape[-1]==2:
                actions=np.concatenate((actions,np.full((len(a[0]),104,1),-1,np.int64)),axis=-1)
            logp,mask=logp.cpu().numpy(),mask.cpu().numpy()
            for i in range(args.environments):
                retained[i]['states'].append(dict(graph=a[0][i],hidden=a[1][i].copy(),
                    active=a[2][i].copy(),op=a[3][i].copy(),site=a[4][i].copy(),
                    roles=kw['agent_types'][i].copy(),action=actions[i].copy(),
                    old_logp=logp[i].reshape(104).copy(),mask=mask[i].reshape(104).copy()))
            if not observed_pool:
                original_call=runner.pool.call
                def observed_call(commands):
                    result=original_call(commands)
                    for index,command,_ in commands:
                        if command=='step':
                            retained[index]['states'][-1]['done']=np.asarray(result[index][2]).reshape(104).astype(bool)
                    return result
                runner.pool.call=observed_call
                observed_pool=True
            return first
        runner.policy.get_actions = checked
        window = {}
        with Progress(out/'status.json', phase='requested_capacity') as hb:
            class WindowProgress:
                def update(self, **values):
                    hb.update(**values)
                    if values.get('rollout_step', 0) >= 26:
                        window.update(values)
                        raise WindowComplete()
            started = time.monotonic()
            try:
                runner.parallel_collect(cases, seeds, heartbeat=WindowProgress())
            except WindowComplete:
                pass
            else:
                raise ValueError('Expected a bounded sampling window')
            sample_seconds = time.monotonic()-started
            if window['environment_steps'] < args.environments*26:
                raise ValueError('Sampling window did not exercise all requested slots')
            runner.policy.get_actions = original
            workers = read_json(out/'sampling/pool_00001/workers.json')
            sample_peak = max(row['cgroup_memory_bytes'] for row in rows)
            projected = int(sample_peak*1.25 + args.environments*190*2**20)
            if projected > args.memory_limit_gib*2**30:
                raise ValueError('Projected retained-trajectory memory exceeds the experiment limit')
            cross_batch_replay=runner.replay_metrics(retained,microbatch=args.microbatch,heartbeat=hb)
            if cross_batch_replay['max_logp_error']>.002 or not cross_batch_replay['decisions']:
                raise ValueError(f'{args.environments}-to-{args.microbatch} behavior likelihood replay failed')
            del retained
            sampling = dict(window=window, seconds=sample_seconds, peak_cgroup_memory_bytes=sample_peak,
                projected_full_memory_bytes=projected, memory_limit_gib=args.memory_limit_gib,
                model_copies=workers['model_copies'], environment_count=workers['environment_count'],
                environment_processes=workers['environment_processes'],
                repeated_forward_and_rng_exact=repeated, full_trajectories_tested=False,
                replay=cross_batch_replay)
            atomic_json(out/'sampling_capacity.json', sampling)
            runner.cache.clear(); torch.cuda.empty_cache()
            hb.update(event='loading_dense_capacity_fixture')
            template = torch.load(args.trajectories, map_location='cpu', weights_only=False)
            group, dense = dense_window(template, args.microbatch,
                lambda graph: tensor_bytes(graph.to_dict()), span=32, tbptt=m['recipe']['tbptt'])
            del template
            if dense['active_microbatch'] != args.microbatch:
                raise ValueError('Dense capacity fixture does not fill the requested microbatch')
            base = read_baseline(m)
            torch.cuda.reset_peak_memory_stats()
            with runner.cache.group():
                # Retained historical actions are forced only to exercise the
                # graph/GRU memory shape. They are never formal on-policy data.
                likelihood = runner.replay_metrics(group, microbatch=args.microbatch, heartbeat=hb)
                unused = max(0, runner.cache.limit-runner.cache.bytes)
                ballast = torch.empty(unused, dtype=torch.uint8, device=runner.device)
                times = []
                for _ in range(2):
                    gradient_mode(runner.policy.ac)
                    runner.policy.actor_optimizer.zero_grad(set_to_none=True)
                    runner.policy.critic_optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(); started = time.monotonic()
                    runner._backward(group, base['costs'], microbatch=args.microbatch, heartbeat=hb)
                    torch.cuda.synchronize(); times.append(time.monotonic()-started)
                    gradients = [torch.isfinite(param.grad).all() for param in runner.policy.ac.parameters()
                                 if param.grad is not None]
                    if not gradients or not torch.stack(gradients).all():
                        raise ValueError('Nonfinite or missing dense-window gradients')
                update = dict(window=dense, backward_seconds=times, finite_gradients=True,
                    optimizer_updates=0, full_update_tested=False, cache_reservation_bytes=unused,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), replay=likelihood)
                del ballast
            result = dict(passed=True, scope='requested_capacity_window_v1', parameters=parameters,
                architecture=VERSION, rng_contract=RNG_CONTRACT, sampling=sampling, update=update,
                environment_parity=parity,
                parent_manifest_sha256=m['manifest_sha256'], checkpoint_sha256=commit['checkpoint_sha256'],
                next_batch=commit['next_batch'], checkpoint_restore_exact=restored['loaded_training_state_exact'],
                cuda_rng_restored_exact=restored['cuda_rng_checked'], updates_executed=0,
                code_files={name:digest_file(ROOT/name) for name in CODE_FILES},
                fixture=dict(path=str(args.trajectories.resolve()),sha256=digest_file(args.trajectories)),
                full_parameter_validation='performed by the guarded formal training batches')
            if subset_proof is not None:
                result['training_subset'] = subset_proof
                relative = 'onpolicy/utils/stage3_b0_subset.py'
                result['code_files'][relative] = digest_file(ROOT/relative)
            result['peak_gpu_mib'] = max(row['gpu']['used_mib'] for row in rows)
            result['gpu_total_mib'] = rows[0]['gpu']['total_mib']
            if result['peak_gpu_mib'] > result['gpu_total_mib']-6144:
                raise ValueError('Requested capacity exceeded GPU headroom')
            atomic_json(out/'result.json', result, overwrite=False)
            hb.update(status='completed', event='requested_capacity_passed')
    finally:
        if runner is not None:
            runner.close()
        stop.set(); monitor.join(timeout=10)


if __name__ == '__main__':
    main()
