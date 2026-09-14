#!/usr/bin/env python3
"""Bounded GPU0 capacity sweep; every failed probe stays in its own process."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import shutil
import sys
import threading
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temporary.replace(path)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(2**20),b''):result.update(block)
    return result.hexdigest()


def gpu():
    row=subprocess.check_output(['nvidia-smi','-i','GPU-744c1334-98c8-5318-e799-7ad15eea1fbf',
        '--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    total,used,util=map(int,row.strip().split(','))
    return dict(total_mib=total,used_mib=used,utilization=util)


def cgroup_memory():
    relative=next(row.split(':',2)[2] for row in Path('/proc/self/cgroup').read_text().splitlines()
                  if row.startswith('0::'))
    return int((Path('/sys/fs/cgroup')/relative.lstrip('/')/'memory.current').read_text())


def dense_window(template,width,graph_bytes,*,span=32,tbptt=8):
    """Match the original aligned TBPTT block with the largest joint graphs.

    Retained behavior hidden states initialize this diagnostic window; this is
    never used for a PPO optimizer step or for the formal training schedule.
    """
    repeated=[template[i%len(template)] for i in range(width)]
    sizes=[[graph_bytes(state['graph']) for state in trajectory['states']] for trajectory in template]
    length=max(len(trajectory['states']) for trajectory in repeated)
    scores=[sum(sizes[i%len(template)][step] for i in range(width)
                if step<len(sizes[i%len(template)])) for step in range(length)]
    peak=max(range(0,length,tbptt),key=lambda first:sum(scores[first:first+tbptt]))
    start=min(peak,max(0,length-span))//tbptt*tbptt
    group=[copy.deepcopy(dict(trajectory,states=trajectory['states'][start:start+span]))
           for trajectory in repeated if start<len(trajectory['states'])]
    return group,dict(start_step=start,peak_tbptt_step=peak,requested_microbatch=width,
        active_microbatch=len(group),events=sum(len(t['states']) for t in group),
        graph_bytes_in_peak_tbptt=sum(scores[peak:peak+tbptt]))


def imports(args):
    m=read(args.manifest)
    sys.path.insert(0,m['source_root'])
    sys.path.insert(0,str(args.module.parent))
    import importlib
    extension=importlib.import_module(args.module.stem)
    from onpolicy.utils.stage3_b_shared_b0 import verify_manifest
    verify_manifest(m,inputs=True)
    return m,extension


def memory_monitor(output,stop,rows):
    import psutil
    own=psutil.Process()
    while not stop.is_set():
        family=[own]+own.children(recursive=True)
        rss=0
        for process in family:
            try:rss+=process.memory_info().rss
            except psutil.NoSuchProcess:pass
        row=dict(unix=time.time(),rss_bytes=rss,cgroup_memory_bytes=cgroup_memory(),
                 gpu=gpu(),available_bytes=psutil.virtual_memory().available)
        rows.append(row)
        with (output/'resources.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
        stop.wait(2)


def probe(args):
    m,extension=imports(args)
    import torch
    from onpolicy.runner.shared.stage3_research_engine import gradient_mode
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import read_baseline
    params=read(args.parameters)
    output=args.output.resolve();output.mkdir(parents=True)
    rows,stop=[],threading.Event()
    monitor=threading.Thread(target=memory_monitor,args=(output,stop,rows),daemon=True)
    monitor.start()
    runner=None
    try:
        state=gpu()
        import psutil
        memory_budget=min(params.get('memory_limit_gib',150)*2**30,
                          psutil.virtual_memory().available+cgroup_memory()-36*2**30)
        # Bound this process before allocating any probe tensors; reserve both
        # the six-GiB study headroom and two GiB for the production validator.
        torch.cuda.set_per_process_memory_fraction((state['total_mib']-8192)/state['total_mib'],0)
        runner=extension.ThroughputEngine(m['frozen_manifest']['path'],config=m['recipe'],
            create_pool=False,runtime=dict(input_cache_mib=params.get('cache_mib',4096)))
        runner.resume(args.reference/'initial.pt',manifest_sha256=m['manifest_sha256'],allow_diagnostic=True)
        if params['kind']=='sample':
            n,lanes=params['environments'],params['lanes']
            runner.sampling_lanes=extension.SamplingLanes(m['frozen_manifest']['path'],m['recipe'],
                output/'lanes',lanes=lanes,environment_count=n,affinity=os.sched_getaffinity(0))
            cases=(m['canary_cases']*((n+7)//8))[:n]
            seeds=[m['recipe']['seed']+10000+i%32 for i in range(n)]
            result=runner.sampling_lanes.collect(runner,cases,seeds,probe_steps=26)
            result['events_per_second']=result['events']/result['measured_seconds']
            result['peak_rss_bytes']=max((r['rss_bytes'] for r in rows),default=0)
            result['peak_cgroup_memory_bytes']=max((r['cgroup_memory_bytes'] for r in rows),default=0)
            # Conservative full-trajectory allowance, calibrated by the actual
            # 5.5-GiB retained 32-visit fixture. RSS shared pages are overcounted.
            result['projected_full_memory_bytes']=int(result['peak_cgroup_memory_bytes']*1.25+n*190*2**20)
            result['full_memory_budget_bytes']=memory_budget
            result['admitted']=result['projected_full_memory_bytes']<memory_budget
        else:
            from onpolicy.utils.stage3_performance import tensor_bytes
            template=torch.load(args.trajectories,map_location='cpu',weights_only=False)
            width=params['microbatch']
            group,window=dense_window(template,width,lambda graph:tensor_bytes(graph.to_dict()))
            del template
            base=read_baseline(m)
            torch.cuda.reset_peak_memory_stats()
            with runner.cache.group():
                torch.cuda.synchronize();started=time.monotonic()
                before=runner.replay_metrics(group,microbatch=width)
                torch.cuda.synchronize();replay_seconds=time.monotonic()-started
                if before['max_logp_error']>.002:raise ValueError('Capacity likelihood replay differs')
                unused=max(0,runner.cache.limit-runner.cache.bytes)
                ballast=torch.empty(unused,dtype=torch.uint8,device='cuda:0')
                times=[]
                for _ in range(2):
                    gradient_mode(runner.policy.ac)
                    runner.policy.actor_optimizer.zero_grad(set_to_none=True)
                    runner.policy.critic_optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize();started=time.monotonic()
                    runner._backward(group,base['costs'],microbatch=width)
                    torch.cuda.synchronize();times.append(time.monotonic()-started)
                result=dict(admitted=True,events=window['events'],window=window,replay_seconds=replay_seconds,
                    backward_seconds=statistics.median(times),cache_reservation_bytes=unused,
                    graph_bytes_per_event=runner.cache.bytes/window['events'],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),actor_updates=0)
                del ballast
        result.update(parameters=params,completed=True,module_sha256=sha(args.module),
            numerical_window_only=True,peak_gpu_mib=max((r['gpu']['used_mib'] for r in rows),default=0))
        if result['peak_gpu_mib']>state['total_mib']-6144:result['admitted']=False
        write(output/'result.json',result)
    except torch.cuda.OutOfMemoryError as error:
        write(output/'result.json',dict(completed=True,admitted=False,capacity_limit='cuda_allocator',
            parameters=params,error=str(error),module_sha256=sha(args.module)))
    except RuntimeError as error:
        if 'CUDA out of memory' not in str(error) and 'CUDA error: out of memory' not in str(error):
            write(output/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
            raise
        write(output/'result.json',dict(completed=True,admitted=False,capacity_limit='sampling_cuda_allocator',
            parameters=params,error=str(error),module_sha256=sha(args.module)))
    except BaseException as error:
        write(output/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        if runner is not None:runner.close()
        stop.set();monitor.join(timeout=10)


def run_child(args,parameters,output):
    output=Path(output)
    if args.reuse_windows:
        previous=args.reuse_windows/output.name/'result.json'
        if previous.exists():
            result=read(previous)
            if (result.get('completed')
                    and result.get('parameters')==parameters and result.get('module_sha256')==sha(args.module)):
                if isinstance(result.get('projected_full_memory_bytes'),(int,float)):
                    import psutil
                    current_budget=min(parameters.get('memory_limit_gib',150)*2**30,
                        psutil.virtual_memory().available+cgroup_memory()-36*2**30)
                    result['admitted']=result['projected_full_memory_bytes']<current_budget
                    result['readmission_memory_budget_bytes']=current_budget
                    if not result['admitted']:result['capacity_limit']='projected_full_memory'
                output.mkdir()
                result['reused_window']=dict(path=str(previous),sha256=sha(previous))
                write(output/'result.json',result)
                return result
    parameter_path=output.with_suffix('.parameters.json');write(parameter_path,parameters)
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'probe',str(args.manifest),
        '--module',str(args.module),'--reference',str(args.reference),'--trajectories',str(args.trajectories),
        '--parameters',str(parameter_path),'--output',str(output)]
    return run_command(args,command,output.with_suffix('.log'),result_path=output/'result.json',timeout=3600)


def run_command(args,command,log_path,*,result_path,timeout=21600):
    import psutil
    import signal
    parameters=read(command[command.index('--parameters')+1]) if 'probe' in command else None
    with Path(log_path).open('x') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic()
        while process.poll() is None:
            write(args.output/'status.json',dict(status='running',phase=Path(log_path).stem,
                child_pid=process.pid,elapsed_seconds=time.monotonic()-start,heartbeat_unix=time.time(),
                command=command))
            usage=psutil.virtual_memory()
            try:
                own=psutil.Process(process.pid)
                family=[own]+own.children(recursive=True)
                rss=sum(p.memory_info().rss for p in family if p.is_running())
            except psutil.NoSuchProcess:
                rss=0
            charged=cgroup_memory()
            projected=None
            memory_budget=min(150*2**30,usage.available+charged-36*2**30)
            if parameters and parameters['kind']=='sample':
                projected=int(charged*1.25+parameters['environments']*190*2**20)
            projection_limit=projected is not None and projected>memory_budget
            limit=charged>150*2**30 or usage.available<32*2**30 or projection_limit
            if limit or time.monotonic()-start>timeout:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=15)
                if limit and 'probe' in command:
                    for _ in range(45):
                        if gpu()['used_mib']<1024 and psutil.virtual_memory().available>36*2**30:break
                        write(args.output/'status.json',dict(status='running',phase='capacity_rejection_cleanup',
                            heartbeat_unix=time.time(),rejected_probe=str(log_path)))
                        time.sleep(2)
                    else:raise RuntimeError('Rejected capacity probe did not release its GPU0 resources')
                    result=dict(completed=True,admitted=False,
                        capacity_limit='projected_full_memory' if projection_limit else 'host_memory_guard',
                        parameters=parameters,module_sha256=sha(args.module),
                        rss_bytes=rss,cgroup_memory_bytes=charged,available_bytes=usage.available,
                        projected_full_memory_bytes=projected,full_memory_budget_bytes=memory_budget)
                    write(result_path,result)
                    return result
                raise TimeoutError(f'Capacity probe timed out: {log_path}')
            time.sleep(5)
    if process.returncode or not Path(result_path).exists():
        raise RuntimeError(f'Probe failed with exit {process.returncode}: {log_path}')
    return read(result_path)


def full_trial(args,params,index):
    directory=args.output/f'full_{index}_{params["environments"]}'
    directory.mkdir()
    fixture_bytes=Path(args.trajectories).stat().st_size*params['environments']//32
    if shutil.disk_usage(directory).free<fixture_bytes+20*2**30:
        raise RuntimeError('Insufficient temporary disk for this complete capacity fixture')
    common=[sys.executable,'-B','-u',str(args.launcher),None,str(args.manifest),
        '--module',str(args.module),'--reference',str(args.reference),'--visits',str(params['environments']),
        '--cache-mib',str(params['cache_mib']),'--microbatch',str(params['microbatch']),'--exclusive-gpu']
    sample_command=common.copy();sample_command[4]='sample'
    sample_command+=['--lanes',str(params['lanes']),'--output',str(directory/'sample')]
    try:
        sample=run_command(args,sample_command,directory/'sample.log',result_path=directory/'sample/result.json')
    except RuntimeError as error:
        if 'AssertionError' not in (directory/'sample.log').read_text(errors='replace'):
            raise
        result=dict(passed=False,parameters=params,rejection='complete sampling numerical comparison',
                    evidence=str(directory/'sample.log'),error=str(error))
        write(directory/'result.json',result)
        return result
    update_command=common.copy();update_command[4]='update'
    update_command+=['--trajectories',str(directory/'sample/trajectories.pt'),'--output',str(directory/'update')]
    try:
        update=run_command(args,update_command,directory/'update.log',result_path=directory/'update/result.json')
    except RuntimeError as error:
        evidence=(directory/'update.log').read_text(errors='replace')
        if not any(marker in evidence for marker in ('OutOfMemoryError','CUDA out of memory','AssertionError','PPOContractError')):
            raise
        result=dict(passed=False,parameters=params,rejection='complete update capacity or numerical comparison',
                    evidence=str(directory/'update.log'),error=str(error),sample_proof=str(directory/'sample/result.json'))
        write(directory/'result.json',result)
        temporary=directory/'sample/trajectories.pt'
        write(directory/'temporary_fixture.json',dict(path=str(temporary),bytes=temporary.stat().st_size,
            removed_after_rejected_diagnostic=True,regeneration_proof=result['sample_proof']))
        temporary.unlink()
        return result
    result=dict(passed=sample['passed'] and update['passed'],parameters=params,
        sample_proof=str(directory/'sample/result.json'),update_proof=str(directory/'update/result.json'),
        sample_seconds=sample['seconds'],update_seconds=update['seconds'],
        seconds_per_visit=(sample['seconds']+update['seconds'])/params['environments'])
    write(directory/'result.json',result)
    # These are this tuner invocation's temporary retained inputs only. Keep
    # checkpoint, exact rollout summaries, timings and all numerical evidence.
    temporary=directory/'sample/trajectories.pt'
    write(directory/'temporary_fixture.json',dict(path=str(temporary),bytes=temporary.stat().st_size,
        sha256=sha(temporary),removed_after_complete_update=True))
    temporary.unlink()
    return result


def choose_complete_trial(trials,**identity):
    passed=[trial for trial in trials if trial.get('passed')]
    if not passed:raise ValueError('No complete trial passed the numerical gate')
    winner=copy.deepcopy(min(passed,key=lambda r:r['seconds_per_visit']))
    winner.update(identity,trials=copy.deepcopy(trials))
    return winner


def controller(args):
    args.output.mkdir(parents=True)
    parent=read(args.manifest)
    gate=Path(parent['root'])/'throughput_acceleration/update16/result.json'
    while not gate.exists():
        write(args.output/'status.json',dict(status='waiting',phase='matched_global32_update16',
            heartbeat_unix=time.time()))
        failed=Path(parent['root'])/'throughput_acceleration/update16/status.json'
        if failed.exists() and read(failed).get('status')=='failed':
            raise RuntimeError('Initial matched update16 did not pass')
        time.sleep(5)
    if not read(gate)['passed']:raise ValueError('Initial matched update16 failed')
    # The user authorized this new capacity study. Preserve the old diagnostic
    # checkpoint and all baselines; never discard a committed training batch.
    if list((Path(parent['root'])/'commits').glob('batch_*.json')):
        raise RuntimeError('Formal commits appeared; a batch-boundary migration is required')
    old_unit=(Path(parent['root'])/'last_service_unit.txt').read_text().strip()
    if subprocess.run(['systemctl','--user','is-active','--quiet',old_unit]).returncode==0:
        subprocess.run(['systemctl','--user','stop',old_unit],check=True)
    study_lock=(Path(parent['root'])/'controller.lock').open('a+')
    fcntl.flock(study_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    for _ in range(30):
        if gpu()['used_mib']<1024:break
        time.sleep(2)
    else:raise RuntimeError('GPU0 was not released by the old diagnostic service')
    write(args.output/'transition.json',dict(previous_unit=old_unit,stopped_unix=time.time(),
        formal_commits=0,retained_reference=str(args.reference),parent_manifest_sha256=parent['manifest_sha256'],
        reason='User requested hardware-limit capacity tuning; old diagnostic artifacts retained'))
    write(Path(parent['root'])/'execution_transition.json',dict(status='capacity_tuning',
        tuner_output=str(args.output),previous_training_commits=0,
        new_training_manifest='Published in tuner deployment/launched.json after complete admission'))
    samples=[]
    for lanes in (4,8,16,32):
        p=dict(kind='sample',environments=32,lanes=lanes,memory_limit_gib=150)
        result=run_child(args,p,args.output/f'sample32_lanes{lanes}')
        samples.append(result)
    admitted=[r for r in samples if r['admitted']]
    if not admitted:raise RuntimeError('No safe sampling lane configuration')
    admitted.sort(key=lambda r:r['events_per_second'],reverse=True)
    fastest=admitted[0]
    lanes=fastest['parameters']['lanes']
    environments=[fastest]
    for count in (64,128,256,512):
        level=[]
        for candidate in admitted[:2]:
            lane_count=candidate['parameters']['lanes']
            p=dict(kind='sample',environments=count,lanes=lane_count,memory_limit_gib=150)
            result=run_child(args,p,args.output/f'sample{count}_lanes{lane_count}')
            samples.append(result)
            if result['admitted']:level.append(result)
            elif result.get('capacity_limit') in ('host_memory_guard','projected_full_memory'):break
        if not level:break
        environments.append(max(level,key=lambda r:r['events_per_second']))
    updates=[]
    for width in (8,16,32,64,128,256):
        p=dict(kind='update',microbatch=width,cache_mib=4096,window='dense_aligned32_v1')
        result=run_child(args,p,args.output/f'update_micro{width}_cache4096')
        updates.append(result)
        if not result['admitted']:break
    admitted=[r for r in updates if r['admitted']]
    if not admitted:raise RuntimeError('No safe update microbatch')
    admitted.sort(key=lambda r:(3*r['replay_seconds']+2*r['backward_seconds'])/r['events'])
    update_candidates=[]
    for rank,row in enumerate(admitted):
        width=row['parameters']['microbatch']
        cache=4096
        if rank<2:
            for size in (8192,12288,16384,24576,32768):
                result=run_child(args,dict(kind='update',microbatch=width,cache_mib=size,window='dense_aligned32_v1'),
                    args.output/f'update_micro{width}_cache{size}')
                updates.append(result)
                if not result['admitted']:break
                cache=size
        update_candidates.append(dict(row,selected_cache_mib=cache))
    # Screen by per-visit work. Full trials below decide among the two leading
    # candidates; throughput estimates do not count as admission evidence.
    mean_steps=statistics.mean(r['steps'] for r in read(args.reference/'rollouts.json'))
    candidates=[]
    finalists=[]
    for update in update_candidates:
        width=update['parameters']['microbatch']
        unit=(3*update['replay_seconds']+2*update['backward_seconds'])/update['events']
        choices=[]
        for row in environments:
            count=row['parameters']['environments']
            if count<width:continue
            choices.append(dict(environments=count,lanes=row['parameters']['lanes'],microbatch=width,
                cache_mib=update['selected_cache_mib'],
                estimated_seconds_per_visit=mean_steps/row['events_per_second']+unit*mean_steps))
        candidates.extend(choices)
        if choices:finalists.append(min(choices,key=lambda r:r['estimated_seconds_per_visit']))
    candidates.sort(key=lambda r:r['estimated_seconds_per_visit'])
    write(args.output/'screen.json',dict(samples=samples,updates=updates,candidates=candidates,finalists=finalists,
        ranking_is_estimated=True,global_visit_budget=7680,ppo_epochs=2))
    full=[]
    for i,candidate in enumerate(finalists):
        full.append(full_trial(args,candidate,i))
        if sum(bool(row['passed']) for row in full)>=2:break
    winner=choose_complete_trial(full,parent_manifest_sha256=parent['manifest_sha256'],selected_unix=time.time(),
        selection='minimum measured complete sampling plus PPO2 seconds per visit among finalists',
        runtime_module=str(args.module),runtime_module_sha256=sha(args.module),launcher=str(args.launcher),
        launcher_sha256=sha(args.launcher))
    write(args.output/'selected.json',winner)
    if args.deployment:
        command=[sys.executable,'-B','-u',str(args.deployment),str(args.manifest),
            '--selection',str(args.output/'selected.json'),'--output',str(args.output/'deployment')]
        run_command(args,command,args.output/'deployment.log',result_path=args.output/'deployment/launched.json')
    write(args.output/'status.json',dict(status='completed',phase='selected_and_launched' if args.deployment else 'selected',
        selected=str(args.output/'selected.json'),heartbeat_unix=time.time()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('probe','controller'))
    p.add_argument('manifest',type=Path)
    p.add_argument('--module',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--trajectories',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parameters',type=Path)
    p.add_argument('--launcher',type=Path)
    p.add_argument('--deployment',type=Path)
    p.add_argument('--reuse-windows',type=Path)
    args=p.parse_args()
    try:(probe if args.command=='probe' else controller)(args)
    except BaseException as error:
        write(args.output/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
        write(args.output/'status.json',dict(status='failed',error=str(error),heartbeat_unix=time.time()))
        raise


if __name__=='__main__':main()
