"""Start seed3 now and migrate the other two at complete epoch boundaries."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil

RUN=Path(__file__).resolve().parent
SOURCE=RUN/'source'
MANIFEST=json.loads((RUN/'manifest.json').read_text())
OLD=Path(MANIFEST['transition']['previous_run'])
PREVIOUS=json.loads((RUN/'previous_manifest.json').read_text())
PYTHON=next(iter(MANIFEST['commands'].values()))['argv'][0]
SOCKET=(RUN/'socket.txt').read_text().strip()
OLD_UNIT='hkbz-stage2-h3f4-seeds123-20260920-r1-gpu0.service'
START=time.time()
state={'status':'starting','started_unix_time':START,'jobs':{},'migrations':{},'pending':[],
       'controller_pid':os.getpid(),'validation_cases':120,'legacy_scheduler_suspended':False}
owned=[];handles=[];legacy_captured=set();migrations={};trainers={};evaluator=None


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def save():
    state['updated_unix_time']=time.time();write(RUN/'suite_status.json',state)


def verify_pid(pid,fragment):
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if fragment not in cmd:raise RuntimeError(f'PID {pid} no longer identifies the expected process')
    return cmd


def affinity_tree(root,cpus):
    rows=subprocess.check_output(['ps','-e','-o','pid=,ppid='],text=True).splitlines()
    edges={}
    for row in rows:
        pid,parent=map(int,row.split());edges.setdefault(parent,[]).append(pid)
    todo=[root];pids=[]
    while todo:
        pid=todo.pop();pids.append(pid);todo.extend(edges.get(pid,[]))
    for pid in pids:
        for task in Path(f'/proc/{pid}/task').glob('*'):
            try:os.sched_setaffinity(int(task.name),cpus)
            except ProcessLookupError:pass
    return pids


def cpuset(value):
    result=set()
    for item in value.split(','):
        limits=item.split('-');result.update(range(int(limits[0]),int(limits[-1])+1))
    return result


def spawn(command,cpus,log,cpu_only=False):
    handle=(RUN/'service_logs'/log).open('x');handles.append(handle)
    env=os.environ.copy()
    if cpu_only:env['CUDA_VISIBLE_DEVICES']=''
    p=subprocess.Popen(['taskset','-c',cpus,*command],cwd=SOURCE,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    owned.append(p);return p


def trainer(seed):
    key=f'H3F4_stage1seed{seed}_bcseed11';cpus=MANIFEST['resource_contract']['train_cpu_sets'][seed-1]
    command=[PYTHON,'-B','-u',str(SOURCE/'onpolicy/scripts/train/run_stage2_resource_manifest_trial.py'),
             '--manifest',str(RUN/'manifest.json'),'--command-key',key,'--gpu','0','--cpu-set',cpus,
             '--shared-eval-socket',SOCKET,'--shared-eval-cpu-set',MANIFEST['resource_contract']['eval_cpu_set'],
             '--eval-workers','12','--record',str(RUN/'records'/f'{key}.json')]
    p=spawn(command,cpus,f'{key}.log');trainers[seed]=p
    state['jobs'][key]={'status':'running','pid':p.pid,'started_unix_time':time.time(),'cpu_set':cpus,'stage1_seed':seed};save()


def original_run(seed):
    key=f'H3F4_stage1seed{seed}_bcseed11'
    return OLD/'source/onpolicy/scripts/results/HKBZ/simple/gnn_mappo'/PREVIOUS['commands'][key]['experiment_name']/'run1'


def capture_if_ready(seed):
    run=original_run(seed);source=run/'models/checkpoint_BCEpoch1.pt'
    required=[source,run/'models/checkpoint_PreSupervised.pt',run/'evaluations/bc_epoch_1.json',
              run/'logs/request_ready_cases_epoch1.json',run/'logs/request_ready_epoch_metrics.jsonl']
    if not all(p.is_file() for p in required):return False
    try:rows=[json.loads(x) for x in required[-1].read_text().splitlines() if x.strip()]
    except json.JSONDecodeError:return False
    rows=[row for row in rows if row['device_bc_epoch']==1]
    if len(rows)!=1:return False
    import torch
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    if checkpoint['stage2_bc_state']['completed_epochs']!=1 or not checkpoint['stage2_bc_state']['at_epoch_boundary']:
        raise RuntimeError('Incomplete epoch checkpoint')
    assert checkpoint.get('supervised_optimizer') and checkpoint['stage2_bc_state'].get('rng')
    assert checkpoint['resource_lookahead_contract']['device_future_intent_horizon']==3
    assert checkpoint['source_m2_sha256']==MANIFEST['stage1_parents'][str(seed)]['sha256']
    dest=RUN/'migration'/f'seed{seed}'/'captured';dest.mkdir(parents=True,exist_ok=False)
    for p in required[:-1]:shutil.copy2(p,dest/p.name)
    (dest/'request_ready_epoch_metrics.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
    pid=MANIFEST['transition']['legacy_trainers'][str(seed)]
    command=verify_pid(pid,PREVIOUS['commands'][f'H3F4_stage1seed{seed}_bcseed11']['experiment_name'])
    assert os.getpgid(pid)==pid
    progress=json.loads((run/'run_status.json').read_text())
    record={'captured_unix_time':time.time(),'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'checkpoint_epoch':1,'optimizer_rng_present':True,'legacy_pid':pid,
            'legacy_progress_at_stop':{k:progress.get(k) for k in ['device_bc_epoch','device_bc_rollout','device_bc_env_step','event']},
            'partial_epoch2_work_policy':'Any updates after the captured Epoch1 boundary are discarded; resume exactly from the saved Epoch1 optimizer/RNG.',
            'legacy_command':command}
    write(dest/'capture.json',record)
    os.killpg(pid,signal.SIGTERM)
    legacy_captured.add(seed);state['migrations'][str(seed)]={'status':'captured_epoch1','capture':str(dest/'capture.json')};save()
    p=spawn([PYTHON,'-B','-u',str(RUN/'migrate.py'),'--seed',str(seed)],'30-31,94-95',f'migrate_seed{seed}.log',cpu_only=True)
    migrations[seed]=p;state['migrations'][str(seed)].update(status='validation120_rebaseline',pid=p.pid);save()
    return True


def main():
    global evaluator
    if (RUN/'suite_status.json').exists():raise RuntimeError('Already attempted; preserve artifacts')
    for name in ['service_logs','records']: (RUN/name).mkdir(exist_ok=True)
    for filename,digest in json.loads((RUN/'launch_spec.json').read_text())['files'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
    for relative,digest in MANIFEST['code_fingerprint'].items():
        assert hashlib.sha256((SOURCE/relative).read_bytes()).hexdigest()==digest,relative
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    assert os.sched_getaffinity(0)==cpuset('0-31,64-95')
    controller=MANIFEST['transition']['legacy_controller_pid']
    verify_pid(controller,'run_stage2_bc_injection_suite.py')
    os.kill(controller,signal.SIGSTOP);state['legacy_scheduler_suspended']=True
    write(OLD/'active_transition.json',{'superseded_by':str(RUN),'authorized_changes':['seed3 immediate','Validation120'],
                                      'legacy_scheduler_suspended':True,'unix':time.time()})
    affinity={}
    for seed in (1,2):
        pid=MANIFEST['transition']['legacy_trainers'][str(seed)]
        verify_pid(pid,f'_stage1seed{seed}_bcseed11')
        affinity[str(seed)]=affinity_tree(pid,cpuset(MANIFEST['resource_contract']['train_cpu_sets'][seed-1]))
    state['legacy_rebound_pids']=affinity;save()
    os.sched_setaffinity(0,cpuset('30-31,94-95'))
    evaluator=spawn([PYTHON,'-B','-u',str(SOURCE/'onpolicy/scripts/train/shared_hkbz_evaluator.py'),
                     '--source-command-json',str(RUN/'evaluator_source.json'),'--socket-path',SOCKET,
                     '--cpu-pool',MANIFEST['resource_contract']['eval_cpu_set'],'--run-dir',str(RUN/'shared_evaluator'),
                     '--eval-dataset-dir',MANIFEST['evaluation_contract']['dataset'],'--max-eval-cases','120',
                     '--cuda-memory-fraction','0.10'],MANIFEST['resource_contract']['eval_cpu_set'],'evaluator.log')
    state['evaluator_pid']=evaluator.pid;save()
    deadline=time.monotonic()+600
    while not Path(SOCKET).exists():
        if evaluator.poll() is not None:raise RuntimeError('Validation120 evaluator failed during startup')
        if time.monotonic()>deadline:raise TimeoutError('Validation120 evaluator startup timeout')
        time.sleep(1)
    trainer(3)
    state['status']='running';save();last=0;old_stop_requested=False
    while True:
        if evaluator.poll() is not None:raise RuntimeError('Validation120 evaluator exited')
        for seed in (2,1):
            if seed not in legacy_captured:capture_if_ready(seed)
        if len(legacy_captured)==2 and not old_stop_requested:
            result=subprocess.run(['systemctl','--no-ask-password','stop','--no-block',OLD_UNIT],capture_output=True,text=True,timeout=15)
            state['legacy_service_stop']={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr};old_stop_requested=True;save()
        for seed,p in list(migrations.items()):
            if p.poll() is None:continue
            if p.returncode:raise RuntimeError(f'Seed{seed} validation migration failed; no auto-retry')
            admitted=json.loads((RUN/'migration'/f'seed{seed}'/'status.json').read_text())
            assert admitted['production_resume_loader_passed'] and admitted['model_optimizer_rng_bitwise_equal']
            assert hashlib.sha256(Path(admitted['target_checkpoint']).read_bytes()).hexdigest()==admitted['target_sha256']
            state['migrations'][str(seed)]={'status':'resuming_epoch2','admission':admitted}
            trainer(seed);del migrations[seed]
        for seed,p in trainers.items():
            key=f'H3F4_stage1seed{seed}_bcseed11';job=state['jobs'][key]
            status_path=SOURCE/'onpolicy/scripts/results/HKBZ/simple/gnn_mappo'/MANIFEST['commands'][key]['experiment_name']/'run1/run_status.json'
            if status_path.exists():
                try:job['progress']=json.loads(status_path.read_text())
                except json.JSONDecodeError:pass
            if p.poll() is not None and job['status']=='running':
                job.update(status='completed' if p.returncode==0 else 'failed',exit_code=p.returncode,ended_unix_time=time.time())
                if p.returncode:raise RuntimeError(f'Seed{seed} trainer failed; no auto-retry')
        if len(trainers)==3 and all(p.poll()==0 for p in trainers.values()):break
        if time.time()-START>48*3600:raise TimeoutError('48 hour hard timeout')
        if time.monotonic()-last>=30:
            last=time.monotonic();save();print('[Validation120]',{k:v['status'] for k,v in state['jobs'].items()},'migrating',list(migrations),flush=True)
        time.sleep(1)
    save()
    result=subprocess.run([PYTHON,'-B',str(SOURCE/'onpolicy/scripts/train/analyze_stage2_bc_injection.py'),'--manifest',str(RUN/'manifest.json')],cwd=SOURCE)
    if result.returncode:raise RuntimeError('Final integrity audit failed')
    state.update(status='completed',ended_unix_time=time.time());save()


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    try:main()
    except BaseException as e:
        state.update(status='failed',error=repr(e),ended_unix_time=time.time());save();raise
    finally:
        for p in reversed(owned):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
        for h in handles:h.close()
