#!/usr/bin/env python3
"""Freeze the measured B0 capacity amendment, prove resume, then launch GPU0."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def read(path):return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for data in iter(lambda:file.read(2**20),b''):h.update(data)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as file:json.dump(value,file,indent=2,sort_keys=True);file.write('\n')


def copy_evidence(source,target,parent_sha,**changes):
    value=read(source)
    if value.get('manifest_sha256')!=parent_sha:
        raise ValueError(f'Parent evidence changed: {source}')
    value.update(changes)
    value['reused_from']=dict(path=str(source),sha256=sha(source),parent_manifest_sha256=parent_sha,
                              fresh_case_queries=0)
    write(target,value)


def prepare(args):
    import yaml
    import psutil
    parent=read(args.manifest);selected=read(args.selection)
    payload=read(Path(__file__).with_name('deployment_payload.json'))
    if not selected['passed'] or selected['parent_manifest_sha256']!=parent['manifest_sha256']:
        raise ValueError('A fully proved capacity winner is required')
    if digest({k:v for k,v in parent.items() if k!='manifest_sha256'})!=parent['manifest_sha256']:
        raise ValueError('Parent manifest identity changed')
    if list((Path(parent['root'])/'commits').glob('*.json')):
        raise ValueError('Parent already has formal commits; do not replace its training cursor')
    for path,key in ((selected['runtime_module'],'runtime_module_sha256'),(selected['launcher'],'launcher_sha256')):
        if sha(path)!=selected[key]:raise ValueError('Selected runtime source changed')
    p=selected['parameters']
    root=Path(parent['root']).with_name(Path(parent['root']).name.replace('_r2_gpu0','_r3_tuned_gpu0'))
    if root==Path(parent['root']) or root.exists():raise FileExistsError(root)
    root.mkdir()
    shutil.copytree(Path(parent['source_root']),root/'source')
    files=copy.deepcopy(parent['code_files'])
    for relative,checksum in files.items():
        if sha(root/'source'/relative)!=checksum:raise ValueError(f'Parent source changed: {relative}')
    for relative,entry in payload['overlays'].items():
        if sha(entry['path'])!=entry['sha256']:raise ValueError(f'Frozen amendment changed: {relative}')
        target=root/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():target.unlink()
        shutil.copyfile(entry['path'],target);target.chmod(0o444)
        files[relative]=sha(target)
    recipe=copy.deepcopy(parent['recipe'])
    memory_max=min(160,int(psutil.virtual_memory().available/2**30)-36)
    if memory_max<48:raise ValueError('Current host memory cannot cover a new B0 training service')
    memory_high=min(144,int(memory_max*.9))
    recipe.update(global_batch=p['environments'],microbatch=p['microbatch'],
                  memory_high_gib=memory_high,memory_max_gib=memory_max)
    config='onpolicy/config/env_stage3_b_shared_b0.yaml'
    target=root/'source'/config;target.unlink()
    target.write_text(yaml.safe_dump(recipe,sort_keys=False));target.chmod(0o444)
    files[config]=sha(target)
    tests=copy.deepcopy(payload['tests'])
    if sha(tests['path'])!=tests['sha256']:raise ValueError('Amendment test report changed')
    shutil.copyfile(tests['path'],root/'tests.xml');tests['path']=str(root/'tests.xml')
    sys.path.insert(0,str(root/'source'))
    from onpolicy.utils.stage3_b_shared_b0 import schedule,budget,verify_manifest
    from onpolicy.utils.stage3_research import digest_json
    if digest(parent['recipe'])!=digest_json(parent['recipe']):raise ValueError('Canonical identity encoder differs')
    plan=schedule(parent['splits']['train'],recipe['seed'],recipe['global_batch'])
    m=copy.deepcopy(parent)
    m.update(root=str(root),source_root=str(root/'source'),recipe=recipe,tests=tests,code_files=files,
        code_sha256=digest(files),schedule_sha256=digest(plan),created_unix=time.time(),
        resource_amendment='Measured GPU0 capacity; same CPU half; phase-separated policy sampling and training',
        throughput_amendment=dict(parent_manifest=str(args.manifest),parent_manifest_sha256=parent['manifest_sha256'],
            selection=str(args.selection),selection_sha256=sha(args.selection),parameters=p,
            previous_global_batch=32,previous_actor_updates=480,
            reason='User requested the highest measured throughput within hardware capacity',
            original_gate='global32 micro8 versus micro4; may be interrupted before completion',
            replacement_gate='complete selected global update versus original micro8 plus fresh-process exact continuation',
            repeated_capacity_inputs_are_diagnostic_only=True,formal_initialization='frozen Stage2 B0'))
    m['throughput_amendment']['budget']=budget(m)
    m['manifest_sha256']=digest({k:v for k,v in m.items() if k!='manifest_sha256'})
    for relative in ('commits','attempts','diagnostics','validator/requests','validator/results','validator/cases'):
        (root/relative).mkdir(parents=True,exist_ok=True)
    write(root/'manifest.json',m)
    verify_manifest(m,inputs=True)
    copy_evidence(Path(parent['root'])/'baselines.json',root/'baselines.json',parent['manifest_sha256'],
                  manifest_sha256=m['manifest_sha256'])
    copy_evidence(Path(parent['root'])/'zero_update_verified.json',root/'zero_update_verified.json',parent['manifest_sha256'],
                  manifest_sha256=m['manifest_sha256'],baseline_sha256=sha(root/'baselines.json'))
    # Evaluation model, native fixed12 grouping and case order are unchanged.
    from onpolicy.utils.stage3_b_shared_b0 import submit_evaluation,validate_request
    old_result=Path(parent['root'])/'validator/results/B0_AR_validation.json'
    control=read(old_result)
    if not control['completed'] or len(control['rows'])!=120:
        raise ValueError('Original complete AR validation baseline is required')
    ident=submit_evaluation(m,m['source']['path'],0,'validation',decoder='AR',baseline=True)
    request=read(root/'validator/requests'/f'{ident}.json');validate_request(m,request)
    for key in ('checkpoint_sha256','cases_sha256','tau','seed','history','decoder'):
        if request[key]!=control['request'][key]:raise ValueError('Reused AR control contract differs')
    control.update(request=request,reused_from=dict(path=str(old_result),sha256=sha(old_result),fresh_case_queries=0))
    write(root/'validator/results'/f'{ident}.json',control)
    proofs={name:dict(path=selected[name+'_proof'],sha256=sha(selected[name+'_proof'])) for name in ('sample','update')}
    checkpoint=read(Path(selected['update_proof']).with_name('checkpoint.json'))
    runtime=dict(study_manifest_sha256=m['manifest_sha256'],proof_manifest=str(args.manifest),
        proof_manifest_sha256=parent['manifest_sha256'],module=selected['runtime_module'],
        module_sha256=selected['runtime_module_sha256'],launcher=selected['launcher'],
        launcher_sha256=selected['launcher_sha256'],global_batch=p['environments'],ppo_epochs=2,
        sampling_environments=p['environments'],sampling_lanes=p['lanes'],microbatch=p['microbatch'],
        input_cache_mib=p['cache_mib'],release_sampling_before_update=True,cuda_memory_headroom_mib=8192,
        proofs=proofs,diagnostic_checkpoint=checkpoint,tests=tests,
        selection=dict(path=str(args.selection),sha256=sha(args.selection)))
    runtime['runtime_sha256']=digest(runtime)
    write(root/'runtime.json',runtime)
    write(args.output/'prepared.json',dict(manifest=str(root/'manifest.json'),runtime=str(root/'runtime.json'),
        training_started=False,budget=budget(m),baseline_queries_reused=780,zero_queries_reused=180))
    return m,runtime,payload


def run(args):
    args.output.mkdir(parents=True)
    m,runtime,payload=prepare(args)
    root=Path(m['root'])
    from onpolicy.scripts.train.run_stage3_b_shared_b0 import check_junit
    from onpolicy.utils.stage3_b_shared_b0 import verify_admission
    env=dict(os.environ,PYTHONPATH=m['source_root'],CUDA_VISIBLE_DEVICES='')
    tests=[str(root/'source'/p) for p in payload['test_files']]
    subprocess.run([payload['pytest_python'],'-B','-m','pytest','-q','-p','no:cacheprovider',*tests,
        '--junitxml='+str(args.output/'frozen_source_tests.xml')],cwd=m['source_root'],env=env,check=True)
    check_junit(args.output/'frozen_source_tests.xml')
    common=[m['python'],'-B','-u',runtime['launcher'],None,str(root/'manifest.json'),
        '--module',runtime['module'],'--runtime',str(root/'runtime.json')]
    for phase,directory in (('continuation','canary'),('resume-proof','resume_check')):
        command=common.copy();command[4]=phase
        command+=['--output',str(args.output/directory)]
        if phase=='resume-proof':command+=['--canary-dir',str(args.output/'canary')]
        with (args.output/f'{phase}.log').open('x') as log:
            subprocess.run(command,cwd=m['source_root'],stdout=log,stderr=subprocess.STDOUT,check=True)
    proof=read(args.output/'canary/result.json');restored=read(args.output/'resume_check/result.json')
    if not proof['passed'] or not restored['passed']:raise ValueError('Fresh continuation admission failed')
    admission=dict(passed=True,manifest_sha256=m['manifest_sha256'],global_batch=runtime['global_batch'],
        physical_microbatch=runtime['microbatch'],gpu_count=1,tests=m['tests'],
        baseline_sha256=sha(root/'baselines.json'),zero_check_sha256=sha(root/'zero_update_verified.json'),
        canary=str(args.output/'canary/result.json'),canary_sha256=sha(args.output/'canary/result.json'),
        resume_check=str(args.output/'resume_check/result.json'),resume_check_sha256=sha(args.output/'resume_check/result.json'),
        full_batch_seconds=proof['batch_seconds'],gate_amendment=m['throughput_amendment'],
        frozen_source_tests=dict(path=str(args.output/'frozen_source_tests.xml'),sha256=sha(args.output/'frozen_source_tests.xml')))
    write(root/'training_admission.json',admission);verify_admission(m)
    unit=f'hkbz-b0-tuned-{int(time.time())}'
    command=['systemd-run','--user','--collect','--unit='+unit,
        '--property=Type=exec','--property=KillMode=control-group','--property=TimeoutStopSec=45',
        '--property=RuntimeMaxSec=1209600',f'--property=MemoryHigh={m["recipe"]["memory_high_gib"]}G',
        f'--property=MemoryMax={m["recipe"]["memory_max_gib"]}G',
        '--property=MemorySwapMax=0','--property=CPUAffinity=0-31 64-95','--property=AllowedCPUs=0-31,64-95',
        '--property=WorkingDirectory='+m['source_root'],
        '--property=StandardOutput=append:'+str(root/'service.log'),
        '--property=StandardError=append:'+str(root/'service.log')]
    for key in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'PYTHONDONTWRITEBYTECODE','PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','HKBZ_STAGE3_WORKSPACE_ROOT'):
        command.append('--setenv='+key+'='+os.environ[key])
    command += [m['python'],'-B','-u',runtime['launcher'],'resume',str(root/'manifest.json'),
                '--module',runtime['module'],'--runtime',str(root/'runtime.json')]
    subprocess.run(command,check=True)
    (root/'last_service_unit.txt').write_text(unit+'.service\n')
    for _ in range(60):
        status=read(root/'run_status.json') if (root/'run_status.json').exists() else {}
        active=subprocess.run(['systemctl','--user','is-active','--quiet',unit+'.service']).returncode==0
        if not active:raise RuntimeError('Tuned training service exited during launch')
        if status.get('phase')=='train' and status.get('status')=='running':
            write(args.output/'launched.json',dict(launched=True,service=unit+'.service',manifest=str(root/'manifest.json'),
                runtime=str(root/'runtime.json'),status=status,formal_initialization='B0',launched_unix=time.time()))
            parent_root=Path(read(args.manifest)['root'])
            transition=parent_root/'execution_transition.json'
            temporary=transition.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(status='tuned_training_launched',
                service=unit+'.service',manifest=str(root/'manifest.json'),tuner_output=str(args.selection.parent),
                formal_initialization='B0',parent_training_commits=0),indent=2)+'\n')
            temporary.replace(transition)
            return
        time.sleep(2)
    raise TimeoutError('Training did not enter its first collection phase')


def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:run(args)
    except BaseException as error:
        args.output.mkdir(parents=True,exist_ok=True)
        path=args.output/'failure.json'
        if not path.exists():write(path,dict(error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
