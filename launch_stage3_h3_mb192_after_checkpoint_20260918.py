#!/usr/bin/env python3
"""Schedule the prepared checkpoint-boundary update; training keeps running meanwhile."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import shlex
import subprocess
import time

WORKSPACE=Path(__file__).resolve().parent
RUN=WORKSPACE/'result/hkbz_train_logs/stage3_h3_r0e8_fresh_20260918_r6_env384_mb192_gpu0'
UNIT='hkbz-stage3-h3-r0e8-fresh-20260918-r6-env384-mb192-gpu0.service'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    request_path=RUN/'switch_request.json'
    q=json.loads(request_path.read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    identity=hashlib.sha256(json.dumps({k:v for k,v in q.items() if k!='request_sha256'},
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (identity!=q['request_sha256'] or q['root']!=str(RUN) or q['after_epoch']!=1
            or q.get('target_profile')!='single_parent_env384_mb192_v1'
            or q.get('requested_minibatch')!=192 or q.get('requested_microbatch')!=192
            or q['resources']['gpu']!=0 or q['resources']['cpus']!='0-31,64-95'):
        raise RuntimeError('Prepared switch differs from the authorized checkpoint-boundary update')
    for record in [q[k] for k in ('origin','tests','plan')]:
        if sha(record['path'])!=record['sha256']:raise RuntimeError('Bound input changed')
    for name,expected in q['source_files'].items():
        if sha(Path(q['source_root'])/name)!=expected:raise RuntimeError('Prepared source changed: '+name)
    user=pwd.getpwnam('fanyx')
    properties={'Type':'exec','User':user.pw_name,'Group':str(user.pw_gid),
        'CPUAffinity':'0-31 64-95','AllowedCPUs':'0-31 64-95',
        'MemoryHigh':'infinity','MemoryMax':'infinity','MemorySwapMax':'infinity',
        'ManagedOOMPreference':'omit','ManagedOOMMemoryPressure':'auto','ManagedOOMSwap':'auto',
        'OOMPolicy':'continue','TasksMax':'4096','KillMode':'control-group','TimeoutStopSec':'45',
        'RuntimeMaxSec':'120h','WorkingDirectory':q['source_root'],
        'StandardOutput':'append:'+str(RUN/'controller.log'),'StandardError':'inherit'}
    env=dict(CUDA_VISIBLE_DEVICES=q['resources']['gpu_uuid'],HKBZ_STAGE3_WORKSPACE_ROOT=str(WORKSPACE),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1')
    command=['systemd-run','--system','--no-ask-password','--unit='+UNIT,
             '--description=Stage3 wait for epoch1 checkpoint then continue env384 mb192 fixed8 GPU0']
    command+=['--property='+k+'='+v for k,v in properties.items()]
    command+=['--setenv='+k+'='+v for k,v in env.items()]
    command+=[q['python'],'-B','-u',str(Path(q['source_root'])/'onpolicy/scripts/train/stage3_h3_checkpoint_switch.py'),str(request_path)]
    if args.dry_run:
        print(shlex.join(command));return
    state=subprocess.check_output(['systemctl','show',UNIT,'-p','ActiveState','--value'],text=True).strip()
    if state in ('active','activating','deactivating'):
        raise RuntimeError('The scheduled continuation service is already active')
    if (RUN/'switch_status.json').exists() or (RUN/'manifest.json').exists():
        raise RuntimeError('This request has already been attempted; inspect its state before retrying')
    pid=q['controller_pid']
    stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    if stat!=q['controller_start_ticks'] or os.fsencode(q['origin']['path']) not in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0'):
        raise RuntimeError('The original controller has changed')
    log=RUN/'controller.log';log.touch(exist_ok=True)
    if os.geteuid()==0:os.chown(log,user.pw_uid,user.pw_gid)
    result=subprocess.run(command,capture_output=True,text=True)
    receipt=RUN/'launch_attempt.json'
    receipt.write_text(json.dumps(dict(unix=time.time(),unit=UNIT,command=command,exit_code=result.returncode,
        stdout=result.stdout,stderr=result.stderr,request_sha256=identity),indent=2)+'\n')
    if os.geteuid()==0:os.chown(receipt,user.pw_uid,user.pw_gid)
    print(result.stdout,end='');print(result.stderr,end='')
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
