#!/usr/bin/env python3
"""Start the prepared GPU0 experiment as fanyx in a root-owned systemd cgroup.

Use --dry-run to inspect the exact command. If systemd authorization is unavailable,
run this script with sudo. No host-wide oomd setting is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import shlex
import subprocess
import time


WORKSPACE = Path(__file__).resolve().parent
RUN = WORKSPACE/'result/hkbz_train_logs/stage3_h3_r0e8_fresh_20260918_r5_env384_mb128_gpu0'
UNIT = 'hkbz-stage3-h3-r0e8-fresh-20260918-r5-env384-mb128-gpu0.service'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = RUN/'manifest.json'
    m = json.loads(manifest.read_text())
    payload = {k: v for k, v in m.items() if k != 'manifest_sha256'}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()
    if identity != m['manifest_sha256'] or m['root'] != str(RUN):
        raise RuntimeError('Prepared experiment identity changed')
    if (m['recipe']['execution_profile'] != 'single_parent_env384_mb128_fresh_v1'
            or m['recipe']['rollout_workers'] != 384
            or m['recipe']['optimizer_minibatch'] != 128 or m['recipe']['microbatch'] != 128
            or m['recipe']['stopping_policy'] != 'fixed_eight_epochs_numeric_gates_v1'
            or m['recipe']['epochs'] != 8
            or m.get('resume_origin')
            or 'optimizer_resize_after_epoch' in m['recipe']
            or m['checkpoints']['epoch8']['sha256'] != '14d63bb44ee96eea38c993bb7d1d1d1ae5f0aa59f979b14a9337e6f96215a4e6'
            or m['execution_mode'] != 'single'
            or m['recipe']['train_tau'] != .03
            or m['recipe']['memory_protection'] != 'disabled'
            or m['resources']['gpu'] != 0 or m['resources']['cpus'] != '0-31,64-95'):
        raise RuntimeError('Prepared experiment differs from the user request')
    for name, expected in m['source_files'].items():
        if sha(Path(m['source_root'])/name) != expected:
            raise RuntimeError('Frozen source changed: '+name)
    for key in ('tests','plan','parent_manifest','baselines','frozen_iga','frozen_manifest'):
        if sha(m[key]['path']) != m[key]['sha256']:
            raise RuntimeError('Bound input changed: '+key)
    for record in m['checkpoints'].values():
        if sha(record['path']) != record['sha256']:
            raise RuntimeError('Bound checkpoint changed')
    user = pwd.getpwnam('fanyx')
    properties = {
        'Type': 'exec', 'User': user.pw_name, 'Group': str(user.pw_gid),
        'CPUAffinity': '0-31 64-95', 'AllowedCPUs': '0-31 64-95',
        'MemoryHigh': 'infinity', 'MemoryMax': 'infinity', 'MemorySwapMax': 'infinity',
        'ManagedOOMPreference': 'omit', 'ManagedOOMMemoryPressure': 'auto',
        'ManagedOOMSwap': 'auto', 'OOMPolicy': 'continue',
        'TasksMax': '4096', 'KillMode': 'control-group', 'TimeoutStopSec': '45',
        'RuntimeMaxSec': '120h', 'WorkingDirectory': m['source_root'],
        'StandardOutput': 'append:'+str(RUN/'controller.log'), 'StandardError': 'inherit',
    }
    env = dict(CUDA_VISIBLE_DEVICES=m['resources']['gpu_uuid'],
               HKBZ_STAGE3_WORKSPACE_ROOT=str(WORKSPACE),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
    command = ['systemd-run','--system','--no-ask-password','--unit='+UNIT,
               '--description=H3 fresh R0 E8 continuation env384 mb128 fixed8 GPU0']
    command += ['--property='+key+'='+value for key,value in properties.items()]
    command += ['--setenv='+key+'='+value for key,value in env.items()]
    command += [m['python'],'-B','-u',str(Path(m['source_root'])/'onpolicy/scripts/train/run_stage3_h3_continuation.py'),
                'run',str(manifest)]
    if args.dry_run:
        print(shlex.join(command))
        return
    state = subprocess.run(['systemctl','show',UNIT,'-p','ActiveState','--value'],
                           capture_output=True,text=True).stdout.strip()
    if state in ('active','activating','deactivating'):
        raise RuntimeError('The experiment unit is already active: '+state)
    if (RUN/'budget_clock.json').exists():
        raise RuntimeError('This prepared run has already started; inspect its state before any resume')
    if (RUN/'resume_migration.json').exists() or list(RUN.glob('arms/*/commits/*.json')):
        raise RuntimeError('Fresh E8 start must not contain imported continuation commits')
    active = subprocess.check_output(['nvidia-smi','--id='+m['resources']['gpu_uuid'],
        '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if active:
        raise RuntimeError('GPU0 has active compute processes: '+active)
    log = RUN/'controller.log'
    log.touch(exist_ok=True)
    if os.geteuid() == 0:
        os.chown(log,user.pw_uid,user.pw_gid)
    completed = subprocess.run(command,capture_output=True,text=True)
    receipt = RUN/'launch_attempt.json'
    receipt.write_text(json.dumps(dict(unix=time.time(),unit=UNIT,
        manifest_sha256=identity,command=command,exit_code=completed.returncode,
        stdout=completed.stdout,stderr=completed.stderr,
        training_user=user.pw_name,global_oomd_service='unchanged'),indent=2)+'\n')
    if os.geteuid() == 0:
        os.chown(receipt,user.pw_uid,user.pw_gid)
    print(completed.stdout,end='')
    print(completed.stderr,end='')
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
