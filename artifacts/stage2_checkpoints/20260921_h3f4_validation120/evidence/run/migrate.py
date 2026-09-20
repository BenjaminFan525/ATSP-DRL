"""Re-evaluate a complete BC epoch and migrate only its evaluation metadata."""
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

RUN=Path(__file__).resolve().parent
SOURCE=RUN/'source'
sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
import yaml
from onpolicy.config.config import get_config
from onpolicy.scripts.train.train_hkbz import parse_args
from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner
from onpolicy.utils.stage2_bc_contract import supervised_gate,json_safe
from onpolicy.utils.training_stage import CANONICAL_RESOURCE_JOINT


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(json_safe(v),indent=2)+'\n');q.replace(p)

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def runner_for(args):
    torch.manual_seed(11)
    r=object.__new__(HKBZ_Runner);r.__dict__.update(vars(args));r.all_args=args;r.device=torch.device('cpu')
    r.policy=GNN_MAPPOPolicy(args,yaml.safe_load(Path(args.ac_config).read_text()),device=r.device)
    r.training_stage=CANONICAL_RESOURCE_JOINT;r.stage2_allow_shared_unfreeze=False
    r.device_bc_train_gnn=False;r.stage2_frozen_state_before=None;r.stage2_bc_at_boundary=True
    r.device_bc_dagger_schedule=[float(x) for x in args.device_bc_dagger_schedule.split(',')]
    r.device_bc_dagger_rng=np.random.default_rng(args.seed)
    return r


def main(seed):
    torch.set_num_threads(1)
    manifest=json.loads((RUN/'manifest.json').read_text());old=json.loads((RUN/'previous_manifest.json').read_text())
    key=f'H3F4_stage1seed{seed}_bcseed11';entry=manifest['commands'][key]
    args=parse_args(entry['argv'][2:],get_config());dest=RUN/'migration'/f'seed{seed}'
    original=Path(manifest['transition']['previous_run'])/'source/onpolicy/scripts/results/HKBZ/simple/gnn_mappo'/old['commands'][key]['experiment_name']/'run1'
    source=dest/'captured/checkpoint_BCEpoch1.pt'
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    assert checkpoint['stage2_bc_state']['completed_epochs']==1 and checkpoint['stage2_bc_state']['at_epoch_boundary']
    assert checkpoint['source_m2_sha256']==entry['source']['sha256']
    r=runner_for(args)
    target_contract=r._stage2_bc_run_contract();previous=checkpoint['stage2_bc_run_contract']
    changed={k for k in previous.keys()|target_contract.keys() if previous.get(k)!=target_contract.get(k)}
    assert changed=={'eval_dataset_dir','max_eval_cases'},changed
    assert previous['max_eval_cases']==60 and target_contract['max_eval_cases']==120
    assert r._resource_lookahead_contract()==checkpoint['resource_lookahead_contract']
    state={'status':'validation120_rebaseline','seed':seed,'source_checkpoint':str(source),'source_sha256':sha(source),
           'started_unix_time':time.time(),'changed_contract_fields':sorted(changed)};write(dest/'status.json',state)
    eval_dir=dest/'evaluations';eval_dir.mkdir(exist_ok=True)
    r.evaluation_dir=str(eval_dir);r.dataset_manifest_path=str(Path(args.eval_dataset_dir).parent/'manifest.json')
    r.eval_dataset_dir=args.eval_dataset_dir;r.pre_ppo_case_makespan={}
    r.best_eval_makespan=r.best_eval_iid_makespan=r.best_eval_composite_makespan=float('inf')
    r.selection_metric='raw';r.selection_weights={'iid':0.5,'ood_stress':0.45,'ood_scale':0.05}
    socket=(RUN/'socket.txt').read_text().strip();metrics={}
    expected={x['case_sha256'] for x in json.loads((RUN/'validation120_cases.json').read_text())}
    for label,model in [('pre_supervised',dest/'captured/checkpoint_PreSupervised.pt'),('bc_epoch_1',source)]:
        output=dest/f'{label}_validation120_raw.json'
        command=[entry['argv'][0],'-B','-u',str(SOURCE/'onpolicy/scripts/train/evaluate_stage1_shared_checkpoint.py'),
                 '--socket-path',socket,'--checkpoint',str(model),'--cpu-set',manifest['resource_contract']['eval_cpu_set'],
                 '--seed','11','--output',str(output),'--label',f'migration_seed{seed}_{label}',
                 '--evaluation-tau','0.3','--n-eval-rollout-threads','12']
        state.update(evaluation=label,command=command);write(dest/'status.json',state)
        result=subprocess.run(command,cwd=SOURCE)
        if result.returncode:raise RuntimeError(f'Validation120 {label} failed with exit {result.returncode}')
        raw=json.loads(output.read_text())['evaluation']
        assert raw['case_count']==raw['completed_count']==120 and raw['cycle_count']==raw['timeout_count']==0
        assert {x['case_sha256'] for x in raw['records']}==expected
        score=r._consume_raw_evaluation(raw,evaluation_label=label)
        metrics[label]=r._evaluation_log_info(score)
        assert metrics[label]['eval_case_count']==120
    gate=supervised_gate(metrics['pre_supervised'],metrics['bc_epoch_1'],max_raw_regression=0,max_stress_regression=30)
    record={'epoch':1,'filename':'checkpoint_BCEpoch1.pt','metrics':metrics['bc_epoch_1'],'gate':gate}
    migrated=copy.deepcopy(checkpoint)
    migrated['stage2_bc_run_contract']=target_contract
    migrated['stage2_bc_state']['pre_supervised_info']=metrics['pre_supervised']
    migrated['stage2_bc_state']['epoch_records']=[record]
    migrated.update(selection_score=metrics['bc_epoch_1']['eval_raw_makespan'],eval_makespan=metrics['bc_epoch_1']['eval_raw_makespan'],stage2_scientific_gate=gate)
    migrated['evaluation_migration']={'source_checkpoint':str(source),'source_sha256':sha(source),
        'original_run':str(original),'original_run_contract':previous,'original_pre_supervised_info':checkpoint['stage2_bc_state']['pre_supervised_info'],
        'original_epoch_records':checkpoint['stage2_bc_state']['epoch_records'],
        'new_dataset':args.eval_dataset_dir,'new_case_count':120,'weights_optimizer_rng_unchanged':True}
    for k in ['model','supervised_optimizer']:assert equal(migrated[k],checkpoint[k]),k
    assert equal(migrated['stage2_bc_state']['rng'],checkpoint['stage2_bc_state']['rng'])
    (dest/'models').mkdir(exist_ok=True)
    target=dest/'models/checkpoint_BCEpoch1.pt';torch.save(migrated,target)
    (dest/'logs').mkdir(exist_ok=True)
    for f in ['request_ready_cases_epoch1.json','request_ready_epoch_metrics.jsonl']:
        shutil.copy2(dest/'captured'/f,dest/'logs'/f)
    # Exercise the real recovery loader and actual Adam deserialization before
    # releasing a migrated trainer. Preflight outputs go outside its run dir.
    probe=runner_for(args);probe.save_dir=str(dest/'restore_probe/models');Path(probe.save_dir).mkdir(parents=True,exist_ok=True)
    probe._restore_stage2_supervised_recovery(str(target))
    assert probe.device_bc_resume_epoch==1 and probe.stage2_bc_completed_epochs==1
    assert equal(probe.policy.ac.state_dict(),checkpoint['model'])
    assert equal(probe.resource_supervised_optimizer_state,checkpoint['supervised_optimizer'])
    assert equal(probe.stage2_bc_pending_rng,checkpoint['stage2_bc_state']['rng'])
    parameters=[p for m in probe._device_bc_trainable_modules() for p in m.parameters()]
    optimizer=torch.optim.Adam(parameters,lr=args.device_bc_lr,eps=args.opti_eps,weight_decay=args.weight_decay)
    optimizer.load_state_dict(probe.resource_supervised_optimizer_state)
    assert equal(optimizer.state_dict(),checkpoint['supervised_optimizer'])
    state.update(status='ready_to_resume',finished_unix_time=time.time(),target_checkpoint=str(target),target_sha256=sha(target),
                 model_optimizer_rng_bitwise_equal=True,production_resume_loader_passed=True,
                 evaluation_cases=120,epoch1_gate=gate)
    state.pop('command',None);write(dest/'status.json',state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1,2],required=True);a=p.parse_args()
    try:main(a.seed)
    except BaseException as e:
        write(RUN/'migration'/f'seed{a.seed}'/'failure.json',{'error':repr(e),'unix':time.time()});raise
