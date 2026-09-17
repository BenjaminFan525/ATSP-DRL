#!/usr/bin/env python3
"""CPU-only environment replay mapping saved first divergences to physical IDs."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import types


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);a=parser.parse_args();root=a.root.resolve()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    torch.set_num_threads(1)
    read=lambda p:json.loads(Path(p).read_text())
    manifest=read(root/'manifest.json');study=read(manifest['study_path']);parent=read(study['parent']['path'])
    sys.path.insert(0,study['source_root'])
    from onpolicy.utils.stage2_frozen import load_frozen_stage2
    from onpolicy.runner.shared.stage3_b_shared_b0_engine import BSharedB0Engine
    from onpolicy.utils.stage3_h3_frozen import PLANNING
    from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
    workspace=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('tau_trace_helpers',workspace/'onpolicy/scripts/train/run_stage3_h3_tau_trace.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    _,args,bundle=load_frozen_stage2(parent['frozen_manifest']['path'])
    def clean(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list,set)):return [clean(x) for x in v]
        if v is None or isinstance(v,(str,int,float,bool)):return v
        return getattr(v,'code',str(v))
    for path in sorted((root/'findings').glob('*.json')):
        finding=read(path);name=finding['case'];target=root/('physical_'+path.name)
        if target.exists():continue
        index=next(i for i,c in enumerate(study['cases']) if c['name']==name);case=study['cases'][index]
        config=BSharedB0Engine.environment_config(types.SimpleNamespace(args=args,bundle=bundle),case['path'],42)
        config.update(PLANNING)
        env=AircraftScheduleEnv(config)
        try:
            env.seed(50000+(index%12)*10000);env.reset_data_cursor();obs,_,info=env.reset()
            stop=finding['step_zero_based']
            with gzip.open(root/'traces'/f'{name}_tau_0p3.pkl.gz','rb') as f:
                for step in range(stop+1):
                    row=pickle.load(f)
                    if mod.tree_digest(obs)!=row['graph_sha256']:
                        raise ValueError(f'Environment replay changed the original graph at {name}:{step}')
                    if step==stop:break
                    actions=row['actions'];actions=np.concatenate([actions,np.full((len(actions),1),-1,dtype=np.int64)],axis=1)
                    obs,_,_,info=env.step(actions)
            changed=finding['active_changed_agents']
            devices={int(i):dict(code=env.device_list[i-env.n_plane_agents].code,
                attributes=clean(vars(env.device_list[i-env.n_plane_agents]))) for i in changed if i>=env.n_plane_agents}
            request_ids={int(action[0]) for ids in ['actions_reference','actions_candidate']
                         for agent,action in zip(finding['changed_agents'],finding[ids]) if agent>=env.n_plane_agents}
            requests={i:clean(env.request_pool[i]) for i in request_ids}
            result=dict(case=name,tau=finding['tau'],step=stop,graph_exact_replay=True,env_time=env.total_time,
                        devices=devices,requests=requests)
            mod.write(target,result)
            print(json.dumps(dict(case=name,tau=finding['tau'],decision=stop+1,devices={k:v['code'] for k,v in devices.items()},
                                  graph_exact_replay=True),ensure_ascii=False))
        finally:env.close()


if __name__=='__main__':main()
