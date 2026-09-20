import argparse,json,time
from pathlib import Path
import torch
import onpolicy.scripts.train.train_hkbz as train
from onpolicy.scripts.train.shared_hkbz_evaluator import configure_runner
r=Path('/data/fanyx/HKBZ-environment/result/hkbz_train_logs/stage1_validation120_best_20260920_r2')
p=r/'diagnostics/multi_ppo_s3_original_batch12'
p.mkdir(parents=True,exist_ok=True)
reference=json.loads((r/'references/multi_ppo_s3_validation60.json').read_text())['cases']
order=[x['case_dir'] for x in sorted(reference,key=lambda x:(x['rank_index'],x['round_index']))]
assert len(order)==60 and len(set(order))==60
train.list_case_folders=lambda *args,**kwargs:list(order)
s=json.loads((r/'source_command.json').read_text());a=s['command']
for k,v in [('--stage1_baseline','multi_ppo'),('--seed','3'),('--n_eval_rollout_threads','12'),('--n_rollout_threads','12'),('--max_eval_cases','60')]:a[a.index(k)+1]=v
(p/'source.json').write_text(json.dumps(s,indent=2))
c=argparse.Namespace(source_command_json=p/'source.json',socket_path=p/'unused.sock',cpu_pool='0-127',run_dir=p/'runner',eval_dataset_dir=r/'inputs/validation',eval_case_offset=0,max_eval_cases=60,cuda_memory_fraction=.9,eval_partition_seed=20260803,eval_partition_stratify_by='profile')
runner,env,pool=configure_runner(c)
try:
 payload=torch.load(r/'checkpoints/multi_ppo_s3.pt',map_location='cpu')
 runner.policy.ac.load_state_dict(payload['model'],strict=True)
 runner.policy.ac.tau=.3;runner.trainer.prep_rollout()
 env.call_each('seed',[((150000)+rank*10000,) for rank in range(12)])
 torch.manual_seed(3);torch.cuda.manual_seed_all(3)
 runner.eval_case_counts=[2]*12
 started=time.time();out=runner._eval_with_envs(evaluation_label='diagnostic_original_batch12_rounds0_1',finalize=False)
 out['evaluation_seconds']=time.time()-started
 old={x['case_dir']:x for x in reference}
 out['reproduction']={'matches':sum(x['makespan']==old[x['case_dir']]['makespan'] for x in out['records']),'count':len(out['records']),'differences':[{k:x[k] for k in ['case_dir','makespan','round_index','rank_index']}|{'old_makespan':old[x['case_dir']]['makespan']} for x in out['records'] if x['makespan']!=old[x['case_dir']]['makespan']]}
 (p/'result.json').write_text(json.dumps(out,indent=2))
 print(out['evaluation_seconds'],out['reproduction'],flush=True)
 print('case0076',[(x['case_dir'],x['makespan']) for x in out['records'] if x['case_dir']=='case_0076'],flush=True)
finally:env.close()
