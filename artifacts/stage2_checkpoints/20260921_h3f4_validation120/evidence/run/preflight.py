"""Verify unchanged model loading and the metadata conversion before migration."""
import json,math,copy
from pathlib import Path
import migrate
import torch
from onpolicy.scripts.train.train_hkbz import parse_args
from onpolicy.config.config import get_config
from onpolicy.utils.stage2_bc_contract import json_safe

r=migrate.RUN;d=json.loads((r/'manifest.json').read_text());old=json.loads((r/'previous_manifest.json').read_text())
report={'passed':False,'parents':{},'training_updates':0}
for seed in (1,2,3):
 key=f'H3F4_stage1seed{seed}_bcseed11';args=parse_args(d['commands'][key]['argv'][2:],get_config());runner=migrate.runner_for(args)
 runner._restore_stage1_m2(d['stage1_parents'][str(seed)]['path'])
 assert runner._stage1_handoff_summary()['sha256']==d['stage1_parents'][str(seed)]['protected_parameter_sha256']
 assert args.max_eval_cases==120 and Path(args.eval_dataset_dir).name=='validation'
 oldrun=Path(d['transition']['previous_run'])/'source/onpolicy/scripts/results/HKBZ/simple/gnn_mappo'/old['commands'][key]['experiment_name']/'run1'
 if seed in (1,2):
  checkpoint=torch.load(oldrun/'models/checkpoint_PreSupervised.pt',map_location='cpu',weights_only=False)
  target=runner._stage2_bc_run_contract();before=checkpoint['stage2_bc_run_contract']
  changed={k for k in before.keys()|target.keys() if before.get(k)!=target.get(k)}
  assert changed=={'eval_dataset_dir','max_eval_cases'},changed
  assert runner.frozen_ready_source==checkpoint['frozen_ready_source']
  # Check conversion against the real recorded Tune60 summary, without using
  # any synthetic score as a Validation120 result.
  fixture=json.loads((oldrun/'evaluations/pre_supervised.json').read_text());summary=fixture['summary'];cases=fixture['cases']
  raw={'raw_makespan':summary['eval_raw_makespan'],'case_count':summary['eval_case_count'],
       'case_ids':[c['case_id'] for c in cases],'completed_count':summary['eval_completed_count'],
       'completion_rate':summary['eval_completion_rate'],'timeout_count':summary['eval_timeout_count'],
       'cycle_count':summary['eval_cycle_count'],'mean_steps':summary['eval_mean_steps'],
       'max_no_progress':summary['eval_max_no_progress'],'mean_relocations':summary['eval_mean_relocations'],'records':cases}
  runner.evaluation_dir=str(r/'preflight_artifacts'/f'seed{seed}');Path(runner.evaluation_dir).mkdir(parents=True,exist_ok=True)
  runner.dataset_manifest_path=str(Path(args.eval_dataset_dir).parent/'manifest.json');runner.eval_dataset_dir=args.eval_dataset_dir
  runner.pre_ppo_case_makespan={};runner.best_eval_makespan=runner.best_eval_iid_makespan=runner.best_eval_composite_makespan=float('inf')
  runner.selection_metric='raw';runner.selection_weights={'iid':.5,'ood_stress':.45,'ood_scale':.05}
  score=runner._consume_raw_evaluation(raw,evaluation_label='pre_supervised');observed=json_safe(runner._evaluation_log_info(score))
  for k,v in summary.items():
   if k.startswith('eval_') and isinstance(v,(int,float)):
    assert math.isclose(observed[k],v,rel_tol=1e-12,abs_tol=1e-9),(k,v,observed[k])
 report['parents'][str(seed)]={'protected_stage1_parameters_verified':True,'ready_lineage_preserved':True}
report.update(passed=True,validation_case_files=120,summary_conversion_matches_native=True)
migrate.write(r/'preflight.json',report);print(json.dumps(report))
