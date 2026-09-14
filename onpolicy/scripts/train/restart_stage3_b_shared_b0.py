#!/usr/bin/env python3
"""Freeze and launch a rebatch continuation from the latest formal commit."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file, digest_json
from onpolicy.utils.stage3_b_shared_b0 import (
    training_schedule, budget, read_commit, manifest_identity, verify_manifest, verify_admission,
    submit_evaluation, validate_request,
)
from onpolicy.utils.stage3_b0_restart import MODE, rebind_checkpoint, verify_restored_state

OVERLAYS = (
    'onpolicy/utils/stage3_b_shared_b0.py',
    'onpolicy/utils/stage3_full_data.py',
    'onpolicy/utils/stage3_b0_restart.py',
    'onpolicy/scripts/train/stage3_b_shared_b0_worker.py',
    'onpolicy/scripts/train/run_stage3_b_shared_b0.py',
    'onpolicy/scripts/train/run_stage3_b_shared_b0_throughput.py',
    'onpolicy/scripts/train/restart_stage3_b_shared_b0.py',
    'onpolicy/envs/HKBZ/test/test_stage3_b0_restart.py',
)
TESTS = ('test_stage3_b_shared_b0.py', 'test_stage3_b0_throughput.py', 'test_stage3_b0_restart.py')
SINGLE_OVERLAYS = (
    'onpolicy/runner/shared/stage3_b_shared_b0_engine.py',
    'onpolicy/utils/stage3_b0_single_model.py',
    'onpolicy/scripts/train/probe_stage3_b0_single_model.py',
    'onpolicy/envs/HKBZ/test/test_stage3_b0_single_model.py',
)
CAPACITY_OVERLAYS = (
    'onpolicy/scripts/train/autotune_stage3_b_shared_b0.py',
    'onpolicy/scripts/train/probe_stage3_b0_requested_capacity.py',
    'onpolicy/envs/HKBZ/test/test_stage3_b0_requested_capacity.py',
    'onpolicy/utils/stage3_b0_subset.py',
    'onpolicy/envs/HKBZ/test/test_stage3_b0_subset.py',
)


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest_file(path))


def environment(source, gpu=''):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(source),
        HKBZ_STAGE3_WORKSPACE_ROOT=str(Path(os.environ.get('HKBZ_STAGE3_WORKSPACE_ROOT', ROOT))),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')


def finite_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tensors(v) for v in value)
    return True


def prepare(args):
    import psutil
    import torch
    import yaml
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states
    from onpolicy.scripts.train.run_stage3_b_shared_b0 import check_junit
    from onpolicy.scripts.train.deploy_stage3_b_shared_b0_tuned import copy_evidence
    if not args.ignore_numerical_equivalence:
        raise ValueError('This continuation needs the explicitly authorized equivalence waiver')
    parent = read_json(args.manifest)
    if manifest_identity(parent) != parent['manifest_sha256']:
        raise ValueError('Parent manifest changed')
    for relative, checksum in parent['code_files'].items():
        if digest_file(Path(parent['source_root']) / relative) != checksum:
            raise ValueError(f'Parent frozen source changed: {relative}')
    old_plan = training_schedule(parent)
    if digest_json(old_plan) != parent['schedule_sha256']:
        raise ValueError('Parent visit schedule changed')
    commits = sorted((Path(parent['root']) / 'commits').glob('batch_*.json'))
    checked = [read_commit(parent, path) for path in commits]
    if not checked or [c['next_batch'] for c in checked] != list(range(1, len(checked) + 1)):
        raise ValueError('A contiguous committed training prefix is required')
    last = checked[-1]
    if last['training_episodes'] >= budget(parent)['training_episodes']:
        raise ValueError('All training visits are already complete')
    root = args.output.resolve()
    root.mkdir()
    source = root / 'source'
    shutil.copytree(parent['source_root'], source)
    files = copy.deepcopy(parent['code_files'])
    capacity_window = bool(args.capacity_window)
    training_selection = getattr(args, 'training_selection', None)
    if training_selection and not capacity_window:
        raise ValueError('Changed training data requires a matched capacity probe')
    single = bool(args.single_model_probe or capacity_window)
    if (args.global_batch, args.microbatch, args.input_cache_mib) != (64, 64, 4096) and not capacity_window:
        raise ValueError('Changed parameters require their bounded capacity evidence')
    for relative in OVERLAYS + (SINGLE_OVERLAYS if single else ()) + (CAPACITY_OVERLAYS if capacity_window else ()):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): target.unlink()
        shutil.copyfile(ROOT / relative, target)
        target.chmod(0o444)
        files[relative] = digest_file(target)
    # The execution module itself stays byte-identical to the measured module.
    runtime_parent = read_json(Path(parent['root']) / 'runtime.json')
    update_module = source / 'onpolicy/utils/stage3_b0_throughput.py'
    if digest_file(update_module) != runtime_parent.get('update_module_sha256', runtime_parent['module_sha256']):
        raise ValueError('Measured execution adapter changed')
    module = source / 'onpolicy/utils/stage3_b0_single_model.py' if single else update_module
    recipe = copy.deepcopy(parent['recipe'])
    maximum = min(144, int(psutil.virtual_memory().available / 2**30) - 40)
    if maximum < 100:
        raise ValueError('Available host memory cannot cover the requested capacity')
    recipe.update(global_batch=args.global_batch, microbatch=args.microbatch, input_cache_mib=args.input_cache_mib,
                  memory_max_gib=maximum, memory_high_gib=int(.9 * maximum))
    if capacity_window:
        recipe['rollout_workers'] = args.global_batch
        recipe['encoder_activation_checkpoint'] = args.encoder_checkpoint
    config = source / 'onpolicy/config/env_stage3_b_shared_b0.yaml'
    config.unlink(); config.write_text(yaml.safe_dump(recipe, sort_keys=False)); config.chmod(0o444)
    files['onpolicy/config/env_stage3_b_shared_b0.yaml'] = digest_file(config)
    tests_file = root / 'tests.xml'
    test_names = TESTS + (('test_stage3_b0_single_model.py',) if single else ())
    if capacity_window:
        test_names += ('test_stage3_b0_requested_capacity.py',)
        if training_selection:
            test_names += ('test_stage3_b0_subset.py',)
    test_paths = [str(source / 'onpolicy/envs/HKBZ/test' / name) for name in test_names]
    subprocess.run([str(args.pytest_python), '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        *test_paths, '--junitxml=' + str(tests_file)], cwd=source, env=environment(source), check=True)
    tests = check_junit(tests_file)
    tests.update(binding(tests_file))
    candidate = args.candidate.resolve()
    selected = read_json(candidate / 'result.json')
    params = selected['parameters']
    if (params['environments'], params['lanes'], params['microbatch'], params['cache_mib']) != (64,16,64,4096):
        raise ValueError('Candidate differs from the user-selected configuration')
    sample = read_json(candidate / 'sample/result.json')
    status = read_json(candidate / 'update/status.json')
    capacity_checkpoint = candidate / 'update/actual.pt'
    capacity = torch.load(capacity_checkpoint, map_location='cpu', weights_only=False)
    if (not sample['passed'] or sample['extension_sha256'] != digest_file(update_module)
            or 'Tensor differs:' not in status.get('error', '')
            or capacity['policy_updates'] != 2 or capacity['next_batch'] != 1
            or not capacity['actor_optim']['state'] or not capacity['critic_optim']['state']
            or not finite_tensors(capacity)):
        raise ValueError('Candidate did not complete the measured finite two-PPO update')
    selection = dict(selected_for_throughput=True, numerical_equivalence_passed=False,
        numerical_equivalence_gate='waived_by_user', parameters=params,
        candidate=binding(candidate / 'result.json'),
        sample_seconds=sample['seconds'],
        update_stage_seconds=status['heartbeat_unix']-status['started_unix'],
        timing_scope='update stage timestamps include checkpoint save and rejected comparison')
    if capacity_window:
        probe = read_json(args.capacity_window)
        params = dict(environments=args.global_batch, microbatch=args.microbatch, lanes=1,
                      cache_mib=args.input_cache_mib, environment_processes=args.environment_processes,
                      encoder_activation_checkpoint=args.encoder_checkpoint)
        if not probe.get('passed') or probe.get('parameters') != params:
            raise ValueError('Requested parameters do not match the capacity window')
        selection.update(parameters=params, sampling_architecture='single_model_batched_v1',
            selected_by_user=True, capacity_window=binding(args.capacity_window),
            selection_reason=f'User requested {args.global_batch} concurrent environments and minibatch {args.microbatch}',
            timing_scope='bounded sampling and dense TBPTT capacity only; full group timing pending',
            end_to_end_speedup_measured=False)
        selection['historical_micro64_timing'] = dict(sample_seconds=selection.pop('sample_seconds'),
            update_stage_seconds=selection.pop('update_stage_seconds'))
        selection['capacity_window_sampling_seconds'] = probe['sampling']['seconds']
        selection['capacity_window_backward_seconds'] = probe['update']['backward_seconds']
    elif single:
        probe = read_json(args.single_model_probe)
        if not probe['passed']:
            raise ValueError('The new sampler requires its complete sampling/replay proof')
        params = dict(params, lanes=1)
        selection.update(parameters=params, sampling_architecture='single_model_batched_v1',
            single_model_probe=binding(args.single_model_probe),
            sample_seconds=probe['sample_seconds'], previous_sample_seconds=sample['seconds'],
            selection_reason='User requested a single shared GPU sampler with true batched inference',
            timing_scope='new sampler measured on next scheduled group; historical update capacity timing only',
            end_to_end_speedup_measured=False)
    atomic_json(root / 'throughput_selection.json', selection, overwrite=False)
    compute = dict(compute_completed=True, all_training_tensors_finite=True,
        numerical_equivalence_passed=False, rejection_preserved=binding(candidate / 'update/status.json'),
        manifest_sha256=sample['manifest_sha256'], extension_sha256=digest_file(update_module),
        global_visits=64, microbatch=64, ppo_epochs=2, input_cache_mib=4096,
        checkpoint=binding(capacity_checkpoint))
    atomic_json(root / 'full_update_compute.json', compute, overwrite=False)
    m = copy.deepcopy(parent)
    m.update(root=str(root), source_root=str(source), recipe=recipe, tests=tests,
        code_files=files, code_sha256=digest_json(files), created_unix=time.time(),
        resource_amendment=f'GPU0 and original CPU half; committed history, global{args.global_batch} micro{args.microbatch} cache{args.input_cache_mib}MiB',
        resume_amendment=dict(mode=MODE, numerical_equivalence_gate='waived_by_user',
            user_instruction='忽略完整数值对照；以首选的参数组合，从最近的checkpoint重启训练',
            parent_manifest=binding(args.manifest), parent_manifest_sha256=parent['manifest_sha256'],
            parent_commit=binding(commits[-1]), completed_batches=len(commits),
            completed_batch_sizes=[len(row['cases']) for row in old_plan[:len(commits)]],
            completed_visits=last['training_episodes'], completed_actor_updates=last['actor_updates'],
            formal_initialization='latest committed Stage3 checkpoint; model/Adam/normalizer/RNG preserved',
            remaining_visits=budget(parent)['training_episodes']-last['training_episodes']))
    if single:
        m['resource_amendment'] = f'GPU0 and original CPU half; one shared sampling/training model, {args.global_batch} environments'
        m['resume_amendment'].update(sampling_architecture='single_model_batched_v1',
            sampling_rng_contract='checkpoint_global_torch_live_slot_order_v1',
            sampling_probe=binding(args.capacity_window or args.single_model_probe),
            user_instruction='忽略完整数值对照；采样使用同一个模型；按照这个思路修改，然后重启实验')
    if capacity_window:
        m['resume_amendment'].update(requested_parameters=params,
            capacity_admission='requested_capacity_window_v1',
            user_instruction=(args.user_instruction or
                f'并发环境设置成{args.global_batch}，minibatch设置{args.microbatch}。从已提交 checkpoint 继续；沿用此前完整数值对照豁免'))
    if training_selection:
        from onpolicy.utils.stage3_b0_subset import make_amendment, verify_selection
        selected_data = verify_selection(parent, binding(training_selection))
        m['splits']['train'] = copy.deepcopy(selected_data['cases'])
        m['training_subset'] = make_amendment(parent, old_plan[:len(commits)], binding(training_selection), root)
        m['resume_amendment']['remaining_visits'] = m['training_subset']['new_training_visits']
        m['resource_amendment'] += '; fixed Train240, eight new subset epochs; original checkpoint history retained'
    plan = training_schedule(m)
    if plan[:len(commits)] != old_plan[:len(commits)]:
        raise ValueError('Continuation changed an already committed group')
    m['schedule_sha256'] = digest_json(plan)
    m['throughput_amendment'] = dict(parent_manifest_sha256=sample['manifest_sha256'],
        selection=str(root / 'throughput_selection.json'),
        selection_sha256=digest_file(root / 'throughput_selection.json'), parameters=params,
        replacement_gate='user-waived equivalence; exact checkpoint load plus finite-update capacity evidence')
    if capacity_window:
        m['throughput_amendment']['replacement_gate'] = (
            'explicit requested parameters; bounded capacity and exact restore; complete on-policy guards before commit')
    m['manifest_sha256'] = manifest_identity(m)
    atomic_json(root / 'manifest.json', m, overwrite=False)
    for relative in ('commits','attempts','diagnostics','validator/requests','validator/results',
                     'validator/cases','resume_import/models','resume_import/updates','resume_import/rollouts'):
        (root / relative).mkdir(parents=True, exist_ok=True)
    for path, old_commit in zip(commits, checked):
        old = torch.load(old_commit['checkpoint'], map_location='cpu', weights_only=False)
        if (old['manifest_sha256'] != parent['manifest_sha256']
                or old['recipe_sha256'] != digest_json(parent['recipe'])
                or old['next_batch'] != old_commit['next_batch'] or old['policy_updates'] != old_commit['actor_updates']):
            raise ValueError('Committed checkpoint payload identity changed')
        rebound = rebind_checkpoint(old, m['manifest_sha256'], recipe)
        checkpoint = root / 'resume_import/models' / f'batch_{old_commit["next_batch"]:04d}.pt'
        torch.save(rebound, checkpoint)
        actual = torch.load(checkpoint, map_location='cpu', weights_only=False)
        for key in old.keys() - {'manifest_sha256', 'recipe_sha256'}:
            compare_states(old[key], actual[key], exact=True)
        update = root / 'resume_import/updates' / Path(old_commit['update']).name
        shutil.copyfile(old_commit['update'], update)
        rollout = Path(old_commit['attempt']) / 'rollouts' / path.name
        if rollout.exists(): shutil.copyfile(rollout, root / 'resume_import/rollouts' / path.name)
        commit = dict(old_commit, manifest_sha256=m['manifest_sha256'], schedule_sha256=m['schedule_sha256'],
            checkpoint=str(checkpoint), checkpoint_sha256=digest_file(checkpoint), update=str(update),
            update_sha256=digest_file(update), attempt=str(root / 'resume_import'),
            inherited_from=binding(path))
        atomic_json(root / 'commits' / path.name, commit, overwrite=False)
    copy_evidence(Path(parent['root'])/'baselines.json',root/'baselines.json',parent['manifest_sha256'],
        manifest_sha256=m['manifest_sha256'])
    copy_evidence(Path(parent['root'])/'zero_update_verified.json',root/'zero_update_verified.json',parent['manifest_sha256'],
        manifest_sha256=m['manifest_sha256'],baseline_sha256=digest_file(root/'baselines.json'))
    for filename in ('B0_AR_validation.json',):
        old_result = Path(parent['root']) / 'validator/results' / filename
        control = read_json(old_result)
        if not control['completed'] or len(control['rows']) != 120:
            raise ValueError('Complete original B0 AR baseline required')
        ident = submit_evaluation(m,m['source']['path'],0,'validation',decoder='AR',baseline=True)
        request = read_json(root/'validator/requests'/f'{ident}.json'); validate_request(m,request)
        for key in ('checkpoint_sha256','cases_sha256','tau','seed','history','decoder'):
            if request[key] != control['request'][key]: raise ValueError('B0 baseline contract changed')
        control.update(request=request,reused_from=binding(old_result))
        atomic_json(root/'validator/results'/f'{ident}.json',control,overwrite=False)
    launcher = source / 'onpolicy/scripts/train/run_stage3_b_shared_b0_throughput.py'
    runtime = dict(study_manifest_sha256=m['manifest_sha256'],
        proof_manifest_sha256=sample['manifest_sha256'],
        module=str(module), module_sha256=digest_file(module),launcher=str(launcher),launcher_sha256=digest_file(launcher),
        global_batch=args.global_batch, microbatch=args.microbatch,
        sampling_environments=args.global_batch, sampling_lanes=params['lanes'],
        input_cache_mib=args.input_cache_mib, ppo_epochs=2, cuda_memory_headroom_mib=8192,release_sampling_before_update=True,
        admission_mode=MODE, numerical_equivalence_gate='waived_by_user',tests=tests,
        selection=binding(root/'throughput_selection.json'),
        proofs=dict(sample=binding(candidate/'sample/result.json'),update=binding(root/'full_update_compute.json')))
    if single:
        runtime.update(sampling_architecture='single_model_batched_v1',
            sampling_rng_contract='checkpoint_global_torch_live_slot_order_v1',
            update_module=str(update_module), update_module_sha256=digest_file(update_module))
        runtime['proofs']['sample'] = binding(args.capacity_window or args.single_model_probe)
    if capacity_window:
        runtime.update(capacity_admission='requested_capacity_window_v1',
            environment_processes=args.environment_processes,
            encoder_activation_checkpoint=args.encoder_checkpoint)
    runtime['runtime_sha256'] = digest_json(runtime)
    atomic_json(root/'runtime.json',runtime,overwrite=False)
    script = source / 'onpolicy/scripts/train/restart_stage3_b_shared_b0.py'
    subprocess.run([m['python'],'-B',str(script),'audit',str(root/'manifest.json')],
        cwd=source,env=environment(source),check=True)
    admission = dict(passed=True,mode=MODE,manifest_sha256=m['manifest_sha256'],
        numerical_equivalence_passed=False,global_batch=args.global_batch,
        physical_microbatch=args.microbatch,gpu_count=1,tests=tests,
        baseline_sha256=digest_file(root/'baselines.json'),zero_check_sha256=digest_file(root/'zero_update_verified.json'),
        restore_check=binding(root/'restore_check.json'),gpu_restore_verification_required_before_collection=True)
    if capacity_window:
        admission.update(capacity_admission='requested_capacity_window_v1',
            full_requested_parameter_validation=False)
    atomic_json(root/'training_admission.json',admission,overwrite=False)
    subprocess.run([m['python'],'-B',str(script),'verify',str(root/'manifest.json')],
        cwd=source,env=environment(source),check=True)
    atomic_json(root/'prepared.json',dict(prepared=True,training_started=False,manifest=str(root/'manifest.json'),
        budget=budget(m),remaining_batches=budget(m)['global_batches']-len(commits),
        remaining_actor_updates=budget(m)['actor_updates']-last['actor_updates'],
        parent_checkpoint=last['checkpoint'],parent_checkpoint_sha256=last['checkpoint_sha256']),overwrite=False)
    print(json.dumps(read_json(root/'prepared.json')),flush=True)


def audit(m):
    import torch
    from onpolicy.runner.shared.stage3_b_shared_b0_engine import BSharedB0Engine
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states
    verify_manifest(m,inputs=True)
    a = m['resume_amendment'];root=Path(m['root'])
    commit = read_commit(m,root/'commits'/f'batch_{a["completed_batches"]:04d}.json')
    parent_commit = read_json(a['parent_commit']['path'])
    parent = torch.load(parent_commit['checkpoint'],map_location='cpu',weights_only=False)
    actual = torch.load(commit['checkpoint'],map_location='cpu',weights_only=False)
    if parent.keys()!=actual.keys():raise ValueError('Checkpoint fields changed during migration')
    for key in parent.keys()-{'manifest_sha256','recipe_sha256'}:
        compare_states(parent[key],actual[key],exact=True)
    runner = BSharedB0Engine(m['frozen_manifest']['path'],device='cpu',create_pool=False,config=m['recipe'])
    try:
        cursor = runner.resume(commit['checkpoint'],manifest_sha256=m['manifest_sha256'])
        proof = verify_restored_state(runner,commit['checkpoint'])
        if cursor!=a['completed_batches'] or runner.policy_updates!=a['completed_actor_updates']:
            raise ValueError('Restored optimizer/update cursor differs')
        proof.update(manifest_sha256=m['manifest_sha256'],fresh_process=True,metadata_only_rebind=True,
            cuda_rng_payload_exact=True,next_batch=cursor,checkpoint_sha256=commit['checkpoint_sha256'],
            parent_checkpoint_sha256=parent_commit['checkpoint_sha256'],training_executed=False)
        atomic_json(root/'restore_check.json',proof,overwrite=False)
    finally:
        runner.close()


def launch(m):
    verify_manifest(m,inputs=True);verify_admission(m)
    root=Path(m['root']);runtime=read_json(root/'runtime.json')
    unit=f'hkbz-b0-resume{runtime["global_batch"]}-{int(time.time())}'
    command=['systemd-run','--user','--collect','--unit='+unit,
        '--property=Type=exec','--property=KillMode=control-group','--property=TimeoutStopSec=45',
        '--property=RuntimeMaxSec=1209600','--property=MemorySwapMax=0',
        '--property=CPUAffinity=0-31 64-95','--property=AllowedCPUs=0-31,64-95',
        f'--property=MemoryHigh={m["recipe"]["memory_high_gib"]}G',
        f'--property=MemoryMax={m["recipe"]["memory_max_gib"]}G',
        '--property=WorkingDirectory='+m['source_root'],
        '--property=StandardOutput=append:'+str(root/'service.log'),
        '--property=StandardError=append:'+str(root/'service.log')]
    env=environment(m['source_root'],m['recipe']['gpu_uuid'])
    env['HKBZ_STAGE3_WORKSPACE_ROOT']=m['workspace_root']
    for key in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','HKBZ_STAGE3_WORKSPACE_ROOT','OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG'):
        command.append('--setenv='+key+'='+env[key])
    command += [m['python'],'-B','-u',runtime['launcher'],'resume',str(root/'manifest.json'),
        '--module',runtime['module'],'--runtime',str(root/'runtime.json')]
    atomic_json(root/'launch_command.json',dict(argv=command),overwrite=False)
    subprocess.run(command,check=True)
    (root/'last_service_unit.txt').write_text(unit+'.service\n')
    print(json.dumps(dict(service=unit+'.service',manifest=str(root/'manifest.json'))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','audit','verify','launch'))
    p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--candidate',type=Path);p.add_argument('--pytest-python',type=Path)
    p.add_argument('--ignore-numerical-equivalence',action='store_true')
    p.add_argument('--single-model-probe',type=Path,
        help='Admit one GPU sampler using a completed 64-environment sampling/replay probe')
    p.add_argument('--capacity-window',type=Path)
    p.add_argument('--global-batch',type=int,default=64)
    p.add_argument('--microbatch',type=int,default=64)
    p.add_argument('--environment-processes',type=int,default=64)
    p.add_argument('--input-cache-mib',type=int,default=4096)
    p.add_argument('--encoder-checkpoint',action='store_true')
    p.add_argument('--user-instruction',help='Record the authorized checkpoint-boundary parameter change')
    p.add_argument('--training-selection', type=Path, help='Frozen deterministic Train240 selection')
    args=p.parse_args()
    if args.command=='prepare':
        if not all((args.output,args.candidate,args.pytest_python)):p.error('Preparation requires output, candidate and pytest Python')
        prepare(args)
    else:
        m=read_json(args.manifest)
        if args.command=='audit':audit(m)
        elif args.command=='launch':launch(m)
        else:
            from onpolicy.scripts.train.run_stage3_b_shared_b0_throughput import verify_runtime
            verify_manifest(m,inputs=True);verify_admission(m)
            r=read_json(Path(m['root'])/'runtime.json')
            verify_runtime(Path(m['root'])/'runtime.json',m,Path(r['module']),Path(r['launcher']))
            print('CONTINUATION_VERIFIED',json.dumps(budget(m)),flush=True)


if __name__=='__main__':main()
