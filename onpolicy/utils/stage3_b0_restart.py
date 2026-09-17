"""Explicit checkpoint continuation with a user-waived equivalence gate."""
from pathlib import Path
import random

from onpolicy.utils.stage3_research import read_json, digest_file, digest_json


MODE = 'checkpoint_continuation_v1'


def bound_json(binding):
    if digest_file(binding['path']) != binding['sha256']:
        raise ValueError('Continuation evidence changed')
    return read_json(binding['path'])


def require_waiver(m):
    a = m.get('resume_amendment', {})
    if (a.get('mode') != MODE or a.get('numerical_equivalence_gate') != 'waived_by_user'
            or not a.get('user_instruction') or not a.get('completed_batches')):
        raise ValueError('Explicit continuation and numerical-equivalence waiver required')
    return a


def rebind_checkpoint(payload, manifest_sha256, recipe):
    """Change only experiment identity; never reinitialize optimizer or RNG."""
    if payload.get('diagnostic_only') or not payload.get('actor_optim', {}).get('state'):
        raise ValueError('A committed checkpoint with nonempty Adam is required')
    if not payload.get('critic_optim', {}).get('state') or payload.get('rng_cuda') is None:
        raise ValueError('Checkpoint lacks critic state or CUDA RNG')
    value = dict(payload)
    value.update(manifest_sha256=manifest_sha256, recipe_sha256=digest_json(recipe))
    return value


def verify_restored_state(runner, checkpoint):
    import numpy as np
    import torch
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states
    expected = torch.load(checkpoint, map_location='cpu', weights_only=False)
    actual = dict(model=runner.policy.ac.state_dict(),
        actor_optim=runner.policy.actor_optimizer.state_dict(),
        critic_optim=runner.policy.critic_optimizer.state_dict(),
        value_normalizer=runner.norm.state_dict(),
        normalization_sha256=runner.normalization_sha256, policy_updates=runner.policy_updates,
        rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state())
    cuda = runner.device.type == 'cuda'
    if cuda:
        actual['rng_cuda'] = torch.cuda.get_rng_state(runner.device)
    for key, value in actual.items():
        compare_states(expected[key], value, exact=True)
    return dict(passed=True, loaded_training_state_exact=True, checked_fields=list(actual),
                cuda_rng_checked=cuda, checkpoint=str(Path(checkpoint).resolve()))


def verify_restart_admission(m, admission):
    amendment = require_waiver(m)
    if admission.get('mode') != MODE or admission.get('numerical_equivalence_passed') is not False:
        raise ValueError('Restart must not claim a numerical-equivalence pass')
    parent = bound_json(amendment['parent_manifest'])
    commit = bound_json(amendment['parent_commit'])
    if (parent['manifest_sha256'] != amendment['parent_manifest_sha256']
            or commit['manifest_sha256'] != parent['manifest_sha256']
            or commit['next_batch'] != amendment['completed_batches']
            or commit['training_episodes'] != amendment['completed_visits']
            or commit['actor_updates'] != amendment['completed_actor_updates']
            or digest_file(commit['checkpoint']) != commit['checkpoint_sha256']):
        raise ValueError('Parent checkpoint continuation identity changed')
    proof = bound_json(admission['restore_check'])
    if (not proof['passed'] or not proof['metadata_only_rebind']
            or not proof['loaded_training_state_exact'] or not proof['cuda_rng_payload_exact']
            or not proof['fresh_process'] or proof['manifest_sha256'] != m['manifest_sha256']
            or proof['parent_checkpoint_sha256'] != commit['checkpoint_sha256']
            or proof['next_batch'] != amendment['completed_batches']
            or digest_file(proof['checkpoint']) != proof['checkpoint_sha256']):
        raise ValueError('Checkpoint restore proof failed')


def verify_restart_runtime(m, runtime):
    require_waiver(m)
    if runtime.get('admission_mode') != MODE or runtime.get('numerical_equivalence_gate') != 'waived_by_user':
        raise ValueError('Runtime did not bind the explicit equivalence waiver')
    sample = bound_json(runtime['proofs']['sample'])
    update = bound_json(runtime['proofs']['update'])
    selection = bound_json(runtime['selection'])
    single = runtime.get('sampling_architecture') == 'single_model_batched_v1'
    window = runtime.get('capacity_admission') == 'requested_capacity_window_v1'
    update_module_sha = runtime.get('update_module_sha256') if single else runtime['module_sha256']
    if single:
        if window:
            verify_requested_capacity(m, runtime, sample)
        else:
            verify_single_model_sample(m, runtime, sample)
        if digest_file(runtime['update_module']) != update_module_sha:
            raise ValueError('Inherited PPO execution module changed')
    for evidence in ((update,) if single else (sample, update)):
        if (evidence['manifest_sha256'] != runtime['proof_manifest_sha256']
                or evidence['extension_sha256'] != update_module_sha):
            raise ValueError('Capacity evidence does not match the execution module')
    if not single and (not sample['passed'] or not sample['rows_exact'] or not sample['rng_exact']
            or sample['visits'] != runtime['global_batch']
            or sample['sampling_lanes'] != runtime['sampling_lanes']):
        raise ValueError('Complete sampling capacity evidence failed')
    if (not update['compute_completed'] or not update['all_training_tensors_finite']
            or update['numerical_equivalence_passed'] is not False
            or update['global_visits'] != (64 if window else runtime['global_batch'])
            or update['microbatch'] != (64 if window else runtime['microbatch']) or update['ppo_epochs'] != 2
            or update['input_cache_mib'] != (4096 if window else runtime['input_cache_mib'])
            or digest_file(update['checkpoint']['path']) != update['checkpoint']['sha256']):
        raise ValueError('Complete update capacity evidence failed')
    selected = selection['parameters']
    if (not selection['selected_for_throughput']
            or (selected['environments'], selected['lanes'], selected['microbatch'], selected['cache_mib'])
            != (runtime['global_batch'], runtime['sampling_lanes'], runtime['microbatch'], runtime['input_cache_mib'])
            or not runtime['release_sampling_before_update'] or runtime['cuda_memory_headroom_mib'] != 8192
            or digest_file(runtime['tests']['path']) != runtime['tests']['sha256']):
        raise ValueError('Restart runtime parameters or regression evidence changed')


def verify_requested_capacity(m, runtime, proof):
    """Admit explicitly requested parameters with honestly scoped window tests."""
    a = require_waiver(m)
    commit = bound_json(a['parent_commit'])
    params = dict(environments=runtime['global_batch'], microbatch=runtime['microbatch'], lanes=1,
        cache_mib=runtime['input_cache_mib'], environment_processes=runtime['environment_processes'],
        encoder_activation_checkpoint=runtime.get('encoder_activation_checkpoint',False))
    sample, update = proof.get('sampling', {}), proof.get('update', {})
    if (a.get('capacity_admission') != 'requested_capacity_window_v1'
            or a.get('requested_parameters') != params
            or runtime['sampling_lanes'] != 1
            or (params['environments'], params['microbatch']) not in ((240, 80), (240, 120), (240, 240))
            or runtime['proofs']['sample'] != a['sampling_probe']
            or not proof.get('passed') or proof.get('scope') != 'requested_capacity_window_v1'
            or proof.get('parameters') != params or proof.get('architecture') != 'single_model_batched_v1'
            or proof.get('rng_contract') != 'checkpoint_global_torch_live_slot_order_v1'
            or proof.get('parent_manifest_sha256') != a['parent_manifest_sha256']
            or proof.get('checkpoint_sha256') != commit['checkpoint_sha256']
            or proof.get('next_batch') != commit['next_batch']
            or not proof.get('checkpoint_restore_exact') or not proof.get('cuda_rng_restored_exact')
            or proof.get('updates_executed') != 0
            or not proof.get('environment_parity', {}).get('passed')
            or not 0 < proof.get('peak_gpu_mib', 0) <= proof.get('gpu_total_mib', 0)-6144
            or sample.get('environment_count') != params['environments']
            or sample.get('environment_processes') != params['environment_processes']
            or not 1 <= params['environment_processes'] <= 64 or sample.get('model_copies') != 1
            or not sample.get('repeated_forward_and_rng_exact')
            or sample.get('window', {}).get('environment_steps', 0) < params['environments']*26
            or sample.get('full_trajectories_tested') is not False
            or not sample.get('replay', {}).get('decisions')
            or not 0 <= sample.get('replay', {}).get('max_logp_error', float('inf')) <= .002
            or not 0 < sample.get('projected_full_memory_bytes', 0) <= m['recipe']['memory_max_gib']*2**30
            or not update.get('finite_gradients') or update.get('optimizer_updates') != 0
            or update.get('full_update_tested') is not False
            or update.get('window', {}).get('active_microbatch') != params['microbatch']
            or update.get('window', {}).get('events', 0) < params['microbatch']*8):
        raise ValueError('Requested parameter capacity evidence failed')
    required = {'onpolicy/utils/stage3_b0_single_model.py',
        'onpolicy/runner/shared/stage3_b_shared_b0_engine.py', 'onpolicy/utils/stage3_b0_throughput.py',
        'onpolicy/scripts/train/autotune_stage3_b_shared_b0.py',
        'onpolicy/scripts/train/probe_stage3_b0_requested_capacity.py'}
    if m.get('training_subset'):
        from onpolicy.utils.stage3_b0_subset import bound, epoch_rows
        subset = m['training_subset']
        selected = bound(subset['selection'])
        initial = epoch_rows(selected, 0)[:params['environments']]
        expected = dict(selection=subset['selection'], cases_sha256=selected['cases_sha256'],
            sampling_cases=[r[0]['path'] for r in initial], sampling_seeds=[r[1] for r in initial],
            formal_training_visits=0)
        if proof.get('training_subset') != expected:
            raise ValueError('Capacity probe did not use the selected Train240 data')
        required.add('onpolicy/utils/stage3_b0_subset.py')
    if set(proof.get('code_files', {})) != required:
        raise ValueError('Requested capacity source binding is incomplete')
    for relative, checksum in proof['code_files'].items():
        if digest_file(Path(m['source_root'])/relative) != checksum:
            raise ValueError(f'Requested capacity source changed: {relative}')
    parent = bound_json(a['parent_manifest'])
    core = 'onpolicy/runner/shared/stage3_b_shared_b0_engine.py'
    if digest_file(Path(parent['source_root'])/core) != digest_file(Path(m['source_root'])/core):
        raise ValueError('Requested-parameter continuation changed the PPO engine')
    if (Path(runtime['module']).resolve() != (Path(m['source_root'])/'onpolicy/utils/stage3_b0_single_model.py').resolve()
            or digest_file(proof['fixture']['path']) != proof['fixture']['sha256']):
        raise ValueError('Requested capacity module or fixture changed')


def verify_single_model_sample(m, runtime, sample):
    """New sampler admission uses real batched trajectories, not legacy parity."""
    a = require_waiver(m)
    parent = bound_json(a['parent_manifest'])
    commit = bound_json(a['parent_commit'])
    if (runtime['sampling_lanes'] != 1 or runtime['global_batch'] != 64
            or Path(runtime['module']).resolve() != (Path(m['source_root'])/'onpolicy/utils/stage3_b0_single_model.py').resolve()
            or runtime['proofs']['sample'] != a.get('sampling_probe')
            or a.get('sampling_architecture') != 'single_model_batched_v1'
            or sample.get('architecture') != a['sampling_architecture']
            or not sample.get('passed') or sample.get('complete_trajectories') != 64
            or sample.get('environment_count') != 64 or sample.get('model_copies') != 1
            or sample.get('first_forward_batch') != 64
            or sample.get('environment_cuda_visible_devices') != ''
            or not sample.get('checkpoint_restore_exact') or not sample.get('cuda_rng_restored_exact')
            or not sample.get('repeated_forward_and_rng_exact')
            or sample.get('rng_contract') != 'checkpoint_global_torch_live_slot_order_v1'
            or sample.get('parent_manifest_sha256') != parent['manifest_sha256']
            or sample.get('checkpoint_sha256') != commit['checkpoint_sha256']
            or sample.get('next_batch') != commit['next_batch']
            or sample.get('updates_executed') != 0
            or not sample.get('replay', {}).get('decisions')
            or not 0 <= sample.get('replay', {}).get('max_logp_error', float('inf')) <= .002):
        raise ValueError('Single-model sampling/replay proof failed')
    required = {'onpolicy/utils/stage3_b0_single_model.py',
                'onpolicy/runner/shared/stage3_b_shared_b0_engine.py',
                'onpolicy/utils/stage3_b0_throughput.py',
                'onpolicy/scripts/train/probe_stage3_b0_single_model.py'}
    if set(sample.get('code_files', {})) != required:
        raise ValueError('Single-model probe code binding is incomplete')
    for relative, checksum in sample['code_files'].items():
        if digest_file(Path(m['source_root'])/relative) != checksum:
            raise ValueError(f'Single-model probe source changed: {relative}')
    # The historical full micro64 update proves only the unchanged PPO math.
    # Compare all engine methods other than the newly admitted collector.
    import ast
    def unchanged_methods(root):
        tree = ast.parse((Path(root)/'onpolicy/runner/shared/stage3_b_shared_b0_engine.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'BSharedB0Engine')
        return {node.name:ast.dump(node, include_attributes=False) for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name != '_collect'}
    if unchanged_methods(m['source_root']) != unchanged_methods(parent['source_root']):
        raise ValueError('Single-model continuation changed inherited PPO or restore methods')
