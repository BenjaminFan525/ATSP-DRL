"""Prepare an immutable batch-size continuation only after its bound commit exists."""
from __future__ import annotations

import copy
from pathlib import Path
import time

from onpolicy.utils.stage3_h3_continuation import (
    MB128_PROFILE, MB128_FRESH_PROFILE, MB192_PROFILE, BATCH_RESIZE_PROFILES,
    MB192_NOPOST_PROFILE, PROTOCOL_CHANGE_PROFILES, EPOCH_BUDGETS, recipe, planned_new_steps,
    FIXED_EPOCHS_POLICY,
)
from onpolicy.utils.stage3_h3_frozen import bind, checked
from onpolicy.utils.stage3_research import atomic_json, digest_file, digest_json, read_json


def request_identity(request):
    return digest_json({k:v for k,v in request.items() if k != 'request_sha256'})


def verify_request(request):
    if request_identity(request) != request['request_sha256']:
        raise ValueError('Checkpoint switch request identity changed')
    old = read_json(checked(request['origin']))
    target = request.get('target_profile', MB128_PROFILE)
    if target not in (*BATCH_RESIZE_PROFILES, *PROTOCOL_CHANGE_PROFILES):
        raise ValueError('Unregistered checkpoint switch target')
    width = 128 if target == MB128_PROFILE else 192
    if any(request.get(k,width) != width for k in ('requested_minibatch','requested_microbatch')):
        raise ValueError('Requested batch dimensions disagree with the registered target')
    origin = {MB128_PROFILE:('single_parent_env384_v1',), MB192_PROFILE:(MB128_FRESH_PROFILE,),
              MB192_NOPOST_PROFILE:(MB192_PROFILE, MB192_NOPOST_PROFILE)}[target]
    if (old['execution_mode'] != 'single'
            or old['recipe']['execution_profile'] not in origin
            or old['resources'] != request['resources']
            or old['resources']['gpu'] != 0 or old['resources']['cpus'] != '0-31,64-95'):
        raise ValueError('Switch origin/resources differ from the authorized single flow')
    if target in (MB192_PROFILE, MB192_NOPOST_PROFILE) and old['recipe'].get('stopping_policy') != FIXED_EPOCHS_POLICY:
        raise ValueError('192-wide profiles must preserve the fixed eight-epoch policy')
    if type(request['after_epoch']) is not int or not 1 <= request['after_epoch'] < 8:
        raise ValueError('Invalid checkpoint boundary')
    epochs = request.get('requested_epochs', old['recipe']['epochs'])
    if epochs not in EPOCH_BUDGETS or epochs < old['recipe']['epochs']:
        raise ValueError('Requested epoch budget is not a registered extension of the origin')
    for key in ('tests','plan'):
        checked(request[key])
    for name, expected in request['source_files'].items():
        if digest_file(Path(request['source_root'])/name) != expected:
            raise ValueError('Prepared switch source changed: '+name)
    return old


def boundary_commit(request):
    """A filename alone is insufficient: require the atomic ledger and its data."""
    old = read_json(checked(request['origin']))
    root = Path(old['root'])
    epoch = request['after_epoch']
    path = root/'arms/C03/commits'/f'epoch_{epoch:04d}.json'
    if not path.exists():
        return None
    arm = read_json(root/'arms/C03/manifest.json')
    if any((root/'arms/C03/commits'/f'epoch_{e:04d}.json').exists()
           for e in range(epoch+1, arm['recipe']['epochs']+1)):
        raise ValueError('The requested checkpoint boundary was already passed')
    if digest_json({k:v for k,v in arm.items() if k != 'manifest_sha256'}) != arm['manifest_sha256']:
        raise ValueError('Origin arm identity changed')
    row = read_json(path)
    if (row['epoch'] != epoch or row['manifest_sha256'] != arm['manifest_sha256']
            or not row['ppo_budget_complete']
            or row['new_ppo_steps'] != planned_new_steps(arm['recipe'],epoch)
            or row['inherited_ppo_steps'] != 96
            or row['cumulative_ppo_steps'] != 96+row['new_ppo_steps']
            or row['training_episodes'] != epoch*384):
        raise ValueError('Checkpoint is incomplete or has an inconsistent optimizer ledger')
    checked(row['checkpoint']); checked(row['update'])
    update = read_json(row['update']['path'])
    if any(update.get(k) != row[k] for k in
           ('epoch','manifest_sha256','new_ppo_steps','cumulative_ppo_steps','inherited_ppo_steps','training_episodes')):
        raise ValueError('Checkpoint and update ledgers disagree')
    return row


def prepare_at_boundary(request):
    from onpolicy.scripts.train.run_stage3_h3_continuation import (
        Controller, identity, migrate_baselines, verify,
    )
    from onpolicy.utils.stage3_h3_resize import describe_origin, import_committed_state
    from onpolicy.scripts.train.run_stage3_h3_frozen import check_tests

    old = verify_request(request); verify(old)
    if boundary_commit(request) is None:
        raise ValueError('Checkpoint boundary has not been committed')
    root = Path(request['root'])
    if (root/'manifest.json').exists():
        raise FileExistsError('This switch was already prepared; do not overwrite it')
    parent = read_json(checked(old['parent_manifest']))
    target = request.get('target_profile', MB128_PROFILE)
    boundary = (request['after_epoch'] if target in BATCH_RESIZE_PROFILES
                else old['recipe'].get('optimizer_resize_after_epoch'))
    epochs = request.get('requested_epochs', old['recipe']['epochs'])
    r = recipe(parent['recipe'], old['resources'], 'C03', execution='single',
               single_profile=target,
               optimizer_resize_after_epoch=boundary,
               stopping_policy=old['recipe'].get('stopping_policy'), epochs=epochs)
    m = copy.deepcopy(old)
    for key in ('resume_origin','reuse_suite'):
        m.pop(key, None)
    m.update(root=str(root),source_root=request['source_root'],source_files=request['source_files'],
             recipe=r,plan=request['plan'],tests=dict(request['tests'],
             passed=check_tests(Path(request['tests']['path']))['passed']),
             workspace_commit=request['workspace_commit'],created_unix=time.time(),
             reuse_suite=request['origin'],switch_request=bind(root/'switch_request.json'))
    m['budget'].update(screen_ppo_steps=planned_new_steps(r,2),
                       max_new_ppo_steps=planned_new_steps(r,r['epochs']),
                       physical_microbatch=r['microbatch'])
    width=r['microbatch'];old_width=old['recipe']['microbatch']
    change = (f'Optimizer minibatch {old_width} to {width}; keep two passes and 384 visits per epoch'
              if target in BATCH_RESIZE_PROFILES else
              f'Remove the two post-pass full replays at {width}/{width}; keep the pre-update '
              'full replay and gate each pass on its applied minibatch pre-step records')
    if epochs != old['recipe']['epochs']:
        change += (f'; extend the local budget from {old["recipe"]["epochs"]} to {epochs} epochs '
                   f'with evaluation epochs {r["evaluation_epochs"]}')
    m['material_passport'].update(authorization=f'User requested {width}/{width} after the next complete checkpoint',
        verification_status='CPU_TESTED_GPU_CAPACITY_PENDING',
        algorithm_change=change)
    m['resume_origin'] = describe_origin(old,m,Path(request['origin']['path']))
    m['manifest_sha256'] = identity(m)
    atomic_json(root/'manifest.json',m,overwrite=False); verify(m)
    migrate_baselines(m,old)
    selection = read_json(Path(old['root'])/'parent_selection.json')
    selection['manifest_sha256'] = m['manifest_sha256']
    atomic_json(root/'parent_selection.json',selection,overwrite=False)
    controller = Controller.__new__(Controller)
    controller.m,controller.path,controller.root = m,root/'manifest.json',root
    _,arm = controller.arm_manifest('C03',selection)
    mapping = import_committed_state(m,arm)
    migrate_baselines(m,old,labels=[f'C03_epoch_{e:04d}' for e in range(1,request['after_epoch']+1)],
                      checkpoint_map=mapping)
    atomic_json(root/'prepared.json',dict(prepared=True,manifest_sha256=m['manifest_sha256'],
        next_epoch=request['after_epoch']+1,solver_queries=0),overwrite=False)
    return m
