"""Audited single-flow checkpoint migration with explicit batch-change boundaries."""
from pathlib import Path

from onpolicy.utils.stage3_b_shared_b0 import SOURCE_SHA
from onpolicy.utils.stage3_h3_continuation import (
    PROTOCOL, MB128_PROFILE, MB128_FRESH_PROFILE, MB192_PROFILE, BATCH_RESIZE_PROFILES,
    PROTOCOL_CHANGE_PROFILES, EPOCH_BUDGETS, schedule, validate_recipe, planned_new_steps,
)
from onpolicy.utils.stage3_h3_frozen import bind, checked
from onpolicy.utils.stage3_research import atomic_json, digest_json, read_json


def verify_recipe_change(before, after):
    validate_recipe(before); validate_recipe(after)
    changes = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    if before.get('execution_mode') != 'single' or after.get('execution_mode') != 'single':
        raise ValueError('Resize requires a single flow')
    if after['execution_profile'] in BATCH_RESIZE_PROFILES:
        expected_origin = MB128_FRESH_PROFILE if after['execution_profile'] == MB192_PROFILE else 'single_parent_env384_v1'
        if (before['execution_profile'] != expected_origin
                or changes != {'execution_profile','optimizer_minibatch','microbatch',
                               'physical_microbatch_candidates','optimizer_resize_after_epoch'}):
            raise ValueError('Batch resize permits only its registered origin at a completed epoch')
    elif after['execution_profile'] in PROTOCOL_CHANGE_PROFILES:
        allowed = {'execution_profile','post_pass_replay','epochs','evaluation_epochs'}
        if (before['execution_profile'] not in (MB192_PROFILE, *PROTOCOL_CHANGE_PROFILES)
                or changes - allowed):
            raise ValueError('Post-pass replay removal permits only its registered origin and field')
        if before['execution_profile'] == MB192_PROFILE and 'post_pass_replay' not in changes:
            raise ValueError('Leaving the post-pass replays on requires the registered field change')
        if after['epochs'] not in EPOCH_BUDGETS or (
                after['epochs'] != before['epochs'] and (before['epochs'], after['epochs']) != (8, 10)):
            raise ValueError('Epoch extension permits only the registered +2 budget')
    elif (changes != {'execution_profile','rollout_workers'}
          or after['execution_profile'] != 'single_parent_env384_v1'):
        raise ValueError('Resize permits only the registered single-flow sampling width change')


def describe_origin(previous, current, suite_path):
    verify_recipe_change(previous['recipe'],current['recipe'])
    for key in ('parent_manifest','baselines','frozen_iga','frozen_manifest','checkpoints','splits','resources'):
        if previous[key] != current[key]:
            raise ValueError('Resize changed a frozen input: '+key)
    before_schedule = schedule(previous['splits']['train'], previous['recipe'])
    after_schedule = schedule(current['splits']['train'], current['recipe'])
    if after_schedule[:len(before_schedule)] != before_schedule:
        raise ValueError('Resize changed the visit schedule')
    permitted = {'onpolicy/utils/stage3_h3_continuation.py',
        'onpolicy/utils/stage3_h3_resize.py',
        'onpolicy/scripts/train/run_stage3_h3_continuation.py',
        'onpolicy/scripts/train/stage3_h3_continuation_worker.py',
        'onpolicy/envs/HKBZ/test/test_stage3_h3_single.py',
        'onpolicy/envs/HKBZ/test/test_stage3_h3_resize.py'}
    batch_change = current['recipe'].get('execution_profile') in BATCH_RESIZE_PROFILES
    protocol_change = current['recipe'].get('execution_profile') in PROTOCOL_CHANGE_PROFILES
    if batch_change or protocol_change:
        permitted |= {'onpolicy/runner/shared/stage3_h3_frozen_engine.py',
                      'onpolicy/utils/stage3_h3_checkpoint_switch.py',
                      'onpolicy/scripts/train/stage3_h3_checkpoint_switch.py',
                      'onpolicy/envs/HKBZ/test/test_stage3_h3_checkpoint_switch.py',
                      'onpolicy/envs/HKBZ/test/test_stage3_h3_mb192_switch.py'}
    if protocol_change:
        permitted |= {'onpolicy/utils/stage3_h3_frozen.py',
                      'onpolicy/envs/HKBZ/test/test_stage3_h3_nopost_replay.py'}
    changed = {k for k in previous['source_files'].keys() | current['source_files'].keys()
               if previous['source_files'].get(k) != current['source_files'].get(k)}
    if changed - permitted:
        raise ValueError('Resize changed model/environment/PPO computation: '+str(changed-permitted))
    root = Path(previous['root'])
    arm_path = root/'arms/C03/manifest.json'
    arm = read_json(arm_path)
    if digest_json({k:v for k,v in arm.items() if k!='manifest_sha256'}) != arm['manifest_sha256']:
        raise ValueError('Previous arm identity changed')
    paths = sorted((root/'arms/C03/commits').glob('epoch_*.json'))
    rows = [read_json(p) for p in paths]
    if not rows or not 1 <= len(rows) < current['recipe']['epochs']:
        raise ValueError('Resume requires an unfinished run with complete committed epochs')
    if batch_change and len(rows) != current['recipe']['optimizer_resize_after_epoch']:
        raise ValueError('Batch resize must use exactly the registered checkpoint boundary')
    for epoch,row in enumerate(rows,1):
        checked(row['checkpoint']); checked(row['update'])
        if (row['epoch'] != epoch or row['manifest_sha256'] != arm['manifest_sha256']
                or not row['ppo_budget_complete'] or row['new_ppo_steps'] != planned_new_steps(arm['recipe'],epoch)
                or row['inherited_ppo_steps'] != 96
                or row['cumulative_ppo_steps'] != 96+planned_new_steps(arm['recipe'],epoch)
                or row['training_episodes'] != 384*epoch):
            raise ValueError('Previous committed training ledger is inconsistent')
    return dict(suite=bind(suite_path),arm_manifest=bind(arm_path),commits=[bind(p) for p in paths],
                epoch=len(rows),cumulative_ppo_steps=rows[-1]['cumulative_ppo_steps'],
                started_unix=read_json(root/'budget_clock.json')['started_unix'],
                changed_source_files=sorted(changed),algorithm_change=batch_change)


def rebind_payload(payload, old_arm, new_arm, commit):
    verify_recipe_change(old_arm['recipe'],new_arm['recipe'])
    expected = dict(protocol=PROTOCOL,source_sha256=SOURCE_SHA,
        manifest_sha256=old_arm['manifest_sha256'],recipe_sha256=digest_json(old_arm['recipe']),
        initialization_sha256=old_arm['parent_checkpoint']['sha256'],
        planning=old_arm['recipe']['planning'],history=old_arm['recipe']['history'],
        next_batch=commit['epoch'],policy_updates=commit['cumulative_ppo_steps'],
        physical_microbatch=old_arm['recipe']['microbatch'],train_tau=.03,
        complete_logical_rollout=True,diagnostic_only=False)
    if any(payload.get(k) != v for k,v in expected.items()):
        raise ValueError('Checkpoint is not the bound complete training state')
    for key in ('model','actor_optim','critic_optim','value_normalizer','normalization_sha256',
                'rng_python','rng_numpy','rng_torch','rng_cuda'):
        if key not in payload or payload[key] is None:
            raise ValueError('Checkpoint state is missing: '+key)
    result = dict(payload,manifest_sha256=new_arm['manifest_sha256'],
                  recipe_sha256=digest_json(new_arm['recipe']))
    if new_arm['recipe'].get('execution_profile') in BATCH_RESIZE_PROFILES:
        if (commit['epoch'] > new_arm['recipe']['optimizer_resize_after_epoch']
                or commit['new_ppo_steps'] != planned_new_steps(new_arm['recipe'],commit['epoch'])):
            raise ValueError('Checkpoint does not match the historical optimizer budget')
        result['physical_microbatch'] = new_arm['recipe']['microbatch']
    elif new_arm['recipe'].get('execution_profile') in PROTOCOL_CHANGE_PROFILES:
        if (commit['new_ppo_steps'] != planned_new_steps(new_arm['recipe'],commit['epoch'])
                or new_arm['recipe']['microbatch'] != old_arm['recipe']['microbatch']):
            raise ValueError('Checkpoint does not match the registered optimizer budget')
        result['physical_microbatch'] = new_arm['recipe']['microbatch']
    return result


def import_committed_state(suite, arm):
    import torch
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states
    origin = suite['resume_origin']; old_arm = read_json(checked(origin['arm_manifest']))
    mapping = {}; receipts = []
    for record in origin['commits']:
        old_commit = read_json(checked(record))
        payload = torch.load(checked(old_commit['checkpoint']),map_location='cpu',weights_only=False)
        rebound = rebind_payload(payload,old_arm,arm,old_commit)
        out = Path(arm['root'])/'imported'/f'epoch_{old_commit["epoch"]:04d}'
        out.mkdir(parents=True)
        checkpoint = out/'checkpoint.pt'
        with checkpoint.open('xb') as handle:
            torch.save(rebound,handle)
        restored = torch.load(checkpoint,map_location='cpu',weights_only=False)
        metadata = {'manifest_sha256','recipe_sha256'}
        if arm['recipe'].get('execution_profile') in (*BATCH_RESIZE_PROFILES, *PROTOCOL_CHANGE_PROFILES):
            metadata.add('physical_microbatch')
        if any(restored[k] != rebound[k] for k in metadata):
            raise ValueError('Serialized checkpoint execution metadata changed')
        compare_states({k:v for k,v in payload.items() if k not in metadata},
                       {k:v for k,v in restored.items() if k not in metadata},exact=True)
        update = read_json(checked(old_commit['update']))
        update['manifest_sha256'] = arm['manifest_sha256']
        update['imported_from'] = old_commit['update']
        atomic_json(out/'update.json',update,overwrite=False)
        commit = dict(old_commit,manifest_sha256=arm['manifest_sha256'],checkpoint=bind(checkpoint),
                      update=bind(out/'update.json'),imported_from=record)
        atomic_json(Path(arm['root'])/'commits'/f'epoch_{commit["epoch"]:04d}.json',commit,overwrite=False)
        mapping[old_commit['checkpoint']['sha256']] = commit['checkpoint']
        receipts.append(dict(epoch=commit['epoch'],source=old_commit['checkpoint'],
            rebound=commit['checkpoint'],all_training_state_exact=True,
            metadata_changes=sorted(metadata),cumulative_ppo_steps=commit['cumulative_ppo_steps']))
    atomic_json(Path(suite['root'])/'resume_migration.json',dict(passed=True,epochs=receipts,
        next_epoch=origin['epoch']+1,remaining_epochs=suite['recipe']['epochs']-origin['epoch'],
        original_wall_budget_preserved=True),overwrite=False)
    return mapping
