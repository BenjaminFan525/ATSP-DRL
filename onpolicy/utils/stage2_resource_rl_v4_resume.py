"""Strict checkpoint restoration and immutable continuation accounting."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from onpolicy.utils.stage2_resource_rl import RESOURCE_PREFIXES, file_sha, summary
from onpolicy.utils.stage2_resource_rl_v4 import ARMS, PROTOCOL


def assert_tree_equal(expected, actual, label='state'):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or expected.dtype != actual.dtype or not torch.equal(expected.cpu(), actual.cpu()):
            raise ValueError(f'Changed tensor in restored {label}.')
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f'Changed keys in restored {label}.')
        for key in expected:
            assert_tree_equal(expected[key], actual[key], f'{label}/{key}')
    elif isinstance(expected, (tuple, list)):
        if len(expected) != len(actual):
            raise ValueError(f'Changed sequence length in restored {label}.')
        for index, (a, b) in enumerate(zip(expected, actual)):
            assert_tree_equal(a, b, f'{label}/{index}')
    elif expected != actual:
        raise ValueError(f'Changed value in restored {label}.')


def restore_learner(learner, checkpoint, arm):
    c = checkpoint['resource_rl_contract']
    s = checkpoint['resource_rl_training_state']
    if c['protocol'] != PROTOCOL or c['arm'] != arm or c['training'] != learner.contract:
        raise ValueError('Checkpoint algorithm/arm/training contract differs.')
    if c['completed_rounds'] != s['collection_round'] or c['case_episodes'] != 24 * c['completed_rounds']:
        raise ValueError('Only complete original 24-case collection checkpoints can resume.')
    if c['protected_before'] != c['protected_after'] or summary(checkpoint['model'], protected=True) != c['protected_before']:
        raise ValueError('Checkpoint changed frozen parameters.')
    learner.policy.load_model_state(checkpoint['model'])
    if summary(learner.policy.ac) != summary(checkpoint['model']):
        raise ValueError('Restored model differs from checkpoint.')
    if summary(learner.policy.ac, RESOURCE_PREFIXES) != c['actor_after']:
        raise ValueError('Restored actor fingerprint differs.')
    # Adam deliberately leaves non-capturable ``step`` tensors on CPU. Its
    # load_state_dict may retain those exact input tensor objects, even for a
    # CUDA model. Clone before loading so independent learners cannot advance
    # one another's bias-correction counters or mutate an immutable checkpoint.
    learner.actor_optim.load_state_dict(copy.deepcopy(s['actor_optim']))
    learner.critic_optim.load_state_dict(copy.deepcopy(s['critic_optim']))
    assert_tree_equal(s['actor_optim'], learner.actor_optim.state_dict(), 'actor Adam')
    assert_tree_equal(s['critic_optim'], learner.critic_optim.state_dict(), 'critic Adam')
    learner.actor_steps, learner.critic_steps = c['actor_steps'], c['critic_steps']
    learner.protected, learner.source_actor = copy.deepcopy(c['protected_before']), copy.deepcopy(c['actor_before'])
    learner.probability_checks = copy.deepcopy(c['probability_checks'])
    learner.minibatch_checks = copy.deepcopy(c['minibatch_checks'])
    torch.set_rng_state(s['torch_rng'].cpu())
    if learner.device.type == 'cuda':
        torch.cuda.set_rng_state(s['cuda_rng'].cpu(), learner.device)
    learner.assert_protected()
    return dict(passed=True, arm=arm, completed_rounds=c['completed_rounds'], actor_steps=learner.actor_steps,
        critic_steps=learner.critic_steps, model_exact=True, actor_adam_exact=True, critic_adam_exact=True,
        critic_warmup_repeated=False, frozen_reference='original_B0_not_resumed_actor',
        rng='saved_torch_restored_then_original_round_specific_seed')


def select_committed_checkpoint(parent, arm):
    parent = Path(parent).resolve()
    result_path = parent / arm / 'result.json'
    state = json.loads(result_path.read_text())
    completed = int(state['completed_rounds'])
    path = parent / arm / f'checkpoint_round_{completed}.pt'
    if Path(state['checkpoint']).resolve() != path or completed < 1:
        raise ValueError('Result does not identify a complete committed checkpoint.')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    c = payload['resource_rl_contract']
    if c['arm'] != arm or c['completed_rounds'] != completed or c['case_episodes'] != state['case_episodes']:
        raise ValueError('Checkpoint/result commit mismatch.')
    if payload['resource_rl_training_state']['collection_round'] != completed:
        raise ValueError('Refusing a partially updated interruption checkpoint.')
    update_path = parent / arm / f'round_{completed}_update.json'
    update = json.loads(update_path.read_text())
    if update['actor_steps'] != c['actor_steps'] or update['critic_steps'] != c['critic_steps']:
        raise ValueError('Checkpoint/update counter mismatch.')
    return dict(path=str(path), sha256=file_sha(path), state=state,
        result_path=str(result_path), result_sha256=file_sha(result_path),
        update_path=str(update_path), update_sha256=file_sha(update_path))


def pending_evaluations(arm, completed, labels, contract, paused=False):
    if paused:
        return []
    result = [arm + f'_round_{r}' for r in contract['evaluation_rounds']
              if r == completed and arm + f'_round_{r}' not in labels]
    if completed == contract['rounds'] and arm + '_H_DIAGNOSTIC' not in labels:
        result.append(arm + '_H_DIAGNOSTIC')
    # Cannot evaluate round 10 using a later checkpoint. Missing earlier evals
    # must be resolved explicitly instead of silently relabeling new weights.
    if any(r < completed and arm + f'_round_{r}' not in labels for r in contract['evaluation_rounds']):
        raise ValueError('An earlier mandatory evaluation is missing at this checkpoint.')
    return result


def continuation_budget(manifest, checkpoints, evaluations, parent_attempted):
    c = manifest['training_contract']
    committed_training = sum(x['state']['case_episodes'] for x in checkpoints.values())
    preserved_eval = sum(len(x['cases']) for x in evaluations.values())
    committed_physical = committed_training + 240 + preserved_eval
    if parent_attempted < committed_physical:
        raise ValueError('Parent attempted ledger is below committed artifact coverage.')
    remaining = {}
    for arm in ARMS:
        state = checkpoints[arm]['state']
        done = state['completed_rounds']
        paused = state['status'] == 'paused_for_kl'
        pending_evaluations(arm, done, evaluations, c, paused)
        future_labels = [arm + f'_round_{r}' for r in c['evaluation_rounds'] if r >= done]
        future_labels += [arm + '_H_DIAGNOSTIC']
        remaining[arm] = dict(training=0 if paused else (c['rounds'] - done) * c['rollout_cases'],
            evaluation=0 if paused else 60 * sum(label not in evaluations for label in future_labels))
        remaining[arm]['total'] = remaining[arm]['training'] + remaining[arm]['evaluation']
    overhead = parent_attempted - committed_physical
    if overhead > 60:
        raise ValueError('Unexpected restart overhead exceeds one uncommitted evaluation (60 cases).')
    limit = manifest['execution']['total_case_episode_limit'] + overhead
    if parent_attempted + sum(v['total'] for v in remaining.values()) > limit:
        raise ValueError('Continuation would exceed the original logical experiment plus recorded interruption overhead.')
    return dict(original_physical_limit=manifest['execution']['total_case_episode_limit'],
        parent_attempted=parent_attempted, parent_committed_physical=committed_physical,
        parent_committed_training=committed_training, restart_uncommitted_case_attempts=overhead,
        physical_limit_including_restart=limit, remaining_by_arm=remaining,
        logical_training_limit=manifest['execution']['total_training_case_episodes'],
        extra_optimization_rounds=0, automatic_retry=False)
