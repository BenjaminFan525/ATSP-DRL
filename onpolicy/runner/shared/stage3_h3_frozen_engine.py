"""Full-shared H3 PPO with independent sampling, SGD and physical batch sizes."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random

import numpy as np
import torch

from onpolicy.runner.shared.stage3_b_shared_b0_engine import PPOContractError
from onpolicy.runner.shared.stage3_research_engine import gradient_mode
from onpolicy.utils.stage3_b0_single_model import ThroughputEngine as SingleModelEngine
from onpolicy.utils.stage3_h3_frozen import PLANNING, PROTOCOL, optimizer_orders
from onpolicy.utils.stage3_b_shared_b0 import HISTORY, SOURCE_SHA
from onpolicy.utils.stage3_performance import DeferredScalars
from onpolicy.utils.stage3_research import digest_file, digest_json


def model_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


class H3FrozenEngine(SingleModelEngine):
    def __init__(self, frozen_manifest, *, config, runtime=None, device='cuda:0', training=True):
        run = dict(runtime or {})
        run.update(sampling_lanes=1, environment_processes=config['environment_processes'],
                   input_cache_mib=config['input_cache_mib'],
                   encoder_activation_checkpoint=config['encoder_activation_checkpoint'])
        if torch.device(device).type == 'cuda':
            # Leave space for CUDA context/workspaces outside the Torch allocator.
            run['cuda_memory_headroom_mib'] = config['gpu_headroom_mib'] + 1024
        super().__init__(frozen_manifest, config=config, runtime=run, device=device,
                         training=training, width=config['rollout_workers'], create_pool=False)
        self.initialization_sha256 = SOURCE_SHA
        self.physical_microbatch = config['microbatch']
        self.commit_ready = True

    def environment_config(self, case, seed):
        config = super().environment_config(case, seed)
        config.update(PLANNING)
        # No heuristic/teacher fallback can initiate an optimization here.
        config['resource_policy'] = 'drl'
        return config

    def warm_start(self, path, *, expected_sha256):
        if digest_file(path) != expected_sha256:
            raise ValueError('Warm-start checkpoint changed')
        p = torch.load(path, map_location='cpu', weights_only=False)
        if expected_sha256 != SOURCE_SHA and (p.get('source_sha256') != SOURCE_SHA
                                             or p.get('diagnostic_only')):
            raise ValueError('Warm start must be a committed B0-descended checkpoint')
        self.policy.ac.load_state_dict(p['model'], strict=True)
        if any(not torch.equal(v.cpu(), p['model'][k]) for k, v in self.policy.ac.state_dict().items()):
            raise ValueError('Warm-start weights were not loaded exactly')
        self.policy.actor_optimizer.state.clear()
        self.policy.critic_optimizer.state.clear()
        self.policy_updates = 0
        self.initialization_sha256 = expected_sha256
        self.commit_ready = True
        self.cache.last_actor = None
        self.assert_frozen()

    def collect_logical(self, cases, seeds, visit_ids, *, logical_id, heartbeat=None):
        if len(cases) != len(seeds) or len(cases) != len(visit_ids):
            raise ValueError('Logical collection coverage mismatch')
        if not self.commit_ready:
            raise PPOContractError('An uncommitted update cannot start another collection')
        identity, version = model_digest(self.policy.ac), self.policy_updates
        result = []
        for start in range(0, len(cases), self.width):
            end = start + self.width
            rows = self.parallel_collect(cases[start:end], seeds[start:end], heartbeat)
            if self.policy_updates != version or model_digest(self.policy.ac) != identity:
                raise PPOContractError('Behavior policy changed between collection waves')
            for t, visit in zip(rows, visit_ids[start:end]):
                t.update(visit_id=visit, logical_id=logical_id, behavior_model_sha256=identity)
            result.extend(rows)
        if len(result) != len(cases) or len({t['visit_id'] for t in result}) != len(result):
            raise PPOContractError('Logical collection has missing/duplicate visits')
        return result

    def _minibatch_backward(self, group, source_costs, microbatch, heartbeat=None):
        count = len(group)
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        health = dict(nonfinite=0, mask_mismatch=0)
        result = dict(actor_loss=0., critic_loss=0.)
        roles = {str(i): dict(decisions=0, kl_sum=0., clip_count=0.) for i in range(3)}
        deferred = DeferredScalars()
        for start in range(0, count, microbatch):
            chunk = group[start:start + microbatch]
            actor_loss = critic_loss = None
            steps = max(len(t['states']) for t in chunk)
            for step, ids, states, graph, lp, mask in self._replay(chunk):
                expected = tensor(np.stack([s['mask'] for s in states]))
                diff = lp - tensor(np.stack([s['old_logp'] for s in states]))
                ratio = diff.clamp(-40, 40).exp()
                advantage = tensor([.01 * (source_costs[chunk[i]['case_id']] - chunk[i]['makespan'])
                                    for i in ids])[:, None]
                loss = -(torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
                         * expected).sum() / count
                deferred.add(health, 'nonfinite', (~torch.isfinite(lp)).sum())
                deferred.add(health, 'mask_mismatch', (mask != expected).sum())
                deferred.add(result, 'actor_loss', loss)
                actor_loss = loss if actor_loss is None else actor_loss + loss
                state_roles = tensor(np.stack([s['roles'] for s in states]))
                for role, row in roles.items():
                    selected = expected.bool() & (state_roles == int(role))
                    d = diff.detach()[selected]
                    row['decisions'] += d.numel()
                    if d.numel():
                        deferred.add(row, 'kl_sum', (d.clamp(-40, 40).exp() - 1 - d).sum())
                        deferred.add(row, 'clip_count', ((d.clamp(-40, 40).exp() - 1).abs() > .2).sum())
                prepared = self.cache.prepare_graph(graph)
                encoded = self.policy.ac._encode_graph(prepared, actor_grad=False)
                values = self.policy.ac.team_critic(encoded['global_emb'].detach()).reshape(-1)
                remaining = tensor([-.01 * (chunk[i]['makespan'] - s['time']) for i, s in zip(ids, states)])
                target = self.norm.normalize(remaining[:, None]).reshape(-1)
                weights = tensor([1. / (count * len(chunk[i]['states'])) for i in ids])
                c_loss = ((values - target.detach()).square() * weights).sum()
                deferred.add(result, 'critic_loss', c_loss)
                deferred.add(health, 'nonfinite', (~torch.isfinite(values)).sum())
                critic_loss = c_loss if critic_loss is None else critic_loss + c_loss
                if (step + 1) % self.config['tbptt'] == 0 or step + 1 == steps:
                    deferred.flush()
                    if any(health.values()):
                        raise PPOContractError(f'Probability/value replay failure: {health}')
                    if actor_loss.requires_grad:
                        actor_loss.backward()
                    critic_loss.backward()
                    actor_loss = critic_loss = None
                if heartbeat and step % 50 == 0:
                    heartbeat.update(event='ppo_backward', update_step=step, physical_chunk_start=start)
        deferred.flush()
        if any(health.values()):
            raise PPOContractError(f'Probability/value replay failure: {health}')
        decisions = sum(row['decisions'] for row in roles.values())
        for row in roles.values():
            row['kl'] = row['kl_sum'] / max(row['decisions'], 1)
        result['pre_step'] = dict(roles=roles, decisions=decisions,
                                 kl=sum(row['kl_sum'] for row in roles.values()) / max(decisions, 1))
        return result

    def update_logical(self, trajectories, source_costs, *, logical_id, shuffle_seed,
                       heartbeat=None, diagnostic=False, minibatch=None, microbatch=None, passes=None):
        batch = minibatch or self.config['optimizer_minibatch']
        physical = microbatch or self.physical_microbatch
        passes = passes or self.config['ppo_epochs']
        if not self.training or not self.normalization_sha256 or not self.commit_ready:
            raise PPOContractError('Update needs a normalized, committed training state')
        if not diagnostic and (len(trajectories), batch, passes) != (384, 64, 2):
            raise PPOContractError('Formal PPO budget changed')
        if physical > batch or batch % physical:
            raise PPOContractError('Physical chunks must partition an optimizer minibatch')
        identity, behavior_version = model_digest(self.policy.ac), self.policy_updates
        if (not trajectories or len({t['visit_id'] for t in trajectories}) != len(trajectories)
                or any(not t['completed'] or t.get('cycle_terminated') or not t['states']
                       or t['behavior_deterministic'] or t['forced_replay']
                       or t['history'] != HISTORY or t['decoder'] != 'AR_sample' or t['tau'] != .03
                       or t['policy_updates'] != behavior_version or t['logical_id'] != logical_id
                       or t['behavior_model_sha256'] != identity for t in trajectories)):
            raise PPOContractError('Expected every complete visit from the same fresh behavior policy')
        if any(not np.isfinite(source_costs.get(t['case_id'], np.nan))
               or source_costs[t['case_id']] <= 0 for t in trajectories):
            raise PPOContractError('Missing H3 source cost')
        orders = optimizer_orders(len(trajectories), batch, passes, shuffle_seed)
        self.policy.ac.tau = .03
        self.commit_ready = False
        applied, completed_passes, stopped, metrics, pass_metrics = 0, 0, None, [], []
        with self.cache.group():
            before = self.replay_metrics(trajectories, microbatch=physical, heartbeat=heartbeat)
            if before['max_logp_error'] > .002 or not before['decisions']:
                raise PPOContractError(f'Behavior likelihood cannot be reproduced: {before}')
            for pass_index, parts in enumerate(orders):
                for part_index, indices in enumerate(parts):
                    if heartbeat:
                        heartbeat.update(event='optimizer_minibatch', ppo_pass=pass_index + 1,
                                         minibatch=part_index + 1, behavior_version=behavior_version,
                                         current_version=self.policy_updates)
                    gradient_mode(self.policy.ac)
                    self.policy.actor_optimizer.zero_grad(set_to_none=True)
                    self.policy.critic_optimizer.zero_grad(set_to_none=True)
                    row = self._minibatch_backward([trajectories[i] for i in indices], source_costs,
                                                   physical, heartbeat)
                    peak_kl = max([row['pre_step']['kl'],
                                   *[v['kl'] for v in row['pre_step']['roles'].values()]])
                    row.update(ppo_pass=pass_index + 1, minibatch=part_index + 1,
                               visit_indices=indices, actor_step_applied=False)
                    if peak_kl > self.config['hard_kl']:
                        raise PPOContractError('Hard KL exceeded; roll back to a completed collection checkpoint')
                    if peak_kl > self.config['soft_kl']:
                        stopped = 'soft_kl'
                        metrics.append(row)
                        break
                    parameters = [p for g in self.policy.actor_optimizer.param_groups for p in g['params']]
                    row['actor_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(
                        parameters, self.config['gradient_clip'], error_if_nonfinite=True))
                    row['critic_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(
                        self.policy.ac.team_critic.parameters(), self.config['gradient_clip'], error_if_nonfinite=True))
                    self.policy.actor_optimizer.step()
                    self.policy.critic_optimizer.step()
                    self.policy_updates += 1
                    applied += 1
                    row.update(actor_step_applied=True, policy_updates=self.policy_updates)
                    metrics.append(row)
                    self.cache.last_actor = None
                post = self.replay_metrics(trajectories, microbatch=physical, heartbeat=heartbeat)
                pass_metrics.append(post)
                if max([post['kl'], *[v['kl'] for v in post['roles'].values()]]) > self.config['hard_kl']:
                    raise PPOContractError('Hard KL after pass; partial collection is not a commit')
                if stopped:
                    break
                completed_passes += 1
                if max([post['kl'], *[v['kl'] for v in post['roles'].values()]]) > self.config['soft_kl']:
                    stopped = 'soft_kl'
                    break
        self.policy.actor_optimizer.zero_grad(set_to_none=True)
        self.policy.critic_optimizer.zero_grad(set_to_none=True)
        self.assert_frozen()
        self.commit_ready = True
        return dict(logical_id=logical_id, behavior_version=behavior_version, behavior_model_sha256=identity,
                    visits=len(trajectories), optimizer_minibatch=batch, microbatch=physical,
                    planned_ppo_steps=len(trajectories) // batch * passes, actual_ppo_steps=applied,
                    completed_passes=completed_passes, stop_reason=stopped,
                    behavior_replay=before, minibatches=metrics, pass_metrics=pass_metrics,
                    optimizer_order_sha256=digest_json(orders), execution_cache=self.cache.last_report,
                    policy_updates=self.policy_updates, teacher_queries=0)

    def save(self, path, *, manifest_sha256, next_batch, diagnostic=False):
        if not diagnostic and not self.commit_ready:
            raise PPOContractError('Only a complete logical collection can be checkpointed for resume')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        p = dict(protocol=PROTOCOL, source_sha256=SOURCE_SHA, manifest_sha256=manifest_sha256,
                 initialization_sha256=self.initialization_sha256,
                 recipe_sha256=digest_json(self.config), history=HISTORY, planning=PLANNING,
                 model=self.policy.ac.state_dict(), actor_optim=self.policy.actor_optimizer.state_dict(),
                 critic_optim=self.policy.critic_optimizer.state_dict(), value_normalizer=self.norm.state_dict(),
                 normalization_sha256=self.normalization_sha256, policy_updates=self.policy_updates,
                 physical_microbatch=self.physical_microbatch, next_batch=next_batch,
                 diagnostic_only=diagnostic, complete_logical_rollout=self.commit_ready,
                 rng_torch=torch.get_rng_state(), rng_numpy=np.random.get_state(),
                 rng_python=random.getstate(),
                 rng_cuda=torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None)
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        try:
            with temporary.open('xb') as f:
                torch.save(p, f); f.flush(); os.fsync(f.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return digest_file(path)

    def resume(self, path, *, manifest_sha256, allow_diagnostic=False):
        p = torch.load(path, map_location='cpu', weights_only=False)
        if (p.get('protocol') != PROTOCOL or p.get('manifest_sha256') != manifest_sha256
                or p.get('source_sha256') != SOURCE_SHA or p.get('history') != HISTORY
                or p.get('planning') != PLANNING or p.get('recipe_sha256') != digest_json(self.config)
                or p.get('initialization_sha256') != self.initialization_sha256
                or not p.get('complete_logical_rollout')
                or (p.get('diagnostic_only') and not allow_diagnostic)
                or p.get('physical_microbatch') not in self.config['physical_microbatch_candidates']):
            raise PPOContractError('Resume identity/complete-collection contract mismatch')
        self.policy.ac.load_state_dict(p['model'], strict=True)
        self.policy.actor_optimizer.load_state_dict(p['actor_optim'])
        self.policy.critic_optimizer.load_state_dict(p['critic_optim'])
        self.norm.load_state_dict(p['value_normalizer'], strict=True)
        self.normalization_sha256 = p['normalization_sha256']
        self.policy_updates = p['policy_updates']
        self.physical_microbatch = p['physical_microbatch']
        random.setstate(p['rng_python']); np.random.set_state(p['rng_numpy'])
        torch.set_rng_state(p['rng_torch'])
        if self.device.type == 'cuda':
            if p['rng_cuda'] is None:
                raise PPOContractError('CUDA RNG missing from checkpoint')
            torch.cuda.set_rng_state(p['rng_cuda'], self.device)
        self.commit_ready = True
        self.cache.last_actor = None
        self.policy.ac.tau = .03
        self.assert_frozen()
        return p['next_batch']
