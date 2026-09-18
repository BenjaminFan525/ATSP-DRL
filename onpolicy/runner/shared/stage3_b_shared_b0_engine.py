"""Native B0 history/decoder with full-shared, visit-mean MAPPO updates.

The old representation engine is intentionally not an initialization path.
One GPU accumulates a complete global batch before clipping or stepping Adam.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import copy
import hashlib
import os
import random
import time

import numpy as np
import torch
import yaml

from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
from onpolicy.runner.shared.stage3_research_engine import EnvironmentPool, authoritative_history, gradient_mode, seed_all
from onpolicy.utils.stage2_frozen import load_frozen_stage2, PLANNING
from onpolicy.utils.stage3_b_shared_b0 import FROZEN, HISTORY, PROTOCOL, SOURCE_SHA, recipe
from onpolicy.utils.stage3_numerics import configure_runtime, install_stable_pool
from onpolicy.utils.stage3_performance import GroupExecutionCache, DeferredScalars
from onpolicy.utils.stage3_research import digest_file, digest_json
from onpolicy.utils.valuenorm import ValueNorm

ACTOR_PREFIXES = ('encoder.', 'plane_sel_enc.', 'actor.', 'device_sel_enc.', 'device_actor.',
                  'transporter_sel_enc.', 'transporter_actor.')
FROZEN_PREFIXES = ('request_ready_head.', 'request_ready_feature.', 'plane_critic.',
                   'device_critic.', 'transporter_critic.')


class PPOContractError(RuntimeError):
    pass


class BSharedB0Engine:
    def __init__(self, frozen_manifest=FROZEN, *, device='cuda:0', width=16, training=True,
                 config=None, create_pool=True):
        self.config = dict(config or recipe())
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index not in (None, 0):
            raise ValueError('B0 study uses only the single visible CUDA device')
        torch.set_num_threads(1)
        configure_runtime()
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        checkpoint, self.args, self.bundle = load_frozen_stage2(frozen_manifest)
        r = self.config
        self.args.lr, self.args.critic_lr = r['actor_lr'], r['critic_lr']
        for role, scale in zip(('shared', 'plane', 'device', 'transporter'), r['actor_lr_scales']):
            setattr(self.args, role + '_actor_lr_scale', scale)
        self.args.shared_encoder_activation_checkpoint = False
        self.policy = GNN_MAPPOPolicy(self.args, yaml.safe_load(self.bundle.path('ac_config').read_text()), self.device)
        self.policy.ac.load_state_dict(checkpoint['model'], strict=True)
        if not all(torch.equal(v.cpu(), checkpoint['model'][k]) for k, v in self.policy.ac.state_dict().items()):
            raise ValueError('Initial model differs from frozen B0')
        self.bundle.validate(self.policy.ac.state_dict())
        self.source_metadata = {k: copy.deepcopy(v) for k, v in checkpoint.items() if k != 'model'}
        self.source_path = str(self.bundle.path('b0'))
        self._frozen = {k: v.detach().cpu().clone() for k, v in self.policy.ac.state_dict().items()
                        if k.startswith(FROZEN_PREFIXES)}
        self._set_trainability(True)
        # The native policy owns separate role heads and exactly one GNN.
        self.policy.reset_optimizers()
        for group in self.policy.actor_optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p.requires_grad]
        self.policy.critic_optimizer = torch.optim.Adam(self.policy.ac.team_critic.parameters(),
            lr=r['critic_lr'], eps=self.policy.opti_eps, weight_decay=self.policy.weight_decay)
        actor_ids = {id(p) for g in self.policy.actor_optimizer.param_groups for p in g['params']}
        critic_ids = {id(p) for g in self.policy.critic_optimizer.param_groups for p in g['params']}
        if actor_ids & critic_ids or not actor_ids or not critic_ids:
            raise ValueError('Actor and team-critic Adam ownership must be disjoint')
        self.norm = ValueNorm(1, device=self.device)
        self.normalization_sha256 = None
        self.policy_updates = 0
        self.width = int(width)
        self.training = bool(training)
        self.cache = GroupExecutionCache(self.policy, r['input_cache_mib'], frozen_features=False)
        self.policy.stage3_execution_cache = self.cache
        self.policy.ac.stage3_execution_cache = self.cache
        install_stable_pool(self.policy)
        self._set_trainability(training)
        self.policy.ac.tau = r['train_tau'] if training else r['evaluation_tau']
        self.policy.ac.eval()
        self.pool = EnvironmentPool(width, timeout=r['ipc_timeout_seconds']) if create_pool else None

    def _set_trainability(self, training):
        if training:
            for name, p in self.policy.ac.named_parameters():
                p.requires_grad_(name.startswith(ACTOR_PREFIXES + ('team_critic.',)))
        else:
            # Preserve native B0 no-grad cuDNN dispatch (requires_grad is part
            # of that dispatch even when autograd is disabled).
            from onpolicy.utils.stage2_resource_rl import configure_trainability
            configure_trainability(self.policy)
        self.policy.ac.eval()

    def assert_frozen(self):
        for name, before in self._frozen.items():
            if not torch.equal(before, self.policy.ac.state_dict()[name].detach().cpu()):
                raise PPOContractError(f'Unused/frozen parameter changed: {name}')

    def environment_config(self, case, seed):
        config = yaml.safe_load(self.bundle.path('env_config').read_text())
        path = Path(case)
        for key, file in (('jobs_path', 'job.json'), ('fixed_res_path', 'fixed_resources.json'),
                          ('mobile_res_path', 'mobile_resources.json'), ('sites_path', 'sites.json'),
                          ('flights_path', 'flights.json')):
            config[key] = str(path / file)
        # Same explicit fields as the production make_train_env contract.
        for key in ('hindsight_reward_mode', 'hindsight_cmax_coef', 'hindsight_shaping_coef',
                    'hindsight_terminal_cmax_coef', 'iga_potential_weights_path', 'iga_potential_beta',
                    'iga_potential_gamma', 'device_deadlock_repeat_limit', 'resource_lateness_coef',
                    'resource_critical_lateness_coef', 'resource_earliness_coef',
                    'resource_slack_criticality_seconds', 'resource_slack_min_weight',
                    'resource_slack_forecast_seconds', 'plane_cycle_repeat_limit',
                    'plane_no_progress_limit', 'plane_relocation_limit', 'plane_first_completion_bonus',
                    'plane_repeat_relocation_penalty', 'plane_reset_job_penalty',
                    'plane_no_progress_penalty', 'plane_cycle_penalty'):
            config[key] = getattr(self.args, key)
        config.update(PLANNING)
        config.update(seed=int(seed), global_feature_mode='f1f2', resource_policy='drl',
            use_domain_rand=False, device_request_capacity_per_plane=5, pair_feature_storage='sparse_legal',
            stage2_resource_v6_observations=False, max_episode_steps=4000)
        config.update(hindsight_reward_mode='team_time', hindsight_terminal_cmax_coef=1.0,
            hindsight_cmax_coef=0.0, hindsight_shaping_coef=0.0, iga_potential_beta=0.0,
            resource_lateness_coef=0.0, resource_critical_lateness_coef=0.0, resource_earliness_coef=0.0)
        for key in ('iga_teacher_dir', 'resource_iga_teacher_dir', 'resource_iga_teacher_index',
                    'joint_iga_teacher_dir', 'joint_iga_teacher_index'):
            config[key] = ''
        return config

    @contextmanager
    def evaluation_mode(self, decoder='H'):
        if decoder not in ('H', 'AR'):
            raise ValueError('Unknown deployment decoder')
        before = {n: p.requires_grad for n, p in self.policy.ac.named_parameters()}
        tau, matching = self.policy.ac.tau, self.policy.ac.device_global_matching
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        self._set_trainability(False)
        self.policy.ac.tau = .3
        self.policy.ac.device_global_matching = decoder == 'H'
        torch.backends.cudnn.allow_tf32 = self.config['evaluation_cudnn_allow_tf32']
        try:
            yield
        finally:
            self.policy.ac.device_global_matching = matching
            self.policy.ac.tau = tau
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            for n, p in self.policy.ac.named_parameters():
                p.requires_grad_(before[n])
            self.policy.ac.eval()

    @torch.no_grad()
    def rollout(self, cases, seeds, *, deterministic=False, retain=False, decoder='H',
                native=False, heartbeat=None, record_times=False):
        if self.pool is None or not 0 < len(cases) <= self.width or len(seeds) != len(cases):
            raise ValueError('Invalid rollout batch/pool')
        if native and not deterministic:
            raise ValueError('Native reference is for zero-gradient evaluation only')
        if not deterministic and (not self.training or not self.policy.ac.device_global_matching):
            raise ValueError('Training requires the registered B0 policy contract')
        if deterministic:
            with self.evaluation_mode(decoder):
                return self._collect(cases, seeds, deterministic=True, retain=retain, decoder=decoder,
                                     native=native, heartbeat=heartbeat, record_times=record_times)
        self._set_trainability(True)
        self.policy.ac.tau = self.config['train_tau']
        return self._collect(cases, seeds, deterministic=False, retain=retain, decoder='AR_sample',
                             native=False, heartbeat=heartbeat, record_times=record_times)

    def _collect(self, cases, seeds, *, deterministic, retain, decoder, native, heartbeat, record_times):
        count, agents = len(cases), 104
        resets = []
        for i, (c, s) in enumerate(zip(cases, seeds)):
            config = self.environment_config(c['path'], 42 if deterministic else s)
            if deterministic:
                config['_stage3_native_reset_seed'] = int(s) * 50000 + i * 10000
            resets.append((i, 'reset', config))
        initial = self.pool.call(resets)
        obs, infos = [initial[i][0] for i in range(count)], [initial[i][2] for i in range(count)]
        hidden = np.zeros((count, agents, 1, 64), np.float32)
        previous = np.full((count, agents, 3), -1, np.int64)
        live = np.ones(count, bool)
        output = [dict(case_id=c['path'], profile=c['profile'], distribution=c['distribution'],
            seed=int(s), actions=[], states=[], times=[], history=HISTORY, decoder=decoder,
            tau=float(self.policy.ac.tau), policy_updates=self.policy_updates,
            behavior_deterministic=deterministic, forced_replay=False) for c, s in zip(cases, seeds)]
        history_hashes = [hashlib.sha256() for _ in cases]
        # Opt-in single-model training uses the checkpoint's global Torch RNG.
        # Environment reset seeds stay per visit. Native evaluation and the
        # historical batch-one sampler retain their original RNG dispatch.
        batched = not deterministic and getattr(self, 'batched_sampling', False)
        rng = []
        if not batched:
            for seed in seeds:
                seed_all(seed)
                rng.append(torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else torch.get_rng_state())
        started, forward_calls, environment_steps = time.monotonic(), 0, 0
        for step in range(self.config['rollout_max_steps']):
            indices = np.flatnonzero(live).tolist()
            if not indices:
                break
            # B0's native evaluator keeps completed slots in the tensor batch.
            # Dropping them changes cuDNN/GEMM reduction shapes and can change
            # close Hungarian rankings, despite identical weights and seeds.
            groups = ([list(range(count))] if deterministic else
                      [indices] if batched else [[i] for i in indices])
            actions_to_step = {}
            for group in groups:
                if not deterministic and not batched:
                    if self.device.type == 'cuda':
                        torch.cuda.set_rng_state(rng[group[0]], self.device)
                    else:
                        torch.set_rng_state(rng[group[0]])
                hist = authoritative_history(previous[group], [infos[i] for i in group])
                active = np.stack([infos[i]['active_agents'] for i in group]).reshape(len(group), agents)
                active[~live[group]] = 0
                roles = np.stack([infos[i]['agent_types'] for i in group]).reshape(len(group), agents)
                graph = [obs[i] for i in group]
                observer = getattr(self.policy.ac, 'stage3_decision_observer', None)
                if observer is not None:
                    observer.begin_batch([cases[i]['distribution'] for i in group])
                if native:
                    from onpolicy.utils.stage2_resource_rl import forward
                    out, _ = forward(self.policy, graph, hidden[group], active, hist, roles,
                                     deterministic=True, hungarian=decoder == 'H')
                    actions, logp, new_h, mask = (out[k] for k in ('actions', 'log_probs', 'rnn_states', 'decision_mask'))
                else:
                    _, actions, logp, new_h, mask = self.policy.get_actions(graph, hidden[group], active,
                        hist[..., 0], hist[..., 1], deterministic=deterministic, agent_types=roles,
                        return_decision_mask=True)
                forward_calls += 1
                if not deterministic and not batched:
                    rng[group[0]] = (torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda'
                                     else torch.get_rng_state())
                actions = actions.cpu().numpy().astype(np.int64)
                # One transfer per output tensor, including when 64 environments
                # share the forward. Do not synchronize separately for each row.
                logp_cpu = logp.cpu().numpy()
                mask_cpu = mask.cpu().numpy()
                hidden_cpu = new_h.cpu().numpy()
                if actions.shape[-1] == 2:
                    actions = np.concatenate((actions, np.full((*actions.shape[:-1], 1), -1, np.int64)), axis=-1)
                for j, i in enumerate(group):
                    if not live[i]:
                        hidden[i] = hidden_cpu[j]
                        previous[i] = actions[j]
                        actions_to_step[i] = actions[j]
                        continue
                    output[i]['actions'].append(actions[j].tolist())
                    now = float(infos[i]['env_total_time'])
                    if record_times:
                        output[i]['times'].append(now)
                    for part in (hist[j], active[j], mask_cpu[j]):
                        history_hashes[i].update(np.ascontiguousarray(part).tobytes())
                    if retain:
                        output[i]['states'].append(dict(graph=obs[i], hidden=hidden[i].copy(),
                            active=active[j].copy(), op=hist[j, :, 0].copy(), site=hist[j, :, 1].copy(),
                            roles=roles[j].copy(), action=actions[j].copy(),
                            old_logp=logp_cpu[j].reshape(agents),
                            mask=mask_cpu[j].reshape(agents), time=now))
                    hidden[i] = hidden_cpu[j]
                    previous[i] = actions[j]
                    actions_to_step[i] = actions[j]
            step_indices = list(range(count)) if deterministic else indices
            results = self.pool.call([(i, 'step', actions_to_step[i]) for i in step_indices])
            environment_steps += len(indices)
            for i in step_indices:
                obs[i], _, done, infos[i] = results[i]
                done = np.asarray(done).reshape(agents).astype(bool)
                hidden[i, done] = 0  # Native per-agent recurrent reset, including inactive terminal agents.
                if retain and live[i]:
                    output[i]['states'][-1]['done'] = done.copy()
                if done.all():
                    live[i] = False
            if heartbeat and step % 25 == 0:
                heartbeat.update(event='rollout', rollout_step=step + 1, live_trajectories=int(live.sum()),
                    forward_calls=forward_calls, environment_steps=environment_steps,
                    collection_seconds=time.monotonic()-started,
                    batched_sampling=batched, inference_batch_size=len(groups[0]))
        summaries = self.pool.call([(i, 'summary', None) for i in range(count)])
        for i, t in enumerate(output):
            t.update(summaries[i])
            t['steps'] = len(t['actions'])
            t['actions_sha256'] = digest_json(t['actions'])
            t['history_sha256'] = history_hashes[i].hexdigest()
            if not t['completed'] or t.get('cycle_terminated') or live[i]:
                raise PPOContractError(f'Incomplete trajectory: {t["case_id"]}: {summaries[i]}')
        return output

    def set_normalization(self, moments, *, expected_sha=None):
        n, total, squares = map(float, moments)
        if n <= 0 or not np.isfinite(moments).all():
            raise ValueError('Invalid training-only normalization moments')
        checksum = digest_json(list(map(float, moments)))
        if expected_sha is not None and checksum != expected_sha:
            raise ValueError('Normalization calibration changed')
        with torch.no_grad():
            self.norm.running_mean.fill_(total / n)
            self.norm.running_mean_sq.fill_(squares / n)
            self.norm.debiasing_term.fill_(1.)
        self.normalization_sha256 = checksum

    def _replay(self, group):
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=self.device)
        hidden = tensor(np.stack([t['states'][0]['hidden'] for t in group]))
        for step in range(max(len(t['states']) for t in group)):
            indices = [i for i, t in enumerate(group) if step < len(t['states'])]
            states = [group[i]['states'][step] for i in indices]
            stack = lambda key: np.stack([s[key] for s in states])
            graph = [s['graph'] for s in states]
            lp, _, mask, new_h = self.policy.evaluate_actions(graph, hidden[indices], stack('active'),
                stack('op'), stack('site'), stack('action'), agent_types=stack('roles'),
                return_decision_mask=True, return_rnn_states=True)
            hidden = hidden.index_copy(0, tensor(indices, torch.long),
                new_h * (~tensor(stack('done'), torch.bool))[:, :, None, None])
            yield step, indices, states, graph, lp.reshape(len(indices), -1), mask.reshape(len(indices), -1)
            if (step + 1) % self.config['tbptt'] == 0:
                hidden = hidden.detach()

    @torch.no_grad()
    def replay_metrics(self, trajectories, *, microbatch=None, heartbeat=None):
        self.policy.ac.eval()
        rows = {str(i): dict(decisions=0, kl_sum=0., max_logp_error=0., clip_count=0.) for i in range(3)}
        width = microbatch or self.config['microbatch']
        for start in range(0, len(trajectories), width):
            for step, _, states, _, lp, mask in self._replay(trajectories[start:start + width]):
                old = torch.as_tensor(np.stack([s['old_logp'] for s in states]), device=self.device)
                expected = torch.as_tensor(np.stack([s['mask'] for s in states]), device=self.device)
                if not torch.isfinite(lp).all() or not torch.equal(mask, expected):
                    raise PPOContractError('Nonfinite likelihood or legal decision-mask drift')
                roles = np.stack([s['roles'] for s in states])
                diff = lp - old
                for role, row in rows.items():
                    select = expected.bool() & torch.as_tensor(roles == int(role), device=self.device)
                    d = diff[select]
                    row['decisions'] += d.numel()
                    if d.numel():
                        row['kl_sum'] += float((d.clamp(-40, 40).exp() - 1 - d).sum())
                        row['max_logp_error'] = max(row['max_logp_error'], float(d.abs().max()))
                        row['clip_count'] += float(((d.exp() - 1).abs() > .2).sum())
                if heartbeat and step % 50 == 0:
                    heartbeat.update(event='likelihood_replay', replay_step=step, replay_microbatch=start // width)
        n = sum(r['decisions'] for r in rows.values())
        for r in rows.values():
            r['kl'] = r['kl_sum'] / max(r['decisions'], 1)
        return dict(kl=sum(r['kl_sum'] for r in rows.values()) / max(n, 1), roles=rows,
            decisions=n, max_logp_error=max(r['max_logp_error'] for r in rows.values()),
            clip_fraction=sum(r['clip_count'] for r in rows.values()) / max(n, 1))

    def _backward(self, trajectories, source_costs, *, microbatch, heartbeat=None, role=None):
        total_count = len(trajectories)
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        health = dict(nonfinite=0, mask_mismatch=0)
        scalars = dict(actor_loss=0., critic_loss=0.)
        deferred = DeferredScalars()
        for start in range(0, total_count, microbatch):
            group = trajectories[start:start + microbatch]
            actor_loss = critic_loss = None
            steps = max(len(t['states']) for t in group)
            for step, indices, states, graph, lp, mask in self._replay(group):
                expected = tensor(np.stack([s['mask'] for s in states]))
                deferred.add(health, 'nonfinite', (~torch.isfinite(lp)).sum())
                deferred.add(health, 'mask_mismatch', (mask != expected).sum())
                train_mask = expected
                if role is not None:
                    train_mask = train_mask * tensor(np.stack([s['roles'] == role for s in states]))
                advantage = tensor([.01 * (source_costs[group[i]['case_id']] - group[i]['makespan'])
                                    for i in indices])[:, None]
                diff = lp - tensor(np.stack([s['old_logp'] for s in states]))
                ratio = diff.clamp(-40, 40).exp()
                loss = -(torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
                         * train_mask).sum() / total_count
                deferred.add(scalars, 'actor_loss', loss)
                actor_loss = loss if actor_loss is None else actor_loss + loss
                if role is None:
                    prepared = self.cache.prepare_graph(graph)
                    encoded = self.policy.ac._encode_graph(prepared, actor_grad=False)
                    values = self.policy.ac.team_critic(encoded['global_emb'].detach()).reshape(-1)
                    remaining = tensor([-.01 * (group[i]['makespan'] - s['time'])
                                        for i, s in zip(indices, states)])
                    target = self.norm.normalize(remaining[:, None]).reshape(-1)
                    weights = tensor([1. / (total_count * len(group[i]['states'])) for i in indices])
                    c_loss = ((values - target.detach()).square() * weights).sum()
                    deferred.add(health, 'nonfinite', (~torch.isfinite(values)).sum())
                    deferred.add(scalars, 'critic_loss', c_loss)
                    critic_loss = c_loss if critic_loss is None else critic_loss + c_loss
                if (step + 1) % self.config['tbptt'] == 0 or step + 1 == steps:
                    deferred.flush()
                    if any(health.values()):
                        raise PPOContractError(f'Probability/value replay failure: {health}')
                    if actor_loss.requires_grad:
                        actor_loss.backward()
                    if critic_loss is not None:
                        critic_loss.backward()
                    actor_loss = critic_loss = None
                if heartbeat and step % 50 == 0:
                    heartbeat.update(event='update_replay', update_step=step, microbatch_start=start)
        return scalars

    def update(self, trajectories, source_costs, *, microbatch=None, heartbeat=None, epochs=None):
        if not self.training or not self.normalization_sha256:
            raise ValueError('Training requires initialized, frozen Train600 ValueNorm')
        if not trajectories or any(not t['completed'] or not t['states'] or t['behavior_deterministic']
                or t['forced_replay'] or t['policy_updates'] != self.policy_updates or t['history'] != HISTORY
                or t['decoder'] != 'AR_sample' or t['tau'] != .03 for t in trajectories):
            raise PPOContractError('Only complete fresh stochastic B0-contract visits can train')
        if any(t['case_id'] not in source_costs or not np.isfinite(source_costs[t['case_id']])
               or source_costs[t['case_id']] <= 0 for t in trajectories):
            raise ValueError('Source costs must be finite positive costs for every visit')
        width, passes = microbatch or self.config['microbatch'], epochs or self.config['ppo_epochs']
        self.policy.ac.tau = .03
        before = self.replay_metrics(trajectories, microbatch=width, heartbeat=heartbeat)
        if before['max_logp_error'] > .002 or not before['decisions']:
            raise PPOContractError(f'Behavior likelihood cannot be reproduced: {before}')
        metrics = []
        with self.cache.group():
            for epoch in range(passes):
                if before['kl'] > self.config['soft_kl']:
                    raise PPOContractError('Soft KL prevented the next complete PPO epoch; partial batch retained')
                gradient_mode(self.policy.ac)
                self.policy.actor_optimizer.zero_grad(set_to_none=True)
                self.policy.critic_optimizer.zero_grad(set_to_none=True)
                row = self._backward(trajectories, source_costs, microbatch=width, heartbeat=heartbeat)
                row['gradient_norm_by_group'] = {}
                actor_parameters = []
                for group in self.policy.actor_optimizer.param_groups:
                    ps = [p for p in group['params'] if p.grad is not None]
                    actor_parameters.extend(ps)
                    row['gradient_norm_by_group'][group['name']] = float(torch.stack(
                        [p.grad.square().sum() for p in ps]).sum().sqrt()) if ps else 0.
                row['actor_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(actor_parameters, 1., error_if_nonfinite=True))
                row['critic_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(
                    self.policy.ac.team_critic.parameters(), 1., error_if_nonfinite=True))
                self.policy.actor_optimizer.step()
                self.policy.critic_optimizer.step()
                self.policy_updates += 1
                # Discard cached adjacent actor encodings before any new pass.
                self.cache.last_actor = None
                before = self.replay_metrics(trajectories, microbatch=width, heartbeat=heartbeat)
                row.update(actor_step_applied=True, policy_updates=self.policy_updates, post_update=before)
                metrics.append(row)
                if max([before['kl'], *[v['kl'] for v in before['roles'].values()]]) > self.config['hard_kl']:
                    raise PPOContractError('Hard KL exceeded; partial batch is not a resumable commit')
        self.assert_frozen()
        advantages = [.01 * (source_costs[t['case_id']] - t['makespan']) for t in trajectories]
        return dict(epochs=metrics, global_visits=len(trajectories), microbatch=width,
            positive_advantage_fraction=float(np.mean(np.asarray(advantages) > 0)),
            negative_advantage_fraction=float(np.mean(np.asarray(advantages) < 0)),
            trajectory_advantages=advantages, execution_cache=self.cache.last_report)

    def gradient_diagnostic(self, trajectories, costs, heartbeat=None):
        vectors = []
        with self.cache.group():
            for role in range(3):
                self.policy.actor_optimizer.zero_grad(set_to_none=True)
                gradient_mode(self.policy.ac)
                self._backward(trajectories, costs, microbatch=self.config['microbatch'], heartbeat=heartbeat, role=role)
                vectors.append(torch.cat([p.grad.detach().cpu().flatten() if p.grad is not None
                    else torch.zeros(p.numel()) for p in self.policy.ac.encoder.parameters()]))
        self.policy.actor_optimizer.zero_grad(set_to_none=True)
        norms = [float(v.norm()) for v in vectors]
        return {'optimizer_steps': 0, 'shared_role_gradient_norms': norms,
                'shared_role_cosines': {f'{i}:{j}': float(torch.dot(vectors[i], vectors[j]) /
                    max(norms[i] * norms[j], 1e-30)) for i in range(3) for j in range(i + 1, 3)}}

    def save(self, path, *, manifest_sha256, next_batch, diagnostic=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        payload = dict(protocol=PROTOCOL, source_sha256=SOURCE_SHA, manifest_sha256=manifest_sha256,
            model=self.policy.ac.state_dict(), actor_optim=self.policy.actor_optimizer.state_dict(),
            critic_optim=self.policy.critic_optimizer.state_dict(), value_normalizer=self.norm.state_dict(),
            normalization_sha256=self.normalization_sha256, policy_updates=self.policy_updates,
            next_batch=next_batch, diagnostic_only=diagnostic, history=HISTORY,
            rng_torch=torch.get_rng_state(), rng_numpy=np.random.get_state(), rng_python=random.getstate(),
            rng_cuda=torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None,
            recipe_sha256=digest_json(self.config))
        temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
        torch.save(payload, temp)
        with temp.open('rb') as f:
            os.fsync(f.fileno())
        os.link(temp, path)
        temp.unlink()
        return digest_file(path)

    def load_weights(self, path, *, manifest_sha256=None):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if digest_file(path) != SOURCE_SHA and (payload.get('protocol') != PROTOCOL
                or payload.get('source_sha256') != SOURCE_SHA or payload.get('diagnostic_only')
                or (manifest_sha256 is not None and payload.get('manifest_sha256') != manifest_sha256)):
            raise ValueError('Evaluation checkpoint does not belong to the B0 study')
        self.policy.ac.load_state_dict(payload['model'], strict=True)
        self.policy_updates = int(payload.get('policy_updates', 0))
        self.assert_frozen()

    def resume(self, path, *, manifest_sha256, allow_diagnostic=False):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        required = ('actor_optim', 'critic_optim', 'value_normalizer', 'normalization_sha256',
                    'rng_torch', 'rng_cuda', 'rng_numpy', 'rng_python', 'next_batch')
        if (any(k not in payload for k in required) or payload.get('protocol') != PROTOCOL
                or payload.get('source_sha256') != SOURCE_SHA or payload.get('history') != HISTORY
                or payload.get('manifest_sha256') != manifest_sha256
                or payload.get('recipe_sha256') != digest_json(self.config)
                or (payload.get('diagnostic_only') and not allow_diagnostic)):
            raise ValueError('Resume identity/state mismatch')
        self.policy.ac.load_state_dict(payload['model'], strict=True)
        self.policy.actor_optimizer.load_state_dict(payload['actor_optim'])
        self.policy.critic_optimizer.load_state_dict(payload['critic_optim'])
        self.norm.load_state_dict(payload['value_normalizer'], strict=True)
        self.normalization_sha256 = payload['normalization_sha256']
        self.policy_updates = payload['policy_updates']
        random.setstate(payload['rng_python'])
        np.random.set_state(payload['rng_numpy'])
        torch.set_rng_state(payload['rng_torch'])
        if self.device.type == 'cuda':
            if payload['rng_cuda'] is None:
                raise ValueError('CUDA resume lacks device RNG')
            torch.cuda.set_rng_state(payload['rng_cuda'], self.device)
        self.policy.ac.tau = .03
        self.assert_frozen()
        return payload['next_batch']

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None
