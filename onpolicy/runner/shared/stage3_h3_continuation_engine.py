"""Full Adam-state forks and passive diagnostics for the H3 temperature study."""
from __future__ import annotations

import math
import random

import numpy as np
import torch

from onpolicy.runner.shared.stage3_h3_frozen_engine import H3FrozenEngine, model_digest
from onpolicy.utils.stage3_h3_continuation import validate_parent_payload, validate_recipe
from onpolicy.utils.stage3_research import digest_file, digest_json
from onpolicy.utils.stage3_performance import DeferredScalars


class DecisionObserver:
    """GPU reductions only; no sampling, mutation of logits, or per-head CPU sync."""
    DISTRIBUTIONS = ('iid', 'ood_stress', 'ood_scale')

    def __init__(self, device):
        self.device = device
        self.totals = torch.zeros((3, 3, 9), dtype=torch.float64, device=device)
        self.labels = None

    def begin_batch(self, distributions):
        self.labels = torch.tensor([self.DISTRIBUTIONS.index(x) for x in distributions],
                                   device=self.device, dtype=torch.long)

    @torch.no_grad()
    def record(self, role, batch_indices, log_probs, selected):
        legal = torch.isfinite(log_probs).sum(-1)
        meaningful = legal > 1
        p = log_probs.detach().softmax(-1)
        logp = p.clamp_min(torch.finfo(p.dtype).tiny).log()
        entropy = -(p * logp).sum(-1)
        normalized = entropy / legal.clamp_min(2).to(p.dtype).log()
        values = torch.stack((torch.ones_like(legal), meaningful, legal*meaningful,
            p.max(-1).values*meaningful, entropy*meaningful, normalized*meaningful,
            (selected != log_probs.argmax(-1))*meaningful,
            ((selected == 0)*meaningful) if role else torch.zeros_like(meaningful),
            (legal == 1)), -1).double()
        labels = self.labels.index_select(0, batch_indices.long())
        # Small fixed categories avoid floating-point atomic accumulation.
        for index in range(3):
            self.totals[role, index] += values[labels == index].sum(0)

    def result(self):
        data = self.totals.cpu().tolist()
        result = {}
        for role, groups in enumerate(data):
            result[str(role)] = {}
            for distribution, v in zip(self.DISTRIBUTIONS, groups):
                n = max(1., v[1])
                result[str(role)][distribution] = dict(head_decisions=int(v[0]),
                    multi_candidate_decisions=int(v[1]), single_candidate_decisions=int(v[8]),
                    legal_candidates_mean=v[2]/n, pmax_mean=v[3]/n,
                    entropy_mean=v[4]/n, normalized_entropy_mean=v[5]/n,
                    non_greedy_fraction=v[6]/n, resource_noop_fraction=v[7]/n if role else None)
        return result


class UpdateObserver:
    def before_step(self, optimizer):
        self.before = [[p.detach().clone() for p in group['params']] for group in optimizer.param_groups]

    @torch.no_grad()
    def after_step(self, optimizer):
        rows = []
        for index, (group, before) in enumerate(zip(optimizer.param_groups, self.before)):
            params = group['params']
            count = sum(p.numel() for p in params)
            if not count:
                continue
            delta2 = torch.stack([(p.detach()-b).double().square().sum() for p, b in zip(params, before)]).sum()
            weight2 = torch.stack([b.double().square().sum() for b in before]).sum()
            values = torch.stack((delta2, weight2)).cpu().tolist()
            rows.append(dict(group=index, parameters=count, lr=group['lr'],
                             update_rms=math.sqrt(values[0]/count),
                             relative_update_l2=math.sqrt(values[0]/max(values[1], 1e-30))))
        del self.before
        return rows


class H3ContinuationEngine(H3FrozenEngine):
    def __init__(self, *args, config, **kwargs):
        validate_recipe(config)
        super().__init__(*args, config=config, **kwargs)
        self.update_observer = UpdateObserver()
        self.decision_stats = {}
        self.value_diagnostics = None

    def fork_parent(self, path, *, expected_sha256, parent_manifest, parent_epoch):
        if digest_file(path) != expected_sha256:
            raise ValueError('Full-state parent bytes changed')
        p = torch.load(path, map_location='cpu', weights_only=False)
        validate_parent_payload(p, parent_manifest, parent_epoch)
        self.policy.ac.load_state_dict(p['model'], strict=True)
        self.policy.actor_optimizer.load_state_dict(p['actor_optim'])
        self.policy.critic_optimizer.load_state_dict(p['critic_optim'])
        self.norm.load_state_dict(p['value_normalizer'], strict=True)
        self.normalization_sha256 = p['normalization_sha256']
        self.initialization_sha256 = expected_sha256
        self.policy_updates = p['policy_updates']
        self.physical_microbatch = self.config['microbatch']
        random.setstate(p['rng_python']); np.random.set_state(p['rng_numpy'])
        torch.set_rng_state(p['rng_torch'])
        if self.device.type == 'cuda':
            if p.get('rng_cuda') is None:
                raise ValueError('CUDA continuation needs the parent CUDA RNG')
            torch.cuda.set_rng_state(p['rng_cuda'], self.device)
        self.policy.ac.tau = self.config['train_tau']
        self.cache.last_actor = None
        self.commit_ready = True
        self.assert_frozen()
        return dict(parent_sha256=expected_sha256, inherited_updates=self.policy_updates,
                    parent_physical_microbatch=p['physical_microbatch'],
                    physical_microbatch=self.physical_microbatch,
                    model_sha256=model_digest(self.policy.ac),
                    actor_optimizer_entries=len(self.policy.actor_optimizer.state),
                    critic_optimizer_entries=len(self.policy.critic_optimizer.state),
                    normalization_sha256=self.normalization_sha256,
                    train_tau=self.config['train_tau'])

    def collect_logical(self, *args, **kwargs):
        observer = DecisionObserver(self.device) if self.config.get('decision_observer', True) else None
        if observer is not None:
            self.policy.ac.stage3_decision_observer = observer
        try:
            rows = super().collect_logical(*args, **kwargs)
        finally:
            if observer is not None:
                del self.policy.ac.stage3_decision_observer
        self.decision_stats = observer.result() if observer is not None else {}
        for row in rows:
            row['behavior_tau'] = row['tau']
            row['advantage_mode'] = self.config['advantage']
        if self.config['advantage'] == 'state_value':
            self.capture_old_values(rows, kwargs.get('heartbeat'))
        if self.config.get('trim_host_heap'):
            # Return free malloc arenas after CPU environment pools exit. No
            # retained trajectory, tensor dtype or value is changed.
            import ctypes
            import gc
            gc.collect()
            trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
            if trim is not None:
                trim(0)
        return rows

    @torch.no_grad()
    def replay_metrics(self, trajectories, *, microbatch=None, heartbeat=None):
        if not self.config.get('deferred_replay_metrics'):
            return super().replay_metrics(trajectories, microbatch=microbatch, heartbeat=heartbeat)
        from onpolicy.runner.shared.stage3_b_shared_b0_engine import PPOContractError
        self.policy.ac.eval()
        rows = {str(i): dict(decisions=0, kl_sum=0., max_logp_error=0., clip_count=0.) for i in range(3)}
        deferred = DeferredScalars()
        health = dict(nonfinite=0, mask_mismatch=0)
        width = microbatch or self.config['microbatch']
        def flush():
            deferred.flush()
            if any(health.values()):
                raise PPOContractError('Nonfinite likelihood or legal decision-mask drift')
        for start in range(0, len(trajectories), width):
            for step, _, states, _, lp, mask in self._replay(trajectories[start:start+width]):
                old = torch.as_tensor(np.stack([s['old_logp'] for s in states]), device=self.device)
                expected = torch.as_tensor(np.stack([s['mask'] for s in states]), device=self.device)
                if mask.shape != expected.shape:
                    raise PPOContractError('Decision-mask shape drift')
                deferred.add(health, 'nonfinite', (~torch.isfinite(lp)).sum())
                deferred.add(health, 'mask_mismatch', (mask != expected).sum())
                roles = np.stack([s['roles'] for s in states])
                diff = lp-old
                for role, row in rows.items():
                    select = expected.bool() & torch.as_tensor(roles == int(role), device=self.device)
                    d = diff[select]
                    row['decisions'] += d.numel()
                    if d.numel():
                        deferred.add(row, 'kl_sum', (d.clamp(-40,40).exp()-1-d).sum())
                        deferred.add(row, 'max_logp_error', d.abs().max(), maximum=True)
                        deferred.add(row, 'clip_count', ((d.exp()-1).abs()>.2).sum())
                if (step+1) % self.config['tbptt'] == 0:
                    flush()
                if heartbeat and step % 50 == 0:
                    heartbeat.update(event='likelihood_replay', replay_step=step, replay_microbatch=start//width)
            flush()
        n = sum(r['decisions'] for r in rows.values())
        for row in rows.values():
            row['kl'] = row['kl_sum']/max(row['decisions'],1)
        return dict(kl=sum(r['kl_sum'] for r in rows.values())/max(n,1), roles=rows, decisions=n,
                    max_logp_error=max(r['max_logp_error'] for r in rows.values()),
                    clip_fraction=sum(r['clip_count'] for r in rows.values())/max(n,1))

    @torch.no_grad()
    def capture_old_values(self, trajectories, heartbeat=None):
        """Compute all baselines before any optimizer/ValueNorm update; never fit on this batch."""
        targets, predictions = [], []
        mean, variance = self.norm.running_mean_var()
        mean, scale = mean.reshape(()), variance.sqrt().reshape(())
        for start in range(0, len(trajectories), self.physical_microbatch):
            chunk = trajectories[start:start+self.physical_microbatch]
            for step in range(max(len(t['states']) for t in chunk)):
                ids = [i for i, t in enumerate(chunk) if step < len(t['states'])]
                states = [chunk[i]['states'][step] for i in ids]
                graph = self.cache.prepare_graph([s['graph'] for s in states])
                encoded = self.policy.ac._encode_graph(graph, actor_grad=False)
                values = self.policy.ac.team_critic(encoded['global_emb']).reshape(-1)*scale+mean
                for i, state, value in zip(ids, states, values.cpu().tolist()):
                    target = -.01*(chunk[i]['makespan']-state['time'])
                    if not math.isfinite(value) or target > 1e-5:
                        raise ValueError('Invalid remaining-time value or inconsistent time origin')
                    state['old_value'] = value
                    state['actor_advantage'] = target-value
                    targets.append(target); predictions.append(value)
                if heartbeat and step % 100 == 0:
                    heartbeat.update(event='frozen_old_values', chunk_start=start, step=step)
        y, v = np.asarray(targets), np.asarray(predictions)
        ev = float(1-np.var(y-v)/max(np.var(y), 1e-12))
        self.value_diagnostics = dict(states=len(y), explained_variance=ev,
            rmse=float(np.sqrt(np.mean((y-v)**2))), baseline_frozen_before_update=True,
            mean_advantage=float(np.mean(y-v)))
        if ev <= 0:
            raise ValueError('Old critic fails the pre-registered positive explained-variance gate')

    def actor_advantage(self, chunk, ids, states, source_costs):
        if self.config['advantage'] == 'source_relative':
            return super().actor_advantage(chunk, ids, states, source_costs)
        return torch.as_tensor([s['actor_advantage'] for s in states],
                               dtype=torch.float32, device=self.device).detach()

    def update_logical(self, trajectories, *args, **kwargs):
        if any(t.get('behavior_tau') != self.config['train_tau']
               or t.get('advantage_mode') != self.config['advantage'] for t in trajectories):
            raise ValueError('Continuation behavior temperature/advantage identity changed')
        result = super().update_logical(trajectories, *args, **kwargs)
        result.update(train_tau=self.config['train_tau'], advantage_mode=self.config['advantage'],
                      value_diagnostics=self.value_diagnostics)
        for row in result['minibatches']:
            if row.get('actor_step_applied'):
                row['gradient_clip_factor'] = min(1., self.config['gradient_clip'] /
                                                  (row['actor_gradient_norm']+1e-6))
        return result
