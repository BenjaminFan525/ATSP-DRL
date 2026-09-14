"""Opt-in adapter; existing Stage2 entry points and running jobs stay untouched.

The dedicated train/diagnostic entry points install this adapter before any
environment is created. The legacy runner's delayed CostIteration import then
selects this subclass. No source file used by the old running job is edited.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.utils import stage2_cost_improvement as legacy
from onpolicy.utils.stage2_cost_execution import (
    EXECUTION_CONTRACT, ExactCostCache, ShadowFanout, array_digest, json_digest, run_fanout,
)
from onpolicy.utils.stage2_cost_timeout import LEGACY_TIMEOUT_CONTRACT, TIMEOUT_CONTRACT

LegacyCostIteration = legacy.CostIteration
_CONFIG = None


def runtime_identity():
    """Cache namespace records actual arithmetic, not just a user-written label."""
    import torch_geometric
    gpu = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=uuid,driver_version',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(',')
    if len(gpu) != 2:
        raise ValueError('Cannot identify physical GPU0 and its driver for the exact cache.')
    return dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
        numpy=np.__version__, torch_geometric=torch_geometric.__version__,
        gpu_uuid=gpu[0].strip(), driver_version=gpu[1].strip(),
        cudnn=torch.backends.cudnn.version(), deterministic=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        float32_precision=torch.get_float32_matmul_precision(),
        cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_name=torch.cuda.get_device_name(0), gpu_capability=list(torch.cuda.get_device_capability(0)))


class FanoutAircraftScheduleEnv(AircraftScheduleEnv):
    def stage2_cost_load_regression_snapshot(self, path, file_sha256, raw_sha256):
        """Read a source-bound archived regression state into the shadow only."""
        import gzip
        compressed = Path(path).read_bytes()
        if hashlib.sha256(compressed).hexdigest() != file_sha256:
            raise ValueError('Archived regression snapshot file changed.')
        raw = gzip.decompress(compressed)
        if hashlib.sha256(raw).hexdigest() != raw_sha256:
            raise ValueError('Archived regression state digest differs.')
        self.stage2_cost_branch_clear()
        self._stage2_cost_snapshot = raw
        return raw_sha256

    def stage2_cost_fanout_restore(self, ids):
        snapshot = getattr(self, '_stage2_cost_snapshot', None)
        if snapshot is None:
            raise RuntimeError('Fanout requires a captured state.')
        self._stage2_cost_fanout = ShadowFanout(snapshot, ids)
        return True

    def stage2_cost_fanout_step(self, requests):
        if not hasattr(self, '_stage2_cost_fanout'):
            raise RuntimeError('Fanout state was not restored.')
        return self._stage2_cost_fanout.step(requests)

    def stage2_cost_branch_clear(self):
        self.__dict__.pop('_stage2_cost_fanout', None)
        return super().stage2_cost_branch_clear()


class AcceleratedCostIteration(LegacyCostIteration):
    def report_execution_progress(self, metrics):
        from onpolicy.scripts.train.run_stage2_resource_manifest_trial import atomic_json
        import time
        atomic_json(Path(self.runner.log_dir) / 'cost_execution_progress.json',
                    dict(epoch=self.epoch, rollout=self.rollout, event_step=self.step,
                         updated_unix_time=time.time(), metrics=metrics))

    def __init__(self, runner, epoch):
        super().__init__(runner, epoch)
        if _CONFIG is None:
            raise RuntimeError('Use the dedicated, source-bound accelerated entry point.')
        self.execution_config = dict(_CONFIG)
        self.execution_counts = dict(batches=0, logical_queries=0, cache_hits=0,
            physical_jobs=0, coalesced_queries=0, vector_steps=0,
            requested_environment_steps=0, executed_environment_steps=0, environment_forks=0,
            environment_rpc_seconds=0., actor_seconds=0., wall_seconds=0.)
        if self.execution_config.get('timeout_contract', LEGACY_TIMEOUT_CONTRACT) == TIMEOUT_CONTRACT:
            self.execution_counts.update(actor_pool_wall_seconds=0., charged_work_seconds=0.,
                                         peak_live_jobs=0, legacy_timeout_avoided_queries=0)
        self.cache = None
        self.actor_pool = None
        if self.enabled:
            runtime = runtime_identity()
            runtime['actor_lanes'] = self.execution_config.get('actor_lanes', 1)
            runtime['fanout_width'] = self.execution_config['fanout_width']
            runtime['inference_mode'] = self.execution_config.get('inference_mode', 'legacy')
            runtime['timeout_contract'] = self.execution_config.get('timeout_contract', LEGACY_TIMEOUT_CONTRACT)
            if runtime['visible_devices'] != '0' or not runtime['deterministic']:
                raise ValueError('Accelerated costs require physical GPU0 deterministic execution.')
            self.arithmetic = runtime
            namespace = json_digest(dict(source=self.execution_config['source_namespace'], runtime=runtime))
            self.cache = ExactCostCache(self.execution_config['cache_dir'], namespace)
            lanes = self.execution_config.get('actor_lanes', 1)
            fast = self.execution_config.get('inference_mode', 'legacy') == 'host_matching'
            if lanes > 1 or fast:
                from onpolicy.utils.stage2_cost_actor_pool import PrivateActorPool
                self.actor_pool = PrivateActorPool(self.policy, lanes, fast_matching=fast)

    def evidence(self, obs, active, history, infos, student, teacher):
        if not self.enabled or not self.selected:
            return []
        prepared = []
        for env in self.selected:
            case = str(np.asarray(infos['case_id']).reshape(-1)[env])
            if self.case_counts.get(case, 0) >= 2:
                raise ValueError('More than two cost states per case/iteration.')
            self.case_counts[case] = self.case_counts.get(case, 0) + 1
            rows = np.flatnonzero((active[env, :, 0] > 0)
                & (np.arange(active.shape[1]) >= self.runner.policy.ac.max_plane_agents))
            legal = obs[env].request_mask_matrix.detach().cpu().numpy().astype(bool)[rows]
            lookahead = obs[env].request_is_lookahead.detach().cpu().numpy().astype(bool).reshape(-1)
            candidates = legacy.joint_candidates(legal, student[env, rows, 0], teacher[env, rows, 0], lookahead)
            if len(candidates) >= 2:
                prepared.append((env, case, rows, lookahead, candidates))
        if not prepared:
            return []
        planes = self.runner.policy.ac.max_plane_agents
        if not np.array_equal(student[:, :planes], self.frozen_actions[:, :planes]):
            raise ValueError('Current learner and target disagree on the protected plane action.')
        directory = Path(self.runner.log_dir) / f'cost_states_epoch{self.epoch}' / f'rollout{self.rollout:03}_step{self.step:06}'
        directory.mkdir(parents=True, exist_ok=False)
        snapshots = self.runner.envs.call_each('stage2_cost_branch_capture',
            [(str(directory / f'env{env:02}.pkl.gz'),) for env in range(len(obs))])
        state_path = directory / 'policy_history.pt.gz'
        try:
            with state_path.open('xb') as raw:
                with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as handle:
                    torch.save({'obs': list(obs), 'active': active, 'history': history,
                        'agent_types': infos.get('agent_types'), 'target_rnn': self.rnn,
                        'target_post_forward_rnn': self.next_rnn, 'learner_rnn': self.learner_rnn,
                        'raw_prefix': self.prefix, 'student': student, 'teacher': teacher,
                        'frozen_actions': self.frozen_actions,
                        'learner_model': {k: v.detach().cpu() for k, v in self.runner.policy.ac.state_dict().items()},
                        'target_checkpoint': str(self.target_path), 'target_file_sha256': self.target_file_sha256}, handle)
            state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
            self.counts['snapshot_bytes'] += sum(x['bytes'] for x in snapshots) + state_path.stat().st_size
            if self.counts['snapshot_bytes'] > 16 * 1024**3:
                raise ValueError('Declared 16-GiB evidence budget exceeded; outputs preserved.')
            context = dict(policy_sha256=self.model_sha256,
                batch_snapshot_sha256=[x['sha256'] for x in snapshots],
                post_forward_recurrent_sha256=array_digest(self.next_rnn),
                history_sha256=array_digest(history), batch_width=len(obs),
                max_steps=self.runner.episode_length, arithmetic=self.arithmetic)
            queries = []
            for env, _, rows, _, candidates in prepared:
                for candidate in candidates:
                    first = self.frozen_actions.copy()
                    first[env] = student[env]
                    first[env, rows, 0], first[env, rows, 1] = candidate['action'], 0
                    queries.append((first, env))
            outcomes, execution = run_fanout(self, queries, width=self.execution_config['fanout_width'],
                                             context=context, cache=self.cache)
            for key, value in execution.items():
                if key == 'peak_live_jobs':
                    self.execution_counts[key] = max(self.execution_counts[key], value)
                else:
                    self.execution_counts[key] += value
            self.execution_counts['batches'] += 1
            self.counts['candidate_queries'] += len(queries)
            self.counts['continuation_vector_steps'] += execution['vector_steps']
            # Actual non-overlapping executor time, not a sum of branch clocks.
            self.counts['continuation_wall_seconds'] += execution['wall_seconds']
            self.counts['completed'] += sum(x['completed'] for x in outcomes)
            self.counts['incomplete'] += sum(not x['completed'] for x in outcomes)
            evidence, cursor = [], 0
            for env, case, rows, lookahead, candidates in prepared:
                first_digests = [array_digest(actions) for actions, _ in queries[cursor:cursor + len(candidates)]]
                group = outcomes[cursor:cursor + len(candidates)]
                cursor += len(candidates)
                row = dict(contract=legacy.COST_CONTRACT, loss_contract=legacy.LOSS_CONTRACT,
                    epoch=self.epoch, step=self.step, case=case, env=env, rows=rows.tolist(),
                    policy_sha256=self.model_sha256, batch_width=len(obs),
                    batch_snapshot_sha256=[x['sha256'] for x in snapshots], environment_snapshots=snapshots,
                    state_path=str(state_path), state_sha256=state_sha,
                    target_checkpoint=str(self.target_path), target_file_sha256=self.target_file_sha256,
                    recurrent_sha256=hashlib.sha256(self.rnn.tobytes()).hexdigest(),
                    post_forward_recurrent_sha256=hashlib.sha256(self.next_rnn.tobytes()).hexdigest(),
                    history_sha256=hashlib.sha256(history.tobytes()).hexdigest(),
                    active_sha256=hashlib.sha256(active.tobytes()).hexdigest(),
                    lookahead=lookahead.tolist(), candidates=[x['action'].tolist() for x in candidates],
                    origins=[x['origin'] for x in candidates], outcomes=group,
                    costs=[x['makespan'] for x in group], execution_contract=EXECUTION_CONTRACT,
                    cache_context=context, first_actions_sha256=first_digests,
                    cache_source_namespace=self.cache.namespace)
                with self.path.open('a') as handle:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                evidence.append(row)
                self.counts['selected_states'] += 1
            if (self.execution_config.get('timeout_contract') == TIMEOUT_CONTRACT
                    and any(not outcome['completed'] for outcome in outcomes)):
                raise RuntimeError('Incomplete cost continuation under v2; all evidence preserved, '
                                   'stop before additional collection/training, no retry.')
            return evidence
        finally:
            self.runner.envs.call('stage2_cost_branch_clear')

    def finish_epoch(self):
        try:
            super().finish_epoch()
        finally:
            self.close()
        path = Path(self.runner.log_dir) / f'cost_execution_epoch{self.epoch}.json'
        with path.open('x') as handle:
            json.dump(dict(contract=EXECUTION_CONTRACT, epoch=self.epoch,
                           enabled=self.enabled, counts=self.execution_counts), handle, sort_keys=True)

    def close(self):
        if self.actor_pool is not None:
            self.actor_pool.close()


def install(config):
    """Explicit process-local adapter. Must precede environment/runner creation."""
    global _CONFIG
    if _CONFIG is not None and _CONFIG != config:
        raise ValueError('Cannot change the executor configuration inside a process.')
    if (config['contract'] != EXECUTION_CONTRACT or not 1 <= config['fanout_width'] <= 16
            or not 1 <= config.get('actor_lanes', 1) <= 4):
        raise ValueError('Invalid accelerated execution configuration.')
    if not Path(config['cache_dir']).is_absolute():
        raise ValueError('Cache path must be absolute and owned by the experiment.')
    if config.get('inference_mode', 'legacy') not in ('legacy', 'host_matching'):
        raise ValueError('Unverified frozen actor inference mode.')
    if config.get('timeout_contract', LEGACY_TIMEOUT_CONTRACT) not in (LEGACY_TIMEOUT_CONTRACT, TIMEOUT_CONTRACT):
        raise ValueError('Unknown cost timeout contract.')
    _CONFIG = dict(config)
    from onpolicy.scripts.train import train_hkbz
    train_hkbz.AircraftScheduleEnv = FanoutAircraftScheduleEnv
    legacy.CostIteration = AcceleratedCostIteration


def load_config(path, expected_sha):
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Executor configuration changed after preparation.')
    config = json.loads(path.read_text())
    root = Path(__file__).resolve().parents[2]
    for relative, expected in config['code_fingerprint'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Executor source changed: {relative}')
    if json_digest(config['code_fingerprint']) != config['source_namespace']:
        raise ValueError('Executor namespace does not match its source fingerprint.')
    install(config)
    return config
