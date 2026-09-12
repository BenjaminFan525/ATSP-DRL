#!/usr/bin/env python3
"""Run frozen B0 H/F evaluations only. No training/search/retry entry point."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
from onpolicy.scripts.train.run_stage2_resource_manifest_trial import atomic_json
from onpolicy.scripts.train.run_stage2_resource_rl import arguments, cpus, metrics, seed
from onpolicy.scripts.train.train_hkbz import make_train_env
from onpolicy.utils.stage2_bc_contract import configure_bc_determinism, ready_head_config
from onpolicy.utils.stage2_bc_hf import (
    ALLOWED_CPUS, ANCHOR, CAPACITY, GPU_UUID, PROTOCOL, SLOTS, binding,
    check_replay, configure_args, current_cases, planning_for, report_suite,
    sha, validate_manifest,
)
from onpolicy.utils.stage2_bc_replay import validate_case_content
from onpolicy.utils.stage2_policy_transfer import verified_frozen_ready_source
from onpolicy.utils.stage2_resource_rl import ResourceLearner, configure_trainability, forward, summary
from onpolicy.utils.stage2_resource_rl_handoff import validate_frozen_lineage
from onpolicy.utils.training_stage import validate_stage2_joint_finetune_checkpoint


class Progress:
    def __init__(self, path, **values):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        now = time.time()
        self.state = dict(protocol=PROTOCOL, status='running', pid=os.getpid(),
            started_unix=now, last_work_progress_unix=now, actor_updates=0,
            critic_updates=0, teacher_queries=0, search_calls=0,
            automatic_training=False, automatic_retry=False,
            confirmed_scientific_success=False, **values)
        self.write()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        self.last_print = 0.

    def heartbeat(self):
        while not self.stop.wait(30):
            self.write()

    def write(self):
        with self.lock:
            now = time.time()
            self.state.update(updated_unix=now, elapsed_seconds=now - self.state['started_unix'],
                seconds_since_work_progress=now - self.state['last_work_progress_unix'],
                stall_suspected=now - self.state['last_work_progress_unix'] > 90,
                stall_action='advisory_only', hard_deadline_unix=None,
                wall_clock_limit_enabled=False)
            atomic_json(self.path, self.state)

    def __call__(self, event, force=False, **values):
        with self.lock:
            self.state.update(event=event, last_work_progress_unix=time.time(), **values)
            if force or time.time() - self.last_print >= 30:
                self.last_print = time.time()
                self.write()
                print('[BC-HF] ' + json.dumps(self.state, ensure_ascii=False), flush=True)

    def finish(self, status, **values):
        self('finished', force=True, status=status, **values)
        self.stop.set()
        self.thread.join(timeout=1)


def verify_resources():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0'
            or not set(os.sched_getaffinity(0)).issubset(cpus(ALLOWED_CPUS))):
        raise ValueError('Physical GPU0 and original CPU half are mandatory.')


def verify_files(m):
    validate_manifest(m)
    for name, digest in m['code_fingerprint'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Pinned implementation changed: {name}')
    for key in ('source', 'stage1_source', 'parent_manifest', 'source_archive', 'engineering_proof'):
        item = m[key]
        if sha(item['path']) != item['sha256']:
            raise ValueError(f'Pinned input changed: {key}')
    if sha(m['source']['evaluation']) != m['source']['evaluation_sha256']:
        raise ValueError('Original B0 evaluation changed.')
    parent = json.loads(Path(m['parent_manifest']['path']).read_text())
    if m['model_arguments'] != {k: parent[k] for k in ('environment_argv', 'training_contract')}:
        raise ValueError('Inherited source/environment arguments changed.')
    proof = json.loads(Path(m['engineering_proof']['path']).read_text())
    if (proof.get('passed') is not True or proof.get('checked_sources') != m['code_fingerprint']):
        raise ValueError('Engineering proof does not cover this implementation.')
    validate_case_content(m['evaluation']['cases'], m['evaluation']['dataset'])


def load_policy(m):
    """Validate the ORIGINAL source contract before explicit counterfactual use."""
    verify_resources()
    configure_bc_determinism(True)
    torch.set_num_threads(1)
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') != GPU_UUID.removeprefix('GPU-'):
        raise ValueError('Wrong physical GPU UUID.')
    torch.cuda.set_per_process_memory_fraction(.40, 0)
    args = arguments(m['model_arguments'])
    payload = torch.load(m['source']['path'], map_location='cpu', weights_only=True)
    if payload['resource_lookahead_contract'] != m['source_planning_contract']:
        raise ValueError('Source planning metadata changed.')
    for key, value in m['source_planning_contract'].items():
        if getattr(args, key, None) != value:
            raise ValueError(f'Original environment planning argument differs: {key}')
    ready = verified_frozen_ready_source(payload)
    lineage = validate_frozen_lineage(payload, m['stage1_source'])
    for key, value in ready_head_config(payload).items():
        setattr(args, key, value)
    seed(100011)
    policy = GNN_MAPPOPolicy(args, yaml.safe_load(Path(args.ac_config).read_text()), torch.device('cuda:0'))
    policy.load_model_state(payload['model'])
    policy.ac.tau = args.evaluation_tau
    policy.ac.device_global_matching = True
    original_validation = validate_stage2_joint_finetune_checkpoint(payload, policy.ac.state_dict(),
        plane_order_mode=policy.ac.plane_order_mode, plane_pair_decoder=policy.ac.plane_pair_decoder,
        global_feature_mode=args.global_feature_mode, planning_contract=m['source_planning_contract'],
        request_ready_time_scale=policy.ac.request_ready_time_scale)
    if summary(policy.ac) != summary(payload['model']):
        raise ValueError('Loaded tensors differ from the immutable B0 source.')
    # Historical no-grad evaluation keeps actor/critic requires_grad metadata.
    # Clearing it changes cuDNN GRU kernel selection on this host (V5 audit).
    # Preserve it, but construct NO optimizer and never enable autograd.
    configure_trainability(policy)
    policy.ac.eval()
    return policy, args, dict(passed=True, frozen_lineage=lineage,
        original_contract_validated=True, original_phase=original_validation.get('phase'),
        frozen_ready_source=ready, tensor_summary=summary(policy.ac),
        historical_requires_grad={n: p.requires_grad for n, p in policy.ac.named_parameters()})


class FrozenEvaluator(ResourceLearner):
    """Reuse the tested collector, without constructing any training optimizer."""
    def __init__(self, policy, args, progress):
        self.policy, self.args, self.progress = policy, args, progress
        self.device = policy.device
        self.contract = dict(rollout_max_steps=4000)
        self.diagnostics = dict(observed_case_events=0, real_request_sum=0,
            lookahead_request_sum=0, maximum_real_requests=0, forward_seconds=0.,
            ordinary=dict(active_rows=0, dispatch=0, wait=0, legal_edges=0),
            transporter=dict(active_rows=0, dispatch=0, wait=0, legal_edges=0))
        self.actor_steps = self.critic_steps = 0

    def collect(self, envs, **kwargs):
        if kwargs != dict(training=False, hungarian=True):
            raise ValueError('This collector permits only frozen Hungarian evaluation.')
        return super().collect(envs, **kwargs)

    def collect_forward(self, policy, obs, hidden, active, history, types, **kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError('H/F sensitivity requires no_grad; no learning is permitted.')
        started = time.perf_counter()
        result, encoded = forward(policy, obs, hidden, active, history, types, **kwargs)
        actions = result['actions'].detach().cpu().numpy()
        self.diagnostics['forward_seconds'] += time.perf_counter() - started
        for i, graph in enumerate(obs):
            active_rows = np.asarray(active[i]).reshape(-1).astype(bool)
            if not active_rows.any():
                continue
            width = int(graph['request'].x.shape[0])
            if width != 1 + CAPACITY * self.args.max_agent_num:
                raise ValueError(f'Request padding width changed: {width}')
            legal = graph.request_mask_matrix.detach().cpu().numpy()
            feature = graph['request'].x.detach().cpu().numpy()
            real = (feature[:, 6] >= 0) & (feature[:, 7] == 0)
            lookahead = graph.request_is_lookahead.detach().cpu().numpy()
            self.diagnostics['observed_case_events'] += 1
            self.diagnostics['real_request_sum'] += int(real.sum())
            self.diagnostics['lookahead_request_sum'] += int((real & lookahead).sum())
            self.diagnostics['maximum_real_requests'] = max(self.diagnostics['maximum_real_requests'], int(real.sum()))
            selected = []
            for agent in np.flatnonzero(active_rows):
                if agent < self.args.max_agent_num:
                    continue
                action = int(actions[i, agent, 0])
                if not 0 <= action < width or not legal[agent, action]:
                    raise ValueError('Frozen actor chose an illegal/padded request.')
                if action:
                    selected.append(action)
                role = 'transporter' if int(types[i, agent]) == policy.ac.AGENT_TYPE_TRANSPORTER else 'ordinary'
                record = self.diagnostics[role]
                record['active_rows'] += 1
                record['dispatch' if action else 'wait'] += 1
                record['legal_edges'] += int(legal[agent].sum())
            if len(selected) != len(set(selected)):
                raise ValueError('Hungarian action assigned one request more than once.')
        return result, encoded


def run_worker(path, label, slot):
    m = json.loads(path.read_text())
    verify_resources()
    verify_files(m)
    if slot not in (0, 1) or not set(os.sched_getaffinity(0)).issubset(cpus(SLOTS[slot])):
        raise ValueError('Worker escaped its declared CPU slot.')
    jobs = {j['label']: j for j in m['jobs']}
    if label not in jobs:
        raise ValueError('Undeclared evaluation job.')
    job, root = jobs[label], path.parent
    status_path = root / 'workers' / label / 'run_status.json'
    if status_path.exists() or (root / 'evaluations' / (label + '.json')).exists():
        raise FileExistsError('No automatic reruns or output reuse.')
    progress = Progress(status_path, phase='loading', label=label, arm=job['arm'],
                        completed_evaluation_cases=0, total_evaluation_cases=60)
    envs = None
    try:
        policy, args, source_check = load_policy(m)
        configure_args(args, job['arm'], native=job['native'])
        learner = FrozenEvaluator(policy, args, progress)
        atomic_json(status_path.parent / 'source_check.json', source_check)
        rows, parts, started = [], [], time.monotonic()
        torch.cuda.reset_peak_memory_stats(0)
        for start in range(0, 60, 12):
            group = m['evaluation']['cases'][start:start + 12]
            env_args = copy.copy(args)
            env_args.seed = 1
            env_args.max_train_cases = env_args.train_sampling_size = env_args.n_rollout_threads = 12
            progress('batch_start', force=True, phase='evaluation', batch=start // 12 + 1,
                event_step=0, completed_cases=0, completed_evaluation_cases=len(rows))
            envs, _ = make_train_env(env_args, case_records=group,
                dataset_override=m['evaluation']['dataset'], evaluation=True)
            seed(1)
            result = learner.collect(envs, training=False, hungarian=True)
            envs.close(); envs = None
            result.pop('frames'); result.pop('targets')
            result['cases'] = current_cases(result['cases'], group)
            atomic_json(root / 'evaluations' / label / f'batch_{start // 12 + 1}.json', result)
            rows.extend(result['cases'])
            parts.append(dict(batch=start // 12 + 1, seconds=result['seconds'],
                trajectory_action_sha256=result['trajectory_action_sha256']))
            progress('batch_completed', force=True, completed_evaluation_cases=len(rows))
        after = summary(policy.ac)
        if (after != source_check['tensor_summary']
                or {n: p.requires_grad for n, p in policy.ac.named_parameters()}
                != source_check['historical_requires_grad']
                or any(p.grad is not None for p in policy.ac.parameters())):
            raise ValueError('Frozen evaluation changed weights/trainability.')
        output = dict(protocol=PROTOCOL, label=label, arm=job['arm'], cases=rows,
            summary=metrics(rows), seconds=time.monotonic() - started, batches=parts,
            diagnostics=learner.diagnostics, tensor_summary_before=source_check['tensor_summary'],
            tensor_summary_after=after, source=m['source'],
            source_planning_contract=m['source_planning_contract'],
            evaluated_planning_contract=planning_for(m['source_planning_contract'], job['arm']),
            request_capacity=0 if job['native'] else CAPACITY,
            interpretation='zero_shot_counterfactual_hf', independent_confirmation=False,
            teacher_queries=0, actor_updates=0, critic_updates=0,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0))
        atomic_json(root / 'evaluations' / (label + '.json'), output)
        progress.finish('completed', phase='completed', completed_evaluation_cases=60,
                        mean_makespan=output['summary']['mean'])
        return 0
    except BaseException as exc:
        progress.finish('failed', error=str(exc), traceback=traceback.format_exc())
        return 1
    finally:
        if envs is not None:
            envs.close()


def run_controller(path):
    from onpolicy.utils.stage2_freeze_guard import reject_stage2_development
    reject_stage2_development()
    m = json.loads(path.read_text())
    verify_resources(); verify_files(m)
    root = path.parent
    if (root / 'run_status.json').exists():
        raise FileExistsError('New suite only; automatic retry is disabled.')
    os.sched_setaffinity(0, cpus('30-31,94-95'))
    progress = Progress(root / 'run_status.json', phase='source_replay',
        completed_jobs=0, total_jobs=14, completed_physical_case_episodes=0,
        physical_case_episode_limit=840, bc_training_started=False)
    running, finished = {}, []
    failed = False

    def launch(job, slot):
        target = root / 'workers' / job['label']
        target.mkdir(parents=True, exist_ok=False)
        with (target / 'worker.log').open('x') as log:
            command = ['taskset', '-c', SLOTS[slot], sys.executable, '-u', str(Path(__file__).resolve()),
                       '--manifest', str(path), '--worker', job['label'], '--slot', str(slot)]
            atomic_json(target / 'command.json', dict(argv=command, cpus=SLOTS[slot], gpu=0))
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        running[job['label']] = (proc, slot)
        progress('worker_started', force=True, label=job['label'], worker_pid=proc.pid,
                 active_jobs=list(running))

    def poll():
        nonlocal failed
        for label, (proc, slot) in list(running.items()):
            result = proc.poll()
            if result is None:
                continue
            del running[label]
            atomic_json(root / 'workers' / label / 'exit.json', dict(exit_code=result, finished_unix=time.time()))
            if result != 0 or not (root / 'evaluations' / (label + '.json')).is_file():
                failed = True
                progress('worker_failed', force=True, failed_job=label, exit_code=result,
                         error='No more jobs will be dispatched; existing independent jobs may finish.')
            else:
                finished.append(label)
                progress('worker_completed', force=True, completed_jobs=len(finished),
                    completed_physical_case_episodes=len(finished) * 60, active_jobs=list(running))
        # The controller does not itself simulate. Propagate real worker
        # progress so a healthy long evaluation is not marked as stalled.
        snapshots = {}
        for label in running:
            state_path = root / 'workers' / label / 'run_status.json'
            if state_path.is_file():
                state = json.loads(state_path.read_text())
                snapshots[label] = {k: state.get(k) for k in ('phase', 'event_step',
                    'completed_cases', 'total_cases', 'completed_evaluation_cases',
                    'last_work_progress_unix', 'stall_suspected')}
        with progress.lock:
            if snapshots:
                progress.state['last_work_progress_unix'] = max(
                    progress.state['last_work_progress_unix'],
                    *(s['last_work_progress_unix'] for s in snapshots.values()))
            progress.state['workers'] = snapshots
            progress.state['active_jobs'] = list(running)
            progress.state['completed_physical_case_episodes'] = len(finished) * 60 + sum(
                s.get('completed_evaluation_cases') or 0 for s in snapshots.values())

    def wait_active():
        while running:
            poll()
            if running:
                time.sleep(2)
        if failed:
            raise RuntimeError('An evaluation failed; all artifacts retained, no automatic retry.')

    try:
        for job in m['jobs'][:2]:
            launch(job, 0); wait_active()
            actual = json.loads((root / 'evaluations' / (job['label'] + '.json')).read_text())
            check = check_replay(m['evaluation']['cases'], actual['cases'])
            atomic_json(root / 'checks' / (job['label'] + '_replay.json'), check)
            if not check['passed']:
                raise ValueError(f"B0 strict source replay failed for {job['label']}; grid is not started.")
        progress('source_replay_passed', force=True, phase='grid')
        pending = list(m['jobs'][2:-1])
        while pending or running:
            poll()
            if not failed:
                occupied = {slot for _, slot in running.values()}
                for slot in range(2):
                    if slot not in occupied and pending:
                        launch(pending.pop(0), slot)
            elif not running:
                raise RuntimeError('Grid stopped dispatching after a failed worker.')
            if running:
                time.sleep(2)
        launch(m['jobs'][-1], 0); wait_active()
        first = json.loads((root / 'evaluations/H2_F4.json').read_text())
        last = json.loads((root / 'evaluations/H2_F4_repeat.json').read_text())
        repeat = check_replay(first['cases'], last['cases'])
        repeat['action_hashes_equal'] = ([p['trajectory_action_sha256'] for p in first['batches']]
            == [p['trajectory_action_sha256'] for p in last['batches']])
        repeat['passed'] = repeat['passed'] and repeat['action_hashes_equal']
        atomic_json(root / 'checks/terminal_repeat.json', repeat)
        if not repeat['passed']:
            raise ValueError('Terminal anchor repeat differs; do not accept the sensitivity ranking.')
        report = report_suite(root)
        atomic_json(root / 'analysis.json', report)
        progress.finish('completed', phase='A_completed_B_teacher_audit_pending',
            completed_jobs=14, completed_physical_case_episodes=840,
            observed_deployment_ranking=report['observed_deployment_ranking'], bc_training_started=False)
        return 0
    except BaseException as exc:
        progress.finish('failed', error=str(exc), traceback=traceback.format_exc(),
                        completed_jobs=len(finished), active_jobs=list(running), bc_training_started=False)
        # Ordinary errors do not kill unrelated independent evaluations. SIGTERM
        # from the operator is handled by systemd's owned control-group policy.
        for proc, _ in running.values():
            proc.wait()
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--worker')
    parser.add_argument('--slot', type=int, default=0)
    cli = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Operator stop')))
    sys.exit(run_worker(cli.manifest.resolve(), cli.worker, cli.slot) if cli.worker
             else run_controller(cli.manifest.resolve()))
