#!/usr/bin/env python3
"""Bounded Stage2 BC/resource-PPO experiment. No retries or auto-promotion."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import signal
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
from onpolicy.config.config import get_config
from onpolicy.scripts.train.run_stage2_resource_manifest_trial import atomic_json
from onpolicy.scripts.train.train_hkbz import make_train_env, parse_args
from onpolicy.utils.stage2_bc_contract import configure_bc_determinism, ready_head_config
from onpolicy.utils.stage2_bc_replay import validate_case_content
from onpolicy.utils.stage2_policy_transfer import verified_frozen_ready_source
from onpolicy.utils.stage2_resource_rl import (
    PROTOCOL, RESOURCE_PREFIXES, ResourceLearner, file_sha, reset_critic, summary,
)
from onpolicy.utils.stage2_resource_rl_handoff import validate_frozen_lineage
from onpolicy.utils.training_stage import validate_stage2_joint_finetune_checkpoint


def cpus(spec):
    result = set()
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        result.update(range(int(lo), int(hi or lo) + 1))
    return result


class Progress:
    def __init__(self, suite, limit):
        self.suite, self.limit = suite, limit
        self.started = time.time()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.state = dict(status='running', started_unix=self.started,
            pid=os.getpid(), phase='preflight', confirmed_scientific_success=False,
            automatic_promotion=False, costs_labels=0)
        self.last = 0.0
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()

    def heartbeat(self):
        os.sched_setaffinity(0, cpus('30-31,94-95'))
        while not self.stop.wait(30):
            self.write()

    def write(self):
        with self.lock:
            self.state.update(updated_unix=time.time(), elapsed_seconds=time.time() - self.started,
                              hard_deadline_unix=self.started + self.limit)
            atomic_json(self.suite / 'run_status.json', self.state)

    def __call__(self, event, force=False, **values):
        with self.lock:
            self.state.update(event=event, **values)
        if time.time() - self.started > self.limit:
            raise TimeoutError(f'Stage2 experiment reached its {self.limit}-second budget.')
        if force or time.time() - self.last >= 30:
            self.last = time.time()
            self.write()
            print('[S2RL] ' + json.dumps({k: v for k, v in self.state.items()
                if k not in ('cases',)}, ensure_ascii=False), flush=True)


def arguments(manifest):
    parser = get_config()
    args = parse_args(manifest['environment_argv'], parser)
    c = manifest['training_contract']
    for key, value in dict(training_stage='resource_joint', stage2_training_mode='bc_resource_rl',
        num_episodes=c['rounds'], ppo_epoch=c['ppo_passes'], device_bc_pretrain_epochs=0,
        hindsight_reward_mode='team_time', hindsight_terminal_cmax_coef=1.0,
        hindsight_cmax_coef=0.0, hindsight_shaping_coef=0.0, iga_potential_beta=0.0,
        resource_lateness_coef=0.0, resource_critical_lateness_coef=0.0,
        resource_earliness_coef=0.0, reward_coef=1 / c['return_scale'], gamma=1.0,
        use_valuenorm=False, use_gae=False, central_team_critic=True,
        adaptive_actor_kl=False, safe_async_graph_clone_workers=0,
        train_sampling_pool_size=0, train_sampling_mode='uniform',
        stage2_research_train_cases='', train_domain_rand=False, n_training_threads=1,
        request_ready_loss_coef=0.0, device_bc_skip_ready_targets=True,
        seed=c['seed'], lr=c['actor_lr'], critic_lr=c['critic_lr']).items():
        setattr(args, key, value)
    if getattr(args, 'stage2_cost_improvement', False):
        raise ValueError('Cost-label training cannot be combined with resource RL.')
    return args


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def enriched(rows, contract):
    lookup = {r['case_dir']: r for r in contract['cases']}
    if len(rows) != len(lookup) or len({r['case_dir'] for r in rows}) != len(lookup):
        raise ValueError('Case coverage differs from manifest.')
    return [{**lookup[r['case_dir']], **r} for r in rows]


def metrics(rows):
    valid = bool(rows) and all(r['completed'] and not r['cycle_terminated'] and not r['timeout'] for r in rows)
    values = [r['makespan'] for r in rows]
    if not valid or not np.isfinite(values).all():
        raise ValueError('Cannot select or summarize an incomplete evaluation.')
    return dict(mean=float(np.mean(values)), maximum=max(values),
        tail10=float(np.mean(sorted(values)[-max(1, int(np.ceil(len(values) * .1))):])),
        groups={g: float(np.mean([r['makespan'] for r in rows if r['distribution'] == g]))
                for g in sorted({r['distribution'] for r in rows})}, cases=len(rows))


def compare(reference, candidate):
    ref = {r['case_sha256']: r for r in reference}
    cur = {r['case_sha256']: r for r in candidate}
    if ref.keys() != cur.keys():
        raise ValueError('Paired comparison case hashes differ.')
    delta = [cur[k]['makespan'] - ref[k]['makespan'] for k in ref]
    before, after = metrics(reference), metrics(candidate)
    return dict(mean_delta=float(np.mean(delta)), relative_improvement=1 - after['mean'] / before['mean'],
        wins=sum(d < -1e-6 for d in delta), ties=sum(abs(d) <= 1e-6 for d in delta),
        losses=sum(d > 1e-6 for d in delta), maximum_positive_delta=max(0, max(delta)),
        group_deltas={g: after['groups'][g] - before['groups'][g] for g in before['groups']},
        maximum_case_delta=after['maximum'] - before['maximum'], tail10_delta=after['tail10'] - before['tail10'])


def run(cli):
    from onpolicy.utils.stage2_freeze_guard import reject_stage2_development
    reject_stage2_development()
    path = cli.manifest.resolve()
    manifest = json.loads(path.read_text())
    if manifest['protocol'] != PROTOCOL:
        raise ValueError('Wrong resource RL protocol.')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('This experiment requires physical GPU0 only.')
    allowed = cpus(manifest['resource_contract']['allowed_cpus'])
    if not set(os.sched_getaffinity(0)).issubset(allowed):
        raise ValueError('Launch must already be confined to the original CPU half.')
    for name, expected in manifest['code_fingerprint'].items():
        if file_sha(ROOT / name) != expected:
            raise ValueError(f'Pinned run source changed: {name}')
    for item in (manifest['source'], manifest['stage1_source'], manifest['source_archive']):
        if file_sha(item['path']) != item['sha256']:
            raise ValueError(f'Immutable source changed: {item["path"]}')
    if file_sha(manifest['source']['evaluation']) != manifest['source']['evaluation_sha256']:
        raise ValueError('Baseline evaluation changed.')
    if file_sha(manifest['teacher_index']) != manifest['teacher_index_sha256']:
        raise ValueError('Locked BC teacher index changed.')
    for key in ('train', 'evaluation'):
        validate_case_content(manifest[key]['cases'], manifest[key]['dataset'])
    suite = path.parent
    if (suite / 'run_status.json').exists():
        raise FileExistsError('No automatic retry or directory reuse.')
    c = manifest['training_contract']
    progress = Progress(suite, c['hard_timeout_seconds'])
    envs = None
    def stop_signal(signum, frame):
        if signum == signal.SIGALRM:
            raise TimeoutError('Predeclared experiment/check deadline reached.')
        raise KeyboardInterrupt(f'Operator/service signal {signum}')
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGALRM, stop_signal)
    signal.alarm(c['check_timeout_seconds'])
    try:
        configure_bc_determinism(True)
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.80, device=0)
        args = arguments(manifest)
        payload = torch.load(manifest['source']['path'], map_location='cpu', weights_only=True)
        verified_frozen_ready_source(payload)
        atomic_json(suite / 'checks' / 'frozen_lineage.json',
                    validate_frozen_lineage(payload, manifest['stage1_source']))
        for key, value in ready_head_config(payload).items():
            setattr(args, key, value)
        policy = GNN_MAPPOPolicy(args, yaml.safe_load(Path(args.ac_config).read_text()), torch.device('cuda:0'))
        policy.load_model_state(payload['model'])
        policy.ac.tau = args.evaluation_tau
        if summary(policy.ac) != summary(payload['model']):
            raise ValueError('Loaded warm-start tensors differ from B0.')
        learner = ResourceLearner(policy, args, c, progress)
        evaluations = {}

        def evaluate(label, hungarian=False):
            nonlocal envs
            progress('evaluation_start', force=True, evaluation_label=label)
            os.sched_setaffinity(0, cpus(manifest['resource_contract']['eval_cpus']))
            eval_args = copy.copy(args)
            eval_args.seed = 1
            eval_args.max_train_cases = eval_args.train_sampling_size = eval_args.n_rollout_threads = len(manifest['evaluation']['cases'])
            envs, _ = make_train_env(eval_args, case_records=manifest['evaluation']['cases'],
                dataset_override=manifest['evaluation']['dataset'], evaluation=True)
            seed(1)
            result = learner.collect(envs, training=False, hungarian=hungarian, audit=label == 'S2RL_F_AR')
            envs.close()
            envs = None
            rows = enriched(result['cases'], manifest['evaluation'])
            output = dict(label=label, cases=rows, summary=metrics(rows), seconds=result['seconds'],
                          independent_confirmation=False, decoder='hungarian' if hungarian else 'autoregressive_argmax')
            atomic_json(suite / 'evaluations' / (label + '.json'), output)
            evaluations[label] = rows
            return rows

        pre = evaluate('S2RL_F_H', hungarian=True)
        expected = {r['case_sha256']: r for r in manifest['evaluation']['cases']}
        replay_errors = [dict(case=r['case_dir'], actual=r['makespan'], expected=expected[r['case_sha256']]['makespan'],
                              steps=r['steps'], expected_steps=expected[r['case_sha256']]['finish_steps'])
            for r in pre if r['makespan'] != expected[r['case_sha256']]['makespan']
            or r['steps'] != expected[r['case_sha256']]['finish_steps']]
        atomic_json(suite / 'checks' / 'source_replay.json', dict(passed=not replay_errors,
                    cases=len(pre), errors=replay_errors, tolerance=0.0))
        if replay_errors:
            raise ValueError(f'Original Hungarian B0 strict replay failed: {replay_errors[:3]}')
        policy.ac.device_global_matching = False
        evaluate('S2RL_F_AR')
        atomic_json(suite / 'checks' / 'probabilities.json', learner.probability_checks)
        if learner.probability_checks['events'] == 0:
            raise ValueError('No probability replay checks were performed.')
        progress('preflight_passed', force=True, phase='training', checks_passed=True)
        signal.alarm(max(1, int(progress.started + c['hard_timeout_seconds'] - time.time())))
        results = {}
        for arm in manifest['execution']['arms']:
            bc = arm == 'S2RL_BC'
            policy.load_model_state(payload['model'])
            policy.ac.device_global_matching = False
            seed(c['seed'])
            if not bc:
                reset_critic(policy.ac)
                policy.capture_bc_reference(payload['model'])
                policy.bc_reference_ac.device_global_matching = False
            learner = ResourceLearner(policy, args, c, progress)
            outdir = suite / arm
            outdir.mkdir(exist_ok=False)
            rounds = []
            for round_index in range(1, c['rounds'] + 1):
                progress('round_start', force=True, phase='training', arm=arm,
                         collection_round=round_index, total_rounds=c['rounds'])
                os.sched_setaffinity(0, cpus(manifest['resource_contract']['train_cpus']))
                args.max_train_cases = args.train_sampling_size = args.n_rollout_threads = len(manifest['train']['cases'])
                envs, _ = make_train_env(args, case_records=manifest['train']['cases'])
                rollout = learner.collect(envs, training=True, bc=bc)
                envs.close()
                envs = None
                training_rows = enriched(rollout['cases'], manifest['train'])
                atomic_json(outdir / f'round_{round_index}_rollout.json', dict(cases=training_rows,
                    **{k: v for k, v in rollout.items() if k not in ('frames', 'targets', 'cases')}))
                progress('update_start', force=True)
                started = time.monotonic()
                update = learner.update(rollout, bc=bc, warmup=round_index == 1)
                update['seconds'] = time.monotonic() - started
                rounds.append(dict(round=round_index, collection_seconds=rollout['seconds'],
                                   update=update, completed_case_episodes=len(training_rows)))
                atomic_json(outdir / f'round_{round_index}_update.json', update)
                del rollout
                checkpoint = {**payload, 'model': {k: v.detach().cpu() for k, v in policy.ac.state_dict().items()},
                    'training_stage': 'resource_joint', 'stage2_training_mode': 'bc_resource_rl',
                    'phase': 'resource_rl_completed' if round_index == c['rounds'] and not bc else 'resource_rl_running',
                    'device_global_matching': False, 'resource_deployment_decoder': 'autoregressive',
                    'resource_rl_contract': dict(protocol=PROTOCOL, arm=arm, source=manifest['source'],
                        training=c, actor_steps=learner.actor_steps, critic_steps=learner.critic_steps,
                        completed_rounds=round_index, case_episodes=round_index * len(training_rows),
                        protected_before=learner.protected, protected_after=summary(policy.ac, protected=True),
                        actor_before=learner.source_actor, actor_after=summary(policy.ac, RESOURCE_PREFIXES),
                        behavior_decoder='autoregressive', evaluation_decoder='autoregressive_argmax',
                        teacher_execution=False, cost_queries=0,
                        probability_checks=learner.probability_checks),
                    'confirmed_scientific_success': False, 'stage2_scientific_gate': {'passed': False},
                    'selected_bc_epoch': payload.get('selected_bc_epoch'),
                    'resource_rl_training_state': dict(actor_optim=learner.actor_optim.state_dict(),
                        critic_optim=learner.critic_optim.state_dict(), torch_rng=torch.get_rng_state())}
                torch.save(checkpoint, outdir / f'checkpoint_round_{round_index}.pt')
                if checkpoint['phase'] == 'resource_rl_completed':
                    handoff = validate_stage2_joint_finetune_checkpoint(checkpoint,
                        policy.ac.state_dict(), plane_order_mode=policy.ac.plane_order_mode,
                        plane_pair_decoder=policy.ac.plane_pair_decoder,
                        global_feature_mode=args.global_feature_mode,
                        planning_contract=payload['resource_lookahead_contract'],
                        request_ready_time_scale=policy.ac.request_ready_time_scale,
                        resource_decoder='autoregressive')
                    atomic_json(outdir / 'static_handoff_check.json', dict(passed=True,
                        validation=handoff, dynamic_transfer_validated=False, stage3_started=False))
            evaluate(arm)
            results[arm] = dict(rounds=rounds, checkpoint=str(outdir / f'checkpoint_round_{c["rounds"]}.pt'),
                versus_ar=compare(evaluations['S2RL_F_AR'], evaluations[arm]),
                versus_original=compare(evaluations['S2RL_F_H'], evaluations[arm]))
            atomic_json(outdir / 'result.json', results[arm])
        contrast = compare(evaluations['S2RL_BC'], evaluations['S2RL_PPO'])
        main = results['S2RL_PPO']
        passed = bool(contrast['mean_delta'] < 0 and all(
            main[k]['relative_improvement'] >= .005
            and max(main[k]['group_deltas'].values()) <= 0
            and main[k]['maximum_case_delta'] <= 0
            for k in ('versus_ar', 'versus_original')))
        atomic_json(suite / 'analysis.json', dict(methods=results, ppo_versus_bc=contrast,
            screen_threshold_passed=passed, confirmed_scientific_success=False,
            goal='fully_better_IGA180_and_at_least_IGA1800', independent_confirmation=False,
            source_retained=not passed, automatic_promotion=False, stage3_started=False))
        progress('completed', force=True, status='completed', phase='completed',
                 screen_threshold_passed=passed, completed_training_case_episodes=96)
        return 0
    except BaseException as exc:
        status = 'operator_stopped' if isinstance(exc, KeyboardInterrupt) else 'timeout' if isinstance(exc, TimeoutError) else 'failed'
        with progress.lock:
            progress.state.update(status=status, error=str(exc), traceback=traceback.format_exc())
        progress.write()
        print(traceback.format_exc(), flush=True)
        return 130 if status == 'operator_stopped' else 1
    finally:
        signal.alarm(0)
        progress.stop.set()
        if envs is not None:
            envs.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    sys.exit(run(parser.parse_args()))
