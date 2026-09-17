#!/usr/bin/env python3
"""Exercise a complete 64-environment sample and its unchanged PPO replay."""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from onpolicy.utils.stage3_b0_single_model import ThroughputEngine, VERSION, RNG_CONTRACT
    from onpolicy.utils.stage3_b0_throughput import capture_rng, restore_rng
    from onpolicy.utils.stage3_b0_restart import verify_restored_state
    from onpolicy.utils.stage3_b_shared_b0 import training_schedule, read_commit, record_summary
    from onpolicy.utils.stage3_research import read_json, atomic_json, digest_file
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import Progress, compare_states
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    m = read_json(args.manifest)
    out = args.output.resolve()
    out.mkdir(parents=True)
    commits = sorted((Path(m['root'])/'commits').glob('batch_*.json'))
    commit = read_commit(m, commits[-1])
    row = training_schedule(m)[commit['next_batch']]
    if len(row['cases']) != 64:
        raise ValueError('This probe requires the next complete 64-visit group')
    runtime = dict(global_batch=64, sampling_environments=64, sampling_lanes=1,
        microbatch=64, input_cache_mib=4096, cuda_memory_headroom_mib=8192,
        sampling_output=str(out/'sampling'), sampling_cpus=sorted(os.sched_getaffinity(0)))
    runner = ThroughputEngine(m['frozen_manifest']['path'], config=m['recipe'],
        width=64, create_pool=False, runtime=runtime)
    files = ('onpolicy/utils/stage3_b0_single_model.py',
             'onpolicy/runner/shared/stage3_b_shared_b0_engine.py',
             'onpolicy/utils/stage3_b0_throughput.py',
             'onpolicy/scripts/train/probe_stage3_b0_single_model.py')
    code = {name:digest_file(ROOT/name) for name in files}
    try:
        runner.resume(commit['checkpoint'], manifest_sha256=m['manifest_sha256'])
        restored = verify_restored_state(runner, commit['checkpoint'])
        atomic_json(out/'resume_verified.json', restored)
        original = runner.policy.get_actions
        repeated = False
        forward_sizes = []
        def checked(*a, **kw):
            nonlocal repeated
            forward_sizes.append(len(a[0]))
            if repeated:
                return original(*a, **kw)
            before = capture_rng()
            first = original(*a, **kw)
            after = capture_rng()
            restore_rng(before)
            second = original(*a, **kw)
            compare_states(first, second, exact=True)
            compare_states(after, capture_rng(), exact=True)
            repeated = True
            return first
        runner.policy.get_actions = checked
        with Progress(out/'status.json', phase='single_model_probe') as hb:
            started = time.monotonic()
            group = runner.parallel_collect(row['cases'], row['seeds'], heartbeat=hb)
            sample_seconds = time.monotonic()-started
            atomic_json(out/'rollouts.json', [record_summary(t) for t in group])
            atomic_json(out/'sample_complete.json', dict(sample_seconds=sample_seconds,
                visits=len(group), environment_steps=sum(t['steps'] for t in group),
                forward_calls=len(forward_sizes), first_forward_batch=forward_sizes[0]))
            started = time.monotonic()
            metrics = runner.replay_metrics(group, microbatch=64, heartbeat=hb)
            replay_seconds = time.monotonic()-started
            if metrics['max_logp_error'] > .002 or not metrics['decisions']:
                raise ValueError(f'Batched behavior replay failed: {metrics}')
            workers = read_json(out/'sampling/pool_00001/workers.json')
            result = dict(passed=True, architecture=VERSION, rng_contract=RNG_CONTRACT,
                parent_manifest_sha256=m['manifest_sha256'], code_files=code,
                checkpoint_sha256=commit['checkpoint_sha256'], next_batch=commit['next_batch'],
                checkpoint_restore_exact=restored['loaded_training_state_exact'],
                cuda_rng_restored_exact=restored['cuda_rng_checked'],
                repeated_forward_and_rng_exact=repeated, complete_trajectories=len(group),
                model_copies=workers['model_copies'], environment_count=workers['environment_count'],
                environment_cuda_visible_devices=workers['environment_cuda_visible_devices'],
                sample_seconds=sample_seconds, replay_seconds=replay_seconds,
                environment_steps=sum(t['steps'] for t in group), forward_calls=len(forward_sizes),
                first_forward_batch=forward_sizes[0], replay=metrics, updates_executed=0,
                peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                numerical_equivalence_to_legacy='not_required_by_user')
            atomic_json(out/'result.json', result, overwrite=False)
            hb.update(status='completed', event='sample_and_replay_passed', **result)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
