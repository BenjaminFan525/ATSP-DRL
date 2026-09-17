#!/usr/bin/env python3
"""Benchmark and launch a separately frozen execution adapter for a B0 study."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess


def load(manifest, module_path):
    m = json.loads(Path(manifest).read_text())
    sys.path.insert(0, str(Path(m['source_root'])))
    sys.path.insert(0, str(Path(module_path).parent))
    runtime = importlib.import_module(Path(module_path).stem)
    from onpolicy.utils.stage3_b_shared_b0 import verify_manifest
    verify_manifest(m, inputs=True)
    return m, runtime


def sample(args, m, extension):
    import torch
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states, read_baseline, Progress
    from onpolicy.utils.stage3_b_shared_b0 import record_summary
    from onpolicy.utils.stage3_research import atomic_json, digest_file
    out = args.output.resolve()
    out.mkdir(parents=True)
    reference = Path(args.reference)
    base = read_baseline(m)
    runner = extension.ThroughputEngine(m['frozen_manifest']['path'], config=m['recipe'],
        create_pool=False, runtime=dict(sampling_output=str(out/'lanes'), sampling_lanes=args.lanes,
            sampling_environments=args.visits,
            release_sampling_before_update=True,
            sampling_cpus=sorted(os.sched_getaffinity(0)), input_cache_mib=args.cache_mib))
    try:
        runner.resume(reference/'initial.pt', manifest_sha256=m['manifest_sha256'], allow_diagnostic=True)
        # Larger capacity probes duplicate the frozen 32-case fixture in
        # distinct environments. It is a timing/numerical oracle, never training.
        repeats=args.visits//32
        cases = m['canary_cases']*4*repeats
        seeds = [m['recipe']['seed']+10000+i for i in range(32)]*repeats
        with Progress(out/'status.json', phase='sampling_benchmark') as hb:
            started = time.monotonic()
            group = runner.parallel_collect(cases, seeds, heartbeat=hb)
            elapsed = time.monotonic()-started
            rows = [record_summary(t) for t in group]
            expected = json.loads((reference/'rollouts.json').read_text())*repeats
            if rows != expected:
                atomic_json(out/'row_mismatches.json', [dict(index=i, expected=a, actual=b)
                    for i,(a,b) in enumerate(zip(expected,rows)) if a!=b])
                raise AssertionError('Parallel collection changed original canary visits')
            previous = torch.load(reference/'micro8.pt', map_location='cpu', weights_only=False)
            for key,value in extension.capture_rng().items():
                compare_states(previous[key], value, exact=True)
            # Keep process-independent retained inputs for update comparisons.
            torch.save(group, out/'trajectories.pt')
            runner.save(out/'after_collection.pt', manifest_sha256=m['manifest_sha256'],
                next_batch=0, diagnostic=True)
            atomic_json(out/'result.json', dict(passed=True, version=extension.VERSION,
                manifest_sha256=m['manifest_sha256'], extension_sha256=digest_file(args.module),
                reference=str(reference), source_checkpoint_sha256=digest_file(reference/'initial.pt'),
                sampling_lanes=args.lanes, active_environments=args.visits, seconds=elapsed,
                visits=args.visits, rows=rows, rows_exact=True, rng_exact=True,
                diagnostic_repeated_inputs=args.visits>32,
                original_canary_colocated=not args.exclusive_gpu,
                release_sampling_before_update=True))
            print('SAMPLING_PASSED',elapsed,flush=True)
    finally:
        runner.close()


def update(args, m, extension):
    import torch
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states, read_baseline, Progress
    from onpolicy.utils.stage3_research import atomic_json, digest_file
    out, reference = args.output.resolve(), Path(args.reference)
    out.mkdir(parents=True)
    base = read_baseline(m)
    # This diagnostic shares GPU0 with the original canary; leave its existing
    # allocations and the study's six-GiB headroom intact even on a failed probe.
    total_mib=torch.cuda.get_device_properties(0).total_memory/2**20
    torch.cuda.set_per_process_memory_fraction((total_mib-8192)/total_mib if args.exclusive_gpu else .50, 0)
    runner = extension.ThroughputEngine(m['frozen_manifest']['path'], config=m['recipe'],
        create_pool=False, runtime=dict(input_cache_mib=args.cache_mib, microbatch=args.microbatch))
    try:
        group = torch.load(args.trajectories, map_location='cpu', weights_only=False)
        if len(group)!=args.visits:
            raise ValueError('Capacity probe trajectory count differs')
        runner.resume(reference/'initial.pt', manifest_sha256=m['manifest_sha256'], allow_diagnostic=True)
        with Progress(out/'status.json', phase='update_benchmark') as hb:
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            result = runner.update(group, base['costs'], heartbeat=hb)
            elapsed = time.monotonic()-started
            checkpoint = out/'actual.pt'
            runner.save(checkpoint, manifest_sha256=m['manifest_sha256'], next_batch=1, diagnostic=True)
            expected = torch.load(reference/'micro8.pt', map_location='cpu', weights_only=False)
            actual = torch.load(checkpoint, map_location='cpu', weights_only=False)
            for key in ('model','actor_optim','critic_optim','value_normalizer','policy_updates'):
                compare_states(expected[key],actual[key], exact=args.microbatch==8 and args.visits==32,
                    rtol=0 if key=='model' else 2e-5)
            initial=torch.load(reference/'initial.pt',map_location='cpu',weights_only=False)
            changed=[name for name,value in actual['model'].items() if not torch.equal(value,initial['model'][name])]
            for prefix in ('encoder.op_embedding.','encoder.convs.0.','encoder.convs.3.',
                           'actor.','device_actor.','transporter_actor.'):
                if not any(name.startswith(prefix) for name in changed):
                    raise AssertionError(f'No demonstrated actor update for {prefix}')
            if not actual['actor_optim']['state'] or not actual['critic_optim']['state']:
                raise AssertionError('Full update must create nonempty Adam moments')
            atomic_json(out/'result.json',dict(passed=True, version=extension.VERSION,
                manifest_sha256=m['manifest_sha256'], extension_sha256=digest_file(args.module),
                microbatch=args.microbatch, global_visits=args.visits, ppo_epochs=2,
                input_cache_mib=args.cache_mib, diagnostic_repeated_inputs=args.visits>32,
                seconds=elapsed, exact_training_state=args.microbatch==8 and args.visits==32, result=result,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                changed_model_tensors=len(changed),all_actor_levels_changed=True,
                reference=str(reference/'micro8.pt'), reference_sha256=digest_file(reference/'micro8.pt')))
            # Save provenance separately to keep older comparison reports readable.
            atomic_json(out/'checkpoint.json',dict(path=str(checkpoint),sha256=digest_file(checkpoint)))
            print('UPDATE_PASSED',elapsed,flush=True)
    finally:
        runner.close()


def verify_runtime(path, m, module, launcher):
    from onpolicy.utils.stage3_research import read_json, digest_file, digest_json
    r=read_json(path)
    proof_manifest=r.get('proof_manifest_sha256',m['manifest_sha256'])
    if (r['study_manifest_sha256']!=m['manifest_sha256']
            or digest_json({k:v for k,v in r.items() if k!='runtime_sha256'})!=r['runtime_sha256']
            or digest_file(module)!=r['module_sha256'] or digest_file(launcher)!=r['launcher_sha256']
            or r['global_batch'] not in (32,64,128,240,256,512) or r['ppo_epochs']!=2
            or r['sampling_environments']!=r['global_batch']
            or r['sampling_lanes'] not in (1,4,8,16,32) or r['microbatch'] not in (8,16,32,64,80,120,128,240,256)
            or r['microbatch']>r['global_batch']
            or r['input_cache_mib'] not in (1024,4096,8192,12288,16384,24576,32768)
            or (m.get('recipe') and (r['global_batch']!=m['recipe']['global_batch']
                or r['microbatch']!=m['recipe']['microbatch']))):
        raise ValueError('Throughput runtime identity or budget changed')
    if m.get('resume_amendment'):
        from onpolicy.utils.stage3_b0_restart import verify_restart_runtime
        verify_restart_runtime(m, r)
        return r
    if proof_manifest!=m['manifest_sha256']:
        if (m['throughput_amendment']['parent_manifest_sha256']!=proof_manifest
                or digest_file(r['selection']['path'])!=r['selection']['sha256']
                or not r['release_sampling_before_update'] or r['cuda_memory_headroom_mib']!=8192):
            raise ValueError('Parent capacity evidence identity changed')
    for kind in ('sample','update'):
        proof=r['proofs'][kind]
        if digest_file(proof['path'])!=proof['sha256']:
            raise ValueError('Throughput proof changed')
        value=read_json(proof['path'])
        if (not value['passed'] or value['manifest_sha256']!=proof_manifest
                or value['extension_sha256']!=r['module_sha256']):
            raise ValueError('Throughput proof does not admit this runtime')
        if kind=='sample' and (value['visits']!=r['global_batch'] or not value['rows_exact']
                or not value['rng_exact'] or value['sampling_lanes']!=r['sampling_lanes']):
            raise ValueError('Parallel collection did not preserve all visits and RNG')
        if kind=='update' and (value['global_visits']!=r['global_batch'] or value['ppo_epochs']!=2
                or value['microbatch']!=r['microbatch']
                or value.get('input_cache_mib',4096)!=r['input_cache_mib']):
            raise ValueError('Complete global update proof missing')
    if digest_file(r['tests']['path'])!=r['tests']['sha256']:
        raise ValueError('Throughput CPU regression evidence changed')
    return r


def worker(args,m,extension,runtime):
    from onpolicy.scripts.train import stage3_b_shared_b0_worker as original
    from onpolicy.utils.stage3_research import atomic_json
    original_engine=original.engine
    original_collect=original.collect
    def engine(manifest,*,training=True,width=None,pool=True):
        if not training:
            return original_engine(manifest,training=False,width=width,pool=pool)
        settings=dict(runtime, sampling_output=str(args.output/'sampling'),
            sampling_cpus=list(range(32))+list(range(64,96)))
        runner=extension.ThroughputEngine(manifest['frozen_manifest']['path'],
            config=manifest['recipe'],training=True,width=runtime['global_batch'],create_pool=False,runtime=settings)
        original_update=runner.update
        def update_with_identity(*a,**kw):
            result=original_update(*a,**kw)
            result.update(runtime_sha256=runtime['runtime_sha256'],sampling_environments=runtime['sampling_environments'],
                sampling_lanes=runtime['sampling_lanes'],module_sha256=runtime['module_sha256'])
            if runtime.get('sampling_architecture'):
                result.update(sampling_architecture=runtime['sampling_architecture'],
                    sampling_rng_contract=runtime['sampling_rng_contract'], sampling_model_copies=1)
            return result
        runner.update=update_with_identity
        return runner
    def collect(runner,cases,seeds,hb):
        if isinstance(runner,extension.ThroughputEngine):
            return runner.parallel_collect(cases,seeds,heartbeat=hb)
        return original_collect(runner,cases,seeds,hb)
    original.engine,original.collect=engine,collect
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/'runtime.json',runtime,overwrite=False)
    sys.argv=[str(original.__file__),str(args.manifest),'--phase',args.phase,'--output',str(args.output)]
    if args.resume_commit:
        sys.argv.extend(['--resume-commit',str(args.resume_commit)])
    original.main()


def continuation(args,m,extension,runtime,*,replay=False):
    """Actual nonempty-Adam continuation, then a separate-process exact replay."""
    import torch
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states,read_baseline,Progress
    from onpolicy.utils.stage3_b_shared_b0 import cpus,record_summary,manifest_identity
    from onpolicy.utils.stage3_research import read_json,atomic_json,digest_file
    os.sched_setaffinity(0,cpus(m['recipe']['trainer_cpus']))
    out=args.output.resolve();out.mkdir(parents=True)
    base=read_baseline(m)
    settings=dict(runtime,sampling_output=str(out/'sampling'),
                  sampling_cpus=list(range(32))+list(range(64,96)))
    runner=extension.ThroughputEngine(m['frozen_manifest']['path'],config=m['recipe'],
        create_pool=False,runtime=settings)
    try:
        with Progress(out/'status.json',phase='fresh_process_resume' if replay else 'continuation_canary') as hb:
            if replay:
                proof=read_json(args.canary_dir/'result.json')
                if (proof['manifest_sha256']!=m['manifest_sha256']
                        or digest_file(proof['source_checkpoint'])!=proof['source_checkpoint_sha256']):
                    raise ValueError('Continuation checkpoint identity changed')
                runner.resume(proof['source_checkpoint'],manifest_sha256=m['manifest_sha256'],allow_diagnostic=True)
                cases,seeds=proof['next_cases'],proof['next_seeds']
            else:
                parent=read_json(runtime['proof_manifest'])
                if (manifest_identity(parent)!=runtime['proof_manifest_sha256']
                        or parent['manifest_sha256']!=runtime['proof_manifest_sha256']):
                    raise ValueError('Parent diagnostic manifest changed')
                checkpoint=runtime['diagnostic_checkpoint']
                if digest_file(checkpoint['path'])!=checkpoint['sha256']:
                    raise ValueError('Selected diagnostic checkpoint changed')
                # Explicit diagnostic transplantation: preserve model/Adam/norm/
                # RNG from the proved capacity update. Formal training starts B0.
                runner.config=parent['recipe']
                runner.resume(checkpoint['path'],manifest_sha256=parent['manifest_sha256'],allow_diagnostic=True)
                runner.config=m['recipe']
                source=out/'nonempty_adam.pt'
                runner.save(source,manifest_sha256=m['manifest_sha256'],next_batch=1,diagnostic=True)
                cases=m['canary_cases'][:4]
                seeds=[m['recipe']['seed']+20000+i for i in range(4)]
            if (not runner.policy.actor_optimizer.state or not runner.policy.critic_optimizer.state
                    or runner.normalization_sha256!=base['normalization_sha256']):
                raise ValueError('Continuation needs proved nonempty Adam and frozen normalization')
            group=runner.parallel_collect(cases,seeds,heartbeat=hb)
            rows=[record_summary(t) for t in group]
            if replay and rows!=proof['expected_next_rows']:
                raise AssertionError('Fresh-process resume changed the next four complete visits')
            result=runner.update(group,base['costs'],heartbeat=hb)
            path=out/('actual_next.pt' if replay else 'expected_next.pt')
            runner.save(path,manifest_sha256=m['manifest_sha256'],next_batch=2,diagnostic=True)
            if replay:
                expected=torch.load(proof['expected_next_checkpoint'],map_location='cpu',weights_only=False)
                actual=torch.load(path,map_location='cpu',weights_only=False)
                for key in ('model','actor_optim','critic_optim','value_normalizer','policy_updates',
                            'rng_python','rng_numpy','rng_torch','rng_cuda'):
                    compare_states(expected[key],actual[key],exact=True)
                value=dict(passed=True,fresh_process=True,manifest_sha256=m['manifest_sha256'],
                    next_collection_equal=True,all_training_state_exact=True,runtime_sha256=runtime['runtime_sha256'])
            else:
                selected=read_json(runtime['selection']['path'])
                value=dict(passed=True,manifest_sha256=m['manifest_sha256'],runtime_sha256=runtime['runtime_sha256'],
                    complete_global_visits=runtime['global_batch'],microbatch_comparison=[8,runtime['microbatch']],
                    source_checkpoint=str(source),source_checkpoint_sha256=digest_file(source),
                    parent_diagnostic_checkpoint=checkpoint,next_cases=cases,next_seeds=seeds,
                    expected_next_rows=rows,expected_next_checkpoint=str(path),update=result,
                    capacity_proofs=runtime['proofs'],batch_seconds=selected['sample_seconds']+selected['update_seconds'],
                    gate_amendment='matched complete global update plus actual nonempty-Adam continuation; diagnostic only')
            atomic_json(out/'result.json',value,overwrite=False)
    finally:
        runner.close()


def resume(args,m,extension,runtime):
    from onpolicy.scripts.train import run_stage3_b_shared_b0 as original
    from onpolicy.utils.stage3_research import atomic_json
    from types import SimpleNamespace
    launch=original.Supervisor.launch
    def accelerated_launch(supervisor,phase,**options):
        if phase not in ('train','diagnostics'):
            return launch(supervisor,phase,**options)
        output=supervisor.output/phase
        output.mkdir(parents=True)
        command=[m['python'],'-B','-u',str(Path(__file__).resolve()),'worker',str(args.manifest),
            '--module',str(args.module),'--runtime',str(args.runtime),'--phase',phase,'--output',str(output)]
        for key,value in options.items():
            command.extend(['--'+key.replace('_','-'),str(value)])
        atomic_json(output/'command.json',dict(argv=command,cwd=m['source_root'],
            runtime_sha256=runtime['runtime_sha256']),overwrite=False)
        with (output/'terminal.log').open('x') as log:
            process=subprocess.Popen(command,cwd=m['source_root'],env=os.environ.copy(),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        supervisor.children.append((phase,process))
        return process
    original.Supervisor.launch=accelerated_launch
    # Preserve original manifest, full admission, cursor validation, validation
    # queue and candidate-selection code. Only train/diagnostic workers change.
    original.run(SimpleNamespace(manifest=args.manifest,commit=None),resume=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command', choices=('sample','update','worker','resume','continuation','resume-proof'))
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--module',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--lanes',type=int,default=16)
    parser.add_argument('--microbatch',type=int,default=8)
    parser.add_argument('--visits',type=int,choices=(32,64,128,256,512),default=32)
    parser.add_argument('--cache-mib',type=int,choices=(4096,8192,12288,16384,24576,32768),default=4096)
    parser.add_argument('--exclusive-gpu',action='store_true')
    parser.add_argument('--trajectories',type=Path)
    parser.add_argument('--runtime',type=Path)
    parser.add_argument('--phase',choices=('train','diagnostics'))
    parser.add_argument('--resume-commit',type=Path)
    parser.add_argument('--canary-dir',type=Path)
    args=parser.parse_args()
    m,extension=load(args.manifest,args.module)
    if args.command in ('sample','update'):
        if args.output is None or args.reference is None:
            parser.error('Benchmarks need --output and --reference')
        (sample if args.command=='sample' else update)(args,m,extension)
    else:
        if args.runtime is None:
            parser.error('Execution requires a proved --runtime')
        runtime=verify_runtime(args.runtime,m,args.module,Path(__file__).resolve())
        if args.command in ('continuation','resume-proof'):
            continuation(args,m,extension,runtime,replay=args.command=='resume-proof')
        else:
            (worker if args.command=='worker' else resume)(args,m,extension,runtime)


if __name__=='__main__':
    main()
