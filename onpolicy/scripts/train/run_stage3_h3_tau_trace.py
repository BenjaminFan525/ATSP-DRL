#!/usr/bin/env python3
"""Replay frozen fixed12 batches and observe first temperature action divergence.

Observers copy intermediate tensors; all original policy operations are retained.
This diagnostic never trains, solves IGA, or edits a running experiment.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
import time
import types

TARGETS = ('case_0020', 'case_0052', 'case_0055', 'case_0114')
TAUS = (.3, .03, .5)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    q.replace(p)


def tag(tau):
    return str(tau).replace('.', 'p')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def observed_function(original, marker, kind, observer):
    """Insert one observer immediately before a temperature arithmetic statement."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    before = ast.dump(tree, include_attributes=False)
    body = tree.body[0].body
    positions = [i for i, node in enumerate(body) if ast.unparse(node).startswith(marker)]
    if len(positions) != 1:
        raise ValueError(f'Expected exactly one observer site in {original.__qualname__}')
    call = ast.parse(f'_tau_trace_observe({kind!r}, locals())').body[0]
    index = positions[0]
    body.insert(index, call)
    check = copy.deepcopy(tree)
    del check.body[0].body[index]
    assert ast.dump(check, include_attributes=False) == before
    namespace = dict(original.__globals__, _tau_trace_observe=observer)
    exec(compile(ast.fix_missing_locations(tree), '<temperature-trace-observer>', 'exec'), namespace)
    return namespace[original.__name__]


def tree_digest(value):
    import numpy as np
    import torch
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            add(x.detach().cpu().contiguous().numpy())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif hasattr(x, 'to_dict'):
            add(x.to_dict())
        elif isinstance(x, dict):
            for k in sorted(x, key=repr):
                add(k); add(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode())
            for item in x:
                add(item)
        else:
            h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


def exact_sum(values):
    return sum((Fraction.from_float(float(x)) for x in values), Fraction())


def matching_analysis(a, b):
    """Compare two physical assignments on each saved scoring matrix."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if not np.array_equal(a['rows'], b['rows']) or a['scores'].shape != b['scores'].shape:
        return dict(comparable=False, reason='active rows or request dimension differs')
    if not np.array_equal(np.isfinite(a['raw']), np.isfinite(b['raw'])):
        return dict(comparable=False, reason='legal support differs')
    A, B = a['assignment'], b['assignment']
    finite = np.isfinite(a['raw']) & np.isfinite(b['raw'])
    result = dict(comparable=True, rows=a['rows'].tolist(),
                  assignment_reference=A.tolist(), assignment_candidate=B.tolist(),
                  raw_exactly_equal=bool(np.array_equal(a['raw'], b['raw'])),
                  raw_max_abs_difference=float(np.max(np.abs(a['raw'][finite]-b['raw'][finite]))) if finite.any() else 0.,
                  changed_agents=a['rows'][A != B].tolist(), matrices={})
    for label, row in [('reference', a), ('candidate', b)]:
        z, scores, look = row['raw'], row['scores'], row['lookahead']
        n, columns = scores.shape
        i = np.arange(n)
        if not np.isfinite(scores[i, A]).all() or not np.isfinite(scores[i, B]).all():
            result['matrices'][label] = dict(comparable=False, reason='cross-assignment is illegal')
            continue
        blocking = lambda assignment: int(((assignment > 0) & ~look[assignment]).sum())
        bonusA = ((A > 0) & ~look[A]).astype(float)*1e6
        bonusB = ((B > 0) & ~look[B]).astype(float)*1e6
        raw_gap = exact_sum(z[i, A])-exact_sum(z[i, B])
        lp_gap = exact_sum(scores[i, A])-exact_sum(scores[i, B])
        deployed_gap = exact_sum(scores[i, A].astype(float)+bonusA)-exact_sum(scores[i, B].astype(float)+bonusB)
        matrix = np.full((n, columns-1+n), -1e30)
        matrix[:, :columns-1] = scores[:, 1:].astype(float)
        matrix[:, :columns-1][np.isfinite(scores[:, 1:]) & ~look[None, 1:]] += 1e6
        matrix[i, columns-1+i] = scores[:, 0]
        matrix[~np.isfinite(matrix)] = -1e30
        chosen = row['assignment']
        chosen_columns = np.where(chosen > 0, chosen-1, columns-1+i)
        chosen_total = exact_sum(matrix[i, chosen_columns])
        alternatives = []
        for k in range(n):
            test = matrix.copy(); test[k, chosen_columns[k]] = -1e30
            _, alt = linear_sum_assignment(test, maximize=True)
            if np.any(test[i, alt] <= -1e25):
                continue
            alternatives.append(exact_sum(matrix[i, alt]))
        second_gap = chosen_total-max(alternatives) if alternatives else None
        result['matrices'][label] = dict(
            blocking_reference=blocking(A), blocking_candidate=blocking(B),
            raw_score_reference_minus_candidate=float(raw_gap), raw_gap_exact=str(raw_gap),
            logprob_reference_minus_candidate=float(lp_gap),
            deployed_reference_minus_candidate=float(deployed_gap),
            chosen_deployed_total=float(chosen_total),
            chosen_minus_best_forced_alternative=float(second_gap) if second_gap is not None else None,
            forced_alternative_count=len(alternatives),
            tie_detected=bool(second_gap == 0) if second_gap is not None else False)
    return result


def compare_frames(base, candidate):
    import numpy as np
    changed = np.flatnonzero(np.any(base['actions'] != candidate['actions'], axis=1))
    active_changed = [int(i) for i in changed if base['active'][i] or candidate['active'][i]]
    result = dict(step_zero_based=candidate['step'], decision_number=candidate['step']+1,
                  time_reference=base['time'], time_candidate=candidate['time'],
                  changed_agents=changed.tolist(), active_changed_agents=active_changed,
                  graph_identical=base['graph_sha256']==candidate['graph_sha256'],
                  recurrent_input_identical=bool(np.array_equal(base['hidden'], candidate['hidden'])),
                  recurrent_input_max_abs=float(np.max(np.abs(base['hidden']-candidate['hidden']))),
                  previous_actions_identical=bool(np.array_equal(base['history'], candidate['history'])),
                  actions_reference=base['actions'][changed].tolist(),
                  actions_candidate=candidate['actions'][changed].tolist(),
                  plane_changes=[], matching=None)
    for agent in active_changed:
        if agent < 24 and agent in base['planes'] and agent in candidate['planes']:
            z, w = base['planes'][agent], candidate['planes'][agent]
            finite = np.isfinite(z) & np.isfinite(w)
            result['plane_changes'].append(dict(agent=agent, raw_identical=bool(np.array_equal(z,w)),
                raw_max_abs=float(np.max(np.abs(z[finite]-w[finite]))) if finite.any() else 0.,
                argmax_reference=int(np.argmax(z)), argmax_candidate=int(np.argmax(w))))
    if base.get('matching') is not None and candidate.get('matching') is not None:
        result['matching'] = matching_analysis(base['matching'], candidate['matching'])
    return result


class Recorder:
    def __init__(self, runner, root, heartbeat):
        import numpy as np
        import torch
        from onpolicy.algorithms.utils import ptr_actor
        self.runner, self.ac, self.root, self.hb = runner, runner.policy.ac, Path(root), heartbeat
        self.current, self.slots, self.streams, self.baselines = {}, {}, {}, {}
        self.findings, self.earliest_input_differences = {}, {}
        self.original_resource = ptr_actor.MaPtrNet.dist
        self.original_plane = ptr_actor.JointPairPtrActor.forward
        ptr_actor.MaPtrNet.dist = observed_function(self.original_resource, 'prob = F.softmax(ptr / tau', 'resource', self.observe)
        ptr_actor.JointPairPtrActor.forward = observed_function(self.original_plane, 'logits = logits / max(float(tau)', 'plane', self.observe)
        original_matching = self.ac._apply_device_global_matching
        def matching(ac, scores, lookahead, active, choices, logp, decision):
            result = original_matching(scores, lookahead, active, choices, logp, decision)
            for b, record in self.current.items():
                rows = torch.nonzero(active[b, ac.max_plane_agents:].bool(), as_tuple=False).flatten()+ac.max_plane_agents
                row_ids = rows.cpu().numpy()
                if not len(row_ids):
                    continue
                raw = np.stack([record['resource_raw'][int(agent)] for agent in row_ids])
                record['matching'] = dict(rows=row_ids, raw=raw, scores=scores[b, rows].detach().cpu().numpy().copy(),
                    lookahead=lookahead[b].cpu().numpy().copy(), assignment=choices[b, rows].cpu().numpy().copy())
            return result
        self.ac._apply_device_global_matching = types.MethodType(matching, self.ac)
        original_get_actions = runner.policy.get_actions
        def get_actions(graph, hidden, active, op, site, **kwargs):
            frame = inspect.currentframe().f_back
            while frame is not None and frame.f_code.co_name != '_collect':
                frame = frame.f_back
            if frame is None:
                raise RuntimeError('Trace requires the original frozen _collect loop')
            local = frame.f_locals
            self.current = {}
            for slot, name in self.slots.items():
                if local['live'][slot]:
                    self.current[slot] = dict(case=name, tau=self.tau, step=local['step'],
                        time=float(local['infos'][slot]['env_total_time']),
                        graph_sha256=tree_digest(graph[slot]), hidden=hidden[slot].copy(),
                        active=active[slot].copy(), history=np.stack([op[slot],site[slot]],axis=-1),
                        planes={}, resource_raw={})
            del frame, local
            output = original_get_actions(graph, hidden, active, op, site, **kwargs)
            actions = output[1].detach().cpu().numpy().astype(np.int64)
            for slot, record in self.current.items():
                name=record['case'];record['actions']=actions[slot].copy()
                record.pop('resource_raw')
                pickle.dump(record,self.streams[name],protocol=5)
                if name in self.baselines:
                    try: base=pickle.load(self.baselines[name])
                    except EOFError: continue
                    if base['step'] != record['step']:
                        raise RuntimeError('Trace steps lost alignment')
                    if name not in self.earliest_input_differences and (
                        base['graph_sha256'] != record['graph_sha256'] or not np.array_equal(base['hidden'],record['hidden'])):
                        self.earliest_input_differences[name] = dict(step=record['step'], graph_identical=base['graph_sha256']==record['graph_sha256'],
                            hidden_max_abs=float(np.max(np.abs(base['hidden']-record['hidden']))))
                    changed=np.any(base['actions']!=record['actions'],axis=1)
                    if name not in self.findings and np.any(changed & (base['active'].astype(bool)|record['active'].astype(bool))):
                        finding=compare_frames(base,record)
                        finding.update(case=name, tau=self.tau, reference_tau=.3,
                            earliest_input_difference=self.earliest_input_differences.get(name))
                        self.findings[name]=finding
                        write(self.root/'findings'/f'{name}_tau_{tag(self.tau)}.json',finding)
                        with gzip.open(self.root/'findings'/f'{name}_tau_{tag(self.tau)}.pkl.gz','wb',compresslevel=1) as f:
                            pickle.dump(dict(reference=base,candidate=record),f,protocol=5)
                        self.hb.update(event='first_action_divergence',case=name,tau=self.tau,
                            decision_number=record['step']+1,changed_agents=finding['active_changed_agents'])
                        print('FIRST_DIVERGENCE',json.dumps(finding,ensure_ascii=False),flush=True)
            self.current={}
            return output
        runner.policy.get_actions=get_actions

    def observe(self,kind,local):
        if not self.current:
            return
        frame=inspect.currentframe().f_back
        while frame is not None:
            caller=frame.f_locals
            if caller.get('self') is self.ac and 'agent_idx' in caller:
                break
            frame=frame.f_back
        if frame is None:
            raise RuntimeError('Cannot map raw scores to the original graph slots')
        agent=int(caller['agent_idx'])
        ids=(caller['role_batch_indices'] if kind=='resource' else
             caller['active_mask_i'].nonzero(as_tuple=False).flatten())
        ids=ids.detach().cpu().tolist()
        tensor=local['ptr'][:,0,:] if kind=='resource' else local['logits']
        targets=[(j,b) for j,b in enumerate(ids) if b in self.current]
        if targets:
            values=tensor.detach().cpu().numpy()
            for j,b in targets:
                self.current[b]['resource_raw' if kind=='resource' else 'planes'][agent]=values[j].copy()
        del frame,caller

    def begin(self,cases,tau):
        self.tau=tau;self.slots={i:c['name'] for i,c in enumerate(cases) if c['name'] in TARGETS}
        self.findings={};self.earliest_input_differences={};self.streams={};self.baselines={}
        directory=self.root/'traces';directory.mkdir(exist_ok=True)
        for name in self.slots.values():
            self.streams[name]=gzip.open(directory/f'{name}_tau_{tag(tau)}.pkl.gz','wb',compresslevel=1)
            if tau!=.3:self.baselines[name]=gzip.open(directory/f'{name}_tau_0p3.pkl.gz','rb')

    def end(self):
        for f in [*self.streams.values(),*self.baselines.values()]:f.close()
        self.current={};self.streams={};self.baselines={}


def self_check(study):
    import torch
    from onpolicy.algorithms.utils.ptr_actor import MaPtrNet,JointPairPtrActor
    torch.set_num_threads(1);torch.manual_seed(17)
    seen=[]
    observer=lambda kind,local:seen.append((kind,local['tau']))
    a=MaPtrNet(8,8);q=torch.randn(3,1,8);k=torch.randn(3,5,8);mask=torch.zeros(3,5,dtype=torch.bool)
    mask[:,4]=True
    original=MaPtrNet.dist;patched=observed_function(original,'prob = F.softmax(ptr / tau','resource',observer)
    b=JointPairPtrActor(16,8,pair_feature_dim=2)
    inputs=dict(query=torch.randn(3,1,16),op_nodes=torch.randn(3,2,8),site_nodes=torch.randn(3,4,8),
        op_valid_mask=torch.ones(3,2,dtype=torch.bool),site_valid_mask=torch.ones(3,2,4,dtype=torch.bool),
        deterministic=True,pair_features=torch.randn(3,2,4,2))
    original_plane=JointPairPtrActor.forward;patched_plane=observed_function(original_plane,'logits = logits / max(float(tau)','plane',observer)
    with torch.no_grad():
        for tau in TAUS:
            assert torch.equal(original(a,q,k,mask,tau),patched(a,q,k,mask,tau))
            x=original_plane(b,**inputs,tau=tau);y=patched_plane(b,**inputs,tau=tau)
            assert all(torch.equal(v,w) for v,w in zip(x,y))
    assert len(seen)==6
    print(json.dumps(dict(observer_ast_preserved=True,synthetic_cpu_bitwise_checks=6,passed=True)))


def run(study_path,output):
    s=read(study_path);base_driver=load_module(s['driver']['path'],'frozen_tau_study')
    s,m=base_driver.load_study(study_path)
    import numpy as np
    import torch
    import psutil
    from onpolicy.runner.shared.stage3_h3_frozen_engine import H3FrozenEngine,model_digest
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import Progress
    from onpolicy.utils.stage3_b0_single_model import CpuEnvironmentPool
    from onpolicy.utils.stage3_b_shared_b0 import cpus,record_summary
    from onpolicy.utils.stage3_h3_frozen import verify_manifest
    from onpolicy.utils.stage3_numerics import configure_runtime
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=s['gpu_uuid']:
        raise ValueError('Only the original physical GPU0 is authorized')
    if not set(os.sched_getaffinity(0)).issubset(cpus(s['cpu_affinity'])):
        raise ValueError('CPU affinity escaped the original validator allocation')
    free=int(subprocess.check_output(['nvidia-smi','--id='+s['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<6144 or psutil.virtual_memory().available<12*2**30:
        raise RuntimeError('Insufficient diagnostic admission headroom')
    verify_manifest(m)
    configure_runtime()
    batches=[s['cases'][start:start+12] for start in [12,48,108]]
    for cases in batches:
        for c in cases:
            for filename,checksum in c['files'].items():
                assert sha(Path(c['path'])/filename)==checksum
    recipe=copy.deepcopy(m['recipe'])
    recipe['gpu_headroom_mib']=torch.cuda.get_device_properties(0).total_memory/2**20-4096-1024
    class TemperatureEngine(H3FrozenEngine):
        diagnostic_tau=.3
        @contextmanager
        def evaluation_mode(self,decoder='H'):
            with super().evaluation_mode(decoder):
                self.policy.ac.tau=self.diagnostic_tau
                yield
    write(root/'manifest.json',dict(protocol='fixed12_temperature_first_divergence_v1',study_path=str(Path(study_path).resolve()),
        study_sha256=s['study_sha256'],checkpoint=s['models']['checkpoint']['checkpoint'],targets=list(TARGETS),
        temperatures=list(TAUS),batches=[[c['name'] for c in b] for b in batches],
        driver_sha256=sha(__file__),source_root=s['source_root'],cpu_affinity=s['cpu_affinity'],gpu_uuid=s['gpu_uuid'],
        torch_allocator_mib=4096,training_mutation=False,iga_solver_queries=0,started_unix=time.time()))
    started=time.monotonic();proofs=[]
    with Progress(root/'status.json',phase='initializing',protocol='fixed12_temperature_first_divergence_v1') as hb:
        runner=TemperatureEngine(m['frozen_manifest']['path'],config=recipe,
            runtime=dict(sampling_cpus=sorted(cpus(s['cpu_affinity'])),sampling_output=str(root/'sampling')),training=False)
        recorder=None
        try:
            runner.pool=CpuEnvironmentPool(12,timeout=recipe['ipc_timeout_seconds'],affinity=cpus(s['cpu_affinity']))
            weights=torch.load(base_driver.checked(s['models']['checkpoint']['checkpoint']),map_location='cpu',weights_only=False)['model']
            runner.policy.ac.load_state_dict(weights,strict=True);runner.policy_updates=12
            before=model_digest(runner.policy.ac)
            recorder=Recorder(runner,root,hb)
            for batch_index,cases in enumerate(batches):
                for tau in TAUS:
                    runner.cache.last_actor=None;runner.diagnostic_tau=tau
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    recorder.begin(cases,tau)
                    hb.update(phase='replay',batch_index=batch_index,tau=tau,case_names=[c['name'] for c in cases])
                    rows=runner.rollout(cases,[1]*12,deterministic=True,decoder='H',native=False,heartbeat=hb)
                    recorder.end()
                    expected={x['case_id']:x for x in read(Path(s['root'])/'results'/f'checkpoint_tau_{tag(tau)}.json')['rows']}
                    mismatches=[]
                    for row in rows:
                        for field in ('makespan','steps','actions_sha256','history_sha256'):
                            if row[field]!=expected[row['case_id']][field]:
                                mismatches.append(dict(case=Path(row['case_id']).name,field=field,expected=expected[row['case_id']][field],actual=row[field]))
                    proof=dict(batch_index=batch_index,tau=tau,cases=12,passed=not mismatches,mismatches=mismatches,
                        weights_unchanged=model_digest(runner.policy.ac)==before,
                        rows=[record_summary(row) for row in rows],peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                    write(root/'reproduction'/f'batch_{batch_index}_tau_{tag(tau)}.json',proof)
                    proofs.append(proof)
                    if mismatches or not proof['weights_unchanged']:
                        raise RuntimeError('Instrumented replay did not exactly reproduce the frozen trajectory')
                    hb.update(event='batch_verified',verified_groups=len(proofs),total_groups=9)
            findings=[read(p) for p in sorted((root/'findings').glob('*.json'))]
            assert len(findings)==8
            verify_manifest(m);base_driver.load_study(study_path)
            write(root/'completed.json',dict(completed=True,reproduced_case_episodes=108,verified_groups=9,
                first_divergences=8,weights_unchanged=True,seconds=time.monotonic()-started))
            hb.update(phase='completed',first_divergences=8,total_seconds=time.monotonic()-started)
        finally:
            if recorder is not None:recorder.end()
            runner.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path)
    args=p.parse_args();s=read(args.study)
    if args.mode=='check':
        sys.path.insert(0,s['source_root']);self_check(s)
    else:
        if args.output is None:raise ValueError('--output required')
        run(args.study,args.output)


if __name__=='__main__':
    main()
