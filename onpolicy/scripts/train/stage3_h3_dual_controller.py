"""Two bounded training processes on one GPU, with epoch barriers for evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from onpolicy.scripts.train.run_stage3_h3_continuation import Controller, WORKER
from onpolicy.utils.stage3_h3_continuation import ARMS, extend_arms
from onpolicy.utils.stage3_h3_frozen import bind, checked
from onpolicy.utils.stage3_research import atomic_json, read_json


class DualController(Controller):
    poll_seconds = 2

    def __init__(self, *args):
        super().__init__(*args)
        self.children = {}

    def stop_child(self):
        for process in self.children.values():
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for process in self.children.values():
            try: process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        self.children.clear()
        super().stop_child()

    def sample(self, label):
        import psutil
        gpu = subprocess.check_output(['nvidia-smi','--id='+self.m['resources']['gpu_uuid'],
            '--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        processes = {}
        for arm, child in self.children.items():
            try:
                p = psutil.Process(child.pid)
                processes[arm] = dict(pid=p.pid,rss_bytes=p.memory_info().rss,
                    cpu_affinity=p.cpu_affinity(),children=len(p.children(recursive=True)))
            except psutil.NoSuchProcess:
                pass
        cgroup = {}
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                base=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
                for name in ('memory.current','memory.peak','memory.swap.current'):
                    if (base/name).exists():cgroup[name]=int((base/name).read_text())
                for name in ('memory.high','memory.max','memory.swap.max','memory.events','memory.pressure'):
                    if (base/name).exists():cgroup[name]=(base/name).read_text().strip()
        with (self.root/'resource_samples.jsonl').open('a') as f:
            f.write(json.dumps(dict(unix=time.time(),phase=label,gpu=gpu,processes=processes,
                cgroup=cgroup,host_available_bytes=psutil.virtual_memory().available))+'\n')

    def jobs(self, entries, label, limit=12*3600):
        """Start every peer before waiting; a failed peer cancels unfinished peers."""
        if not entries:
            return 0.
        if self.children:
            raise RuntimeError('Concurrent job groups must not overlap')
        if len(entries)>2 or len({e[0] for e in entries})!=len(entries):
            raise ValueError('At most one process per temperature arm')
        self.wait_gpu()
        started=time.time(); last_sample=0.; logs=[]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=self.m['resources']['gpu_uuid'],
            HKBZ_STAGE3_WORKSPACE_ROOT=self.m['workspace_root'],PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0')
        try:
            for arm,command,output in entries:
                output.mkdir(parents=True,exist_ok=True)
                log=(output/'worker.log').open('a');logs.append(log)
                self.children[arm]=subprocess.Popen(command,cwd=self.m['source_root'],env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                codes={a:p.poll() for a,p in self.children.items()}
                if any(c is not None and c!=0 for c in codes.values()):
                    raise RuntimeError(f'Parallel phase {label} failed: {codes}')
                if all(c is not None for c in codes.values()):
                    break
                self.status(status='running',phase=label,phase_started_unix=started,
                    child_pids={a:p.pid for a,p in self.children.items()},
                    outputs={a:str(o) for a,_,o in entries},resources=self.m['resources'])
                if time.time()-last_sample>=30:
                    self.sample(label);last_sample=time.time()
                if (time.time()-started>limit or
                        time.time()-self.started>self.m['budget']['max_wall_seconds']):
                    raise TimeoutError('Parallel phase reached its wall budget: '+label)
                time.sleep(self.poll_seconds)
            self.children.clear()
            return time.time()-started
        except BaseException:
            self.stop_child();raise
        finally:
            for log in logs:log.close()

    def entry(self,arm,item,mode,output,extra=()):
        # Bind before Python imports: native libraries may create helper threads
        # before the worker's main-thread sched_setaffinity() call.
        return (arm,['/usr/bin/taskset','--cpu-list',item[1]['recipe']['trainer_cpus'],
                     self.m['python'],'-B','-u',str(Path(self.m['source_root'])/WORKER),
                     mode,str(item[0]),'--output',str(output),*extra],output)

    def canaries(self,arms):
        for mode,complete in [('canary','expected.pt'),('canary-resume','passed.json')]:
            entries=[]
            for arm,item in arms.items():
                out=Path(item[1]['root'])/'canary'
                if not (out/complete).exists():entries.append(self.entry(arm,item,mode,out))
            self.jobs(entries,'dual_'+mode,limit=3600)
        for arm,item in arms.items():
            proof=read_json(Path(item[1]['root'])/'canary/passed.json')
            if not proof['passed'] or proof['manifest_sha256']!=item[1]['manifest_sha256']:
                raise ValueError('Invalid parallel full-state restore proof: '+arm)

    def capacity(self,arms):
        """Identical diagnostic windows, first singly and then simultaneously."""
        path=self.root/'dual_capacity.json'
        if path.exists():
            record=read_json(path)
            if record['manifest_sha256']!=self.m['manifest_sha256'] or not record['passed']:
                raise ValueError('Invalid dual capacity receipt')
            for r in record['results']:checked(r)
            return
        records=[]
        for mode in ('single','parallel'):
            pending=[]
            for arm,item in arms.items():
                out=self.root/'capacity'/mode/arm
                if not (out/'result.json').exists():pending.append(self.entry(arm,item,'capacity',out))
                records.append(bind(out/'result.json') if (out/'result.json').exists() else out/'result.json')
            if mode=='single':
                for entry in pending:self.jobs([entry],'capacity_single_'+entry[0],limit=1800)
            else:self.jobs(pending,'capacity_parallel',limit=1800)
        results=[bind(x) if isinstance(x,Path) else x for x in records]
        values=[read_json(checked(x)) for x in results]
        if any(not x['passed'] for x in values):raise ValueError('Dual capacity admission failed')
        serial=sum(x['measured_seconds'] for x in values[:2])
        parallel=max(x['ended_unix'] for x in values[2:])-min(x['started_unix'] for x in values[2:])
        atomic_json(path,dict(passed=True,manifest_sha256=self.m['manifest_sha256'],results=results,
            serial_compute_seconds=serial,parallel_compute_span_seconds=parallel,
            paired_window_throughput_ratio=serial/parallel,
            scope=f'Native {self.m["recipe"]["rollout_workers"]}-slot inference and '
                  f'{self.m["recipe"]["microbatch"]}-trajectory TBPTT windows; not a complete epoch speed claim'),overwrite=False)

    def train_stage(self,arms,live,first,last,parent):
        for epoch in range(first,last+1):
            entries=[]
            for arm in live:
                root=Path(arms[arm][1]['root'])
                if not (root/'commits'/f'epoch_{epoch:04d}.json').exists():
                    if last==2 and time.time()-self.started>self.m['budget']['screen_wall_seconds']:
                        raise TimeoutError('Paired screen reached its 36-hour budget')
                    out=self.attempt/arm/f'epoch_{epoch:04d}'
                    entries.append(self.entry(arm,arms[arm],'train',out,('--epoch',str(epoch))))
            self.jobs(entries,f'dual_epoch_{epoch:04d}',limit=18*3600)
            # Both GPU training processes exit before either fixed-12 evaluator
            # starts. Existing committed epochs are never trained a second time.
            live=[a for a in live if self.train_to(a,arms[a],epoch,parent)]
            if not live:break
        return live

    def run(self):
        if self.m['recipe'].get('memory_protection') == 'disabled':
            from onpolicy.utils.stage3_memory_runtime import verify_unprotected_memory
            receipt = verify_unprotected_memory()
            atomic_json(self.attempt/'memory_runtime.json', receipt, overwrite=False)
        parent=self.baseline()
        arms={a:self.arm_manifest(a,parent) for a in ARMS}
        self.canaries(arms)
        self.capacity(arms)
        live=self.train_stage(arms,list(ARMS),1,2,parent)
        live=self.train_stage(arms,live,3,4,parent)
        summaries={a:read_json(self.root/'arms'/a/'validation/epoch_0004.json')['versus_parent'] for a in live}
        active=extend_arms(summaries)
        decision=dict(stage=4,extend_arms=active,summaries=summaries,
            rule='validation mean gain >=0.3%, stress nonworse and risk gates',solver_queries=0)
        path=self.root/'promotion_epoch4.json'
        if path.exists():
            if read_json(path)!=decision:raise ValueError('Promotion decision changed')
        else:atomic_json(path,decision,overwrite=False)
        self.train_stage(arms,active,5,8,parent)
        self.finish(parent,active)
