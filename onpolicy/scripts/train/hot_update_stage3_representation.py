#!/usr/bin/env python3
"""Rolling checkpoint handoff; never inject code into a live Python process.

The old controller is reversibly suspended while its workers continue. After
the GPU gate, a new controller owns the shared validators and replaces each
trainer only at a fully published checkpoint with no later committed update.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.scripts.train.run_stage3_representation import Suite, verify
from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat
from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file
from onpolicy.utils.stage3_representation import (ARMS, external_resources, is_descendant,
    screen_candidates, risk_pass, endpoint_pass, schedule)
from onpolicy.utils.stage3_hot_update import prepare, validate_hot_update, numa_plan, resume_payload_allowed


class ProcessIdentity:
    """PID + birth token, checked on every signal (including cleanup)."""
    def __init__(self, pid, expected_command=None):
        self.pid = int(pid)
        self.ticks = self.stat()[19]
        self.command = Path(f"/proc/{self.pid}/cmdline").read_text().replace("\0", " ")
        if expected_command and expected_command not in self.command:
            raise ValueError(f"PID {pid} is not the expected study process")

    def stat(self):
        return Path(f"/proc/{self.pid}/stat").read_text().rsplit(")",1)[1].split()

    def alive(self):
        try:
            row = self.stat()
            return row[19] == self.ticks and row[0] not in ("Z","X")
        except FileNotFoundError:
            return False

    def send(self, sig, group=False):
        if not self.alive():
            return
        if group:
            if os.getpgid(self.pid) != self.pid:
                raise ValueError("Refuse to signal a process group not led by the recorded worker")
            os.killpg(self.pid, sig)
        else:
            os.kill(self.pid, sig)

    def pause(self):
        self.send(signal.SIGSTOP)
        deadline = time.time()+5
        while self.alive() and self.stat()[0] not in ("T","t"):
            if time.time() > deadline:
                raise TimeoutError("Could not suspend the recorded process")
            time.sleep(.01)

    def terminate(self):
        self.send(signal.SIGTERM, group=True)
        self.send(signal.SIGCONT, group=True)
        deadline = time.time()+30
        while self.alive() and time.time() < deadline:
            time.sleep(.05)
        if self.alive():
            self.send(signal.SIGKILL, group=True)
            raise TimeoutError("Owned worker did not close after handoff; artifacts retained")


def latest_committed(output):
    updates = sorted((Path(output)/"updates").glob("group_*.json"))
    return read_json(updates[-1])["training_episodes"] if updates else 0


def checkpoint_candidate(output):
    completed = latest_committed(output)
    candidate = Path(output)/f"models/episodes_{completed:06d}.pt"
    return candidate if 0 < completed < 960 and completed % 240 == 0 and candidate.is_file() else None


def set_process_tree_affinity(process, cpus):
    if not process.alive():
        return
    # CPU placement changes only; no global thread-count or precision change.
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or not is_descendant(int(path.name), process.pid):
            continue
        try:
            for thread in (path/"task").iterdir():
                os.sched_setaffinity(int(thread.name), cpus)
        except (FileNotFoundError, ProcessLookupError):
            pass


class RollingSuite(Suite):
    def __init__(self, manifest, manifest_path):
        super().__init__(manifest, manifest_path=manifest_path)
        self.upgrade = manifest["hot_update"]
        self.attempt = Path(self.upgrade["attempt_dir"])
        self.parent = validate_hot_update(manifest)
        self.legacy = ProcessIdentity(self.upgrade["legacy_controller_pid"], self.upgrade["parent_manifest"])
        actual_pid = subprocess.check_output(["systemctl","--user","show",self.upgrade["legacy_service"],
            "-p","MainPID","--value"],text=True).strip()
        if actual_pid != str(self.legacy.pid):
            raise ValueError("Legacy service controller PID changed")
        self.plan = numa_plan(manifest["resources"]["core_groups"],manifest["resources"]["gpus"])
        topology = subprocess.check_output(["nvidia-smi","topo","-m"],text=True)
        for gpu in range(8):
            row = next((line.split() for line in topology.splitlines() if line.startswith(f"GPU{gpu}\t") or line.startswith(f"GPU{gpu} ")),None)
            if not row or len(row) < 11 or row[10] != str(gpu//4):
                raise ValueError("GPU NUMA topology differs from reviewed mapping")
        self.old_trainers = {}
        for arm in ARMS:
            output = self.root/f"train/{arm}_to960"
            state = read_json(output/"status.json")
            process = ProcessIdentity(state["pid"],str(output))
            if not is_descendant(process.pid,self.legacy.pid):
                raise ValueError("Trainer is not owned by the recorded legacy study")
            self.old_trainers[arm] = dict(process=process,output=output)
        self.old_validators = []
        for path in (self.root/"validator/workers").glob("*/status.json"):
            state = read_json(path)
            try:
                process = ProcessIdentity(state["pid"],str(path.parent))
            except FileNotFoundError:
                continue
            if process.alive() and is_descendant(process.pid,self.legacy.pid):
                self.old_validators.append(process)
        if len(self.old_validators) != 2:
            raise ValueError("Expected the existing two-worker shared validator pool")
        self.irreversible = False
        self.outputs = {}

    def resources(self):
        external = external_resources(self.m["resources"]["gpus"],os.getpid())
        external = [r for r in external if not (self.legacy.alive() and is_descendant(r["pid"],self.legacy.pid))]
        if external:
            raise RuntimeError(f"External GPU jobs appeared; will not preempt them: {external}")
        atomic_json(self.root/"resource_leases.json",dict(updated_unix=time.time(),plan=self.plan,
            external_reservations=[],rolling_from_controller=self.legacy.pid if self.legacy.alive() else None))
        return self.plan

    def canary(self, hb):
        folder = self.attempt/"gpu_canary"
        folder.mkdir(exist_ok=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.m["resources"]["gpus"][0]["uuid"],
            HKBZ_STAGE3_GPU_TEST="1", HKBZ_STAGE3_CANARY_REPORT_DIR=str(folder),
            HKBZ_STAGE3_TEST_PARENT_MANIFEST=self.upgrade["parent_manifest"],
            CUBLAS_WORKSPACE_CONFIG=":4096:8")
        command = [self.m["execution"]["python"],"-m","pytest","-q","-p","no:cacheprovider",
            str(ROOT/"onpolicy/envs/HKBZ/test/test_stage3_performance.py"),"-k","cuda"]
        with (folder/"pytest.log").open("x") as log:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        token = ProcessIdentity(child.pid)
        begin = time.time()
        try:
            while child.poll() is None:
                hb.update(event="gpu_equivalence_canary",canary_pid=child.pid,elapsed_seconds=time.time()-begin)
                if time.time()-begin > 1800:
                    raise TimeoutError("GPU equivalence canary timed out")
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"GPU equivalence canary failed; see {folder/'pytest.log'}")
        finally:
            if child.poll() is None:
                token.terminate()
        rows = [read_json(folder/f"E{i}_cuda_0.json") for i in range(4)]
        if not all(r["passed"] and r["peak_reserved_gib"] < 10 for r in rows):
            raise ValueError("GPU canary correctness/memory gate failed")
        atomic_json(folder/"result.json",dict(passed=True,results=rows,command=command,
            production_precision_unchanged=True),overwrite=False)

    def replace_validators(self, hb):
        begin = time.time()
        while True:
            with (self.queue.root/"claim.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                # Claim lock closes the idle -> new request race.
                if not list((self.queue.root/"running").glob("*.json")):
                    for process in self.old_validators:
                        process.pause()
                    self.irreversible = True
                    for process in self.old_validators:
                        process.terminate()
                    break
            if time.time()-begin > self.m["timeouts_seconds"]["validator_request"]:
                raise TimeoutError("Could not drain old validators at a request boundary")
            hb.update(event="draining_old_validators",pending_validations=self.queue.pending_count())
            time.sleep(2)
        for i,placement in enumerate(self.plan["validators"]):
            self.start(f"pool{i}_{self.upgrade['attempt_id']}","validator",placement)

    def ensure_due_validation(self, checkpoint, arm, episodes):
        if episodes % 480:
            return
        request = f"{arm}_e{episodes:06d}"
        matches = [self.queue.root/state/f"{request}.json" for state in ("pending","running","results","failed")]
        existing = [path for path in matches if path.exists()]
        if existing:
            # Publication may move pending -> running -> results while read.
            result = self.queue.poll(request)
            if result is not None and result["checkpoint_sha256"] != digest_file(checkpoint):
                raise ValueError("Existing validation refers to a different checkpoint")
            return
        from onpolicy.scripts.train.stage3_representation_worker import submit
        submit(self.parent,checkpoint,arm,episodes,ARMS[arm]["encoder"])

    def migrate(self, arm, record, hb):
        process,output = record["process"],record["output"]
        if not process.alive():
            result = read_json(output/"result.json")
            if not result.get("completed") and not result.get("guard"):
                raise RuntimeError(f"Legacy worker {arm} exited without a valid result")
            self.outputs[arm] = output
            return True
        checkpoint = checkpoint_candidate(output)
        if checkpoint is None:
            return False
        process.pause()
        accepted = False
        try:
            episodes = int(checkpoint.stem.split("_")[-1])
            if latest_committed(output) != episodes:
                return False  # Never rewind a later completed update.
            import torch
            payload = torch.load(checkpoint,map_location="cpu",weights_only=False)
            plan = schedule(self.m["splits"]["train_pilot120"],self.m["training"]["seed"],ARMS[arm]["batch"])
            resume_payload_allowed(payload,self.m,arm,960,plan)
            required = ("model","actor_optim","critic_optim","role_value_normalizers","rng_torch",
                "rng_numpy","rng_python","rng_cuda","elite_buffer","elite_usage","auxiliary_steps","validation_requests")
            if any(key not in payload for key in required) or payload["rng_cuda"] is None:
                raise ValueError("Handoff checkpoint is not a complete CUDA training state")
            del payload
            snapshot = read_json(output/"status.json")
            self.ensure_due_validation(checkpoint,arm,episodes)
            atomic_json(self.attempt/f"handoffs/{arm}.json",dict(arm=arm,checkpoint=str(checkpoint),
                checkpoint_sha256=digest_file(checkpoint),training_episodes=episodes,
                last_committed_episodes=latest_committed(output),old_pid=process.pid,old_start_ticks=process.ticks,
                old_status=snapshot,committed_updates_discarded=0,
                in_flight_work="Any uncommitted work after this checkpoint is retained as diagnostic artifacts only",
                created_unix=time.time()),overwrite=False)
            process.terminate()
            accepted = True
            index = list(ARMS).index(arm)
            name = f"{arm}_to960_{self.upgrade['attempt_id']}"
            self.start(name,"train",self.plan["trainers"][index],arm=arm,until=960,resume_checkpoint=str(checkpoint))
            self.outputs[arm] = self.root/"train"/name
            hb.update(event="arm_handed_off",last_handoff_arm=arm,last_handoff_episodes=episodes)
            return True
        finally:
            if not accepted:
                process.send(signal.SIGCONT)

    def retire_controller(self):
        # Every old worker is already retired/completed. A new service owns
        # new workers, so systemd's old cgroup cleanup cannot kill them.
        self.legacy.send(signal.SIGTERM)
        self.legacy.send(signal.SIGCONT)
        deadline = time.time()+45
        while self.legacy.alive() and time.time() < deadline:
            time.sleep(.1)
        if self.legacy.alive():
            raise TimeoutError("Legacy controller did not finish its owned-only cleanup")

    def finish_study(self, hb):
        while any(j["phase"] == "train" and j["process"].poll() is None for j in self.jobs.values()):
            self.check(hb); self.resources()
            hb.update(phase="screen_960",event="training",screen_outputs={a:str(p) for a,p in self.outputs.items()})
            time.sleep(10)
        self.check(hb)
        screen = {arm:self.evaluation(arm,960,hb) for arm,output in self.outputs.items()
            if read_json(output/"result.json").get("completed")}
        chosen = screen_candidates(screen)
        atomic_json(self.root/"screen_admission.json",dict(results=screen,selected=chosen,
            threshold=.005,max_extensions=2,created_unix=time.time()),overwrite=False)
        outcomes = {}
        for endpoint in (1440,1920):
            if not chosen:
                break
            hb.update(phase=f"pilot_to_{endpoint}")
            next_outputs = {}
            for arm in chosen:
                name = f"{arm}_to{endpoint}_{self.upgrade['attempt_id']}"
                previous = endpoint-480
                checkpoint = self.outputs[arm]/f"models/episodes_{previous:06d}.pt"
                placement = self.plan["trainers"][list(ARMS).index(arm)]
                self.start(name,"train",placement,arm=arm,until=endpoint,resume_checkpoint=str(checkpoint))
                next_outputs[arm] = self.root/"train"/name
            while any(j["phase"] == "train" and j["process"].poll() is None for j in self.jobs.values()):
                self.check(hb); self.resources(); time.sleep(10)
            self.check(hb)
            survivors = []
            self.outputs.update(next_outputs)
            for arm in chosen:
                if not read_json(self.outputs[arm]/"result.json").get("completed"):
                    continue
                result = self.evaluation(arm,endpoint,hb)
                outcomes.setdefault(arm,[]).append(result)
                if result["summary"]["gain_fraction"] >= .01 and risk_pass(result["summary"]):
                    survivors.append(arm)
            atomic_json(self.root/f"pilot_admission_{endpoint}.json",dict(results=outcomes,
                eligible_next_chunk=survivors if endpoint == 1440 else [],
                failed_1440_not_called_completed_1920=True),overwrite=False)
            chosen = survivors
        atomic_json(self.root/"result.json",dict(completed=True,screen=screen,pilot=outcomes,
            passed_arms=[a for a,r in outcomes.items() if endpoint_pass(r)],single_seed_only=True,
            finalblind_opened=False,automatic_extension_4800=False,
            execution_update=self.upgrade["attempt_id"],evidence_scope="single-seed screening, not cross-seed robustness"),overwrite=False)
        atomic_json(self.root/"validator/STOP",dict(reason="all requested validations drained"),overwrite=False)
        while any(j["phase"] == "validator" and j["process"].poll() is None for j in self.jobs.values()):
            self.check(hb); time.sleep(2)

    def run(self):
        verify(self.m)
        try:
            self.legacy.pause()
            for name in ("status.json","resource_leases.json","active_execution.json"):
                if (self.root/name).exists():
                    target = self.attempt/"pre_update"/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(self.root/name,target)
            with ProgressHeartbeat(self.root/"status.json",phase="screen_960",active_manifest=str(self.manifest_path),
                    hot_update_attempt=self.upgrade["attempt_id"],legacy_controller_pid=self.legacy.pid) as hb:
                atomic_json(self.attempt/"started.json",dict(pid=os.getpid(),created_unix=time.time(),
                    manifest=str(self.manifest_path)),overwrite=False)
                self.resources()
                self.canary(hb)
                self.resources()
                for index,record in enumerate(self.old_trainers.values()):
                    set_process_tree_affinity(record["process"],self.plan["trainers"][index]["cpus"])
                os.sched_setaffinity(0,self.plan["controller"])
                self.replace_validators(hb)
                atomic_json(self.root/"active_execution.json",dict(manifest=str(self.manifest_path),
                    code_root=str(ROOT),hot_update_attempt=self.upgrade["attempt_id"],
                    parent_manifest=self.upgrade["parent_manifest"],started_unix=time.time()))
                waiting = dict(self.old_trainers)
                last = 0
                while waiting:
                    for arm,record in list(waiting.items()):
                        if self.migrate(arm,record,hb):
                            del waiting[arm]
                    if time.time()-last > 10:
                        self.check(hb)
                        self.resources()
                        hb.update(event="rolling_checkpoint_handoff",waiting_for_checkpoint=list(waiting),
                            screen_outputs={a:str(p) for a,p in self.outputs.items()},
                            legacy_training_episodes={a:latest_committed(r["output"]) for a,r in waiting.items()})
                        last = time.time()
                    if time.time()-read_json(self.attempt/"started.json")["created_unix"] > self.m["timeouts_seconds"]["train"]:
                        raise TimeoutError("Legacy checkpoint handoff exceeded training hard timeout")
                    time.sleep(1)
                self.retire_controller()
                with (self.root/"controller.lock").open("a") as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    atomic_json(self.attempt/"handoff_complete.json",dict(completed=True,
                        outputs={a:str(p) for a,p in self.outputs.items()},created_unix=time.time()),overwrite=False)
                    self.finish_study(hb)
        except BaseException:
            atomic_json(self.attempt/"activation_failure.json",dict(traceback=traceback.format_exc(),
                irreversible=self.irreversible,created_unix=time.time()),overwrite=False)
            if not self.irreversible:
                # Canary failure has not terminated any old worker. Resume the
                # old controller and restore its dashboard without rewinding.
                try:
                    for name in ("status.json","resource_leases.json","active_execution.json"):
                        if (self.attempt/"pre_update"/name).exists():
                            shutil.copy2(self.attempt/"pre_update"/name,self.root/name)
                finally:
                    self.legacy.send(signal.SIGCONT)
            else:
                # Fail closed, just like the original suite: no unattended
                # orphan trainers or implicit retries after a broken migration.
                self.retire_controller()
            raise
        finally:
            self.stop_owned()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--parent",required=True,type=Path)
    p.add_argument("--attempt-id",required=True)
    p.add_argument("--verification",required=True,type=Path)
    p.add_argument("--legacy-service",required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest",required=True,type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.parent,args.attempt_id,ROOT,args.verification,args.legacy_service),flush=True)
        return
    manifest = read_json(args.manifest)
    attempt = Path(manifest["hot_update"]["attempt_dir"])
    with (Path(manifest["root"])/"hot_update.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (attempt/"started.json").exists():
            raise FileExistsError("No implicit retry of a hot-update activation")
        def stop(signum,frame):
            raise KeyboardInterrupt(f"Hot-update stop signal {signum}; owned study only")
        signal.signal(signal.SIGTERM,stop)
        RollingSuite(manifest,args.manifest).run()


if __name__ == "__main__":
    main()
