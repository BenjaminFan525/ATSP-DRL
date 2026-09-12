"""Immutable single-seed 4x2 study, gates and shared multi-consumer queue."""
from __future__ import annotations
from collections import Counter
import fcntl
from pathlib import Path
import os
import subprocess

from onpolicy.utils.stage3_research import (digest_json, trajectory_seed, safety_gate,
                                          EvaluationQueue)

ARMS = {f"{encoder}_{batch}": {"encoder": encoder, "batch": batch}
        for encoder in ("E0", "E1", "E2", "E3") for batch in ("T0", "T1")}


def schedule(cases, seed, batch, passes=2):
    """Matched exposure in every 32-trajectory macroblock, fresh 8/update."""
    if batch not in ("T0", "T1") or len(cases) % 4:
        raise ValueError("Need T0/T1 and a case count divisible by four")
    result = []
    for visit in range(passes):
        ordered = sorted(cases, key=lambda c: digest_json([seed, "case-order", visit, c["content_sha256"]]))
        for first in range(0, len(ordered), 4):
            block = ordered[first:first+4]
            for update in range(4):
                pairs = ([(block[update], replica) for replica in range(8)] if batch == "T0"
                         else [(case, 2*update + replica) for case in block for replica in range(2)])
                result.append({"visit": visit, "macroblock": visit*len(cases)//4 + first//4,
                    "cases": [case for case, _ in pairs],
                    "seeds": [trajectory_seed(seed, case["content_sha256"], visit, replica)
                              for case, replica in pairs]})
    return result


def risk_pass(summary):
    return safety_gate(summary) and summary["regression_over_5pct_fraction"] <= .05


def screen_candidates(results):
    eligible = [(name, row) for name, row in results.items()
                if row["training_episodes"] == 960 and row["summary"]["gain_fraction"] >= .005
                and risk_pass(row["summary"])]
    return [name for name, _ in sorted(eligible, key=lambda item:
            (-item[1]["summary"]["gain_fraction"], item[0]))[:2]]


def endpoint_pass(results):
    by_episode = {r["training_episodes"]: r["summary"] for r in results}
    return all(e in by_episode and by_episode[e]["gain_fraction"] >= .01
               and risk_pass(by_episode[e]) for e in (1440, 1920))


class RepresentationQueue(EvaluationQueue):
    @staticmethod
    def identity(request):
        return digest_json({k: request[k] for k in ("checkpoint_sha256", "cases_sha256",
            "contract_sha256", "code_sha256", "tau", "seed", "evaluation_protocol",
            "training_exploration", "representation", "protocol_sha256")})

    def claim(self):
        # Serialize the short rename scan, NOT evaluations. Old single-worker
        # fail_interrupted() must never be called by individual pool workers.
        with (self.root / "claim.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return super().claim()

    def pending_count(self, arm=None):
        prefix = f"{arm}_" if arm else ""
        return sum(1 for state in ("pending", "running")
                   for path in (self.root/state).glob(f"{prefix}*.json"))


def cpu_list(text):
    cpus = set()
    for part in text.strip().split(","):
        if not part:
            continue
        ends = part.split("-")
        cpus.update(range(int(ends[0]), int(ends[-1])+1))
    return cpus


def cpu_text(cpus):
    return ",".join(str(c) for c in sorted(cpus))


def discover_topology():
    output = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
    cores = {}
    for line in output.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, socket, node = map(int, line.split(","))
        cores.setdefault((node, socket, core), []).append(cpu)
    return [sorted(cores[k]) for k in sorted(cores)]


def allocate_resources(core_groups, excluded_cpus, free_gpus):
    available = [c for c in core_groups if not set(c) & set(excluded_cpus)]
    # Keep all SMT siblings together. All 64 physical cores => 2+6+8*7.
    if len(available) < 15 or not free_gpus:
        return {"controller": [], "validators": [], "trainers": []}
    controller = sum(available[:2], [])
    validators = [{"cpus": sum(available[2+3*i:5+3*i], []), "gpu": free_gpus[i % len(free_gpus)],
                   "cuda_memory_fraction": .15} for i in range(min(2, len(free_gpus)))]
    rest = available[8:]
    slots = min(len(free_gpus), len(rest)//7, 8)
    trainers = [{"gpu": gpu, "cpus": sum(rest[7*i:7*(i+1)], []), "cuda_memory_fraction": .65}
                for i, gpu in enumerate(free_gpus[:slots])]
    return {"controller": controller, "validators": validators, "trainers": trainers}


def is_descendant(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            pid = int(stat[1])
        except (OSError, ValueError, IndexError):
            break
    return False


def external_resources(gpus, owner_pid, reservations=()):
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                      "--format=csv,noheader,nounits"], text=True)
    by_uuid = {gpu["uuid"]: gpu["index"] for gpu in gpus}
    observed = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        uuid, pid = (v.strip() for v in line.split(","))
        if uuid in by_uuid and not is_descendant(int(pid), owner_pid):
            observed[int(pid)] = by_uuid[uuid]
    # Keep reservations across idle CUDA intervals, until the original PID exits.
    for row in reservations:
        try:
            current = Path(f"/proc/{row['pid']}/stat").read_text().rsplit(")", 1)[1].split()[19]
            if current == row["start_ticks"]:
                observed[row["pid"]] = row["gpu"]
        except (OSError, IndexError):
            pass
    rows = []
    for pid, gpu in observed.items():
        try:
            rows.append({"pid": pid, "gpu": gpu, "cpus": sorted(os.sched_getaffinity(pid)),
                "start_ticks": Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19],
                "command": Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ")})
        except (OSError, IndexError):
            continue
    return rows
