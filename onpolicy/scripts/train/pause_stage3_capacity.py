#!/usr/bin/env python3
"""Reversible pause of the explicit live Stage3 service; never kills workers.

Pause only controller/worker mains. Environment children may finish outstanding
IPC replies and then wait, avoiding expiration of an unanswered poll on resume.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(pid):
    row = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": int(pid), "start_ticks": row[19], "state": row[0],
            "command": Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ")}


def send(record, sig):
    actual = identity(record["pid"])
    if any(actual[key] != record[key] for key in ("pid", "start_ticks", "command")):
        raise ValueError("Recorded process identity changed; refusing signal")
    os.kill(record["pid"], sig)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pause", "resume", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    unit = "hkbz-s3repr-probe-guard-resume-20260908-r1.service"
    if args.action == "pause":
        if args.receipt.exists():
            raise FileExistsError("Pause receipt already exists")
        pid = int(subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "MainPID", "--value"], text=True))
        controller = identity(pid)
        if "resume_stage3_after_probe.py" not in controller["command"] or str(args.root) not in controller["command"]:
            raise ValueError("Unexpected live controller")
        paused = []
        try:
            send(controller, signal.SIGSTOP)
            paused.append(controller)
            time.sleep(.1)
            status = json.loads((args.root / "status.json").read_text())
            if status["pid"] != pid or status["status"] != "running":
                raise ValueError("Study dashboard is not this running controller")
            workers = []
            for key, code in status["worker_exit_codes"].items():
                if code is not None:
                    continue
                phase, name = key.split("/", 1)
                output = args.root / ("validator/workers" if phase == "validator" else phase) / name
                state = json.loads((output / "status.json").read_text())
                record = identity(state["pid"])
                if str(output) not in record["command"]:
                    raise ValueError("Unexpected worker command")
                record.update(task=key, output=str(output), status=state)
                send(record, signal.SIGSTOP)
                paused.append(record)
                workers.append(record)
            if sum(r["task"].startswith("train/") for r in workers) != 8:
                raise ValueError("Expected eight running training arms")
            time.sleep(.2)
            if any(identity(r["pid"])["state"] not in ("T", "t") for r in paused):
                raise RuntimeError("Some main process did not pause")
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt = {"unit": unit, "root": str(args.root), "paused_unix": time.time(),
                "controller": controller, "workers": workers, "prior_status": status,
                "method": "SIGSTOP mains only; IPC children finish replies and idle",
                "training_state_retained_in_memory": True, "committed_work_discarded": 0}
            write_json(args.receipt, receipt)
            write_json(args.root / "status.json", {**status, "status": "paused", "event": "user_capacity_retest_pause",
                       "paused_unix": receipt["paused_unix"], "pause_receipt": str(args.receipt)})
        except BaseException:
            for record in reversed(paused):
                send(record, signal.SIGCONT)
            raise
    else:
        receipt = json.loads(args.receipt.read_text())
        for record in receipt["workers"] + [receipt["controller"]]:
            if args.action == "resume":
                send(record, signal.SIGCONT)
            print(record.get("task", "controller"), identity(record["pid"]))


if __name__ == "__main__":
    main()
