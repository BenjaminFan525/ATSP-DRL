#!/usr/bin/env python3
"""Record source-bound CPU regressions before creating an execution snapshot."""
import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from onpolicy.utils.stage3_hot_update import OVERLAYS
from onpolicy.utils.stage3_research import digest_file, atomic_json

TESTS = (
    "test_stage3_performance.py", "test_stage3_hot_update.py",
    "test_stage3_representation.py", "test_stage3_representation_orchestration.py",
    "test_stage3_diagnostic_protocol.py", "test_stage3_diagnostic_resume.py",
    "test_stage3_local_exploration.py", "test_stage3_local_resources.py",
    "test_stage3_sampling_audit.py", "test_joint_training.py",
    "test_stage1_learning_baselines.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    hashes = {relative:digest_file(ROOT/relative) for relative in OVERLAYS}
    for relative in OVERLAYS:
        if relative.endswith(".py"):
            ast.parse((ROOT/relative).read_text(),filename=relative)
        elif relative.endswith(".sh"):
            subprocess.run(["bash","-n",str(ROOT/relative)],check=True)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    command = [sys.executable,"-m","pytest","-q","-p","no:cacheprovider"] + [
        str(ROOT/"onpolicy/envs/HKBZ/test"/name) for name in TESTS]
    begin = time.time()
    with (args.output/"pytest.log").open("x") as log:
        result = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    current = {relative:digest_file(ROOT/relative) for relative in OVERLAYS}
    passed = result.returncode == 0 and hashes == current
    atomic_json(args.output/"result.json",dict(passed=passed,overlay_hashes=hashes,command=command,
        exit_code=result.returncode,seconds=time.time()-begin,created_unix=time.time(),
        source_unchanged=hashes==current,gpu_excluded=True,
        material_passport=dict(origin_skill="academic-research-suite/experiment-agent",origin_mode="run",
            origin_date="2026-09-08",verification_status="CPU verified" if passed else "failed",
            version_label="stage3_execution_perf_v1")),overwrite=False)
    print(args.output/"result.json",flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
