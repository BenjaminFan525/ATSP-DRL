"""Launch only this approved reproduction as an isolated user-owned system service."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

RUN = Path(__file__).resolve().parent
UNIT = 'hkbz-stage2-h3f4-valid120-parallel3-20260920-r2-gpu0.service'


def main():
    manifest_path = RUN / 'manifest.json'
    if (RUN / 'launch_attempt.json').exists() or (RUN / 'suite_status.json').exists():
        raise RuntimeError('This launch has already been attempted; preserve its evidence.')
    preflight = json.loads((RUN / 'preflight.json').read_text())
    assert preflight['passed'] and set(preflight['parents']) == {'1', '2', '3'}
    manifest = json.loads(manifest_path.read_text())
    spec = json.loads((RUN / 'launch_spec.json').read_text())
    for filename, digest in spec['files'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
    command = [
        'systemd-run', '--system', '--no-ask-password', '--unit=' + UNIT,
        '--description=Stage2 H3F4 immediate seed3 and Validation120 boundary migration GPU0',
        '--property=Type=exec', '--property=User=fanyx', '--property=Group=1001',
        '--property=CPUAffinity=0-31 64-95', '--property=AllowedCPUs=0-31 64-95',
        '--property=MemoryMax=120G', '--property=MemorySwapMax=16G',
        '--property=TasksMax=1024', '--property=LimitNOFILE=1048576',
        '--property=KillMode=control-group', '--property=OOMPolicy=stop',
        '--property=TimeoutStopSec=60', '--property=RuntimeMaxSec=49h',
        '--property=WorkingDirectory=' + str(RUN / 'source'),
        '--property=StandardOutput=append:' + str(RUN / 'controller.log'),
        '--property=StandardError=inherit',
    ]
    environment = {
        'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(RUN / 'source'),
        'MPLCONFIGDIR': str(RUN / 'runtime/matplotlib'),
        'WANDB_MODE': 'disabled', 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1',
    }
    command.extend('--setenv=' + key + '=' + value for key, value in environment.items())
    python = next(iter(manifest['commands'].values()))['argv'][0]
    command += [python, '-B', '-u', str(RUN / 'coordinator.py')]
    record = {'unix': time.time(), 'unit': UNIT, 'command': command,
              'manifest_draft_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'training_user': 'fanyx', 'global_oomd_service': 'unchanged',
              'memory_limit_gib': 120, 'hard_timeout_hours': 48}
    (RUN / 'launch_attempt.json').write_text(json.dumps(record, indent=2) + '\n')
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    record.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    (RUN / 'launch_attempt.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
