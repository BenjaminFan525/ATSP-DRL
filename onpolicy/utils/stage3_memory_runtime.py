"""Verify the user-requested per-experiment memory policy before GPU work starts."""
from __future__ import annotations

import os
from pathlib import Path
import time


def verify_unprotected_memory(*, cgroup_root=Path('/sys/fs/cgroup'), membership=None):
    if membership is None:
        membership = Path('/proc/self/cgroup').read_text()
    line = next((line for line in membership.splitlines() if line.startswith('0::')), None)
    if line is None:
        raise RuntimeError('Unified cgroup membership is required')
    base = cgroup_root/line[3:].lstrip('/')
    # systemd 249 honors the omit attribute only on root-owned cgroups.
    if base.stat().st_uid != 0:
        raise RuntimeError('systemd 249 requires a root-owned service cgroup for an effective oomd exemption')
    try:
        omit = os.getxattr(base, 'user.oomd_omit')
    except OSError as exc:
        raise RuntimeError('The experiment has no effective systemd-oomd exemption') from exc
    if omit not in (b'1', b'yes', b'true'):
        raise RuntimeError('The systemd-oomd omit attribute is not enabled')
    limits = {}
    node = base
    while node != cgroup_root:
        values = {}
        for name in ('memory.high','memory.max','memory.swap.max'):
            path = node/name
            if not path.exists():
                raise RuntimeError(f'Memory controller not available: {path}')
            value = path.read_text().strip()
            if value != 'max':
                raise RuntimeError(f'Memory protection remains active: {path}={value}')
            values[name] = value
        limits[str(node)] = values
        node = node.parent
    return dict(verified_unix=time.time(), pid=os.getpid(), cgroup=str(base),
                cgroup_owner_uid=0, oomd_omit=omit.decode(), limits=limits,
                kernel_global_oom_behavior='unchanged', global_oomd_service='unchanged')
