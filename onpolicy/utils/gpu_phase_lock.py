"""Cross-process serialization for memory-heavy phases on one GPU."""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def exclusive_gpu_phase(lock_path: str, label: str) -> Iterator[float]:
    """Hold an advisory file lock and yield the queue wait in seconds.

    Rollout inference remains concurrent.  Only callers that explicitly wrap
    a memory-heavy phase participate, so an empty path preserves the legacy
    single-process behavior exactly.
    """

    path_text = str(lock_path or "").strip()
    if not path_text:
        yield 0.0
        return

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        raise ValueError("GPU phase lock path must be absolute.")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o600)
    started = time.monotonic()
    print(f"[GPUPhase] waiting label={label} lock={path}", flush=True)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        waited = max(0.0, time.monotonic() - started)
        print(
            f"[GPUPhase] acquired label={label} wait_seconds={waited:.3f}",
            flush=True,
        )
        try:
            yield waited
        finally:
            print(f"[GPUPhase] released label={label}", flush=True)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
