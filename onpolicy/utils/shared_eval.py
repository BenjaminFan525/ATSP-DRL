"""Local protocol helpers for the per-GPU shared HKBZ evaluator."""

from __future__ import annotations

import os
import time
from multiprocessing.connection import Client, wait
from pathlib import Path


PROTOCOL_VERSION = 2
AUTHKEY = b'hkbz-shared-eval-v1'


def parse_cpu_set(spec: str) -> frozenset[int]:
    cpus: set[int] = set()
    for raw_part in str(spec or '').split(','):
        part = raw_part.strip()
        if not part:
            continue
        bounds = part.split('-', 1)
        try:
            start = int(bounds[0])
            stop = int(bounds[-1])
        except ValueError as error:
            raise ValueError(f'Invalid logical CPU range {part!r}.') from error
        if start < 0 or stop < start:
            raise ValueError(f'Invalid logical CPU range {part!r}.')
        cpus.update(range(start, stop + 1))
    if not cpus:
        raise ValueError('A non-empty logical CPU set is required.')
    return frozenset(cpus)


def format_cpu_set(cpus) -> str:
    values = sorted(set(int(cpu) for cpu in cpus))
    if not values:
        return ''
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f'{start}-{previous}')
        start = previous = value
    ranges.append(str(start) if start == previous else f'{start}-{previous}')
    return ','.join(ranges)


def set_affinity(pid: int, cpus) -> None:
    requested = set(int(cpu) for cpu in cpus)
    if not requested:
        raise ValueError('Refusing to apply an empty CPU affinity.')
    os.sched_setaffinity(int(pid), requested)
    observed = os.sched_getaffinity(int(pid))
    if observed != requested:
        raise RuntimeError(
            f'CPU affinity mismatch for pid={pid}: '
            f'requested={format_cpu_set(requested)}, '
            f'observed={format_cpu_set(observed)}'
        )


def set_process_affinity(pid: int, cpus) -> None:
    """Apply affinity to every existing thread in a process."""
    task_dir = Path(f'/proc/{int(pid)}/task')
    if not task_dir.is_dir():
        raise ProcessLookupError(pid)
    task_ids = [int(entry.name) for entry in task_dir.iterdir() if entry.name.isdigit()]
    for task_id in task_ids:
        try:
            set_affinity(task_id, cpus)
        except ProcessLookupError:
            # A short-lived runtime thread may exit between enumeration and
            # sched_setaffinity; all persistent threads are still verified.
            continue


class SharedEvalClient:
    """Blocking client; trainers naturally release their CPU slice while queued."""

    def __init__(self, socket_path: str, timeout_seconds: float = 7200.0):
        self.socket_path = str(Path(socket_path))
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0.0:
            raise ValueError('Shared-evaluator timeout must be positive.')

    def _connect(self):
        deadline = time.monotonic() + min(self.timeout_seconds, 60.0)
        last_error = None
        while time.monotonic() < deadline:
            try:
                return Client(
                    self.socket_path,
                    family='AF_UNIX',
                    authkey=AUTHKEY,
                )
            except (FileNotFoundError, ConnectionRefusedError, OSError) as error:
                last_error = error
                time.sleep(0.25)
        raise TimeoutError(
            f'Could not connect to shared evaluator {self.socket_path}: '
            f'{last_error}'
        )

    def request(self, payload: dict) -> dict:
        request = {'protocol_version': PROTOCOL_VERSION, **payload}
        connection = self._connect()
        try:
            connection.send(request)
            if not wait([connection], timeout=self.timeout_seconds):
                raise TimeoutError(
                    'Shared validation exceeded queue/execution timeout of '
                    f'{self.timeout_seconds:.1f}s.'
                )
            response = connection.recv()
        finally:
            connection.close()
        if not isinstance(response, dict):
            raise RuntimeError(
                f'Shared evaluator returned {type(response).__name__}, expected dict.'
            )
        if not response.get('ok', False):
            raise RuntimeError(
                'Shared evaluator failed: '
                f'{response.get("error", "unknown evaluator error")}\n'
                f'{response.get("traceback", "")}'.rstrip()
            )
        return response

    def ping(self) -> dict:
        return self.request({'operation': 'ping'})
