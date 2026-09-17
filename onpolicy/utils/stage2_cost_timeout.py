"""Engineering work budgets for shared execution; never scheduling costs.

An overlapping host latency is not private compute time. Allocate each
non-overlapping wall interval exactly once across its participating jobs.
The original 600-second budget is kept, with an independent 600-second hard
no-progress deadline and the unchanged service/global experiment deadline.
"""
import math
import os
import signal
import threading

LEGACY_TIMEOUT_CONTRACT = 'duplicated_rpc_and_overlapping_actor_latency_v1'
TIMEOUT_CONTRACT = 'amortized_shared_wall_and_progress_deadline_v2'


def shared_wall_charges(wall_seconds, weights):
    """Weighted allocation, not a claim to measure isolated CPU/GPU time."""
    weights = [float(weight) for weight in weights]
    if (not math.isfinite(wall_seconds) or wall_seconds < 0
            or any(not math.isfinite(weight) or weight < 0 for weight in weights)):
        raise ValueError('Invalid wall interval or actor latency.')
    if not weights:
        return []
    total = math.fsum(weights)
    if total == 0:
        weights, total = [1.] * len(weights), float(len(weights))
    return [wall_seconds * weight / total for weight in weights]


class HardProgressDeadline:
    """Main-process watchdog; a blocked GPU/RPC cannot hang cleanup forever.

    Fatal exit 124 is deliberate: raising into an unresponsive CUDA thread or
    environment cleanup can hang again. The owning bounded controller sees
    the nonzero exit, never retries, exits its service, and systemd KillMode=
    control-group reaps the owned workers. Already-persisted files remain.
    """
    def __init__(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Progress deadline must be finite and positive.')
        self.seconds = seconds
        self.previous = None

    @staticmethod
    def expired(*_):
        os.write(2, b'[CostProgressDeadline] No completed RPC/inference progress; hard timeout, no retry.\n')
        os._exit(124)

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise ValueError('Cost progress watchdog must be owned by the main thread.')
        if any(signal.getitimer(signal.ITIMER_REAL)):
            raise ValueError('Do not override another real-time alarm.')
        self.previous = signal.signal(signal.SIGALRM, self.expired)
        self.touch()
        return self

    def touch(self):
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, self.previous)

