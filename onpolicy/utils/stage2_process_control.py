"""Race-safe pidfds for Conda builds missing Python's optional Linux wrappers.

Fallback ABI is deliberately restricted to Linux x86-64 LP64, checked against
the host headers and Linux v6.8 arch/x86/entry/syscalls/syscall_64.tbl:
https://github.com/torvalds/linux/blob/v6.8/arch/x86/entry/syscalls/syscall_64.tbl
This invokes the same kernel calls and preserves their permission failures.
It never falls back to signaling a numeric PID with kill().
"""
from __future__ import annotations

import ctypes
import os
import platform
import signal
import sys


def _syscall(number, *arguments):
    if sys.platform != 'linux' or platform.machine() != 'x86_64' or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise NotImplementedError('Native Python pidfd support or verified Linux x86-64 ABI required.')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def pidfd_open(pid):
    native = getattr(os, 'pidfd_open', None)
    if callable(native):
        return native(pid, 0)
    if not isinstance(pid, int) or not 0 < pid <= 2**31 - 1:
        raise ValueError('A positive Linux process ID is required.')
    return _syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_send_signal(fd, signum):
    native = getattr(signal, 'pidfd_send_signal', None)
    if callable(native):
        return native(fd, signum, None, 0)
    if not isinstance(fd, int) or fd < 0:
        raise ValueError('An open pidfd is required.')
    _syscall(424, ctypes.c_int(fd), ctypes.c_int(signum), ctypes.c_void_p(), ctypes.c_uint(0))
