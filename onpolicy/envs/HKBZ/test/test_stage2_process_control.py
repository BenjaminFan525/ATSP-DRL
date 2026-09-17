"""Kernel capability checks touch only this test process and its own child."""
import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from onpolicy.utils import stage2_process_control as control


class PidfdCompatibilityTests(unittest.TestCase):
    def test_missing_python_wrappers_use_real_pidfd_and_signal_zero(self):
        with patch.object(os, 'pidfd_open', None, create=True), \
                patch.object(signal, 'pidfd_send_signal', None, create=True):
            fd = control.pidfd_open(os.getpid())
            try:
                self.assertFalse(os.get_inheritable(fd))
                control.pidfd_send_signal(fd, 0)
            finally:
                os.close(fd)

    def test_native_wrappers_are_preferred_without_abi_assumptions(self):
        with patch.object(os, 'pidfd_open', return_value=123, create=True) as opened, \
                patch.object(signal, 'pidfd_send_signal', return_value=None, create=True) as sent, \
                patch.object(control, '_syscall') as fallback:
            self.assertEqual(control.pidfd_open(42), 123)
            control.pidfd_send_signal(123, 0)
            opened.assert_called_once_with(42, 0)
            sent.assert_called_once_with(123, 0, None, 0)
            fallback.assert_not_called()

    def test_native_permission_failure_is_not_bypassed(self):
        with patch.object(os, 'pidfd_open', side_effect=PermissionError(errno.EPERM, 'denied'), create=True), \
                patch.object(control, '_syscall') as fallback:
            with self.assertRaises(PermissionError):
                control.pidfd_open(42)
            fallback.assert_not_called()

    def test_unverified_platform_and_invalid_pid_fail_closed(self):
        with patch.object(os, 'pidfd_open', None, create=True):
            with patch.object(control.platform, 'machine', return_value='unknown'):
                with self.assertRaises(NotImplementedError):
                    control.pidfd_open(42)
            for pid in (-1, 0, 2**40):
                with self.assertRaises(ValueError):
                    control.pidfd_open(pid)

    def test_closed_fd_propagates_kernel_error(self):
        fd = control.pidfd_open(os.getpid())
        os.close(fd)
        with patch.object(signal, 'pidfd_send_signal', None, create=True):
            with self.assertRaises(OSError) as error:
                control.pidfd_send_signal(fd, 0)
        self.assertEqual(error.exception.errno, errno.EBADF)

    def test_only_owned_child_is_stopped_resumed_and_terminated(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        fd = control.pidfd_open(child.pid)
        try:
            control.pidfd_send_signal(fd, signal.SIGSTOP)
            for _ in range(100):
                raw = (Path('/proc') / str(child.pid) / 'stat').read_text()
                if raw[raw.rfind(')') + 2:].split()[0] in ('T', 't'):
                    break
                time.sleep(.01)
            else:
                self.fail('Owned synthetic child did not stop.')
            control.pidfd_send_signal(fd, signal.SIGCONT)
            control.pidfd_send_signal(fd, signal.SIGTERM)
            self.assertEqual(child.wait(timeout=5), -signal.SIGTERM)
            with self.assertRaises(ProcessLookupError):
                control.pidfd_send_signal(fd, 0)
        finally:
            if child.poll() is None:
                control.pidfd_send_signal(fd, signal.SIGCONT)
                control.pidfd_send_signal(fd, signal.SIGKILL)
                child.wait(timeout=5)
            os.close(fd)


if __name__ == '__main__':
    unittest.main()
