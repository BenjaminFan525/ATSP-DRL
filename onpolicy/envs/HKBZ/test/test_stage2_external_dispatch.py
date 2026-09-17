import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from onpolicy.scripts.train.run_stage2_resource_manifest_trial import atomic_json, apply_runtime_overrides
from onpolicy.utils.stage2_external_dispatch import (
    acquire_job_lock, guard_launch, marker_path, process_identity, read_marker,
)


class ExternalDispatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manifest = Path(self.temp.name) / 'manifest.json'
        self.manifest.write_text('{"commands": {}}\n')
        self.key = 'B2_R1_seed11'
        self.args = argparse.Namespace(manifest=self.manifest, command_key=self.key,
            record=Path(self.temp.name) / 'adoption.json', external_owner_pid=None,
            cpu_set='0-1', experiment_name=None, cuda_memory_fraction=None)

    def state(self, **extra):
        value = {'manifest': str(self.manifest.resolve()), 'command_key': self.key,
            'manifest_sha256': hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            'supervisor_identity': process_identity(os.getpid()), 'status': 'launching',
            'cpu_set': '0-1', 'deadline_unix_time': time.time() + 60}
        value.update(extra)
        atomic_json(marker_path(self.manifest, self.key), value)
        return value

    def test_lock_is_exclusive_and_inheritable(self):
        fd = acquire_job_lock(self.manifest, self.key)
        self.addCleanup(os.close, fd)
        self.assertTrue(os.get_inheritable(fd))
        with self.assertRaises(BlockingIOError):
            acquire_job_lock(self.manifest, self.key)

    def test_completed_claim_is_adopted_without_training(self):
        self.state(status='completed', exit_code=0, trainer_pid=42)
        self.assertEqual(guard_launch(self.args, atomic_json), 0)
        self.assertEqual(json.loads(self.args.record.read_text())['status'], 'adopted_completed')

    def test_failed_claim_propagates_failure_not_retry(self):
        self.state(status='failed', exit_code=7)
        self.assertEqual(guard_launch(self.args, atomic_json), 7)

    def test_live_external_job_is_waited_for_under_exclusive_lock(self):
        fd = acquire_job_lock(self.manifest, self.key)
        self.addCleanup(os.close, fd)
        self.state(status='running', trainer_pid=42)
        timer = threading.Timer(0.1, lambda: self.state(status='completed', exit_code=0, trainer_pid=42))
        timer.start()
        self.addCleanup(timer.join)
        self.assertEqual(guard_launch(self.args, atomic_json), 0)
        self.assertEqual(json.loads(self.args.record.read_text())['trainer_pid'], 42)

    def test_only_claiming_supervisors_child_can_execute(self):
        self.state()
        self.args.external_owner_pid = os.getpid()
        with patch('onpolicy.utils.stage2_external_dispatch.os.getppid', return_value=os.getpid()):
            self.assertIsNone(guard_launch(self.args, atomic_json))

    def test_stale_owner_refuses_retry(self):
        self.state(supervisor_identity={'pid': 999999999, 'start_ticks': 0, 'boot_id': 'none'})
        with self.assertRaisesRegex(RuntimeError, 'disappeared'):
            guard_launch(self.args, atomic_json)

    def test_manifest_mutation_rejected(self):
        self.state(status='completed', exit_code=0)
        self.manifest.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'frozen manifest'):
            read_marker(self.manifest, self.key)

    def test_wrong_key_and_owner_rejected(self):
        with self.assertRaises(ValueError):
            marker_path(self.manifest, '../escape')
        self.state()
        self.args.external_owner_pid = os.getpid()
        with self.assertRaisesRegex(RuntimeError, 'claiming supervisor'):
            guard_launch(self.args, atomic_json)

    def test_allocator_override_is_only_resource_option(self):
        self.args.cuda_memory_fraction = 0.10
        command = ['python', 'train.py', '--cuda_memory_fraction', '0.4', '--seed', '11']
        result = apply_runtime_overrides(command, self.args)
        self.assertEqual(result, {'cuda_memory_fraction': 0.10})
        self.assertEqual(command[command.index('--seed') + 1], '11')
        self.assertEqual(len(command), 6)
        self.assertEqual(command[command.index('--cuda_memory_fraction') + 1], '0.1')

    def test_invalid_allocator_rejected(self):
        for value in (0, -1, 1.1, float('nan')):
            self.args.cuda_memory_fraction = value
            with self.assertRaises(ValueError):
                apply_runtime_overrides([], self.args)


if __name__ == '__main__':
    unittest.main()
