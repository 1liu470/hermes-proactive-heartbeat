"""Lock semantics for the heartbeat tick (the double-send guard).

Run:  python -m unittest discover -s tests -p "test_*.py" -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_STORE_PATH = os.path.join(os.path.dirname(_HERE), "store.py")
_spec = importlib.util.spec_from_file_location("ph_store_under_test", _STORE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["ph_store_under_test"] = _mod
_spec.loader.exec_module(_mod)


class TestTickLock(unittest.TestCase):
    def setUp(self):
        self.sandbox = tempfile.mkdtemp(prefix="ph-lock-")
        self._old = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self.sandbox

    def tearDown(self):
        if self._old is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old

    def test_second_acquire_is_refused(self):
        fd = _mod.acquire_tick_lock()
        self.assertIsNotNone(fd)
        self.assertIsNone(_mod.acquire_tick_lock(), "a concurrent tick must not proceed")
        _mod.release_tick_lock(fd)
        fd2 = _mod.acquire_tick_lock()
        self.assertIsNotNone(fd2, "lock must be reusable after release")
        _mod.release_tick_lock(fd2)

    def test_stale_lock_is_reclaimed(self):
        fd = _mod.acquire_tick_lock()
        self.assertIsNotNone(fd)
        os.close(fd)
        lock_path = _mod.state_dir() / "tick.lock"
        old = time.time() - 3600
        os.utime(lock_path, (old, old))
        fd2 = _mod.acquire_tick_lock(ttl_seconds=180)
        self.assertIsNotNone(fd2, "a crashed holder must not block the loop forever")
        _mod.release_tick_lock(fd2)

    def test_release_is_idempotent(self):
        fd = _mod.acquire_tick_lock()
        _mod.release_tick_lock(fd)
        _mod.release_tick_lock(None)  # must not raise
        self.assertFalse((_mod.state_dir() / "tick.lock").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
