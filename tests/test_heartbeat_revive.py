"""心跳线程的复活判据：看线程还在不在，而不是看那个 Event。

踩过的坑：`plugins disable` 会把 stop Event 置位，`enable` 之后 register 再跑一遍，
旧写法见到 `stop.is_set()` 就 return。于是插件显示 enabled、钩子照常注入，心跳却
永久哑掉，日志里一行错都没有——机器上看起来一切正常，只是它从此不再主动开口。

判据只能是 thread.is_alive()：线程还在收尾就别插手，线程已经退出就必须重建。
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERMES_SRC = Path(os.path.expanduser("~/AppData/Local/hermes/hermes-agent"))
PLUGIN_DIR = Path(__file__).resolve().parent.parent

SANDBOX = Path(tempfile.mkdtemp(prefix="ph-revive-"))
os.environ["HERMES_HOME"] = str(SANDBOX)

for p in (str(HERMES_SRC), str(HERMES_SRC / "venv/Lib/site-packages"), str(PLUGIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ph_revive_under_test", PLUGIN_DIR / "__init__.py",
    submodule_search_locations=[str(PLUGIN_DIR)])
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


class StubCtx:
    plugin_id = "proactive-heartbeat"

    def __init__(self, **settings):
        self.settings = settings

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_hook(self, *a, **k):
        pass

    def register_tool(self, **k):
        pass

    def register_command(self, *a, **k):
        pass

    def on_unload(self, *a, **k):
        pass


class TestHeartbeatRevives(unittest.TestCase):
    def setUp(self):
        rt = mod._RUNTIME
        self._saved_stop = rt.get("stop")
        self._saved_thread = rt.get("thread")
        rt["stop"] = None
        rt["thread"] = None
        rt["ctx"] = None
        rt["tick_fn"] = None
        rt["interval"] = None

    def tearDown(self):
        rt = mod._RUNTIME
        stop = rt.get("stop")
        if stop is not None:
            stop.set()
        th = rt.get("thread")
        if th is not None and th.is_alive():
            th.join(timeout=2)
        rt["stop"] = self._saved_stop
        rt["thread"] = self._saved_thread

    def test_dead_thread_is_rebuilt_even_if_stop_is_set(self):
        dead = threading.Thread(target=lambda: None, name="dead")
        dead.start()
        dead.join()  # 已退出
        rt = mod._RUNTIME
        rt["thread"] = dead
        rt["stop"] = threading.Event()
        rt["stop"].set()  # 上一次卸载留下的痕迹

        mod._ensure_thread(StubCtx(tick_seconds=900))

        self.assertIsNot(rt["stop"], None)
        self.assertFalse(rt["stop"].is_set(), "旧痕迹必须清掉，否则新线程立刻退出")
        self.assertIsNot(rt["thread"], dead, "线程已死，必须换新的")
        self.assertTrue(rt["thread"].is_alive())

    def test_live_thread_is_not_duplicated(self):
        gate = threading.Event()

        def _wait():
            gate.wait(5)

        live = threading.Thread(target=_wait, name="live", daemon=True)
        live.start()
        try:
            rt = mod._RUNTIME
            rt["thread"] = live
            rt["stop"] = threading.Event()
            rt["stop"].set()

            mod._ensure_thread(StubCtx(tick_seconds=900))

            self.assertIs(rt["thread"], live, "线程还活着就别另起一个，否则两个循环同扫一份账本")
        finally:
            gate.set()
            live.join(timeout=2)

    def test_interval_comes_from_config(self):
        mod._ensure_thread(StubCtx(tick_seconds=120))
        self.assertEqual(mod._RUNTIME["interval"], 120.0)
        # 下限保护：配置里给个荒谬的小值，不能让它变成忙等
        mod._RUNTIME["thread"].is_alive()
        stop = mod._RUNTIME["stop"]
        stop.set()
        mod._RUNTIME["thread"].join(timeout=2)
        mod._RUNTIME["stop"] = None
        mod._RUNTIME["thread"] = None
        mod._ensure_thread(StubCtx(tick_seconds=1))
        self.assertGreaterEqual(mod._RUNTIME["interval"], 15.0)


if __name__ == "__main__":
    unittest.main()
