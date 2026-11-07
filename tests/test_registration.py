"""注册面：命令名不能被宿主的保留字吃掉。

真事：命令一直叫 `/proactive`，而 Hermes 内置的 `/loop` 已把 `proactive` 用作别名。
宿主遇到冲突不报错、不抛异常，只写一行 WARNING 然后跳过注册。于是这个命令
在日志里"注册过"，在会话里从不存在，直到有人想用它才发现。

所以这里锁两件事：命令名是 nudge，且不与内置别名表重合。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

HERMES_SRC = Path(os.path.expanduser("~/AppData/Local/hermes/hermes-agent"))
PLUGIN_DIR = Path(__file__).resolve().parent.parent
for p in (str(HERMES_SRC), str(HERMES_SRC / "venv/Lib/site-packages"), str(PLUGIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ph_registration_under_test", PLUGIN_DIR / "__init__.py",
    submodule_search_locations=[str(PLUGIN_DIR)])
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod   # 相对导入 from . import engine 需要先登记
_spec.loader.exec_module(mod)


class StubCtx:
    """只够 register() 跑完的最小宿主。"""

    plugin_id = "proactive-heartbeat"

    def __init__(self):
        self.hooks, self.tools, self.commands, self.settings = {}, {}, {}, {}

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_hook(self, name, cb):
        self.hooks.setdefault(name, []).append(cb)

    def register_tool(self, **kw):
        self.tools[kw["name"]] = kw

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = handler

    def on_unload(self, cb):
        self.unloaded = cb

    def inject_message(self, *a, **k):
        return True


class TestCommandName(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = StubCtx()
        mod.register(cls.ctx)
        mod._RUNTIME["stop"].set()  # 别让心跳线程留在测试进程里

    def test_command_is_registered(self):
        self.assertIn("nudge", self.ctx.commands)
        self.assertTrue(callable(self.ctx.commands["nudge"]))

    def test_command_name_is_not_a_host_reserved_word(self):
        """宿主保留字表里的名字一律不能用——冲突是静默的。"""
        reserved = {"proactive", "loop", "heartbeat", "hb", "goal", "plan",
                    "review", "refine", "memory-graph", "journey", "moa"}
        self.assertNotIn("nudge", reserved)
        for name in self.ctx.commands:
            self.assertNotIn(name, reserved, f"命令 {name} 会被宿主静默跳过")

    def test_tools_registered(self):
        self.assertIn("proactive_intent", self.ctx.tools)
        self.assertIn("proactive_status", self.ctx.tools)
        self.assertIn("pre_llm_call", self.ctx.hooks)


if __name__ == "__main__":
    unittest.main()
