"""投递闸门测试：消息只准进指定的私聊，别处一律不发。

背景：之前的"回退到最近会话"会把主动消息插进桌面会话；用户明确要求
除了指定对话之外任何地方都不能出现。所以这条不是配置项，是两道硬闸：

  allowed_platforms  平台白名单，空 = 一律不发（fail closed）
  require_direct     session_key 必须带 :dm:/:private: 一类标记

  python -m unittest discover -s tests -p "test_*.py" -v
"""

from __future__ import annotations

import logging
import os
import sys
import unittest
from pathlib import Path

HERMES_SRC = Path(os.path.expanduser("~/AppData/Local/hermes/hermes-agent"))
PLUGIN_DIR = Path(__file__).resolve().parent.parent
for p in (str(HERMES_SRC), str(HERMES_SRC / "venv/Lib/site-packages"), str(PLUGIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from plugins.plugin_loader import load_plugin_module  # noqa: E402

mod = load_plugin_module("plugins.proactive-heartbeat", PLUGIN_DIR, parents=("plugins",),
                         logger=logging.getLogger("delivery-test"))
assert mod is not None

DM_KEY = "agent:main:qqbot:dm:0D888441B0B68A310D1B1DF7"
GROUP_KEY = "agent:main:qqbot:group:123456789"
DESKTOP_KEY = "20261104_173611_d47612"


class Ctx:
    plugin_id = "proactive-heartbeat"

    def __init__(self, **settings):
        self.settings = settings
        self.sent = []

    def get_config(self, k, d=None):
        return self.settings.get(k, d)

    def inject_message(self, content, role="user", session_key=None):
        self.sent.append(session_key)
        return True

    def register_tool(self, **k): pass
    def register_command(self, *a, **k): pass
    def register_hook(self, *a, **k): pass
    def on_unload(self, cb): pass


def state_with(sessions, last=None, last_platform=None):
    return {"runtime": {"sessions": sessions,
                        "last_session_key": last,
                        "last_platform": last_platform}}


ALL_SESSIONS = {"desktop": {"key": DESKTOP_KEY},
                "qqbot": {"key": DM_KEY}}


class TestPlatformGate(unittest.TestCase):
    def test_only_qqbot_even_in_all_mode(self):
        """all 模式下也只能发到白名单里的平台，桌面必须被拦。"""
        ctx = Ctx(delivery_mode="all", allowed_platforms=["qqbot"], require_direct=True)
        ok, used = mod._deliver(ctx, state_with(ALL_SESSIONS), "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(ctx.sent, [DM_KEY])
        self.assertNotIn(DESKTOP_KEY, ctx.sent)

    def test_empty_allowlist_refuses_everything(self):
        """留空不是"不限制"，是"限制到零"——这条闸生来就是为了限制。"""
        ctx = Ctx(delivery_mode="all", allowed_platforms=[], require_direct=True)
        ok, used = mod._deliver(ctx, state_with(ALL_SESSIONS), "hi", 0)
        self.assertFalse(ok)
        self.assertEqual(ctx.sent, [])

    def test_string_allowlist_is_parsed(self):
        ctx = Ctx(delivery_mode="all", allowed_platforms="qqbot, telegram", require_direct=True)
        ok, used = mod._deliver(ctx, state_with(ALL_SESSIONS), "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(used, "qqbot")


class TestFallbackIsGated(unittest.TestCase):
    def test_no_fallback_to_desktop(self):
        """找不到 QQ 会话时，绝不能退而求其次发到最近会话。"""
        ctx = Ctx(delivery_mode="platform:qqbot", allowed_platforms=["qqbot"], require_direct=True)
        st = state_with({"desktop": {"key": DESKTOP_KEY}}, last=DESKTOP_KEY, last_platform="desktop")
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertFalse(ok)
        self.assertEqual(ctx.sent, [], "回退目标必须同样过闸")

    def test_fallback_works_when_target_is_allowed(self):
        ctx = Ctx(delivery_mode="platform:qqbot", allowed_platforms=["qqbot"], require_direct=True)
        st = state_with({}, last=DM_KEY, last_platform="qqbot")
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(ctx.sent, [DM_KEY])


class TestDirectOnly(unittest.TestCase):
    def test_group_session_is_blocked(self):
        ctx = Ctx(delivery_mode="all", allowed_platforms=["qqbot"], require_direct=True)
        ok, used = mod._deliver(ctx, state_with({"qqbot": {"key": GROUP_KEY}}), "hi", 0)
        self.assertFalse(ok)
        self.assertEqual(ctx.sent, [])

    def test_direct_session_passes(self):
        ctx = Ctx(delivery_mode="all", allowed_platforms=["qqbot"], require_direct=True)
        ok, used = mod._deliver(ctx, state_with({"qqbot": {"key": DM_KEY}}), "hi", 0)
        self.assertTrue(ok)

    def test_direct_check_can_be_disabled(self):
        ctx = Ctx(delivery_mode="all", allowed_platforms=["qqbot"], require_direct=False)
        ok, used = mod._deliver(ctx, state_with({"qqbot": {"key": GROUP_KEY}}), "hi", 0)
        self.assertTrue(ok, "显式关掉才允许群发")


class TestDefaultsAreStrict(unittest.TestCase):
    def test_plugin_defaults_block_desktop(self):
        """默认配置下（不传任何 settings），桌面必须发不出去。"""
        ctx = Ctx()   # 全部走 DEFAULTS
        ok, used = mod._deliver(ctx, state_with(ALL_SESSIONS), "hi", 0)
        self.assertEqual(ctx.sent, [DM_KEY])
        self.assertFalse(ctx.sent == [DESKTOP_KEY])


if __name__ == "__main__":
    unittest.main(verbosity=2)
