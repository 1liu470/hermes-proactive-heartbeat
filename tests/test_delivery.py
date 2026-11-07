"""投递闸门：消息只准进指定的私聊，别处一律不发。

2026-11-07 改版后的语义：

  allowed_platforms   平台白名单，空 = 一律不发（fail closed）
  require_direct      必须是 direct；unknown（如桌面纯会话号）也算不安全
  delivery_mode       platform:X（单）/ platforms:X,Y（多）/ active（白名单内活跃的）
  all                 **已删除**——它的语义是广播，配了也拒发

回退只在白名单内的会话里挑，`last_session_key` 不再是回退源：
那个字段可能是桌面，之前正是它把消息漏到别处的。

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

DM_KEY = "agent:main:qqbot:dm:<qq_id>"
GROUP_KEY = "agent:main:qqbot:group:123456789"
DESKTOP_KEY = "20260101_000000_abcdef"
TG_DM = "agent:main:telegram:dm:555000111"
TG_GROUP = "agent:main:telegram:group:-100777"
DC_DM = "agent:main:discord:dm:987654321"


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


ALL_SESSIONS = {"desktop": {"key": DESKTOP_KEY, "at": 100},
                "qqbot": {"key": DM_KEY, "at": 200}}


class TestNoBroadcast(unittest.TestCase):
    def test_all_mode_is_refused(self):
        """all 会广播到每个会话，语义已删除——配了也不发。"""
        ctx = Ctx(delivery_mode="all", allowed_platforms=["qqbot"], require_direct=True)
        ok, used = mod._deliver(ctx, state_with(ALL_SESSIONS), "hi", 0)
        self.assertFalse(ok)
        self.assertEqual(ctx.sent, [])

    def test_empty_allowlist_refuses_everything(self):
        """留空不是"不限制"，是"限制到零"。"""
        ctx = Ctx(delivery_mode="platform:qqbot", allowed_platforms=[], require_direct=True)
        ok, used = mod._deliver(ctx, state_with(ALL_SESSIONS), "hi", 0)
        self.assertFalse(ok)
        self.assertEqual(ctx.sent, [])

    def test_unknown_mode_falls_back_inside_allowlist(self):
        """写错的 mode 不放大权限，只回退到白名单内的会话。"""
        ctx = Ctx(delivery_mode="whatever", allowed_platforms=["qqbot"], require_direct=True)
        ok, used = mod._deliver(ctx, state_with(ALL_SESSIONS), "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(ctx.sent, [DM_KEY])


class TestPlatformGate(unittest.TestCase):
    def test_single_platform(self):
        ctx = Ctx(delivery_mode="platform:qqbot", allowed_platforms=["qqbot"], require_direct=True)
        ok, used = mod._deliver(ctx, state_with(ALL_SESSIONS), "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(used, "qqbot")
        self.assertEqual(ctx.sent, [DM_KEY])
        self.assertNotIn(DESKTOP_KEY, ctx.sent)

    def test_multi_platform_sends_to_both(self):
        """多平台：qqbot 与 telegram 各发一次。"""
        ctx = Ctx(delivery_mode="platforms:qqbot,telegram",
                  allowed_platforms=["qqbot", "telegram"], require_direct=True)
        st = state_with({"qqbot": {"key": DM_KEY, "at": 200},
                         "telegram": {"key": TG_DM, "at": 300}})
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(ctx.sent, [DM_KEY, TG_DM])
        self.assertEqual(used, "qqbot,telegram")

    def test_multi_platform_respects_allowlist(self):
        """mode 里写了 telegram，但白名单没有——只发白名单内的那个。"""
        ctx = Ctx(delivery_mode="platforms:qqbot,telegram",
                  allowed_platforms=["qqbot"], require_direct=True)
        st = state_with({"qqbot": {"key": DM_KEY, "at": 200},
                         "telegram": {"key": TG_DM, "at": 300}})
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(ctx.sent, [DM_KEY])

    def test_string_allowlist_is_parsed(self):
        ctx = Ctx(delivery_mode="platforms:qqbot,telegram",
                  allowed_platforms="qqbot, telegram", require_direct=True)
        st = state_with({"qqbot": {"key": DM_KEY, "at": 200},
                         "telegram": {"key": TG_DM, "at": 300}})
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(used, "qqbot,telegram")

    def test_active_mode_uses_allowlisted_sessions(self):
        ctx = Ctx(delivery_mode="active", allowed_platforms=["qqbot", "telegram"],
                  require_direct=True)
        st = state_with({"qqbot": {"key": DM_KEY, "at": 200},
                         "telegram": {"key": TG_DM, "at": 300}})
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(set(ctx.sent), {DM_KEY, TG_DM})


class TestFallbackIsGated(unittest.TestCase):
    def test_no_fallback_to_desktop(self):
        """找不到指定平台时，绝不能退而求其次发到桌面。"""
        ctx = Ctx(delivery_mode="platform:qqbot", allowed_platforms=["qqbot"], require_direct=True)
        st = state_with({"desktop": {"key": DESKTOP_KEY, "at": 100}},
                        last=DESKTOP_KEY, last_platform="desktop")
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertFalse(ok)
        self.assertEqual(ctx.sent, [], "回退目标必须同样过闸")

    def test_last_session_key_is_not_a_fallback(self):
        """只有 last_session_key 没有 sessions 时，不回退——那个字段可能是桌面。"""
        ctx = Ctx(delivery_mode="platform:qqbot", allowed_platforms=["qqbot"], require_direct=True)
        st = state_with({}, last=DESKTOP_KEY, last_platform="desktop")
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertFalse(ok)
        self.assertEqual(ctx.sent, [])

    def test_fallback_uses_newest_allowlisted_session(self):
        """mode 指向一个没有会话的平台时，回退到白名单内最近的那个。"""
        ctx = Ctx(delivery_mode="platform:telegram", allowed_platforms=["qqbot"],
                  require_direct=True)
        ok, used = mod._deliver(ctx, state_with(ALL_SESSIONS), "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(ctx.sent, [DM_KEY])


class TestDirectOnly(unittest.TestCase):
    def test_group_session_is_blocked(self):
        ctx = Ctx(delivery_mode="platform:qqbot", allowed_platforms=["qqbot"], require_direct=True)
        ok, used = mod._deliver(ctx, state_with({"qqbot": {"key": GROUP_KEY, "at": 1}}), "hi", 0)
        self.assertFalse(ok)
        self.assertEqual(ctx.sent, [])

    def test_direct_session_passes(self):
        ctx = Ctx(delivery_mode="platform:qqbot", allowed_platforms=["qqbot"], require_direct=True)
        ok, used = mod._deliver(ctx, state_with({"qqbot": {"key": DM_KEY, "at": 1}}), "hi", 0)
        self.assertTrue(ok)

    def test_telegram_group_blocked(self):
        ctx = Ctx(delivery_mode="platform:telegram", allowed_platforms=["telegram"],
                  require_direct=True)
        ok, used = mod._deliver(ctx, state_with({"telegram": {"key": TG_GROUP, "at": 1}}), "hi", 0)
        self.assertFalse(ok)

    def test_desktop_blocked_even_if_allowlisted(self):
        """把 desktop 加进白名单也没用——它的键判不出私聊，unknown 一律拒。"""
        ctx = Ctx(delivery_mode="platform:desktop", allowed_platforms=["desktop"],
                  require_direct=True)
        ok, used = mod._deliver(ctx, state_with({"desktop": {"key": DESKTOP_KEY, "at": 1}}), "hi", 0)
        self.assertFalse(ok)
        self.assertEqual(ctx.sent, [])

    def test_direct_check_can_be_disabled(self):
        ctx = Ctx(delivery_mode="platform:qqbot", allowed_platforms=["qqbot"], require_direct=False)
        ok, used = mod._deliver(ctx, state_with({"qqbot": {"key": GROUP_KEY, "at": 1}}), "hi", 0)
        self.assertTrue(ok, "显式关掉才允许群发")


class TestActiveAnywhere(unittest.TestCase):
    """active + `*`：Hermes 接通了哪个平台，就往哪个平台的私聊发。

    平台接通这一步由人做——接不通就没有会话，插件不会凭空造一个。
    """

    def test_reaches_every_connected_platform(self):
        ctx = Ctx(delivery_mode="active", allowed_platforms=["*"], require_direct=True)
        st = state_with({"qqbot": {"key": DM_KEY, "at": 100},
                         "telegram": {"key": TG_DM, "at": 300},
                         "discord": {"key": DC_DM, "at": 200}})
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(set(ctx.sent), {DM_KEY, TG_DM, DC_DM})
        self.assertEqual(set(used.split(",")), {"qqbot", "telegram", "discord"})

    def test_only_connected_platforms(self):
        """telegram 没接通就别想着往那儿发。"""
        ctx = Ctx(delivery_mode="active", allowed_platforms=["*"], require_direct=True)
        ok, used = mod._deliver(ctx, state_with({"qqbot": {"key": DM_KEY, "at": 1}}), "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(ctx.sent, [DM_KEY])

    def test_desktop_excluded(self):
        ctx = Ctx(delivery_mode="active", allowed_platforms=["*"], require_direct=True)
        st = state_with({"desktop": {"key": DESKTOP_KEY, "at": 999},
                         "qqbot": {"key": DM_KEY, "at": 100}})
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(ctx.sent, [DM_KEY])
        self.assertNotIn(DESKTOP_KEY, ctx.sent)

    def test_groups_still_blocked(self):
        """接通了但只有群会话，那就一条也不发——不因为放开跨平台就发到群里。"""
        ctx = Ctx(delivery_mode="active", allowed_platforms=["*"], require_direct=True)
        st = state_with({"telegram": {"key": TG_GROUP, "at": 1},
                         "discord": {"key": "agent:main:discord:channel:123", "at": 2}})
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertFalse(ok)
        self.assertEqual(ctx.sent, [])

    def test_mixed_direct_and_group(self):
        ctx = Ctx(delivery_mode="active", allowed_platforms=["*"], require_direct=True)
        st = state_with({"qqbot": {"key": DM_KEY, "at": 1},
                         "telegram": {"key": TG_GROUP, "at": 2}})
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(ctx.sent, [DM_KEY])

    def test_explicit_allowlist_narrows_it(self):
        """白名单写了具体平台，即使接通了别的也不发。"""
        ctx = Ctx(delivery_mode="active", allowed_platforms=["qqbot"], require_direct=True)
        st = state_with({"qqbot": {"key": DM_KEY, "at": 1},
                         "telegram": {"key": TG_DM, "at": 2}})
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(ctx.sent, [DM_KEY])

    def test_no_sessions_means_silence(self):
        """一个平台都没接通，就一句话都不说。"""
        ctx = Ctx(delivery_mode="active", allowed_platforms=["*"], require_direct=True)
        ok, used = mod._deliver(ctx, state_with({}), "hi", 0)
        self.assertFalse(ok)
        self.assertEqual(ctx.sent, [])


class TestDefaultsAreStrict(unittest.TestCase):
    def test_plugin_defaults_block_desktop(self):
        """默认配置下（不传任何 settings），桌面必须发不出去。"""
        ctx = Ctx()
        ok, used = mod._deliver(ctx, state_with(ALL_SESSIONS), "hi", 0)
        self.assertEqual(ctx.sent, [DM_KEY])
        self.assertNotIn(DESKTOP_KEY, ctx.sent)

    def test_defaults_reach_multiple_platforms(self):
        """默认就是 active + `*`，接通的平台都该收到。"""
        ctx = Ctx()
        st = state_with({"qqbot": {"key": DM_KEY, "at": 1},
                         "telegram": {"key": TG_DM, "at": 2},
                         "desktop": {"key": DESKTOP_KEY, "at": 3}})
        ok, used = mod._deliver(ctx, st, "hi", 0)
        self.assertTrue(ok)
        self.assertEqual(set(ctx.sent), {DM_KEY, TG_DM})


if __name__ == "__main__":
    unittest.main(verbosity=2)
