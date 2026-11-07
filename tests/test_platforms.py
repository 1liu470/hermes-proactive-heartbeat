"""平台判定：从会话键判出一对一还是群聊。

判定靠结构而不是猜子串。Hermes 的键是

    agent:<profile>:<platform>:<chat_type>:<chat_id>

`chat_type` 由各适配器给出（dm / group / channel / thread / forum）。
本机实测：qqbot 是 `agent:main:qqbot:dm:<id>`，桌面是纯会话号
`20260101_000000_abcdef`——后者没有平台前缀，必然判成 unknown。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from platforms import classify, describe, is_allowed_target, parse, platform_of  # noqa: E402

QQ_DM = "agent:main:qqbot:dm:<qq_id>"
QQ_GROUP = "agent:main:qqbot:group:123456789"
TG_DM = "agent:main:telegram:dm:123456789"
TG_GROUP = "agent:main:telegram:group:-1001234"
DC_CHANNEL = "agent:main:discord:channel:987654321"
SLACK_THREAD = "agent:main:slack:thread:1712345678.123456"
DESKTOP = "20260101_000000_abcdef"


class TestParse(unittest.TestCase):
    def test_gateway_key(self):
        info = parse(QQ_DM)
        self.assertEqual(info["platform"], "qqbot")
        self.assertEqual(info["chat_type"], "dm")
        self.assertEqual(info["chat_id"], "<qq_id>")

    def test_local_key_is_not_a_gateway_key(self):
        self.assertIsNone(parse(DESKTOP))

    def test_short_prefix_is_not_guessed(self):
        self.assertIsNone(parse("agent:main:qqbot"))

    def test_empty(self):
        self.assertIsNone(parse(""))


class TestClassify(unittest.TestCase):
    def test_direct_across_platforms(self):
        for key in (QQ_DM, TG_DM):
            self.assertEqual(classify(key), "direct", key)

    def test_group_across_platforms(self):
        for key in (QQ_GROUP, TG_GROUP, DC_CHANNEL, SLACK_THREAD):
            self.assertEqual(classify(key), "group", key)

    def test_local_session_is_unknown(self):
        """桌面的键没有平台前缀，判不出来——这时必须当作不安全的。"""
        self.assertEqual(classify(DESKTOP), "unknown")

    def test_empty_is_unknown(self):
        self.assertEqual(classify(""), "unknown")

    def test_legacy_key_falls_back_to_marks(self):
        """旧格式没有 agent: 前缀，靠 :dm: 这类标记兜底。"""
        self.assertEqual(classify("some:dm:abc"), "direct")
        self.assertEqual(classify("some:group:abc"), "group")


class TestPlatformOf(unittest.TestCase):
    def test_extracts_platform(self):
        self.assertEqual(platform_of(TG_DM), "telegram")
        self.assertEqual(platform_of(DESKTOP), "")


class TestDescribe(unittest.TestCase):
    def test_readable(self):
        self.assertEqual(describe(QQ_DM), "qqbot/dm")
        self.assertEqual(describe(DESKTOP), "local/unknown")


class TestAllowedTarget(unittest.TestCase):
    def test_direct_in_allowlist_passes(self):
        ok, why = is_allowed_target(QQ_DM, "qqbot", ["qqbot"], True)
        self.assertTrue(ok, why)

    def test_empty_allowlist_blocks(self):
        ok, why = is_allowed_target(QQ_DM, "qqbot", [], True)
        self.assertFalse(ok)
        self.assertEqual(why, "no-allowlist")

    def test_platform_not_allowlisted(self):
        ok, why = is_allowed_target(TG_DM, "telegram", ["qqbot"], True)
        self.assertFalse(ok)
        self.assertEqual(why, "platform")

    def test_key_platform_must_match_config(self):
        """键里写 telegram，配置说 qqbot——不一致按更严的来。"""
        ok, why = is_allowed_target(TG_DM, "qqbot", ["qqbot"], True)
        self.assertFalse(ok)
        self.assertEqual(why, "platform-mismatch")

    def test_group_blocked_when_direct_required(self):
        ok, why = is_allowed_target(TG_GROUP, "telegram", ["telegram"], True)
        self.assertFalse(ok)
        self.assertTrue(why.startswith("not-direct"))

    def test_unknown_blocked_when_direct_required(self):
        ok, why = is_allowed_target(DESKTOP, "desktop", ["desktop"], True)
        self.assertFalse(ok)
        self.assertEqual(why, "not-direct:unknown")

    def test_group_passes_when_direct_not_required(self):
        ok, why = is_allowed_target(TG_GROUP, "telegram", ["telegram"], False)
        self.assertTrue(ok, why)


if __name__ == "__main__":
    unittest.main(verbosity=2)
