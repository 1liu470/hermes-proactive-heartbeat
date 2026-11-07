"""反馈记账必须幂等：一条提醒被扫过十次，也只算一笔。

真机账本：一条事项（wx:…）在 22:55、22:56、22:58、23:24、23:50 被判了
五次 ignored —— 因为每轮对话都问一次"他回了吗"，而原来每问一次就记一笔。
ignore_streak 涨到 5，门槛被自己吓高；用户可能只是还没到桌前。

同时锁住第二条：判过一次 ignored 之后 pending 不能清掉。窗口还没关，
他五分钟后回「收到」，应该能升级成 engaged，而不是被当成一条早已作废的记录。
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
import time
import unittest
from pathlib import Path

HERMES_SRC = Path(os.path.expanduser("~/AppData/Local/hermes/hermes-agent"))
PLUGIN_DIR = Path(__file__).resolve().parent.parent

# 必须在导入插件之前定好 HERMES_HOME，否则插件会写到真实账本上。
SANDBOX = Path(tempfile.mkdtemp(prefix="ph-idem-"))
os.environ["HERMES_HOME"] = str(SANDBOX)

for p in (str(HERMES_SRC), str(HERMES_SRC / "venv/Lib/site-packages"), str(PLUGIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ph_idempotent_under_test", PLUGIN_DIR / "__init__.py",
    submodule_search_locations=[str(PLUGIN_DIR)])
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


class StubCtx:
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
        pass

    def inject_message(self, *a, **k):
        return True


class TestFeedbackIsRecordedOnce(unittest.TestCase):
    def setUp(self):
        self.ctx = StubCtx()
        mod.register(self.ctx)
        S = mod.S
        now = time.time()

        # 两个用例共用一个 sandbox 目录，必须回到干净账本，否则上一条的
        # ignore_streak 会带过来，断言就成了"累计值"而不是"这次记了几笔"。
        def seed(st):
            st.clear()
            st.update(json.loads(json.dumps(S.DEFAULTS)))
            st["runtime"].update({
                "pending": {"item_id": "it_x", "at": now - 120, "session_key": "k",
                            "platforms": "qqbot", "title": "交实验报告", "type": "submit"},
                "last_user_turn_at": now,
            })
        S.update(seed)
        self.now = now

    def tearDown(self):
        mod._RUNTIME["stop"].set()

    def _say(self, text):
        return mod._on_pre_llm_call(user_message=text, platform="qqbot",
                                    session_id="s1", turn_id=1)

    def _state(self):
        return mod.S.load()

    def test_five_scans_record_one_ignore(self):
        for i in range(5):
            self._say(f"在吗，随便聊聊第{i}句")
        st = self._state()
        self.assertEqual(st["ignore_streak"], 1,
                         "一轮对话扫一次，不该每次都记一笔无视")
        self.assertIsNotNone(st["runtime"]["pending"],
                             "窗口还没关，pending 要留着等他回")

    def test_reply_after_ignore_still_upgrades_to_engaged(self):
        for i in range(3):
            self._say(f"随便聊聊第{i}句")
        self.assertEqual(self._state()["ignore_streak"], 1)
        self._say("收到，实验报告我这就交")
        st = self._state()
        self.assertEqual(st["ignore_streak"], 0, "他真回了，连续无视要清零")
        self.assertIsNone(st["runtime"]["pending"], "结算完就该收摊")
        self.assertEqual(st.get("type_stats", {}).get("submit", {}).get("engaged"), 1)


if __name__ == "__main__":
    unittest.main()
