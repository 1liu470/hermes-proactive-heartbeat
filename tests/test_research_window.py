"""检索窗口要覆盖「刚错过」那几天，且不得无界扩张。

真账本里十一件事，五件已逾期、六件无期限，一次资料都没查过：旧判据下界是 0，
hours 一旦为负就立刻返回 False。于是提醒越紧急的时候，功课越早停止——
他一开始就错过的事，反而拿到一句空口提醒。

边界必须两头都钉住：逾期七天内要查（还能补），逾过一周就闭嘴（陈旧账）；
未来太远也不查，那时候查到的通知多半会改。
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERMES_SRC = Path(os.path.expanduser("~/AppData/Local/hermes/hermes-agent"))
PLUGIN_DIR = Path(__file__).resolve().parent.parent

SANDBOX = Path(tempfile.mkdtemp(prefix="ph-window-"))
os.environ["HERMES_HOME"] = str(SANDBOX)

for p in (str(HERMES_SRC), str(HERMES_SRC / "venv/Lib/site-packages"), str(PLUGIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ph_window_under_test", PLUGIN_DIR / "__init__.py",
    submodule_search_locations=[str(PLUGIN_DIR)])
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)

E, RS = mod.E, mod.RS
NOW = 1_800_000_000.0


def item(due_offset_hours, kind="deadline"):
    return E.Item.from_dict({
        "id": "t:%s" % due_offset_hours,
        "title": "申请国家助学金",
        "kind": kind,
        "status": "open",
        "created_at": NOW - 3 * 86400,
        "nudges": 0,
        "due": None if due_offset_hours is None else NOW + due_offset_hours * 3600,
    })


class WindowTests(unittest.TestCase):
    def test_upcoming_inside_window(self):
        self.assertTrue(RS.should_research(item(30), NOW, 72))

    def test_upcoming_far_outside(self):
        self.assertFalse(RS.should_research(item(100), NOW, 72))

    def test_just_overdue_still_researched(self):
        # 刚过期两小时：最该查「能不能补、找谁」。
        self.assertTrue(RS.should_research(item(-2), NOW, 72))

    def test_overdue_within_grace(self):
        self.assertTrue(RS.should_research(item(-72), NOW, 72))

    def test_overdue_past_grace(self):
        self.assertFalse(RS.should_research(item(-240), NOW, 72))

    def test_grace_zero_restores_old_behaviour(self):
        # 把宽限显式设成 0，必须回到「逾期即不查」，说明它不是靠默认值兜住的意外行为。
        self.assertFalse(RS.should_research(item(-2), NOW, 72, 0))

    def test_no_due_never_researched(self):
        self.assertFalse(RS.should_research(item(None), NOW, 72))

    def test_negative_grace_is_clamped(self):
        # 负数宽限不该把窗口反过来扩成「只查逾期」。
        self.assertTrue(RS.should_research(item(10), NOW, 72, -500))
        self.assertFalse(RS.should_research(item(-10), NOW, 72, -500))

    def test_open_loop_not_researched_by_default(self):
        # 无期限事项的标题常带微信昵称，直接发出去等于把私事送去检索。
        # 默认不查；要放开得先解决脱敏，不能靠这一条改。
        self.assertFalse(RS.should_research(item(30, kind="open_loop"), NOW, 72))

    def test_sense_uses_note(self):
        # sense 类不按时间窗判，只认 note 里显式写下的「查：」检索词。
        it = item(None, kind="sense")
        it.note = "他最近在问合成生物学竞赛"
        self.assertFalse(RS.should_research(it, NOW, 72))
        it.note = "他最近在问合成生物学竞赛 查：合成生物学 iGEM 校内选拔"
        self.assertTrue(RS.should_research(it, NOW, 72))


class GatherTests(unittest.TestCase):
    def test_overdue_item_produces_cached_hits(self):
        class Ctx:
            def __init__(self):
                self.calls = []

            def dispatch_tool(self, name, args):
                self.calls.append(args["query"])
                return {"data": {"web": [{
                    "title": "国家助学金补交说明",
                    "url": "https://example.edu/a",
                    "description": "逾期可向学院提交书面申请",
                }]}}

            def get_config(self, key, default=None):
                return default

        ctx, state = Ctx(), {"research": {}, "knowledge": {}, "research_calls": {}}
        hits = RS.gather(ctx, item(-5), state, NOW, window_hours=72)
        self.assertEqual(len(hits), 1)
        self.assertEqual(len(ctx.calls), 1)
        # 缓存与计数都要落地，否则下一轮还得重查，日限额也形同虚设。
        self.assertTrue(state["research"])
        self.assertEqual(sum(state["research_calls"].values()), 1)
        self.assertIn("补交", RS.format_intel(hits))


if __name__ == "__main__":
    unittest.main(verbosity=2)
