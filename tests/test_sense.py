"""自感知层测试：三条规则的触发与抑制。

Run:  python -m unittest discover -s tests -p "test_*.py" -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("ph_sense_pkg")
_pkg.__path__ = [_ROOT]
sys.modules["ph_sense_pkg"] = _pkg


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eng = _load("ph_sense_pkg.engine", _ROOT + "/engine.py")
res = _load("ph_sense_pkg.research", _ROOT + "/research.py")
sn = _load("ph_sense_pkg.sense", _ROOT + "/sense.py")

NOW = 1_900_000_000.0
DAY = 86400.0


def mk_item(i, title, *, kind="deadline", hours=48, tags=None):
    return {"id": i, "title": title, "kind": kind, "due": NOW + hours * 3600,
            "weight": 0.6, "created_at": NOW - DAY, "nudges": 0,
            "last_nudged_at": None, "status": "open",
            "tags": tags or [], "note": "", "type": ""}


def state_with(items=None, terms=None, goals=None, knowledge=None):
    return {"items": items or [], "runtime": {"term_stats": terms or {}},
            "goals": goals or [], "knowledge": knowledge or {}}


class TestRulePrepHole(unittest.TestCase):
    def test_attend_without_knowledge_raises_hypothesis(self):
        s = state_with([mk_item("wx:1", "周六上午在纪念馆集合")])
        hs = sn.hypotheses(s, NOW)
        self.assertTrue(any(h["rule"] == "临期无备" for h in hs))

    def test_knowledge_present_silences_it(self):
        s = state_with([mk_item("wx:1", "周六上午在纪念馆集合")],
                       knowledge={"周六上午在纪念馆集合": {"at": NOW - DAY, "hits": [{"title": "x"}]}})
        hs = sn.hypotheses(s, NOW)
        self.assertFalse(any(h["rule"] == "临期无备" for h in hs))

    def test_submit_type_is_not_covered_by_this_rule(self):
        # 要交的东西不需要"提前备资料"，这条规则只管要去/要办
        s = state_with([mk_item("wx:2", "重新上传实验报告")])
        hs = sn.hypotheses(s, NOW)
        self.assertFalse(any(h["rule"] == "临期无备" for h in hs))

    def test_far_away_deadline_does_not_fire(self):
        s = state_with([mk_item("wx:3", "下月开会", hours=24 * 20)])
        hs = sn.hypotheses(s, NOW)
        self.assertFalse(any(h["rule"] == "临期无备" for h in hs))


class TestRuleTopicHeat(unittest.TestCase):
    def _terms(self, n=6, days=None, last_age=DAY):
        days = days if days is not None else ["2026-11-01", "2026-11-03"]
        return {"大创": {"n": n, "first": NOW - 4 * DAY, "last": NOW - last_age, "days": days}}

    def test_repeated_topic_without_item_fires(self):
        s = state_with(terms=self._terms())
        hs = sn.hypotheses(s, NOW)
        self.assertTrue(any(h["rule"] == "话题加热" for h in hs))

    def test_topic_already_in_ledger_is_silent(self):
        s = state_with([mk_item("wx:9", "申请大创项目")], terms=self._terms())
        hs = sn.hypotheses(s, NOW)
        self.assertFalse(any(h["rule"] == "话题加热" for h in hs))

    def test_single_day_burst_is_not_heat(self):
        terms = self._terms(n=9, days=["2026-11-04"], last_age=1800)
        s = state_with(terms=terms)
        hs = sn.hypotheses(s, NOW)
        self.assertFalse(any(h["rule"] == "话题加热" for h in hs))

    def test_stale_topic_is_ignored(self):
        terms = self._terms(n=9, days=["2026-10-01", "2026-10-03"], last_age=10 * DAY)
        s = state_with(terms=terms)
        hs = sn.hypotheses(s, NOW)
        self.assertFalse(any(h["rule"] == "话题加热" for h in hs))


class TestRuleStall(unittest.TestCase):
    def test_stalled_goal_with_near_deadline_fires(self):
        s = state_with(
            [mk_item("wx:4", "提交申请材料", hours=24 * 5)],
            goals=[{"id": "G3.1", "level": "goal", "domain": "apply",
                    "title": "大创申报", "due": "2030-04-20",
                    "last_touch": NOW - 20 * DAY}])
        hs = sn.hypotheses(s, NOW)
        self.assertTrue(any(h["rule"] == "目标失联" for h in hs))

    def test_recently_touched_goal_is_quiet(self):
        s = state_with(
            [mk_item("wx:4", "提交申请材料", hours=24 * 5)],
            goals=[{"id": "G3.1", "level": "goal", "domain": "apply",
                    "title": "大创申报", "due": "2030-04-20",
                    "last_touch": NOW - 2 * DAY}])
        hs = sn.hypotheses(s, NOW)
        self.assertFalse(any(h["rule"] == "目标失联" for h in hs))

    def test_never_tracked_goal_is_silent_during_observation(self):
        """刚同步、从没记录过：不能说"半个月没动"，那是把"我不知道"当成"你没做"。"""
        s = state_with(
            [mk_item("wx:4", "提交申请材料", hours=24 * 5)],
            goals=[{"id": "G3.1", "level": "goal", "domain": "apply",
                    "title": "大创申报", "due": "2030-04-20"}])
        s["longterm"] = {"synced_at": NOW - 2 * DAY}
        hs = sn.hypotheses(s, NOW)
        self.assertFalse(any(h["rule"] == "目标失联" for h in hs))

    def test_never_tracked_after_observation_uses_honest_wording(self):
        s = state_with(
            [mk_item("wx:4", "提交申请材料", hours=24 * 5)],
            goals=[{"id": "G3.1", "level": "goal", "domain": "apply",
                    "title": "大创申报", "due": "2030-04-20"}])
        s["longterm"] = {"synced_at": NOW - 20 * DAY}
        hs = [h for h in sn.hypotheses(s, NOW) if h["rule"] == "目标失联"]
        self.assertEqual(len(hs), 1)
        self.assertIn("还没有任何推进记录", hs[0]["basis"])
        self.assertNotIn("半个月", hs[0]["basis"])


class TestPickGating(unittest.TestCase):
    def test_daily_cap_stops_it(self):
        s = state_with([mk_item("wx:1", "周六上午在纪念馆集合")])
        s["sense"] = {"calls": {sn.time.strftime("%Y-%m-%d"): 2}}
        self.assertIsNone(sn.pick(s, NOW, daily_cap=2))

    def test_recently_sent_hypothesis_is_skipped(self):
        s = state_with([mk_item("wx:1", "周六上午在纪念馆集合")])
        h = sn.hypotheses(s, NOW)[0]
        s["sense"] = {"done": {h["key"]: NOW - DAY}}
        self.assertIsNone(sn.pick(s, NOW, repeat_days=7))

    def test_denied_hypothesis_not_retried_within_30min(self):
        s = state_with([mk_item("wx:1", "周六上午在纪念馆集合")])
        h = sn.hypotheses(s, NOW)[0]
        s["sense"] = {"last_attempt": {"key": h["key"], "at": NOW - 600}}
        self.assertIsNone(sn.pick(s, NOW))

    def test_marks_and_counts(self):
        s = state_with([mk_item("wx:1", "周六上午在纪念馆集合")])
        h = sn.pick(s, NOW)
        self.assertIsNotNone(h)
        sn.mark(s, h["key"], NOW)
        self.assertEqual(s["sense"]["calls"][sn.time.strftime("%Y-%m-%d")], 1)
        self.assertIsNone(sn.pick(s, NOW + 60), "刚标记的同一条不该立刻再来")


class TestResearchHandoff(unittest.TestCase):
    def test_sense_item_carries_its_own_query(self):
        it = eng.Item(id="sense:topic:大创", title="大创", kind="sense",
                      created_at=NOW, note="最近提了 6 次｜查：大创 是什么 怎么做")
        self.assertTrue(res.should_research(it, NOW, 72))
        self.assertEqual(res._query_from_note(it), "大创 是什么 怎么做")


if __name__ == "__main__":
    unittest.main(verbosity=2)
