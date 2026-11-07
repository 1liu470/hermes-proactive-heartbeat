"""Unit tests for the pure decision engine (no Hermes imports, no network).

Run:  python -m unittest discover -s <plugin dir> -p "test_*.py" -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_PATH = os.path.join(os.path.dirname(_HERE), "engine.py")
_spec = importlib.util.spec_from_file_location("ph_engine_under_test", _ENGINE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["ph_engine_under_test"] = _mod  # dataclasses resolves the module during exec
_spec.loader.exec_module(_mod)

Item = _mod.Item
DIALS = _mod.DIALS
classify_feedback = _mod.classify_feedback
gate = _mod.gate
in_quiet_hours = _mod.in_quiet_hours
next_adaptive_offset = _mod.next_adaptive_offset
novelty = _mod.novelty
parse_hhmm = _mod.parse_hhmm
relevance = _mod.relevance
score_item = _mod.score_item
terms = _mod.terms
urgency = _mod.urgency

NOW = 1_800_000_000.0  # fixed clock: 2027-01-15-ish


def _item(**kw):
    base = dict(id="it_1", title="提交生物工程实验报告", kind="deadline",
                due=NOW + 86400, weight=0.6, created_at=NOW - 86400)
    base.update(kw)
    return Item(**base)


class TestTimeWindows(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_hhmm("23:00"), 1380)
        self.assertEqual(parse_hhmm("8:05"), 485)
        self.assertIsNone(parse_hhmm("25:00"))
        self.assertIsNone(parse_hhmm("bad"))

    def test_wrap_midnight(self):
        start, end = parse_hhmm("23:00"), parse_hhmm("08:00")
        self.assertTrue(in_quiet_hours(23 * 60 + 30, start, end))
        self.assertTrue(in_quiet_hours(3 * 60, start, end))
        self.assertFalse(in_quiet_hours(12 * 60, start, end))
        self.assertFalse(in_quiet_hours(8 * 60, start, end))

    def test_same_day_window(self):
        start, end = parse_hhmm("13:00"), parse_hhmm("14:00")
        self.assertTrue(in_quiet_hours(13 * 60 + 1, start, end))
        self.assertFalse(in_quiet_hours(14 * 60, start, end))


class TestSalience(unittest.TestCase):
    def test_urgency_is_monotonic_near_due(self):
        far = urgency(_item(due=NOW + 5 * 86400), NOW)
        near = urgency(_item(due=NOW + 3 * 86400), NOW)
        close = urgency(_item(due=NOW + 6 * 3600), NOW)
        overdue = urgency(_item(due=NOW - 60), NOW)
        self.assertLess(far, near)
        self.assertLess(near, close)
        self.assertGreater(overdue, 0.95, "just-missed stays near the top")
        self.assertLessEqual(overdue, 1.0)
        stale = urgency(_item(due=NOW - 14 * 86400), NOW)
        self.assertLess(stale, 0.30, "two weeks late is a backlog item, not an emergency")
        self.assertGreaterEqual(stale, 0.20, "but it never leaves the pool entirely")
        self.assertGreater(far, 0.0)

    def test_open_loop_never_saturates(self):
        u = urgency(Item(id="x", title="读文献", kind="open_loop", created_at=NOW - 10 * 86400), NOW)
        self.assertLessEqual(u, 0.55)

    def test_novelty_refractory(self):
        it = _item(nudges=1, last_nudged_at=NOW - 3600)
        self.assertEqual(novelty(it, NOW), 0.0)
        it2 = _item(nudges=1, last_nudged_at=NOW - 3 * 86400)
        self.assertAlmostEqual(novelty(it2, NOW), 0.5, places=3)

    def test_relevance_cjk_bigrams(self):
        it = _item(title="合成生物学 实验报告 提交")
        ctx = terms("今天在写合成生物学的实验报告")
        self.assertGreater(relevance(it, ctx), 0.0)
        self.assertEqual(relevance(it, terms("天气 股票 篮球")), 0.0)

    def test_score_components_present(self):
        comp = score_item(_item(), NOW, terms("实验报告"))
        for key in ("total", "urgency", "importance", "novelty", "relevance", "cost"):
            self.assertIn(key, comp)


class TestGate(unittest.TestCase):
    def _gate(self, **kw):
        args = dict(items=[_item()], now=NOW, local_hm=14 * 60, dial="normal",
                    quiet_hours=("23:00", "08:00"), min_idle_minutes=25,
                    cooldown_minutes=90, daily_budget=3, daily_count=0,
                    last_injection_at=None, last_user_turn_at=NOW - 3600,
                    ignore_streak=0, context_terms=set(), adaptive_offset=0.0)
        args.update(kw)
        return gate(**args)

    def test_passes_for_imminent_deadline(self):
        v = self._gate()
        self.assertTrue(v.allowed, v.reason)
        self.assertEqual(v.item.id, "it_1")

    def test_blocks_while_user_is_talking(self):
        v = self._gate(last_user_turn_at=NOW - 60)
        self.assertFalse(v.allowed)
        self.assertIn("user_active", v.reason)

    def test_urgent_lane_lets_a_closing_deadline_through(self):
        """静默 3 分钟 + 硬期限 4 小时内 → 走紧急通道，不被 idle 拦。"""
        v = self._gate(items=[_item(due=NOW + 4 * 3600, weight=0.8)],
                       last_user_turn_at=NOW - 200)
        self.assertTrue(v.allowed, v.reason)
        self.assertTrue(v.urgent)
        self.assertEqual(v.reason, "urgent")

    def test_urgent_lane_still_blocks_far_deadlines(self):
        v = self._gate(items=[_item(due=NOW + 30 * 3600, weight=0.8)],
                       last_user_turn_at=NOW - 200)
        self.assertFalse(v.allowed)
        self.assertIn("user_active", v.reason)

    def test_urgent_lane_respects_the_three_minute_floor(self):
        v = self._gate(items=[_item(due=NOW + 4 * 3600, weight=0.8)],
                       last_user_turn_at=NOW - 30)
        self.assertFalse(v.allowed, "刚说完话不该被紧急通道打断")

    def test_blocks_on_budget_and_cooldown(self):
        self.assertIn("daily_budget", self._gate(daily_count=3).reason)
        self.assertIn("cooldown", self._gate(last_injection_at=NOW - 600).reason)

    def test_blocks_in_quiet_hours(self):
        self.assertEqual(self._gate(local_hm=2 * 60).reason, "quiet_hours")

    def test_off_dial_is_silent(self):
        self.assertEqual(self._gate(dial="off").reason, "dial=off")

    def test_far_future_item_below_threshold(self):
        v = self._gate(items=[_item(due=NOW + 30 * 86400, weight=0.2)])
        self.assertFalse(v.allowed)
        self.assertIn("below_threshold", v.reason)

    def test_nudged_item_does_not_retrigger(self):
        v = self._gate(items=[_item(nudges=2, last_nudged_at=NOW - 1800)])
        self.assertFalse(v.allowed)

    def test_nudge_cap_is_hard_even_when_score_is_high(self):
        """分数够高也不能无限复述。同一条会址集合三天里发过三次，就是缺这道闸。"""
        v = self._gate(items=[_item(nudges=2, due=NOW + 86400, weight=1.0)])
        self.assertFalse(v.allowed)
        self.assertEqual(v.reason, "no_open_items")

    def test_nudge_cap_grants_one_extra_for_a_closing_deadline(self):
        """硬期限临门允许突破一次——那是最后一次值得打断的机会。"""
        v = self._gate(items=[_item(nudges=2, due=NOW + 3 * 3600, weight=1.0)])
        self.assertTrue(v.allowed)
        again = self._gate(items=[_item(nudges=3, due=NOW + 3 * 3600, weight=1.0)])
        self.assertFalse(again.allowed)

    def test_nudge_cap_is_configurable(self):
        v = self._gate(items=[_item(nudges=3, due=NOW + 86400, weight=1.0)], max_nudges=4)
        self.assertTrue(v.allowed)

    def test_adaptive_offset_raises_the_bar(self):
        ok = self._gate()
        self.assertTrue(ok.allowed)
        strict = self._gate(adaptive_offset=0.14, items=[_item(weight=0.45, due=NOW + 20 * 3600)])
        self.assertFalse(strict.allowed)

    def test_skip_reason_wins(self):
        self.assertIn("platform:cron", self._gate(skip_reason="platform:cron").reason)


class TestFeedback(unittest.TestCase):
    def test_classification(self):
        pending = {"at": NOW - 60}
        self.assertEqual(classify_feedback(pending, NOW, True), "engaged")
        self.assertIsNone(classify_feedback(pending, NOW, False))
        self.assertEqual(classify_feedback(pending, NOW + 3600, False), "ignored")
        self.assertIsNone(classify_feedback(None, NOW, True))

    def test_expired_probe_is_dropped_not_judged(self):
        """隔了一天再回一句"好的"，不该算有效，也不该算无视——两者都在污染账本。"""
        stale = {"at": NOW - 13 * 3600, "title": "确认集合"}
        self.assertEqual(classify_feedback(stale, NOW, True, "好的"), "expired")
        # 窗口内仍正常归因
        fresh = {"at": NOW - 10 * 60, "title": "确认集合"}
        self.assertEqual(classify_feedback(fresh, NOW, True, "好的"), "engaged")
        # 边界：正好 12 小时仍算窗口内
        edge = {"at": NOW - 12 * 3600 + 60, "title": "确认集合"}
        self.assertNotEqual(classify_feedback(edge, NOW, True, "好的"), "expired")

    def test_offset_moves_both_ways_with_caps(self):
        self.assertEqual(next_adaptive_offset(0.0, "ignored"), 0.04)
        self.assertEqual(next_adaptive_offset(0.12, "ignored"), 0.14)
        self.assertEqual(next_adaptive_offset(0.14, "ignored"), 0.14)
        self.assertEqual(next_adaptive_offset(0.04, "engaged"), 0.01)
        self.assertEqual(next_adaptive_offset(0.0, "engaged"), 0.0)

    def test_dials_are_ordered(self):
        self.assertLess(DIALS["eager"][0], DIALS["normal"][0])
        self.assertLess(DIALS["normal"][0], DIALS["quiet"][0])
        self.assertEqual(DIALS["off"][1], 0)



class TestContinuityCurve(unittest.TestCase):
    """长期承诺不能因为「放得久」而被折价。

    旧实现 0.97**days 在第 28 天只剩 0.42，比昨天新建的事项还轻——
    一个扛了三周还没办的事，恰恰是「还在」才更该说。这条测试锁住曲线。
    """

    def _val(self, days):
        base = 1_700_000_000.0
        it = Item(id="c1", title="扛着的承诺", created_at=base)
        return _mod.continuity(it, base + days * 86400.0)

    def test_fresh_item_is_full_weight(self):
        self.assertAlmostEqual(self._val(0), 1.0, places=6)

    def test_bottom_is_around_day_ten(self):
        vals = {d: self._val(d) for d in range(7, 22)}
        bottom = min(vals, key=vals.get)
        self.assertIn(bottom, (9, 10, 11), msg=f"谷底落在第 {bottom} 天")
        self.assertGreater(vals[bottom], 0.74)
        self.assertLess(vals[bottom], 0.78)

    def test_long_arc_returns_to_full_weight(self):
        self.assertGreaterEqual(self._val(28), 0.999)
        self.assertGreaterEqual(self._val(40), 0.999)

    def test_never_below_three_quarters(self):
        for d in range(0, 61):
            self.assertGreaterEqual(self._val(d), 0.74, msg=f"day {d}")

    def test_day_ten_beats_plain_decay(self):
        """旧公式同一天只有 0.737，新曲线必须高于它。"""
        self.assertGreater(self._val(10), 0.97 ** 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
