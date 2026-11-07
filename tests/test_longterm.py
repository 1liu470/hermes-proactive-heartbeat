"""长期层单测：归因、周体检、周节律。

Run:  python -m unittest discover -s tests -p "test_*.py" -v
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from datetime import datetime, timezone

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
_pkg = types.ModuleType("ph_lt_pkg")
_pkg.__path__ = [_ROOT]
sys.modules["ph_lt_pkg"] = _pkg


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eng = _load("ph_lt_pkg.engine", _ROOT + "/engine.py")
lt = _load("ph_lt_pkg.longterm", _ROOT + "/longterm.py")

TZ = timezone.utc
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=TZ).timestamp()

GOALS = [
    {"id": "D0", "level": "north", "domain": "exit", "title": "推免进交大或中科大", "weight": 1.0},
    {"id": "D1", "level": "domain", "domain": "exit", "title": "分流进生物工程", "weight": 1.0},
    {"id": "D2", "level": "domain", "domain": "research", "title": "在 Ali 组形成可验证的科研产出"},
    {"id": "G1.1", "level": "goal", "domain": "academic", "title": "大一上期末全部必修 ≥85，加权均分进前 10%"},
    {"id": "G2.1", "level": "goal", "domain": "research", "title": "某课题 综述第 8 章"},
    {"id": "G3.1", "level": "goal", "domain": "contest", "title": "大创 2027 年度项目申报"},
]


def _item(title, **kw):
    return eng.Item(id=kw.pop("id", "x:1"), title=title, **kw)


class TestAttribute(unittest.TestCase):
    def test_matches_by_shared_terms(self):
        gid = lt.attribute(_item("写某课题综述的第 8 章"), GOALS)
        self.assertEqual(gid, "G2.1")

    def test_refuses_to_guess_below_threshold(self):
        self.assertIsNone(lt.attribute(_item("交学生证"), GOALS))

    def test_north_is_never_the_target(self):
        gid = lt.attribute(_item("推免进交大或中科大 准备材料"), GOALS)
        self.assertNotEqual(gid, "D0")


class TestDigest(unittest.TestCase):
    def _state(self):
        return {"goals": [dict(g) for g in GOALS]}

    def test_counts_moved_and_stalled(self):
        s = self._state()
        lt.touch(s, "G2.1", NOW - 2 * 86400)
        d = lt.weekly_digest(s, NOW)
        self.assertEqual(len(d["alive"]), 1)
        self.assertIn("research", d["alive"])
        self.assertGreaterEqual(len(d["dead"]), 1)

    def test_text_states_numbers_without_moralising(self):
        s = self._state()
        lt.touch(s, "G2.1", NOW - 86400)
        txt = lt.digest_text(lt.weekly_digest(s, NOW))
        self.assertIn("这周动了的", txt)
        for bad in ("懒惰", "加油", "不够"):
            self.assertNotIn(bad, txt)

    def test_old_touch_counts_as_stalled(self):
        s = self._state()
        lt.touch(s, "G2.1", NOW - 30 * 86400)
        d = lt.weekly_digest(s, NOW)
        self.assertNotIn("research", d["alive"])


class TestReviewCadence(unittest.TestCase):
    def test_fires_after_threshold_once_per_week(self):
        from datetime import timedelta
        sun = datetime(2026, 11, 8, 21, 0, tzinfo=TZ).timestamp()   # 周日 21:00
        s = {"longterm": {}}
        self.assertTrue(lt.review_due(s, sun, weekday=6, hour=20, tz=TZ))
        s["longterm"]["last_review_at"] = sun
        self.assertFalse(lt.review_due(s, sun + 3600, weekday=6, hour=20, tz=TZ))
        self.assertTrue(lt.review_due(s, sun + 7 * 86400, weekday=6, hour=20, tz=TZ))

    def test_not_before_hour(self):
        wed = datetime(2026, 11, 4, 12, 0, tzinfo=TZ).timestamp()
        self.assertFalse(lt.review_due({"longterm": {}}, wed, weekday=6, hour=20, tz=TZ))


class TestReviewUrgency(unittest.TestCase):
    def test_review_item_clears_the_bar(self):
        it = eng.Item(id="lt:2026-W45", title="本周目标体检", kind="review",
                      weight=0.85, created_at=NOW)
        self.assertAlmostEqual(eng.urgency(it, NOW), 0.85, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
