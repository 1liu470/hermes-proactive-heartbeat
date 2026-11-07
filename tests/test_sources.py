"""University OS signal mapping (pure functions, synthetic fixtures).

Run:  python -m unittest discover -s tests -p "test_*.py" -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# sources.py does `from . import engine`; load it as a package member instead.
_PKG = "ph_uo_pkg"
_pkg_dir = _ROOT
_spec = importlib.util.spec_from_file_location(
    _PKG, os.path.join(_pkg_dir, "__init__.py"),
    submodule_search_locations=[_pkg_dir],
)
# Loading the plugin __init__ pulls Hermes imports; load a bare namespace instead.
import types  # noqa: E402

_pkg_mod = types.ModuleType(_PKG)
_pkg_mod.__path__ = [_pkg_dir]
sys.modules[_PKG] = _pkg_mod
_load(_PKG + ".engine", os.path.join(_root := _ROOT, "engine.py"))
src = _load(_PKG + ".sources", os.path.join(_root, "sources.py"))

TZ = timezone.utc  # fixtures below are written in UTC for determinism
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=TZ).timestamp()


def _state():
    return {
        "treeStats": {
            "overdue": [
                {"id": "M2.1.1", "title": "见导师", "due": "2026-10-31",
                 "level": "milestone", "domain": "research"},
                {"id": "T1", "title": "交作业", "due": "2026-11-01", "level": "task"},
            ],
            "today3": [
                {"id": "T2", "title": "今天做", "due": "2026-11-04", "level": "task"},
                {"id": "M2.1.1", "title": "见导师", "due": "2026-10-31", "level": "milestone"},
            ],
        },
        "keyDates": [
            {"id": 11, "date": "2026-11-06", "label": "50 米考核", "kind": "exam"},
            {"id": 12, "date": "2026-12-30", "label": "太远的考试", "kind": "exam"},
            {"id": 13, "date": "2026-11-10", "label": "非考试节点", "kind": "holiday"},
        ],
        "policies": [
            {"id": "p1", "title": "大创双选", "months": [11], "window": "11 月上旬", "urgent": 1},
            {"id": "p2", "title": "春季的事", "months": [3], "window": "3 月", "urgent": 1},
            {"id": "p3", "title": "不紧急", "months": [11], "window": "11 月", "urgent": 0, "critical": 0},
        ],
    }


class TestUniversitySignals(unittest.TestCase):
    def test_overdue_becomes_weighted_deadline(self):
        items = src.items_from_state(_state(), NOW, TZ)
        by_id = {i.id: i for i in items}
        self.assertIn("uo:M2.1.1", by_id)
        it = by_id["uo:M2.1.1"]
        self.assertAlmostEqual(it.weight, 0.8, places=2)
        self.assertIsNotNone(it.due)
        self.assertLess(it.due, NOW, "overdue item keeps its past due date")
        self.assertAlmostEqual(by_id["uo:T1"].weight, 0.55, places=2)

    def test_today3_does_not_duplicate_overdue(self):
        items = src.items_from_state(_state(), NOW, TZ)
        ids = [i.id for i in items]
        self.assertEqual(ids.count("uo:M2.1.1"), 1)
        self.assertIn("uo:T2", ids)

    def test_keydates_window_and_kind(self):
        items = src.items_from_state(_state(), NOW, TZ)
        ids = {i.id for i in items}
        self.assertIn("uo:kd:11", ids)
        self.assertNotIn("uo:kd:12", ids, "beyond 14 days")
        self.assertNotIn("uo:kd:13", ids, "not an exam/milestone kind")

    def test_policy_only_current_or_next_month_and_flagged(self):
        items = src.items_from_state(_state(), NOW, TZ)
        ids = {i.id for i in items}
        self.assertIn("uo:pol:p1", ids)
        self.assertNotIn("uo:pol:p2", ids)
        self.assertNotIn("uo:pol:p3", ids, "urgent/critical must be set")
        pol = [i for i in items if i.id == "uo:pol:p1"][0]
        self.assertEqual(datetime.fromtimestamp(pol.due, TZ).day, 30, "due pinned to month end")

    def test_nudged_history_is_applied(self):
        nudged = {"uo:M2.1.1": {"at": NOW - 3600, "n": 2, "created_at": NOW - 86400}}
        items = src.items_from_state(_state(), NOW, TZ, nudged=nudged)
        it = [i for i in items if i.id == "uo:M2.1.1"][0]
        self.assertEqual(it.nudges, 2)
        self.assertAlmostEqual(it.last_nudged_at, NOW - 3600)

    def test_empty_or_malformed_state_yields_nothing(self):
        self.assertEqual(src.items_from_state({}, NOW, TZ), [])
        self.assertEqual(src.items_from_state({"treeStats": None, "policies": "x"}, NOW, TZ), [])

    def test_fetch_returns_none_when_unreachable(self):
        self.assertIsNone(src.fetch_state("http://127.0.0.1:9", timeout=0.5, cache_seconds=0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
