"""类型级偏好测试：分类与门槛调整。

Run:  python -m unittest tests.test_type_prefs -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("ph_type_pkg")
_pkg.__path__ = [_ROOT]
sys.modules["ph_type_pkg"] = _pkg

spec = importlib.util.spec_from_file_location("ph_type_pkg.engine", _ROOT + "/engine.py")
eng = importlib.util.module_from_spec(spec)
sys.modules["ph_type_pkg.engine"] = eng
spec.loader.exec_module(eng)

NOW = 1_900_000_000.0


def I(title, **kw):
    return eng.Item(id=kw.pop("id", "t:1"), title=title, **kw)


class TestItemType(unittest.TestCase):
    def test_explicit_type_wins(self):
        self.assertEqual(eng.item_type(I("随便什么", type="attend")), "attend")

    def test_infers_submit(self):
        self.assertEqual(eng.item_type(I("重新上传实验报告")), "submit")

    def test_infers_attend(self):
        self.assertEqual(eng.item_type(I("周六上午九点半在纪念馆集合")), "attend")

    def test_infers_apply(self):
        self.assertEqual(eng.item_type(I("申请奖学金")), "apply")

    def test_unknown_falls_back(self):
        self.assertEqual(eng.item_type(I("把桌子擦一下")), "other")


class TestTypeOffset(unittest.TestCase):
    def test_no_offset_below_min_samples(self):
        stats = {"submit": {"sent": 4, "engaged": 4}}
        self.assertEqual(eng.type_offset(stats, "submit"), 0.0)

    def test_high_response_lowers_bar(self):
        stats = {"submit": {"sent": 10, "engaged": 9}}
        off = eng.type_offset(stats, "submit")
        self.assertGreater(off, 0.0)
        self.assertLessEqual(off, 0.08)

    def test_low_response_raises_bar(self):
        stats = {"attend": {"sent": 10, "engaged": 0}}
        off = eng.type_offset(stats, "attend")
        self.assertLess(off, 0.0)
        self.assertGreaterEqual(off, -0.08)

    def test_offset_actually_changes_score(self):
        item = I("申请奖学金", kind="deadline", due=NOW + 3600, weight=0.6)
        base = eng.score_item(item, NOW, set())
        better = eng.score_item(item, NOW, set(), type_offset=0.08)
        self.assertGreater(better["total"], base["total"])
        self.assertEqual(better["type_offset"], 0.08)


if __name__ == "__main__":
    unittest.main(verbosity=2)
