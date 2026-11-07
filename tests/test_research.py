"""研究层测试：主题键、知识库跨事项复用、每日上限。

Run:  python -m unittest discover -s tests -p "test_*.py" -v
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("ph_res_pkg")
_pkg.__path__ = [_ROOT]
sys.modules["ph_res_pkg"] = _pkg


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eng = _load("ph_res_pkg.engine", _ROOT + "/engine.py")
res = _load("ph_res_pkg.research", _ROOT + "/research.py")

NOW = 1_900_000_000.0


class _LLM:
    def complete(self, *a, **kw):
        raise RuntimeError("no model in unit test")


class Ctx:
    def __init__(self):
        self.calls = 0
        self.llm = _LLM()

    def dispatch_tool(self, name, args, **kw):
        self.calls += 1
        return json.dumps({"data": {"web": [
            {"title": "办事指南", "url": "https://example.com/a", "description": "需要带三份材料"}]}})


def item(i, title, hours=24):
    return eng.Item(id=i, title=title, kind="deadline", due=NOW + hours * 3600, weight=0.6)


class TestTopicKey(unittest.TestCase):
    def test_strips_year_and_space(self):
        self.assertEqual(res._topic_key("大创 2027 年度项目申报"), "大创年度项目申报")

    def test_keeps_distinct_topics_distinct(self):
        self.assertNotEqual(res._topic_key("奖学金 材料"), res._topic_key("大创 申报"))


class TestKnowledgeReuse(unittest.TestCase):
    def test_second_similar_item_hits_cache(self):
        ctx, state = Ctx(), {}
        first = res.gather(ctx, item("wx:1", "大创项目申报材料"), state, NOW)
        self.assertEqual(len(first), 1, "首查应拿到情报")
        calls_after_first = ctx.calls

        # 另一件事，同一主题：不该再联网
        second = res.gather(ctx, item("wx:2", "大创项目申报 提交"), state, NOW + 3600)
        self.assertEqual(len(second), 1)
        self.assertEqual(ctx.calls, calls_after_first, "同主题应复用知识库，不再检索")

    def test_unrelated_item_still_searches(self):
        ctx, state = Ctx(), {}
        res.gather(ctx, item("wx:1", "大创项目申报材料"), state, NOW)
        before = ctx.calls
        res.gather(ctx, item("wx:2", "申请奖学金材料"), state, NOW + 60)
        self.assertEqual(ctx.calls, before + 1)


class TestDailyCap(unittest.TestCase):
    def test_cap_blocks_further_research(self):
        ctx, state = Ctx(), {"research_calls": {res.time.strftime("%Y-%m-%d"): 6}}
        hits = res.gather(ctx, item("wx:9", "随便一件要紧的事"), state, NOW, daily_cap=6)
        self.assertEqual(hits, [])
        self.assertEqual(ctx.calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
