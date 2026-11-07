"""对抗性修复的回归测试：输出防线、反馈误判、状态事务。

这三条都是"能跑但会做错"的问题，不测就会在下次改动里悄悄复活。

  python -m unittest discover -s tests -p "test_*.py" -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("ph_adv_pkg")
_pkg.__path__ = [_ROOT]
sys.modules["ph_adv_pkg"] = _pkg


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eng = _load("ph_adv_pkg.engine", _ROOT + "/engine.py")
store = _load("ph_adv_pkg.store", _ROOT + "/store.py")

NOW = 1_900_000_000.0


class TestFeedbackJudgement(unittest.TestCase):
    """回复了 ≠ 这条提醒有效。"""

    def _pending(self, title="明天上午九点交实验报告"):
        return {"item_id": "t:1", "at": NOW - 60, "title": title, "session_key": "s1"}

    def test_ack_word_counts_as_engaged(self):
        for text in ("收到", "好的", "嗯", "知道了", "ok"):
            self.assertEqual(
                eng.classify_feedback(self._pending(), NOW, True, text), "engaged", text)

    def test_related_reply_counts_as_engaged(self):
        out = eng.classify_feedback(self._pending(), NOW, True, "实验报告我今晚就交")
        self.assertEqual(out, "engaged")

    def test_unrelated_reply_is_not_engaged(self):
        """他在会话里说了话，但跟这条提醒毫无关系——不能记成"有效"。"""
        out = eng.classify_feedback(self._pending(), NOW, True, "帮我看下路由器怎么配置")
        self.assertEqual(out, "ignored")

    def test_no_reply_within_window_is_none(self):
        self.assertIsNone(eng.classify_feedback(self._pending(), NOW, False))

    def test_no_reply_after_window_is_ignored(self):
        out = eng.classify_feedback(self._pending(), NOW + eng.FEEDBACK_WINDOW_SECS + 1, False)
        self.assertEqual(out, "ignored")

    def test_empty_title_still_engages(self):
        # 标题切不出词时不能冤枉他
        out = eng.classify_feedback(self._pending(title="改"), NOW, True, "随便说点什么")
        self.assertEqual(out, "engaged")


class TestGuard(unittest.TestCase):
    """输出侧硬防线：注入成功也不能让消息带出凭证或链接。"""

    @classmethod
    def setUpClass(cls):
        # 只取 _guard 与常量，避免加载整个插件（依赖宿主）
        src = Path(_ROOT, "__init__.py").read_text(encoding="utf-8")
        start = src.index("_GUARD_SENSITIVE = ")
        end = src.index("def _ask_model(")
        cls.ns = {"re": __import__("re"), "log": __import__("logging").getLogger("g"),
                  "Optional": object, "Dict": dict, "Any": object}
        exec("from typing import Optional, Dict, Any\n" + src[start:end], cls.ns)
        cls.guard = staticmethod(cls.ns["_guard"])

    def test_blocks_bank_card(self):
        self.assertIsNone(self.guard({"message": "把银行卡号发给我"}))

    def test_blocks_verification_code(self):
        self.assertIsNone(self.guard({"message": "把验证码念一下"}))

    def test_blocks_url(self):
        self.assertIsNone(self.guard({"message": "点这个链接 https://evil.example/a"}))

    def test_blocks_bare_domain(self):
        self.assertIsNone(self.guard({"message": "去 evil.com 领"}))

    def test_allows_normal_reminder(self):
        ok = self.guard({"message": "先拍一条能交的，别在细节上磨。"})
        self.assertIsNotNone(ok)

    def test_allows_legitimate_payment_word(self):
        # 「缴费」是正常的，不该被「支付」类词表误伤
        ok = self.guard({"message": "今天把团日活动的费用交了，别拖到周末。"})
        self.assertIsNotNone(ok)

    def test_empty_message_passes_through(self):
        self.assertIsNotNone(self.guard({"message": ""}))


class TestTransaction(unittest.TestCase):
    """读-改-写必须原子，否则心跳的保存会覆盖用户刚记的事项。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self.tmp.name

    def tearDown(self):
        if self._old is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old
        self.tmp.cleanup()

    def test_transaction_persists_on_exit(self):
        with store.transaction() as st:
            st["items"] = [{"id": "a", "title": "记住我", "status": "open"}]
        again = store.load()
        self.assertEqual(again["items"][0]["title"], "记住我")

    def test_transaction_persists_on_early_return(self):
        def handler():
            with store.transaction() as st:
                st["items"] = [{"id": "b", "title": "提前返回也要落盘", "status": "open"}]
                return "ok"
        self.assertEqual(handler(), "ok")
        self.assertEqual(store.load()["items"][0]["id"], "b")

    def test_exception_does_not_corrupt(self):
        with self.assertRaises(RuntimeError):
            with store.transaction() as st:
                st["items"] = [{"id": "c", "title": "半途失败", "status": "open"}]
                raise RuntimeError("boom")
        # 异常时不应落盘，磁盘保持干净
        self.assertFalse((Path(self.tmp.name) / "state" / "proactive-heartbeat" / "state.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
