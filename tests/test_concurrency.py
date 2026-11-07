"""并发写入的回归测试：钩子清掉的东西，不能被心跳的旧快照写回来。

背景——2026-11-07 在真实 state 里观察到的症状：
  · pending 停在一天前的 17:32 不消失
  · last_user_turn_at 冻结在昨天 14:35，用户在实时对话却"从未说话"
  · 同一条 item 的 feedback 被反复结算，adaptive_offset 在 0.0/0.04 之间来回跳
根因不是某一个函数，是 store 的读改写不互斥：load() 与 save() 各自持锁，
中间那段（一次 tick 里含模型调用与联网检索，数秒）是敞开的。心跳拿几秒前的
快照整份写回，就把钩子在这期间的修改抹掉了。

这组测试锁住三件事：
  1. 增量提交不会覆盖别人在这期间改的字段
  2. transaction 块内异常不落盘（不留半截状态）
  3. 源码里不再出现 load() → save() 的整份覆盖写法
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import store as S  # noqa: E402


class _Sandbox(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="ph-race-")
        self._old = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self.tmp.name
        S.save({
            "version": 1,
            "dial": "normal",
            "adaptive_offset": 0.0,
            "ignore_streak": 0,
            "items": [],
            "runtime": {
                "daily_key": "2026-11-07",
                "daily_count": 1,
                "pending": {"item_id": "msg:demo-0001", "at": 1793957545.0, "title": "确认集合"},
                "last_user_turn_at": 1793862893.0,
                "context_terms": [],
            },
            "stats": {"considered": 0, "gated": 0, "declined_by_model": 0, "sent": 3},
        })

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old
        self.tmp.cleanup()


class TestNoLostUpdate(_Sandbox):
    def test_hook_clear_survives_a_later_incremental_commit(self) -> None:
        """钩子清掉 pending，心跳随后只加计数——pending 必须仍然是空的。"""
        snap = S.load()                     # 心跳 tick 开头的快照
        self.assertIsNotNone(snap["runtime"]["pending"])

        with S.transaction() as st:         # 钩子：清 pending、刷新活跃时间
            st["runtime"]["pending"] = None
            st["runtime"]["last_user_turn_at"] = 1794062537.0

        def _arm(st):                       # 心跳事后记账：只碰自己的字段
            st["runtime"]["daily_count"] = int(st["runtime"]["daily_count"]) + 1
            st["stats"]["sent"] = int(st["stats"]["sent"]) + 1

        S.update(_arm)

        now = S.load()
        self.assertIsNone(now["runtime"]["pending"], "pending 被旧快照写回来了")
        self.assertEqual(now["runtime"]["last_user_turn_at"], 1794062537.0,
                         "last_user_turn_at 被旧快照写回去了")
        self.assertEqual(now["runtime"]["daily_count"], 2)
        self.assertEqual(now["stats"]["sent"], 4)
        self.assertEqual(snap["runtime"]["daily_count"], 1, "快照本身不该被就地改动")

    def test_two_threads_writing_different_fields_both_land(self) -> None:
        """两个线程各改一个字段，交错写入后两个改动都要在。"""
        out = {}

        def _hook():
            with S.transaction() as st:
                st["runtime"]["context_terms"] = ["周三组会"]
                time.sleep(0.05)
            out["hook"] = True

        def _tick():
            time.sleep(0.01)
            S.update(lambda st: st["runtime"].__setitem__("last_platform", "qqbot"))
            out["tick"] = True

        t1, t2 = threading.Thread(target=_hook), threading.Thread(target=_tick)
        t1.start(); t2.start(); t1.join(); t2.join()

        now = S.load()
        self.assertEqual(now["runtime"]["context_terms"], ["周三组会"])
        self.assertEqual(now["runtime"]["last_platform"], "qqbot")

    def test_transaction_does_not_persist_on_exception(self) -> None:
        """事务块里抛异常，磁盘保持原样，不留半截写入。"""
        with self.assertRaises(RuntimeError):
            with S.transaction() as st:
                st["runtime"]["pending"] = None
                st["runtime"]["daily_count"] = 99
                raise RuntimeError("boom")

        now = S.load()
        self.assertIsNotNone(now["runtime"]["pending"])
        self.assertEqual(now["runtime"]["daily_count"], 1)


class TestNoWholeFileOverwrite(unittest.TestCase):
    def test_init_has_no_load_then_save(self) -> None:
        """源码里不该再有 `S.save(state)` 这种整份覆盖。

        全量写回是这一整类 bug 的源头：只要调用方手里有一份旧快照，
        写回去就等于撤销别人在这期间做的一切。允许 store.py 内部使用（那才是
        实现），也允许 `_handle_*` 之外确实需要重建整份状态的迁移代码。
        """
        src = (_ROOT / "__init__.py").read_text(encoding="utf-8")
        hits = [ln for ln in src.splitlines() if re.search(r"\bS\.save\(", ln)]
        self.assertEqual(hits, [], f"仍有整份覆盖写法：{hits}")


if __name__ == "__main__":
    unittest.main()
