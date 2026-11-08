"""两件长期运行的账要算清：淤积的事项该退场，做掉的事该算数。

真账本里三件事项逾期十四到二十三天还挂在 open，六件无期限事项一年也不会
自己结束；每次 tick 都要陪跑一遍打分，候选池越跑越脏。

另一头是真账本上发七次、回应零次。不是他不理，是真人直接去办了、不回「收到」。
只认文字，等于把「照做但没吭声」记成「无视」，门槛越学越偏。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

HERMES_SRC = Path(os.path.expanduser("~/AppData/Local/hermes/hermes-agent"))
PLUGIN_DIR = Path(__file__).resolve().parent.parent

SANDBOX = Path(tempfile.mkdtemp(prefix="ph-life-"))
os.environ["HERMES_HOME"] = str(SANDBOX)

for p in (str(HERMES_SRC), str(HERMES_SRC / "venv/Lib/site-packages"), str(PLUGIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ph_life_under_test", PLUGIN_DIR / "__init__.py",
    submodule_search_locations=[str(PLUGIN_DIR)])
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)

E = mod.E
NOW = 1_800_000_000.0


def mk(iid="i1", status="open", due_offset_days=None, kind="deadline"):
    return E.Item.from_dict({
        "id": iid, "title": "交学生证", "kind": kind, "status": status,
        "created_at": NOW - 20 * 86400,
        "due": None if due_offset_days is None else NOW + due_offset_days * 86400,
    })


class RetireTests(unittest.TestCase):
    def test_overdue_beyond_window_goes_stale(self):
        it = mk(due_offset_days=-8)
        out = E.retire_stale([it], NOW, days=7)
        self.assertEqual(len(out), 1)
        self.assertEqual(it.status, "stale")
        self.assertEqual(it.closed_at, NOW)

    def test_recently_overdue_survives(self):
        it = mk(due_offset_days=-3)
        self.assertEqual(E.retire_stale([it], NOW, days=7), [])
        self.assertEqual(it.status, "open")

    def test_future_untouched(self):
        it = mk(due_offset_days=2)
        self.assertEqual(E.retire_stale([it], NOW, days=7), [])
        self.assertEqual(it.status, "open")

    def test_open_loop_never_retired(self):
        # 无期限事项本来就长期存在，按时间摘它等于删掉他的承诺。
        it = mk(kind="open_loop", due_offset_days=None)
        self.assertEqual(E.retire_stale([it], NOW, days=7), [])
        self.assertEqual(it.status, "open")

    def test_already_closed_not_touched(self):
        it = mk(status="done", due_offset_days=-30)
        self.assertEqual(E.retire_stale([it], NOW, days=7), [])
        self.assertEqual(it.status, "done")
        self.assertIsNone(it.closed_at)

    def test_zero_days_retires_anything_overdue(self):
        it = mk(due_offset_days=-0.01)
        self.assertEqual(len(E.retire_stale([it], NOW, days=0)), 1)

    def test_idempotent(self):
        it = mk(due_offset_days=-30)
        E.retire_stale([it], NOW, days=7)
        self.assertEqual(E.retire_stale([it], NOW + 60, days=7), [])

    def test_nothing_to_retire_returns_empty(self):
        self.assertEqual(E.retire_stale([], NOW, days=7), [])


class EngagementTests(unittest.TestCase):
    def pend(self, iid="i1", at=None):
        return {"item_id": iid, "at": NOW - 3600 if at is None else at}

    def test_done_counts(self):
        self.assertEqual(E.infer_engagement(self.pend(), [mk(status="done")], NOW), "done")

    def test_dropped_counts(self):
        self.assertEqual(E.infer_engagement(self.pend(), [mk(status="dropped")], NOW), "done")

    def test_still_open_is_not_evidence(self):
        self.assertEqual(E.infer_engagement(self.pend(), [mk()], NOW), "")

    def test_stale_is_not_his_answer(self):
        # 系统自己摘掉的不算回应，否则等于把自己写的账当他的证词。
        self.assertEqual(E.infer_engagement(self.pend(), [mk(status="stale")], NOW), "")

    def test_vanished_item_is_not_evidence(self):
        # 归档、外部源重扫都会让事项离开主表。把它当成「他做完了」，
        # 等于自己给自己发奖，账就脏了。宁可漏记。
        self.assertEqual(E.infer_engagement(self.pend(), [], NOW), "")

    def test_outside_window_ignored(self):
        old = {"item_id": "i1", "at": NOW - 30 * 3600}
        self.assertEqual(E.infer_engagement(old, [mk(status="done")], NOW), "")

    def test_no_pending(self):
        self.assertEqual(E.infer_engagement(None, [mk(status="done")], NOW), "")
        self.assertEqual(E.infer_engagement({}, [mk(status="done")], NOW), "")

    def test_other_item_done_does_not_count(self):
        items = [mk("i1", status="open"), mk("i2", status="done")]
        self.assertEqual(E.infer_engagement(self.pend("i1"), items, NOW), "")

    def test_window_hours_configurable(self):
        p = {"item_id": "i1", "at": NOW - 3 * 3600}
        self.assertEqual(E.infer_engagement(p, [mk(status="done")], NOW, window_hours=2), "")
        self.assertEqual(E.infer_engagement(p, [mk(status="done")], NOW, window_hours=6), "done")


class ArchiveTests(unittest.TestCase):
    def test_old_closed_moves_out(self):
        it = mk(status="done")
        it.closed_at = NOW - 40 * 86400
        keep, out = E.split_closed([it], NOW, days=30)
        self.assertEqual(keep, [])
        self.assertEqual(len(out), 1)

    def test_recent_closed_stays(self):
        it = mk(status="done")
        it.closed_at = NOW - 2 * 86400
        keep, out = E.split_closed([it], NOW, days=30)
        self.assertEqual(len(keep), 1)
        self.assertEqual(out, [])

    def test_open_never_archived(self):
        it = mk(status="open")
        it.created_at = NOW - 400 * 86400
        keep, out = E.split_closed([it], NOW, days=30)
        self.assertEqual(len(keep), 1)
        self.assertEqual(out, [])

    def test_falls_back_to_last_nudged_at(self):
        # 老账本里 done 的事项多半没有 closed_at，用最后提醒时间兜底。
        it = mk(status="done")
        it.closed_at = None
        it.last_nudged_at = NOW - 60 * 86400
        keep, out = E.split_closed([it], NOW, days=30)
        self.assertEqual(len(out), 1)


class RoundTripTests(unittest.TestCase):
    def test_closed_at_survives_serialisation(self):
        it = mk(due_offset_days=-30)
        E.retire_stale([it], NOW, days=7)
        back = E.Item.from_dict(it.to_dict())
        self.assertEqual(back.status, "stale")
        self.assertEqual(back.closed_at, NOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
