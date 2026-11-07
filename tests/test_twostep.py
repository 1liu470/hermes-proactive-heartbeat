"""两段式裁决：先判断，判为发才写话。

要锁住的是三件事：
1. 判为不发 → 只调用一次模型（不能因为"顺手写一句"而多花钱多花时间）
2. 判为发   → 两次调用，且最终发给用户的是第二段产出的那句话
3. 写不出话 → 整次放弃，不硬发半句
"""

from __future__ import annotations

import logging
import os
import sys
import time
import unittest
from pathlib import Path

HERMES_SRC = Path(os.path.expanduser("~/AppData/Local/hermes/hermes-agent"))
PLUGIN_DIR = Path(__file__).resolve().parent.parent
for p in (str(HERMES_SRC), str(HERMES_SRC / "venv/Lib/site-packages"), str(PLUGIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from plugins.plugin_loader import load_plugin_module  # noqa: E402

mod = load_plugin_module("plugins.proactive-heartbeat", PLUGIN_DIR, parents=("plugins",),
                         logger=logging.getLogger("twostep-test"))
assert mod is not None

Item = mod.E.Item


class Reply:
    def __init__(self, text):
        self.text = text


class Llm:
    """按顺序吐出预设回复，并记录每次调用的 purpose。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, **kw):
        self.calls.append(kw.get("purpose"))
        if not self.replies:
            return Reply("")
        return Reply(self.replies.pop(0))

    @property
    def count(self):
        return len(self.calls)


class Ctx:
    plugin_id = "proactive-heartbeat"

    def __init__(self, replies):
        self.llm = Llm(replies)

    def get_config(self, k, d=None):
        return d


def make_item(now):
    return Item(id="t1", title="交实验报告", due=now + 3600, weight=0.8,
                created_at=now - 2 * 86400)


class TestTwoStep(unittest.TestCase):
    def setUp(self):
        self.now = time.time()

    def test_declined_costs_one_call(self):
        ctx = Ctx(['{"send": false, "reason": "没有可执行的下一步"}'])
        out = mod._ask_model(ctx, make_item(self.now), {}, self.now)
        self.assertIsNotNone(out)
        self.assertFalse(out["send"])
        self.assertEqual(ctx.llm.count, 1, "判为不发时不该再调一次措辞")
        self.assertEqual(ctx.llm.calls, ["proactive-judge"])

    def test_send_costs_two_and_uses_written_line(self):
        ctx = Ctx(['{"send": true, "reason": "今天截止", "used": [1], "speech": "action"}',
                   "今天下午三点前把报告传上去，别等到晚上系统卡。"])
        out = mod._ask_model(ctx, make_item(self.now), {}, self.now)
        self.assertTrue(out["send"])
        self.assertEqual(ctx.llm.count, 2)
        self.assertEqual(ctx.llm.calls, ["proactive-judge", "proactive-write"])
        self.assertIn("下午三点前", out["message"])
        self.assertEqual(out["speech"], "action")

    def test_empty_draft_aborts_the_reminder(self):
        """写不出话，就不发——宁可沉默也不发半句。"""
        ctx = Ctx(['{"send": true, "reason": "今天截止"}', "   "])
        self.assertIsNone(mod._ask_model(ctx, make_item(self.now), {}, self.now))
        self.assertEqual(ctx.llm.count, 2)

    def test_broken_judge_json_stays_silent(self):
        ctx = Ctx(["我觉得应该发吧"])
        self.assertIsNone(mod._ask_model(ctx, make_item(self.now), {}, self.now))
        self.assertEqual(ctx.llm.count, 1)

    def test_model_error_stays_silent(self):
        class Boom(Llm):
            def complete(self, messages, **kw):
                self.calls.append(kw.get("purpose"))
                raise RuntimeError("model offline")

        ctx = Ctx([])
        ctx.llm = Boom([])
        self.assertIsNone(mod._ask_model(ctx, make_item(self.now), {}, self.now))
        self.assertEqual(ctx.llm.count, 1)


class TestDraftCleaning(unittest.TestCase):
    def test_strips_quotes(self):
        self.assertEqual(mod._clean_draft('「明早八点前发过去」'), "明早八点前发过去")

    def test_strips_a_json_envelope(self):
        """模型在"只写一句话"的阶段仍回 JSON 时，剥壳而不能原样发出去。

        simulate 的 stub 就撞出过这个：注入正文里整段是
        {"send": true, "message": ..., "reason": ...}。真模型目前没犯，
        但这条路径此前没有任何防护，模型换一种说法就会原样发到用户手机上。
        """
        shell = '{"send": true, "message": "今晚十点前把稳态推导补完", "reason": "临近截止"}'
        self.assertEqual(mod._clean_draft(shell), "今晚十点前把稳态推导补完")
        self.assertEqual(mod._clean_draft('{"text": "回一句收到就行"}'), "回一句收到就行")
        # 剥不出可用的句子就交空串，让调用方整次作废，而不是发一段 JSON
        self.assertEqual(mod._clean_draft('{"send": false}'), "")
        self.assertEqual(mod._clean_draft('{"broken": '), '{"broken":')

    def test_strips_code_fence(self):
        self.assertEqual(mod._clean_draft("```\n明早八点前发过去\n```"), "明早八点前发过去")

    def test_keeps_only_first_paragraph(self):
        self.assertEqual(mod._clean_draft("明早八点前发过去\n\n（这句话的意思是…）"),
                         "明早八点前发过去")

    def test_empty_stays_empty(self):
        self.assertEqual(mod._clean_draft(""), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
