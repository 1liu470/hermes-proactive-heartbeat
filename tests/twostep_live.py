"""用真实宿主模型跑一次两段式裁决，确认它真的多调一次、真的在判为不发时只调一次。

与 test_twostep.py 的区别：那里是假 LLM，锁的是控制流；这里是真的 PluginLlm，
测的是"拆分之后模型还能不能给出可用的一句话"。
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

HERMES_SRC = Path(os.path.expanduser("~/AppData/Local/hermes/hermes-agent"))
PLUGIN_DIR = Path(__file__).resolve().parent.parent
for p in (str(HERMES_SRC), str(HERMES_SRC / "venv/Lib/site-packages"), str(PLUGIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.plugin_llm import PluginLlm  # noqa: E402
from plugins.plugin_loader import load_plugin_module  # noqa: E402

mod = load_plugin_module("plugins.proactive-heartbeat", PLUGIN_DIR, parents=("plugins",),
                         logger=logging.getLogger("live-test"))
Item = mod.E.Item


class Ctx:
    plugin_id = "proactive-heartbeat"

    def __init__(self):
        self.llm = PluginLlm(plugin_id="proactive-heartbeat")

    def get_config(self, key, default=None):
        return default


now = time.time()
cases = [
    ("该发：1.5 小时后截止，还没有动作", Item(id="L1", title="今晚 23:00 前上传实验报告签字页",
                                              due=now + 1.5 * 3600, weight=0.8,
                                              created_at=now - 86400)),
    ("该发：45 分钟后开会，需要带证件", Item(id="L2", title="带上学生证去学院办公室核验",
                                            due=now + 0.75 * 3600, weight=0.7,
                                            created_at=now - 3600)),
    ("不该发：三周前的旧账", Item(id="L3", title="交学生证",
                                  due=now - 21 * 86400, weight=0.6,
                                  created_at=now - 30 * 86400)),
]

for label, item in cases:
    ctx = Ctx()
    t0 = time.time()
    out = mod._ask_model(ctx, item, {}, now)
    dt = time.time() - t0
    print(f"\n=== {label}  ({dt:.1f}s)")
    if out is None:
        print("  结果：None（模型失败或写不出话 → 不发）")
    else:
        print(f"  send   : {out['send']}")
        print(f"  reason : {out['reason']}")
        print(f"  speech : {out['speech']}")
        print(f"  message: {out['message']!r}")
