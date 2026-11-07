"""Regression test for the double-send bug.

Two heartbeats racing for the same moment must produce exactly one model call
and one injection.  Reproduces the real failure shape (hot reload left two
threads alive: two `sent` rows 0.59s apart, different wording).

Run:  ~/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe tests/race.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

HERMES_SRC = Path(os.path.expanduser("~/AppData/Local/hermes/hermes-agent"))
PLUGIN_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(HERMES_SRC))
sys.path.insert(0, str(PLUGIN_DIR))

os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="ph-race-")
logging.basicConfig(level=logging.WARNING)

from plugins.plugin_loader import load_plugin_module  # noqa: E402

mod = load_plugin_module("plugins.proactive-heartbeat", PLUGIN_DIR, parents=("plugins",),
                         logger=logging.getLogger("race"))
assert mod is not None


class R:
    def __init__(self, text):
        self.text, self.model, self.usage = text, "stub", None


class Llm:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def complete(self, messages, **kw):
        with self.lock:
            self.calls += 1
        time.sleep(0.3)  # widen the window a real model call would also open
        return R(json.dumps({"send": True, "message": "该动手了。", "reason": "截止近"}, ensure_ascii=False))


class Ctx:
    plugin_id = "proactive-heartbeat"

    def __init__(self):
        self.hooks, self.tools, self.commands, self.injected = {}, {}, {}, []
        self.llm = Llm()
        self.settings = {}

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_hook(self, name, cb):
        self.hooks.setdefault(name, []).append(cb)

    def register_tool(self, **kw):
        self.tools[kw["name"]] = kw

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = handler

    def on_unload(self, cb):
        pass

    def inject_message(self, content, role="user", *, session_key=None, origin=None):
        with self.llm.lock:
            self.injected.append(content)
        return True


ctx = Ctx()
mod.register(ctx)
mod._RUNTIME["stop"].set()

mod._handle_intent(action="add", title="提交实验报告", due="12h", kind="deadline", weight=0.8)
st = mod.S.load()
st["runtime"].update({"last_session_key": "ses_race", "last_platform": "desktop",
                      "last_user_turn_at": time.time() - 3600})
mod.S.save(st)

barrier = threading.Barrier(3)
errors = []


def worker():
    try:
        barrier.wait()   # all three fire in the same instant
        mod._tick(ctx)
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)


threads = [threading.Thread(target=worker) for _ in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("model calls:", ctx.llm.calls, "(expect 1)")
print("injections :", len(ctx.injected), "(expect 1)")
print("errors     :", errors)
print("VERDICT:", "PASS" if ctx.llm.calls == 1 and len(ctx.injected) == 1 and not errors else "FAIL")
