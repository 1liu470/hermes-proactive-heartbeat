"""End-to-end simulation of proactive-heartbeat against a stub host context.

Loads the plugin through Hermes's real plugin loader (so relative imports and
the `register(ctx)` contract are exercised), then drives one full tick:
intent -> gate -> model verdict -> inject_message -> state write.

Run with the Hermes runtime interpreter:
  ~/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe simulate.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

HERMES_SRC = Path(os.path.expanduser("~/AppData/Local/hermes/hermes-agent"))
# Simulate the source tree, not the installed copy, so the test can never pass
# against stale code.
PLUGIN_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(HERMES_SRC))
sys.path.insert(0, str(PLUGIN_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Freeze the clock at 14:00 today. The gate refuses to speak 23:00–08:00, which
# is correct behaviour and would make this script fail every night after 11pm —
# a red light that means "you ran it late", not "the plugin is broken".
_real_time = time.time
_midday = time.localtime(_real_time())
_frozen = time.mktime((_midday.tm_year, _midday.tm_mon, _midday.tm_mday, 14, 0, 0, 0, 0, -1))
time.time = lambda: _frozen

# Isolate state so a simulation never touches the live intent ledger.
SANDBOX = Path(tempfile.mkdtemp(prefix="ph-sim-"))
os.environ["HERMES_HOME"] = str(SANDBOX)
print("sandbox HERMES_HOME =", SANDBOX)

from plugins.plugin_loader import load_plugin_module  # noqa: E402

mod = load_plugin_module(
    "plugins.proactive-heartbeat", PLUGIN_DIR, parents=("plugins",), logger=logging.getLogger("sim")
)
assert mod is not None, "plugin module failed to load"
print("module:", mod.__name__, "| siblings:",
      [n for n in sys.modules if n.startswith("plugins.proactive-heartbeat")])


class StubLlmResult:
    def __init__(self, text):
        self.text = text
        self.model = "stub"
        self.usage = None


class StubLlm:
    def __init__(self):
        self.calls = []

    def complete(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        return StubLlmResult(
            json.dumps(
                {"send": True, "message": "今晚十一点前把实验报告的稳态推导补完，别拖到明早。",
                 "reason": "截止在12小时后且有具体动作"},
                ensure_ascii=False,
            )
        )


class StubCtx:
    plugin_id = "proactive-heartbeat"

    def __init__(self):
        self.hooks = {}
        self.tools = {}
        self.commands = {}
        self.injected = []
        self.llm = StubLlm()
        self._unload = []
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
        self._unload.append(cb)

    def inject_message(self, content, role="user", *, session_key=None, origin=None):
        self.injected.append({"role": role, "session_key": session_key, "content": content})
        return True


ctx = StubCtx()
mod.register(ctx)
print("hooks:", list(ctx.hooks), "| tools:", list(ctx.tools), "| commands:", list(ctx.commands))
mod._RUNTIME["stop"].set()  # keep the background thread quiet; we drive ticks by hand

# --- 1. record an intent through the public tool ---------------------------
out = mod._handle_intent(action="add", title="提交生物工程实验报告", due="12h", kind="deadline", weight=0.8)
print("intent add ->", out)
item_id = json.loads(out)["id"]

# --- 2. prime the runtime as if a session had been idle for an hour --------
st = mod.S.load()
st["runtime"].update({
    "last_session_key": "agent:main:qqbot:dm:sim0001",
    "last_platform": "qqbot",
    # 主动消息只进私聊；桌面的纯会话号判不出 chat_type，会被闸门拒掉，
    # 所以这里造一个 QQ 私聊会话，模拟"人在 QQ 里说过话"。
    "sessions": {"qqbot": {"key": "agent:main:qqbot:dm:sim0001", "at": time.time()}},
    "last_user_turn_at": time.time() - 3600,
    "context_terms": ["实验报告", "稳态"],
})
st["dial"] = "normal"
mod.S.save(st)

# --- 3. one real tick ------------------------------------------------------
mod._tick(ctx)
print("model calls:", len(ctx.llm.calls))
print("injected:", len(ctx.injected))
if ctx.injected:
    print("--- injected directive ---")
    print(ctx.injected[0]["content"])
print("--- decisions.jsonl ---")
print(mod.S.decisions_path().read_text(encoding="utf-8").strip())

# --- 4. a second immediate tick must be blocked by cooldown ----------------
before = len(ctx.injected)
mod._tick(ctx)
print("second tick injected delta:", len(ctx.injected) - before, "(expect 0: cooldown)")

# --- 5. feedback loop: user answers -> engaged, streak resets -------------
pending = mod.S.load()["runtime"]["pending"]
print("pending armed:", bool(pending), "| live session_key:", repr(mod._session_env("HERMES_SESSION_KEY")))
mod._on_pre_llm_call(session_id="s", user_message="好，我去写。", platform="desktop", is_first_turn=False)
st = mod.S.load()
print("after reply -> ignore_streak:", st["ignore_streak"], "offset:", st["adaptive_offset"], "pending:", st["runtime"]["pending"])

# --- 6. self-injection must not count as user activity --------------------
before_ts = mod.S.load()["runtime"]["last_user_turn_at"]
prev_content = ctx.injected[0]["content"]
mod._on_pre_llm_call(session_id="s", user_message=prev_content, platform="desktop", is_first_turn=False)
after_ts = mod.S.load()["runtime"]["last_user_turn_at"]
print("idle clock untouched by own injection:", before_ts == after_ts)

# --- 7. slash command surface --------------------------------------------
print("--- /proactive status ---")
print(mod._handle_command("status"))
print("--- /proactive off ---")
print(mod._handle_command("off"))
mod._tick(ctx)
print("tick while off injected delta:", len(ctx.injected) - before - 0, "(no new line expected above)")
print("SANDBOX:", SANDBOX)
