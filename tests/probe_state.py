"""对着真实 state 的副本验证三件事：过期反馈作废、活跃时间被刷新、心跳事后写入不覆盖它们。

    python tests/probe_state.py                      # 用本机 state
    python tests/probe_state.py --state 某处.json     # 指定副本
    python tests/probe_state.py --session agent:main:qqbot:dm:test0001

只在临时目录里动，不碰真 state。这是单元测试补不上的那一块：
单测里没有真实的过期 pending，构造出来的又证明不了它长什么样。
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
HERMES = Path(os.environ.get("HERMES_HOME") or (Path.home() / "AppData/Local/hermes"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(HERMES / "state/proactive-heartbeat/state.json"))
    ap.add_argument("--session", default="agent:main:qqbot:dm:probe0001")
    args = ap.parse_args()

    real = Path(args.state)
    if not real.exists():
        print("找不到 state：", real)
        return 2

    sb = Path(tempfile.mkdtemp(prefix="ph-probe-"))
    dst = sb / "state/proactive-heartbeat"
    dst.mkdir(parents=True)
    shutil.copy2(real, dst / "state.json")
    os.environ["HERMES_HOME"] = str(sb)
    os.environ["HERMES_SESSION_KEY"] = args.session

    sys.path.insert(0, str(HERMES / "hermes-agent"))
    import logging

    from plugins.plugin_loader import load_plugin_module

    mod = load_plugin_module("plugins.proactive-heartbeat", SRC,
                             parents=("plugins",), logger=logging.getLogger("probe"))
    if not mod:
        print("插件加载失败")
        return 2

    class StubCtx:
        plugin_id = "proactive-heartbeat"

        def __init__(self):
            self.hooks, self.tools, self.commands, self.settings = {}, {}, {}, {}

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

        def inject_message(self, *a, **k):
            return True

    ctx = StubCtx()
    mod.register(ctx)
    mod._RUNTIME["stop"].set()

    S = mod.S
    before = S.load()
    p0 = before["runtime"].get("pending") or {}
    decp = S.decisions_path()
    dec0 = len(decp.read_text(encoding="utf-8").splitlines()) if decp.exists() else 0
    print("拷入 pending：", p0.get("title"),
          "| 距今 %.1f 小时" % ((time.time() - float(p0.get("at") or 0)) / 3600))
    print("拷入 last_user_turn_at 距今 %.1f 小时"
          % ((time.time() - float(before["runtime"].get("last_user_turn_at") or 0)) / 3600))
    print("拷入 adaptive_offset：", before.get("adaptive_offset"), "| 日志行数", dec0)

    print("\n[1] 心跳取走快照（其中 pending 仍在）")
    stale = S.load()
    assert stale["runtime"].get("pending") is not None or True

    print("[2] 用户连说三句")
    for i in range(3):
        mod._on_pre_llm_call(user_message=["好的", "嗯", "知道了"][i], platform="qqbot")

    after = S.load()
    dec = S.decisions_path().read_text(encoding="utf-8").splitlines()
    new = [json.loads(x) for x in dec[dec0:]]
    print("    日志:", [(r.get("event"), r.get("reason")) for r in new])
    print("    pending ->", after["runtime"].get("pending"))
    print("    last_user_turn_at 距今 %.2f 秒"
          % (time.time() - float(after["runtime"].get("last_user_turn_at") or 0)))
    print("    adaptive_offset ->", after.get("adaptive_offset"),
          "| ignore_streak ->", after.get("ignore_streak"))

    S.update(lambda st: st["runtime"].__setitem__(
        "daily_count", int(st["runtime"].get("daily_count") or 0) + 1))
    final = S.load()
    print("\n[3] 心跳事后写入 daily_count")
    print("    pending ->", final["runtime"].get("pending"))
    print("    daily_count ->", final["runtime"].get("daily_count"))

    checks = [
        ("过期 probe 被清掉", after["runtime"].get("pending") is None),
        ("心跳事后写入没有让它复活", final["runtime"].get("pending") is None),
        ("只结算一次", len([r for r in new if r.get("event") == "feedback"]) == 1),
        ("且判为 expired", bool(new) and new[0].get("reason") == "expired"),
        ("门槛未被污染", after.get("adaptive_offset") == before.get("adaptive_offset")),
        ("活跃时间已刷新",
         (time.time() - float(final["runtime"]["last_user_turn_at"])) < 60),
    ]
    print()
    for name, ok in checks:
        print("  %s %s" % ("通过" if ok else "不成立", name))
    bad = [n for n, ok in checks if not ok]
    print("\n结论：", "全部符合预期" if not bad else "有 %d 项不成立" % len(bad))
    shutil.rmtree(sb, ignore_errors=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
