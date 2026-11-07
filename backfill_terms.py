"""从会话历史回填话题词频，让 R2「话题加热」立刻有真实数据。

为什么需要它：term_stats 是从零开始攒的，攒够 5 次要几天。
而 state.db 里已经有 14 天的对话，直接读一遍就能让 R2 当天可用。

计数口径是「出现在多少个不同的日子」——"他连续三天在提这件事"比
"他一口气说了十遍"更能说明这件事一直放在心上。同一个会话里的重复不算。

只读 state.db，不写它。用法：
  python backfill_terms.py [天数]
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ph_engine", HERE / "engine.py")
engine = importlib.util.module_from_spec(spec)
sys.modules["ph_engine"] = engine
spec.loader.exec_module(engine)

TZ = timezone(timedelta(hours=8))
STATE = Path(os.environ.get("HERMES_HOME", Path.home() / "AppData/Local/hermes")) / \
    "state" / "proactive-heartbeat" / "state.json"
DB = Path(os.environ.get("HERMES_HOME", Path.home() / "AppData/Local/hermes")) / "state.db"


def day_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, TZ).strftime("%Y-%m-%d")


#: 这些标记说明那条 "user" 消息其实是宿主注入的上下文，不是他本人说的话。
INJECTION_MARKERS = (
    "[LIVE-TIME]", "[CROSS-SESSION", "[PROACTIVE", "[IMPORTANT", "[OUT-OF-BAND",
    "[time-gap", "[CROSS-SESSION DOSSIERS]", "<available_skills>", "<memory",
    "MEDIA:", "::preview", "AppData/Local/hermes", "HERMES_",
)


def is_real_user_message(text: str) -> bool:
    """把注入块筛掉。实测 1491 条 'user' 消息里相当一部分是环境与召回注入，
    不筛的话词频榜前十五名全是 and / any / appdata / hermes / 200。"""
    if not text or len(text.strip()) < 2:
        return False
    return not any(m in text for m in INJECTION_MARKERS)


def main(days: int = 14) -> int:
    if not DB.exists():
        print("找不到 state.db")
        return 1
    now = time.time()
    since = now - days * 86400

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT s.id AS sid, s.started_at, s.last_activity_at, m.content
        FROM messages m JOIN sessions s ON s.id = m.session_id
        WHERE m.role = 'user'
          AND s.last_activity_at IS NOT NULL
          AND s.last_activity_at > ?
        ORDER BY s.started_at
    """, (since,)).fetchall()
    con.close()
    print(f"读到 {len(rows)} 条消息，覆盖最近 {days} 天")

    stats: dict = {}
    seen_in_day: set = set()          # (day, term) 只算一次
    skipped = 0
    for r in rows:
        text = r["content"] or ""
        if not is_real_user_message(text):
            skipped += 1
            continue
        ts = float(r["last_activity_at"] or r["started_at"] or 0)
        if ts <= 0:
            continue
        day = day_of(ts)
        for term in engine.topic_terms(text, limit=40):
            # 只统计中文词：英文与数字几乎都来自代码、路径与注入文本
            if not re.fullmatch(r"[\u4e00-\u9fff]{2,}", term):
                continue
            key = (day, term)
            s = stats.get(term)
            if s is None:
                stats[term] = {"n": 0, "first": ts, "last": ts, "days": [day]}
                s = stats[term]
            else:
                if ts < s["first"]:
                    s["first"] = ts
                if ts > s["last"]:
                    s["last"] = ts
                if day not in s["days"]:
                    s["days"].append(day)
            if key not in seen_in_day:
                seen_in_day.add(key)
                s["n"] = int(s["n"]) + 1

    if not STATE.exists():
        print("找不到插件 state.json")
        return 1
    state = json.loads(STATE.read_text(encoding="utf-8"))
    rt = state.setdefault("runtime", {})
    old = len(rt.get("term_stats") or {})
    rt["term_stats"] = stats
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

    hot = sorted(stats.items(), key=lambda kv: (-len(kv[1]["days"]), -kv[1]["n"]))[:15]
    print(f"写入 {len(stats)} 个词（替换旧账 {old} 个）\n")
    print("跨天最多的话题词：")
    for term, s in hot:
        print(f"  {term:<10} 出现在 {len(s['days'])} 天 | 共 {s['n']} 次 | 最后 {day_of(s['last'])}")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    raise SystemExit(main(n))
