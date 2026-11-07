"""External signal sources for proactive-heartbeat.

Currently one: **University OS** (a separate local project, Node + SQLite, serving
`http://127.0.0.1:<port>/api/state`).

Boundary rules taken from that project's own charter, and enforced here:

* **Read-only, over public HTTP.**  Never open its SQLite file, never POST, never
  write its `tree.json`.  The plugin is a reader of published facts, not a second
  writer of a single-writer store.
* **No project data in this repository.**  This module knows field *names*
  (`treeStats.overdue`, `tree.nodes[].due`, `policies[].months`), never values.
  Anything it learns lives in the plugin's own runtime state.
* **Degrade silently.**  Service down, timeout, malformed body -> that source
  contributes nothing this tick.  A reminder loop that breaks when a webpage is
  closed is worse than no reminder loop.

Why bother: the University OS already computes what matters — 144 tree nodes with
real due dates, a policy calendar (分流/推免/大创/期末 windows), and an `overdue`
bucket.  Re-typing those into the plugin would guarantee drift.  Reading them
means one source of truth and zero maintenance.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import engine as E

_CACHE_LOCK = threading.RLock()
_CACHE: Dict[str, Tuple[float, Optional[dict]]] = {}

#: level -> subjective importance.  A blown milestone matters more than a blown task.
_LEVEL_WEIGHT = {"north": 0.9, "domain": 0.75, "goal": 0.7, "milestone": 0.8, "task": 0.55}

#: keyDates kinds that deserve an interruption when they get close.
_EXAM_KINDS = {"exam", "quiz", "milestone"}


def fetch_state(base_url: str, *, timeout: float = 3.0, cache_seconds: float = 300.0) -> Optional[dict]:
    """GET /api/state with a short timeout and a small cache. Never raises."""
    url = base_url.rstrip("/") + "/api/state"
    now = time.time()
    with _CACHE_LOCK:
        hit = _CACHE.get(url)
        if hit and now - hit[0] < cache_seconds:
            return hit[1]
    data: Optional[dict] = None
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                  "User-Agent": "proactive-heartbeat/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        parsed = json.loads(raw.decode("utf-8", "replace"))
        if isinstance(parsed, dict):
            data = parsed
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        data = None
    with _CACHE_LOCK:
        _CACHE[url] = (now, data)
    return data


def _end_of_day(day: datetime) -> float:
    return day.replace(hour=23, minute=59, second=0, microsecond=0).timestamp()


def _parse_date(text: Any, tz: timezone) -> Optional[float]:
    if not isinstance(text, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M"):
        try:
            return _end_of_day(datetime.strptime(text, fmt).replace(tzinfo=tz))
        except ValueError:
            continue
    return None


def _month_end(year: int, month: int, tz: timezone) -> float:
    if month == 12:
        nxt = datetime(year + 1, 1, 1, tzinfo=tz)
    else:
        nxt = datetime(year, month + 1, 1, tzinfo=tz)
    return (nxt - timedelta(days=1)).replace(hour=23, minute=59).timestamp()


def items_from_state(state: dict, now: float, tz: timezone,
                     nudged: Optional[Dict[str, Dict[str, Any]]] = None,
                     *, max_items: int = 14) -> List[E.Item]:
    """Turn a University OS snapshot into candidate Items (pure; no I/O).

    Four signals, each with a reason to exist:

    1. `treeStats.overdue`  — work whose date already passed and nobody closed it.
       Highest value: it is a real, dated, self-authored commitment.
    2. `treeStats.today3`   — the three things that system ranked for today, still
       dated today or later.  Its ranking is already better than anything
       recomputed here, so the plugin defers to it.
    3. `keyDates`           — exams and calendar milestones inside 14 days.
    4. `policies`           — 分流/推免/大创/期末 windows in this or next month.
       These carry no exact date, so they are given a month-end due date and the
       window text as evidence; a wrong-day warning is worse than a late one.
    """
    nudged = nudged or {}
    out: List[E.Item] = []
    seen: set = set()

    def add(item_id: str, title: str, due: Optional[float], weight: float, note: str) -> None:
        key = "uo:" + item_id
        if key in seen or not title:
            return
        seen.add(key)
        rec = nudged.get(key) or {}
        out.append(E.Item(
            id=key, title=title.strip()[:120], kind="deadline", due=due,
            weight=max(0.0, min(1.0, weight)),
            created_at=float(rec.get("created_at") or now),
            nudges=int(rec.get("n") or 0),
            last_nudged_at=float(rec["at"]) if rec.get("at") else None,
            note=note[:120],
        ))

    stats = state.get("treeStats") or {}
    # Group overdue siblings by their parent id (T2.1.1.1 / .2 / .3 -> T2.1.1).
    # Five fragments of one past event must not become five interruptions: keep the
    # earliest-due member as the group's representative and say how many it stands for.
    groups: Dict[str, List[dict]] = {}
    for node in (stats.get("overdue") or []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "")
        parent = node_id.rsplit(".", 1)[0] if "." in node_id else node_id
        groups.setdefault(parent, []).append(node)
    for parent, members in groups.items():
        members.sort(key=lambda n: str(n.get("due") or "9999"))
        head = members[0]
        due = _parse_date(head.get("due"), tz)
        extra = f"，同组 {len(members)} 条" if len(members) > 1 else ""
        add(str(head.get("id") or ""), str(head.get("title") or ""), due,
            _LEVEL_WEIGHT.get(str(head.get("level") or ""), 0.55),
            f"大学项目·逾期（{head.get('level')}/{head.get('domain')}{extra}）")

    today_key = datetime.fromtimestamp(now, tz).strftime("%Y-%m-%d")
    for node in (stats.get("today3") or [])[:3]:
        if not isinstance(node, dict):
            continue
        due = _parse_date(node.get("due"), tz)
        if due is not None and due < now - 86400:
            continue  # already counted as overdue; do not double-count
        add(str(node.get("id") or ""), str(node.get("title") or ""), due, 0.5,
            "大学项目·今日三件")

    for kd in (state.get("keyDates") or []):
        if not isinstance(kd, dict) or str(kd.get("kind") or "") not in _EXAM_KINDS:
            continue
        due = _parse_date(kd.get("date"), tz)
        if due is None or not (now - 86400 <= due <= now + 14 * 86400):
            continue
        add("kd:" + str(kd.get("id")), str(kd.get("label") or ""), due, 0.6,
            f"大学项目·校历（{kd.get('kind')}）")

    local = datetime.fromtimestamp(now, tz)
    this_month, next_month = local.month, (local.month % 12) + 1
    this_year = local.year
    next_year = this_year + (1 if this_month == 12 else 0)
    for pol in (state.get("policies") or []):
        if not isinstance(pol, dict):
            continue
        months = pol.get("months") or []
        if not isinstance(months, list):
            continue
        if not (pol.get("urgent") or pol.get("critical")):
            continue
        if this_month in months:
            due = _month_end(this_year, this_month, tz)
        elif next_month in months:
            due = _month_end(next_year, next_month, tz)
        else:
            continue
        add("pol:" + str(pol.get("id")), str(pol.get("title") or ""), due, 0.65,
            f"大学项目·政策窗口（{pol.get('window') or ''}）")

    out.sort(key=lambda i: (i.due or 9e18))
    return out[:max_items]
