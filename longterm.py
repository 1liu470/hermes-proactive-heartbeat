"""长期层：把"接下来几小时做什么"接回"这半年往哪走"。

为什么需要它：事件层（微信抽的事项、临期任务）是短期的——
做完就消失，下周账本又是空的。要看出"哪条路在动、哪条路停了"，
得先有一副长期骨架，再把零散事项挂上去。

这一层只做两件事：

1. **归因**：把事件层的事项挂到目标上（关键词重叠，宁可漏不误挂）；
2. **周体检**：每周问一次"哪个域这周动了、哪个域很久没动"，
   产出一句人话，交给同一套裁决决定说不说。

目标从哪来：本层不再自带任何外部数据源。`state["goals"]` 为空时，
归因与周体检自动空转，不报错、不打扰。要启用，把节点写进 goals 即可。

不做的事：不改目标的 `done`（那是事实判断，只能由人填），
不替用户设定新目标，不把"没动"解释成"你不够努力"——只陈述数字。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from . import engine as E

#: 一个域超过这么多天没有任何推进信号，周体检里点出来
STALL_DAYS = 14.0


def sync_goals(state: Dict[str, Any], goals: List[Dict[str, Any]], now: float) -> int:
    """把目标写进 state，保留已有的 last_touch 等本地字段。"""
    if not goals:
        return 0
    old = {g.get("id"): g for g in (state.get("goals") or [])}
    merged = []
    for g in goals:
        prev = old.get(g["id"]) or {}
        merged.append({**g, "last_touch": prev.get("last_touch"),
                       "touch_count": int(prev.get("touch_count") or 0)})
    state["goals"] = merged
    state.setdefault("longterm", {})["synced_at"] = now
    return len(merged)


def _terms(text: str) -> set:
    return E.terms(text)


def attribute(item: E.Item, goals: List[Dict[str, Any]]) -> Optional[str]:
    """把一个事项挂到最相关的目标上。

    判据是词面重叠（CJK 双字 + 英文词），阈值 2 个共同词，避免"交学生证"
    被硬挂到"证件零遗漏"以外的任何地方。挂不上就返回 None——不猜。
    """
    text = " ".join([item.title, item.note or "", " ".join(item.tags or [])])
    it_terms = _terms(text)
    if not it_terms:
        return None
    best, best_hits = None, 0
    for g in goals:
        if g.get("level") == "north":
            continue
        hits = len(it_terms & _terms(g.get("title") or ""))
        if hits > best_hits:
            best, best_hits = g, hits
    return best["id"] if best and best_hits >= 2 else None


def _goal_catalog(goals: List[Dict[str, Any]], limit: int = 24) -> str:
    """给模型看的候选目标清单，带序号便于它只回一个数字。"""
    rows = []
    for g in goals:
        if g.get("level") not in ("domain", "goal"):
            continue
        rows.append("%d. %s" % (len(rows) + 1, g.get("title") or ""))
        if len(rows) >= limit:
            break
    return "\n".join(rows)


def attribute_llm(ctx, state: Dict[str, Any], item: E.Item,
                  goals: List[Dict[str, Any]], now: float,
                  *, timeout: float = 20.0) -> Optional[str]:
    """语义归因（第二层）：让本地模型判断这件事属于哪个长期目标。

    为什么要这一层：词面重叠在这套数据上是 0 命中，而长期层的全部意义
    就是把"交学生证"接回"报名、材料、证件零遗漏"。挂不上，长期层就是死的。

    结果按事项标题缓存，同一件事只问一次。模型说"挂不上"也缓存（存空串），
    免得一件无关的事被反复追问。
    """
    cache = state.setdefault("goal_attr", {})
    # 键里必须带目标集合的指纹：目标改了、增加了，旧归因就作废。
    # 只按标题缓存过一次，结果在目标变动后仍返回旧答案——那是个真 bug。
    sig = "|".join("%s:%s" % (g.get("id") or "", g.get("title") or "")
                   for g in goals if g.get("level") in ("domain", "goal"))
    key = "%s#%s" % ((item.title or "")[:80], hashlib.md5(sig.encode("utf-8")).hexdigest()[:8])
    if key in cache:
        return cache[key] or None

    catalog = _goal_catalog(goals)
    if not catalog:
        return None
    use = [g for g in goals if g.get("level") in ("domain", "goal")]
    prompt = (
        "下面是这个人的长期目标清单：\n" + catalog +
        "\n\n现在有一件具体的事：\n- %s\n- 备注：%s\n- 标签：%s\n" % (
            item.title or "", (item.note or "")[:400], "、".join(item.tags or [])[:120]) +
        "\n这件事在推动哪个长期目标？只回一个序号。\n"
        "判断标准：这件事做完或做砸，会不会影响哪个目标的进度、成绩或材料？"
        "会 → 就是它。\n"
        "只有当它跟所有目标都毫无因果关系时（纯娱乐、与学业和课题无关的社交），才回 0。\n"
        "注意一：把日用小事务归到它所服务的那个更大的目标上是正确的——"
        "交证件属于「证件零遗漏」，上传报告属于对应的课题，这些都要挂上，不要因为"
        "「只是小事」就回 0。\n"
        "注意二：序号必须是真的对上号，不能因为「看起来像个正经事」就随便挑一个。"
        "拿不准就回 0——挂错比挂不上更有害：挂错会让一件无关的杂事一直算作某个目标的进展。\n"
        "只输出数字。"
    )
    # ctx.llm.complete 不接受 seed（其 _gate 只放行固定几个 kwargs），
    # 实测同一问题会给出不同答案。于是每件事问 3 次取多数，靠投票稳住。
    picks: List[int] = []
    for _ in range(3):
        try:
            res = ctx.llm.complete(
                [{"role": "system", "content": "你只负责判断一件事属于哪个长期目标，只输出一个数字。"},
                 {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=8, timeout=timeout,
                purpose="goal-attribute",
            )
        except Exception:
            continue
        text = (getattr(res, "text", None) or getattr(res, "content", None) or "")
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        if not digits:
            continue
        try:
            picks.append(int(digits[:3]))
        except ValueError:
            continue
    gid = ""
    if picks:
        # 多数票；同票时取较大的序号以外，一律以"至少 2 票"为门槛，否则当没把握
        counts: Dict[int, int] = {}
        for k in picks:
            counts[k] = counts.get(k, 0) + 1
        top, top_n = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
        if top_n >= 2 and 1 <= top <= len(use):
            gid = str(use[top - 1].get("id") or "")
    cache[key] = gid
    if len(cache) > 300:
        for old_key in list(cache)[:80]:
            cache.pop(old_key, None)
    return gid or None


def attribute_best(ctx, state: Dict[str, Any], item: E.Item,
                   goals: List[Dict[str, Any]], now: float) -> Optional[str]:
    """两层归因：先词面（零成本），挂不上再问模型（有缓存）。"""
    gid = attribute(item, goals)
    if gid:
        return gid
    if not goals:
        return None
    return attribute_llm(ctx, state, item, goals, now)


def touch(state: Dict[str, Any], goal_id: Optional[str], now: float) -> None:
    """记录一次推进信号（事项完成、提醒被响应都算）。"""
    if not goal_id:
        return
    for g in state.get("goals") or []:
        if g.get("id") == goal_id:
            g["last_touch"] = now
            g["touch_count"] = int(g.get("touch_count") or 0) + 1
            return


def weekly_digest(state: Dict[str, Any], now: float) -> Dict[str, Any]:
    """按域统计最近 7 天的推进信号。纯计算，不调模型。"""
    goals = [g for g in (state.get("goals") or []) if g.get("level") == "goal"]
    domains = [g for g in (state.get("goals") or []) if g.get("level") == "domain"]
    domain_name = {d["id"]: d["title"] for d in domains}

    week_ago = now - 7 * 86400
    rows = []
    for g in goals:
        touched = g.get("last_touch")
        rows.append({
            "id": g["id"],
            "domain": g.get("domain") or "",
            "title": g.get("title") or "",
            "due": g.get("due"),
            "days_since_touch": round((now - float(touched)) / 86400, 1) if touched else None,
            "touched_this_week": bool(touched and float(touched) >= week_ago),
            "touch_count": int(g.get("touch_count") or 0),
        })

    by_domain: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = r["domain"] or "未归类"
        slot = by_domain.setdefault(d, {"name": domain_name.get(d, d), "moved": 0, "total": 0,
                                        "stalled": []})
        slot["total"] += 1
        if r["touched_this_week"]:
            slot["moved"] += 1
        elif r["days_since_touch"] is None or r["days_since_touch"] > STALL_DAYS:
            slot["stalled"].append(r["title"])

    return {
        "at": now,
        "by_domain": by_domain,
        "goals": rows,
        "dead": [d for d, s in by_domain.items() if s["moved"] == 0],
        "alive": [d for d, s in by_domain.items() if s["moved"] > 0],
    }


def digest_text(digest: Dict[str, Any], limit: int = 3) -> str:
    """把体检结果压成给人看的几句话。只陈述数字，不做道德判断。"""
    by_domain = digest.get("by_domain") or {}
    if not by_domain:
        return ""
    moved = [(d, s) for d, s in by_domain.items() if s["moved"] > 0]
    dead = [(d, s) for d, s in by_domain.items() if s["moved"] == 0]
    parts = []
    if moved:
        moved.sort(key=lambda kv: -kv[1]["moved"])
        parts.append("这周动了的：" + "、".join(f"{s['name']}（{s['moved']}/{s['total']}）"
                                              for _, s in moved[:limit]))
    if dead:
        stalled = [t for _, s in dead for t in s["stalled"]][:limit]
        parts.append(f"没动的 {len(dead)} 个域" + ("：" + "、".join(stalled) if stalled else ""))
    return "；".join(parts)


def review_due(state: Dict[str, Any], now: float, *, weekday: int = 6,
               hour: int = 20, tz=None) -> bool:
    """每周一次（默认周日 20 点后）。同一天只触发一次。"""
    import datetime as _dt

    tz = tz or _dt.timezone(_dt.timedelta(hours=8))
    local = _dt.datetime.fromtimestamp(now, tz)
    if local.weekday() < weekday:
        return False
    if local.hour < hour:
        return False
    last = float((state.get("longterm") or {}).get("last_review_at") or 0)
    if last:
        last_local = _dt.datetime.fromtimestamp(last, tz)
        if last_local.isocalendar()[:2] == local.isocalendar()[:2]:
            return False
    return True
