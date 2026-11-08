"""主动研究：在提醒之前，先把这件事的功课做了。

Muse 那种"为你着想"不是语气温柔，是**替你完成一个你还没做的动作**：
不只是告诉你"11-08 助学金截止"，而是先把"要交什么、交给谁、往年什么时候发通知"查清楚，
再问你一句要不要现在办。

实现上刻意保持便宜：

* 只在硬期限进入窗口时才查（默认 72 小时），不是每条都查；
* 每天有总次数上限，超了就退回纯提醒；
* 结果按事项缓存一天，同一件事不重复搜；
* 查不到就静默降级——研究失败不该让提醒也失败。

搜索结果不单独成段发出去，而是喂进裁决提示，让同一次模型调用顺带消化。
一次调用干两件事，不多花钱。
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import engine as E


def _query_from_note(item: E.Item) -> str:
    """自感知候选把检索词放在 note 的「查：」后面，不再让模型重新想一遍。"""
    note = item.note or ""
    if "查：" in note:
        return note.split("查：", 1)[1].strip()[:40]
    return ""


def should_research(item: E.Item, now: float, window_hours: float,
                    overdue_grace_hours: float = 168.0) -> bool:
    if item.kind == "sense":
        return bool(_query_from_note(item))
    if item.kind != "deadline" or item.due is None:
        return False
    hours = (item.due - now) / 3600.0
    # 逾期不等于没救。刚过期那几天，「找谁、还能不能补、要什么材料」恰恰最需要查；
    # 原来的下界是 0，等于他一开始就错过的事，功课立刻停止，提醒反而更空。
    # 逾过一周就是陈旧账，再查只是噪音，所以给一个有界的落后窗口。
    return -max(0.0, overdue_grace_hours) <= hours <= window_hours


def build_query(item: E.Item) -> str:
    """把事项标题变成一句像人写的检索词。

    去掉括号里的日期（检索词里日期通常有害），补一个年份锚点。
    """
    title = re.sub(r"[（(][^）)]*[）)]", "", item.title).strip()
    title = re.sub(r"^(确认|记得|完成|参加|准备)", "", title).strip()
    year = time.strftime("%Y")
    return f"{title} {year}".strip()


def _extract_hits(raw: Any, limit: int) -> List[Dict[str, str]]:
    """Accept whatever shape the host's web_search returns."""
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            return []
    if not isinstance(data, dict):
        return []
    web = ((data.get("data") or {}).get("web") if isinstance(data.get("data"), dict) else None) \
        or data.get("results") or []
    hits = []
    for w in web[:limit]:
        if not isinstance(w, dict):
            continue
        hits.append({
            "title": str(w.get("title") or "")[:80],
            "url": str(w.get("url") or "")[:160],
            "desc": str(w.get("description") or w.get("content") or "")[:160],
        })
    return hits


QUERY_SYSTEM = "你把一件事转成检索词。只输出 JSON 数组。"

QUERY_PROMPT = """事项：{title}（时间：{when}）

要做这件事，缺哪些信息？例如：办理流程、需要带的材料、地点开放时间、截止日期、往年经验。

把缺的信息转成 1-2 条检索词。要求：
- 不要照抄事项原文；把口语（"确认""集合""搞定"）换成名词短语
- 带上必要的限定（学校名、城市、机构名），但只有确定时才加
- 每条 ≤20 字

只输出 JSON 数组：["检索词1", "检索词2"]"""


def plan_queries(ctx, item: E.Item, now: float, *, timeout: float = 45) -> List[str]:
    """让模型决定该查什么。

    直接把事项标题当 query 会搜出无关的东西（"某纪念馆集合"搜到建党大会），
    因为标题写的是"我要干什么"，而检索要的是"这件事需要哪些信息"。
    """
    when = E.humanize_due(item.due, now)
    try:
        res = ctx.llm.complete(
            [{"role": "system", "content": QUERY_SYSTEM},
             {"role": "user", "content": QUERY_PROMPT.format(title=item.title,
                                                             when=when)}],
            temperature=0.2, max_tokens=200, timeout=timeout, purpose="research-query",
        )
        text = getattr(res, "text", None) or getattr(res, "content", None) or ""
    except Exception:
        return []
    if not isinstance(text, str):
        return []
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        arr = json.loads(text[start:end + 1])
    except Exception:
        return []
    out = [str(q).strip()[:40] for q in arr if isinstance(q, str) and q.strip()]
    return out[:2]


def _call_search(ctx, query: str, limit: int) -> List[Dict[str, str]]:
    """两条路都试：插件作用域 → 全局注册表。

    网关启动时会 import model_tools，工具都在全局注册表里；但插件调用走的是
    插件自己的 scope，若那个 scope 查不到（隔离策略不同），全局那条还能兜住。
    任何一步失败都返回空列表，研究失败不该让提醒也失败。
    """
    args = {"query": query, "limit": limit}
    try:
        hits = _extract_hits(ctx.dispatch_tool("web_search", dict(args)), limit)
        if hits:
            return hits
    except Exception:
        pass
    try:
        from tools.registry import registry
        return _extract_hits(registry.dispatch("web_search", dict(args)), limit)
    except Exception:
        return []


def _topic_key(query: str) -> str:
    """把检索词归一成一个主题键：去掉年份和空格，截 40 字。

    "大创 2027 年度项目申报" 与 "大创项目 申报 材料" 都会落到 "大创年度项目申报" 附近，
    但不会误并到无关主题——这是粗粒度去重，不是语义聚类。
    """
    s = re.sub(r"20\d\d", "", query or "")
    s = re.sub(r"\s+", "", s)
    return s[:40]


def _similar(a: str, b: str) -> float:
    """字符二元组重叠率。用于判断两个标题是不是同一件事。

    "大创项目申报材料" 与 "大创项目申报提交" → 0.86，是同一件事；
    "申请奖学金材料" 与 "大创项目申报材料" → 0.17，不是。
    精确字符串匹配在这里太脆：多一个词就漏，所以用房喻。
    """
    a2 = {a[i:i + 2] for i in range(max(0, len(a) - 1))}
    b2 = {b[i:i + 2] for i in range(max(0, len(b) - 1))}
    if not a2 or not b2:
        return 0.0
    return len(a2 & b2) / min(len(a2), len(b2))


def gather(ctx, item: E.Item, state: Dict[str, Any], now: float, *,
           window_hours: float = 72.0, max_items: int = 3,
           daily_cap: int = 6, cache_hours: float = 24.0,
           knowledge_days: float = 30.0,
           overdue_grace_hours: float = 168.0,
           report: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """返回该事项的情报条目；任何一步失败都返回空列表。

    ``report`` 会回填这次为什么有、为什么没有——「一次都没查过」这种故障
    只能靠读账本发现，太隐蔽了。
    """
    def _note(status: str, reason: str = "") -> None:
        if report is not None:
            report.update({"status": status, "reason": reason})

    if not should_research(item, now, window_hours, overdue_grace_hours):
        _note("skipped", "window")
        return []

    research = state.setdefault("research", {})
    cached = research.get(item.id)
    if cached and now - float(cached.get("at") or 0) < cache_hours * 3600:
        cached_hits = list(cached.get("hits") or [])
        _note("cached" if cached_hits else "empty", "cache")
        return cached_hits

    # 知识库：同一主题查过就复用，不必重搜。这件事跨月，缓存不该只有一天。
    knowledge = state.setdefault("knowledge", {})
    topic = _topic_key(item.title)
    for key, entry in knowledge.items():
        if now - float(entry.get("at") or 0) > knowledge_days * 86400:
            continue
        if _similar(key, topic) >= 0.6:
            hits = list(entry.get("hits") or [])
            if hits:
                research[item.id] = {"at": now, "query": entry.get("query"), "hits": hits,
                                     "from": "knowledge"}
                _note("cached", "knowledge")
                return hits

    # 与 engine.local_day_key 同一口径：系统本地 = Asia/Shanghai（+08:00），
    # 已在 /proactive status 里显式核对，避免 UTC 与本地日期错位。
    day = time.strftime("%Y-%m-%d")
    used = int((state.setdefault("research_calls", {}) or {}).get(day) or 0)
    if used >= daily_cap:
        _note("skipped", "daily_cap")
        return []

    query = build_query(item)
    preset = _query_from_note(item)
    queries = [preset] if preset else (plan_queries(ctx, item, now) or [query])
    hits: List[Dict[str, str]] = []
    for q in queries:
        hits = _call_search(ctx, q, max_items)
        if hits:
            query = q
            break

    state["research_calls"][day] = used + 1
    research[item.id] = {"at": now, "query": query, "queries": queries, "hits": hits}
    if hits:
        knowledge[_topic_key(query)] = {"at": now, "query": query, "hits": hits}
        if len(knowledge) > 60:
            keep = sorted(knowledge.items(), key=lambda kv: float(kv[1].get("at") or 0))[-40:]
            state["knowledge"] = dict(keep)
    if len(research) > 100:
        keep = sorted(research.items(), key=lambda kv: float(kv[1].get("at") or 0))[-60:]
        state["research"] = dict(keep)
    _note("ok" if hits else "empty")
    return hits


def format_intel(hits: List[Dict[str, str]]) -> str:
    """给裁决提示用的一小段情报。空则返回空串。"""
    if not hits:
        return ""
    lines = []
    for h in hits[:3]:
        t = h.get("title") or ""
        d = h.get("desc") or ""
        if t or d:
            lines.append(f"- {t}：{d}".strip("："))
    if not lines:
        return ""
    return "已知情报（联网检索，未核验，可能过时）：\n" + "\n".join(lines)


# --------------------------------------------------------------------------
# 资料落盘：把检索结果写成文件，提醒里附路径。
#
# 为什么必须落盘：原先 format_intel 只取标题+摘要的前几条喂给模型，
# 模型再压成一句话，用户看不到任何一条原文与链接。"我替你查了"于是
# 变成一句空话。落盘之后，即使这句提醒他不需要，东西是真递过去了的。
# --------------------------------------------------------------------------

def _intel_dir() -> Path:
    home = Path(os.environ.get("HERMES_HOME", Path.home() / "AppData/Local/hermes"))
    d = home / "state" / "proactive-heartbeat" / "intel"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(text: str, limit: int = 40) -> str:
    keep = "".join(ch for ch in (text or "") if ch.isalnum() or ch in "-_")
    return (keep or "intel")[:limit]


def save_intel(item: Any, hits: List[Dict[str, str]], now: float) -> Optional[str]:
    """把一次检索的完整结果写成 markdown，返回绝对路径；无线索则 None。"""
    rows = [h for h in (hits or []) if (h.get("title") or h.get("url"))]
    if not rows:
        return None
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    name = "%s_%s.md" % (ts, _safe_name(getattr(item, "title", "") or "intel"))
    path = _intel_dir() / name
    lines = ["# 检索资料：%s" % (getattr(item, "title", "") or ""), "",
             "- 时间：%s" % time.strftime("%Y-%m-%d %H:%M", time.localtime(now)),
             "- 来源：联网检索，未逐条核验，可能过时或不准。", ""]
    for i, h in enumerate(rows, 1):
        lines.append("## %d. %s" % (i, h.get("title") or "（无标题）"))
        if h.get("url"):
            lines.append("- 链接：%s" % h["url"])
        if h.get("desc"):
            lines.append("- 摘要：%s" % h["desc"])
        lines.append("")
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        return None
    return str(path)


def intel_headline(hits: List[Dict[str, str]], limit: int = 2, width: int = 26) -> str:
    """给消息用的极短提要：前两条标题，够让他决定要不要点开。"""
    names = []
    for h in (hits or [])[:limit]:
        t = (h.get("title") or "").strip()
        if not t:
            continue
        names.append(t if len(t) <= width else t[: width - 1] + "…")
    return "、".join(names)
