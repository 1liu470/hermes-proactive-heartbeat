"""自感知层：没有事项的时候，推测他此刻需要什么。

和事件层的区别：事件层是"有人告诉我有件事"，自感知是**没人说，但从上下文里看出来**。
Muse 那种体贴感就来自这里——不是等你交办，而是先想到。

三条规则，都能追溯到可验证的信号，不做心理推断：

R1 临期无备：七天内有硬期限，这类事（要去 / 要办）他从来没提前准备过，
   知识库里也没有对应资料 → 他大概不知道要带什么、走什么流程。
R2 话题加热：某个词在最近几天被他反复提到（≥5 次、跨 ≥2 天），
   但账本里没有对应事项 → 他心里装着这件事，还没落成待办。
R3 目标失联：某个长期目标半个月没动，而它的域里已经有十天内的期限
   → 不是催他努力，是提醒他"这条线快断了"。

三条都不猜情绪、不猜关系、不猜健康。猜不出就不猜——默认沉默同样适用于这一层。
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from . import engine as E

#: 同一个假设，七天内不重复提
REPEAT_DAYS = 7.0
#: R2 的门槛：跨几个不同日子、总共几次、最近还提过
MIN_DAYS = 2
MIN_MENTIONS = 4
MIN_SPAN_DAYS = 2.0
#: 最后一次提及距今多久之内还算"还在惦记"。3 天太窄：
#: 实测"分流"10-27~11-01 提了 4 次，11-05 再看时 age=99 小时就被判定过气，
#: 可这正是"反复说、一直没办"的典型。放到 5 天。
RECENT_DAYS = 5.0


def _due_ts(value: Any) -> Optional[float]:
    """目标树里的 due 是 'YYYY-MM-DD' 字符串，按北京时间解析。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    import datetime as _dt
    try:
        d = _dt.datetime.strptime(str(value)[:10], "%Y-%m-%d")
        return d.replace(tzinfo=_dt.timezone(_dt.timedelta(hours=8))).timestamp()
    except Exception:
        return None


def _open(state: Dict[str, Any]) -> List[E.Item]:
    """state["items"] 落盘后是 dict 列表，这里统一转成 Item。

    之前直接调 dict.is_open() 会在第一次真实 tick 就抛 AttributeError——
    单测用 dict 构造数据才把它逼出来。
    """
    out: List[E.Item] = []
    for raw in (state.get("items") or []):
        it = raw if isinstance(raw, E.Item) else E.Item.from_dict(raw)
        if it.is_open():
            out.append(it)
    return out


def _titles(state: Dict[str, Any]) -> str:
    return " ".join(i.title for i in _open(state))


def _already_covered(term: str, state: Dict[str, Any]) -> bool:
    return term in _titles(state)


def _knowledge_has(state: Dict[str, Any], title: str) -> bool:
    from .research import _similar, _topic_key
    topic = _topic_key(title)
    for key in (state.get("knowledge") or {}):
        if _similar(key, topic) >= 0.5:
            return True
    return False


def hypotheses(state: Dict[str, Any], now: float, *,
               window_days: float = 7.0, stall_days: float = 14.0) -> List[Dict[str, Any]]:
    """产出候选假设，纯计算，不调模型。按置信度降序。"""
    out: List[Dict[str, Any]] = []
    items = _open(state)

    # R1 临期无备
    for it in items:
        if it.kind != "deadline" or it.due is None:
            continue
        hours = (it.due - now) / 3600.0
        if not (0 < hours <= window_days * 24):
            continue
        if E.item_type(it) not in ("attend", "apply"):
            continue
        if _knowledge_has(state, it.title):
            continue
        out.append({
            "key": f"prep:{it.id}",
            "rule": "临期无备",
            "title": it.title,
            "basis": f"{E.humanize_due(it.due, now)}；这类事他通常不提前备，库里也没资料",
            "query": f"{re.sub(r'[（(][^）)]*[）)]', '', it.title)[:20]} 需要带什么 流程",
            "weight": 0.62,
            "conf": 0.62,
        })

    # R2 话题加热
    stats = (state.get("runtime") or {}).get("term_stats") or {}
    ctx_terms = set((state.get("runtime") or {}).get("context_terms") or [])
    today = time.strftime("%Y-%m-%d")
    for term, s in stats.items():
        n = int(s.get("n") or 0)
        days = s.get("days") or []
        if n < MIN_MENTIONS or len(term) < 2:
            continue
        # 按"提到过几个不同日子"判，而不是消息条数——
        # 一个下午连说十遍是同一个念头，连续三天提起才是心里装着的事。
        if len(days) < MIN_DAYS:
            continue
        first, last = float(s.get("first") or 0), float(s.get("last") or 0)
        if (last - first) < MIN_SPAN_DAYS * 86400:
            continue
        # 今天还在提的，是他正在处理的事，不需要猜；正在聊的词同理。
        # 要找的是"前几天反复说、今天没再提、账本里也没有"的那一类——
        # 那才是惦记着却掉在地上的事。
        if today in days or term in ctx_terms:
            continue
        age = now - last
        if age < 86400 or age > RECENT_DAYS * 86400:
            continue
        if _already_covered(term, state):
            continue
        # 只认"具体实体"，不认抽象名词。
        # 这条规则是被真实数据打出来的：放宽到所有名词后，跨天最多的假设变成了
        # 形式 / 数据 / 代价 / 元素 / 公式——那是讨论时的用词，不是惦记的事。
        # 判据是：在校园实体词表里，或本身带数字/英文（四六级、iGEM 这类专名）。
        if term not in E.USER_WORDS and not re.search(r"[0-9A-Za-z]", term):
            continue
        # 提得越多越可信，但上限压在硬期限之下
        conf = min(0.66, 0.52 + 0.02 * min(n - MIN_MENTIONS, 7))
        out.append({
            "key": f"topic:{term}",
            "rule": "话题加热",
            "title": term,
            "basis": f"{len(days)} 天里提了 {n} 次，最后一次是 {int(age/3600)} 小时前，账本里没有对应的事",
            # 不联网：一个孤立的词构造不出有效检索。
            # 实测「分流 是什么 怎么做」搜回来的是 VPN 分流与电路分流定律，
            # 塞进裁决提示只会误导。R2 只是"提醒你想过这件事"，不需要资料。
            "query": "",
            "weight": conf,
            "conf": conf,
        })

    # R3 目标失联（不搜，直接说）
    synced = float((state.get("longterm") or {}).get("synced_at") or 0)
    for g in (state.get("goals") or []):
        if g.get("level") != "goal":
            continue
        due_ts = _due_ts(g.get("due"))
        if due_ts is None:
            continue
        days_left = (due_ts - now) / 86400.0
        if not (0 < days_left <= 60):
            continue
        touch = g.get("last_touch")
        if not touch:
            # 从没有推进记录：不知道他做过没有，只知道这本账没记过。
            # 记账不足 14 天就不下结论——把"我不知道"说成"你半个月没动"是最坏的误报。
            if not synced or (now - synced) < stall_days * 86400:
                continue
            basis = f"这条线 {int(days_left)} 天后到期，这本账还没有任何推进记录"
        else:
            idle_days = (now - float(touch)) / 86400.0
            if idle_days < stall_days:
                continue
            basis = f"这条线 {int(days_left)} 天后到期，而它 {int(idle_days)} 天没有推进"
        out.append({
            "key": f"stall:{g.get('id')}",
            "rule": "目标失联",
            "title": g.get("title") or "",
            "basis": basis,
            "query": "",
            "weight": 0.58,
            "conf": 0.58,
        })

    out.sort(key=lambda h: -h["conf"])
    return out


def pick(state: Dict[str, Any], now: float, *, daily_cap: int = 2,
         repeat_days: float = REPEAT_DAYS) -> Optional[Dict[str, Any]]:
    """选出这一轮唯一要处理的那个假设。任何条件不满足就返回 None。"""
    sense = state.setdefault("sense", {})
    done = sense.get("done") or {}
    day = time.strftime("%Y-%m-%d")
    used = int((sense.get("calls") or {}).get(day) or 0)
    if used >= daily_cap:
        return None

    for h in hypotheses(state, now):
        last = float(done.get(h["key"]) or 0)
        if last and (now - last) < repeat_days * 86400:
            continue
        # 上一个假设刚被模型否决（"此刻不该说"），半小时内不再拿同一个去问一遍
        attempt = sense.get("last_attempt") or {}
        if attempt.get("key") == h["key"] and (now - float(attempt.get("at") or 0)) < 1800:
            continue
        sense["last_attempt"] = {"key": h["key"], "at": now}
        return h
    return None


def mark(state: Dict[str, Any], key: str, now: float) -> None:
    sense = state.setdefault("sense", {})
    done = sense.setdefault("done", {})
    done[key] = now
    day = time.strftime("%Y-%m-%d")
    calls = sense.setdefault("calls", {})
    calls[day] = int(calls.get(day) or 0) + 1
    if len(done) > 200:
        keep = sorted(done.items(), key=lambda kv: float(kv[1]))[-120:]
        sense["done"] = dict(keep)


def to_item(h: Dict[str, Any], now: float) -> E.Item:
    note = h["basis"]
    if h.get("query"):
        note = f"{note}｜查：{h['query']}"
    return E.Item(id=f"sense:{h['key']}", title=h["title"], kind="sense",
                  weight=float(h["weight"]), created_at=now, note=note)
