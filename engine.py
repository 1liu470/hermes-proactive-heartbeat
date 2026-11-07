"""Pure decision engine for the proactive-heartbeat plugin.

No Hermes imports here on purpose: every rule below is unit-testable with plain
Python.  The plugin layer (``__init__.py``) only feeds it observations and acts
on its verdicts.

Lineage of the rules (all verified against source, not recalled):

* Generative Agents (joonspk-research/generative_agents) — salience is a weighted
  sum of recency/relevance/importance, with geometric decay
  (``recency_decay = 0.99``, retrieval weights all 1.0) and an attention budget
  (``att_bandwidth = 3``, ``retention = 5``).  Reflection only runs once
  accumulated importance crosses ``importance_trigger_max = 150``.
* ProactiveAgent (thunlp) — a fixed-cadence heartbeat over a *coalesced* event
  stream; the model is only asked when a bucket carries signal, its answer is
  ``Proactive_Task: null`` when no help is needed, and the previous turn's
  user feedback conditions the next proposal ("try not to disturb the user when
  they ignore your assistance").
* Muse (Meta) — the product rule that sets the bar: a proactive message "has to
  be genuinely helpful and worth the interruption", with a user-facing dial and
  an audible decision trail.

What this engine adds beyond the three: an explicit *interruption cost* term and
a deterministic pre-gate, so the model is never asked a question whose answer is
already "don't".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# Dials
# --------------------------------------------------------------------------

#: dial -> (threshold, daily budget, min idle minutes, cooldown minutes)
DIALS: Dict[str, Tuple[float, int, int, int]] = {
    "off":    (9.99, 0, 10 ** 9, 10 ** 9),
    "quiet":  (0.76, 2, 60, 180),
    "normal": (0.62, 8, 15, 40),
    "eager":  (0.50, 14, 8, 20),
}

TERM_RE = re.compile(r"[a-z0-9_]{3,}", re.I)
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

#: 虚词与高频无信息词。双字滑窗最大的问题是把"这个怎么搞"切成
#: 这个/个怎/怎么/么搞——"个怎"这种碎片在任何词典里都不存在，
#: 却会被当成他"反复提到的话题"。分词 + 停用词才治得了。
STOPWORDS = frozenset("""
这个 那个 哪个 什么 怎么 怎样 这么 那么 多少 为什么 如何 是否
我们 你们 他们 她们 它们 咱们 自己 别人 大家 各位
可以 需要 应该 可能 已经 还是 就是 不是 没有 因为 所以 但是 而且 然后
如果 虽然 不过 或者 以及 关于 对于 由于 因此 于是 只是 还有 一样
现在 今天 明天 昨天 后天 时候 时间 东西 事情 地方 方面 问题 情况 内容
一个 一些 一直 一下 一起 一遍 这种 那种 这样 那样 这里 那里 这些 那些
知道 觉得 认为 看看 说说 想想 试试 帮忙 麻烦 谢谢 好的 收到 明白
全部 一般 一致 大概 基本 主要 重要 直接 具体 相关 不同 一样 各种
""".split())

#: 纯确认词：这类回复本身就是对提醒的回应，不必再看内容相关度。
ACK_WORDS = ("收到", "好的", "嗯", "知道了", "明白", "行了", "ok", "okay",
             "了解", "记下", "已办", "搞定", "完成了", "谢谢")

#: 校园场景的常见简称，jieba 词表里没有，会被切成"大"+"创"。
#: 可以在插件配置 user_words 里追加自己的（比赛名、课程名、项目简称）。
USER_WORDS = (
    "大创", "推免", "保研", "绩点", "加权", "选课", "组会", "分流",
    "四六级", "考研", "实习", "简历", "助研", "学工", "班团", "团支书",
    "助学金", "奖学金", "实验报告", "预习报告", "毕业论文", "开题",
)

_JIEBA: Any = None


def _jieba():
    """懒加载分词器。没装就返回 None，回退到双字切分。"""
    global _JIEBA
    if _JIEBA is None:
        try:
            import jieba  # type: ignore
            try:
                jieba.setLogLevel(60)   # 不往日志里打"Building prefix dict…"
            except Exception:
                pass
            for w in USER_WORDS:
                try:
                    jieba.add_word(w, freq=2000)
                except Exception:
                    pass
            _JIEBA = jieba
        except Exception:
            _JIEBA = False
    return _JIEBA or None


def terms(text: str, limit: int = 80) -> Set[str]:
    """Bag of terms: latin words ≥3 chars, Chinese words via 分词.

    早先用的是 CJK 双字滑窗，实测 136 个词里大半是跨词边界的碎片
    （"个怎""要登""站字"），既不能作为"他反复提到的话题"，也让 relevance
    项形同虚设。有 jieba 就用词，没有就退回双字并过滤停用词——
    插件不能因为宿主没装分词库就崩。
    """
    if not text:
        return set()
    out: Set[str] = set()
    for w in TERM_RE.findall(text.lower()):
        if w not in STOPWORDS:
            out.add(w)

    jb = _jieba()
    if jb is not None:
        for raw in jb.cut(text):
            w = raw.strip().lower()
            if len(w) < 2 or w in STOPWORDS:
                continue
            if not (CJK_RE.fullmatch(w) or TERM_RE.fullmatch(w)):
                continue          # 标点、数字、混排
            out.add(w)
    else:
        for run in CJK_RE.findall(text):
            if len(run) == 1:
                continue
            for i in range(len(run) - 1):
                bg = run[i:i + 2]
                if bg not in STOPWORDS:
                    out.add(bg)

    if len(out) > limit:
        out = set(sorted(out)[:limit])
    return out


def topic_terms(text: str, limit: int = 40) -> Set[str]:
    """只取名词性词，用于"他反复提到的话题"。

    为什么不能用 terms()：实测他 14 天里跨天最多的词是
    直接 / 相关 / 下载 / 优化 / 不能 / 完成 / 继续——这些是给我下的指令，
    不是他关心的事。真正有信息的是名词：助学金、大创、推免、实验报告。
    jieba 不可用时退回 terms()，宁可粗糙也不要空。
    """
    jb = _jieba()
    if jb is None:
        return terms(text, limit)
    try:
        import jieba.posseg as pseg
    except Exception:
        return terms(text, limit)

    out: Set[str] = set()
    for word, flag in pseg.cut(text or ""):
        w = word.strip().lower()
        if len(w) < 2 or w in STOPWORDS:
            continue
        # 排除法而不是白名单：白名单会把未登录词（大创、推免、分流）一起误杀，
        # 而这些恰恰是最该被捕捉的简称。只排除明确的虚词、动词、形容词词性。
        if flag[0] in "d c p u r m q y e o w f t v a":
            continue
        if not (CJK_RE.fullmatch(w) or TERM_RE.fullmatch(w)):
            continue
        out.add(w)
    if len(out) > limit:
        out = set(sorted(out)[:limit])
    return out


# --------------------------------------------------------------------------
# Items: deadlines, open loops, watches
# --------------------------------------------------------------------------

@dataclass
class Item:
    id: str
    title: str
    kind: str = "open_loop"          # deadline | open_loop | watch
    due: Optional[float] = None      # epoch seconds
    weight: float = 0.55             # 0..1 subjective importance
    created_at: float = 0.0
    nudges: int = 0
    last_nudged_at: Optional[float] = None
    status: str = "open"             # open | done | dropped
    tags: List[str] = field(default_factory=list)
    note: str = ""
    type: str = ""                   # submit | attend | apply | reply | other

    def is_open(self) -> bool:
        return self.status == "open"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "kind": self.kind, "due": self.due,
            "weight": self.weight, "created_at": self.created_at, "nudges": self.nudges,
            "last_nudged_at": self.last_nudged_at, "status": self.status,
            "tags": list(self.tags), "note": self.note, "type": self.type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Item":
        return cls(
            id=str(d.get("id") or ""), title=str(d.get("title") or ""),
            kind=str(d.get("kind") or "open_loop"),
            due=float(d["due"]) if d.get("due") is not None else None,
            weight=float(d.get("weight", 0.55)),
            created_at=float(d.get("created_at") or 0.0),
            nudges=int(d.get("nudges") or 0),
            last_nudged_at=float(d["last_nudged_at"]) if d.get("last_nudged_at") else None,
            status=str(d.get("status") or "open"),
            tags=[str(t) for t in (d.get("tags") or [])],
            note=str(d.get("note") or ""), type=str(d.get("type") or ""),
        )


# --------------------------------------------------------------------------
# Salience
# --------------------------------------------------------------------------

def urgency(item: Item, now: float) -> float:
    """0..1. Deadlines follow a piecewise curve; open loops age slowly."""
    if item.kind == "review":
        # 周体检没有截止时间，但它是"值得说一句"的固定事件
        return 0.85
    if item.kind == "sense":
        # 推测出来的需求：比硬期限低一档（前者 0.85），但必须能过门槛——
        # 定 0.55 时实测总分只有 0.57，永远低于 0.62，功能等于不存在。
        # 0.72 让它在没有更紧的事时说得出口，同时抢不过真正的截止。
        return 0.72
    if item.due is None:
        if not item.created_at:
            return 0.35
        days = max(0.0, (now - item.created_at) / 86400.0)
        # An open loop is worth less the longer it sits untouched, but never zero:
        # it is a standing commitment, not a timer.
        return min(0.55, 0.20 + 0.05 * days) * (0.6 ** (max(0.0, days - 21.0) / 7.0))
    hours = (item.due - now) / 3600.0
    if hours <= 0:
        # Overdue does not stay maximally urgent forever.  A deadline missed an hour
        # ago deserves a word; one missed last week is a backlog item, and nagging
        # about a "prepare for the meeting" task after the meeting happened is pure
        # noise.  Geometric decay with a floor keeps it in the pool without shouting.
        days_late = (-hours) / 24.0
        return max(0.20, 0.88 ** days_late)
    if hours <= 24:
        return 0.90 - 0.20 * (hours / 24.0)
    if hours <= 72:
        return 0.70 - 0.30 * ((hours - 24.0) / 48.0)
    if hours <= 168:
        return 0.40 - 0.25 * ((hours - 72.0) / 96.0)
    return 0.15


def novelty(item: Item, now: float) -> float:
    """First sighting 1.0, then 1/(1+n), and zero inside a 24h refractory window."""
    if item.last_nudged_at and (now - item.last_nudged_at) < 86400:
        return 0.0
    return min(1.0, 1.0 / (1 + max(0, item.nudges)))


def relevance(item: Item, context_terms: Set[str]) -> float:
    if not context_terms:
        return 0.0
    item_terms = terms(" ".join([item.title, *item.tags, item.note]))
    if not item_terms:
        return 0.0
    hits = len(item_terms & context_terms)
    return min(1.0, hits / 3.0)


def continuity(item: Item, now: float) -> float:
    """两条曲线取大：短期按天衰减，长期回升。

    只有 0.97**days 时的含义是「老了就不值得说」，第 28 天只剩 0.42。
    但对一个扛了三周的承诺，恰恰是「还在」才更该说。所以第 7 天起并入
    一条回升曲线，第 28 天回到 1.0。

    实测：第 7 天 0.808（谷底前后），第 10 天 0.760（全局最低），
    第 14 天 0.813，第 21 天 0.907，第 28 天 1.000。
    谷底绝不能低于 1.0 —— 放置一周的事项不该比刚建的还轻。
    """
    born = item.created_at or now
    days = max(0.0, (now - born) / 86400.0)
    short = 0.97 ** days
    if days < 7.0:
        return short
    long_arc = 0.72 + 0.28 * min(1.0, (days - 7.0) / 21.0)
    return max(short, long_arc)


def interruption_cost(*, daily_count: int, daily_budget: int, ignore_streak: int,
                      user_active: bool) -> float:
    """0..1. What it costs to speak right now (GA's attention bandwidth, made explicit)."""
    budget_pressure = 0.0 if daily_budget <= 0 else min(1.0, daily_count / daily_budget)
    streak_pressure = min(1.0, ignore_streak / 3.0)
    active_pressure = 1.0 if user_active else 0.0
    return min(1.0, 0.30 * budget_pressure + 0.45 * streak_pressure + 0.25 * active_pressure)


def score_item(item: Item, now: float, context_terms: Set[str], *,
               daily_count: int = 0, daily_budget: int = 3, ignore_streak: int = 0,
               user_active: bool = False, type_offset: float = 0.0) -> Dict[str, float]:
    """Weighted salience minus interruption cost, plus the components for the log."""
    u = urgency(item, now)
    i = max(0.0, min(1.0, item.weight))
    n = novelty(item, now)
    r = relevance(item, context_terms)
    c = continuity(item, now)
    base = 0.50 * u + 0.25 * i + 0.15 * n + 0.10 * r
    decayed = base * (0.40 + 0.60 * c)
    cost = interruption_cost(daily_count=daily_count, daily_budget=daily_budget,
                             ignore_streak=ignore_streak, user_active=user_active)
    total = max(0.0, min(1.0, decayed - 0.30 * cost + type_offset))
    return {"total": round(total, 3), "urgency": round(u, 3), "importance": round(i, 3),
            "novelty": round(n, 3), "relevance": round(r, 3), "continuity": round(c, 3),
            "cost": round(cost, 3), "base": round(base, 3),
            "type_offset": round(type_offset, 3)}


# --------------------------------------------------------------------------
# 类型级偏好：哪一类提醒他真的会动
# --------------------------------------------------------------------------

TYPES = ("submit", "attend", "apply", "reply", "other")

_TYPE_RULES = (
    ("submit", r"提交|上传|交(?!代)|写|报告|作业|材料|填|补交"),
    ("attend", r"会议|集合|培训|讲座|活动|到场|见面|面试|答辩"),
    ("apply", r"申请|办理|报名|核对|审批|缴费|领取|确认信息"),
    ("reply", r"回复|答复|表态|接龙"),
)


def item_type(item: Item) -> str:
    """优先用抽取时给的 type；没有就按标题动词猜，猜不出归 other。"""
    t = (item.type or "").strip().lower()
    if t in TYPES:
        return t
    for name, pat in _TYPE_RULES:
        if re.search(pat, item.title or ""):
            return name
    return "other"


def type_offset(type_stats: Optional[Dict[str, Any]], t: str, *,
                min_samples: int = 5, cap: float = 0.08) -> float:
    """按历史响应率给该类事调整门槛。

    样本人少时不乱动——五条以内的"命中率"没有统计意义。
    响应率高说明这类话他愿意听，降门槛；低就抬。上限 ±0.08，避免某一类
    把整个账本带偏。
    """
    d = (type_stats or {}).get(t) or {}
    n = int(d.get("sent") or 0)
    if n < min_samples:
        return 0.0
    rate = float(d.get("engaged") or 0) / n
    return max(-cap, min(cap, (rate - 0.5) * 0.2))


# --------------------------------------------------------------------------
# Time windows
# --------------------------------------------------------------------------

def parse_hhmm(text: str) -> Optional[int]:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", text or "")
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    return h * 60 + mi


def in_quiet_hours(local_hm: int, start_hm: int, end_hm: int) -> bool:
    """Handles windows that wrap midnight (23:00 -> 08:00)."""
    if start_hm == end_hm:
        return False
    if start_hm < end_hm:
        return start_hm <= local_hm < end_hm
    return local_hm >= start_hm or local_hm < end_hm


def local_day_key(now: float, tz: timezone) -> str:
    return datetime.fromtimestamp(now, tz).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------

@dataclass
class Verdict:
    allowed: bool
    reason: str
    item: Optional[Item] = None
    components: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.0
    urgent: bool = False

    def to_log(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed, "reason": self.reason,
            "item_id": self.item.id if self.item else None,
            "item_title": self.item.title if self.item else None,
            "score": self.components.get("total"),
            "components": self.components, "threshold": self.threshold,
            "urgent": self.urgent,
        }


def gate(*, items: Sequence[Item], now: float, local_hm: int, dial: str,
         quiet_hours: Tuple[str, str], min_idle_minutes: int, cooldown_minutes: int,
         daily_budget: int, daily_count: int, last_injection_at: Optional[float],
         last_user_turn_at: Optional[float], ignore_streak: int, context_terms: Set[str],
         adaptive_offset: float = 0.0, skip_reason: str = "",
         urgent_window_hours: float = 6.0, urgent_min_idle_seconds: float = 180.0,
         type_stats: Optional[Dict[str, Any]] = None,
         max_nudges: int = 2) -> Verdict:
    """The deterministic half of proactivity: everything decidable without a model.

    Two idle lanes, not one.  The quiet lane is the default (user must be away for
    `min_idle_minutes`).  The urgent lane exists because the original single-lane
    design meant a user who chats all day never hears about a deadline closing in
    six hours: a hard deadline inside the urgent window may interrupt once the user
    has been silent for `urgent_min_idle_seconds` — still at a turn boundary, still
    never mid-generation.
    """
    threshold, dial_budget, dial_idle, dial_cooldown = DIALS.get(dial, DIALS["normal"])
    if skip_reason:
        return Verdict(False, skip_reason, threshold=threshold)
    if dial == "off":
        return Verdict(False, "dial=off", threshold=threshold)

    start_hm = parse_hhmm(quiet_hours[0])
    end_hm = parse_hhmm(quiet_hours[1])
    if start_hm is not None and end_hm is not None and in_quiet_hours(local_hm, start_hm, end_hm):
        return Verdict(False, "quiet_hours", threshold=threshold)

    budget = min(daily_budget, dial_budget)
    if daily_count >= budget:
        return Verdict(False, f"daily_budget({daily_count}/{budget})", threshold=threshold)

    user_active = last_user_turn_at is not None and (now - last_user_turn_at) < 3600
    best: Optional[Item] = None
    best_comp: Dict[str, float] = {}
    for it in items:
        if not it.is_open() or not it.title:
            continue
        # 催满就退场。没有这条上限，一条分数够高的事项会隔几小时再发一次，
        # 实测同一条会址集合三天里发了三次——用户没回，不代表要再说一遍。
        # 硬期限临门（6 小时内）例外一次，那是最后一次值得打断的机会。
        cap = max(1, int(max_nudges))
        if (it.kind == "deadline" and it.due is not None
                and 0 <= (it.due - now) <= urgent_window_hours * 3600):
            cap += 1
        if it.nudges >= cap:
            continue
        comp = score_item(it, now, context_terms, daily_count=daily_count,
                          daily_budget=budget, ignore_streak=ignore_streak,
                          user_active=user_active,
                          type_offset=type_offset(type_stats, item_type(it)))
        if best is None or comp["total"] > best_comp.get("total", -1):
            best, best_comp = it, comp

    if best is None:
        return Verdict(False, "no_open_items", threshold=threshold)

    # 冷却必须在选出 best 之后再判：否则紧急通道会被常规冷却先挡死，
    # "静默三分钟就能说话"只是一句空话。硬期限临门时冷却压到三分之一。
    soon = (best.kind == "deadline" and best.due is not None
            and 0 <= (best.due - now) <= urgent_window_hours * 3600)
    if last_injection_at is not None:
        gap_min = (now - last_injection_at) / 60.0
        cool = max(cooldown_minutes, dial_cooldown)
        if soon:
            cool = max(10.0, cool / 3.0)
        if gap_min < cool:
            return Verdict(False, f"cooldown({int(gap_min)}<{int(cool)}m)",
                           item=best, components=best_comp, threshold=threshold)

    urgent = False
    idle_need = max(min_idle_minutes, dial_idle)
    if last_user_turn_at is not None:
        idle_min = (now - last_user_turn_at) / 60.0
        if idle_min < idle_need:
            if not (soon and idle_min * 60 >= urgent_min_idle_seconds):
                return Verdict(False, f"user_active(idle {int(idle_min)}<{idle_need}m)",
                               item=best, components=best_comp, threshold=threshold)
            urgent = True

    eff = min(0.98, threshold + max(0.0, adaptive_offset))
    if best_comp["total"] < eff:
        return Verdict(False, f"below_threshold({best_comp['total']}<{round(eff, 3)})",
                       item=best, components=best_comp, threshold=eff)
    return Verdict(True, "urgent" if urgent else "passed", item=best,
                   components=best_comp, threshold=eff, urgent=urgent)


# --------------------------------------------------------------------------
# Feedback adaptation (ProactiveAgent's "don't disturb when ignored")
# --------------------------------------------------------------------------

FEEDBACK_WINDOW_SECS = 45 * 60
# 应答期限：超过这个时间，pending 一律作废而不归因。
# 与 WINDOW 的区别——WINDOW 判"没回话算不算无视"，TTL 判"这条 probe 还有没有意义"。
FEEDBACK_TTL_SECS = 12 * 3600


def classify_feedback(pending: Optional[Dict[str, Any]], now: float,
                      user_replied: bool, user_text: str = "") -> Optional[str]:
    """Return 'engaged' | 'ignored' | None for a pending injection.

    判定不能只看"他回没回话"：同一会话里说任何一句都算 engaged 的话，
    他随口回个"嗯"或继续聊别的，都会被记成"这条提醒有效"，门槛一路下调，
    结果是越用越吵。所以要看回的内容跟这件事有没有关系。
    """
    if not pending:
        return None
    at = float(pending.get("at") or 0)
    # 超过应答期限就直接作废。窗口内判 engaged/ignored，窗口外连"回没回"都不可靠：
    # 一条一天前的提醒，今天随口回句"好的"记成有效会一路调低门槛，
    # 记成无视则是凭空记账。两种归因都在污染账本，不如丢弃。
    if now - at > FEEDBACK_TTL_SECS:
        return "expired"
    if user_replied:
        text = (user_text or "").strip()
        title = str(pending.get("title") or "")
        # 纯确认词（收到/好的/嗯）本身就是对提醒的回应，直接算 engaged
        if any(a in text for a in ACK_WORDS):
            return "engaged"
        title_terms = terms(title)
        reply_terms = terms(text)
        # 标题切不出词（太短），或他确实提到了这件事 → engaged
        if not title_terms or (title_terms & reply_terms):
            return "engaged"
        # 回了话，但跟这件事毫无关系：这条提醒没落到他心里
        return "ignored"
    if now - at >= FEEDBACK_WINDOW_SECS:
        return "ignored"
    return None


def next_adaptive_offset(current: float, outcome: str) -> float:
    """Ignored: raise the bar 0.04 per miss (cap 0.14). Engaged: relax 0.03."""
    if outcome == "ignored":
        return round(min(0.14, current + 0.04), 3)
    if outcome == "engaged":
        return round(max(0.0, current - 0.03), 3)
    return current


def humanize_due(due: Optional[float], now: float) -> str:
    if due is None:
        return "无截止时间"
    delta = due - now
    if delta <= 0:
        return f"已逾期 {abs(delta) / 3600:.0f} 小时"
    if delta < 3600:
        return f"还有 {delta / 60:.0f} 分钟"
    if delta < 86400:
        return f"还有 {delta / 3600:.1f} 小时"
    return f"还有 {delta / 86400:.1f} 天"


def summarize_items(items: Iterable[Item], now: float, limit: int = 5) -> str:
    rows = []
    for it in sorted(items, key=lambda x: (x.due or 9e18))[:limit]:
        if not it.is_open():
            continue
        rows.append(f"- {it.title}（{humanize_due(it.due, now)}）")
    return "\n".join(rows)
