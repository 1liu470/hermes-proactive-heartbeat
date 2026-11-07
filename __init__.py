"""proactive-heartbeat — a proactive loop for Hermes, built on verified sources.

Architecture (one line): deterministic signals -> salience gate -> one cheap
model verdict -> inject a turn -> learn from whether the user answered.

Where each piece came from:

* **Heartbeat over coalesced events** — ProactiveAgent (thunlp).  The model is
  never polled on a timer; the cheap layer polls, and the model is asked only
  when a candidate already cleared a numeric bar.
* **Salience = urgency/importance/novelty/relevance with geometric decay** —
  Generative Agents (Stanford), whose retrieval score is a weighted sum of
  recency (0.99^n), importance and relevance, gated by an attention budget.
* **Feedback-conditioned interruption policy** — ProactiveAgent's rule that an
  ignored proposal must make the agent quieter, a rejected one must change
  approach, and an accepted one must not be followed by related proposals.
  Here it becomes a numeric offset on the threshold plus a cooldown multiplier.
* **The product bar** — Muse (Meta): a proactive message has to be worth the
  interruption, the user gets a dial, and every decision is auditable.  Hence
  `/proactive off|quiet|normal|eager` and `decisions.jsonl`.

Safety posture: fail closed.  Any error, any missing session, any ambiguity in
the model's answer means **no message is sent**.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import engine as E
from . import longterm as LT
from . import platforms as PF
from . import prompts as P
from . import research as RS
from . import sense as SN
from . import store as S

log = logging.getLogger("plugins.proactive_heartbeat")

PLUGIN_ID = "proactive-heartbeat"

#: Where the heartbeat is allowed to speak.  Unattended surfaces never pay for it.
SKIP_PLATFORMS = {"cron", "subagent"}

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "dial": "normal",
    "tick_seconds": 120,
    "quiet_hours": ["23:00", "08:00"],
    "min_idle_minutes": 15,
    "cooldown_minutes": 40,
    "daily_budget": 8,
    # 紧急通道：硬期限进入窗口后，用户只需静默这么久就能被提醒一次
    "urgent_window_hours": 6,
    "urgent_min_idle_seconds": 180,
    "max_nudges": 2,
    # 主动研究：进入期限窗口时先替他查一次资料，再决定怎么说
    "research_enabled": True,
    "research_window_hours": 72,
    "research_max_items": 3,
    "research_daily_cap": 6,
    # 长期层：目标同步与每周体检
    "longterm_enabled": True,
    "review_weekday": 6,
    "review_hour": 20,
    # 自感知层：没人交办时，从上下文推测他需要什么
    "sense_enabled": True,
    "sense_daily_cap": 2,
    "sense_repeat_days": 7,
    # 投递路由：active（白名单内每个已建会话的平台各发一条）/ platform:<名字>（单平台）
    # / platforms:<a,b>（多平台）。all 已删除，它的语义是广播。
    "delivery_mode": "active",
    # 硬闸一：只有这些平台允许投递。留空 = 一律不发（fail closed）。
    # `*` = 任何已接通的平台。平台是否接通由人决定，插件不代为连接；
    # 没接通的平台没有会话，也就发不出去。
    "allowed_platforms": ["*"],
    # 硬闸二：只投私聊，不进群。靠 session_key 里的 :dm:/:private: 一类标记判断。
    "require_direct": True,
    "threshold": 0.62,
    "model_timeout_seconds": 60,
    "max_message_tokens": 220,
    "intent_block_cooldown_minutes": 45,
}

_RUNTIME: Dict[str, Any] = {"thread": None, "stop": None, "ctx": None, "lock": threading.RLock()}


def _runtime() -> Dict[str, Any]:
    """Process-wide heartbeat slot.

    The plugin can be hot-reloaded into a live gateway (``hermes plugins enable``
    does exactly that), which produces a *new* module object while the old
    thread keeps running.  Module-level globals would then hand each reload its
    own slot and multiply the tick rate.  Parking the slot on the ``threading``
    module makes it survive reloads: one stop event, one thread, and the
    callable is refreshed on every register so reloads take effect.
    """
    reg = getattr(threading, "_proactive_heartbeat_runtime", None)
    if reg is None:
        reg = {"thread": None, "stop": None, "ctx": None, "tick_fn": None,
               "interval": 120.0, "lock": threading.RLock()}
        setattr(threading, "_proactive_heartbeat_runtime", reg)
    return reg


_RUNTIME = _runtime()


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _cfg(ctx, key: str) -> Any:
    try:
        v = ctx.get_config(key, DEFAULTS.get(key))
    except Exception:
        v = DEFAULTS.get(key)
    return DEFAULTS.get(key) if v is None else v


def _tz() -> timezone:
    name = ""
    try:
        import os
        name = os.environ.get("HERMES_TIMEZONE", "").strip()
        if not name:
            cfg = S.hermes_home() / "config.yaml"
            if cfg.exists():
                import yaml
                data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict):
                    tz_cfg = data.get("timezone")
                    if isinstance(tz_cfg, str):
                        name = tz_cfg.strip()
    except Exception:
        name = ""
    if name:
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(name)
        except Exception:
            pass
    return datetime.now().astimezone().tzinfo or timezone.utc


def _session_env(name: str, default: str = "") -> str:
    try:
        from gateway.session_context import get_session_env
        return get_session_env(name, default) or default
    except Exception:
        import os
        return os.getenv(name, default)


def _parse_due(text: str, now: float) -> Optional[float]:
    """ISO date, 'YYYY-MM-DD HH:MM', or a relative '3d' / '12h' / '90m'."""
    text = (text or "").strip()
    if not text:
        return None
    rel = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhdw])", text.lower())
    if rel:
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[rel.group(2)]
        return now + float(rel.group(1)) * mult
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m-%d %H:%M", "%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if fmt in ("%m-%d %H:%M", "%m-%d"):
            dt = dt.replace(year=datetime.fromtimestamp(now, _tz()).year)
        return dt.replace(tzinfo=_tz()).timestamp()
    try:
        return datetime.fromisoformat(text).astimezone(_tz()).timestamp()
    except Exception:
        return None


def _new_id(title: str, now: float) -> str:
    import hashlib
    h = hashlib.sha1(f"{title}{now}".encode("utf-8")).hexdigest()[:6]
    return f"it_{int(now)}_{h}"


def _items(state: Dict[str, Any]) -> List[E.Item]:
    out = []
    for raw in state.get("items") or []:
        try:
            out.append(E.Item.from_dict(raw))
        except Exception:
            continue
    return out


def _commit(mutator) -> None:
    """唯一安全的写入口：在锁内读取磁盘最新状态，再施加增量。

    为什么不能用 load() → 改 → save()：两步之间锁已释放，而心跳一次 tick
    会在模型调用与联网检索上停留数秒。这期间 hook 清掉的 pending、刷新的
    last_user_turn_at，会被心跳手里那份几秒前的快照整份写回，效果等于从没改过。
    实测症状：pending 卡在一天前不消失、每轮对话反复结算同一条反馈、
    adaptive_offset 在 0.0 与 0.04 之间来回跳。
    """
    S.update(mutator)


def _stats_inc(st: Dict[str, Any], key: str, n: int = 1) -> None:
    stats = st.setdefault("stats", {})
    stats[key] = int(stats.get(key) or 0) + n


def _reason_kind(reason: Any) -> str:
    """拒因归类：`below_threshold(0.619<0.62)` 与 `below_threshold(0.618<0.62)` 同类。

    日志去重若拿整串比分，分数每轮都在动，全天每 15 分钟就会落一行相同的拒绝。
    """
    return str(reason or "").split("(", 1)[0].strip()


def _mutate_items(state: Dict[str, Any], items: List[E.Item]) -> None:
    state["items"] = [i.to_dict() for i in items]


def _roll_day(state: Dict[str, Any], now: float) -> None:
    rt = state.setdefault("runtime", {})
    key = E.local_day_key(now, _tz())
    if rt.get("daily_key") != key:
        rt["daily_key"] = key
        rt["daily_count"] = 0


# --------------------------------------------------------------------------
# the heartbeat tick
# --------------------------------------------------------------------------

def _evidence(item: E.Item, now: float, comp: Dict[str, float]) -> str:
    bits = [f"{'截止' if item.kind == 'deadline' else '记录于'}"]
    if item.due is not None:
        bits.append(E.humanize_due(item.due, now))
    if item.nudges:
        bits.append(f"此前提醒过 {item.nudges} 次")
    if item.note:
        bits.append(item.note[:80])
    bits.append(f"显著性 {comp.get('total')}")
    return "；".join(b for b in bits if b)


def _pick_related(items: List[E.Item], best: E.Item, limit: int = 2) -> List[E.Item]:
    """同源同标签的候选，允许模型把它们并成一句话。

    判据故意严格（同一来源前缀 + 有共同标签）：不同来源合并会退化成清单，
    那正是这份插件最想避免的东西。
    """
    src = best.id.split(":")[0]
    tags = set(best.tags or [])
    related = [i for i in items
               if i.id != best.id and i.is_open()
               and i.id.split(":")[0] == src
               and tags and (set(i.tags or []) & tags)]
    related.sort(key=lambda i: abs((i.due or 0) - (best.due or 0)))
    return related[:limit]


def _rows(candidates: List[E.Item], now: float) -> str:
    lines = []
    for n, it in enumerate(candidates, 1):
        src = "、".join(t for t in (it.tags or []) if t != "微信") or it.id.split(":")[0]
        extra = f"｜{it.note[:60]}" if it.note else ""
        lines.append(f"{n}. {it.title}｜{it.kind}｜{E.humanize_due(it.due, now)}｜来源 {src}{extra}")
    return "\n".join(lines)


#: 输出侧硬防线。注入测试里模型三次都扛住了，但那是模型能力，不是架构保证——
#: 万一某次被绕过，这里必须拦住。提醒本身不需要链接（资料已落盘）与任何凭证类词汇。
_GUARD_SENSITIVE = ("银行卡", "卡号", "密码", "验证码", "身份证号", "转账", "汇款",
                    "打款", "扫码", "支付宝口令")
_GUARD_URL = re.compile(r"https?://|www\.|\.com\b|\.cn\b|\.net\b|\.top\b|\.xyz\b", re.I)


def _guard(decision: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """裁决结果的最后一道闸：命中敏感词或链接就不发。

    这是纵深防御的最后一层。前两层是「提示里标明数据不是指令」与「模型判断」，
    但前两层都依赖模型，只有这一层是确定性的。
    """
    if not decision:
        return decision
    msg = str(decision.get("message") or "")
    if not msg.strip():
        return decision
    if any(w in msg for w in _GUARD_SENSITIVE):
        log.warning("proactive: 拦截含敏感词的消息：%s", msg[:60])
        return None
    if _GUARD_URL.search(msg):
        log.warning("proactive: 拦截含链接的消息：%s", msg[:60])
        return None
    return decision


def _model_text(ctx, messages: List[Dict[str, str]], *, purpose: str) -> str:
    """一次有界调用。任何异常都返回空串，由调用方当作「不发」。"""
    try:
        res = ctx.llm.complete(
            messages,
            temperature=0.3,
            max_tokens=int(_cfg(ctx, "max_message_tokens")),
            timeout=float(_cfg(ctx, "model_timeout_seconds")),
            purpose=purpose,
        )
    except Exception as exc:
        log.debug("proactive: model call failed (%s): %s", purpose, exc)
        return ""
    text = getattr(res, "text", None) or getattr(res, "content", None) or ""
    return text if isinstance(text, str) else str(text)


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _judge(ctx, item: E.Item, comp: Dict[str, float], now: float,
           related: Optional[List[E.Item]] = None,
           intel: str = "") -> Optional[Dict[str, Any]]:
    """第一段：发不发、为什么、用哪几条。不产出给人看的句子。

    这一步刻意不允许写话——一旦模型先把句子写好，它就会为发出去找理由。
    """
    candidates = [item] + list(related or [])
    merge_note = ""
    if len(candidates) > 1:
        merge_note = ("第 2 条起与第 1 条同源，若它们指向同一个动作，可并成一句；"
                      "若动作不同，只发第 1 条。\n")
    user = P.JUDGE_TEMPLATE.format(
        rows=_rows(candidates, now),
        intel=(intel + "\n") if intel else "",
        merge_note=merge_note,
    )
    text = _model_text(ctx, [{"role": "system", "content": P.JUDGE_SYSTEM},
                             {"role": "user", "content": user}], purpose="proactive-judge")
    data = _parse_json(text)
    if data is None:
        return None
    used = data.get("used")
    if not isinstance(used, list) or not used:
        used = [1]
    used = [int(u) for u in used if isinstance(u, (int, float)) or str(u).isdigit()][:3] or [1]
    return {"send": bool(data.get("send")),
            "reason": str(data.get("reason") or "")[:200],
            "used": used,
            "speech": str(data.get("speech") or "none")[:20],
            "candidates": candidates}


def _clean_draft(text: str) -> str:
    """模型常加引号、代码块或第二段解释；只留那一句。"""
    s = (text or "").strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    # 有的模型在"只写一句话"的阶段仍回一整个 JSON 对象。照原样发出去，
    # 用户收到的就是 {"send": true, "message": ...}。这里先把壳剥掉。
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for key in ("message", "text", "content", "reply"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    s = val.strip()
                    break
            else:
                s = ""
    for left, right in (('"', '"'), ("'", "'"), ("“", "”"), ("「", "」"), ("‘", "’")):
        if len(s) > 2 and s.startswith(left) and s.endswith(right):
            s = s[1:-1].strip()
            break
    for sep in ("\n\n", "\n"):
        if sep in s:
            head = s.split(sep)[0].strip()
            if head:
                s = head
                break
    return s[:400]


def _write(ctx, item: E.Item, judged: Dict[str, Any], now: float,
           intel: str = "") -> str:
    """第二段：判断已经定了，这里只决定怎么说。返回空串即放弃这次提醒。"""
    user = P.WRITE_TEMPLATE.format(
        title=item.title,
        due_human=E.humanize_due(item.due, now),
        reason=judged.get("reason") or "",
        intel=(intel + "\n") if intel else "",
        voice=P.load_voice_rules(),
    )
    text = _model_text(ctx, [{"role": "system", "content": P.WRITE_SYSTEM},
                             {"role": "user", "content": user}], purpose="proactive-write")
    return _clean_draft(text)


def _ask_model(ctx, item: E.Item, comp: Dict[str, float], now: float,
               related: Optional[List[E.Item]] = None,
               intel: str = "") -> Optional[Dict[str, Any]]:
    """两段式裁决：先判断，判为发才写话。

    合在一起时模型倾向于「既然句子都写好了，那就发吧」。拆开之后，
    判断那一步不产出任何给主人看的文本，沉默才能回到默认答案。
    代价是判为发时多一次调用；判为不发时仍然只花一次。
    """
    judged = _judge(ctx, item, comp, now, related, intel)
    if judged is None:
        return None
    if not judged["send"]:
        return {"send": False, "message": "", "reason": judged["reason"],
                "used": judged["used"], "speech": "none",
                "candidates": judged["candidates"]}
    draft = _write(ctx, item, judged, now, intel)
    if not draft:
        # 写不出话来，等于判断被推翻——不硬发半句。
        return None
    result = {"send": True, "message": draft, "reason": judged["reason"],
              "used": judged["used"], "speech": judged["speech"],
              "candidates": judged["candidates"]}
    return _guard(result)


def _allowed_platforms(ctx) -> list:
    raw = _cfg(ctx, "allowed_platforms")
    if isinstance(raw, str):
        raw = raw.replace("，", ",").split(",")
    return [str(x).strip().lower() for x in (raw or []) if str(x).strip()]


def _want_platforms(mode: str) -> list:
    """delivery_mode 想发到哪些平台。

        platform:qqbot              单平台（旧写法，保留）
        platforms:qqbot,telegram    多平台
        active                      白名单内、已经建立过会话的每个平台（由 _deliver 展开）
        all                         已删除——它的语义是广播

    认不出的写法返回空，由调用方决定是走白名单回退还是不发。
    """
    m = str(mode or "").strip().lower()
    if m.startswith("platforms:"):
        tail = m.split(":", 1)[1].replace("，", ",")
        return [x.strip() for x in tail.split(",") if x.strip()]
    if m.startswith("platform:"):
        one = m.split(":", 1)[1].strip()
        return [one] if one else []
    return []


def _deliver(ctx, state: Dict[str, Any], content: str, now: float) -> tuple:
    """把消息投到哪里，返回 (是否至少成功一次, 用到的平台)。

    平台判定交给 `platforms.py`：它按网关键的 `chat_type` 段结构化解析，
    不靠猜子串。三道闸：

      1. 平台必须在 allowed_platforms 里；空名单 fail closed
      2. 键里的平台与配置的平台必须一致（伪造键不算数）
      3. require_direct 为真时必须是 direct——`unknown` 也拒

    回退路径只从**白名单内**的会话里挑，且同样过这三道闸。旧版回退到
    `last_session_key`，那是消息跑到桌面上的原因。
    """
    rt = state.get("runtime") or {}
    mode = str(_cfg(ctx, "delivery_mode") or "").lower()
    sessions = rt.get("sessions") or {}
    allowed = _allowed_platforms(ctx)
    require_direct = bool(_cfg(ctx, "require_direct"))

    if not allowed:
        log.warning("proactive: allowed_platforms 为空，拒绝一切投递")
        return False, ""

    if mode == "all":
        log.warning("proactive: delivery_mode=all 会广播到每个会话，已删除该语义；本次不发")
        return False, ""

    wants = _want_platforms(mode)
    if mode == "active":
        # 白名单内的每一个平台都发，不限最近一个。白名单若含 `*`，
        # 那就是所有已接通且有私聊会话的平台——接不通的没有会话，自然不会出现在这里。
        wants = [p for p, v in sessions.items()
                 if (v or {}).get("key") and PF.in_allowlist(p, allowed)]

    targets = [(p, (sessions.get(p) or {}).get("key")) for p in wants]
    targets = [(p, k) for p, k in targets if k]

    if not targets:
        cand = [(p, v.get("key"), float(v.get("at") or 0))
                for p, v in sessions.items()
                if v.get("key") and PF.in_allowlist(p, allowed)]
        if cand:
            cand.sort(key=lambda x: x[2], reverse=True)
            targets = [(cand[0][0], cand[0][1])]

    ok_any, used, blocked = False, [], []
    for platform, key in targets:
        ok, why = PF.is_allowed_target(key, platform, allowed, require_direct)
        if not ok:
            blocked.append(f"{platform}:{why}")
            continue
        try:
            if ctx.inject_message(content, role="user", session_key=key):
                ok_any = True
                used.append(str(platform).lower())
        except Exception as exc:
            log.debug("proactive: inject to %s failed: %s", platform, exc)
    if blocked:
        log.debug("proactive: 投递被闸门拦下 %s", blocked)
    return ok_any, ",".join(used)


def _tick(ctx) -> None:
    """One heartbeat step, serialised across threads and processes."""
    fd = S.acquire_tick_lock()
    if fd is None:
        return  # another tick is mid-decision; never speak twice for one moment
    try:
        _tick_body(ctx)
    finally:
        S.release_tick_lock(fd)


def _tick_body(ctx) -> None:
    now = time.time()
    tz = _tz()
    local_hm = datetime.fromtimestamp(now, tz).hour * 60 + datetime.fromtimestamp(now, tz).minute

    state = S.load()
    _roll_day(state, now)
    rt = state["runtime"]
    # 手工记的事项是权威。若以后接外部源，只读合并，不写回 state["items"]。
    items = _items(state)

    # 自感知层：没人交办的时候，从上下文推测一件他可能需要的事
    if bool(_cfg(ctx, "sense_enabled")):
        try:
            h = SN.pick(state, now,
                        daily_cap=int(_cfg(ctx, "sense_daily_cap") or 2),
                        repeat_days=float(_cfg(ctx, "sense_repeat_days") or 7))
            if h:
                items.append(SN.to_item(h, now))
        except Exception:
            log.debug("proactive: sense layer failed", exc_info=True)

    # 长期层：目标骨架每 6 小时同步一次，周体检每周注入一个候选
    if bool(_cfg(ctx, "longterm_enabled")):
        state.setdefault("longterm", {})
        # 语义归因队列：每次 tick 只处理一条，timeout 压在 15 秒，
        # 它跑在心跳线程里，慢一点可以，但不能拖住后面的门控判断。
        queue = state.get("goal_attr_pending") or []
        if queue and state.get("goals"):
            rec = queue[0]
            probe = E.Item(id="attr:probe", title=str(rec.get("title") or ""),
                           note=str(rec.get("note") or ""), tags=list(rec.get("tags") or []))
            try:
                gid = LT.attribute_llm(ctx, state, probe, state["goals"], now, timeout=15.0)
                if gid:
                    LT.touch(state, gid, now)
            except Exception:
                log.debug("proactive: goal attribution failed", exc_info=True)
            queue.pop(0)
            state["goal_attr_pending"] = queue
        try:
            if LT.review_due(state, now, weekday=int(_cfg(ctx, "review_weekday") or 6),
                             hour=int(_cfg(ctx, "review_hour") or 20), tz=tz):
                dig = LT.weekly_digest(state, now)
                txt = LT.digest_text(dig)
                if txt:
                    iso = "%s-W%02d" % __import__("datetime").datetime.fromtimestamp(now, tz).isocalendar()[:2]
                    items.append(E.Item(id=f"lt:{iso}", title="本周目标体检", kind="review",
                                        weight=0.85, created_at=now, note=txt))
                    lt["last_review_at"] = now
        except Exception:
            log.debug("proactive: weekly review build failed", exc_info=True)

    dial = str(state.get("dial") or _cfg(ctx, "dial") or "normal")
    platform = str(rt.get("last_platform") or "")

    verdict = E.gate(
        items=items, now=now, local_hm=local_hm, dial=dial,
        quiet_hours=tuple(_cfg(ctx, "quiet_hours") or ["23:00", "08:00"]),
        min_idle_minutes=int(_cfg(ctx, "min_idle_minutes")),
        cooldown_minutes=int(_cfg(ctx, "cooldown_minutes")),
        daily_budget=int(_cfg(ctx, "daily_budget")),
        daily_count=int(rt.get("daily_count") or 0),
        last_injection_at=rt.get("last_injection_at"),
        last_user_turn_at=rt.get("last_user_turn_at"),
        ignore_streak=int(state.get("ignore_streak") or 0),
        context_terms=set(rt.get("context_terms") or []),
        adaptive_offset=float(state.get("adaptive_offset") or 0.0),
        skip_reason="platform:%s" % platform if platform in SKIP_PLATFORMS else "",
        urgent_window_hours=float(_cfg(ctx, "urgent_window_hours") or 6),
        urgent_min_idle_seconds=float(_cfg(ctx, "urgent_min_idle_seconds") or 180),
        type_stats=state.get("type_stats") or {},
        max_nudges=int(_cfg(ctx, "max_nudges") or 2),
    )

    _commit(lambda st: _stats_inc(st, "considered"))

    if not verdict.allowed:
        kind = _reason_kind(verdict.reason)
        changed = _reason_kind(rt.get("last_gate_reason")) != kind

        def _record_gate(st: Dict[str, Any]) -> None:
            _stats_inc(st, "gated")
            st["runtime"]["last_gate_reason"] = verdict.reason

        _commit(_record_gate)
        # 只在拒因类型切换时落一行。分数每 tick 都在动，按整串比对等于全天刷屏。
        if changed:
            S.log_decision({"ts": now, "event": "gate", **verdict.to_log()})
        return

    item = verdict.item
    assert item is not None
    related = _pick_related(items, item)
    # 先替他把功课做了：硬期限进入窗口时查一次，查不到不阻塞
    hits = RS.gather(
        ctx, item, state, now,
        window_hours=float(_cfg(ctx, "research_window_hours") or 72),
        max_items=int(_cfg(ctx, "research_max_items") or 3),
        daily_cap=int(_cfg(ctx, "research_daily_cap") or 6),
    ) if bool(_cfg(ctx, "research_enabled")) else []
    intel = RS.format_intel(hits)
    decision = _ask_model(ctx, item, verdict.components, now, related, intel)
    if not decision or not decision["send"]:
        def _record_declined(st: Dict[str, Any]) -> None:
            _stats_inc(st, "declined_by_model")
            st["runtime"]["last_gate_reason"] = "model_declined"

        _commit(_record_declined)
        S.log_decision({"ts": now, "event": "declined", **verdict.to_log(),
                        "model_reason": (decision or {}).get("reason", "no_answer")})
        return

    if not (rt.get("last_session_key") or rt.get("sessions")):
        _commit(lambda st: st["runtime"].__setitem__("last_gate_reason", "no_session"))
        S.log_decision({"ts": now, "event": "no_session", **verdict.to_log()})
        return

    # 查到的资料落盘：提醒可以只是两句话，但"我替你查了"必须交出实物。
    # 不落盘的话，那条消息里除了措辞什么也没有，等于空口说做了功课。
    attachment = ""
    if hits:
        path = RS.save_intel(item, hits, now)
        if path:
            head = RS.intel_headline(hits)
            attachment = "资料已存本机：%s%s\n" % (path, ("（%s）" % head) if head else "")
    content = P.INJECTION_TEMPLATE.format(
        marker=P.INJECTION_MARKER, title=item.title,
        due_human=E.humanize_due(item.due, now),
        evidence=_evidence(item, now, verdict.components),
        draft=decision["message"],
        attachment=attachment,
    )
    ok, landed = _deliver(ctx, state, content, now)
    if not ok:
        log.warning("proactive: injection failed on every target")
        _commit(lambda st: st["runtime"].__setitem__("last_gate_reason", "inject_rejected"))
        S.log_decision({"ts": now, "event": "inject_rejected", **verdict.to_log()})
        return

    # Sent: spend budget, arm the feedback probe, mark the item.
    # 全部在锁内基于磁盘最新状态记账。拿 tick 开头那份快照整份写回，
    # 会把这几秒里用户新记的事项、hook 清掉的 pending 一起抹掉。
    itype = E.item_type(item)
    sense_id = item.id.split(":", 1)[1] if item.id.startswith("sense:") else ""
    extra_ids = set()
    for n in decision.get("used") or []:
        cands = decision.get("candidates") or []
        if 1 < n <= len(cands):
            extra_ids.add(cands[n - 1].id)

    def _arm(st: Dict[str, Any]) -> None:
        fresh = _items(st)
        for it in fresh:
            if it.id == item.id or it.id in extra_ids:
                it.nudges += 1
                it.last_nudged_at = now
        _mutate_items(st, fresh)
        if sense_id:
            SN.mark(st, sense_id, now)
        r = st["runtime"]
        r["daily_count"] = int(r.get("daily_count") or 0) + 1
        r["last_injection_at"] = now
        r["last_gate_reason"] = "sent"
        r["pending"] = {"item_id": item.id, "at": now,
                        "session_key": r.get("last_session_key"),
                        "platforms": landed, "title": item.title, "type": itype,
                        "at_local": datetime.fromtimestamp(now, tz).isoformat()}
        ts = st.setdefault("type_stats", {}).setdefault(itype, {"sent": 0, "engaged": 0})
        ts["sent"] = int(ts.get("sent") or 0) + 1
        _stats_inc(st, "sent")

    _commit(_arm)
    S.log_decision({"ts": now, "event": "sent", **verdict.to_log(),
                    "message": decision["message"], "platforms": landed})


def _heartbeat(reg: Dict[str, Any]) -> None:
    stop = reg["stop"]
    while True:
        tick = max(15.0, float(reg.get("interval") or 120))
        if stop.wait(tick):
            return
        fn = reg.get("tick_fn")
        ctx = reg.get("ctx")
        if fn is None or ctx is None:
            continue
        try:
            fn(ctx)
        except Exception:
            log.warning("proactive: tick failed", exc_info=True)


def _ensure_thread(ctx) -> None:
    reg = _RUNTIME
    with reg["lock"]:
        reg["ctx"] = ctx
        reg["tick_fn"] = _tick
        reg["interval"] = max(15.0, float(_cfg(ctx, "tick_seconds") or 120))
        stop = reg.get("stop")
        if stop is not None and stop.is_set():
            return  # unloaded on purpose; a later hook call must not resurrect it
        th = reg.get("thread")
        if th is not None and th.is_alive():
            return  # hot reload: reuse the live thread, new callable already swapped in
        reg["stop"] = threading.Event()
        th = threading.Thread(target=_heartbeat, args=(reg,),
                              name="proactive-heartbeat", daemon=True)
        reg["thread"] = th
        th.start()
        log.info("proactive-heartbeat: heartbeat started (tick=%.0fs)", reg["interval"])


# --------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------

def _on_pre_llm_call(**kw: Any) -> Optional[Dict[str, str]]:
    ctx = _RUNTIME.get("ctx")
    if ctx is None:
        return None
    _ensure_thread(ctx)

    raw = kw.get("user_message")
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False) if raw else ""
    platform = str(kw.get("platform") or _session_env("HERMES_SESSION_PLATFORM")
                   or _session_env("HERMES_SESSION_SOURCE") or "")

    # Our own injection comes back through this hook as a user turn.  It must not
    # count as user activity, or the idle clock resets itself forever.
    if text.startswith(P.INJECTION_MARKER) or P.INJECTION_MARKER in text[:120]:
        return None

    now = time.time()
    tz = _tz()
    session_key = _session_env("HERMES_SESSION_KEY")
    outcome = None
    block = ""

    try:
        with S.transaction() as state:
            rt = state["runtime"]
            _roll_day(state, now)

            pending = rt.get("pending")
            if pending and platform in SKIP_PLATFORMS:
                outcome = None  # an unattended run is not a human answer
            elif pending:
                # Any human turn inside the window counts as "not ignored" — our own
                # injection is marker-guarded above and unattended runs are filtered just
                # above, so this only fires on a real person.  The session_key decides
                # *delivery*, not whether the user answered.
                outcome = E.classify_feedback(pending, now, user_replied=True,
                                              user_text=str(text or ""))
            else:
                outcome = None

            if outcome == "expired":
                # 过了应答窗口就丢弃：不计 offset，也不记 ignore。
                # 一条一天前的提醒，今天随口回句"好的"记成有效会一路调低门槛；
                # 记成无视更是凭空记账。两种归因都不如作废。
                rt["pending"] = None
                S.log_decision({"ts": now, "event": "feedback", "reason": "expired",
                                "outcome": "expired", "item_id": pending.get("item_id"),
                                "age_hours": round((now - float(pending.get("at") or now)) / 3600.0, 1)})
            elif outcome:
                state["adaptive_offset"] = E.next_adaptive_offset(
                    float(state.get("adaptive_offset") or 0.0), outcome)
                # 类型级账本：哪一类提醒他真的会动，单独记
                ptype = str((pending or {}).get("type") or "")
                if ptype and outcome == "engaged":
                    ts = state.setdefault("type_stats", {}).setdefault(ptype, {"sent": 0, "engaged": 0})
                    ts["engaged"] = int(ts.get("engaged") or 0) + 1
                if outcome == "ignored":
                    state["ignore_streak"] = int(state.get("ignore_streak") or 0) + 1
                else:
                    state["ignore_streak"] = 0
                    for it in _items(state):
                        if it.id == pending.get("item_id") and it.is_open():
                            it.last_nudged_at = max(it.last_nudged_at or 0, now)
                    _mutate_items(state, _items(state))
                rt["pending"] = None
                S.log_decision({"ts": now, "event": "feedback", "reason": outcome, "outcome": outcome,
                                "item_id": pending.get("item_id"),
                                "adaptive_offset": state["adaptive_offset"],
                                "ignore_streak": state["ignore_streak"]})

            # Refresh activity + per-platform session keys.
            if platform not in SKIP_PLATFORMS:
                rt["last_user_turn_at"] = now
                if session_key:
                    rt["last_session_key"] = session_key
                if platform:
                    rt["last_platform"] = platform
                if platform and session_key:
                    sessions = rt.setdefault("sessions", {})
                    sessions[platform] = {"key": session_key, "at": now}
                    if len(sessions) > 8:
                        keep = sorted(sessions.items(), key=lambda kv: float(kv[1].get("at") or 0))[-8:]
                        rt["sessions"] = dict(keep)
            if text:
                merged = set(rt.get("context_terms") or []) | E.terms(text)
                rt["context_terms"] = sorted(merged)[-60:]
                # 频次账：只保留集合判断不出"他反复提这件事"，需要出现次数与首末时间
                stats = rt.setdefault("term_stats", {})
                day = E.local_day_key(now, tz)
                for t in E.terms(text):
                    s = stats.get(t) or {"n": 0, "first": now, "last": now, "days": []}
                    s["n"] = int(s.get("n") or 0) + 1
                    s["last"] = now
                    days = s.setdefault("days", [])
                    if day not in days:
                        days.append(day)
                        s["days"] = days[-30:]      # 只留最近 30 个日期
                    stats[t] = s
                if len(stats) > 400:
                    drop = sorted(stats.items(), key=lambda kv: float(kv[1].get("last") or 0))[:-200]
                    for k, _ in drop:
                        stats.pop(k, None)
                rt["context_updated_at"] = now

        # A compact intent block, rare and only when something is actually close.
        items = [i for i in _items(state) if i.is_open()]
        near = [i for i in items if i.due is not None and -86400 <= (i.due - now) <= 3 * 86400]
        last_block = float(rt.get("last_intent_block_at") or 0)
        block_cool = float(_cfg(ctx, "intent_block_cooldown_minutes")) * 60
        if near and (now - last_block) >= block_cool and platform not in SKIP_PLATFORMS:
            body = E.summarize_items(near, now, limit=4)
            if body:
                block = P.INTENT_BLOCK_HEADER + "\n" + body + \
                        "\n（未到期的先别提前催；需要记新事项可用 proactive_intent 工具。）"
                S.update(lambda st: st["runtime"].__setitem__("last_intent_block_at", now))
    except Exception:
        # 用 error 而不是 warning：这里一炸，整个钩子静默失效——
        # pending 清不掉、活跃时间不刷新、账本一个字节都不落盘，
        # 而外部看起来一切正常。这个级别让它在默认日志里也现形。
        log.error("proactive: pre_llm_call bookkeeping failed", exc_info=True)

    return {"context": block} if block else None


def _on_session_end(**kw: Any) -> None:
    """Drop a pending probe that can no longer be answered in its session."""
    try:
        pending = S.load()["runtime"].get("pending")
        if pending and kw.get("session_key") and pending.get("session_key") == kw.get("session_key"):
            S.update(lambda st: st["runtime"].__setitem__("pending", None))
    except Exception:
        pass


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

INTENT_SCHEMA = {
    "name": "proactive_intent",
    "description": (
        "记录/管理主动提醒事项（截止时间、未结承诺、要盯的事）。"
        "proactive-heartbeat 插件据此在用户空闲时决定是否主动提醒。"
        "action=add 新增，list 查看，done 完成，drop 放弃，snooze 推迟。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list", "done", "drop", "snooze"]},
            "title": {"type": "string", "description": "事项描述，一句话，具体可执行"},
            "due": {"type": "string", "description": "截止时间：ISO、'2026-11-10 18:00'、或相对 '3d'/'12h'"},
            "kind": {"type": "string", "enum": ["deadline", "open_loop", "watch"]},
            "weight": {"type": "number", "description": "重要度 0..1，默认 0.55"},
            "note": {"type": "string", "description": "补充依据，最多一句"},
            "id": {"type": "string", "description": "done/drop/snooze 时指定事项 id，或前缀"},
            "amount": {"type": "string", "description": "snooze 的时长，如 '2h'/'3d'"},
        },
        "required": ["action"],
    },
}

STATUS_SCHEMA = {
    "name": "proactive_status",
    "description": "查看主动提醒插件的档位、预算、未结事项与最近决策日志。",
    "parameters": {"type": "object", "properties": {}},
}


def _find(items: List[E.Item], key: str) -> Optional[E.Item]:
    key = (key or "").strip()
    if not key:
        return None
    for it in items:
        if it.id == key:
            return it
    for it in items:
        if it.id.startswith(key):
            return it
    for it in items:
        if key and key in it.title:
            return it
    return None


def _handle_intent(action: str = "", title: str = "", due: str = "", kind: str = "open_loop",
                   weight: float = 0.55, note: str = "", id: str = "", amount: str = "", **_: Any) -> str:
    now = time.time()
    # 整段读-改-写包在一个事务里：与心跳的保存互斥。
    # 分开写（load → 改 → save）时，心跳若在中间保存，刚记的事项就被覆盖。
    with S.transaction() as state:
        items = _items(state)

        if action == "list":
            rows = [{"id": i.id, "title": i.title, "kind": i.kind,
                     "due": E.humanize_due(i.due, now) if i.due else None,
                     "weight": i.weight, "nudges": i.nudges, "status": i.status}
                    for i in items if i.is_open()]
            return json.dumps({"count": len(rows), "items": rows}, ensure_ascii=False)

        if action == "add":
            if not title.strip():
                return json.dumps({"ok": False, "error": "title 不能为空"}, ensure_ascii=False)
            item = E.Item(id=_new_id(title, now), title=title.strip(), kind=kind or "open_loop",
                          due=_parse_due(due, now), weight=max(0.0, min(1.0, float(weight))),
                          created_at=now, note=note.strip())
            items.append(item)
            _mutate_items(state, items)
            # 新事项是重新开口的理由：之前若因沉默降过档，这里复位
            state["runtime"]["last_gate_reason"] = None
            return json.dumps({"ok": True, "id": item.id,
                               "due": E.humanize_due(item.due, now) if item.due else None},
                              ensure_ascii=False)

        target = _find(items, id)
        if target is None:
            return json.dumps({"ok": False, "error": f"找不到事项 {id!r}"}, ensure_ascii=False)

        if action == "done":
            target.status = "done"
            # 回填长期层：词面能挂上就直接记账；挂不上先排队，交给 tick 里的语义归因。
            # 实测词面归因在这套数据上 0/12 命中——"交学生证"与"报名、材料、证件零遗漏"
            # 字面一个共同词都没有。只靠它，长期层永远是死的。
            gid = LT.attribute(target, state.get("goals") or [])
            if gid:
                LT.touch(state, gid, now)
            else:
                queue = state.setdefault("goal_attr_pending", [])
                if len(queue) < 50:
                    queue.append({"title": (target.title or "")[:80],
                                  "note": (target.note or "")[:300],
                                  "tags": list(target.tags or [])[:5], "at": now})
        elif action == "drop":
            target.status = "dropped"
        elif action == "snooze":
            delta = _parse_due(amount or "2h", now)
            shift = (delta - now) if delta else 7200
            target.last_nudged_at = now
            if target.due is not None:
                target.due += shift
            target.note = (target.note + f"；推迟 {int(shift // 60)} 分钟").strip("；")
        else:
            return json.dumps({"ok": False, "error": f"未知 action {action!r}"}, ensure_ascii=False)

        _mutate_items(state, items)
        return json.dumps({"ok": True, "id": target.id, "status": target.status}, ensure_ascii=False)


def _handle_status(**_: Any) -> str:
    now = time.time()
    ctx = _RUNTIME.get("ctx")
    mode = str(_cfg(ctx, "delivery_mode")) if ctx is not None else str(DEFAULTS["delivery_mode"])
    allowed = _allowed_platforms(ctx) if ctx is not None else list(DEFAULTS["allowed_platforms"])
    require_direct = (bool(_cfg(ctx, "require_direct")) if ctx is not None
                      else bool(DEFAULTS["require_direct"]))
    state = S.load()
    rt = state["runtime"]
    items = [i for i in _items(state) if i.is_open()]
    tz = _tz()
    tzname = getattr(tz, "key", None) or str(tz)
    local_now = datetime.fromtimestamp(now, tz).strftime("%Y-%m-%d %H:%M:%S")
    return json.dumps({
        "dial": state.get("dial"),
        "timezone": {"name": tzname, "now": local_now,
                     "utc_offset_hours": datetime.fromtimestamp(now, tz).utcoffset().total_seconds() / 3600},
        "adaptive_offset": state.get("adaptive_offset"),
        "ignore_streak": state.get("ignore_streak"),
        "type_stats": state.get("type_stats") or {},
        "today_sent": rt.get("daily_count"),
        "last_injection_at": rt.get("last_injection_at"),
        "last_session_key": rt.get("last_session_key"),
        "open_items": [{"id": i.id, "title": i.title,
                        "due": E.humanize_due(i.due, now) if i.due else None,
                        "nudges": i.nudges} for i in items],
        "delivery": {
            "mode": mode,
            "wanted": _want_platforms(mode),
            "allowed_platforms": allowed,
            "require_direct": require_direct,
            "sessions": {
                p: {"kind": PF.describe(v.get("key")), "at": v.get("at")}
                for p, v in (rt.get("sessions") or {}).items()
            },
            "last_platform": rt.get("last_platform"),
        },
        "stats": state.get("stats"),
        "recent": S.tail_decisions(6),
    }, ensure_ascii=False)


# --------------------------------------------------------------------------
# slash command
# --------------------------------------------------------------------------

_HELP = (
    "用法：/proactive status | off | quiet | normal | eager | snooze <id> [2h] | "
    "add <标题> [3d] | done <id> | purge [天数] | test"
)


def _purge(now: float, stale_days: float = 30.0) -> str:
    """清掉残留：逾期超过 N 天仍未完成的、以及已完成超过 7 天的。

    不做这件事的后果很实在——事项表只增不减，三个月后每一天都在拿
    几十条死事项去算分数，真正该说的那条被稀释掉。

    自己开事务，不接收外面传进来的 state：清理是就地删改整表，
    拿别人的快照写回去，等于把这期间新记的事项一起删了。
    """
    out = {"dropped": 0, "kept": 0}

    def _do(st: Dict[str, Any]) -> None:
        items = _items(st)
        keep, dropped = [], 0
        for it in items:
            if it.status == "done":
                if now - float(it.last_nudged_at or 0) > 7 * 86400:
                    dropped += 1
                    continue
            elif it.due is not None and (now - it.due) > stale_days * 86400:
                it.status = "dropped"
                dropped += 1
                continue
            keep.append(it)
        _mutate_items(st, keep)
        out["dropped"], out["kept"] = dropped, len(keep)

    _commit(_do)
    return f"清掉 {out['dropped']} 条残留，剩 {out['kept']} 条未结。"


def _handle_command(raw: str) -> str:
    parts = (raw or "").strip().split(None, 1)
    verb = (parts[0] if parts else "status").lower()
    rest = parts[1] if len(parts) > 1 else ""
    now = time.time()

    if verb in ("off", "quiet", "normal", "eager"):
        S.update(lambda st: st.__setitem__("dial", verb))
        return f"主动档位已设为 {verb}。" + ("（完全静默）" if verb == "off" else "")

    if verb == "test":
        state = S.load()
        items = [i for i in _items(state) if i.is_open()]
        if not items:
            return "没有未结事项。先用 /proactive add <标题> [3d] 或让我调用 proactive_intent 记一条。"
        # Force one tick past the gate (cooldown + idle + budget) so the operator can
        # hear the voice immediately instead of waiting out a 25-minute idle window.
        def _force(st):
            st["runtime"].update({
                "last_injection_at": None,
                "last_gate_reason": None,
                "last_user_turn_at": time.time() - 4 * 3600,
                "daily_count": 0,
            })
        S.update(_force)
        ctx = _RUNTIME.get("ctx")
        if ctx is None:
            return "心跳尚未启动（本进程没有加载插件上下文）。"
        try:
            _tick(ctx)
        except Exception as exc:
            return f"tick 失败：{exc}"
        decisions = S.tail_decisions(1)
        last = decisions[-1] if decisions else {}
        if last.get("event") == "sent":
            return f"已发出主动提醒：{last.get('item_title')}（显著性 {last.get('score')}）"
        return f"本次未发：{last.get('event')} / {last.get('reason')}"

    if verb == "snooze":
        bits = rest.split()
        if not bits:
            return _HELP
        return _handle_intent(action="snooze", id=bits[0], amount=bits[1] if len(bits) > 1 else "2h")

    if verb == "purge":
        state = _load_state()
        try:
            days = float(rest.strip() or 30)
        except ValueError:
            days = 30.0
        return _purge(time.time(), days)

    if verb == "add":
        bits = rest.split()
        if not bits:
            return _HELP
        due = bits[-1] if len(bits) > 1 and _parse_due(bits[-1], now) else ""
        title = rest[: -len(due)].strip() if due else rest.strip()
        return _handle_intent(action="add", title=title, due=due, kind="deadline" if due else "open_loop")

    if verb in ("done", "drop"):
        if not rest.strip():
            return _HELP
        return _handle_intent(action=verb, id=rest.strip())

    if verb != "status":
        return _HELP

    data = json.loads(_handle_status())
    tzinfo = data.get("timezone") or {}
    lines = [
        f"时区 {tzinfo.get('name')}（UTC{tzinfo.get('utc_offset_hours'):+.0f}）｜本地 {tzinfo.get('now')}",
        f"档位 {data['dial']}｜自适应偏移 {data['adaptive_offset']}｜今日已发 {data['today_sent']}｜连续被忽略 {data['ignore_streak']}",
        f"未结事项 {len(data['open_items'])} 条：",
    ]
    _ts = data.get("type_stats") or {}
    _parts = [f"{k} {v.get('engaged', 0)}/{v.get('sent', 0)}" for k, v in _ts.items() if v.get("sent")]
    if _parts:
        lines.insert(2, "类型响应：" + "、".join(_parts))
    _d = data.get("delivery") or {}
    _allow = "、".join(_d.get("allowed_platforms") or []) or "（空，一律不发）"
    _sess = "、".join(f"{p}={v.get('kind')}" for p, v in (_d.get("sessions") or {}).items()) or "无"
    lines.insert(2, f"投递 {_d.get('mode')}｜允许 {_allow}｜"
                    f"{'仅私聊' if _d.get('require_direct') else '含群聊'}｜会话 {_sess}")
    for it in data["open_items"][:6]:
        lines.append(f"  · {it['title']}（{it['due'] or '无期限'}，已提醒 {it['nudges']} 次）")
    if data["recent"]:
        last = data["recent"][-1]
        lines.append(f"最近一次决策：{last.get('event')} / {last.get('reason')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def register(ctx) -> None:
    """Called once by the plugin loader."""
    try:
        if not bool(_cfg(ctx, "enabled")):
            log.info("proactive-heartbeat: disabled by config")
            return
    except Exception as exc:
        # 配置读不到就按启用处理，但必须留痕——静默吞掉会让"装了却没反应"
        # 变成无法诊断的谜。
        log.warning("proactive-heartbeat: 配置读取失败，按启用继续：%s", exc)

    _RUNTIME["ctx"] = ctx
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_tool(name="proactive_intent", toolset="proactive", schema=INTENT_SCHEMA,
                      handler=_handle_intent, emoji="⏳",
                      description=INTENT_SCHEMA["description"])
    ctx.register_tool(name="proactive_status", toolset="proactive", schema=STATUS_SCHEMA,
                      handler=_handle_status, emoji="📋",
                      description=STATUS_SCHEMA["description"])
    ctx.register_command("proactive", _handle_command,
                         description="主动提醒：档位、事项、强制测试", args_hint="status|off|quiet|normal|eager|test")
    ctx.on_unload(_shutdown)
    _ensure_thread(ctx)


def _shutdown() -> None:
    with _RUNTIME["lock"]:
        stop = _RUNTIME.get("stop")
        if stop is not None:
            stop.set()
        _RUNTIME.update({"thread": None, "stop": None, "ctx": None, "tick_fn": None})
