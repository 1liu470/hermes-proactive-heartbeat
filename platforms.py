"""投递目标判定：这条会话键是不是一对一对话。

Hermes 的网关键有固定结构：

    agent:<profile>:<platform>:<chat_type>:<chat_id>
    例：agent:main:qqbot:dm:<qq_id>
        agent:main:telegram:group:-1001234567

`chat_type` 取自各适配器（`dm` / `group` / `channel` / `thread` / `forum`），
是**协议层给的事实**，不是从字符串里猜出来的。所以判定分三级：

  1. 按结构解出 chat_type —— 最可靠
  2. 解不出来，看有没有 `:dm:` 一类标记 —— 兼容旧格式
  3. 都不成立 —— 返回 unknown

`unknown` 在 require_direct 为真时一律不发。宁可哑，不可错发。
本地会话（桌面）的键是纯会话号如 `20260101_000000_abcdef`，没有平台前缀，
必然落在 unknown —— 这正是我们想要的。
"""

from __future__ import annotations

from typing import Dict, Optional

#: 一对一的 chat_type。
DIRECT_TYPES = frozenset({"dm", "private", "p2p", "direct", "im"})

#: 有第三人在场的 chat_type。
GROUP_TYPES = frozenset({"group", "channel", "thread", "forum", "room", "guild", "supergroup"})

#: 结构化解析失败时的兜底标记。
_MARK_PREFIX = ":"


def parse(key: str) -> Optional[Dict[str, str]]:
    """把网关键拆成字段；不是网关键就返回 None。

    只认第一段是 `agent` 的五段以上键。少一段都不猜——猜错的方向是
    「把群聊当私聊」，代价比沉默大得多。
    """
    raw = str(key or "").strip()
    if not raw:
        return None
    parts = raw.split(":")
    if len(parts) < 5 or parts[0].lower() != "agent":
        return None
    return {
        "profile": parts[1],
        "platform": parts[2].lower(),
        "chat_type": parts[3].lower(),
        "chat_id": ":".join(parts[4:]),
    }


def classify(key: str) -> str:
    """返回 `direct` / `group` / `unknown`。"""
    info = parse(key)
    if info:
        ctype = info["chat_type"]
        if ctype in DIRECT_TYPES:
            return "direct"
        if ctype in GROUP_TYPES:
            return "group"
        return "unknown"

    low = str(key or "").strip().lower()
    if not low:
        return "unknown"
    if any(f"{_MARK_PREFIX}{m}{_MARK_PREFIX}" in low for m in DIRECT_TYPES):
        return "direct"
    if any(f"{_MARK_PREFIX}{m}{_MARK_PREFIX}" in low for m in GROUP_TYPES):
        return "group"
    return "unknown"


def platform_of(key: str) -> str:
    """键里带的平台名；本地会话（无前缀）返回空串。"""
    info = parse(key)
    return info["platform"] if info else ""


def describe(key: str) -> str:
    """给人看的一行：`qqbot/dm`、`desktop/unknown`。"""
    info = parse(key)
    if info:
        return f"{info['platform']}/{info['chat_type']}"
    kind = classify(key)
    return f"local/{kind}"


def is_allowed_target(key: str, platform: str, allowed: list, require_direct: bool) -> tuple:
    """返回 (能不能发, 拒因)。

    平台名以**配置里登记的那个**为准，不以键里的为准——键可以伪造，
    配置不能。两者不一致时按更严的那个来。
    """
    plat = str(platform or "").strip().lower()
    if not allowed:
        return False, "no-allowlist"
    if plat not in allowed:
        return False, "platform"
    key_plat = platform_of(key)
    if key_plat and plat and key_plat != plat:
        return False, "platform-mismatch"
    if require_direct:
        kind = classify(key)
        if kind != "direct":
            return False, f"not-direct:{kind}"
    return True, ""
