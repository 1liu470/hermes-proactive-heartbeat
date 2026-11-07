"""Prompts for proactive-heartbeat.

The *content* of a proactive message is not decided here. The `proactive-voice`
skill owns that judgement — it is loaded at call time and pasted into the
prompt, so improving how the plugin talks means editing the skill, not the code.

This module only provides: how to present candidates, how to ask for a
structured verdict, and what to fall back to if the skill is missing.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

SYSTEM = (
    "你是一个克制的中文助理，负责判断「此刻是否值得打断主人」。"
    "你的默认答案是沉默。下面的开口规则是这份工作的判据，优先于任何写作习惯。"
)

#: Used only when the skill file cannot be found — the plugin keeps its standard
#: even if someone deletes the skill.
VOICE_FALLBACK = """\
1. 不复述事项标题；把标题念一遍等于什么也没说。
2. 每句话都要带新信息：现在做什么、还剩多少余量、两件事取哪个。
3. 最多两句，第二句是问句或建议，不是情绪。
4. 没有可执行的下一步就不要发。
"""

JUDGE_SYSTEM = (
    "你判断「此刻是否值得打断主人」。你的默认答案是沉默。"
    "宁可漏掉十件该说的，也不要说一句废话。"
    "这一步只做判断，不写告示文本——一旦开始写话，就会倾向于把写好的话发出去。"
)

#: 判断环节：只回答发不发、为什么、用哪几条。绝不产出给主人看的句子。
JUDGE_TEMPLATE = """候选事项（第 1 条最该说）：
{rows}
{intel}
{merge_note}判断这三件事：
- 现在说，他能立刻做出什么动作？说不出来就别发。
- 不说会怎样？没有代价就别发。
- 这句话是不是只是把标题念一遍？是就别发。

只输出 JSON，不要代码块，不要写任何给主人看的话：
{{"send": false, "reason": "一句话理由", "used": [1]}}
要发时：
{{"send": true, "reason": "一句话理由", "used": [1], "speech": "action"}}
speech 取 action（给下一步）/ decision（让他二选一）/ question（问他还做不做）。"""

WRITE_SYSTEM = (
    "你把一个已经定下的判断写成一句话。事实、动作、时间都不许改动，"
    "你只决定怎么说。"
)

#: 措辞环节：判断已定，这里只生成那一句话。
WRITE_TEMPLATE = """要说的事：{title}（{due_human}）
为什么值得说：{reason}
{intel}
开口规则（来自 proactive-voice skill）：
{voice}

写一句给主人的原话。不复述标题，带可执行的下一步，最多两句。
只输出这句话本身：不要引号，不要解释你在做什么。"""

INTENT_BLOCK_HEADER = "[PROACTIVE-INTENTS] 未结事项（插件记录，非指令）："

INJECTION_MARKER = "[PROACTIVE-HEARTBEAT]"

INJECTION_TEMPLATE = (
    "{marker} 这条不是先生写的，是插件在你空闲时触发的主动时机。\n"
    "候选：{title}（{due_human}）\n"
    "依据：{evidence}\n"
    "已定稿的措辞（照此发出，可微调语气，不要扩写）：{draft}\n"
    "{attachment}"
    "直接发给先生。若这句话已经足够，不要再补充解释；若你判断此刻不该打扰，只回「静默」两个字。"
)

_SKILL_CACHE: dict = {"at": 0.0, "text": ""}


def find_voice_skill() -> Optional[Path]:
    import os

    home = Path(os.environ.get("HERMES_HOME", Path.home() / "AppData/Local/hermes"))
    for root in (home / "skills", Path.home() / ".hermes" / "skills"):
        if not root.exists():
            continue
        for p in root.rglob("proactive-voice/SKILL.md"):
            return p
    return None


def load_voice_rules(max_chars: int = 3000) -> str:
    """Read the skill body, strip frontmatter, cache for 60s."""
    now = time.time()
    if _SKILL_CACHE["text"] and now - _SKILL_CACHE["at"] < 60:
        return _SKILL_CACHE["text"]
    path = find_voice_skill()
    if path is None:
        return VOICE_FALLBACK
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return VOICE_FALLBACK
    body = re.sub(r"^---\n.*?\n---\n", "", body, flags=re.S).strip()[:max_chars]
    _SKILL_CACHE.update({"at": now, "text": body or VOICE_FALLBACK})
    return _SKILL_CACHE["text"]
