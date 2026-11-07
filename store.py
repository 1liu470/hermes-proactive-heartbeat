"""Atomic JSON state store for proactive-heartbeat.

State lives outside the plugin directory ($HERMES_HOME/state/proactive-heartbeat/)
so reinstalling or upgrading the plugin never destroys the user's intent list.
Reads tolerate corruption (return defaults, keep a .broken copy); writes are
temp-file + os.replace so a crash mid-write cannot truncate the ledger.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from contextlib import contextmanager
import time
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.RLock()

DEFAULTS: Dict[str, Any] = {
    "version": 1,
    "dial": "normal",
    "adaptive_offset": 0.0,
    "ignore_streak": 0,
    "items": [],
    "runtime": {
        "daily_key": "",
        "daily_count": 0,
        "last_injection_at": None,
        "last_user_turn_at": None,
        "last_session_key": "",
        "last_platform": "",
        "context_terms": [],
        "context_updated_at": None,
        "pending": None,
    },
    "stats": {"considered": 0, "gated": 0, "declined_by_model": 0, "sent": 0},
}


def hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    try:
        from hermes_cli.main import get_hermes_home  # type: ignore
        return Path(get_hermes_home())
    except Exception:
        pass
    for cand in (Path.home() / ".hermes", Path.home() / "AppData" / "Local" / "hermes"):
        if cand.exists():
            return cand
    return Path.home() / ".hermes"


def state_dir() -> Path:
    d = hermes_home() / "state" / "proactive-heartbeat"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path() -> Path:
    return state_dir() / "state.json"


def decisions_path() -> Path:
    return state_dir() / "decisions.jsonl"


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load() -> Dict[str, Any]:
    with _LOCK:
        path = state_path()
        if not path.exists():
            return json.loads(json.dumps(DEFAULTS))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("state root is not an object")
            return _deep_merge(json.loads(json.dumps(DEFAULTS)), data)
        except Exception:
            try:
                shutil.copy2(path, path.with_suffix(f".broken-{int(time.time())}.json"))
            except Exception:
                pass
            return json.loads(json.dumps(DEFAULTS))


def save(state: Dict[str, Any]) -> None:
    with _LOCK:
        path = state_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)


def update(mutator) -> Dict[str, Any]:
    """Load-modify-save under one lock. *mutator* receives the dict in place."""
    with _LOCK:
        state = load()
        mutator(state)
        save(state)
        return state


@contextmanager
def transaction():
    """Load-modify-save 一体化，横跨整个 with 块持锁。

    为什么需要：工具调用原来是 load() → 改 → save()，两步之间锁已释放。
    心跳若在这几百毫秒里保存（它也在写同一个文件），用户刚记的事项就被覆盖——
    这是"我加了，回头就没了"的典型成因。包成事务后两者互斥。
    """
    with _LOCK:
        state = load()
        yield state
        save(state)


def acquire_tick_lock(ttl_seconds: float = 180.0) -> Optional[int]:
    """Cross-process, cross-thread tick mutex via an exclusive-create lock file.

    Why a file and not a threading.Lock: a hot plugin reload leaves the old
    module's heartbeat thread alive while the new one starts, and a gateway
    restart can briefly overlap two processes.  Both share only the filesystem,
    and both would otherwise read "not sent yet", call the model, and deliver
    twice (observed: two `sent` rows 0.59s apart).  O_EXCL is atomic across
    processes; the TTL reclaims a lock left behind by a crash.
    """
    path = state_dir() / "tick.lock"
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > ttl_seconds:
                    path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            return None
        except OSError:
            return None
        try:
            os.write(fd, f"{os.getpid()} {time.time():.0f}".encode())
        except OSError:
            pass
        return fd
    return None


def release_tick_lock(fd: Optional[int]) -> None:
    path = state_dir() / "tick.lock"
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def log_decision(record: Dict[str, Any]) -> None:
    with _LOCK:
        try:
            with decisions_path().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass


def tail_decisions(limit: int = 10) -> list:
    path = decisions_path()
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except Exception:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out
