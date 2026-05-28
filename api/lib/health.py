"""
경량 인메모리 로그 버퍼 + health 스냅샷.

목적:
  Vercel 로그는 retention이 짧고(24h), 디스크 영속이 어려움. 인메모리
  ring buffer로 최근 N개 에러/경고만 보관하면 운영 중에 /api/health로
  현재 상태와 최근 이슈를 한 번에 확인 가능.

용법:
  from lib.health import log_error, log_warn, snapshot
  log_error("vix_fetch", "Yahoo 403")
  log_warn("options_flow", "Polygon rate limit")
  data = snapshot()  # 최근 N개 + 카운터
"""
import time
from collections import deque
from datetime import datetime
import pytz

NY = pytz.timezone("America/New_York")

MAX_LOG_ENTRIES = 100
_logs = deque(maxlen=MAX_LOG_ENTRIES)
_counters = {"error": 0, "warn": 0, "info": 0}


def _push(level, source, message, extra=None):
    entry = {
        "ts": datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": int(time.time()),
        "level": level,
        "source": source,
        "message": str(message)[:500],
    }
    if extra:
        entry["extra"] = extra
    _logs.append(entry)
    _counters[level] = _counters.get(level, 0) + 1


def log_error(source, message, extra=None):
    _push("error", source, message, extra)


def log_warn(source, message, extra=None):
    _push("warn", source, message, extra)


def log_info(source, message, extra=None):
    _push("info", source, message, extra)


def snapshot(limit=50):
    """Return recent log entries + counters for the health endpoint."""
    entries = list(_logs)[-limit:]
    return {
        "process_uptime_sec": int(time.time() - _process_start),
        "counters": dict(_counters),
        "recent": entries,
        "buffer_size": len(_logs),
        "buffer_max": MAX_LOG_ENTRIES,
    }


_process_start = time.time()
