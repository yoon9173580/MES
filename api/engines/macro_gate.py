"""
LAYER 1 — Macro Event Gate
중요 매크로 이벤트(FOMC/CPI/NFP/PPI) 발표 전후 진입 차단.

Vercel serverless에서는 외부 캘린더 API 호출 비용/지연을 피하기 위해
규칙 기반 + 알려진 FOMC 날짜를 임베드해서 사용한다.
"""
from datetime import datetime, timedelta, time as dtime
import pytz

NY = pytz.timezone("America/New_York")

# ── FOMC 결정일 (회의 2일차, 발표 시각 14:00 ET) ─────────────
# 8 meetings/year, 회의 후 발표 시각 = 14:00 ET (Powell 회견 14:30~15:30)
# Fed가 매년 7월말~8월 다음해 일정 발표. 2026년 일정은 Fed 공식.
FOMC_DATES = [
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
    # 2027 (잠정 — Fed가 다음 일정 발표 시 업데이트)
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16",
    "2027-07-28", "2027-09-22", "2027-11-03", "2027-12-15",
]

# 이벤트별 발표 시각 (모두 ET)
EVENT_TIMES = {
    "FOMC": dtime(14, 0),   # FOMC statement 14:00 ET
    "CPI":  dtime(8, 30),   # BLS 8:30 ET
    "NFP":  dtime(8, 30),   # BLS first Friday 8:30 ET
    "PPI":  dtime(8, 30),   # BLS 8:30 ET (CPI 하루 전)
    "FOMC_MINUTES": dtime(14, 0),  # 3주 후 minutes
}

# 블록 윈도우 (이벤트 전후 분 단위)
BLOCK_WINDOWS = {
    "FOMC": (180, 180),     # FOMC: 3h before / 3h after (Powell 회견 포함)
    "CPI":  (60,  120),     # CPI: 1h before / 2h after
    "NFP":  (60,  120),     # NFP: 1h before / 2h after
    "PPI":  (30,   60),     # PPI: 30m before / 1h after
    "FOMC_MINUTES": (30, 60),
}


def _first_friday(year: int, month: int) -> int:
    """월의 첫 금요일 일자."""
    first = datetime(year, month, 1)
    days_to_first_fri = (4 - first.weekday()) % 7
    return 1 + days_to_first_fri


def _nfp_release_dates(year: int) -> list:
    """매월 첫 금요일 = NFP 발표일."""
    return [datetime(year, m, _first_friday(year, m)).date() for m in range(1, 13)]


def _cpi_release_dates(year: int) -> list:
    """CPI는 보통 10~15일 사이 화/수.
    정확한 날짜는 매월 BLS 일정에 의존하나, 보수적으로 10~16일 평일을 모두 윈도우로 본다.
    """
    dates = []
    for m in range(1, 13):
        # 평균적으로 두 번째 주 화요일 또는 수요일
        # 보수적으로 10~16일 사이 영업일을 모두 잠재 후보로
        for day in range(10, 17):
            try:
                d = datetime(year, m, day).date()
                if d.weekday() in (1, 2, 3):  # Tue/Wed/Thu
                    dates.append(d)
                    break  # 한 달에 한 번만
            except ValueError:
                continue
    return dates


def _ppi_release_dates(year: int) -> list:
    """PPI는 보통 CPI 하루 전(또는 다음날). 9~13일 평일 후보."""
    dates = []
    for m in range(1, 13):
        for day in range(9, 14):
            try:
                d = datetime(year, m, day).date()
                if d.weekday() in (0, 1, 2):  # Mon/Tue/Wed
                    dates.append(d)
                    break
            except ValueError:
                continue
    return dates


def _build_calendar(year: int) -> list:
    """주어진 연도의 매크로 이벤트 캘린더 — 정렬된 리스트."""
    events = []
    # FOMC (hardcoded)
    for d in FOMC_DATES:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d").date()
            if dt.year == year:
                events.append({"date": dt, "kind": "FOMC", "time": EVENT_TIMES["FOMC"]})
        except ValueError:
            continue
    # NFP
    for d in _nfp_release_dates(year):
        events.append({"date": d, "kind": "NFP", "time": EVENT_TIMES["NFP"]})
    # CPI
    for d in _cpi_release_dates(year):
        events.append({"date": d, "kind": "CPI", "time": EVENT_TIMES["CPI"]})
    # PPI
    for d in _ppi_release_dates(year):
        events.append({"date": d, "kind": "PPI", "time": EVENT_TIMES["PPI"]})
    return sorted(events, key=lambda e: (e["date"], e["time"]))


def _event_datetime(event: dict) -> datetime:
    """이벤트 일자+시각을 NY tz datetime으로 변환."""
    d = event["date"]
    t = event["time"]
    naive = datetime(d.year, d.month, d.day, t.hour, t.minute)
    return NY.localize(naive)


def _is_blocked(now: datetime, event: dict) -> tuple:
    """이벤트의 블록 윈도우 안에 now가 있으면 (True, minutes_to_event) 반환."""
    pre, post = BLOCK_WINDOWS.get(event["kind"], (60, 60))
    event_dt = _event_datetime(event)
    delta_min = (event_dt - now).total_seconds() / 60.0
    if -post <= delta_min <= pre:
        return True, delta_min
    return False, delta_min


def calculate_macro_gate(now_et: datetime = None) -> dict:
    """
    Layer 1 결과를 반환.

    Returns
    -------
    dict with keys:
        score        : int  — 0 (informational; Layer 1은 gate 역할)
        max          : int  — 0
        gate_passed  : bool — False면 진입 차단
        status       : str  — BLOCKED / CLEAR / WARNING
        active_event : str or None — 현재 블로킹 중인 이벤트명
        next_event   : dict or None — 다음 매크로 이벤트 메타
        detail       : str  — 사용자에게 표시할 메시지
    """
    if now_et is None:
        now_et = datetime.now(NY)
    elif now_et.tzinfo is None:
        now_et = NY.localize(now_et)
    elif now_et.tzinfo != NY:
        now_et = now_et.astimezone(NY)

    # 올해 + 내년 캘린더 (연말에 다음해 이벤트도 봐야 함)
    cal = _build_calendar(now_et.year) + _build_calendar(now_et.year + 1)

    # 1) 현재 블로킹 중인 이벤트가 있는지
    for ev in cal:
        blocked, delta = _is_blocked(now_et, ev)
        if blocked:
            sign = "+" if delta < 0 else "-"
            mins = int(abs(delta))
            phase = "지난" if delta < 0 else "남은"
            return {
                "score": 0,
                "max": 0,
                "gate_passed": False,
                "status": "BLOCKED",
                "active_event": ev["kind"],
                "next_event": {
                    "kind": ev["kind"],
                    "date": ev["date"].strftime("%Y-%m-%d"),
                    "time": ev["time"].strftime("%H:%M"),
                    "minutes_offset": int(delta),
                },
                "detail": f"🚫 {ev['kind']} 발표 윈도우 — {phase} {mins}분 ({ev['date'].strftime('%m/%d')} {ev['time'].strftime('%H:%M')} ET)",
            }

    # 2) 향후 24시간 내 이벤트가 있으면 경고
    soon = None
    for ev in cal:
        event_dt = _event_datetime(ev)
        delta_min = (event_dt - now_et).total_seconds() / 60.0
        if 0 < delta_min <= 1440:  # 24h
            soon = (ev, delta_min)
            break

    if soon:
        ev, delta_min = soon
        hours = delta_min / 60.0
        return {
            "score": 0,
            "max": 0,
            "gate_passed": True,
            "status": "WARNING",
            "active_event": None,
            "next_event": {
                "kind": ev["kind"],
                "date": ev["date"].strftime("%Y-%m-%d"),
                "time": ev["time"].strftime("%H:%M"),
                "minutes_until": int(delta_min),
            },
            "detail": f"⚠️ {ev['kind']} 발표까지 {hours:.1f}h ({ev['date'].strftime('%m/%d')} {ev['time'].strftime('%H:%M')} ET)",
        }

    # 3) Clear
    return {
        "score": 0,
        "max": 0,
        "gate_passed": True,
        "status": "CLEAR",
        "active_event": None,
        "next_event": None,
        "detail": "✓ Macro calendar clear (24h)",
    }
