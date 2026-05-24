"""
LAYER 5 — Time Window Filter
0DTE에서 시간이 전부. 어느 시간대에 진입하느냐가 승률을 30% 이상 바꿈.
"""
from datetime import datetime


# ── Time Window Map (EST 기준) ──────────────────────────────────────
# Each window: (start_min, end_min, score, label, emoji, description)
WINDOWS = [
    (570, 600,   0, "OPEN_CHAOS",  "❌", "Gap open chaos — avoid"),
    (600, 630,   5, "FORMING",     "⚠️", "Direction forming — caution"),
    (630, 690,  20, "PRIME",       "🟢", "Best window — max liquidity"),
    (690, 720,   8, "TRANSITION",  "⚠️", "Pre-lunch transition"),
    (720, 780,   0, "LUNCH_LULL",  "❌", "Lunch lull — avoid"),
    (780, 840,   8, "REENTRY",     "⚠️", "Afternoon re-entry possible"),
    (840, 885,  15, "GAMMA",       "🟡", "Gamma window — 2nd best"),
    (885, 960,   0, "GAMMA_BOMB",  "❌", "Gamma explosion — avoid"),
]

# ── Day-of-Week Bias ────────────────────────────────────────────────
DAY_BIAS = {
    0: {"label": "Monday",    "note": "Gap reversal common",        "adj": 0},
    1: {"label": "Tuesday",   "note": "Normal trend day",           "adj": 0},
    2: {"label": "Wednesday", "note": "Mid-week, watch Fed speak",  "adj": 0},
    3: {"label": "Thursday",  "note": "Pre-NFP positioning",        "adj": 0},
    4: {"label": "Friday",    "note": "0DTE expiry day — caution",  "adj": -5},
}


def calculate_time_score(now_et: datetime) -> dict:
    """
    Returns time window score (0–20) and metadata.

    Parameters
    ----------
    now_et : datetime
        Current time in ET (Eastern Time).

    Returns
    -------
    dict with keys:
        score      : int (0–20)
        max        : int (20)
        window     : str label
        emoji      : str
        description: str
        day_bias   : dict
        next_window: dict or None — countdown to next good window
        is_blocked : bool — True if current window is "AVOID"
    """
    h, m = now_et.hour, now_et.minute
    t_min = h * 60 + m
    weekday = now_et.weekday()

    # ── Find current window ─────────────────────────────────────
    score, label, emoji, desc = 0, "CLOSED", "⏸️", "Market closed"
    is_blocked = False

    for start, end, w_score, w_label, w_emoji, w_desc in WINDOWS:
        if start <= t_min < end:
            score, label, emoji, desc = w_score, w_label, w_emoji, w_desc
            if w_score == 0 and start >= 570:  # During market hours but avoid
                is_blocked = True
            break

    # ── Day-of-week adjustment ──────────────────────────────────
    day_info = DAY_BIAS.get(weekday, {"label": "Unknown", "note": "", "adj": 0})
    score = max(0, score + day_info["adj"])

    # ── Calculate next good window ──────────────────────────────
    next_window = None
    if score < 15:  # Not already in a prime/gamma window
        for start, end, w_score, w_label, w_emoji, w_desc in WINDOWS:
            if w_score >= 15 and start > t_min:
                mins_until = start - t_min
                next_window = {
                    "window": w_label,
                    "emoji": w_emoji,
                    "minutes_until": mins_until,
                    "countdown": f"{mins_until // 60}h {mins_until % 60}m",
                    "starts_at": f"{start // 60:02d}:{start % 60:02d}",
                }
                break

    return {
        "score": score,
        "max": 20,
        "window": label,
        "emoji": emoji,
        "description": desc,
        "day_bias": day_info,
        "next_window": next_window,
        "is_blocked": is_blocked,
        "current_time": now_et.strftime("%H:%M"),
    }
