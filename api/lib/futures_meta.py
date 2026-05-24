"""
선물(MES/ES) 메타 헬퍼 — 분기 만기/롤 윈도우/컨트랙트 코드.

CME quarterly ticker codes:
  H = March  | M = June | U = September | Z = December
"""
from datetime import datetime


_MES_MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}


def third_friday(year, month):
    """Day-of-month (int) of the 3rd Friday in (year, month)."""
    first = datetime(year, month, 1)
    days_to_first_fri = (4 - first.weekday()) % 7
    return 1 + days_to_first_fri + 14


def is_quarterly_roll_window(dt):
    """True during the 3 trading days before 3rd-Friday of Mar/Jun/Sep/Dec.

    ES/MES futures roll on the Thursday before third Friday. Volume splits
    between front and back month for a few sessions either side — wider
    spreads and erratic fills make new entries risky.
    """
    if dt.month not in _MES_MONTH_CODES:
        return False
    third_fri = third_friday(dt.year, dt.month)
    return (third_fri - 3) <= dt.day <= (third_fri - 1)


def next_quarterly_month(dt):
    """(year, month) of the next quarterly roll month from dt's perspective."""
    quarter_months = [3, 6, 9, 12]
    for m in quarter_months:
        if m > dt.month:
            return dt.year, m
        if m == dt.month and dt.day < third_friday(dt.year, m) - 3:
            return dt.year, m
    # Wrapped past December
    return dt.year + 1, 3


def days_to_next_roll(dt):
    """Calendar days until the next quarterly roll Wednesday (3 days before 3rd Fri)."""
    y, m = next_quarterly_month(dt)
    roll_day = third_friday(y, m) - 3   # Wednesday before 3rd Friday
    target = datetime(y, m, roll_day)
    return max(0, (target.date() - dt.date()).days)


def current_mes_contract(dt):
    """Active MES contract code (e.g. 'MESM26' for Jun 2026).

    Front-month is the quarterly contract whose 3rd-Friday expiry has not
    yet passed (or has just passed within the same week — we conservatively
    advance to the next contract on the Wednesday before expiry).
    """
    y, m = next_quarterly_month(dt)
    # If today is within the roll window, the front month is the *next* quarter
    if dt.month in _MES_MONTH_CODES and not is_quarterly_roll_window(dt):
        third_fri = third_friday(dt.year, dt.month)
        if dt.day < third_fri:
            y, m = dt.year, dt.month
    return f"MES{_MES_MONTH_CODES[m]}{y % 100:02d}"
