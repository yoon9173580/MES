"""
LAYER 7 — Risk Management & Position Sizing
이기는 시그널보다 크게 잃지 않는 구조가 장기 생존의 핵심.
"""
from datetime import datetime
import pytz

NY = pytz.timezone("America/New_York")

# ── Risk Parameters ─────────────────────────────────────────────────
MAX_TRADE_LOSS_PCT    = 2.0    # 1 trade max loss: 2% of account
MAX_DAILY_LOSS_PCT    = 6.0    # Daily max loss: 6% of account
MAX_WEEKLY_LOSS_PCT   = 10.0   # Weekly max loss: 10% of account
CONSECUTIVE_LOSS_LOCK = 3      # 3 consecutive losses → lockout
MAX_DAILY_TRADES      = 3      # Max entries per day
MIN_RR_RATIO          = 2.0    # Minimum risk-reward ratio
AUTO_STOP_LOSS_PCT    = 50.0   # Auto stop-loss at -50% premium
AUTO_TAKE_PROFIT_PCT  = 100.0  # First take-profit at +100% premium


def _count_today_trades(portfolio: dict) -> int:
    """Count trades executed today."""
    today = datetime.now(NY).strftime("%Y-%m-%d")
    history = portfolio.get("history", [])
    return sum(
        1 for h in history
        if (h.get("date") == today or h.get("time", "").startswith(today))
        and h.get("action") == "BUY"
    )


def _count_consecutive_losses(portfolio: dict) -> int:
    """Count consecutive losses from most recent trades."""
    history = portfolio.get("history", [])
    sell_trades = [h for h in history if h.get("action") == "SELL"]

    consecutive = 0
    for trade in sell_trades:  # history is newest-first (insert(0, ...))
        pnl = trade.get("pnl", trade.get("revenue", 0) - trade.get("cost", 0))
        if pnl < 0:
            consecutive += 1
        else:
            break
    return consecutive


def _calculate_daily_drawdown(portfolio: dict) -> float:
    """Calculate today's drawdown as percentage of initial balance."""
    initial = portfolio.get("initial_balance", 2000.0)
    current = portfolio.get("current_value", initial)
    if initial <= 0:
        return 0.0
    return max(0.0, ((initial - current) / initial) * 100)


def _calculate_weekly_pnl(portfolio: dict) -> float:
    """Calculate this week's P&L as percentage."""
    # Simplified: use total return as proxy
    return portfolio.get("total_return_pct", 0.0)


def calculate_position_size(portfolio: dict, signal_grade: str, entry_price: float) -> dict:
    """
    Dynamic position sizing based on account balance and signal strength.

    Returns
    -------
    dict with keys:
        max_shares      : int
        max_risk_amount : float
        sizing_reason   : str
    """
    cash = portfolio.get("cash", 0)
    initial = portfolio.get("initial_balance", 2000.0)
    max_risk = initial * (MAX_TRADE_LOSS_PCT / 100)

    if signal_grade == "STRONG":
        allocation = 1.0   # Full position
    elif signal_grade == "MODERATE":
        allocation = 0.5   # Half position
    else:
        allocation = 0.0   # No position

    risk_amount = max_risk * allocation
    max_shares = int(min(cash, risk_amount) // entry_price) if entry_price > 0 else 0

    return {
        "max_shares": max_shares,
        "max_risk_amount": round(risk_amount, 2),
        "allocation_pct": int(allocation * 100),
        "sizing_reason": f"{int(allocation * 100)}% allocation ({signal_grade} signal)",
    }


def check_risk_rules(portfolio: dict) -> dict:
    """
    Layer 7: Comprehensive risk management check.

    Returns
    -------
    dict with keys:
        passed           : bool — True if all risk checks pass
        score            : int (0 or negative)
        lockout          : bool — True if trading should be blocked
        lockout_reason   : str or None
        details          : dict — individual check results
        daily_trades     : int
        consecutive_losses: int
        daily_drawdown   : float
        strikes_remaining: int
        trades_remaining : int
    """
    details = {}
    lockout = False
    lockout_reason = None
    score = 0

    # ── 3-Strike Rule ───────────────────────────────────────────
    consecutive = _count_consecutive_losses(portfolio)
    strikes_remaining = max(0, CONSECUTIVE_LOSS_LOCK - consecutive)

    if consecutive >= CONSECUTIVE_LOSS_LOCK:
        lockout = True
        lockout_reason = f"🔒 3-STRIKE LOCKOUT ({consecutive} consecutive losses)"
        score = -100
    details["three_strike"] = {
        "passed": consecutive < CONSECUTIVE_LOSS_LOCK,
        "detail": f"{consecutive}/{CONSECUTIVE_LOSS_LOCK} strikes",
        "consecutive_losses": consecutive,
    }

    # ── Daily Drawdown Limit ────────────────────────────────────
    drawdown = _calculate_daily_drawdown(portfolio)
    if drawdown >= MAX_DAILY_LOSS_PCT:
        lockout = True
        lockout_reason = f"🔒 DAILY LOSS LIMIT ({drawdown:.1f}% > {MAX_DAILY_LOSS_PCT}%)"
        score = -100
    details["daily_drawdown"] = {
        "passed": drawdown < MAX_DAILY_LOSS_PCT,
        "detail": f"{drawdown:.1f}% / {MAX_DAILY_LOSS_PCT}% limit",
        "current_pct": drawdown,
    }

    # ── Daily Trade Count ───────────────────────────────────────
    today_trades = _count_today_trades(portfolio)
    trades_remaining = max(0, MAX_DAILY_TRADES - today_trades)

    if today_trades >= MAX_DAILY_TRADES:
        lockout = True
        lockout_reason = f"🔒 MAX TRADES REACHED ({today_trades}/{MAX_DAILY_TRADES})"
        score = min(score, -50)
    details["daily_trades"] = {
        "passed": today_trades < MAX_DAILY_TRADES,
        "detail": f"{today_trades}/{MAX_DAILY_TRADES} trades today",
    }

    # ── Weekly Loss Limit ───────────────────────────────────────
    weekly_pnl = _calculate_weekly_pnl(portfolio)
    if weekly_pnl <= -MAX_WEEKLY_LOSS_PCT:
        lockout = True
        lockout_reason = f"🔒 WEEKLY LOSS LIMIT ({weekly_pnl:.1f}%)"
        score = -100
    details["weekly_limit"] = {
        "passed": weekly_pnl > -MAX_WEEKLY_LOSS_PCT,
        "detail": f"Weekly P&L: {weekly_pnl:+.1f}%",
    }

    return {
        "passed": not lockout,
        "score": score,
        "lockout": lockout,
        "lockout_reason": lockout_reason,
        "details": details,
        "daily_trades": today_trades,
        "consecutive_losses": consecutive,
        "daily_drawdown": round(drawdown, 2),
        "strikes_remaining": strikes_remaining,
        "trades_remaining": trades_remaining,
    }
