"""
LAYER 7 — Risk Management & Position Sizing (Dashboard / score_engine view)

This implements the rich UI risk layer (3-strike, daily DD, trade counts).
The *live paper bot* (v10_runner + v10_strategy) uses its own state machine +
vix_risk_pct + the 3-strike logic recorded in KV state.

See api/v10_constants.py and api/v10_runner.py for the production risk numbers.
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
    """Count entries executed today using the ledger-style trade_log (OPEN events)."""
    today = datetime.now(NY).strftime("%Y-%m-%d")
    log = portfolio.get("trade_log") or []
    count = sum(1 for e in log if e.get("event") == "OPEN" and e.get("date") == today)
    if count:
        return count
    # Fallback: scan history for today's entries (covers older records lacking trade_log)
    history = portfolio.get("history") or []
    return sum(1 for h in history if h.get("date") == today)


def _count_consecutive_losses(portfolio: dict) -> int:
    """Count consecutive losses from the most recent closed trades.

    history is maintained newest-first via insert(0, …) in api/data.py.
    Closed records carry pnl_locked=True or status="CLOSED".
    """
    history = portfolio.get("history") or []
    closed = [
        h for h in history
        if h.get("pnl_locked") or h.get("status") == "CLOSED" or h.get("pnl") is not None
    ]

    consecutive = 0
    for trade in closed:
        pnl = trade.get("pnl")
        if pnl is None:
            continue
        if pnl < 0:
            consecutive += 1
        else:
            break
    return consecutive


def _calculate_daily_drawdown(portfolio: dict) -> float:
    """Calculate today's drawdown as percentage of day-open equity.

    Anchor is daily_start_value (refreshed once per trading day by
    api/data.py); falls back to initial_balance if the anchor isn't
    set yet (first call ever on a new portfolio).
    """
    anchor = (portfolio.get("daily_start_value")
              or portfolio.get("initial_balance", 500000.0)
              or 500000.0)
    current = portfolio.get("current_value", anchor)
    if anchor <= 0:
        return 0.0
    return max(0.0, ((anchor - current) / anchor) * 100)


def _calculate_weekly_pnl(portfolio: dict) -> float:
    """Calculate this week's P&L as percentage."""
    # Simplified: use total return as proxy
    return portfolio.get("total_return_pct", 0.0)


def calculate_position_size(portfolio: dict, signal_grade: str, entry_price: float) -> dict:
    """
    Dynamic position sizing based on MES Futures Day Margin ($50 per contract)
    and 1.5% account risk rule.

    Returns
    -------
    dict with keys:
        max_shares      : int (represents contracts for futures)
        contracts       : int
        max_risk_amount : float
        sizing_reason   : str
    """
    cash = portfolio.get("cash", 0)
    initial = portfolio.get("initial_balance", 500000.0)
    
    # Dashboard sim only. Live bot uses vix-scaled risk from v10_constants (up to 2.5%).
    RISK_PCT = 0.015
    max_risk = initial * RISK_PCT

    if signal_grade == "STRONG":
        allocation = 1.0   # Full position
    elif signal_grade == "MODERATE":
        allocation = 0.5   # Half position
    else:
        allocation = 0.0   # No position

    risk_amount = max_risk * allocation
    
    # MES Day Margin is $50. We also assume expected stop-loss risk is ~$15 (3 points of MES)
    MES_DAY_MARGIN = 50.0
    expected_risk_per_contract = 15.0

    if risk_amount > 0:
        contracts_by_risk = int(risk_amount / expected_risk_per_contract)
        contracts_by_margin = int(cash / MES_DAY_MARGIN)
        contracts = max(1, min(contracts_by_risk, contracts_by_margin))
    else:
        contracts = 0

    return {
        "max_shares": contracts,  # For backward-compatibility with option shares
        "contracts": contracts,
        "max_risk_amount": round(risk_amount, 2),
        "allocation_pct": int(allocation * 100),
        "sizing_reason": f"{int(allocation * 100)}% allocation ({signal_grade} signal) - {contracts} MES Contracts",
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
        "max": 0,  # Risk is a pass/fail gate, not a score contributor (excluded from active_max sum)
        "lockout": lockout,
        "lockout_reason": lockout_reason,
        "details": details,
        "daily_trades": today_trades,
        "consecutive_losses": consecutive,
        "daily_drawdown": round(drawdown, 2),
        "strikes_remaining": strikes_remaining,
        "trades_remaining": trades_remaining,
    }
