"""
Central Score Engine — 7-Layer Orchestrator
승리 공식 = 레짐 감지 × 옵션 플로우 편향 × 최적 타이밍 × 명확한 진입 트리거 × 리스크 한도

Layer 1 (매크로)      : Gate 필터. Fail 즉시 중단.           [Future: API]
Layer 2 (레짐)        : 최대 40점                            [ACTIVE]
Layer 3 (옵션 플로우)  : 최대 30점                            [Future: Unusual Whales]
Layer 4 (상관관계)     : 최대 20점                            [ACTIVE]
Layer 5 (시간 창)     : 최대 20점                            [ACTIVE]
Layer 6 (진입 트리거)  : 최대 30점                            [ACTIVE]
Layer 7 (리스크 체크)  : Pass/Fail (Fail 시 lockout)          [ACTIVE]
──────────────────────────────────────────────────────────────
총점 최대             : 140점 (Layer 3 제외 현재: 110점)
정규화 기준           : 75점 이상 → 시그널 발화
"""
from datetime import datetime
import pandas as pd

from engines.regime import calculate_regime_score
from engines.correlation import calculate_correlation_score
from engines.time_window import calculate_time_score
from engines.technical import calculate_technical_score
from engines.risk_manager import check_risk_rules


# ── Signal Grade Thresholds (loaded from env for security / easy tuning) ──
import os
GRADE_STRONG   = int(os.getenv("GRADE_STRONG", "90"))
GRADE_MODERATE = int(os.getenv("GRADE_MODERATE", "75"))
GRADE_WEAK     = int(os.getenv("GRADE_WEAK", "60"))


def determine_signal_grade(total_score: int) -> dict:
    """
    Map total score to signal grade and trading action.

    Returns dict with: grade, label, emoji, action, color
    """
    if total_score >= GRADE_STRONG:
        return {
            "grade": "STRONG",
            "label": "STRONG SIGNAL",
            "emoji": "🟢",
            "action": "Full position entry",
            "color": "#3dd68c",
        }
    elif total_score >= GRADE_MODERATE:
        return {
            "grade": "MODERATE",
            "label": "MODERATE SIGNAL",
            "emoji": "🟡",
            "action": "Half position entry",
            "color": "#f5c451",
        }
    elif total_score >= GRADE_WEAK:
        return {
            "grade": "WEAK",
            "label": "STANDBY",
            "emoji": "🟠",
            "action": "Monitor — do not enter",
            "color": "#f5a623",
        }
    else:
        return {
            "grade": "NONE",
            "label": "NO SIGNAL",
            "emoji": "🔴",
            "action": "No entry — conditions insufficient",
            "color": "#f07178",
        }


def run_score_engine(now_et: datetime,
                     spy_price: float,
                     vix_price: float,
                     vix3m_price: float,
                     prev_close: float,
                     vwap: float,
                     vol_ratio: float,
                     range_value: float,
                     pcts: dict,
                     spy_history: pd.DataFrame,
                     portfolio: dict,
                     session_name: str) -> dict:
    """
    Run all scoring layers and produce final signal output.

    Parameters
    ----------
    now_et       : datetime in ET
    spy_price    : float — current SPY price
    vix_price    : float — current VIX
    vix3m_price  : float — current VIX3M
    prev_close   : float — SPY previous close
    vwap         : float — current VWAP
    vol_ratio    : float — current volume / 20-SMA volume
    range_value  : float — today's high-low range
    pcts         : dict — % changes for SPY, QQQ, IWM, DIA, etc.
    spy_history  : DataFrame — 5-min OHLC bars
    portfolio    : dict — paper portfolio state
    session_name : str — REGULAR, PRE-MARKET, AFTER-HOURS, etc.

    Returns
    -------
    dict — Complete score engine output for dashboard consumption
    """
    layers = {}
    is_market_open = session_name == "REGULAR"

    # ── LAYER 1: Macro Gate [Future Implementation] ─────────────
    # Placeholder — will integrate economic calendar API
    layers["macro_gate"] = {
        "score": 0,
        "max": 0,
        "status": "NOT_IMPLEMENTED",
        "detail": "Macro calendar not yet connected",
        "gate_passed": True,  # Default pass until implemented
    }

    # ── LAYER 2: Market Regime ──────────────────────────────────
    layers["regime"] = calculate_regime_score(
        vix_price=vix_price,
        vix3m_price=vix3m_price,
        spy_price=spy_price,
        prev_close=prev_close,
        spy_history=spy_history,
    )

    # ── LAYER 3: Options Flow [Future Implementation] ───────────
    # Placeholder — will integrate Unusual Whales API
    layers["options_flow"] = {
        "score": 0,
        "max": 30,
        "status": "NOT_IMPLEMENTED",
        "detail": "Options flow data not yet connected",
    }

    # ── LAYER 4: Correlation ────────────────────────────────────
    layers["correlation"] = calculate_correlation_score(pcts)

    # ── LAYER 5: Time Window ────────────────────────────────────
    layers["time_window"] = calculate_time_score(now_et)

    # ── LAYER 6: Technical Triggers ─────────────────────────────
    layers["technical"] = calculate_technical_score(
        spy_price=spy_price,
        vwap=vwap,
        vol_ratio=vol_ratio,
        range_value=range_value,
        spy_history=spy_history,
    )

    # ── LAYER 7: Risk Management ────────────────────────────────
    layers["risk"] = check_risk_rules(portfolio)

    # ── TOTAL SCORE CALCULATION ─────────────────────────────────
    # Sum layers 2 + 4 + 5 + 6 (Layer 1 = gate, Layer 3 = future, Layer 7 = lockout)
    active_scores = [
        layers["regime"]["score"],
        layers["correlation"]["score"],
        layers["time_window"]["score"],
        layers["technical"]["score"],
    ]
    total_score = sum(active_scores)

    # Current max (excluding Layer 3 which is not implemented)
    active_max = (
        layers["regime"]["max"] +
        layers["correlation"]["max"] +
        layers["time_window"]["max"] +
        layers["technical"]["max"]
    )  # = 40 + 20 + 20 + 30 = 110

    # Normalize score to 0-100 scale for signal grade
    normalized = int((total_score / active_max) * 100) if active_max > 0 else 0
    normalized = max(0, normalized)

    # ── SIGNAL GRADE ────────────────────────────────────────────
    signal = determine_signal_grade(normalized)

    # ── RISK OVERRIDE ───────────────────────────────────────────
    # If risk check fails, override signal to NO SIGNAL
    if not layers["risk"]["passed"]:
        signal = {
            "grade": "LOCKED",
            "label": layers["risk"]["lockout_reason"] or "RISK LOCKOUT",
            "emoji": "🔒",
            "action": "Trading locked — risk limit reached",
            "color": "#f07178",
        }
        normalized = 0

    # ── MARKET CLOSED OVERRIDE ──────────────────────────────────
    if not is_market_open:
        signal["label"] = f"MARKET {session_name}"
        signal["action"] = "Market not in session"

    # ── DIRECTION BIAS (Counter-Trend Inversion Applied) ──────────
    raw_bias = layers["technical"].get("direction_bias", "NEUTRAL")
    if raw_bias == "CALL":
        direction_bias = "PUT"
    elif raw_bias == "PUT":
        direction_bias = "CALL"
    else:
        direction_bias = "NEUTRAL"

    # ── BUILD FINAL OUTPUT ──────────────────────────────────────
    return {
        "total_score": normalized,
        "raw_score": total_score,
        "max_score": active_max,
        "signal": signal,
        "direction_bias": direction_bias,
        "layers": layers,
        "is_market_open": is_market_open,
    }
