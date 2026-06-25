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
from engines.ml_weights import get_ml_multipliers
from engines.macro_gate import calculate_macro_gate
from engines.options_flow import calculate_options_flow_score


# ── Signal Grade Thresholds (Dashboard UI only) ──
# These control the pretty labels (STRONG / MODERATE) shown in the rich dashboard.
# The *hard entry gate* for the live paper bot is MIN_SCORE from v10_constants (default 65).
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

    # ── LAYER 1: Macro Gate (FOMC/CPI/NFP/PPI) ───────────────────
    layers["macro_gate"] = calculate_macro_gate(now_et)

    # ── LAYER 2: Market Regime ──────────────────────────────────
    layers["regime"] = calculate_regime_score(
        vix_price=vix_price,
        vix3m_price=vix3m_price,
        spy_price=spy_price,
        prev_close=prev_close,
        spy_history=spy_history,
    )

    # ── LAYER 3: Options Flow (Yahoo SPY chain) ──────────────────
    layers["options_flow"] = calculate_options_flow_score("SPY")

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
    # ── APPLY ML ADAPTIVE WEIGHTS ───────────────────────────────
    # ML multipliers range 0.5~1.5. Only apply to positive scores so a
    # poor regime doesn't get amplified into a deeper negative, and cap
    # the result at the layer's nominal max so total never exceeds the
    # denominator used for normalization.
    ml = get_ml_multipliers()
    for layer_key in ("regime", "correlation", "technical"):
        s = layers[layer_key]["score"]
        m = layers[layer_key]["max"]
        if s > 0:
            layers[layer_key]["score"] = min(m, int(s * ml.get(layer_key, 1.0)))

    # Sum layers 2 + 4 + 5 + 6 (Layer 1 = gate, Layer 3 = future, Layer 7 = lockout)
    # Options flow only counts when data is live (NO_DATA → skip).
    of_live = layers["options_flow"].get("status") == "LIVE"
    active_scores = [
        layers["regime"]["score"],
        layers["correlation"]["score"],
        layers["time_window"]["score"],
        layers["technical"]["score"],
    ]
    if of_live:
        active_scores.append(layers["options_flow"]["score"])
    total_score = sum(active_scores)

    # Max is computed dynamically — when options flow data isn't available,
    # the score is normalized against only the layers that voted.
    active_max = (
        layers["regime"]["max"] +
        layers["correlation"]["max"] +
        layers["time_window"]["max"] +
        layers["technical"]["max"]
    )  # = 50 + 20 + 20 + 30 = 120 (regime max is 50, not 40 as old comment said)
    if of_live:
        active_max += layers["options_flow"]["max"]  # +30 → 150

    # Normalize score to 0-100 scale for signal grade
    normalized = int((total_score / active_max) * 100) if active_max > 0 else 0
    normalized = max(0, normalized)

    # ── SIGNAL GRADE ────────────────────────────────────────────
    signal = determine_signal_grade(normalized)

    # ── REGIME-AWARE STRATEGY FLAG ──────────────────────────────
    # Trending regime (low VIX): follow momentum signals.
    # Counter-trend regime (high VIX): trade RSI/band extremes only.
    is_trending = vix_price is not None and vix_price < 22.0

    # ── RUNAWAY TREND VETO FILTER ───────────────────────────────
    # RSI extremes are the *entry signal* in counter-trend mode, so the
    # RSI veto only applies when trending. ADX/sector vetoes still apply
    # — fading a fully-established 35+ ADX move is dangerous either way.
    is_runaway_trend = False
    runaway_reason = ""

    # 1. ADX Runaway check (both modes)
    # Use the same default as v10 production gate when possible.
    try:
        from ..v10_constants import ADX_RUNAWAY as _ADX_RUNAWAY
    except Exception:
        _ADX_RUNAWAY = 40.0
    adx_val = layers["regime"].get("details", {}).get("adx", {}).get("value")
    if adx_val is not None and adx_val >= _ADX_RUNAWAY:
        is_runaway_trend = True
        runaway_reason = f"Extreme ADX ({adx_val:.1f} >= {_ADX_RUNAWAY})"

    # 2. RSI Runaway check — trending mode only
    rsi_val = layers["technical"].get("rsi")
    if is_trending and rsi_val is not None and (rsi_val >= 80.0 or rsi_val <= 20.0):
        is_runaway_trend = True
        runaway_reason = f"Extreme RSI ({rsi_val:.1f})"

    # 3. Synchronized Sector Breakout — VIX-adaptive threshold
    spy_ret = pcts.get("SPY", 0.0)
    qqq_ret = pcts.get("QQQ", 0.0)
    iwm_ret = pcts.get("IWM", 0.0)
    vix_ref = vix_price if vix_price and vix_price > 0 else 18.0
    if vix_ref < 15:
        sector_thresh = 0.8
    elif vix_ref < 22:
        sector_thresh = 1.2
    elif vix_ref < 30:
        sector_thresh = 1.8
    else:
        sector_thresh = 2.5
    if ((spy_ret > sector_thresh and qqq_ret > sector_thresh and iwm_ret > sector_thresh)
        or (spy_ret < -sector_thresh and qqq_ret < -sector_thresh and iwm_ret < -sector_thresh)):
        is_runaway_trend = True
        runaway_reason = (f"Synchronized Sector Breakout ({sector_thresh:.1f}% threshold) "
                          f"SPY:{spy_ret:+.2f}%, QQQ:{qqq_ret:+.2f}%, IWM:{iwm_ret:+.2f}%")

    # Apply Veto
    if is_runaway_trend:
        signal = {
            "grade": "LOCKED",
            "label": "RUNAWAY VETO",
            "emoji": "⚠️",
            "action": f"Vetoed: {runaway_reason} — Runway trend danger",
            "color": "#f07178",
        }
        normalized = 0

    # ── MACRO GATE OVERRIDE ─────────────────────────────────────
    # FOMC/CPI/NFP/PPI 발표 윈도우 진입 시 신호 봉쇄.
    if not layers["macro_gate"]["gate_passed"]:
        signal = {
            "grade": "LOCKED",
            "label": f"MACRO {layers['macro_gate']['active_event']}",
            "emoji": "🚫",
            "action": layers["macro_gate"]["detail"],
            "color": "#f07178",
        }
        normalized = 0

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

    # ── DIRECTION BIAS (Regime-Aware Strategy Switching) ────────
    raw_bias = layers["technical"].get("direction_bias", "NEUTRAL")
    of_dir = layers["options_flow"].get("direction", "NEUTRAL") if of_live else "NEUTRAL"

    if is_trending:
        # Trend following: technical layer drives. Options flow is a
        # *confirmation gate* — if it actively disagrees with technical
        # (LONG vs SHORT), de-confidence to NEUTRAL. Agreement or
        # NEUTRAL options flow leaves technical's call intact.
        if of_live and raw_bias in ("LONG", "SHORT") and of_dir in ("LONG", "SHORT") and of_dir != raw_bias:
            direction_bias = "NEUTRAL"
        else:
            direction_bias = raw_bias
    else:
        # Mean reversion: ignore momentum (raw_bias) and require an actual
        # extreme — naive inversion of trend signals has no statistical edge.
        # Bias only when RSI is far from neutral OR price is outside the
        # VWAP 2-SD band (over-extension fade).
        rsi_val_mr = layers["technical"].get("rsi")
        band_region = layers["technical"].get("details", {}).get("vwap_bands", {}).get("region", "NEUTRAL")
        if rsi_val_mr is not None and rsi_val_mr <= 35:
            direction_bias = "LONG"   # oversold bounce
        elif rsi_val_mr is not None and rsi_val_mr >= 65:
            direction_bias = "SHORT"  # overbought fade
        elif band_region == "LONG_FADE":
            direction_bias = "LONG"   # below 2SD — mean revert up
        elif band_region == "SHORT_FADE":
            direction_bias = "SHORT"  # above 2SD — mean revert down
        else:
            direction_bias = "NEUTRAL"
        # In counter-trend mode, options flow disagreement also disqualifies.
        if of_live and direction_bias in ("LONG", "SHORT") and of_dir in ("LONG", "SHORT") and of_dir != direction_bias:
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
