"""
LAYER 6 — Technical Entry Triggers
Layer 1~5가 모두 통과(스코어 60+)한 경우에만 실행.
이 레이어에서 실제 Call/Put 방향 결정.
"""
import pandas as pd


def _score_vwap_position(spy_price: float, vwap: float) -> tuple:
    """Price vs VWAP: above = Call bias, below = Put bias."""
    if spy_price is None or vwap is None:
        return 0, "NEUTRAL", "VWAP data unavailable"
    dist = spy_price - vwap
    pct = (dist / vwap) * 100 if vwap != 0 else 0

    if dist > 0:
        return 10, "CALL", f"Above VWAP by ${dist:+.2f} ({pct:+.2f}%)"
    else:
        return 10, "PUT", f"Below VWAP by ${dist:+.2f} ({pct:+.2f}%)"


def _score_vwap_bands(spy_price: float, vwap: float, spy_history: pd.DataFrame) -> tuple:
    """
    VWAP ± 1SD, ± 2SD band analysis.
    Overextension beyond 2SD → fading opportunity.
    """
    if spy_history is None or spy_history.empty or vwap is None or spy_price is None:
        return 0, "Band data unavailable"

    try:
        tp = (spy_history["High"] + spy_history["Low"] + spy_history["Close"]) / 3.0
        deviation = ((tp - vwap) ** 2).mean() ** 0.5
        upper_1sd = vwap + deviation
        lower_1sd = vwap - deviation
        upper_2sd = vwap + 2 * deviation
        lower_2sd = vwap - 2 * deviation

        if spy_price > upper_2sd:
            return -5, f"Over-extended above 2SD — fade risk"
        elif spy_price < lower_2sd:
            return -5, f"Over-extended below 2SD — fade risk"
        elif spy_price > upper_1sd:
            return 5, f"Above 1SD — momentum"
        elif spy_price < lower_1sd:
            return 5, f"Below 1SD — momentum"
        else:
            return 0, f"Within 1SD — neutral zone"
    except Exception:
        return 0, "Band calculation error"


def _score_volume(vol_ratio: float) -> tuple:
    """Volume confirmation: breakout/bounce needs 1.5x+ average volume."""
    if vol_ratio is None:
        return 0, "Volume data unavailable"
    if vol_ratio >= 2.0:
        return 10, f"Strong volume {vol_ratio:.2f}x — confirmed"
    elif vol_ratio >= 1.5:
        return 7, f"Adequate volume {vol_ratio:.2f}x"
    elif vol_ratio >= 1.0:
        return 3, f"Normal volume {vol_ratio:.2f}x"
    else:
        return 0, f"Weak volume {vol_ratio:.2f}x — caution"


def _score_range(range_value: float, atr: float = None) -> tuple:
    """
    Daily range adequacy check.
    If already moved most of ATR, overextension risk.
    """
    if range_value is None:
        return 0, "Range data unavailable"
    if atr and atr > 0:
        ratio = range_value / atr
        if ratio > 1.5:
            return 0, f"Range ${range_value:.2f} = {ratio:.1f}x ATR — overextended"
        elif ratio > 1.0:
            return 5, f"Range ${range_value:.2f} = {ratio:.1f}x ATR — extended"
        else:
            return 10, f"Range ${range_value:.2f} = {ratio:.1f}x ATR — room to move"
    else:
        if range_value >= 3.0:
            return 10, f"Range ${range_value:.2f} — adequate"
        elif range_value >= 2.0:
            return 5, f"Range ${range_value:.2f} — marginal"
        else:
            return 0, f"Range ${range_value:.2f} — too tight"


def _calculate_rsi(spy_history: pd.DataFrame, period: int = 14) -> float:
    """Calculate RSI from close prices."""
    if spy_history is None or len(spy_history) < period + 1:
        return None
    try:
        delta = spy_history["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        last = rsi.dropna()
        return float(last.iloc[-1]) if not last.empty else None
    except Exception:
        return None


def _score_momentum(rsi: float) -> tuple:
    """RSI momentum: >60 = Call bias, <40 = Put bias."""
    if rsi is None:
        return 0, "NEUTRAL", "RSI unavailable"
    if rsi >= 70:
        return 5, "CALL", f"RSI {rsi:.1f} — Overbought (caution)"
    elif rsi >= 60:
        return 10, "CALL", f"RSI {rsi:.1f} — Bullish momentum"
    elif rsi <= 30:
        return 5, "PUT", f"RSI {rsi:.1f} — Oversold (caution)"
    elif rsi <= 40:
        return 10, "PUT", f"RSI {rsi:.1f} — Bearish momentum"
    else:
        return 0, "NEUTRAL", f"RSI {rsi:.1f} — Neutral"


def calculate_technical_score(spy_price: float, vwap: float,
                              vol_ratio: float, range_value: float,
                              spy_history: pd.DataFrame = None) -> dict:
    """
    Calculate Layer 6 technical entry trigger score.

    Returns
    -------
    dict with keys:
        score          : int (max ~30)
        max            : int (30)
        direction_bias : str (CALL / PUT / NEUTRAL)
        details        : dict — component breakdown
        rsi            : float or None
    """
    details = {}
    bias_votes = {"CALL": 0, "PUT": 0, "NEUTRAL": 0}

    # VWAP position
    vwap_score, vwap_dir, vwap_detail = _score_vwap_position(spy_price, vwap)
    details["vwap_position"] = {"score": vwap_score, "detail": vwap_detail}
    bias_votes[vwap_dir] += 1

    # VWAP bands
    band_score, band_detail = _score_vwap_bands(spy_price, vwap, spy_history)
    details["vwap_bands"] = {"score": band_score, "detail": band_detail}

    # Volume
    vol_score, vol_detail = _score_volume(vol_ratio)
    details["volume"] = {"score": vol_score, "detail": vol_detail}

    # Range
    range_score, range_detail = _score_range(range_value)
    details["range"] = {"score": range_score, "detail": range_detail}

    # RSI Momentum
    rsi = _calculate_rsi(spy_history)
    rsi_score, rsi_dir, rsi_detail = _score_momentum(rsi)
    details["momentum"] = {"score": rsi_score, "detail": rsi_detail}
    bias_votes[rsi_dir] += 1

    # Total (cap at 30)
    total = min(30, vwap_score + band_score + vol_score + range_score + rsi_score)

    # Direction bias — majority vote
    if bias_votes["CALL"] > bias_votes["PUT"]:
        direction = "CALL"
    elif bias_votes["PUT"] > bias_votes["CALL"]:
        direction = "PUT"
    else:
        direction = "NEUTRAL"

    return {
        "score": total,
        "max": 30,
        "direction_bias": direction,
        "details": details,
        "rsi": rsi,
    }
