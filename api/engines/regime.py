"""
LAYER 2 — Market Regime Detection
오늘 시장이 어떤 성격인지 분류. 레짐에 따라 전략 방향이 달라짐.
"""
import pandas as pd


# ── Regime Labels ───────────────────────────────────────────────────
REGIME_TRENDING  = "TRENDING"
REGIME_CHOPPY    = "CHOPPY"
REGIME_BREAKOUT  = "BREAKOUT"
REGIME_UNKNOWN   = "UNKNOWN"

REGIME_STRATEGIES = {
    REGIME_TRENDING:  "추세 방향 진입, VWAP 리젝션 후 재진입",
    REGIME_CHOPPY:    "극단값 페이딩, 작은 목표, 빠른 청산",
    REGIME_BREAKOUT:  "돌파 확인 후 재테스트 대기 진입",
    REGIME_UNKNOWN:   "레짐 불명 — 관망 권장",
}


def _score_vix(vix_price: float) -> tuple:
    """Score VIX absolute level. Returns (score, detail_str)."""
    if vix_price is None:
        return 0, "VIX unavailable"
    if 14 <= vix_price <= 20:
        return 15, f"VIX {vix_price:.1f} — Normal range"
    elif 20 < vix_price <= 30:
        return 0, f"VIX {vix_price:.1f} — Elevated caution"
    elif vix_price > 30:
        return -20, f"VIX {vix_price:.1f} — FEAR regime"
    else:
        return -5, f"VIX {vix_price:.1f} — Below threshold"


def _score_vix_term_structure(vix_price: float, vix3m_price: float) -> tuple:
    """
    VIX vs VIX3M spread.
    Contango (VIX < VIX3M) = normal → +10
    Backwardation (VIX > VIX3M) = fear → -15
    """
    if vix_price is None or vix3m_price is None:
        return 0, "Term structure unavailable"
    spread = vix_price - vix3m_price
    if spread < 0:
        return 10, f"Contango ({spread:+.2f}) — Normal"
    elif spread == 0:
        return 0, f"Flat term structure"
    else:
        return -15, f"Backwardation ({spread:+.2f}) — FEAR"


def _score_gap(spy_price: float, prev_close: float) -> tuple:
    """Gap analysis: Gap Up/Down > 0.5% signals directional bias."""
    if spy_price is None or prev_close is None or prev_close == 0:
        return 0, "Gap data unavailable"
    gap_pct = ((spy_price / prev_close) - 1.0) * 100
    if gap_pct > 0.5:
        return 5, f"Gap Up {gap_pct:+.2f}% — Bullish bias"
    elif gap_pct < -0.5:
        return 5, f"Gap Down {gap_pct:+.2f}% — Bearish bias"
    else:
        return 0, f"Flat open ({gap_pct:+.2f}%)"


def _calculate_adx(spy_history: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate ADX from 5-minute OHLC data.
    Returns ADX value or None if insufficient data.
    """
    if spy_history is None or len(spy_history) < period + 1:
        return None

    try:
        high = spy_history["High"]
        low = spy_history["Low"]
        close = spy_history["Close"]

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        adx = dx.rolling(window=period).mean()

        last_adx = adx.dropna()
        return float(last_adx.iloc[-1]) if not last_adx.empty else None
    except Exception:
        return None


def _score_adx(adx_value: float) -> tuple:
    """Score ADX: ≥25 = trending, <20 = choppy."""
    if adx_value is None:
        return 0, "ADX unavailable"
    if adx_value >= 25:
        return 15, f"ADX {adx_value:.1f} — Strong trend"
    elif adx_value >= 20:
        return 5, f"ADX {adx_value:.1f} — Weak trend"
    else:
        return 0, f"ADX {adx_value:.1f} — Choppy/Ranging"


def _classify_regime(adx_value, vix_price, range_value, opening_range_broken=False):
    """Classify market regime based on indicators."""
    if opening_range_broken and adx_value and adx_value >= 25:
        return REGIME_BREAKOUT
    if adx_value and adx_value >= 25:
        return REGIME_TRENDING
    if adx_value and adx_value < 20:
        return REGIME_CHOPPY
    if vix_price and vix_price > 25:
        return REGIME_CHOPPY
    return REGIME_UNKNOWN


def calculate_regime_score(vix_price: float, vix3m_price: float,
                           spy_price: float, prev_close: float,
                           spy_history: pd.DataFrame = None) -> dict:
    """
    Calculate Layer 2 regime score.

    Parameters
    ----------
    vix_price   : float — Current VIX
    vix3m_price : float — Current VIX3M (3-month VIX)
    spy_price   : float — Current SPY price
    prev_close  : float — SPY previous close
    spy_history : DataFrame — 5-min OHLC for ADX calculation

    Returns
    -------
    dict with keys:
        score      : int (can be negative, max ~40)
        max        : int (40)
        regime     : str (TRENDING/CHOPPY/BREAKOUT/UNKNOWN)
        strategy   : str — Recommended approach for this regime
        details    : dict — Individual component scores
        vix_spread : float or None
    """
    details = {}

    # VIX absolute level
    vix_score, vix_detail = _score_vix(vix_price)
    details["vix"] = {"score": vix_score, "detail": vix_detail}

    # VIX term structure
    term_score, term_detail = _score_vix_term_structure(vix_price, vix3m_price)
    details["vix_term"] = {"score": term_score, "detail": term_detail}

    # Gap analysis
    gap_score, gap_detail = _score_gap(spy_price, prev_close)
    details["gap"] = {"score": gap_score, "detail": gap_detail}

    # ADX
    adx_value = _calculate_adx(spy_history)
    adx_score, adx_detail = _score_adx(adx_value)
    details["adx"] = {"score": adx_score, "detail": adx_detail, "value": adx_value}

    # Total regime score
    total = vix_score + term_score + gap_score + adx_score

    # Regime classification
    regime = _classify_regime(adx_value, vix_price, None)
    strategy = REGIME_STRATEGIES.get(regime, "")

    # VIX spread for display
    vix_spread = None
    if vix_price is not None and vix3m_price is not None:
        vix_spread = round(vix_price - vix3m_price, 2)

    return {
        "score": total,
        "max": 40,
        "regime": regime,
        "strategy": strategy,
        "details": details,
        "vix_spread": vix_spread,
    }
