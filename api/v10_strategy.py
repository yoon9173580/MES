"""
v10 SHARED decision engine — single source of truth for BOTH the backtest
(thorough_backtest_futures.py) and the live bot (api/v10_runner.py).

The backtest is the ground truth. To guarantee the live paper bot trades the
*same* strategy that produced the published Sharpe, both sides import the pure
functions below instead of each re-implementing the math.

Everything here operates on plain inputs (daily OHLC history + a morning bar
slice + VIX) so it has no dependency on how the data was sourced (CSV replay vs
live broker feed).

Price space: ES/MES index points (e.g. ~5300). The live bot multiplies SPY by
ES_PER_SPY before calling in, so SL/TP land on the same scale the backtest used.
"""
import os
import sys
from datetime import time as dtime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    from engines.regime import calculate_regime_score
    from engines.correlation import calculate_correlation_score
    from engines.time_window import calculate_time_score
    from engines.technical import calculate_technical_score
except ImportError:  # imported as api.v10_strategy from repo root
    from api.engines.regime import calculate_regime_score
    from api.engines.correlation import calculate_correlation_score
    from api.engines.time_window import calculate_time_score
    from api.engines.technical import calculate_technical_score


# ── v10 strategy constants (mirror thorough_backtest_futures.py v10 profile) ──
ATR_SL_MULT = 1.5            # SL = 1.5 × ATR
TP_MULT = 2.5               # TP = 2.5 × SL  (v10 lever)
ATR_MIN = 8.0               # skip days whose 14-day ATR < 8 points
MIN_SCORE = 65              # v10.6: ML hard-skip ON (SKIP_AFTER_N=30, THRESH=0.35) → 171 trades, Sharpe 2.32
VIX_THRESHOLD = 25.0        # below → trend-follow, at/above → mean-reversion
VIX_SHORT_FILTER = 20.0     # daily-bias SHORT filter: skip SHORT when VIX < this
VIX_BEAR_MIN = 25.0         # Option B min: ADX-triggered TREND_BEAR only when VIX≥25
VIX_CRISIS = 30.0           # Option A: crisis bear → override back to trend-follow
ADX_BEAR_TREND = 25.0       # Option B: trending bear (VIX_BEAR_MIN≤VIX<30 + ADX>25) → trend-follow
ADX_RUNAWAY = 40.0
# Option C: VIX-based position sizing
VIX_SIZE_25 = 25.0
VIX_SIZE_35 = 35.0
RISK_PCT_FULL = 0.025       # VIX < 25 (v10.3: 2.5% — targets ~31% annual)
RISK_PCT_BEAR = 0.010       # 25 ≤ VIX < 35
RISK_PCT_CRISIS = 0.007     # VIX ≥ 35
RSI_UPPER = 90.0
RSI_LOWER = 10.0
SECTOR_THRESHOLD = 1.8
NR7_SCORE_BOOST = 5
PULLBACK_SCORE_BOOST = 5
SL_MIN_PTS = 2.0
SL_CAP_PTS = 22.0           # v10.3: widened from 15 — fewer whipsaw stops, higher WR
ENTRY_TIME = dtime(10, 30)  # single PRIME entry
EXIT_TIME = dtime(15, 30)   # EOD flatten

# Intraday exit management (mirror thorough_backtest_futures.py)
TRAILING_ACTIVATION = 0.5   # arm trailing once profit ≥ 0.5×ATR
TRAILING_STEP = 0.25        # trail at best − 0.25×ATR
BREAKEVEN_AT = 0.25         # move stop to breakeven once profit ≥ 0.25×ATR
ES_SLIPPAGE_PTS = 0.25      # 1-tick slippage cushion at breakeven


def vix_risk_pct(vix):
    """Option C: scale position risk down in elevated-VIX regimes."""
    if vix >= VIX_SIZE_35:
        return RISK_PCT_CRISIS
    if vix >= VIX_SIZE_25:
        return RISK_PCT_BEAR
    return RISK_PCT_FULL


# ── Daily-history helpers (identical math to the backtest) ────────────────────
def calc_atr(daily_highs, daily_lows, daily_closes, period=14):
    """ATR(period) from the most-recent daily OHLC, newest entry LAST.

    Mirrors thorough_backtest_futures.calc_atr: True Range averaged over the
    last `period` days, needs ≥10 valid days else falls back to 4.0.
    """
    n = len(daily_closes)
    tr_list = []
    # iterate newest→older over the prior `period` days (skip today = last idx)
    for j in range(1, period + 1):
        i = n - 1 - j
        if i < 0:
            break
        prev_c = daily_closes[i - 1] if i - 1 >= 0 else daily_closes[i]
        tr = max(daily_highs[i] - daily_lows[i],
                 abs(daily_highs[i] - prev_c),
                 abs(daily_lows[i] - prev_c))
        tr_list.append(tr)
    return float(np.mean(tr_list)) if len(tr_list) >= 10 else 4.0


def check_nr7(daily_highs, daily_lows):
    """Today's range is the narrowest of the last 7 days (Crabel NR7).

    daily_* newest LAST; the final entry is *today*.
    """
    n = len(daily_highs)
    if n < 8:
        return False
    today_range = daily_highs[-1] - daily_lows[-1]
    prev_ranges = [daily_highs[-1 - j] - daily_lows[-1 - j] for j in range(1, 7)
                   if n - 1 - j >= 0]
    return len(prev_ranges) >= 6 and today_range < min(prev_ranges)


def check_3day_pullback(daily_closes):
    """3+ consecutive down closes immediately before today (newest LAST)."""
    n = len(daily_closes)
    if n < 5:
        return False
    consec = 0
    for j in range(1, 4):
        i = n - 1 - j
        if i - 1 < 0:
            break
        if daily_closes[i] < daily_closes[i - 1]:
            consec += 1
        else:
            break
    return consec >= 3


def check_daily_bias(daily_closes, day_open):
    """True (bullish) if today's open is above the prior 20-day SMA close."""
    n = len(daily_closes)
    if n < 21:
        return True
    prior20 = daily_closes[n - 21:n - 1]
    return day_open > float(np.mean(prior20)) if len(prior20) >= 20 else True


def synthetic_sector_pcts(spy_morning_ret):
    """Backtest derives QQQ/IWM/DIA from SPY's morning return (no real feed).

    Live MUST use the same synthetic values so correlation scoring matches.
    """
    return {
        "SPY": spy_morning_ret,
        "QQQ": spy_morning_ret * 1.2 if spy_morning_ret >= 0 else spy_morning_ret * 1.3,
        "IWM": spy_morning_ret * 0.9,
        "DIA": spy_morning_ret * 0.8,
    }


def stop_loss_points(atr_val, gamma_mult=1.0, sl_cap=SL_CAP_PTS):
    """SL distance in points: min(max(1.5×gamma_mult×ATR, 2), cap).

    PRIME window: gamma_mult=1.0, sl_cap=15. GAMMA (afternoon, v9): 0.75 / 10.
    """
    return min(max(ATR_SL_MULT * gamma_mult * atr_val, SL_MIN_PTS), sl_cap)


def init_position(trade_dir, entry_price, atr, sl_points, tp_mult=TP_MULT):
    """Build the exit-management state for a freshly opened position.

    Mirrors thorough_backtest_futures.py lines 564-571: fixed TP, initial SL,
    high-water mark, and the breakeven/trailing flags.
    """
    tp_points = sl_points * tp_mult
    if trade_dir == "LONG":
        tp_target = entry_price + tp_points
        sl_target = entry_price - sl_points
    else:
        tp_target = entry_price - tp_points
        sl_target = entry_price + sl_points
    return {
        "trade_dir": trade_dir, "entry_price": entry_price, "atr": atr,
        "sl_points": sl_points, "tp_target": tp_target, "sl_target": sl_target,
        "best_price": entry_price,
        "breakeven_activated": False, "trailing_activated": False,
    }


def manage_bar(pos, bar_high, bar_low):
    """Advance one bar of intraday management; mutate `pos` in place.

    Returns (exit_price, exit_type) if the position should close this bar, else
    (None, None). exit_type ∈ {TP, TRAIL, BE, SL}. Bit-for-bit identical to the
    backtest's minute loop (lines 583-636) so live trailing/BE matches.
    """
    td = pos["trade_dir"]
    entry = pos["entry_price"]
    atr = pos["atr"]

    if td == "LONG":
        if bar_high > pos["best_price"]:
            pos["best_price"] = bar_high
        profit = pos["best_price"] - entry
        # TP first (prioritise locking the gain)
        if bar_high >= pos["tp_target"]:
            return pos["tp_target"], "TP"
        if not pos["breakeven_activated"] and profit >= BREAKEVEN_AT * atr:
            pos["sl_target"] = entry + ES_SLIPPAGE_PTS
            pos["breakeven_activated"] = True
        if profit >= TRAILING_ACTIVATION * atr:
            tsl = pos["best_price"] - TRAILING_STEP * atr
            if tsl > pos["sl_target"]:
                pos["sl_target"] = tsl
                pos["trailing_activated"] = True
        if bar_low <= pos["sl_target"]:
            etype = ("TRAIL" if pos["trailing_activated"]
                     else "BE" if pos["breakeven_activated"] else "SL")
            return pos["sl_target"], etype
    else:  # SHORT
        if bar_low < pos["best_price"]:
            pos["best_price"] = bar_low
        profit = entry - pos["best_price"]
        if bar_low <= pos["tp_target"]:
            return pos["tp_target"], "TP"
        if not pos["breakeven_activated"] and profit >= BREAKEVEN_AT * atr:
            pos["sl_target"] = entry - ES_SLIPPAGE_PTS
            pos["breakeven_activated"] = True
        if profit >= TRAILING_ACTIVATION * atr:
            tsl = pos["best_price"] + TRAILING_STEP * atr
            if tsl < pos["sl_target"]:
                pos["sl_target"] = tsl
                pos["trailing_activated"] = True
        if bar_high >= pos["sl_target"]:
            etype = ("TRAIL" if pos["trailing_activated"]
                     else "BE" if pos["breakeven_activated"] else "SL")
            return pos["sl_target"], etype
    return None, None


def evaluate_entry(
    *,
    daily_highs, daily_lows, daily_closes,
    day_open, entry_price, entry_ts, vix_val,
    morning_df,
    min_score=MIN_SCORE, tp_mult=TP_MULT, atr_min=ATR_MIN,
    no_mean_reversion=False, gamma_mult=1.0, sl_cap=SL_CAP_PTS,
):
    """Pure v10 entry decision. Returns a dict; `enter` is the gate.

    Inputs are the same quantities the backtest's inner loop holds at 10:30:
      daily_*      : list of prior+today daily OHLC, newest LAST (today included)
      day_open     : today's session open price
      entry_price  : price at the 10:30 entry bar
      entry_ts     : tz-aware datetime of the entry bar (for time_window score)
      vix_val      : VIX level
      morning_df   : DataFrame of bars up to 10:30, columns capitalised
                     (Open/High/Low/Close/Volume), tz-aware index

    Mirrors thorough_backtest_futures.py lines ~460-653 exactly.
    """
    out = {"enter": False, "reasons": [], "atr": None, "score": None,
           "boosted_score": None, "boost_reasons": "", "direction": None,
           "sl_points": None, "tp_points": None, "strategy": None,
           "grade": "WEAK"}

    # [ATR] dynamic SL + dead-market floor
    atr_val = calc_atr(daily_highs, daily_lows, daily_closes)
    out["atr"] = atr_val
    if atr_min is not None and atr_val < atr_min:
        out["reasons"].append(f"atr {atr_val:.1f} < {atr_min}")
        return out

    window_sl = stop_loss_points(atr_val, gamma_mult, sl_cap)

    # Pre-score daily context
    is_nr7 = check_nr7(daily_highs, daily_lows)
    is_pullback = check_3day_pullback(daily_closes)
    daily_trend_long = check_daily_bias(daily_closes, day_open)

    # Morning metrics
    if morning_df is None or len(morning_df) < 5:
        out["reasons"].append("morning slice <5 bars")
        return out
    spy_morning_ret = ((entry_price / day_open) - 1.0) * 100
    pcts = synthetic_sector_pcts(spy_morning_ret)
    vol_sum = morning_df["Volume"].sum()
    vwap_morning = ((morning_df["High"] * morning_df["Volume"]).sum() / vol_sum
                    if vol_sum > 0 else entry_price)
    range_morning = float(morning_df["High"].max() - morning_df["Low"].min())
    avg_5 = morning_df["Volume"].tail(5).mean()
    avg_all = morning_df["Volume"].mean()
    vol_ratio = avg_5 / avg_all if avg_all > 0 else 1.0

    # Score engine — 4 layers, backtest normalization
    try:
        regime = calculate_regime_score(
            vix_price=vix_val, vix3m_price=vix_val * 1.08,
            spy_price=entry_price, prev_close=day_open, spy_history=morning_df)
        corr = calculate_correlation_score(pcts)
        time_win = calculate_time_score(entry_ts)
        tech = calculate_technical_score(entry_price, vwap_morning, vol_ratio,
                                         range_morning, morning_df)
        active_scores = [regime["score"], corr["score"], time_win["score"], tech["score"]]
        active_max = regime["max"] + corr["max"] + time_win["score"] + tech["max"]
        if active_max <= 0:
            active_max = 110
        normalized = int((sum(active_scores) / active_max) * 100)
        direction = tech.get("direction_bias", "NEUTRAL")
    except Exception as e:
        out["reasons"].append(f"score error {e}")
        return out

    out["score"] = normalized

    # Score boosting
    boosted = normalized
    boost_reasons = []
    if is_nr7:
        boosted += NR7_SCORE_BOOST
        boost_reasons.append("NR7")
    if is_pullback and direction == "CALL":
        boosted += PULLBACK_SCORE_BOOST
        boost_reasons.append("3DAY_PB")
    out["boosted_score"] = boosted
    out["boost_reasons"] = ",".join(boost_reasons)

    # Runaway trend veto — original symmetric check (pre-v10.2b)
    # Direction-aware veto interacted badly with MEAN_REVERSION (score direction ≠ trade direction).
    # Keep symmetric veto; TREND_BEAR SHORT exemption is handled by the strategy switch alone.
    adx_val = regime.get("details", {}).get("adx", {}).get("value")
    rsi_val = tech.get("rsi")
    s, q, i_ = pcts["SPY"], pcts["QQQ"], pcts["IWM"]

    is_runaway = False
    if adx_val is not None and adx_val >= ADX_RUNAWAY:
        is_runaway = True
    if rsi_val is not None and (rsi_val >= RSI_UPPER or rsi_val <= RSI_LOWER):
        is_runaway = True
    if (s > SECTOR_THRESHOLD and q > SECTOR_THRESHOLD and i_ > SECTOR_THRESHOLD) or \
       (s < -SECTOR_THRESHOLD and q < -SECTOR_THRESHOLD and i_ < -SECTOR_THRESHOLD):
        is_runaway = True

    out["grade"] = "STRONG" if boosted >= 88 else "MODERATE" if boosted >= min_score else "WEAK"
    if boosted < min_score:
        out["reasons"].append(f"score {boosted} < {min_score}")
        return out
    if direction not in ("CALL", "PUT", "LONG", "SHORT"):
        out["reasons"].append(f"direction {direction}")
        return out
    if is_runaway:
        out["reasons"].append("runaway veto")
        return out

    is_bull = direction in ("CALL", "LONG")
    is_bear = direction in ("PUT", "SHORT")

    # Daily-bias filter: skip SHORT in bullish daily trend (low VIX)
    if daily_trend_long and is_bear and vix_val < VIX_SHORT_FILTER:
        out["reasons"].append("daily-bias SHORT skip")
        return out

    # Adaptive strategy switch — Option A only (B removed: hurts bull-market MEAN_REV)
    # VIX < 20  : bull market → trend-follow
    # 20 ≤ VIX < 30: mean-reversion (unchanged from v10.1)
    # VIX ≥ 30  : crisis bear → trend-follow override (Option A)
    if no_mean_reversion or vix_val < VIX_THRESHOLD:
        is_trending = True
        bear_mode = False
    elif vix_val >= VIX_CRISIS:
        is_trending = True          # Option A: true crisis → follow the move
        bear_mode = True
    else:
        is_trending = False         # moderate stress → mean-reversion (v10.1 behaviour)
        bear_mode = False

    if is_trending:
        trade_dir = "LONG" if is_bull else "SHORT"
        strategy = "TREND_BEAR" if bear_mode else "TREND_FOLLOW"
    else:
        trade_dir = "SHORT" if is_bull else "LONG"
        strategy = "MEAN_REVERSION"

    out.update({
        "enter": True,
        "direction": trade_dir,
        "strategy": strategy,
        "bear_mode": bear_mode,
        "sl_points": round(window_sl, 4),
        "tp_points": round(window_sl * tp_mult, 4),
    })
    return out
