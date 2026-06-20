#!/usr/bin/env python3
"""
thorough_backtest_v11.py
========================
MES Futures Day-Trading Backtest — v11.0
ML Scoring via Walk-Forward LightGBM (22 features, OR breakout, ATR exits)

Architecture:
- Hard filters: 10:30 PRIME bar only, ATR>=8, max 1 trade/day, 3-strike lockout
- ML gate: WalkForwardMLv11 predicts P(win) from 22 features
  * Warmup: first 25 trades entered unconditionally (collect training data)
  * After warmup: skip if P(win) < 0.52
- Direction: OR breakout (9:30-10:00 range) then momentum fallback
- Exit: ATR-based SL/TP, breakeven, trailing stop, EOD flatten at 15:30

Run:
    python3 thorough_backtest_v11.py \
        --csv MES_1min_data_et_rth.csv \
        --start 2023-03-27 \
        --out backtest_v11.json
"""

import argparse
import json
import math
import os
import sys
import warnings
from collections import deque
from datetime import datetime, date, time as dtime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
PNL_PER_POINT = 5.0          # $5 per MES point per contract
START_BALANCE = 10_000.0
SLIPPAGE_PTS  = 0.25         # per side, in MES points
COMMISSION    = 0.62         # $ per contract per side (round-trip = 1.24)
SL_CAP        = 22.0         # max SL in MES points (tunable via --sl-cap)
RISK_PCT      = 0.020        # base risk per trade (tunable via --risk-pct)

PRIME_TIME    = dtime(10, 30)
OR_START      = dtime(9, 30)
OR_END        = dtime(10, 0)
EOD_EXIT      = dtime(15, 30)

ATR_MIN       = 8.0
ATR_PERIOD    = 14
ATR_NORM_REF  = 20.0        # "normal" ATR reference for normalisation

FEATURE_NAMES = [
    "mom_5m", "mom_15m", "mom_30m", "mom_day",
    "or_breakout", "or_position", "or_width_norm",
    "gap_norm",
    "atr_norm", "atr_expand",
    "vol_ratio",
    "vwap_dist",
    "rsi_norm", "ema_dist",
    "prev_range", "prev_close_pos",
    "direction", "atr_regime",
    "dow_norm", "month_norm",
    "recent_wr5", "streak_norm",
]


# ── Walk-Forward ML ──────────────────────────────────────────────────────────
class WalkForwardMLv11:
    """
    Walk-forward LightGBM classifier that predicts P(win) for each trade signal.

    Training data accumulates from completed trades.  The model is refit every
    RETRAIN_EVERY completed trades once MIN_TRAIN examples are available.
    Before WARMUP_TRADES have been entered, no ML gate is applied.
    """

    MIN_TRAIN      = 15
    WARMUP_TRADES  = 25
    RETRAIN_EVERY  = 5
    ENTRY_THRESH   = 0.52

    SIZE_HIGH = 0.62
    SIZE_MED  = 0.55
    SIZE_LOW  = 0.44

    def __init__(self, entry_thresh: float = 0.52, warmup: int = 25):
        self.entry_thresh = entry_thresh
        self.WARMUP_TRADES = warmup

        # Try LightGBM, fall back to LogisticRegression
        try:
            import lightgbm as lgb
            self._use_lgbm = True
            self._model = lgb.LGBMClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.08,
                num_leaves=7,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                reg_alpha=0.5,
                class_weight="balanced",
                verbose=-1,
                random_state=42,
            )
        except ImportError:
            from sklearn.linear_model import LogisticRegression
            self._use_lgbm = False
            self._model = LogisticRegression(C=0.2, max_iter=500, random_state=42)

        self._fitted = False
        self._X: List[List[float]] = []   # feature rows
        self._y: List[int] = []           # labels: 1=win, 0=loss

        self.trades_entered   = 0
        self.trades_completed = 0
        self._last_retrain_at = 0          # trades_completed count at last fit

        # For ML stat tracking
        self.ml_filtered = 0
        self.ml_passed   = 0

        # Rolling win-rate / streak state (used to build features)
        self._results: deque = deque(maxlen=5)  # 1=win, 0=loss, recent first
        self._consec_wins  = 0
        self._consec_losses = 0

        # Feature importances (filled after fitting)
        self.feature_importances: Dict[str, float] = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def ml_active(self) -> bool:
        """True once warmup is complete and a model has been fitted."""
        return self.trades_entered >= self.WARMUP_TRADES and self._fitted

    def predict_proba(self, features: List[float]) -> float:
        """Return P(win).  If ML not yet active, returns 0.5 (neutral)."""
        if not self._fitted:
            return 0.5
        x = np.array(features, dtype=float).reshape(1, -1)
        try:
            return float(self._model.predict_proba(x)[0, 1])
        except Exception:
            return 0.5

    def should_enter(self, features: List[float]) -> Tuple[bool, float]:
        """
        Returns (enter: bool, p_win: float).
        During warmup always returns True.
        After warmup, skip if p_win < entry_thresh.
        """
        p = self.predict_proba(features)
        if not self.ml_active():
            return True, p
        if p >= self.entry_thresh:
            self.ml_passed += 1
            return True, p
        else:
            self.ml_filtered += 1
            return False, p

    def size_multiplier(self, p_win: float) -> float:
        """Contract multiplier based on ML confidence."""
        if not self.ml_active():
            return 1.0
        if p_win >= self.SIZE_HIGH:
            return 1.3
        if p_win >= self.SIZE_MED:
            return 1.15
        if p_win < self.SIZE_LOW:
            return 0.8
        return 1.0

    def record_entry(self, features: List[float]) -> None:
        """Called when a trade is entered (store features for labelling later)."""
        self.trades_entered += 1
        self._pending_features = features  # will be labelled on record_exit

    def record_exit(self, win: bool) -> None:
        """Called when a trade completes; adds labelled example and may retrain."""
        label = 1 if win else 0
        if hasattr(self, "_pending_features") and self._pending_features is not None:
            self._X.append(list(self._pending_features))
            self._y.append(label)
            self._pending_features = None

        # Update streak / recent win-rate state
        self._results.appendleft(label)
        if win:
            self._consec_wins  += 1
            self._consec_losses = 0
        else:
            self._consec_losses += 1
            self._consec_wins   = 0

        self.trades_completed += 1

        # Retrain if enough data and due
        if (len(self._y) >= self.MIN_TRAIN and
                self.trades_completed - self._last_retrain_at >= self.RETRAIN_EVERY):
            self._fit()

    def rolling_features(self) -> Tuple[float, float]:
        """Return (recent_wr5, streak_norm) for feature building."""
        wr5 = float(np.mean(list(self._results))) if self._results else 0.5
        streak = np.clip(
            (self._consec_wins - self._consec_losses) / 5.0, -1.0, 1.0
        )
        return wr5, float(streak)

    # ── Private ───────────────────────────────────────────────────────────────

    def _fit(self) -> None:
        X = np.array(self._X, dtype=float)
        y = np.array(self._y, dtype=int)

        # Need at least one positive and one negative example
        if len(np.unique(y)) < 2:
            return

        try:
            self._model.fit(X, y)
            self._fitted = True
            self._last_retrain_at = self.trades_completed

            # Extract feature importances
            if self._use_lgbm:
                imp = self._model.feature_importances_
                total = float(imp.sum()) or 1.0
                self.feature_importances = {
                    FEATURE_NAMES[i]: float(imp[i]) / total
                    for i in range(min(len(FEATURE_NAMES), len(imp)))
                }
            else:
                coef = np.abs(self._model.coef_[0])
                total = float(coef.sum()) or 1.0
                self.feature_importances = {
                    FEATURE_NAMES[i]: float(coef[i]) / total
                    for i in range(min(len(FEATURE_NAMES), len(coef)))
                }
        except Exception as e:
            pass  # keep previous model if refit fails


# ── Feature helpers ──────────────────────────────────────────────────────────

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute RSI from a 1-D array of closes. Returns value in [0,100]."""
    if len(closes) < period + 1:
        return 50.0
    diffs = np.diff(closes[-period - 1:])
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _ema(values: np.ndarray, span: int) -> float:
    """Return final EMA value using pandas ewm for correctness."""
    if len(values) == 0:
        return float(values[-1]) if len(values) else 0.0
    k = 2.0 / (span + 1)
    ema = float(values[0])
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


# ── Preprocessing: build daily & intra-day lookup tables ───────────────────

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    From raw 1-min OHLCV build:
      - daily_df : date-indexed DataFrame with daily OHLCV and ATR14
      - day_bars : dict mapping date -> DataFrame of 1-min bars for that session

    Returns (daily_df, day_bars_dict)
    """
    df = df.copy()
    df["date"] = df.index.date

    # ── Daily OHLCV ──────────────────────────────────────────────────────────
    daily = (
        df.groupby("date")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .sort_index()
    )

    # ── True Range and ATR14 ─────────────────────────────────────────────────
    prev_close = daily["close"].shift(1)
    daily["tr"] = np.maximum(
        daily["high"] - daily["low"],
        np.maximum(
            (daily["high"] - prev_close).abs(),
            (daily["low"] - prev_close).abs(),
        ),
    )
    daily["atr14"] = daily["tr"].rolling(ATR_PERIOD, min_periods=1).mean()

    # ── Day bars dict ─────────────────────────────────────────────────────────
    day_bars: Dict[date, pd.DataFrame] = {}
    for d, grp in df.groupby("date"):
        day_bars[d] = grp.sort_index()

    return daily, day_bars


# ── Per-day feature computation ──────────────────────────────────────────────

def compute_features(
    trade_date: date,
    bars_today: pd.DataFrame,
    daily_df: pd.DataFrame,
    ml: WalkForwardMLv11,
    vol_history: deque,          # deque of vol_at_1030 values (last 20 days)
) -> Optional[Dict]:
    """
    Compute the 22 features for a trade at 10:30 ET on trade_date.
    Returns dict of feature values, or None if data insufficient.
    """
    # ── ATR14 ────────────────────────────────────────────────────────────────
    if trade_date not in daily_df.index:
        return None
    atr14 = float(daily_df.loc[trade_date, "atr14"])
    if atr14 <= 0:
        return None

    # ── Previous day stats ────────────────────────────────────────────────────
    dates_so_far = daily_df.index[daily_df.index <= trade_date].tolist()
    if len(dates_so_far) < 2:
        return None
    prev_date = dates_so_far[-2]
    prev_row  = daily_df.loc[prev_date]
    prev_close = float(prev_row["close"])
    prev_high  = float(prev_row["high"])
    prev_low   = float(prev_row["low"])
    prev_range_abs = prev_high - prev_low

    # ── Day open ─────────────────────────────────────────────────────────────
    day_open = float(daily_df.loc[trade_date, "open"])

    # ── Opening Range: 9:30–10:00 ────────────────────────────────────────────
    or_mask = bars_today.index.time < OR_END
    or_bars = bars_today[or_mask]
    if len(or_bars) == 0:
        return None
    or_high = float(or_bars["high"].max())
    or_low  = float(or_bars["low"].min())
    or_width = or_high - or_low
    if or_width <= 0:
        or_width = atr14 * 0.1   # avoid division by zero

    # ── Bars up to and including 10:30 ───────────────────────────────────────
    prime_mask = bars_today.index.time <= PRIME_TIME
    prime_bars = bars_today[prime_mask]
    if len(prime_bars) == 0:
        return None

    prime_bar  = prime_bars.iloc[-1]
    price_1030 = float(prime_bar["close"])
    vol_1030   = float(prime_bars["volume"].sum())

    # Need bars at various lookbacks for momentum
    all_times = prime_bars.index
    def close_at_or_before(t: dtime) -> Optional[float]:
        mask = all_times.time <= t
        if not mask.any():
            return None
        return float(prime_bars[mask].iloc[-1]["close"])

    close_1025 = close_at_or_before(dtime(10, 25)) or price_1030
    close_1015 = close_at_or_before(dtime(10, 15)) or price_1030
    close_1000 = close_at_or_before(dtime(10, 0))  or price_1030

    # ── Momentum features ─────────────────────────────────────────────────────
    def clip_mom(val: float) -> float:
        return float(np.clip(val / atr14, -1.0, 1.0))

    mom_5m  = clip_mom(price_1030 - close_1025)
    mom_15m = clip_mom(price_1030 - close_1015)
    mom_30m = clip_mom(price_1030 - close_1000)
    mom_day = clip_mom(price_1030 - day_open)

    # ── OR features ──────────────────────────────────────────────────────────
    if price_1030 > or_high:
        or_breakout = 1.0
    elif price_1030 < or_low:
        or_breakout = -1.0
    else:
        or_breakout = 0.0

    or_position  = float(np.clip((price_1030 - or_low) / or_width, 0.0, 1.0))
    or_width_norm = float(np.clip(or_width / atr14, 0.0, 1.0))

    # ── Gap ──────────────────────────────────────────────────────────────────
    gap_norm = float(np.clip((day_open - prev_close) / atr14, -1.0, 1.0))

    # ── Volatility ───────────────────────────────────────────────────────────
    atr_norm   = float(np.clip(atr14 / ATR_NORM_REF, 0.0, 1.0))
    prev_range_norm = float(prev_row["high"] - prev_row["low"])
    if prev_range_norm > 0:
        atr_expand = float(np.clip(or_width / prev_range_norm, 0.0, 1.0))
    else:
        atr_expand = 0.5

    # ── Volume ratio ─────────────────────────────────────────────────────────
    avg_vol = float(np.mean(list(vol_history))) if vol_history else vol_1030
    if avg_vol <= 0:
        avg_vol = vol_1030 or 1.0
    log_today = math.log(vol_1030 + 1)
    log_avg   = math.log(avg_vol + 1)
    vol_ratio = float(np.clip(log_today / (log_avg + 1e-9), 0.0, 2.0)) / 2.0

    # ── VWAP at 10:30 ────────────────────────────────────────────────────────
    tp  = (prime_bars["high"] + prime_bars["low"] + prime_bars["close"]) / 3.0
    tvp = (tp * prime_bars["volume"]).sum()
    tv  = prime_bars["volume"].sum()
    vwap = float(tvp / tv) if tv > 0 else price_1030
    vwap_dist = float(np.clip((price_1030 - vwap) / atr14, -1.0, 1.0))

    # ── RSI (14-period on 1-min closes in prime window) ───────────────────────
    closes_arr = prime_bars["close"].values.astype(float)
    rsi_val    = _rsi(closes_arr, 14)
    rsi_norm   = rsi_val / 100.0

    # ── EMA distance (EMA20 on 1-min closes) ─────────────────────────────────
    ema20      = _ema(closes_arr, 20)
    ema_dist   = float(np.clip((price_1030 - ema20) / atr14, -1.0, 1.0))

    # ── Previous day features ─────────────────────────────────────────────────
    prev_range_feat = float(np.clip(prev_range_abs / atr14, 0.0, 2.0)) / 2.0
    if prev_range_abs > 0:
        prev_close_pos = float(np.clip((prev_close - prev_low) / prev_range_abs, 0.0, 1.0))
    else:
        prev_close_pos = 0.5

    # ── Direction (determined from OR breakout or momentum) ───────────────────
    if or_breakout == 1.0:
        direction_val = 1.0   # LONG
    elif or_breakout == -1.0:
        direction_val = 0.0   # SHORT
    else:
        direction_val = 1.0 if mom_30m > 0 else 0.0

    # ── ATR regime ────────────────────────────────────────────────────────────
    if atr14 < 8.0:
        atr_regime = 0.0
    elif atr14 <= 15.0:
        atr_regime = 0.5
    else:
        atr_regime = 1.0

    # ── Calendar ─────────────────────────────────────────────────────────────
    dt_obj   = datetime.combine(trade_date, dtime(0, 0))
    dow_norm  = dt_obj.weekday() / 4.0          # Mon=0 → 0.0, Fri=4 → 1.0
    month_norm = (dt_obj.month - 1) / 11.0

    # ── ML rolling features ───────────────────────────────────────────────────
    recent_wr5, streak_norm = ml.rolling_features()

    return {
        "mom_5m":        mom_5m,
        "mom_15m":       mom_15m,
        "mom_30m":       mom_30m,
        "mom_day":       mom_day,
        "or_breakout":   or_breakout,
        "or_position":   or_position,
        "or_width_norm": or_width_norm,
        "gap_norm":      gap_norm,
        "atr_norm":      atr_norm,
        "atr_expand":    atr_expand,
        "vol_ratio":     vol_ratio,
        "vwap_dist":     vwap_dist,
        "rsi_norm":      rsi_norm,
        "ema_dist":      ema_dist,
        "prev_range":    prev_range_feat,
        "prev_close_pos": prev_close_pos,
        "direction":     direction_val,
        "atr_regime":    atr_regime,
        "dow_norm":      dow_norm,
        "month_norm":    month_norm,
        "recent_wr5":    recent_wr5,
        "streak_norm":   streak_norm,
        # Extra context (not in FEATURE_NAMES, used by sim only)
        "_price_1030":   price_1030,
        "_atr14":        atr14,
        "_or_high":      or_high,
        "_or_low":       or_low,
        "_vol_1030":     vol_1030,
        "_direction_str": "LONG" if direction_val == 1.0 else "SHORT",
    }


# ── Trade simulation ─────────────────────────────────────────────────────────

def simulate_trade(
    direction: str,
    entry_price: float,
    atr14: float,
    bars_after: pd.DataFrame,   # 1-min bars from entry bar onwards (inclusive)
) -> Tuple[float, str, float]:
    """
    Simulate a single trade with ATR-based SL/TP, breakeven, trailing stop.
    Returns (exit_price, exit_type, exit_bar_time_str).

    SL  = clip(1.5*ATR, 2.0, 22.0)
    TP  = 2.5 * SL
    BE  : if profit >= 0.25*ATR, move SL to entry
    TSL : if profit >= 0.5*ATR, trail at best_price ∓ 0.25*ATR
    EOD : force exit at 15:30 bar close
    """
    sl_pts = float(np.clip(1.5 * atr14, 2.0, SL_CAP))
    tp_pts = 2.5 * sl_pts

    if direction == "LONG":
        sl_price  = entry_price - sl_pts
        tp_price  = entry_price + tp_pts
    else:
        sl_price  = entry_price + sl_pts
        tp_price  = entry_price - tp_pts

    be_triggered  = False
    tsl_triggered = False
    best_price    = entry_price
    tsl_trail_pts = 0.25 * atr14

    for i, (ts, bar) in enumerate(bars_after.iterrows()):
        t = ts.time()
        is_eod = (t >= EOD_EXIT)

        bar_high  = float(bar["high"])
        bar_low   = float(bar["low"])
        bar_close = float(bar["close"])

        if direction == "LONG":
            # Check SL (low touches stop)
            if bar_low <= sl_price:
                return sl_price, "SL", str(ts)

            # Check TP (high touches target)
            if bar_high >= tp_price:
                return tp_price, "TP", str(ts)

            # Update best price and trailing logic
            if bar_high > best_price:
                best_price = bar_high

            profit = best_price - entry_price
            if not be_triggered and profit >= 0.25 * atr14:
                sl_price = max(sl_price, entry_price)
                be_triggered = True
            if not tsl_triggered and profit >= 0.5 * atr14:
                tsl_triggered = True
            if tsl_triggered:
                tsl_stop = best_price - tsl_trail_pts
                sl_price = max(sl_price, tsl_stop)

        else:  # SHORT
            # Check SL
            if bar_high >= sl_price:
                return sl_price, "SL", str(ts)

            # Check TP
            if bar_low <= tp_price:
                return tp_price, "TP", str(ts)

            if bar_low < best_price:
                best_price = bar_low

            profit = entry_price - best_price
            if not be_triggered and profit >= 0.25 * atr14:
                sl_price = min(sl_price, entry_price)
                be_triggered = True
            if not tsl_triggered and profit >= 0.5 * atr14:
                tsl_triggered = True
            if tsl_triggered:
                tsl_stop = best_price + tsl_trail_pts
                sl_price = min(sl_price, tsl_stop)

        if is_eod:
            return bar_close, "EOD", str(ts)

    # No exit hit — exit at last bar close
    last_bar = bars_after.iloc[-1]
    return float(last_bar["close"]), "END", str(bars_after.index[-1])


# ── Statistics helpers ────────────────────────────────────────────────────────

def compute_sharpe(daily_rets: List[float]) -> float:
    arr = np.array(daily_rets, dtype=float)
    if len(arr) < 2 or arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std() * math.sqrt(252))


def compute_sortino(daily_rets: List[float]) -> float:
    arr = np.array(daily_rets, dtype=float)
    down = arr[arr < 0]
    if len(down) < 2 or down.std() == 0:
        return 0.0
    return float(arr.mean() / down.std() * math.sqrt(252))


def compute_max_drawdown(equity_curve: List[float]) -> float:
    arr = np.array(equity_curve, dtype=float)
    peak = arr[0]
    max_dd = 0.0
    for v in arr:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    start_balance: float = START_BALANCE,
    entry_thresh: float = 0.52,
    warmup_trades: int = 25,
) -> Dict:
    """
    Main simulation engine.
    Returns a results dict matching the required JSON schema.
    """
    # ── Date filtering ────────────────────────────────────────────────────────
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]

    if len(df) == 0:
        raise ValueError("No data in requested date range.")

    print(f"[v11] Loaded {len(df):,} bars from "
          f"{df.index[0].date()} to {df.index[-1].date()}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    daily_df, day_bars = preprocess(df)
    all_trade_dates = sorted(day_bars.keys())

    # ── State ─────────────────────────────────────────────────────────────────
    balance        = start_balance
    equity_curve   = [balance]
    daily_rets     = []             # daily return fractions

    trades         : List[Dict] = []
    total_trades   = 0
    wins           = 0
    losses         = 0
    gross_profit   = 0.0
    gross_loss     = 0.0

    # 3-strike lockout
    consec_losses  = 0
    lockout_active = False
    lockout_reset_tomorrow = False   # flag: reset lockout at next eligible day

    vol_history    = deque(maxlen=20)   # vol at 10:30 for last 20 trading days

    ml = WalkForwardMLv11(entry_thresh=entry_thresh, warmup=warmup_trades)

    yearly_pnl : Dict[str, float] = {}
    yearly_start_bal: Dict[str, float] = {}

    prev_balance = balance

    print("\n[v11] Starting simulation...\n")

    for trade_date in all_trade_dates:
        bars_today = day_bars[trade_date]
        year_key   = str(trade_date.year)

        if year_key not in yearly_start_bal:
            yearly_start_bal[year_key] = balance
            yearly_pnl[year_key]       = 0.0

        # ── ATR gate ─────────────────────────────────────────────────────────
        if trade_date not in daily_df.index:
            continue
        atr14 = float(daily_df.loc[trade_date, "atr14"])
        if atr14 < ATR_MIN:
            continue

        # ── Lockout gate ──────────────────────────────────────────────────────
        if lockout_reset_tomorrow:
            consec_losses = 0
            lockout_active = False
            lockout_reset_tomorrow = False

        if lockout_active:
            # still in lockout: skip but note we will reset next day
            lockout_reset_tomorrow = True
            continue

        # ── Check we have a 10:30 bar ─────────────────────────────────────────
        prime_mask = bars_today.index.time == PRIME_TIME
        if not prime_mask.any():
            continue

        # ── Compute features ──────────────────────────────────────────────────
        feat = compute_features(trade_date, bars_today, daily_df, ml, vol_history)
        if feat is None:
            continue

        price_1030 = feat["_price_1030"]
        or_high    = feat["_or_high"]
        or_low     = feat["_or_low"]
        direction  = feat["_direction_str"]
        vol_1030   = feat["_vol_1030"]

        # Update volume history (used for next day's feature computation)
        vol_history.append(vol_1030)

        # Feature vector (ordered by FEATURE_NAMES)
        feat_vec = [feat[k] for k in FEATURE_NAMES]

        # ── ML gate ───────────────────────────────────────────────────────────
        enter, p_win = ml.should_enter(feat_vec)
        if not enter:
            continue

        # ── Position sizing ───────────────────────────────────────────────────
        sl_pts = float(np.clip(1.5 * atr14, 2.0, SL_CAP))
        risk_dollar   = balance * RISK_PCT
        base_contracts = max(1, int(risk_dollar / (sl_pts * PNL_PER_POINT)))
        max_contracts  = max(1, int((balance * 0.06) / (sl_pts * PNL_PER_POINT)))

        size_mult  = ml.size_multiplier(p_win)
        contracts  = min(round(base_contracts * size_mult), max_contracts)
        contracts  = max(1, contracts)

        # ── Entry price (10:30 bar close + slippage) ──────────────────────────
        if direction == "LONG":
            entry_price = price_1030 + SLIPPAGE_PTS
        else:
            entry_price = price_1030 - SLIPPAGE_PTS

        # ── Simulate trade on bars AFTER 10:30 through EOD ───────────────────
        after_mask = bars_today.index.time >= PRIME_TIME
        bars_after = bars_today[after_mask]

        exit_price, exit_type, exit_ts = simulate_trade(
            direction, entry_price, atr14, bars_after
        )

        # Add slippage to exit
        if direction == "LONG":
            exit_price_net = exit_price - SLIPPAGE_PTS
        else:
            exit_price_net = exit_price + SLIPPAGE_PTS

        # ── P&L ───────────────────────────────────────────────────────────────
        if direction == "LONG":
            raw_pnl = (exit_price_net - entry_price) * PNL_PER_POINT * contracts
        else:
            raw_pnl = (entry_price - exit_price_net) * PNL_PER_POINT * contracts

        commission_total = COMMISSION * contracts * 2   # entry + exit sides
        net_pnl = raw_pnl - commission_total

        win = net_pnl > 0
        balance += net_pnl

        # ── Update state ──────────────────────────────────────────────────────
        ml.record_entry(feat_vec)
        ml.record_exit(win)

        total_trades += 1
        if win:
            wins += 1
            gross_profit += net_pnl
            consec_losses = 0
        else:
            losses += 1
            gross_loss   += abs(net_pnl)
            consec_losses += 1
            if consec_losses >= 3:
                lockout_active = True

        equity_curve.append(balance)
        yearly_pnl[year_key] = yearly_pnl.get(year_key, 0.0) + net_pnl

        # Daily return (simple fraction)
        day_ret = (balance - prev_balance) / prev_balance if prev_balance > 0 else 0.0
        daily_rets.append(day_ret)
        prev_balance = balance

        trades.append({
            "date":           str(trade_date),
            "direction":      direction,
            "entry_price":    round(entry_price, 4),
            "exit_price":     round(exit_price_net, 4),
            "exit_type":      exit_type,
            "sl_points":      round(sl_pts, 4),
            "atr":            round(atr14, 4),
            "contracts":      contracts,
            "pnl":            round(net_pnl, 2),
            "balance":        round(balance, 2),
            "ml_confidence":  round(p_win, 4),
            "ml_active":      ml.ml_active(),
            "or_breakout":    feat["or_breakout"],
            "mom_30m":        round(feat["mom_30m"], 4),
        })

    # ── Post-loop statistics ──────────────────────────────────────────────────
    total_pnl    = balance - start_balance
    pnl_pct      = total_pnl / start_balance * 100.0
    win_rate     = wins / total_trades if total_trades > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win  = gross_profit / wins   if wins   > 0 else 0.0
    avg_loss = gross_loss  / losses  if losses > 0 else 0.0
    rr_ratio = avg_win / avg_loss    if avg_loss > 0 else float("inf")

    max_dd = compute_max_drawdown(equity_curve)

    # Annualised return
    if len(all_trade_dates) < 2:
        years = 1.0
    else:
        first = datetime.combine(all_trade_dates[0], dtime(0, 0))
        last  = datetime.combine(all_trade_dates[-1], dtime(0, 0))
        years = max((last - first).days / 365.25, 1 / 365.25)

    cagr = (balance / start_balance) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    annual_return = cagr * 100.0

    sharpe  = compute_sharpe(daily_rets)
    sortino = compute_sortino(daily_rets)
    calmar  = (cagr / max_dd) if max_dd > 0 else float("inf")

    # Yearly pnl: format nicely
    yearly_pnl_out = {yr: round(pnl, 2) for yr, pnl in sorted(yearly_pnl.items())}

    # Feature importances (from last ML fit)
    fi = {k: round(v, 4) for k, v in
          sorted(ml.feature_importances.items(), key=lambda x: -x[1])}

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  MES Futures v11.0 — ML Scoring (WalkForward LightGBM)")
    print("=" * 65)
    print(f"  Period     : {all_trade_dates[0]} → {all_trade_dates[-1]}")
    print(f"  Total Trades: {total_trades}  |  Wins: {wins}  |  Losses: {losses}")
    print(f"  Win Rate   : {win_rate*100:.1f}%")
    print(f"  Profit Factor: {profit_factor:.2f}   RR: {rr_ratio:.2f}x")
    print(f"  Start Balance: ${start_balance:,.2f}")
    print(f"  End Balance  : ${balance:,.2f}")
    print(f"  Total P&L    : ${total_pnl:+,.2f}  ({pnl_pct:+.1f}%)")
    print(f"  Annual Return: {annual_return:.1f}%")
    print(f"  Max Drawdown : {max_dd*100:.1f}%")
    print(f"  Sharpe Ratio : {sharpe:.2f}")
    print(f"  Sortino Ratio: {sortino:.2f}")
    print(f"  Calmar Ratio : {calmar:.2f}")
    print(f"  ML Filtered  : {ml.ml_filtered}  |  ML Passed: {ml.ml_passed}")
    print("-" * 65)
    print("  Yearly P&L:")
    for yr, pnl in sorted(yearly_pnl_out.items()):
        print(f"    {yr}: ${pnl:+,.2f}")
    print("-" * 65)
    if fi:
        print("  Top Feature Importances:")
        for i, (k, v) in enumerate(fi.items()):
            if i >= 5:
                break
            print(f"    {k:20s}: {v:.4f}")
    print("=" * 65)

    return {
        "model":        "MES Futures v11.0 — ML Scoring (OR breakout, 22 features)",
        "period":       f"{all_trade_dates[0]} to {all_trade_dates[-1]}",
        "product":      "MES $5/pt",
        "strategy":     "WalkForward LightGBM · P(win)≥0.52 · OR breakout · ATR SL · 10:30 PRIME",
        "total_trades": total_trades,
        "prime_trades": total_trades,
        "wins":         wins,
        "losses":       losses,
        "win_rate":     round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "rr_ratio":     round(rr_ratio, 4),
        "annual_return": round(annual_return, 2),
        "max_drawdown":  round(max_dd, 4),
        "sharpe_ratio":  round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio":  round(calmar, 4),
        "start_balance": start_balance,
        "end_balance":   round(balance, 2),
        "total_pnl":     round(total_pnl, 2),
        "pnl_pct":       round(pnl_pct, 2),
        "years":         round(years, 4),
        "yearly_pnl":    yearly_pnl_out,
        "feature_importance": fi,
        "ml_stats": {
            "warmup_trades": ml.WARMUP_TRADES,
            "ml_filtered":   ml.ml_filtered,
            "ml_passed":     ml.ml_passed,
        },
        "trades": trades,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global SL_CAP, RISK_PCT
    parser = argparse.ArgumentParser(
        description="MES Futures v11.0 — Walk-Forward ML Backtest"
    )
    parser.add_argument(
        "--csv",
        default="MES_1min_data_et_rth.csv",
        help="Path to 1-min MES OHLCV CSV (timestamp,open,high,low,close,volume)",
    )
    parser.add_argument("--start",   default="2023-03-27", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=None,         help="End date YYYY-MM-DD")
    parser.add_argument("--balance", type=float, default=START_BALANCE,
                        help="Starting balance in USD")
    parser.add_argument("--entry-thresh", type=float, default=0.60,
                        help="ML P(win) threshold for entry")
    parser.add_argument("--warmup",  type=int, default=25,
                        help="Number of trades before ML gate activates")
    parser.add_argument("--sl-cap",  type=float, default=SL_CAP,
                        help="Max SL in MES points (default 22)")
    parser.add_argument("--risk-pct", type=float, default=RISK_PCT,
                        help="Base risk per trade as fraction of balance (default 0.025)")
    parser.add_argument("--out",     default="backtest_v11.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    # Apply tunable globals from CLI
    SL_CAP   = args.sl_cap
    RISK_PCT = args.risk_pct

    # ── Load CSV ──────────────────────────────────────────────────────────────
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path)

    print(f"[v11] Loading CSV: {csv_path}")
    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )
    df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]
    df.sort_index(inplace=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    results = run_backtest(
        df,
        start_date=args.start,
        end_date=args.end,
        start_balance=args.balance,
        entry_thresh=args.entry_thresh,
        warmup_trades=args.warmup,
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)

    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    print(f"\n[v11] Results saved → {out_path}")


if __name__ == "__main__":
    main()
