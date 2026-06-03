#!/usr/bin/env python3
"""
S&P 500 Futures (ES) Backtest - Pro Trader Strategy Integration
Deterministic minute-by-minute simulation - NO Theta decay, NO bid-ask friction.

Integrated Strategies (from top futures traders):
  - Toby Crabel NR7 Volatility Filter (narrow range -> breakout boost)
  - 3-Day Pullback Mean Reversion (60-65% WR statistical edge)
  - Gap Context Filter (small gap fade, large gap follow)
  - Daily Trend Bias (20 SMA macro alignment)
  - ATR-Based Dynamic SL (1.5x ATR, adapts to volatility)
  - Kelly-Informed Position Sizing (10% risk, margin-aware)

v5 Configuration (High-Frequency Dual-Window):
  - Entry: PRIME 10:30 AM + GAMMA 14:00 PM | Exit: 15:30 PM
  - SL = 1.5x ATR PRIME / 1.125x ATR GAMMA (tighter for 90-min window)
  - MIN_SCORE = 78 (MODERATE grade; STRONG = 88+)
  - Risk = 1.5% per trade | LockoutStrikes=3 | LockoutDays=1
  - Margin = $50/contract (MES day-trading margin)

Product: Micro E-mini S&P 500 (MES)
  - 1 ES contract = $50 per point of S&P 500
  - Tick size: 0.25 points ($12.50 per tick)
  - Commission: ~$1.25 per side per contract (round-trip ~$2.50)
  - Day Margin: ~$500 per ES contract (AMP/NinjaTrader intraday)
"""
import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime, timedelta, time as dtime
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
import pytz
warnings.filterwarnings("ignore")

# ML imports
try:
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[!] ML libraries not found. pip install scikit-learn lightgbm")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
from engines.regime import calculate_regime_score
from engines.correlation import calculate_correlation_score
from engines.time_window import calculate_time_score
from engines.technical import calculate_technical_score
from v10_strategy import (
    evaluate_entry as _shared_evaluate_entry,
    init_position as _shared_init_position,
    manage_bar as _shared_manage_bar,
)

NY = pytz.timezone("America/New_York")

# -- MES Contract Specifications (matches api/data.py live trading) --
ES_MULTIPLIER = 5.0        # $5 per point of S&P 500 (MES — Micro E-mini)
ES_COMMISSION_RT = 0.50    # Round-trip commission per MES contract
ES_SLIPPAGE_PTS = 0.25     # 1 tick slippage per side
ES_DAY_MARGIN = 50.0       # Day-trading margin per MES contract

# -- Strategy Parameters v10.1 --
MIN_SCORE = 60              # PRIME window minimum (v10.1: score≥60, ~3.1 days/trade)
GAMMA_MIN_SCORE = 83        # GAMMA window minimum (scores naturally lower at 14:00)
RISK_PCT = 0.015
MARGIN_UTIL = 0.95
EXIT_TIME = dtime(15, 30)
VIX_THRESHOLD = 20.0
ADX_RUNAWAY = 40.0
RSI_UPPER = 90.0
RSI_LOWER = 10.0
SECTOR_THRESHOLD = 1.8
LOCKOUT_STRIKES = 5         # lenient lockout (was 2)
LOCKOUT_DAYS = 0            # no cooldown period (was 1)
ATR_SL_MULT = 1.5
TP_MULT = 2.5               # TP = 2.5x SL (v10.1)
TRAILING_ACTIVATION = 0.5
TRAILING_STEP = 0.25
BREAKEVEN_AT = 0.25

# -- Entry Windows: PRIME (10:30~11:30) + REENTRY (13:00~13:30) + GAMMA (14:00~14:45) --
ENTRY_WINDOWS = [
    dtime(10, 30), dtime(10, 45), dtime(11, 0), dtime(11, 15), dtime(11, 30),  # PRIME (20pts)
    dtime(13, 0),  dtime(13, 30),                                              # REENTRY (afternoon)
    dtime(14, 0),  dtime(14, 15), dtime(14, 30), dtime(14, 45),                # GAMMA (15pts)
]
MAX_TRADES_PER_DAY = 4       # Up to 4 positions per day (PRIME + REENTRY + GAMMA)

# -- Pro Strategy Bonuses --
NR7_SCORE_BOOST = 5          # Score boost on NR7 days (Crabel)
PULLBACK_SCORE_BOOST = 5     # Score boost on 3-day pullback (Mean Reversion)


def load_vix_data():
    """Load yfinance historical VIX data for alignment."""
    import yfinance as yf
    print("[*] Fetching historical VIX data for backtest...")
    try:
        vix_df = yf.download("^VIX", start="2018-01-01", end="2026-05-25", interval="1d", progress=False)
        if not vix_df.empty:
            if isinstance(vix_df.columns, pd.MultiIndex):
                return vix_df["Close"].squeeze()
            return vix_df["Close"]
    except Exception as e:
        print(f"Warning: Could not fetch VIX data ({e}). Defaulting to 18.0 VIX.")
    return pd.Series(dtype=float)


def build_daily_ohlc(days_dict, trading_days):
    """Build daily OHLC from minute bars for NR7, ATR, etc."""
    daily = {}
    for ds in trading_days:
        bars = days_dict[ds]
        daily[ds] = {
            "open": bars[0][1],
            "high": max(b[2] for b in bars),
            "low": min(b[3] for b in bars),
            "close": bars[-1][4]
        }
    return daily


def calc_atr(daily_ohlc, trading_days, idx, period=14):
    """Calculate ATR(period) from daily OHLC."""
    tr_list = []
    for j in range(1, period + 1):
        if idx - j < 0:
            break
        ds = trading_days[idx - j]
        if ds not in daily_ohlc:
            continue
        d = daily_ohlc[ds]
        prev_ds = trading_days[idx - j - 1] if idx - j - 1 >= 0 else ds
        prev_d = daily_ohlc.get(prev_ds, d)
        tr = max(d["high"] - d["low"],
                 abs(d["high"] - prev_d["close"]),
                 abs(d["low"] - prev_d["close"]))
        tr_list.append(tr)
    return np.mean(tr_list) if len(tr_list) >= 10 else 4.0


def check_nr7(daily_ohlc, trading_days, idx):
    """Check if today is an NR7 day (Toby Crabel)."""
    if idx < 7:
        return False
    ds = trading_days[idx]
    if ds not in daily_ohlc:
        return False
    today_range = daily_ohlc[ds]["high"] - daily_ohlc[ds]["low"]
    prev_ranges = []
    for j in range(1, 7):
        if idx - j >= 0:
            prev_ds = trading_days[idx - j]
            if prev_ds in daily_ohlc:
                d = daily_ohlc[prev_ds]
                prev_ranges.append(d["high"] - d["low"])
    if len(prev_ranges) >= 6 and today_range < min(prev_ranges):
        return True
    return False


def check_3day_pullback(daily_ohlc, trading_days, idx):
    """Check if there were 3+ consecutive down closes (mean reversion signal)."""
    if idx < 4:
        return False
    consecutive_down = 0
    for j in range(1, 4):
        if idx - j < 0 or idx - j - 1 < 0:
            break
        prev_ds = trading_days[idx - j]
        prev2_ds = trading_days[idx - j - 1]
        if prev_ds in daily_ohlc and prev2_ds in daily_ohlc:
            if daily_ohlc[prev_ds]["close"] < daily_ohlc[prev2_ds]["close"]:
                consecutive_down += 1
            else:
                break
    return consecutive_down >= 3


def check_daily_bias(daily_ohlc, trading_days, idx, spy_open):
    """Check if price is above 20-day SMA (daily trend filter)."""
    if idx < 20:
        return True  # Default bullish if not enough data
    closes = []
    for j in range(1, 21):
        if idx - j >= 0:
            ds = trading_days[idx - j]
            if ds in daily_ohlc:
                closes.append(daily_ohlc[ds]["close"])
    if len(closes) >= 20:
        return spy_open > np.mean(closes)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward ML System
# ─────────────────────────────────────────────────────────────────────────────
class WalkForwardML:
    """
    Walk-Forward ML 사이징+필터링 시스템.
    - SKIP_AFTER_N 건 이후: P(win) < SKIP_THRESH → 진입 차단
    - P(win) >= SIZE_HIGH  → 계약수 +30%
    - P(win) >= SIZE_MED   → 계약수 +15%
    - P(win) <  SIZE_LOW   → 계약수 -20%
    """
    MIN_TRAIN    = 15
    SKIP_AFTER_N = 25    # start hard-filtering after 25 completed trades
    SKIP_THRESH  = 0.35  # skip if confidence < this threshold (loosened from 0.43 for more trades)
    RETRAIN_N    = 5
    SIZE_HIGH  = 0.62   # +30%
    SIZE_MED   = 0.55   # +15%
    SIZE_LOW   = 0.44   # -20%

    FEATURE_NAMES = [
        "score_norm", "vix_norm", "atr_pct",
        "is_long", "day_of_week", "month_norm",
        "recent_wr5", "streak_norm",
        "open_momentum", "vix_regime",
    ]

    def __init__(self):
        self.X: List[List[float]] = []
        self.y: List[int] = []
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.n_since_retrain = 0
        self.conf_history: List[float] = []        # all predictions (including skipped)
        self.executed_confs: List[float] = []      # confs for executed trades only
        self.feature_importance: Dict[str, float] = {}

    @staticmethod
    def features(score: float, vix: float, atr: float, price: float,
                 direction: str, date_dt, recent_wr5: float,
                 consec_wins: int, consec_losses: int,
                 day_open: float = 0.0) -> List[float]:
        atr_pct    = min(atr / max(price, 1), 0.05) / 0.05
        vix_norm   = min(vix, 40) / 40.0
        streak     = (min(consec_wins, 5) - min(consec_losses, 5)) / 5.0
        open_mom   = float(np.clip((price - day_open) / max(atr, 1), -2, 2) / 2.0)
        vix_regime = float(np.clip(vix / max(atr * 0.4, 1), 0, 3) / 3.0)
        return [
            score / 100.0,
            vix_norm,
            atr_pct,
            1.0 if direction == "LONG" else 0.0,
            date_dt.weekday() / 4.0,
            (date_dt.month - 1) / 11.0,
            recent_wr5,
            streak,
            open_mom,
            vix_regime,
        ]

    def predict(self, feat: List[float]) -> float:
        if not self.is_trained or self.model is None:
            return 0.5
        try:
            from sklearn.preprocessing import StandardScaler
            X = self.scaler.transform(np.array(feat).reshape(1, -1))
            return float(self.model.predict_proba(X)[0][1])
        except Exception:
            return 0.5

    def update(self, feat: List[float], won: bool) -> None:
        self.X.append(feat)
        self.y.append(1 if won else 0)
        if self.conf_history:
            self.executed_confs.append(self.conf_history[-1])
        self.n_since_retrain += 1
        if len(self.X) >= self.MIN_TRAIN and self.n_since_retrain >= self.RETRAIN_N:
            self._fit()

    def _fit(self) -> None:
        X = np.array(self.X)
        y = np.array(self.y)
        if len(np.unique(y)) < 2:
            return
        try:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X)
            if ML_AVAILABLE:
                import lightgbm as lgb
                params = dict(
                    n_estimators=60, max_depth=3, learning_rate=0.1,
                    num_leaves=7, min_child_samples=4, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=1.5,
                    objective="binary", verbose=-1, n_jobs=1,
                    class_weight="balanced", random_state=42,
                )
                self.model = lgb.LGBMClassifier(**params)
                self.model.fit(Xs, y)
                imp = self.model.feature_importances_
                total = max(float(imp.sum()), 1e-9)
                self.feature_importance = {
                    n: round(float(v) / total, 4)
                    for n, v in zip(self.FEATURE_NAMES, imp)
                }
            else:
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(C=0.3, max_iter=1000, random_state=42, class_weight="balanced")
                self.model.fit(Xs, y)
                coef = np.abs(self.model.coef_[0])
                total = max(float(coef.sum()), 1e-9)
                self.feature_importance = {
                    n: round(float(v) / total, 4)
                    for n, v in zip(self.FEATURE_NAMES, coef)
                }
            self.is_trained = True
            self.n_since_retrain = 0
        except Exception as e:
            pass

    def should_skip(self) -> bool:
        """Returns True if this trade should be skipped (low ML confidence after sufficient training)."""
        if not self.is_trained or len(self.y) < self.SKIP_AFTER_N or not self.conf_history:
            return False
        return self.conf_history[-1] < self.SKIP_THRESH

    def apply_sizing(self, num_contracts: int, max_contracts: int) -> int:
        if not self.is_trained or not self.conf_history:
            return num_contracts
        conf = self.conf_history[-1]
        if conf >= self.SIZE_HIGH:
            return min(int(num_contracts * 1.30), max_contracts)
        elif conf >= self.SIZE_MED:
            return min(int(num_contracts * 1.15), max_contracts)
        elif conf < self.SIZE_LOW:
            return max(int(num_contracts * 0.80), 1)
        return num_contracts

    def stats(self) -> Dict:
        if not self.y:
            return {}
        conf_arr = np.array(self.conf_history) if self.conf_history else np.array([0.5])
        # Use aligned executed_confs (same length as y) for accurate high_conf_wr
        confs_exec = self.executed_confs if self.executed_confs else [0.5] * len(self.y)
        wins_in_high = sum(
            1 for c, w in zip(confs_exec, self.y)
            if c >= self.SIZE_HIGH and w == 1
        )
        n_high = sum(1 for c in confs_exec if c >= self.SIZE_HIGH)
        return {
            "total_predictions":  len(self.conf_history),
            "training_samples":   len(self.X),
            "avg_confidence":     round(float(conf_arr.mean()), 3),
            "high_conf_wr":       round(wins_in_high / max(n_high, 1), 3),
            "historical_wr":      round(sum(self.y) / max(len(self.y), 1), 3),
            "feature_importance": self.feature_importance,
            "confidence_series":  [round(c, 3) for c in self.conf_history],
        }


def run_futures_backtest(csv_path: str, start_str: str = "2023-03-25",
                         end_str: str = "2026-03-25",
                         start_balance: float = 10000.0,
                         fixed_size: bool = False,
                         out_path: str = "backtest_futures.json",
                         vix_max: Optional[float] = None,
                         atr_min: Optional[float] = None,
                         no_mean_reversion: bool = False):
    t_start = time.time()
    # Pre-compute fixed contracts using start_balance (no reinvestment)
    _fixed_sl_ref = 15.0  # reference SL for fixed sizing (PRIME cap)
    _fixed_risk_per = (_fixed_sl_ref + ES_SLIPPAGE_PTS * 2) * ES_MULTIPLIER + ES_COMMISSION_RT
    FIXED_CONTRACTS = max(1, int((start_balance * RISK_PCT) / _fixed_risk_per))
    print("=" * 80)
    print("  MICRO E-MINI (MES) - PRO TRADER STRATEGY INTEGRATION")
    print(f"  ATR SL={ATR_SL_MULT}x | Risk={RISK_PCT*100:.1f}% | Margin=${ES_DAY_MARGIN:.0f}")
    print(f"  NR7 + 3Day Pullback + Gap + Daily Bias | MIN_SCORE={MIN_SCORE}")
    sizing_mode = f"FIXED ({FIXED_CONTRACTS}계약)" if fixed_size else "DYNAMIC (잔고비례)"
    print(f"  포지션 사이징: {sizing_mode}")
    print("=" * 80)

    # 1. Load VIX
    vix_series = load_vix_data()

    # 2. Load CSV
    print(f"[*] Loading historical 1-minute bars from {csv_path}...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df):,} rows in {time.time()-t0:.1f}s.")

    # Parse timestamps — CSV is already in NY local time (generated by gen_synthetic_1min.py)
    print("[*] Parsing timestamps...")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize(NY, ambiguous="infer", nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert(NY)
    df = df.sort_index()

    # Filter dates
    start_dt = pd.to_datetime(start_str).tz_localize(NY)
    end_dt = pd.to_datetime(end_str).tz_localize(NY) + timedelta(days=1)
    df_filtered = df[(df.index >= start_dt) & (df.index < end_dt)].copy()

    if df_filtered.empty:
        print("ERROR: No data found in date range.")
        return

    print(f"[*] Filtered date range: {start_str} ~ {end_str} ({len(df_filtered):,} rows).")

    # Group bars by day
    days_dict = {}
    for ts, row in df_filtered.iterrows():
        day_str = ts.strftime("%Y-%m-%d")
        if day_str not in days_dict:
            days_dict[day_str] = []
        days_dict[day_str].append((ts, row["open"], row["high"], row["low"], row["close"], row["volume"]))

    trading_days = sorted(list(days_dict.keys()))
    print(f"[*] Identified {len(trading_days)} trading days in dataset.")

    # Build daily OHLC for pro strategies
    daily_ohlc = build_daily_ohlc(days_dict, trading_days)

    balance = start_balance
    trades = []
    wins, losses = 0, 0
    consecutive_losses = 0
    consecutive_wins = 0
    max_consec_wins = 0
    max_consec_losses = 0
    lockout_cooldown = 0
    monthly_pnl: Dict[str, float] = {}    # "YYYY-MM" -> net P&L
    daily_balance: Dict[str, float] = {}  # "YYYY-MM-DD" -> closing balance

    # ML Walk-Forward System 초기화
    ml = WalkForwardML()

    pbar = tqdm(trading_days, desc="Backtesting Days")

    for day_idx, day_str in enumerate(pbar):
        # -- Layer 7: Lockout --
        if lockout_cooldown > 0:
            lockout_cooldown -= 1
            continue

        day_bars = days_dict[day_str]
        if len(day_bars) < 60:
            continue

        df_day = pd.DataFrame(day_bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]).set_index("timestamp")
        spy_o = float(df_day["Open"].iloc[0])

        # Get VIX
        try:
            vix_val = float(vix_series.loc[day_str])
        except:
            vix_val = 18.0

        # ===== Shared v10 decision inputs (newest-LAST daily history incl. today) =====
        # The entry decision (ATR, NR7, pullback, daily-bias, scoring, direction,
        # SL/TP) is delegated to api/v10_strategy.evaluate_entry so the live paper
        # bot trades the *identical* logic. Only ML sizing + the minute-by-minute
        # exit simulation remain backtest-only below.
        _hist = trading_days[:day_idx + 1]
        _dh = [daily_ohlc[d]["high"]  for d in _hist if d in daily_ohlc]
        _dl = [daily_ohlc[d]["low"]   for d in _hist if d in daily_ohlc]
        _dc = [daily_ohlc[d]["close"] for d in _hist if d in daily_ohlc]

        # ── Optional VIX filter (per-run override; ATR floor handled in module) ──
        if vix_max is not None and vix_val > vix_max:
            continue

        # ===== Multi-Window Entry Scan: PRIME (10:30~11:15) + GAMMA (14:00~14:30) =====
        trades_today = 0
        last_exit_time_today = dtime(0, 0)   # track last exit to prevent overlap
        for entry_time in ENTRY_WINDOWS:
            if trades_today >= MAX_TRADES_PER_DAY:
                break
            # Skip if last position hasn't closed yet (prevent concurrent overlap)
            if entry_time <= last_exit_time_today:
                continue

            is_gamma = (entry_time >= dtime(13, 0))   # afternoon (REENTRY+GAMMA): lower threshold, tighter SL

            # Find entry bar for this window
            entry_bar = None
            for ts, o, h, l, c, v in day_bars:
                if ts.time() >= entry_time:
                    entry_bar = (ts, o, h, l, c, v)
                    break
            if not entry_bar:
                continue

            ts_entry, spy_entry_price, _, _, _, _ = entry_bar

            # Slice bars up to entry time for scoring
            df_morning = df_day[df_day.index.time <= entry_time].copy()
            if len(df_morning) < 5:
                continue
            df_morning.columns = [col.capitalize() for col in df_morning.columns]

            # ── Shared v10 entry decision (identical logic runs in the live bot) ──
            gamma_mult = 0.75 if is_gamma else 1.0
            sl_cap = 10.0 if is_gamma else 15.0
            effective_min = GAMMA_MIN_SCORE if is_gamma else MIN_SCORE
            decision = _shared_evaluate_entry(
                daily_highs=_dh, daily_lows=_dl, daily_closes=_dc,
                day_open=spy_o, entry_price=spy_entry_price, entry_ts=ts_entry,
                vix_val=vix_val, morning_df=df_morning,
                min_score=effective_min, tp_mult=TP_MULT, atr_min=atr_min,
                no_mean_reversion=no_mean_reversion,
                gamma_mult=gamma_mult, sl_cap=sl_cap,
            )
            atr_val = decision["atr"]
            if not decision["enter"]:
                continue

            window_sl = decision["sl_points"]
            normalized = decision["score"]
            boosted_score = decision["boosted_score"]
            boost_reasons = decision["boost_reasons"].split(",") if decision["boost_reasons"] else []
            grade = decision["grade"]
            trade_dir = decision["direction"]
            strategy_used = decision["strategy"]

            # ── ML Walk-Forward 사이징 ───────────────────────────────────────
            recent_wr5 = (sum(1 for t in trades[-5:] if t["pnl"] > 0)
                          / max(len(trades[-5:]), 1))
            _day_dt = datetime.strptime(day_str, "%Y-%m-%d")
            ml_feat = WalkForwardML.features(
                score=normalized, vix=vix_val, atr=atr_val,
                price=spy_entry_price, direction=trade_dir,
                date_dt=_day_dt, recent_wr5=recent_wr5,
                consec_wins=consecutive_wins, consec_losses=consecutive_losses,
                day_open=spy_o,
            )
            ml_conf = ml.predict(ml_feat)
            ml.conf_history.append(round(ml_conf, 3))
            if ml.should_skip():
                continue  # ML hard filter: skip low-confidence trades
            # ────────────────────────────────────────────────────────────────

            # Position Sizing
            if fixed_size:
                num_contracts = FIXED_CONTRACTS
            else:
                max_risk_dollar = balance * RISK_PCT
                risk_per_contract = (window_sl + ES_SLIPPAGE_PTS * 2) * ES_MULTIPLIER + ES_COMMISSION_RT
                num_contracts = int(max_risk_dollar / risk_per_contract)
                if num_contracts == 0:
                    num_contracts = 1
                max_by_margin = int((balance * MARGIN_UTIL) / ES_DAY_MARGIN)
                if max_by_margin == 0:
                    max_by_margin = 1
                num_contracts = min(num_contracts, max_by_margin)
                if num_contracts * ES_DAY_MARGIN > balance:
                    continue

            # ML 사이징 적용
            if not fixed_size:
                max_by_margin_ref = int((balance * MARGIN_UTIL) / ES_DAY_MARGIN)
                num_contracts = ml.apply_sizing(num_contracts, max_by_margin_ref)

            # Minute-by-Minute Simulation — exit management delegated to the
            # shared module so the live worker (api/v10_runner.v10_worker)
            # trails/breaks-even identically.
            entry_price = spy_entry_price
            _pos = _shared_init_position(trade_dir, entry_price, atr_val, window_sl, TP_MULT)

            exit_price = None
            exit_type = "EOD"
            exit_time_str = f"{EXIT_TIME.hour}:{EXIT_TIME.minute:02d}"

            for ts_bar, o_bar, h_bar, l_bar, c_bar, v_bar in day_bars:
                if ts_bar.time() <= entry_time:
                    continue
                if ts_bar.time() > EXIT_TIME:
                    break
                _xp, _xt = _shared_manage_bar(_pos, h_bar, l_bar)
                if _xt is not None:
                    exit_price = _xp
                    exit_type = _xt
                    exit_time_str = ts_bar.strftime("%H:%M")
                    break

            # EOD fallback exit
            if exit_price is None:
                eod_price = None
                for bar_ts, bar_o, bar_h, bar_l, bar_c, bar_v in reversed(day_bars):
                    if bar_ts.time() <= EXIT_TIME:
                        eod_price = bar_c; break
                if eod_price is None:
                    eod_price = float(df_day["Close"].iloc[-1])
                exit_price = eod_price
                exit_type = "EOD"

            # P&L Calculation
            point_pnl = (exit_price - entry_price) if trade_dir == "LONG" else (entry_price - exit_price)
            net_point_pnl = point_pnl - (ES_SLIPPAGE_PTS * 2)
            gross_pnl = net_point_pnl * ES_MULTIPLIER * num_contracts
            total_pnl = gross_pnl - (ES_COMMISSION_RT * num_contracts)

            balance += total_pnl
            if total_pnl > 0:
                wins += 1
                consecutive_wins += 1
                consecutive_losses = 0
                max_consec_wins = max(max_consec_wins, consecutive_wins)
            else:
                losses += 1
                consecutive_losses += 1
                consecutive_wins = 0
                max_consec_losses = max(max_consec_losses, consecutive_losses)
                if consecutive_losses >= LOCKOUT_STRIKES:
                    lockout_cooldown = LOCKOUT_DAYS
                    consecutive_losses = 0
                prev_balance = balance - total_pnl
                if prev_balance > 0 and abs(total_pnl) / prev_balance >= 0.06:
                    lockout_cooldown = LOCKOUT_DAYS

            # Monthly tracking
            month_key = day_str[:7]
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + total_pnl
            daily_balance[day_str] = round(balance, 2)

            trades.append({
                "date": day_str,
                "window": "GAMMA" if is_gamma else "PRIME",
                "grade": grade,
                "score": normalized,
                "boosted_score": boosted_score,
                "boost_reasons": ",".join(boost_reasons) if boost_reasons else "",
                "direction": trade_dir,
                "strategy": strategy_used,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "exit_type": exit_type,
                "exit_time": exit_time_str,
                "sl_points": round(window_sl, 2),
                "atr": round(atr_val, 2),
                "point_pnl": round(point_pnl, 2),
                "contracts": num_contracts,
                "pnl": round(total_pnl, 2),
                "balance": round(balance, 2),
                "vix": round(vix_val, 1),
                "ml_confidence": round(ml_conf, 3),
                "ml_active": ml.is_trained,
            })

            # ML 결과 학습 (거래 완료 후)
            ml.update(ml_feat, won=(total_pnl > 0))

            trades_today += 1
            # Track exit time to prevent next window from entering while position open
            try:
                last_exit_time_today = datetime.strptime(exit_time_str, "%H:%M").time()
            except Exception:
                last_exit_time_today = EXIT_TIME

        pbar.set_postfix({"Bal": f"${balance:,.0f}", "WR": f"{wins/(wins+losses)*100 if wins+losses>0 else 0:.0f}%"})

    pbar.close()

    # -- Summary --
    total_trades = wins + losses
    total_pnl = balance - start_balance
    wr = (wins / total_trades * 100) if total_trades > 0 else 0

    # Derive number of years from actual date span
    if trades:
        d0 = datetime.strptime(trades[0]["date"], "%Y-%m-%d")
        d1 = datetime.strptime(trades[-1]["date"], "%Y-%m-%d")
        years = max((d1 - d0).days / 365.25, 0.01)
    else:
        years = 3.0
    annual_ret = ((balance / start_balance) ** (1 / years) - 1) * 100 if balance > 0 else 0

    # Drawdown
    peak = start_balance
    max_dd = 0.0
    dd_series: List[float] = []
    for t in trades:
        if t["balance"] > peak:
            peak = t["balance"]
        dd = (peak - t["balance"]) / peak * 100
        dd_series.append(round(dd, 2))
        if dd > max_dd:
            max_dd = dd

    avg_w = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if wins > 0 else 0.0
    avg_l = np.mean([t["pnl"] for t in trades if t["pnl"] <= 0]) if losses > 0 else 0.0
    gross_wins = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_losses = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = round(gross_wins / gross_losses, 2) if gross_losses > 0 else float("inf")

    # ── Risk-adjusted metrics ──────────────────────────────────────────────
    # Per-trade returns (vs. previous balance)
    trade_returns: List[float] = []
    prev_bal = start_balance
    for t in trades:
        trade_returns.append(t["pnl"] / prev_bal)
        prev_bal = t["balance"]

    # Annualisation factor: average calendar days between trades → daily equivalent
    if len(trades) >= 2:
        total_cal_days = (
            datetime.strptime(trades[-1]["date"], "%Y-%m-%d") -
            datetime.strptime(trades[0]["date"], "%Y-%m-%d")
        ).days
        avg_days_per_trade = total_cal_days / len(trades)
    else:
        avg_days_per_trade = 10.0
    ann_factor = np.sqrt(252.0 / max(avg_days_per_trade, 1))

    rf_per_trade = 0.05 * avg_days_per_trade / 365.25
    excess = [r - rf_per_trade for r in trade_returns]
    ret_mean = float(np.mean(excess)) if excess else 0.0
    ret_std  = float(np.std(excess,  ddof=1)) if len(excess) > 1 else 1e-9
    sharpe = round((ret_mean / ret_std) * ann_factor, 2) if ret_std > 0 else 0.0

    neg_excess = [r for r in excess if r < 0]
    downside_std = float(np.sqrt(np.mean([r ** 2 for r in neg_excess]))) if neg_excess else 1e-9
    sortino = round((ret_mean / downside_std) * ann_factor, 2) if downside_std > 0 else 0.0

    calmar = round(annual_ret / max_dd, 2) if max_dd > 0 else 0.0

    # ── Monthly / yearly breakdown ─────────────────────────────────────────
    yearly_pnl: Dict[str, float] = {}
    for ym, pnl in monthly_pnl.items():
        yr = ym[:4]
        yearly_pnl[yr] = yearly_pnl.get(yr, 0.0) + pnl

    # ── Exit / pro-filter counts ───────────────────────────────────────────
    nr7_trades = sum(1 for t in trades if "NR7" in t.get("boost_reasons", ""))
    pb_trades  = sum(1 for t in trades if "3DAY_PB" in t.get("boost_reasons", ""))
    trail_exits = sum(1 for t in trades if t.get("exit_type") == "TRAIL")
    be_exits    = sum(1 for t in trades if t.get("exit_type") == "BE")
    sl_exits    = sum(1 for t in trades if t.get("exit_type") == "SL")
    eod_exits   = sum(1 for t in trades if t.get("exit_type") == "EOD")
    tp_exits    = sum(1 for t in trades if t.get("exit_type") == "TP")

    long_wins  = sum(1 for t in trades if t["direction"] in ("LONG", "CALL")  and t["pnl"] > 0)
    long_total = sum(1 for t in trades if t["direction"] in ("LONG", "CALL"))
    short_wins = sum(1 for t in trades if t["direction"] in ("SHORT", "PUT") and t["pnl"] > 0)
    short_total = sum(1 for t in trades if t["direction"] in ("SHORT", "PUT"))

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  MICRO E-MINI (MES) - PRO STRATEGY v4 RESULTS")
    print("=" * 80)
    prime_cnt = sum(1 for t in trades if t.get("window") == "PRIME")
    gamma_cnt = sum(1 for t in trades if t.get("window") == "GAMMA")
    strong_cnt = sum(1 for t in trades if t.get("grade") == "STRONG")
    moderate_cnt = sum(1 for t in trades if t.get("grade") == "MODERATE")

    print(f"  Period:            {start_str} ~ {end_str}  ({years:.1f}y)")
    print(f"  Product:           Micro E-mini S&P 500 (MES) [${ES_MULTIPLIER:.0f}/pt]")
    print(f"  Strategy:          ATR SL={ATR_SL_MULT}x + Trail + BE | Risk={RISK_PCT*100:.1f}%")
    print(f"  Entry Windows:     PRIME(10:30)={prime_cnt} | GAMMA(14:00)={gamma_cnt}")
    print(f"  Grade Breakdown:   STRONG(≥88)={strong_cnt} | MODERATE(≥{MIN_SCORE})={moderate_cnt}")
    print(f"  Pro Filters:       NR7 + 3Day Pullback + Gap + Daily Bias")
    print(f"  Starting Balance:  ${start_balance:,.2f}")
    print(f"  Final Balance:     ${balance:,.2f}")
    print(f"  Total P&L:         ${total_pnl:+,.2f} ({total_pnl/start_balance*100:+.1f}%)")
    print(f"  Annual Return:     {annual_ret:+.1f}%")
    print(f"  Total Trades:      {total_trades}  (LONG={long_total}, SHORT={short_total})")
    print(f"  Win Rate:          {wr:.1f}% ({wins}W / {losses}L)")
    print(f"    LONG  WR:        {long_wins/max(long_total,1)*100:.1f}%  ({long_wins}/{long_total})")
    print(f"    SHORT WR:        {short_wins/max(short_total,1)*100:.1f}%  ({short_wins}/{short_total})")
    print(f"  Avg Win:           ${avg_w:+,.2f}")
    print(f"  Avg Loss:          ${avg_l:+,.2f}")
    print(f"  R:R Ratio:         {abs(avg_w/avg_l):.2f}" if avg_l != 0 else "  R:R Ratio:         N/A")
    print(f"  Profit Factor:     {pf}")
    print(f"  Max Drawdown:      {max_dd:.1f}%")
    print(f"  Max Consec. Wins:  {max_consec_wins}")
    print(f"  Max Consec. Loss:  {max_consec_losses}")
    print(f"  ── Risk-Adjusted ──────────────────────────────")
    print(f"  Sharpe Ratio:      {sharpe:.2f}  (annualised, RF=5%)")
    print(f"  Sortino Ratio:     {sortino:.2f}  (downside deviation)")
    print(f"  Calmar Ratio:      {calmar:.2f}  (annual / max-DD)")
    print(f"  ── Exit Types ─────────────────────────────────")
    print(f"  Exit Types:        EOD={eod_exits} | TP={tp_exits} | SL={sl_exits} | TRAIL={trail_exits} | BE={be_exits}")
    print(f"  NR7 Boosted:       {nr7_trades} trades")
    print(f"  3Day PB Boosted:   {pb_trades} trades")
    print(f"  ── Monthly P&L ────────────────────────────────")
    for ym in sorted(monthly_pnl):
        sign = "+" if monthly_pnl[ym] >= 0 else ""
        print(f"    {ym}:  {sign}${monthly_pnl[ym]:,.0f}")
    print(f"  ── Yearly P&L ─────────────────────────────────")
    for yr in sorted(yearly_pnl):
        sign = "+" if yearly_pnl[yr] >= 0 else ""
        print(f"    {yr}:  {sign}${yearly_pnl[yr]:,.0f}")
    print(f"  Running Time:      {time.time()-t_start:.1f}s")
    print("=" * 80)

    sizing_label = f"FIXED {FIXED_CONTRACTS}계약" if fixed_size else f"DYNAMIC Risk={RISK_PCT*100:.1f}%"
    results = {
        "model": f"MES Futures Pro Strategy v10.1 (10:30 PRIME · TP×{TP_MULT} · ATR>8 · Score≥{MIN_SCORE})",
        "period": f"{start_str} ~ {end_str}",
        "product": f"Micro E-mini S&P 500 (MES) [${ES_MULTIPLIER:.0f}/pt]",
        "strategy": f"ATR SL={ATR_SL_MULT}x · TP={TP_MULT}xSL · MinScore={MIN_SCORE} · ML Walk-Forward",
        "fixed_size": fixed_size,
        "fixed_contracts": FIXED_CONTRACTS if fixed_size else None,
        "prime_trades": prime_cnt,
        "gamma_trades": gamma_cnt,
        "strong_trades": strong_cnt,
        "moderate_trades": moderate_cnt,
        "start_balance": start_balance,
        "end_balance": round(balance, 2),
        "total_pnl": round(total_pnl, 2),
        "pnl_pct": round(total_pnl / start_balance * 100, 1),
        "annual_return": round(annual_ret, 1),
        "years": round(years, 2),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 1),
        "long_trades": long_total,
        "long_wins": long_wins,
        "short_trades": short_total,
        "short_wins": short_wins,
        "avg_win": round(avg_w, 2),
        "avg_loss": round(avg_l, 2),
        "rr_ratio": round(abs(avg_w / avg_l), 2) if avg_l != 0 else None,
        "profit_factor": pf,
        "max_drawdown": round(max_dd, 1),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "exit_counts": {"EOD": eod_exits, "TP": tp_exits, "SL": sl_exits, "TRAIL": trail_exits, "BE": be_exits},
        "nr7_boosted_trades": nr7_trades,
        "pullback_boosted_trades": pb_trades,
        "monthly_pnl": {k: round(v, 2) for k, v in sorted(monthly_pnl.items())},
        "yearly_pnl": {k: round(v, 2) for k, v in sorted(yearly_pnl.items())},
        "daily_balance": daily_balance,
        "drawdown_series": [round(d, 2) for d in dd_series],
        "ml_stats": ml.stats(),
        "trades": trades,
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] Saved results to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S&P 500 Futures (ES) Pro Strategy Backtest")
    parser.add_argument("--csv", type=str, default="SPY_1min_synthetic.csv")
    parser.add_argument("--start", type=str, default="2023-03-25")
    parser.add_argument("--end", type=str, default="2026-03-25")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--fixed-size", action="store_true",
                        help="Use fixed contract count (no reinvestment effect)")
    parser.add_argument("--profile", type=str, default="v10", choices=["v10", "v9", "v4"],
                        help="v10 = v4 + TP×2.5 + ATR>8 floor + ML-skip off (default/recommended); "
                             "v9 = wide PRIME+REENTRY+GAMMA; v4 = narrow 10:30 legacy defaults")
    parser.add_argument("--out", type=str, default="backtest_futures.json",
                        help="Output JSON path")
    # ── Tunable strategy parameters ──
    parser.add_argument("--min-score", type=int, default=None,
                        help="Override MIN_SCORE threshold (default: profile default, 88)")
    parser.add_argument("--tp-mult", type=float, default=None,
                        help="Override TP_MULT: TP = N × SL (default 1.5)")
    parser.add_argument("--vix-max", type=float, default=None,
                        help="Skip entry if daily VIX above this value (default: no cap)")
    parser.add_argument("--atr-min", type=float, default=None,
                        help="Skip entry if daily ATR below this value in points (default: no floor)")
    parser.add_argument("--no-mean-reversion", action="store_true",
                        help="Always follow signal direction; disable VIX-triggered mean-reversion mode")
    parser.add_argument("--no-ml-skip", action="store_true",
                        help="Disable ML hard-skip filter (keeps all score-passing trades)")

    args = parser.parse_args()

    # ── profile overrides ──
    if args.profile in ("v4", "v10"):
        ENTRY_WINDOWS = [dtime(10, 30)]   # single PRIME entry only
        MAX_TRADES_PER_DAY = 1            # one high-conviction trade per day
        LOCKOUT_STRIKES = 3               # stricter 3-strike lockout
        LOCKOUT_DAYS = 1                  # 1-day cooldown after lockout
        WalkForwardML.SKIP_AFTER_N = 9999  # disable ML skip — score-purity over filter noise
        WalkForwardML.SKIP_THRESH = 0.43

    if args.profile == "v10":
        # v10.1: TP×2.5 and MIN_SCORE=60 are now the module-level defaults.
        if args.atr_min is None:
            args.atr_min = 8.0
        print(f"[*] PROFILE: v10.1 — 10:30 PRIME · TP×{TP_MULT} · ATR>8 · Score≥{MIN_SCORE}")
    elif args.profile == "v4":
        print("[*] PROFILE: v4 — narrow 10:30 PRIME only, legacy defaults")

    # ── per-run overrides (after profile defaults) ──
    if args.min_score is not None:
        MIN_SCORE = args.min_score
        print(f"[*] OVERRIDE: MIN_SCORE = {MIN_SCORE}")
    if args.tp_mult is not None:
        TP_MULT = args.tp_mult
        print(f"[*] OVERRIDE: TP_MULT = {TP_MULT}")
    else:
        print(f"[*] TP_MULT = {TP_MULT} (global default)")
    if args.no_ml_skip:
        WalkForwardML.SKIP_AFTER_N = 9999
        print("[*] OVERRIDE: ML hard-skip disabled")

    if not os.path.exists(args.csv):
        print(f"ERROR: Could not find CSV file at {args.csv}")
        sys.exit(1)

    run_futures_backtest(
        csv_path=args.csv,
        start_str=args.start,
        end_str=args.end,
        start_balance=args.balance,
        fixed_size=args.fixed_size,
        out_path=args.out,
        vix_max=args.vix_max,
        atr_min=args.atr_min,
        no_mean_reversion=args.no_mean_reversion,
    )
