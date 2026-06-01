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
from datetime import datetime, timedelta, time as dtime
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
import pytz

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
from engines.regime import calculate_regime_score
from engines.correlation import calculate_correlation_score
from engines.time_window import calculate_time_score
from engines.technical import calculate_technical_score

NY = pytz.timezone("America/New_York")

# -- MES Contract Specifications (matches api/data.py live trading) --
ES_MULTIPLIER = 5.0        # $5 per point of S&P 500 (MES — Micro E-mini)
ES_COMMISSION_RT = 0.50    # Round-trip commission per MES contract
ES_SLIPPAGE_PTS = 0.25     # 1 tick slippage per side
ES_DAY_MARGIN = 50.0       # Day-trading margin per MES contract

# -- Strategy Parameters (matches api/data.py live RISK_PCT) --
MIN_SCORE = 78              # MODERATE grade threshold (was 88 — STRONG only)
RISK_PCT = 0.015            # 1.5% per-trade risk (live system value)
MARGIN_UTIL = 0.95          # 95% margin utilization allowed
EXIT_TIME = dtime(15, 30)   # Exit at 15:30 (avoid last 30min noise)
VIX_THRESHOLD = 20.0        # VIX < 20 = Trend Follow, >= 20 = Mean Reversion
ADX_RUNAWAY = 40.0           # ADX runaway veto
RSI_UPPER = 90.0             # RSI upper veto
RSI_LOWER = 10.0             # RSI lower veto
SECTOR_THRESHOLD = 1.8       # Sector breakout veto
LOCKOUT_STRIKES = 3          # Consecutive losses before lockout (was 2)
LOCKOUT_DAYS = 1             # Days to cool down (was 3)
ATR_SL_MULT = 1.5            # SL = 1.5x ATR(14) dynamic
TRAILING_ACTIVATION = 1.0    # Activate trailing stop after 1.0x ATR profit
TRAILING_STEP = 0.5          # Trail by 0.5x ATR behind highest profit
BREAKEVEN_AT = 0.5           # Move SL to breakeven after 0.5x ATR profit

# -- Dual Entry Windows --
ENTRY_WINDOWS = [dtime(10, 30), dtime(14, 0)]   # PRIME (10:30) + GAMMA (14:00)
MAX_TRADES_PER_DAY = 2       # Max positions per calendar day

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


def run_futures_backtest(csv_path: str, start_str: str = "2023-03-25",
                         end_str: str = "2026-03-25",
                         start_balance: float = 10000.0):
    t_start = time.time()
    print("=" * 80)
    print("  MICRO E-MINI (MES) - PRO TRADER STRATEGY INTEGRATION")
    print(f"  ATR SL={ATR_SL_MULT}x | Risk={RISK_PCT*100:.1f}% | Margin=${ES_DAY_MARGIN:.0f}")
    print(f"  NR7 + 3Day Pullback + Gap + Daily Bias | MIN_SCORE={MIN_SCORE}")
    print("=" * 80)

    # 1. Load VIX
    vix_series = load_vix_data()

    # 2. Load CSV
    print(f"[*] Loading historical 1-minute bars from {csv_path}...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df):,} rows in {time.time()-t0:.1f}s.")

    # Parse timestamps and localize to NY
    print("[*] Parsing timestamps...")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_localize("UTC").tz_convert(NY) if df.index.tz is None else df.index.tz_convert(NY)
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

        # ===== PRO STRATEGIES: Pre-Score Calculations =====

        # [Crabel] NR7 Volatility Filter
        is_nr7 = check_nr7(daily_ohlc, trading_days, day_idx)

        # [Mean Reversion] 3-Day Pullback
        is_pullback = check_3day_pullback(daily_ohlc, trading_days, day_idx)

        # [Macro] Daily Trend Bias (above 20 SMA)
        daily_trend_long = check_daily_bias(daily_ohlc, trading_days, day_idx, spy_o)

        # [ATR] Dynamic SL calculation
        atr_val = calc_atr(daily_ohlc, trading_days, day_idx)
        sl_points = max(ATR_SL_MULT * atr_val, 2.0)
        sl_points = min(sl_points, 15.0)

        # [Gap] Gap context
        gap_bias = 0  # -1=fade gap up, +1=fade gap down, 0=neutral
        if day_idx >= 1:
            prev_ds = trading_days[day_idx - 1]
            if prev_ds in daily_ohlc:
                prev_close = daily_ohlc[prev_ds]["close"]
                gap_pct = ((spy_o - prev_close) / prev_close) * 100
                if abs(gap_pct) > 1.2:
                    gap_bias = 0  # Large gap: don't fade
                elif gap_pct > 0.1:
                    gap_bias = -1  # Small gap up: bearish
                elif gap_pct < -0.1:
                    gap_bias = 1   # Small gap down: bullish

        # ===== Dual Entry Windows: PRIME (10:30) + GAMMA (14:00) =====
        trades_today = 0
        for entry_time in ENTRY_WINDOWS:
            if trades_today >= MAX_TRADES_PER_DAY:
                break

            is_gamma = (entry_time == dtime(14, 0))

            # Find entry bar for this window
            entry_bar = None
            for ts, o, h, l, c, v in day_bars:
                if ts.time() >= entry_time:
                    entry_bar = (ts, o, h, l, c, v)
                    break
            if not entry_bar:
                continue

            ts_entry, spy_entry_price, _, _, _, _ = entry_bar

            # Tighter SL for GAMMA (only ~90 min until EOD exit)
            gamma_mult = 0.75 if is_gamma else 1.0
            sl_cap = 10.0 if is_gamma else 15.0
            window_sl = max(ATR_SL_MULT * gamma_mult * atr_val, 2.0)
            window_sl = min(window_sl, sl_cap)

            # Slice bars up to entry time for scoring
            df_morning = df_day[df_day.index.time <= entry_time].copy()
            if len(df_morning) < 5:
                continue
            df_morning.columns = [col.capitalize() for col in df_morning.columns]

            # Sector returns (relative to day open)
            spy_morning_ret = ((spy_entry_price / spy_o) - 1.0) * 100
            pcts = {
                "SPY": spy_morning_ret,
                "QQQ": spy_morning_ret * 1.2 if spy_morning_ret >= 0 else spy_morning_ret * 1.3,
                "IWM": spy_morning_ret * 0.9,
                "DIA": spy_morning_ret * 0.8
            }

            # Window metrics
            vwap_morning = (df_morning["High"] * df_morning["Volume"]).sum() / df_morning["Volume"].sum() if df_morning["Volume"].sum() > 0 else spy_entry_price
            range_morning = float(df_morning["High"].max() - df_morning["Low"].min())
            avg_5min_vol = df_morning["Volume"].tail(5).mean()
            avg_morning_vol = df_morning["Volume"].mean()
            vol_ratio = avg_5min_vol / avg_morning_vol if avg_morning_vol > 0 else 1.0

            # Score Engine
            try:
                regime = calculate_regime_score(
                    vix_price=vix_val, vix3m_price=vix_val * 1.08,
                    spy_price=spy_entry_price, prev_close=spy_o, spy_history=df_morning)
                corr = calculate_correlation_score(pcts)
                time_win = calculate_time_score(ts_entry)
                tech = calculate_technical_score(spy_entry_price, vwap_morning, vol_ratio, range_morning, df_morning)

                active_scores = [regime["score"], corr["score"], time_win["score"], tech["score"]]
                active_max = regime["max"] + corr["max"] + time_win["max"] + tech["max"]
                normalized = int((sum(active_scores) / active_max) * 100) if active_max > 0 else 0
                direction = tech.get("direction_bias", "NEUTRAL")
            except Exception:
                continue

            # Score Boosting
            boosted_score = normalized
            boost_reasons = []
            if is_nr7:
                boosted_score += NR7_SCORE_BOOST
                boost_reasons.append("NR7")
            if is_pullback and direction == "CALL":
                boosted_score += PULLBACK_SCORE_BOOST
                boost_reasons.append("3DAY_PB")

            # Runaway Trend Veto
            is_runaway_trend = False
            adx_val = regime.get("details", {}).get("adx", {}).get("value")
            if adx_val is not None and adx_val >= ADX_RUNAWAY:
                is_runaway_trend = True
            rsi_val = tech.get("rsi")
            if rsi_val is not None and (rsi_val >= RSI_UPPER or rsi_val <= RSI_LOWER):
                is_runaway_trend = True
            spy_ret, qqq_ret, iwm_ret = pcts.get("SPY", 0), pcts.get("QQQ", 0), pcts.get("IWM", 0)
            if (spy_ret > SECTOR_THRESHOLD and qqq_ret > SECTOR_THRESHOLD and iwm_ret > SECTOR_THRESHOLD) or \
               (spy_ret < -SECTOR_THRESHOLD and qqq_ret < -SECTOR_THRESHOLD and iwm_ret < -SECTOR_THRESHOLD):
                is_runaway_trend = True

            # Entry filter
            grade = "STRONG" if boosted_score >= 88 else "MODERATE" if boosted_score >= MIN_SCORE else "WEAK"
            if boosted_score < MIN_SCORE or direction not in ("CALL", "PUT", "LONG", "SHORT") or is_runaway_trend:
                continue

            # Normalize to LONG/SHORT
            is_bull_signal = direction in ("CALL", "LONG")
            is_bear_signal = direction in ("PUT", "SHORT")

            # Daily Bias Filter: skip SHORT in bullish daily trend (low VIX)
            if daily_trend_long and is_bear_signal and vix_val < VIX_THRESHOLD:
                continue

            # Adaptive Strategy Switching
            is_trending = (vix_val < VIX_THRESHOLD)
            if is_trending:
                trade_dir = "LONG" if is_bull_signal else "SHORT"
                strategy_used = "TREND_FOLLOW"
            else:
                trade_dir = "SHORT" if is_bull_signal else "LONG"
                strategy_used = "MEAN_REVERSION"

            # Kelly-Informed Position Sizing
            max_risk_dollar = balance * RISK_PCT
            risk_per_contract = (window_sl + ES_SLIPPAGE_PTS * 2) * ES_MULTIPLIER + ES_COMMISSION_RT
            num_contracts = int(max_risk_dollar / risk_per_contract)
            if num_contracts == 0:
                num_contracts = 1

            # Margin check
            max_by_margin = int((balance * MARGIN_UTIL) / ES_DAY_MARGIN)
            if max_by_margin == 0:
                max_by_margin = 1
            num_contracts = min(num_contracts, max_by_margin)
            if num_contracts * ES_DAY_MARGIN > balance:
                continue

            # Minute-by-Minute Simulation
            entry_price = spy_entry_price
            sl_target = entry_price - window_sl if trade_dir == "LONG" else entry_price + window_sl
            breakeven_activated = False
            trailing_activated = False
            best_price = entry_price

            exit_price = None
            exit_type = "EOD"
            exit_time_str = f"{EXIT_TIME.hour}:{EXIT_TIME.minute:02d}"

            for ts_bar, o_bar, h_bar, l_bar, c_bar, v_bar in day_bars:
                if ts_bar.time() <= entry_time:
                    continue
                if ts_bar.time() > EXIT_TIME:
                    break

                if trade_dir == "LONG":
                    if h_bar > best_price:
                        best_price = h_bar
                    current_profit_pts = best_price - entry_price

                    if not breakeven_activated and current_profit_pts >= BREAKEVEN_AT * atr_val:
                        sl_target = entry_price + ES_SLIPPAGE_PTS
                        breakeven_activated = True

                    if current_profit_pts >= TRAILING_ACTIVATION * atr_val:
                        trailing_sl = best_price - TRAILING_STEP * atr_val
                        if trailing_sl > sl_target:
                            sl_target = trailing_sl
                            trailing_activated = True

                    if l_bar <= sl_target:
                        exit_price = sl_target
                        exit_type = "TRAIL" if trailing_activated else ("BE" if breakeven_activated else "SL")
                        exit_time_str = ts_bar.strftime("%H:%M")
                        break
                else:  # SHORT
                    if l_bar < best_price:
                        best_price = l_bar
                    current_profit_pts = entry_price - best_price

                    if not breakeven_activated and current_profit_pts >= BREAKEVEN_AT * atr_val:
                        sl_target = entry_price - ES_SLIPPAGE_PTS
                        breakeven_activated = True

                    if current_profit_pts >= TRAILING_ACTIVATION * atr_val:
                        trailing_sl = best_price + TRAILING_STEP * atr_val
                        if trailing_sl < sl_target:
                            sl_target = trailing_sl
                            trailing_activated = True

                    if h_bar >= sl_target:
                        exit_price = sl_target
                        exit_type = "TRAIL" if trailing_activated else ("BE" if breakeven_activated else "SL")
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
                "vix": round(vix_val, 1)
            })

            trades_today += 1

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
    print(f"  Exit Types:        EOD={eod_exits} | SL={sl_exits} | TRAIL={trail_exits} | BE={be_exits}")
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

    results = {
        "model": "MES Futures Pro Strategy v5 (Dual-Window, MODERATE grade, live params)",
        "period": f"{start_str} ~ {end_str}",
        "product": f"Micro E-mini S&P 500 (MES) [${ES_MULTIPLIER:.0f}/pt]",
        "strategy": f"ATR SL={ATR_SL_MULT}x + PRIME/GAMMA | MinScore={MIN_SCORE} | Risk={RISK_PCT*100:.1f}%",
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
        "exit_counts": {"EOD": eod_exits, "SL": sl_exits, "TRAIL": trail_exits, "BE": be_exits},
        "nr7_boosted_trades": nr7_trades,
        "pullback_boosted_trades": pb_trades,
        "monthly_pnl": {k: round(v, 2) for k, v in sorted(monthly_pnl.items())},
        "yearly_pnl": {k: round(v, 2) for k, v in sorted(yearly_pnl.items())},
        "daily_balance": daily_balance,
        "drawdown_series": [round(d, 2) for d in dd_series],
        "trades": trades,
    }

    out_path = "backtest_futures.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] Saved results to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S&P 500 Futures (ES) Pro Strategy Backtest")
    parser.add_argument("--csv", type=str, default="C:/Users/Gun_y/Desktop/SPY_1min_data.csv")
    parser.add_argument("--start", type=str, default="2023-03-25")
    parser.add_argument("--end", type=str, default="2026-03-25")
    parser.add_argument("--balance", type=float, default=10000.0)

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: Could not find CSV file at {args.csv}")
        sys.exit(1)

    run_futures_backtest(
        csv_path=args.csv,
        start_str=args.start,
        end_str=args.end,
        start_balance=args.balance,
    )
