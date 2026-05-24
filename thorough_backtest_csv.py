#!/usr/bin/env python3
"""
Thorough 0DTE Options Backtest using Local SPY 1-Minute CSV Data & Real Score Engine
Deterministic minute-by-minute simulation with Black-Scholes pricing fallback.
"""
import os
import sys
import math
import json
import time
import argparse
from datetime import datetime, timedelta, time as dtime
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engines.score_engine import run_score_engine
from engines.regime import calculate_regime_score
from engines.correlation import calculate_correlation_score
from engines.time_window import calculate_time_score
from engines.technical import calculate_technical_score

NY = pytz.timezone("America/New_York")

# ── Black-Scholes ────────────────────────────────────────────────
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bs_price(S, K, T, r, sigma, opt="call"):
    if T <= 0: return max(S - K, 0) if opt == "call" else max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def dynamic_slippage(vix, spy_range_pct=0.0):
    """VIX-adaptive slippage per contract side."""
    if vix >= 30 or spy_range_pct >= 2.0:
        return 0.08
    if vix >= 25 or spy_range_pct >= 1.5:
        return 0.06
    if vix >= 20 or spy_range_pct >= 1.0:
        return 0.05
    return 0.03

def load_vix_data():
    """Load yfinance historical VIX data for alignment."""
    import yfinance as yf
    print("[*] Fetching historical VIX data for backtest...")
    try:
        vix_df = yf.download("^VIX", start="2018-01-01", end="2026-05-23", interval="1d", progress=False)
        if not vix_df.empty:
            if isinstance(vix_df.columns, pd.MultiIndex):
                return vix_df["Close"].squeeze()
            return vix_df["Close"]
    except Exception as e:
        print(f"Warning: Could not fetch VIX data ({e}). Defaulting to 18.0 VIX.")
    return pd.Series(dtype=float)

def run_thorough_csv_backtest(csv_path: str, start_str: str = "2024-05-01", end_str: str = "2025-05-01", start_balance: float = 2000.0, invert: bool = True, eod_only: bool = True):
    t_start = time.time()
    print("=" * 80)
    print("  SPY 0DTE DETERMINISTIC MINUTE-BY-MINUTE BACKTEST (LOCAL CSV)")
    print(f"  Configuration: Invert Direction = {invert} | EOD Exits Only = {eod_only}")
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
    
    balance = start_balance
    trades = []
    wins, losses = 0, 0
    r_rate = 0.05
    consecutive_losses = 0
    lockout_cooldown = 0
    
    # SPREAD PARAMETERS
    SPREAD_WIDTH = 5.0
    TP_PCT = 1.0  # +100% Take Profit
    SPREAD_PCT = 0.03 # 3% bid-ask spread friction
    MIN_SCORE = 90
    
    pbar = tqdm(trading_days, desc="Backtesting Days")
    
    for day_str in pbar:
        # Check consecutive losses lockout (Applying original 3-Strike Lockout: cool-down for 3 trading days)
        if lockout_cooldown > 0:
            lockout_cooldown -= 1
            continue
        day_bars = days_dict[day_str]
        if len(day_bars) < 60: # need sufficient intraday bars
            continue
            
        df_day = pd.DataFrame(day_bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]).set_index("timestamp")
        
        spy_o = float(df_day["Open"].iloc[0])
        
        # Get VIX
        try:
            vix_val = float(vix_series.loc[day_str])
        except:
            vix_val = 18.0
            
        # Run Score Engine at 10:30 AM
        entry_time = dtime(10, 30)
        entry_bar = None
        for ts, o, h, l, c, v in day_bars:
            if ts.time() >= entry_time:
                entry_bar = (ts, o, h, l, c, v)
                break
                
        if not entry_bar:
            continue
            
        ts_entry, spy_entry_o, _, _, _, _ = entry_bar
        
        # Slice morning bars up to 10:30 AM
        df_morning = df_day[df_day.index.time <= entry_time].copy()
        df_morning.columns = [col.capitalize() for col in df_morning.columns] # capitalize for engines

        # Sector returns approximation strictly based on morning return
        spy_morning_ret = ((spy_entry_o / spy_o) - 1.0) * 100
        pcts = {
            "SPY": spy_morning_ret,
            "QQQ": spy_morning_ret * 1.2 if spy_morning_ret >= 0 else spy_morning_ret * 1.3,
            "IWM": spy_morning_ret * 0.9,
            "DIA": spy_morning_ret * 0.8
        }
        
        # Calculate morning technical metrics strictly up to 10:30 AM
        vwap_morning = (df_morning["High"] * df_morning["Volume"]).sum() / df_morning["Volume"].sum() if df_morning["Volume"].sum() > 0 else spy_entry_o
        range_morning = float(df_morning["High"].max() - df_morning["Low"].min())
        vol_ratio = 1.6 # Good volume ratio for momentum
        
        # Check scores using 10:30 AM NY time context
        try:
            regime = calculate_regime_score(
                vix_price=vix_val,
                vix3m_price=vix_val * 1.08, # VIX3M is higher than VIX during Contango
                spy_price=spy_entry_o,
                prev_close=spy_o,
                spy_history=df_morning
            )
            corr = calculate_correlation_score(pcts)
            time_win = calculate_time_score(ts_entry)
            tech = calculate_technical_score(spy_entry_o, vwap_morning, vol_ratio, range_morning, df_morning)
            
            active_scores = [regime["score"], corr["score"], time_win["score"], tech["score"]]
            total_score = sum(active_scores)
            active_max = regime["max"] + corr["max"] + time_win["max"] + tech["max"]
            normalized = int((total_score / active_max) * 100) if active_max > 0 else 0
            
            grade = "STRONG" if normalized >= MIN_SCORE else "MODERATE" if normalized >= 75 else "WEAK" if normalized >= 60 else "NONE"
            direction = tech.get("direction_bias", "NEUTRAL")
        except Exception:
            continue
            
        # ── RUNAWAY TREND VETO FILTER ───────────────────────────────
        is_runaway_trend = False
        
        # 1. ADX Runaway
        adx_val = regime.get("details", {}).get("adx", {}).get("value")
        if adx_val is not None and adx_val >= 35.0:
            is_runaway_trend = True
            
        # 2. RSI Runaway
        rsi_val = tech.get("rsi")
        if rsi_val is not None and (rsi_val >= 80.0 or rsi_val <= 20.0):
            is_runaway_trend = True
            
        # 3. Synchronized Sector Breakout
        spy_ret = pcts.get("SPY", 0.0)
        qqq_ret = pcts.get("QQQ", 0.0)
        iwm_ret = pcts.get("IWM", 0.0)
        if (spy_ret > 1.2 and qqq_ret > 1.2 and iwm_ret > 1.2) or (spy_ret < -1.2 and qqq_ret < -1.2 and iwm_ret < -1.2):
            is_runaway_trend = True
            
        # Entry Filters
        if normalized < MIN_SCORE or grade != "STRONG" or direction not in ("CALL", "PUT") or is_runaway_trend:
            continue
            
        # ── ENTER TRADE SIMULATION ──
        # Handle option direction inversion (counter-trend fading)
        if invert:
            opt = "put" if direction == "CALL" else "call"
        else:
            opt = "call" if direction == "CALL" else "put"
            
        iv = vix_val / 100.0
        K_buy = round(spy_entry_o)
        K_sell = K_buy + SPREAD_WIDTH if opt == "call" else K_buy - SPREAD_WIDTH
        
        # Entry time remaining = 5.5 hours to 4:00 PM (330 minutes)
        T_entry = 330.0 / (252.0 * 390.0)
        
        long_entry = bs_price(spy_entry_o, K_buy, T_entry, r_rate, iv, opt)
        short_entry = bs_price(spy_entry_o, K_sell, T_entry, r_rate, iv, opt)
        
        # Add bid-ask friction + dynamic slippage
        spy_range_pct = (range_morning / spy_o) * 100 if spy_o > 0 else 0
        slip = dynamic_slippage(vix_val, spy_range_pct)
        net_debit = (long_entry - short_entry) * (1 + SPREAD_PCT) + slip * 2
        if net_debit <= 0.05:
            continue
        tp_price = net_debit * (1 + TP_PCT)
        
        # Sizing (Applying original Layer 7 Risk Manager: strictly 2.0% max trade loss)
        max_risk = balance * 0.02
        num_contracts = int(max_risk / (net_debit * 100))
        if num_contracts == 0:
            # Allow 1 contract for small accounts if it doesn't exceed 10% of total balance
            if net_debit * 100 <= balance * 0.10:
                num_contracts = 1
            else:
                continue
        
        # ── MINUTE-BY-MINUTE SIMULATION ──
        exit_val = None
        exit_type = "EOD"
        exit_time_str = "16:00"
        
        if not eod_only:
            # Loop through each minute bar starting from entry time (10:30 AM) to 4:00 PM
            for ts_bar, o_bar, h_bar, l_bar, c_bar, v_bar in day_bars:
                if ts_bar.time() <= entry_time:
                    continue
                if ts_bar.time() > dtime(16, 0):
                    break
                    
                # Remaining time at this specific bar
                minutes_to_close = max(1.0, (datetime.combine(ts_bar.date(), dtime(16,0)).replace(tzinfo=NY) - ts_bar).total_seconds() / 60.0)
                T_rem = minutes_to_close / (252.0 * 390.0)
                
                # Check worst price for SL and best price for TP
                if opt == "call":
                    best_underlying = h_bar
                else:
                    best_underlying = l_bar
                    
                # Calculate option spread values
                best_opt_val = max(bs_price(best_underlying, K_buy, T_rem, r_rate, iv, opt) - bs_price(best_underlying, K_sell, T_rem, r_rate, iv, opt), 0.0)
                best_exit = best_opt_val * (1 - SPREAD_PCT)
                
                # Check exit conditions
                if best_exit >= tp_price:
                    exit_val = tp_price - slip
                    exit_type = "TP"
                    exit_time_str = ts_bar.strftime("%H:%M")
                    break
                    
        # EOD Close fallback
        if exit_val is None:
            spy_eod = float(df_day["Close"].iloc[-1])
            eod_opt_val = max(bs_price(spy_eod, K_buy, 1e-6, r_rate, iv, opt) - bs_price(spy_eod, K_sell, 1e-6, r_rate, iv, opt), 0.0)
            exit_val = max(eod_opt_val * (1 - SPREAD_PCT) - slip, 0.0)
            exit_type = "EOD"
            exit_time_str = "16:00"
            
        # P&L calculation
        total_pnl = (exit_val - net_debit) * 100 * num_contracts
        # Cap loss to premium paid at entry (no negative account balance possible beyond premium)
        total_pnl = max(total_pnl, -net_debit * 100 * num_contracts)
        
        balance += total_pnl
        if total_pnl > 0:
            wins += 1
            consecutive_losses = 0
        else:
            losses += 1
            consecutive_losses += 1
            
            # Apply consecutive loss lockout (3 strikes)
            if consecutive_losses >= 3:
                lockout_cooldown = 3
                consecutive_losses = 0
                
            # Apply daily loss limit (MAX_DAILY_LOSS_PCT = 6.0%)
            # If today's loss exceeds 6% of the balance prior to this trade, lock trading for 3 days
            prev_balance = balance - total_pnl
            if prev_balance > 0 and abs(total_pnl) / prev_balance >= 0.06:
                lockout_cooldown = 3
            
        trades.append({
            "date": day_str,
            "score": normalized,
            "orig_direction": direction,
            "trade_direction": opt.upper(),
            "K_buy": K_buy,
            "K_sell": K_sell,
            "net_debit": round(net_debit, 2),
            "exit_val": round(exit_val, 2),
            "exit_type": exit_type,
            "exit_time": exit_time_str,
            "contracts": num_contracts,
            "pnl": round(total_pnl, 2),
            "balance": round(balance, 2),
            "vix": round(vix_val, 1)
        })
        
        pbar.set_postfix({"Balance": f"${balance:,.2f}", "WR": f"{wins/(wins+losses)*100 if wins+losses>0 else 0:.1f}%"})
        
    pbar.close()
    
    # ── Summary ──
    total_trades = wins + losses
    total_pnl = balance - start_balance
    wr = (wins / total_trades * 100) if total_trades > 0 else 0
    
    peak = start_balance
    max_dd = 0
    for t in trades:
        if t["balance"] > peak: peak = t["balance"]
        dd = (peak - t["balance"]) / peak * 100
        if dd > max_dd: max_dd = dd
        
    avg_w = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if wins > 0 else 0
    avg_l = np.mean([t["pnl"] for t in trades if t["pnl"] <= 0]) if losses > 0 else 0
    pf = round(sum(t["pnl"] for t in trades if t["pnl"] > 0) / abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0)), 2) if losses > 0 else float("inf")
    
    print("\n" + "=" * 80)
    print("  CSV 1-MINUTE BAR THOROUGH BACKTEST RESULTS")
    print("=" * 80)
    print(f"  Period:            {start_str} ~ {end_str}")
    print(f"  Invert Direction:  {invert}")
    print(f"  EOD Only Exits:    {eod_only}")
    print(f"  Starting Balance:  ${start_balance:,.2f}")
    print(f"  Final Balance:     ${balance:,.2f}")
    print(f"  Total P&L:         ${total_pnl:+,.2f} ({total_pnl/start_balance*100:+.1f}%)")
    print(f"  Total Trades:      {total_trades}")
    print(f"  Win Rate:          {wr:.1f}% ({wins}W / {losses}L)")
    print(f"  Avg Win:           ${avg_w:+,.2f}")
    print(f"  Avg Loss:          ${avg_l:+,.2f}")
    print(f"  Profit Factor:     {pf}")
    print(f"  Max Drawdown:      {max_dd:.1f}%")
    print(f"  Running Time:      {time.time()-t_start:.1f}s")
    print("=" * 80)
    
    results = {
        "model": "CSV 1-Min Deterministic v4",
        "period": f"{start_str} ~ {end_str}",
        "invert": invert,
        "eod_only": eod_only,
        "start_balance": start_balance,
        "end_balance": round(balance, 2),
        "total_pnl": round(total_pnl, 2),
        "pnl_pct": round(total_pnl / start_balance * 100, 1),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 1),
        "avg_win": round(avg_w, 2),
        "avg_loss": round(avg_l, 2),
        "profit_factor": pf,
        "max_drawdown": round(max_dd, 1),
        "trades": trades
    }
    
    with open("backtest_v4.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[*] Saved results to backtest_v4.json")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic 0DTE Options CSV Backtest Tool")
    parser.add_argument("--csv", type=str, default="C:/Users/Gun_y/Desktop/SPY_1min_data.csv", help="Path to local 1-minute CSV data")
    parser.add_argument("--start", type=str, default="2024-05-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2025-05-01", help="End date YYYY-MM-DD")
    parser.add_argument("--balance", type=float, default=2000.0, help="Initial balance")
    parser.add_argument("--no-invert", action="store_true", help="Do not invert direction (run standard direction)")
    parser.add_argument("--no-eod-only", action="store_true", help="Enable intraday TP checks (default EOD only)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"ERROR: Could not find CSV file at {args.csv}")
        sys.exit(1)
        
    run_thorough_csv_backtest(
        csv_path=args.csv,
        start_str=args.start,
        end_str=args.end,
        start_balance=args.balance,
        invert=not args.no_invert,
        eod_only=not args.no_eod_only
    )
