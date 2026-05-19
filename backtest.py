"""
SPY 0DTE Signal Machine — Tuned Backtest v3
Debit spreads + No SL (capped risk) + STRONG only + VIX sizing
"""
import math, json, sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
import yfinance as yf
from scipy.stats import norm

NY = pytz.timezone("America/New_York")

# ── Black-Scholes ────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, opt="call"):
    if T <= 0: return max(S - K, 0) if opt == "call" else max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ── Indicators ───────────────────────────────────────────────────

def calc_rsi(series, period=14):
    d = series.diff()
    g = d.where(d > 0, 0.0).rolling(period).mean()
    l = (-d.where(d < 0, 0.0)).rolling(period).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def calc_adx(df, period=14):
    if len(df) < period + 1: return pd.Series(dtype=float)
    h, l, c = df["High"], df["Low"], df["Close"]
    pm = h.diff().where((h.diff() > l.diff().abs()) & (h.diff() > 0), 0.0)
    mm = l.diff().abs().where((l.diff().abs() > h.diff()) & (l.diff().abs() > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    pdi = 100 * (pm.rolling(period).mean() / atr)
    mdi = 100 * (mm.rolling(period).mean() / atr)
    dx = (pdi - mdi).abs() / (pdi + mdi) * 100
    return dx.rolling(period).mean()

# ── Scoring Engine ───────────────────────────────────────────────

def score_day(row, vix, qqq_pct, iwm_pct, adx, rsi):
    # Regime
    vix_sc = 15 if 14 <= vix <= 20 else (0 if vix <= 30 else -20) if vix > 20 else -5
    adx_sc = 15 if adx and adx >= 25 else (5 if adx and adx >= 20 else 0)
    gap = ((row["Open"] / row["PrevClose"]) - 1) * 100 if row.get("PrevClose") else 0
    regime = vix_sc + adx_sc + (5 if abs(gap) > 0.5 else 0)

    # Correlation
    sp = row.get("PctChange", 0)
    qa = (sp >= 0 and qqq_pct >= 0) or (sp < 0 and qqq_pct < 0)
    ss = all(v >= 0 for v in [sp, qqq_pct, iwm_pct]) or all(v < 0 for v in [sp, qqq_pct, iwm_pct])
    corr = max(0, min(20, (10 if qa else -5) + (5 if iwm_pct > 0.3 else (-3 if iwm_pct < -0.3 else 0)) + (5 if ss else 0)))

    # Technical
    vwap = row.get("VWAP", row["Close"])
    vr = row.get("VolRatio", 0)
    dr = row["High"] - row["Low"]
    d = "CALL" if row["Open"] > vwap else "PUT"
    vol_sc = 10 if vr >= 2.0 else (7 if vr >= 1.5 else (3 if vr >= 1.0 else 0))
    rng_sc = 10 if dr >= 3.0 else (5 if dr >= 2.0 else 0)
    rsi_sc = 10 if rsi and (rsi >= 60 or rsi <= 40) else 0
    tech = min(30, 10 + vol_sc + rng_sc + rsi_sc)

    raw = regime + corr + 20 + tech  # 20 = prime window
    norm_score = max(0, int((raw / 110) * 100))
    grade = "STRONG" if norm_score >= 90 else "MODERATE" if norm_score >= 75 else "WEAK" if norm_score >= 60 else "NONE"
    return norm_score, grade, d


def run_backtest(days=30, balance=2000.0):
    print("=" * 80)
    print("  SPY 0DTE BACKTEST v3 — DEBIT SPREADS + VIX SIZING + SMART ENTRY")
    print("=" * 80)

    end = datetime.now(NY)
    start = end - timedelta(days=int(days * 2))

    print(f"\n[*] Fetching data...")
    spy_d = yf.Ticker("SPY").history(start=start, end=end, interval="1d")
    qqq_d = yf.Ticker("QQQ").history(start=start, end=end, interval="1d")
    iwm_d = yf.Ticker("IWM").history(start=start, end=end, interval="1d")
    vix_d = yf.Ticker("^VIX").history(start=start, end=end, interval="1d")

    if spy_d.empty: print("ERROR: No data"); return

    spy_d["PrevClose"] = spy_d["Close"].shift(1)
    spy_d["PctChange"] = spy_d["Close"].pct_change() * 100
    spy_d["RSI"] = calc_rsi(spy_d["Close"])
    spy_d["ADX"] = calc_adx(spy_d)
    spy_d["VWAP"] = (spy_d["Volume"] * (spy_d["High"] + spy_d["Low"] + spy_d["Close"]) / 3).cumsum() / spy_d["Volume"].cumsum()
    spy_d["VolRatio"] = spy_d["Volume"] / spy_d["Volume"].rolling(20).mean()

    qqq_pcts = qqq_d["Close"].pct_change() * 100
    iwm_pcts = iwm_d["Close"].pct_change() * 100

    dates = spy_d.index[-days:]
    r = 0.05
    SPREAD_PCT = 0.03   # bid-ask spread
    SLIPPAGE = 0.02
    TP_PCT = 0.80       # +80% take profit
    SL_PCT = 1.00       # no SL — max loss = debit paid (spread is capped)
    SPREAD_WIDTH = 3    # $3 wide debit spread (buy ATM, sell OTM+3)
    MIN_VIX = 15.0
    MIN_SCORE = 90      # STRONG signals only

    trades = []
    wins, losses = 0, 0
    initial_balance = balance

    print(f"[*] {len(dates)} trading days | Debit spread width: ${SPREAD_WIDTH}\n")
    hdr = f"{'Date':<11} {'Sc':>3} {'G':<2} {'Dir':<4} {'K':>5}/{'>5'} {'Debit':>6} {'Exit':>6} {'Move':>7} {'P&L':>7} {'Ex':>3} {'Bal':>10}"
    print(hdr)
    print("-" * len(hdr) + "-" * 10)

    for date in dates:
        ds = date.strftime("%m/%d")
        try:
            row = spy_d.loc[date]
            spy_o, spy_c = float(row["Open"]), float(row["Close"])
            spy_h, spy_l = float(row["High"]), float(row["Low"])
            prev_c = float(row["PrevClose"]) if pd.notna(row["PrevClose"]) else spy_o
        except: continue

        # VIX — align by finding closest date
        vix_val = 18.0
        try:
            # Try exact match first, then nearest
            if date in vix_d.index:
                vix_val = float(vix_d.loc[date]["Close"])
            else:
                # Find nearest date within 3 days
                for offset in range(4):
                    check = date - timedelta(days=offset)
                    if check in vix_d.index:
                        vix_val = float(vix_d.loc[check]["Close"])
                        break
        except: pass

        try: qqq_p = float(qqq_pcts.loc[date])
        except: qqq_p = 0
        try: iwm_p = float(iwm_pcts.loc[date])
        except: iwm_p = 0

        adx_v = float(row["ADX"]) if pd.notna(row.get("ADX", np.nan)) else None
        rsi_v = float(row["RSI"]) if pd.notna(row.get("RSI", np.nan)) else None

        row_dict = {
            "Open": spy_o, "Close": spy_c, "High": spy_h, "Low": spy_l,
            "PrevClose": prev_c, "PctChange": float(row["PctChange"]) if pd.notna(row["PctChange"]) else 0,
            "VWAP": float(row["VWAP"]) if pd.notna(row["VWAP"]) else spy_o,
            "VolRatio": float(row["VolRatio"]) if pd.notna(row["VolRatio"]) else 0,
        }

        score, grade, direction = score_day(row_dict, vix_val, qqq_p, iwm_p, adx_v, rsi_v)

        # Entry filter: score >= 85 and VIX >= 15
        if score < MIN_SCORE or grade != "STRONG":
            print(f"{ds:<11} {score:>3} {'X':<2} {grade:<4} {'':>5} {'':>5} {'':>6} {'':>6} {'':>7} {'SKIP':>7} {'':>3} ${balance:>9,.0f}")
            continue
        if vix_val < MIN_VIX:
            print(f"{ds:<11} {score:>3} {'V':<2} {grade:<4} {'':>5} {'':>5} {'':>6} {'':>6} {'':>7} {'LOWV':>7} {'':>3} ${balance:>9,.0f}")
            continue

        # ── DEBIT SPREAD SIMULATION ──
        opt = "call" if direction == "CALL" else "put"
        iv = vix_val / 100.0

        # Strikes: buy ATM, sell OTM
        K_buy = round(spy_o)  # ATM
        if opt == "call":
            K_sell = K_buy + SPREAD_WIDTH  # sell higher strike call
        else:
            K_sell = K_buy - SPREAD_WIDTH  # sell lower strike put

        T_entry = 5.5 / (252 * 6.5)  # 10:30 AM
        T_mid = 3.0 / (252 * 6.5)    # 1:00 PM
        T_exit = 1.0 / (252 * 6.5)   # 3:30 PM

        # Entry: debit = long premium - short premium
        long_entry = bs_price(spy_o, K_buy, T_entry, r, iv, opt)
        short_entry = bs_price(spy_o, K_sell, T_entry, r, iv, opt)
        net_debit = (long_entry - short_entry) * (1 + SPREAD_PCT) + SLIPPAGE * 2
        if net_debit <= 0.05: continue

        # Max profit = spread width - debit (for in-spread moves)
        max_profit_per = SPREAD_WIDTH - net_debit
        # Max loss = net debit paid

        # ── Dynamic VIX-based sizing ──
        # Higher VIX = more confident sizing (more premium to work with)
        if vix_val >= 25:
            risk_pct = 0.08  # 8% risk in high-vol
        elif vix_val >= 20:
            risk_pct = 0.06
        else:
            risk_pct = 0.04  # 4% risk in normal vol

        max_risk = balance * risk_pct
        num_contracts = max(1, int(max_risk / (net_debit * 100)))

        # ── Intraday simulation: check TP/SL with high/low ──
        tp_price = net_debit * (1 + TP_PCT)
        sl_price = net_debit * (1 - SL_PCT)

        # Calculate spread value at best/worst/close
        def spread_value(S, T_rem):
            lp = bs_price(S, K_buy, T_rem, r, iv, opt)
            sp = bs_price(S, K_sell, T_rem, r, iv, opt)
            return max(lp - sp, 0)

        if opt == "call":
            best_val = spread_value(spy_h, T_mid)
            worst_val = spread_value(spy_l, T_mid)
        else:
            best_val = spread_value(spy_l, T_mid)
            worst_val = spread_value(spy_h, T_mid)

        eod_val = spread_value(spy_c, T_exit)

        # Determine exit
        exit_note = "EOD"
        if worst_val * (1 - SPREAD_PCT) <= sl_price:
            exit_val = max(sl_price - SLIPPAGE, 0)
            exit_note = "SL"
        elif best_val * (1 - SPREAD_PCT) >= tp_price:
            exit_val = tp_price - SLIPPAGE
            exit_note = "TP"
        else:
            exit_val = max(eod_val * (1 - SPREAD_PCT) - SLIPPAGE, 0)

        # P&L
        pnl_per = (exit_val - net_debit) * 100
        total_pnl = pnl_per * num_contracts
        total_pnl = max(total_pnl, -net_debit * 100 * num_contracts)  # can't lose more than debit

        balance += total_pnl
        if total_pnl > 0: wins += 1
        else: losses += 1

        spy_move = spy_c - spy_o
        g = "G" if grade == "STRONG" else "H"

        trades.append({
            "date": date.strftime("%Y-%m-%d"), "score": score, "grade": grade,
            "direction": direction, "K_buy": K_buy, "K_sell": K_sell,
            "net_debit": round(net_debit, 2), "exit_val": round(exit_val, 2),
            "spy_move": round(spy_move, 2), "contracts": num_contracts,
            "pnl": round(total_pnl, 2), "balance": round(balance, 2),
            "vix": round(vix_val, 1), "exit_type": exit_note
        })

        print(f"{ds:<11} {score:>3} {g:<2} {opt:<4} {K_buy:>5}/{K_sell:>3} ${net_debit:>5.2f} ${exit_val:>5.2f} ${spy_move:>+6.2f} ${total_pnl:>+6.0f} {exit_note:>3} ${balance:>9,.0f}")

    # ── Summary ──
    total_trades = wins + losses
    total_pnl = balance - initial_balance
    wr = (wins / total_trades * 100) if total_trades > 0 else 0

    peak = initial_balance
    max_dd = 0
    for t in trades:
        if t["balance"] > peak: peak = t["balance"]
        dd = (peak - t["balance"]) / peak * 100
        if dd > max_dd: max_dd = dd

    avg_w = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if wins > 0 else 0
    avg_l = np.mean([t["pnl"] for t in trades if t["pnl"] <= 0]) if losses > 0 else 0
    sharpe = 0
    if trades:
        rets = [t["pnl"] / initial_balance for t in trades]
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(252) if np.std(rets) > 0 else 0

    # Profit factor
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    print("\n" + "=" * 80)
    print("  BACKTEST v3 RESULTS — DEBIT SPREADS")
    print("=" * 80)
    print(f"  Period:            {days} trading days")
    print(f"  Starting Balance:  ${initial_balance:,.2f}")
    print(f"  Final Balance:     ${balance:,.2f}")
    print(f"  Total P&L:         ${total_pnl:+,.2f} ({total_pnl/initial_balance*100:+.1f}%)")
    print(f"  Total Trades:      {total_trades}")
    print(f"  Win Rate:          {wr:.1f}% ({wins}W / {losses}L)")
    print(f"  Avg Win:           ${avg_w:+,.2f}")
    print(f"  Avg Loss:          ${avg_l:+,.2f}")
    print(f"  Profit Factor:     {pf}")
    print(f"  Max Drawdown:      {max_dd:.1f}%")
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"  ---")
    print(f"  Strategy:          ${SPREAD_WIDTH} wide debit spread (ATM/OTM)")
    print(f"  Exit:              TP +{TP_PCT*100:.0f}% / SL -{SL_PCT*100:.0f}% / EOD")
    print(f"  VIX filter:        >= {MIN_VIX}")
    print(f"  Score filter:      >= {MIN_SCORE}")
    print(f"  Sizing:            Dynamic (4-8% risk based on VIX)")
    print("=" * 80)

    results = {
        "model": "Debit Spread v3", "period_days": days,
        "start_balance": initial_balance, "end_balance": round(balance, 2),
        "total_pnl": round(total_pnl, 2), "pnl_pct": round(total_pnl / initial_balance * 100, 1),
        "total_trades": total_trades, "wins": wins, "losses": losses,
        "win_rate": round(wr, 1), "avg_win": round(avg_w, 2), "avg_loss": round(avg_l, 2),
        "profit_factor": pf, "max_drawdown": round(max_dd, 1),
        "sharpe": round(sharpe, 2), "trades": trades
    }
    with open("backtest_v3.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [*] Saved to backtest_v3.json")
    return results

if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_backtest(days=days)
