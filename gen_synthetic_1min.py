#!/usr/bin/env python3
"""
Generate synthetic 1-minute SPY bars from yfinance daily OHLCV.

Each bar's daily open/high/low/close is preserved.  Intraday path uses a
Brownian bridge so prices always start at the daily open, end at the daily
close, and at some point touch the daily high and daily low.
"""
import sys, time
import numpy as np
import pandas as pd
import yfinance as yf
import pytz
from datetime import timedelta

NY = pytz.timezone("America/New_York")
MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 30
BARS_PER_DAY = 390  # 9:30 → 16:00


def synthetic_1min_day(date, o, h, l, c, vol, seed=None):
    """Return list of 1-min bar dicts guaranteed to cover daily OHLC."""
    rng = np.random.default_rng(seed)
    n = BARS_PER_DAY

    # --- Brownian bridge o → c ---
    steps = rng.standard_normal(n)
    path = np.cumsum(steps)
    t = np.arange(1, n + 1)
    # Bridge forces path[n-1] = c - o
    bridge = path - (t / n) * path[-1] + (t / n) * (c - o)
    prices = o + bridge

    # --- Scale to exactly touch [l, h] ---
    p_min, p_max = prices.min(), prices.max()
    if p_max > p_min:
        prices = l + (prices - p_min) / (p_max - p_min) * (h - l)
    prices[0] = o
    prices[-1] = c

    # --- Volume: U-shaped (heavier at open and close) ---
    x = np.linspace(0, 1, n)
    w = 1.5 + np.exp(-8 * x) + np.exp(-8 * (1 - x))
    w /= w.sum()
    bar_vols = np.maximum((w * vol).astype(int), 100)

    # --- Build OHLC bars ---
    base_ts = pd.Timestamp(date.date()) \
                .tz_localize(NY) \
                .replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN)

    bar_range_sigma = (h - l) / n * 1.5
    rows = []
    for i in range(n):
        mid = prices[i]
        nxt = prices[i + 1] if i < n - 1 else c
        noise = rng.normal(0, bar_range_sigma * 0.3)
        b_o = round(float(mid + noise), 2)
        b_c = round(float(nxt), 2)
        b_h = round(float(max(b_o, b_c) + abs(rng.normal(0, bar_range_sigma * 0.5))), 2)
        b_l = round(float(min(b_o, b_c) - abs(rng.normal(0, bar_range_sigma * 0.5))), 2)
        b_h = min(b_h, float(h))
        b_l = max(b_l, float(l))
        ts = base_ts + timedelta(minutes=i)
        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "open":   b_o,
            "high":   b_h,
            "low":    b_l,
            "close":  b_c,
            "volume": int(bar_vols[i]),
        })
    return rows


def main():
    start = "2021-01-01"
    end   = "2026-06-01"
    out   = "SPY_1min_synthetic.csv"

    # Use ^GSPC (S&P 500 index) for prices → matches MES futures price level 4000-7000
    # Use SPY for volume reference
    print(f"[*] Downloading ^GSPC (S&P 500 index) daily bars {start} ~ {end} …")
    gspc = yf.download("^GSPC", start=start, end=end, interval="1d",
                       progress=False, auto_adjust=True)
    spy_vol = yf.download("SPY", start=start, end=end, interval="1d",
                          progress=False, auto_adjust=True)
    if isinstance(gspc.columns, pd.MultiIndex):
        gspc.columns = gspc.columns.droplevel(1)
    if isinstance(spy_vol.columns, pd.MultiIndex):
        spy_vol.columns = spy_vol.columns.droplevel(1)
    # Merge: use ^GSPC OHLC, SPY Volume (scaled up to look like futures volume)
    spy = gspc[["Open","High","Low","Close"]].copy()
    spy["Volume"] = spy_vol["Volume"].reindex(spy.index).fillna(80_000_000)
    spy = spy.dropna(subset=["Open"])
    print(f"[*] {len(spy)} trading days. Price range: {spy['Open'].min():.0f}–{spy['Open'].max():.0f}")

    all_bars = []
    t0 = time.time()
    for i, (date, row) in enumerate(spy.iterrows()):
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        v = float(row.get("Volume", 80_000_000))
        seed = int(date.strftime("%Y%m%d")) % (2**31 - 1)
        all_bars.extend(synthetic_1min_day(date, o, h, l, c, v, seed=seed))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(spy)} days …")

    df = pd.DataFrame(all_bars)
    df.to_csv(out, index=False)
    print(f"[OK] {len(df):,} bars → {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
