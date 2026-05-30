#!/usr/bin/env python3
"""
오늘(또는 지정 날짜)의 알고리듬 실행을 재현하여
어느 시점에 진입 조건이 충족됐는지 (혹은 안 됐는지) 보여준다.

용법:
    python replay_today.py                # 가장 최근 거래일
    python replay_today.py --date 2026-05-28

출력:
    - 5분 간격으로 score / grade / entry_reason 표시
    - STRONG 시점 발견 시 강조
    - 발견되면 portfolio.json에 backfill 옵션 제공

데이터 소스: Yahoo Finance (무료, 60일 1-min 한계 내).
Alpaca/Polygon 키 설정 시 자동 사용.
"""
import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, "api")

import pandas as pd
import numpy as np
import yfinance as yf
import pytz

NY = pytz.timezone("America/New_York")


def fetch_minute_bars(symbol: str, date_str: str) -> pd.DataFrame:
    """Yahoo 1-min for the given date. Returns regular-session bars only."""
    target = datetime.strptime(date_str, "%Y-%m-%d").date()
    start = target
    end = target + timedelta(days=1)
    df = yf.Ticker(symbol).history(start=start, end=end, interval="1m")
    if df.empty:
        return df
    # Yahoo returns tz-aware in NY usually; ensure ET
    if df.index.tz is None:
        df.index = df.index.tz_localize(NY)
    else:
        df.index = df.index.tz_convert(NY)
    # Filter to regular session 9:30 ~ 16:00 ET
    df = df.between_time("09:30", "16:00")
    return df


def replay_day(date_str: str, verbose: bool = True) -> dict:
    """Run the score engine every 5 min through the trading day."""
    from engines.score_engine import run_score_engine

    print(f"[*] Fetching SPY/QQQ/IWM/^VIX 1-min bars for {date_str}...")
    spy = fetch_minute_bars("SPY", date_str)
    qqq = fetch_minute_bars("QQQ", date_str)
    iwm = fetch_minute_bars("IWM", date_str)
    dia = fetch_minute_bars("DIA", date_str)
    vix = fetch_minute_bars("^VIX", date_str)

    if spy.empty:
        print(f"[!] No SPY 1-min data for {date_str} — likely not a trading day or beyond Yahoo's 60-day window.")
        return {"date": date_str, "trading_day": False, "frames": [], "strong_minutes": []}

    spy_open = float(spy["Open"].iloc[0])
    prev_day = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
    spy_prev = yf.Ticker("SPY").history(start=prev_day, end=date_str, interval="1d")
    spy_prev_close = float(spy_prev["Close"].iloc[-1]) if not spy_prev.empty else spy_open

    # Synthetic portfolio (clean state for replay)
    portfolio = {
        "cash": 500000, "positions": {}, "history": [], "trade_log": [],
        "initial_balance": 500000, "current_value": 500000,
        "daily_start_value": 500000, "daily_session_date": date_str,
    }

    # Iterate 5-min steps
    frames = []
    strong_minutes = []
    spy_h_so_far = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    times = sorted(spy.index.unique())
    last_print_time = None
    for t in times:
        # Append this minute to running history
        bar = spy.loc[[t]]
        spy_h_so_far = pd.concat([spy_h_so_far, pd.DataFrame({
            "Open":   bar["Open"].values,
            "High":   bar["High"].values,
            "Low":    bar["Low"].values,
            "Close":  bar["Close"].values,
            "Volume": bar["Volume"].values,
        }, index=bar.index)])

        if t.minute % 5 != 0 or t == last_print_time:
            continue
        last_print_time = t

        spy_p = float(bar["Close"].iloc[0])
        # VIX value at this time (closest prior)
        if not vix.empty:
            vix_at = vix.loc[vix.index <= t]
            vix_p = float(vix_at["Close"].iloc[-1]) if not vix_at.empty else 18.0
        else:
            vix_p = 18.0

        # Sector pcts (vs prior session close approximation: use first bar open)
        def _pct_from_open(df, t_):
            if df.empty: return 0.0
            row = df.loc[df.index <= t_]
            if row.empty: return 0.0
            ref = float(df["Open"].iloc[0])
            cur = float(row["Close"].iloc[-1])
            return (cur / ref - 1) * 100 if ref else 0.0

        pcts = {
            "SPY": _pct_from_open(spy, t),
            "QQQ": _pct_from_open(qqq, t),
            "IWM": _pct_from_open(iwm, t),
            "DIA": _pct_from_open(dia, t),
        }

        # VWAP, vol ratio, range from spy_h_so_far
        tp = (spy_h_so_far["High"] + spy_h_so_far["Low"] + spy_h_so_far["Close"]) / 3.0
        vol = spy_h_so_far["Volume"].astype(float)
        vwap_v = float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else spy_p
        vol_recent = float(spy_h_so_far["Volume"].tail(5).mean()) if len(spy_h_so_far) >= 5 else float(vol.mean() or 0)
        vol_sma = float(spy_h_so_far["Volume"].rolling(20).mean().iloc[-1]) if len(spy_h_so_far) >= 20 else 1.0
        vol_r = vol_recent / vol_sma if vol_sma > 0 else 1.0
        d_range = float(spy_h_so_far["High"].max() - spy_h_so_far["Low"].min())

        result = run_score_engine(
            now_et=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
            spy_price=spy_p, vix_price=vix_p, vix3m_price=None,
            prev_close=spy_prev_close, vwap=vwap_v,
            vol_ratio=vol_r, range_value=d_range,
            pcts=pcts, spy_history=spy_h_so_far,
            portfolio=portfolio, session_name="REGULAR",
        )

        frame = {
            "time":  t.strftime("%H:%M"),
            "spy":   round(spy_p, 2),
            "vix":   round(vix_p, 2),
            "score": result["total_score"],
            "grade": result["signal"]["grade"],
            "bias":  result["direction_bias"],
            "tw":    result["layers"]["time_window"]["window"],
            "regime": result["layers"]["regime"]["regime"],
        }
        frames.append(frame)
        if frame["grade"] == "STRONG" and frame["bias"] in ("LONG", "SHORT"):
            strong_minutes.append(frame)

        if verbose:
            tag = "🟢" if frame["grade"] == "STRONG" else ("🟡" if frame["grade"] == "MODERATE" else " ")
            print(f"  {tag} {frame['time']}  score={frame['score']:3d} {frame['grade']:8s} {frame['bias']:8s}  tw={frame['tw']:12s} vix={frame['vix']:5.2f} regime={frame['regime']}")

    return {
        "date": date_str,
        "trading_day": True,
        "total_frames": len(frames),
        "strong_minutes": strong_minutes,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (defaults to most recent trading day in NY)")
    parser.add_argument("--out",  default="replay_result.json", help="Output JSON path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.date:
        date_str = args.date
    else:
        now = datetime.now(NY)
        # If today is Sat/Sun, walk back to Friday
        d = now
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        date_str = d.strftime("%Y-%m-%d")

    print(f"=== Replaying {date_str} ===\n")
    result = replay_day(date_str, verbose=not args.quiet)
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))

    print()
    print(f"=== Summary ===")
    print(f"  Frames evaluated: {result['total_frames']}")
    print(f"  STRONG opportunities: {len(result['strong_minutes'])}")
    if result["strong_minutes"]:
        print(f"  Earliest STRONG: {result['strong_minutes'][0]['time']} → score={result['strong_minutes'][0]['score']} {result['strong_minutes'][0]['bias']}")
        print(f"  Latest STRONG:   {result['strong_minutes'][-1]['time']} → score={result['strong_minutes'][-1]['score']} {result['strong_minutes'][-1]['bias']}")
    else:
        print(f"  → No STRONG signal during regular hours. Algo correctly did not fire today.")
    print(f"  Result saved: {args.out}")


if __name__ == "__main__":
    main()
