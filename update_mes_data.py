#!/usr/bin/env python3
"""
Incrementally update MES 1-min data and regenerate the RTH (ET) file.

Pipeline:
  1. Read existing raw CSV (MES_1min_data.csv, UTC) → find last timestamp
  2. Download new bars from Databento (last_ts → --end, default now) for MES.c.0
  3. Merge + dedupe on timestamp → rewrite raw CSV
  4. Regenerate RTH file (MES_1min_data_et_rth.csv): UTC → US/Eastern,
     keep 09:30–15:59, drop tz. Validated to reproduce the existing file exactly.

Requires DATABENTO_API_KEY in the environment (or .env).

Usage:
  python3 update_mes_data.py [--end 2026-06-23] \
      [--raw MES_1min_data.csv] [--rth MES_1min_data_et_rth.csv]
"""
import os
import sys
import argparse
from datetime import datetime, timezone

import pandas as pd
import pytz

ET = pytz.timezone("US/Eastern")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def regenerate_rth(raw_df: pd.DataFrame) -> pd.DataFrame:
    """UTC raw → ET RTH (09:30–15:59), tz-naive index. Matches existing file."""
    df = raw_df.copy()
    df.index = df.index.tz_localize("UTC").tz_convert(ET)
    rth = df.between_time("09:30", "15:59")
    rth.index = rth.index.tz_localize(None)
    rth.index.name = "timestamp"
    return rth


def main():
    p = argparse.ArgumentParser(description="Incremental MES data updater")
    p.add_argument("--symbol", default="MES.c.0")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today UTC)")
    p.add_argument("--raw", default="MES_1min_data.csv")
    p.add_argument("--rth", default="MES_1min_data_et_rth.csv")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    raw_path = args.raw if os.path.isabs(args.raw) else os.path.join(here, args.raw)
    rth_path = args.rth if os.path.isabs(args.rth) else os.path.join(here, args.rth)

    # ── Existing raw ──────────────────────────────────────────────────────────
    raw = pd.read_csv(raw_path, parse_dates=["timestamp"]).set_index("timestamp")
    raw.sort_index(inplace=True)
    last_ts = raw.index.max()
    # Databento start is inclusive; resume one minute after last bar
    start = (last_ts + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")
    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"[*] Existing raw: {len(raw):,} bars, last = {last_ts} UTC")
    print(f"[*] Fetching new bars: {start} → {end}")

    if pd.Timestamp(end) <= last_ts.normalize():
        print(f"[!] --end {end} not after last bar {last_ts.date()}; nothing to fetch.")
        return

    # ── Download ──────────────────────────────────────────────────────────────
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("[ERROR] DATABENTO_API_KEY not set in environment / .env")
        sys.exit(2)

    import databento as db
    client = db.Historical(api_key)
    try:
        cost = client.metadata.get_cost(
            dataset="GLBX.MDP3", schema="ohlcv-1m", symbols=args.symbol,
            stype_in="continuous", start=start, end=end,
        )
        print(f"[*] Estimated cost: ${cost:.4f}")
    except Exception as e:
        print(f"[!] Cost estimate failed (continuing): {e}")

    data = client.timeseries.get_range(
        dataset="GLBX.MDP3", schema="ohlcv-1m", symbols=args.symbol,
        stype_in="continuous", start=start, end=end,
    )
    new = data.to_df()
    if new.empty:
        print("[!] No new bars returned. Up to date.")
        return

    new = new[["open", "high", "low", "close", "volume"]].copy()
    new.index = new.index.tz_convert("UTC").tz_localize(None)
    new.index.name = "timestamp"
    print(f"[*] Downloaded {len(new):,} new bars: {new.index.min()} → {new.index.max()}")

    # ── Merge + dedupe ────────────────────────────────────────────────────────
    merged = pd.concat([raw, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    added = len(merged) - len(raw)
    merged.to_csv(raw_path)
    print(f"[OK] Raw updated: {len(merged):,} bars (+{added:,}). Saved {raw_path}")

    # ── Regenerate RTH ────────────────────────────────────────────────────────
    rth = regenerate_rth(merged)
    rth.to_csv(rth_path)
    print(f"[OK] RTH regenerated: {len(rth):,} bars, last = {rth.index.max()}. Saved {rth_path}")


if __name__ == "__main__":
    main()
