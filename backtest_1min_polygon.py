"""
ES Futures Signal Engine -- 5-Year 1-Minute Bar Backtest
=========================================================
Data Source: Polygon.io (정확한 1분봉 데이터)
Period: 2021-06-01 ~ 2026-05-23 (약 1,240 거래일)
Contract: MES Micro-Size ($5/pt)

Usage:
  set POLYGON_API_KEY=your_key_here
  python backtest_1min_polygon.py

Or create .env file with:
  POLYGON_API_KEY=your_key_here
"""

import os, sys, json, time, math, warnings
from datetime import datetime, timedelta, date
from pathlib import Path
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")

# Try loading from .env if not in environment
if not POLYGON_KEY:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("POLYGON_API_KEY="):
                POLYGON_KEY = line.split("=", 1)[1].strip()

if not POLYGON_KEY:
    print("=" * 60)
    print("  ERROR: POLYGON_API_KEY not found!")
    print()
    print("  Set it via environment variable:")
    print("    set POLYGON_API_KEY=your_key_here")
    print("    python backtest_1min_polygon.py")
    print()
    print("  Or add to .env file:")
    print("    POLYGON_API_KEY=your_key_here")
    print()
    print("  Get free key at: https://polygon.io/")
    print("=" * 60)
    sys.exit(1)

# MES Futures Constants
ES_MULT = 5.0
ES_COMM = 0.50        # Round-trip commission
ES_SLIP = 0.25        # 1 tick slippage per side
ES_MARGIN = 50.0      # Day margin per contract
ATR_SL_MULT = 1.5
RISK_PCT = 0.12       # 12% Kelly risk per trade
START_BAL = 500000.0

# Risk Limits
MAX_DD_PCT = 6.0
STREAK_LOCK = 3
MAX_DAILY = 3

# Score thresholds
GRADE_STRONG = 90
GRADE_MODERATE = 75

# Time Windows (minutes since midnight ET)
PRIME_START = 630    # 10:30
PRIME_END   = 690    # 11:30
GAMMA_START = 840    # 14:00
GAMMA_END   = 885    # 14:45
MARKET_OPEN = 570    # 09:30
MARKET_CLOSE = 960   # 16:00
EOD_EXIT    = 930    # 15:30 force close

# Data cache directory
CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

START_DATE = date(2021, 6, 1)
END_DATE   = date(2026, 5, 23)

print("=" * 70)
print("  MES FUTURES -- 5-YEAR 1-MINUTE BACKTEST (Polygon.io)")
print(f"  Period: {START_DATE} ~ {END_DATE}")
print(f"  Contract: MES Micro-Size ($5/pt)")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD (Polygon.io REST API)
# ══════════════════════════════════════════════════════════════════════

def polygon_aggs(symbol, from_date, to_date, timespan="minute", multiplier=1,
                 limit=50000):
    """Download aggregates from Polygon REST API with auto-pagination."""
    all_results = []
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": limit,
        "apiKey": POLYGON_KEY,
    }
    
    while True:
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                # Rate limited - wait and retry
                print("    [Rate limited, waiting 60s...]", end="", flush=True)
                time.sleep(62)
                print(" retrying", flush=True)
                continue
            if r.status_code == 403:
                print(f"    [ERROR 403] API key may be invalid or plan doesn't support this data")
                return pd.DataFrame()
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    [Error] {e}")
            time.sleep(5)
            continue
        
        results = data.get("results", [])
        if not results:
            break
        all_results.extend(results)
        
        # Check for next page
        next_url = data.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": POLYGON_KEY}
        else:
            break
    
    if not all_results:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("US/Eastern")
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close",
                             "v": "Volume", "vw": "VWAP"})
    df = df.set_index("datetime")
    df = df[["Open", "High", "Low", "Close", "Volume", "VWAP"]]
    return df


def download_with_cache(symbol, from_date, to_date, timespan="minute"):
    """Download data in monthly chunks with disk caching."""
    cache_file = CACHE_DIR / f"{symbol}_{timespan}_{from_date}_{to_date}.parquet"
    
    if cache_file.exists():
        print(f"  {symbol}: Loading from cache...")
        return pd.read_parquet(cache_file)
    
    print(f"  {symbol}: Downloading {timespan} bars {from_date} -> {to_date}...")
    
    # Download in monthly chunks to avoid rate limits
    chunks = []
    current = from_date
    chunk_num = 0
    
    while current < to_date:
        # Monthly chunk (Polygon free tier: max 2 years back, 5 calls/min)
        chunk_end = min(current + timedelta(days=30), to_date)
        chunk_cache = CACHE_DIR / f"{symbol}_{timespan}_{current}_{chunk_end}.parquet"
        
        if chunk_cache.exists():
            df = pd.read_parquet(chunk_cache)
            if not df.empty:
                chunks.append(df)
                current = chunk_end + timedelta(days=1)
                continue
        
        chunk_num += 1
        sys.stdout.write(f"\r    Chunk {chunk_num}: {current} -> {chunk_end} ... ")
        sys.stdout.flush()
        
        df = polygon_aggs(symbol, str(current), str(chunk_end), timespan)
        
        if not df.empty:
            df.to_parquet(chunk_cache)
            chunks.append(df)
            sys.stdout.write(f"{len(df)} bars\n")
        else:
            sys.stdout.write("0 bars\n")
        
        current = chunk_end + timedelta(days=1)
        
        # Rate limit: Polygon free = 5 calls/minute
        time.sleep(13)  # ~4.6 calls/min to be safe
    
    if not chunks:
        return pd.DataFrame()
    
    result = pd.concat(chunks)
    result = result[~result.index.duplicated(keep='first')]
    result = result.sort_index()
    
    # Save combined cache
    result.to_parquet(cache_file)
    print(f"  {symbol}: Total {len(result)} bars cached.")
    return result


def download_daily_with_cache(symbol, from_date, to_date):
    """Download daily bars (used for VIX which has no meaningful 1-min data)."""
    cache_file = CACHE_DIR / f"{symbol}_daily_{from_date}_{to_date}.parquet"
    
    if cache_file.exists():
        print(f"  {symbol}: Loading daily from cache...")
        return pd.read_parquet(cache_file)
    
    print(f"  {symbol}: Downloading daily bars...")
    df = polygon_aggs(symbol, str(from_date), str(to_date), "day", 1)
    
    if not df.empty:
        df.to_parquet(cache_file)
    print(f"  {symbol}: {len(df)} daily bars")
    return df


# ══════════════════════════════════════════════════════════════════════
# INDICATORS (computed on 1-minute data)
# ══════════════════════════════════════════════════════════════════════

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_adx(high, low, close, period=14):
    pdm = high.diff()
    mdm = low.diff().abs()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
                    axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    pdi = 100 * pdm.ewm(span=period, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(span=period, adjust=False).mean() / atr
    dx = (pdi - mdi).abs() / (pdi + mdi) * 100
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, atr


def compute_daily_indicators(spy_1m, vix_daily, qqq_1m, iwm_1m, dia_1m):
    """
    Pre-compute daily-level indicators from 1-min data.
    Returns a dict keyed by date string with all needed values.
    """
    print("  Computing daily indicators from 1-min data...")
    
    # Group by trading date
    spy_1m["date"] = spy_1m.index.date
    
    daily = {}
    dates = sorted(spy_1m["date"].unique())
    
    # Pre-compute rolling indicators on 5-min resampled data for efficiency
    spy_5m = spy_1m.resample("5min").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna()
    
    spy_5m["RSI"] = calc_rsi(spy_5m["Close"], 14)
    spy_5m["ADX"], spy_5m["ATR"] = calc_adx(spy_5m["High"], spy_5m["Low"], spy_5m["Close"], 14)
    
    # Daily aggregates
    spy_daily_agg = spy_1m.resample("D").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna()
    
    spy_daily_agg["Prev_Close"] = spy_daily_agg["Close"].shift(1)
    spy_daily_agg["Range"] = spy_daily_agg["High"] - spy_daily_agg["Low"]
    spy_daily_agg["Vol20"] = spy_daily_agg["Volume"].rolling(20).mean()
    spy_daily_agg["VolR"] = spy_daily_agg["Volume"] / spy_daily_agg["Vol20"]
    spy_daily_agg["Pct"] = spy_daily_agg["Close"].pct_change() * 100
    
    # Do the same for QQQ, IWM, DIA
    peer_pcts = {}
    for name, df in [("QQQ", qqq_1m), ("IWM", iwm_1m), ("DIA", dia_1m)]:
        if df is not None and not df.empty:
            d = df.resample("D").agg({"Close": "last"}).dropna()
            d["Pct"] = d["Close"].pct_change() * 100
            peer_pcts[name] = d
    
    return spy_1m, spy_5m, spy_daily_agg, peer_pcts, vix_daily


# ══════════════════════════════════════════════════════════════════════
# SCORING ENGINE (exact replica of engines/)
# ══════════════════════════════════════════════════════════════════════

def score_regime(vix_val, vix3m_val, spy_price, prev_close, adx_val):
    s = 0
    if vix_val is not None:
        if 14 <= vix_val <= 20: s += 15
        elif 20 < vix_val <= 30: s += 0
        elif vix_val > 30: s -= 20
        else: s -= 5
    if vix_val is not None and vix3m_val is not None:
        spread = vix_val - vix3m_val
        if spread < 0: s += 10
        elif spread > 0: s -= 15
    if prev_close and prev_close > 0:
        gap = abs(((spy_price / prev_close) - 1) * 100)
        if gap > 0.5: s += 5
    if adx_val is not None:
        if adx_val >= 25: s += 15
        elif adx_val >= 20: s += 5
    return max(0, min(40, s))


def score_correlation(spy_pct, qqq_pct, iwm_pct, dia_pct):
    s = 0
    synced = False
    if (spy_pct >= 0 and qqq_pct >= 0) or (spy_pct < 0 and qqq_pct < 0): s += 10
    else: s -= 5
    if iwm_pct > 0.3: s += 5
    elif iwm_pct < -0.3: s -= 3
    all_up = spy_pct >= 0 and qqq_pct >= 0 and iwm_pct >= 0
    all_dn = spy_pct < 0 and qqq_pct < 0 and iwm_pct < 0
    synced = all_up or all_dn
    if synced: s += 5
    if (spy_pct >= 0 and dia_pct >= 0) or (spy_pct < 0 and dia_pct < 0): s += 3
    return max(0, min(20, s)), synced


def score_time(t_min, weekday):
    s = 0
    if 630 <= t_min < 690: s = 20     # PRIME
    elif 840 <= t_min < 885: s = 15    # GAMMA
    elif 600 <= t_min < 630: s = 5     # FORMING
    elif 690 <= t_min < 720: s = 8     # TRANSITION
    elif 780 <= t_min < 840: s = 8     # REENTRY
    else: s = 0
    if weekday == 4: s = max(0, s - 5)
    return s


def score_technical(spy_price, vwap, vol_r, d_range, rsi):
    s = 0; bc = 0; bp = 0
    if vwap and vwap > 0:
        s += 10
        if spy_price > vwap: bc += 1
        else: bp += 1
        dp = abs(spy_price - vwap) / vwap * 100
        if dp > 2: s -= 5
        elif dp > 1: s += 5
    if vol_r >= 2: s += 10
    elif vol_r >= 1.5: s += 7
    elif vol_r >= 1: s += 3
    if d_range >= 3: s += 10
    elif d_range >= 2: s += 5
    if rsi is not None:
        if rsi >= 70: s += 5; bc += 1
        elif rsi >= 60: s += 10; bc += 1
        elif rsi <= 30: s += 5; bp += 1
        elif rsi <= 40: s += 10; bp += 1
    d = "CALL" if bc > bp else ("PUT" if bp > bc else "NEUTRAL")
    return max(0, min(30, s)), d, rsi


def check_runaway(adx, rsi, spy_pct, qqq_pct, iwm_pct):
    if adx is not None and adx >= 35: return True
    if rsi is not None and (rsi >= 80 or rsi <= 20): return True
    if spy_pct > 1.2 and qqq_pct > 1.2 and iwm_pct > 1.2: return True
    if spy_pct < -1.2 and qqq_pct < -1.2 and iwm_pct < -1.2: return True
    return False


# ══════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════

def run_backtest(spy_1m, spy_5m, spy_daily, peer_pcts, vix_daily):
    """
    Bar-by-bar 1-minute backtest.
    
    Strategy:
    1. Score computed every 5 minutes (efficiency)
    2. Entry checked during PRIME (10:30-11:30) and GAMMA (14:00-14:45)
    3. SL/TP checked every 1-minute bar
    4. EOD forced exit at 15:30
    """
    
    cash = START_BAL
    trades = []
    equity_daily = {}
    max_eq = START_BAL
    max_dd = 0.0
    c_losses = 0
    total_pnl = 0.0
    monthly = {}
    yearly = {}
    lockouts = 0
    
    # Current position state
    position = None  # {es_dir, entry_price, contracts, sl, tp, margin, date, entry_time}
    
    dates = sorted(spy_daily.index)
    total_days = len(dates)
    
    print(f"\n  Backtesting {total_days} trading days bar-by-bar...")
    last_pct = -1
    
    for day_idx, day_date in enumerate(dates):
        # Progress
        pct = int(day_idx / total_days * 100)
        if pct % 5 == 0 and pct != last_pct:
            sys.stdout.write(f"\r    Progress: {pct}% ({day_idx}/{total_days})")
            sys.stdout.flush()
            last_pct = pct
        
        day_str = str(day_date)[:10]
        day_dt = pd.Timestamp(day_date)
        
        # Skip if day_date is a Timestamp with time component, extract date
        if hasattr(day_date, 'date'):
            trade_date = day_date.date() if callable(day_date.date) else day_date
        else:
            trade_date = day_date
        
        weekday = trade_date.weekday() if hasattr(trade_date, 'weekday') else 0
        if weekday >= 5:
            continue
        
        # Daily context
        row = spy_daily.loc[day_date]
        prev_close = row.get("Prev_Close")
        d_range = row.get("Range", 3.0)
        vol_r = row.get("VolR", 1.0)
        spy_pct = row.get("Pct", 0.0)
        
        if prev_close is None or (isinstance(prev_close, float) and np.isnan(prev_close)):
            prev_close = row["Open"]
        if isinstance(d_range, float) and np.isnan(d_range): d_range = 3.0
        if isinstance(vol_r, float) and np.isnan(vol_r): vol_r = 1.0
        if isinstance(spy_pct, float) and np.isnan(spy_pct): spy_pct = 0.0
        
        # VIX for this day
        vix_val = 18.0
        vix3m_val = 18.0
        if vix_daily is not None and not vix_daily.empty:
            # Find nearest VIX date
            vix_dates = vix_daily.index
            mask = vix_dates <= day_dt
            if mask.any():
                vix_row = vix_daily.loc[vix_dates[mask][-1]]
                vix_val = float(vix_row["Close"]) if not np.isnan(vix_row["Close"]) else 18.0
        
        # Peer % changes
        qqq_pct = 0.0; iwm_pct = 0.0; dia_pct = 0.0
        for name, df in peer_pcts.items():
            if df is not None and not df.empty:
                mask = df.index <= day_dt
                if mask.any():
                    val = float(df.loc[df.index[mask][-1]]["Pct"])
                    if not np.isnan(val):
                        if name == "QQQ": qqq_pct = val
                        elif name == "IWM": iwm_pct = val
                        elif name == "DIA": dia_pct = val
        
        # Get 1-min bars for this day
        day_start = pd.Timestamp(f"{day_str} 09:30:00", tz="US/Eastern")
        day_end = pd.Timestamp(f"{day_str} 16:00:00", tz="US/Eastern")
        day_bars = spy_1m.loc[day_start:day_end]
        
        if day_bars.empty:
            equity_daily[day_str] = cash
            continue
        
        # Get recent 5-min RSI and ADX
        end_5m = day_end
        recent_5m = spy_5m.loc[:end_5m].tail(50)
        
        rsi_val = None; adx_val = None; atr_val = None
        if not recent_5m.empty:
            if "RSI" in recent_5m.columns:
                last_rsi = recent_5m["RSI"].dropna()
                if not last_rsi.empty:
                    rsi_val = float(last_rsi.iloc[-1])
            if "ADX" in recent_5m.columns:
                last_adx = recent_5m["ADX"].dropna()
                if not last_adx.empty:
                    adx_val = float(last_adx.iloc[-1])
            if "ATR" in recent_5m.columns:
                last_atr = recent_5m["ATR"].dropna()
                if not last_atr.empty:
                    atr_val = float(last_atr.iloc[-1])
        
        daily_trades = 0
        
        # Reset consecutive losses at start of each new day
        if c_losses >= STREAK_LOCK:
            c_losses = 2  # Carry 2, need 1 more for re-lock
        
        # ── BAR-BY-BAR LOOP ──────────────────────────────────────
        for bar_time, bar in day_bars.iterrows():
            bar_price = float(bar["Close"])
            bar_high = float(bar["High"])
            bar_low = float(bar["Low"])
            bar_vwap = float(bar.get("VWAP", bar_price))
            
            hour = bar_time.hour
            minute = bar_time.minute
            t_min = hour * 60 + minute
            
            # Skip pre-market
            if t_min < MARKET_OPEN or t_min >= MARKET_CLOSE:
                continue
            
            # ── MANAGE OPEN POSITION ─────────────────────────────
            if position is not None:
                es_dir = position["es_dir"]
                entry_p = position["entry_price"]
                contracts = position["contracts"]
                margin = position["margin"]
                sl_p = position["sl"]
                tp_p = position["tp"]
                
                # Check SL/TP on this bar
                exit_type = None
                exit_price = None
                
                if es_dir == "LONG":
                    if bar_low <= sl_p:
                        exit_type = "SL"; exit_price = sl_p
                    elif bar_high >= tp_p:
                        exit_type = "TP"; exit_price = tp_p
                elif es_dir == "SHORT":
                    if bar_high >= sl_p:
                        exit_type = "SL"; exit_price = sl_p
                    elif bar_low <= tp_p:
                        exit_type = "TP"; exit_price = tp_p
                
                # EOD forced exit
                if exit_type is None and t_min >= EOD_EXIT:
                    exit_type = "EOD"
                    exit_price = bar_price
                
                if exit_type:
                    # Close position
                    if es_dir == "LONG":
                        ppnl = exit_price - entry_p - ES_SLIP
                    else:
                        ppnl = entry_p - exit_price - ES_SLIP
                    
                    pnl = round(ppnl * ES_MULT * contracts - ES_COMM * contracts, 2)
                    cash += pnl + margin  # Return margin + P&L
                    total_pnl += pnl
                    
                    win = pnl > 0
                    if win: c_losses = 0
                    else: c_losses += 1
                    
                    mk = day_str[:7]; yk = day_str[:4]
                    monthly[mk] = monthly.get(mk, 0) + pnl
                    yearly[yk] = yearly.get(yk, 0) + pnl
                    
                    trades.append({
                        "date": day_str,
                        "entry_time": position["entry_time"],
                        "exit_time": f"{hour:02d}:{minute:02d}",
                        "dir": position["direction"],
                        "es": es_dir,
                        "grade": position["grade"],
                        "entry": round(entry_p, 2),
                        "exit": round(exit_price, 2),
                        "sl": round(sl_p, 2),
                        "tp": round(tp_p, 2),
                        "sl_pts": position["sl_pts"],
                        "tp_pts": position["tp_pts"],
                        "ctrs": contracts,
                        "type": exit_type,
                        "pnl": pnl,
                        "ppnl": round(ppnl, 2),
                        "score": position["score"],
                        "vix": round(vix_val, 1),
                        "rsi": round(rsi_val, 1) if rsi_val else 0,
                        "adx": round(adx_val, 1) if adx_val else 0,
                        "trend": position["trending"],
                        "win": win,
                    })
                    
                    position = None
                    continue
            
            # ── CHECK ENTRY ──────────────────────────────────────
            if position is None and daily_trades < MAX_DAILY:
                # Only enter during PRIME or GAMMA windows
                in_prime = PRIME_START <= t_min < PRIME_END
                in_gamma = GAMMA_START <= t_min < GAMMA_END
                
                if not (in_prime or in_gamma):
                    continue
                
                # Compute score every 5 minutes
                if t_min % 5 != 0:
                    continue
                
                # Update RSI/ADX from recent 5m bars
                recent_5m_now = spy_5m.loc[:bar_time].tail(30)
                if not recent_5m_now.empty:
                    r = recent_5m_now["RSI"].dropna()
                    if not r.empty: rsi_val = float(r.iloc[-1])
                    a = recent_5m_now["ADX"].dropna()
                    if not a.empty: adx_val = float(a.iloc[-1])
                    at = recent_5m_now["ATR"].dropna()
                    if not at.empty: atr_val = float(at.iloc[-1])
                
                # Score
                regime_s = score_regime(vix_val, vix3m_val, bar_price, prev_close, adx_val)
                corr_s, synced = score_correlation(spy_pct, qqq_pct, iwm_pct, dia_pct)
                time_s = score_time(t_min, weekday)
                tech_s, raw_dir, _ = score_technical(bar_price, bar_vwap, vol_r, d_range, rsi_val)
                
                total_raw = regime_s + corr_s + time_s + tech_s
                norm = int((total_raw / 110) * 100)
                norm = max(0, min(100, norm))
                
                # Grade
                if norm >= GRADE_STRONG: grade = "STRONG"
                elif norm >= GRADE_MODERATE: grade = "MODERATE"
                else: grade = "NONE"
                
                # Runaway veto
                if check_runaway(adx_val, rsi_val, spy_pct, qqq_pct, iwm_pct):
                    grade = "LOCKED"; norm = 0
                
                # VIX adaptive direction
                trending = vix_val < 22.0
                if trending:
                    direction = raw_dir
                else:
                    if raw_dir == "CALL": direction = "PUT"
                    elif raw_dir == "PUT": direction = "CALL"
                    else: direction = "NEUTRAL"
                
                # Risk check
                risk_ok = True
                if c_losses >= STREAK_LOCK:
                    risk_ok = False; lockouts += 1
                dd_pct = max(0, (START_BAL - cash) / START_BAL * 100)
                if dd_pct >= MAX_DD_PCT: risk_ok = False
                
                # Entry criteria
                can_enter = (
                    grade in ("STRONG", "MODERATE") and
                    direction in ("CALL", "PUT") and
                    time_s >= 15 and
                    risk_ok
                )
                
                if can_enter:
                    es_dir = "LONG" if direction == "CALL" else "SHORT"
                    alloc = 1.0 if grade == "STRONG" else 0.5
                    
                    # Entry at current bar close + slippage
                    entry_price = bar_price + ES_SLIP if es_dir == "LONG" else bar_price - ES_SLIP
                    
                    # ATR-based SL/TP
                    atr_proxy = max(d_range, 2.0) if atr_val is None else max(atr_val, 2.0)
                    sl_pts = max(ATR_SL_MULT * atr_proxy, 2.0)
                    sl_pts = min(sl_pts, 15.0)
                    tp_pts = sl_pts * 2.0
                    
                    # Sizing
                    risk_per = sl_pts * ES_MULT + ES_COMM
                    max_risk = cash * RISK_PCT * alloc
                    max_m = int(cash * 0.95 / ES_MARGIN) if ES_MARGIN > 0 else 0
                    ctrs = min(max(1, int(max_risk / risk_per)), max_m)
                    
                    if ctrs > 0 and ctrs * ES_MARGIN <= cash:
                        margin = ctrs * ES_MARGIN
                        
                        if es_dir == "LONG":
                            sl_p = entry_price - sl_pts
                            tp_p = entry_price + tp_pts
                        else:
                            sl_p = entry_price + sl_pts
                            tp_p = entry_price - tp_pts
                        
                        cash -= margin  # Lock margin
                        
                        position = {
                            "es_dir": es_dir,
                            "direction": direction,
                            "entry_price": entry_price,
                            "contracts": ctrs,
                            "margin": margin,
                            "sl": sl_p, "tp": tp_p,
                            "sl_pts": round(sl_pts, 2),
                            "tp_pts": round(tp_pts, 2),
                            "date": day_str,
                            "entry_time": f"{hour:02d}:{minute:02d}",
                            "score": norm,
                            "grade": grade,
                            "trending": trending,
                        }
                        
                        daily_trades += 1
        
        # End of day: force close any remaining position
        if position is not None:
            last_bar = day_bars.iloc[-1]
            exit_price = float(last_bar["Close"])
            es_dir = position["es_dir"]
            entry_p = position["entry_price"]
            contracts = position["contracts"]
            margin = position["margin"]
            
            if es_dir == "LONG":
                ppnl = exit_price - entry_p - ES_SLIP
            else:
                ppnl = entry_p - exit_price - ES_SLIP
            
            pnl = round(ppnl * ES_MULT * contracts - ES_COMM * contracts, 2)
            cash += pnl + margin
            total_pnl += pnl
            
            win = pnl > 0
            if win: c_losses = 0
            else: c_losses += 1
            
            mk = day_str[:7]; yk = day_str[:4]
            monthly[mk] = monthly.get(mk, 0) + pnl
            yearly[yk] = yearly.get(yk, 0) + pnl
            
            trades.append({
                "date": day_str, "entry_time": position["entry_time"],
                "exit_time": "15:59", "dir": position["direction"],
                "es": es_dir, "grade": position["grade"],
                "entry": round(entry_p, 2), "exit": round(exit_price, 2),
                "sl": round(position["sl"], 2), "tp": round(position["tp"], 2),
                "sl_pts": position["sl_pts"], "tp_pts": position["tp_pts"],
                "ctrs": contracts, "type": "EOD_FORCE",
                "pnl": pnl, "ppnl": round(ppnl, 2),
                "score": position["score"], "vix": round(vix_val, 1),
                "rsi": round(rsi_val, 1) if rsi_val else 0,
                "adx": round(adx_val, 1) if adx_val else 0,
                "trend": position["trending"], "win": win,
            })
            position = None
        
        # Track equity
        equity_daily[day_str] = cash
        if cash > max_eq: max_eq = cash
        dd = (max_eq - cash) / max_eq * 100 if max_eq > 0 else 0
        if dd > max_dd: max_dd = dd
    
    print(f"\r    Progress: 100% ({total_days}/{total_days})     ")
    
    return {
        "trades": trades, "equity": equity_daily,
        "final": cash, "total_pnl": total_pnl,
        "max_dd": max_dd, "lockouts": lockouts,
        "monthly": monthly, "yearly": yearly,
    }


# ══════════════════════════════════════════════════════════════════════
# RESULTS PRINTER
# ══════════════════════════════════════════════════════════════════════

def print_results(res):
    trades = res["trades"]
    n = len(trades)
    fe = res["final"]
    tp = res["total_pnl"]
    dd = res["max_dd"]
    
    w = sum(1 for t in trades if t["win"])
    l = n - w
    tr = ((fe - START_BAL) / START_BAL) * 100
    wr = (w / n * 100) if n > 0 else 0
    aw = np.mean([t["pnl"] for t in trades if t["win"]]) if w > 0 else 0
    al = np.mean([t["pnl"] for t in trades if not t["win"]]) if l > 0 else 0
    at = tp / n if n > 0 else 0
    gp = sum(t["pnl"] for t in trades if t["win"])
    gl = abs(sum(t["pnl"] for t in trades if not t["win"]))
    pf = gp / gl if gl > 0 else float("inf")
    
    pnls = [t["pnl"] for t in trades]
    if len(pnls) > 1 and np.std(pnls) > 0:
        sh = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252)
    else:
        sh = 0
    
    # Streaks
    mcw = mcl = cur = 0
    for t in trades:
        if t["win"]:
            cur = cur + 1 if cur > 0 else 1
            mcw = max(mcw, cur)
        else:
            cur = cur - 1 if cur < 0 else -1
            mcl = max(mcl, abs(cur))
    
    lt = [t for t in trades if t["es"] == "LONG"]
    st = [t for t in trades if t["es"] == "SHORT"]
    lwr = (sum(1 for t in lt if t["win"]) / len(lt) * 100) if lt else 0
    swr = (sum(1 for t in st if t["win"]) / len(st) * 100) if st else 0
    
    exit_stats = {}
    for t in trades:
        k = t["type"]
        if k not in exit_stats: exit_stats[k] = {"n": 0, "w": 0, "pnl": 0}
        exit_stats[k]["n"] += 1
        exit_stats[k]["pnl"] += t["pnl"]
        if t["win"]: exit_stats[k]["w"] += 1
    
    tt = [t for t in trades if t["trend"]]
    ct = [t for t in trades if not t["trend"]]
    twr = (sum(1 for t in tt if t["win"]) / len(tt) * 100) if tt else 0
    cwr = (sum(1 for t in ct if t["win"]) / len(ct) * 100) if ct else 0
    
    strong_t = [t for t in trades if t["grade"] == "STRONG"]
    mod_t = [t for t in trades if t["grade"] == "MODERATE"]
    
    print("\n" + "=" * 70)
    print("  5-YEAR 1-MIN BACKTEST RESULTS -- MES FUTURES (Polygon.io)")
    print("=" * 70)
    print(f"""
  PERFORMANCE
  ---------------
  Starting:      ${START_BAL:>12,.2f}
  Final:         ${fe:>12,.2f}
  Net P&L:       ${tp:>+12,.2f}
  Return:        {tr:>+11.2f}%
  Max Drawdown:  {dd:>11.2f}%
  Sharpe Ratio:  {sh:>11.2f}
  Profit Factor: {pf:>11.2f}

  TRADE STATS
  ---------------
  Total Trades:  {n:>8d}
  Wins:          {w:>8d}  ({wr:.1f}%)
  Losses:        {l:>8d}
  Avg Win:       ${aw:>+10,.2f}
  Avg Loss:      ${al:>+10,.2f}
  Avg Trade:     ${at:>+10,.2f}
  Max Win Streak:{mcw:>8d}
  Max Loss Stk:  {mcl:>8d}
  Lockout Days:  {res['lockouts']:>8d}
""")
    
    print(f"  DIRECTION")
    print(f"    LONG:  {len(lt):>5d} trades | WR: {lwr:>5.1f}% | P&L: ${sum(t['pnl'] for t in lt):>+10,.2f}")
    print(f"    SHORT: {len(st):>5d} trades | WR: {swr:>5.1f}% | P&L: ${sum(t['pnl'] for t in st):>+10,.2f}")
    
    print(f"\n  VIX REGIME")
    print(f"    Trend (VIX<22):   {len(tt):>5d} | WR: {twr:>5.1f}% | P&L: ${sum(t['pnl'] for t in tt):>+10,.2f}")
    print(f"    Counter (VIX>=22):{len(ct):>5d} | WR: {cwr:>5.1f}% | P&L: ${sum(t['pnl'] for t in ct):>+10,.2f}")
    
    print(f"\n  SIGNAL GRADE")
    print(f"    STRONG:   {len(strong_t):>5d} trades")
    print(f"    MODERATE: {len(mod_t):>5d} trades")
    
    print(f"\n  EXIT TYPES")
    for k, v in sorted(exit_stats.items()):
        ew = (v["w"] / v["n"] * 100) if v["n"] > 0 else 0
        print(f"    {k:>10s}: {v['n']:>5d} ({ew:>5.1f}% WR) P&L: ${v['pnl']:>+10,.2f}")
    
    print(f"\n  YEARLY RETURNS")
    for y, p in sorted(res["yearly"].items()):
        pct = (p / START_BAL) * 100
        bar = "+" * max(0, int(pct / 2)) if pct > 0 else "-" * max(0, int(abs(pct) / 2))
        print(f"    {y}: ${p:>+10,.2f} ({pct:>+6.1f}%) {bar}")
    
    print(f"\n  MONTHLY RETURNS (last 24)")
    for m, p in sorted(res["monthly"].items())[-24:]:
        bar = "+" * max(0, int(p / 100)) if p > 0 else "-" * max(0, int(abs(p) / 100))
        print(f"    {m}: ${p:>+8,.0f} {bar}")
    
    if trades:
        print(f"\n  SAMPLE TRADES (first 20)")
        print(f"    {'Date':>12} {'Enter':>5} {'Exit':>5} {'Dir':>5} {'Grd':>3} {'Entry$':>8} {'Exit$':>8} {'Type':>4} {'#':>2} {'P&L':>9} {'Score':>3}")
        for t in trades[:20]:
            print(f"    {t['date']:>12} {t['entry_time']:>5} {t['exit_time']:>5} {t['es']:>5} {t['grade'][:3]:>3} {t['entry']:>8.2f} {t['exit']:>8.2f} {t['type']:>4} {t['ctrs']:>2} ${t['pnl']:>+8,.0f} {t['score']:>3}")
    
    return {
        "period": f"{START_DATE} ~ {END_DATE}",
        "data_source": "Polygon.io 1-minute bars",
        "start_bal": START_BAL, "final": round(fe, 2),
        "pnl": round(tp, 2), "return_pct": round(tr, 2),
        "max_dd": round(dd, 2), "sharpe": round(sh, 2),
        "pf": round(pf, 2) if pf != float("inf") else "inf",
        "trades": n, "wins": w, "losses": l, "wr": round(wr, 1),
        "avg_win": round(aw, 2), "avg_loss": round(al, 2),
        "direction": {
            "long": {"n": len(lt), "wr": round(lwr, 1), "pnl": round(sum(t["pnl"] for t in lt), 2)},
            "short": {"n": len(st), "wr": round(swr, 1), "pnl": round(sum(t["pnl"] for t in st), 2)},
        },
        "vix_regime": {
            "trend": {"n": len(tt), "wr": round(twr, 1), "pnl": round(sum(t["pnl"] for t in tt), 2)},
            "counter": {"n": len(ct), "wr": round(cwr, 1), "pnl": round(sum(t["pnl"] for t in ct), 2)},
        },
        "exit_types": {k: {"n": v["n"], "wr": round((v["w"]/v["n"]*100) if v["n"]>0 else 0, 1), "pnl": round(v["pnl"], 2)} for k, v in exit_stats.items()},
        "yearly": {k: round(v, 2) for k, v in sorted(res["yearly"].items())},
        "monthly": {k: round(v, 2) for k, v in sorted(res["monthly"].items())},
        "all_trades": trades,
        "equity": {k: round(v, 2) for k, v in list(res["equity"].items())[-500:]},
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n[1/3] DOWNLOADING 1-MINUTE DATA (Polygon.io)...")
    print("  (First run downloads & caches; subsequent runs use cache)")
    
    # Download 1-min bars for SPY, QQQ, IWM, DIA
    spy_1m = download_with_cache("SPY", START_DATE, END_DATE, "minute")
    qqq_1m = download_with_cache("QQQ", START_DATE, END_DATE, "minute")
    iwm_1m = download_with_cache("IWM", START_DATE, END_DATE, "minute")
    dia_1m = download_with_cache("DIA", START_DATE, END_DATE, "minute")
    
    # VIX daily (no meaningful 1-min data for regime detection)
    vix_daily = download_daily_with_cache("VIX", START_DATE, END_DATE)  # Use "VIX" ticker on Polygon
    if vix_daily.empty:
        # Polygon uses different VIX tickers
        print("  Trying I:VIX...")
        time.sleep(13)
        vix_daily = download_daily_with_cache("I:VIX", START_DATE, END_DATE)
    
    if spy_1m.empty:
        print("\n  ERROR: No SPY data downloaded. Check API key and plan.")
        sys.exit(1)
    
    print(f"\n  SPY 1-min bars: {len(spy_1m):,}")
    print(f"  QQQ 1-min bars: {len(qqq_1m):,}")
    print(f"  IWM 1-min bars: {len(iwm_1m):,}")
    print(f"  DIA 1-min bars: {len(dia_1m):,}")
    print(f"  VIX daily bars: {len(vix_daily):,}")
    
    print("\n[2/3] COMPUTING INDICATORS...")
    spy_1m, spy_5m, spy_daily, peer_pcts, vix_d = compute_daily_indicators(
        spy_1m, vix_daily, qqq_1m, iwm_1m, dia_1m
    )
    
    print("\n[3/3] RUNNING 1-MIN BAR-BY-BAR BACKTEST...")
    results = run_backtest(spy_1m, spy_5m, spy_daily, peer_pcts, vix_d)
    
    # Print and save
    json_results = print_results(results)
    
    output_path = Path(__file__).parent / "backtest_1min_results.json"
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n  Results saved: {output_path}")
    print(f"  Cache dir: {CACHE_DIR}")
    print("=" * 70)
