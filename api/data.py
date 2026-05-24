"""
Vercel Serverless API — /api/data
MES Futures Signal Engine — 7-Layer Score Engine
Google SSO only + Session Cookie (Last Update: 2026-05-25)
"""
import math, json, os, time, traceback, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler
from datetime import datetime, timedelta
import pytz, requests

NY = pytz.timezone("America/New_York")
import pandas as pd
import numpy as np
from engines.score_engine import run_score_engine


STARTING_BALANCE = 10000.0
TRADING_START_DATE = "2026-05-25"   # Day 1 of MES futures paper trading

# MES Futures Contract Specs
ES_MULTIPLIER    = 5.0      # $5/pt (Micro E-mini S&P 500)
ES_COMMISSION_RT = 0.50     # Round-trip commission per contract
ES_SLIPPAGE_PTS  = 0.25     # 1 tick slippage per side
ES_DAY_MARGIN    = 50.0     # Day-trading margin per contract
ES_TICK_SIZE     = 0.25     # Minimum price increment
ATR_SL_MULT      = 1.5      # SL = 1.5x ATR proxy (range-based)
RISK_PCT         = 0.12     # 12% Kelly-informed risk per trade
DAILY_LOSS_LIMIT = 0.06     # Halt trading if daily drawdown > 6%
MAX_OPEN_TRADES  = 1        # Max 1 MES position simultaneously

# ── Backtest Summary (embedded static data — no file read at runtime) ─
BACKTEST_SUMMARY = {
    "mes_futures": {
        "model": "MES Futures (ES backtest scaled to MES)",
        "period": "2023-03-25 ~ 2026-03-25",
        "period_days": 1095,
        "strategy": "ATR SL=1.5x | 2:1 RR | Risk=12% | Entry 10:30 EST",
        "total_trades": 78,
        "wins": 52,
        "losses": 26,
        "win_rate": 66.7,
        "profit_factor": 2.91,
        "avg_win_mes": 26.36,
        "avg_loss_mes": -18.11,
        "max_drawdown_pct": 4.6,
        "annual_return_pct": 23.8,
        "total_pnl_pct": 90.0,
        "sharpe": 3.0,
        "note": "ES $50/pt backtest. MES is 1/10th ($5/pt)."
    },
    "debit_spread_v3": {
        "model": "SPY 0DTE Debit Spread v3 (legacy reference)",
        "period_days": 750,
        "total_trades": 122,
        "win_rate": 74.6,
        "profit_factor": 2.55,
        "max_drawdown_pct": 21.1,
        "total_pnl_pct": 570.0,
        "sharpe": 5.54,
        "note": "Legacy SPY 0DTE strategy — reference only."
    }
}



ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_HEADERS = {
    "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
    "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", ""),
}
FLASHALPHA_API_KEY = os.getenv("FLASHALPHA_API_KEY", "")
FLASHALPHA_API_URL = "https://lab.flashalpha.com/v1"
_VIX_CACHE = {"at": 0.0, "vix": 18.0, "vix3m": None}
VIX_CACHE_SEC = int(os.getenv("VIX_CACHE_SEC", "45"))
YAHOO_UA = "Mozilla/5.0 (compatible; ESFutures/2.0)"


class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if pd.isna(obj): return None
        except (TypeError, ValueError): pass
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, pd.Timestamp): return str(obj)
        return super().default(obj)


# ── Alpaca Data Fetchers ────────────────────────────────────────────

def _alpaca_snapshots(symbols):
    """Fetch latest snapshots for multiple stock symbols."""
    url = f"{ALPACA_DATA_URL}/v2/stocks/snapshots"
    r = requests.get(url, headers=ALPACA_HEADERS,
                     params={"symbols": ",".join(symbols), "feed": "iex"}, timeout=5)
    r.raise_for_status()
    return r.json()


def _alpaca_bars(symbol, timeframe="5Min"):
    """Fetch intraday bars for technical analysis (capped for latency)."""
    now = datetime.now(NY)
    start = now.replace(hour=4, minute=0, second=0, microsecond=0)
    minutes = max(30, int((now - start).total_seconds() // 60))
    bar_limit = min(160, max(40, minutes // 5 + 25))
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    r = requests.get(url, headers=ALPACA_HEADERS, params={
        "timeframe": timeframe, "start": start.isoformat(),
        "limit": bar_limit, "adjustment": "raw", "feed": "iex",
    }, timeout=5)
    r.raise_for_status()
    bars = r.json().get("bars", [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                             "c": "Close", "v": "Volume", "t": "Timestamp"})
    return df


def _vix_fallback():
    """Fast VIX/VIX3M via Yahoo quote API + short TTL cache."""
    now = time.time()
    if now - _VIX_CACHE["at"] < VIX_CACHE_SEC:
        return _VIX_CACHE["vix"], _VIX_CACHE["vix3m"]

    vix_p, vix3m_p = _VIX_CACHE["vix"], _VIX_CACHE["vix3m"]
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v7/finance/quote",
            params={"symbols": "^VIX,^VIX3M"},
            headers={"User-Agent": YAHOO_UA},
            timeout=4,
        )
        if r.status_code == 200:
            for q in r.json().get("quoteResponse", {}).get("result", []):
                sym = q.get("symbol")
                px = q.get("regularMarketPrice")
                if px is None:
                    continue
                if sym == "^VIX":
                    vix_p = float(px)
                elif sym == "^VIX3M":
                    vix3m_p = float(px)
    except Exception:
        pass

    _VIX_CACHE.update({"at": now, "vix": vix_p, "vix3m": vix3m_p})
    return vix_p, vix3m_p


def _flashalpha_spy_summary():
    """Fetch SPY summary data from FlashAlpha API (volume, VWAP, etc.)."""
    try:
        url = f"{FLASHALPHA_API_URL}/stock/spy/summary"
        r = requests.get(
            url,
            headers={"X-Api-Key": FLASHALPHA_API_KEY},
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            return {
                "price": data.get("price"),
                "vwap": data.get("vwap"),
                "open": data.get("open"),
                "high": data.get("high"),
                "low": data.get("low"),
                "volume": data.get("volume"),
                "bid": data.get("bid"),
                "ask": data.get("ask"),
                "spread": data.get("spread"),
                "update_time": data.get("update_time"),
            }
    except Exception as e:
        pass
    return None


def get_market_status(dt):
    """
    Returns market status for smart API usage:
      'regular'      → 9:30 ~ 16:00 ET (full live)
      'pre_market'   → 8:30 ~ 9:30 ET (1h before open)
      'after_hours'  → 16:00 ~ 17:00 ET (1h after close)
      'closed'       → everything else (including weekends & holidays)

    This allows us to:
    - Keep fresh data during extended hours (pre/after)
    - Aggressively reduce calls deep into the night / weekends
    """
    # 1. Weekends
    if dt.weekday() >= 5:
        return 'closed'

    # 2. NYSE holidays 2026
    nyse_holidays_2026 = {
        datetime(2026, 1, 1).date(),
        datetime(2026, 1, 19).date(),
        datetime(2026, 2, 16).date(),
        datetime(2026, 4, 3).date(),
        datetime(2026, 5, 25).date(),
        datetime(2026, 6, 19).date(),
        datetime(2026, 7, 3).date(),
        datetime(2026, 9, 7).date(),
        datetime(2026, 11, 26).date(),
        datetime(2026, 12, 25).date(),
    }
    if dt.date() in nyse_holidays_2026:
        return 'closed'

    t_min = dt.hour * 60 + dt.minute

    # 3. Regular trading hours
    if 570 <= t_min <= 960:           # 9:30 ~ 16:00
        return 'regular'

    # 4. Pre-market (1 hour before)
    if 510 <= t_min < 570:            # 8:30 ~ 9:30
        return 'pre_market'

    # 5. After-hours (1 hour after)
    if 960 < t_min <= 1020:           # 16:00 ~ 17:00
        return 'after_hours'

    # 6. Deep night / early morning
    return 'closed'


def is_market_open(dt):
    """Legacy helper — returns True only during regular hours (for backward compatibility)."""
    return get_market_status(dt) == 'regular'

_MARKET_DATA_CACHE = {
    "fetched_at": 0.0,
    "snaps": {},
    "spy_h": None,
    "vix": (18.0, None),
    "flashalpha": None,
    "timing_ms": {}
}

def _fetch_market_bundle(all_stocks):
    """Parallel market data fetch (snapshots, bars, VIX, portfolio, FlashAlpha) with dynamic caching."""
    global _MARKET_DATA_CACHE
    
    now = datetime.now(NY)
    status = get_market_status(now)

    if status in ('regular', 'pre_market', 'after_hours'):
        # Treat extended hours (1h before/after) the same as regular for freshness
        ttl = 10
        fetch_options = True
    else:
        # Fully closed (night, weekends, holidays) → very aggressive caching
        ttl = 7200          # 2 hours
        fetch_options = False
    
    current_time = time.time()
    use_cache = False
    if _MARKET_DATA_CACHE["fetched_at"] > 0:
        elapsed = current_time - _MARKET_DATA_CACHE["fetched_at"]
        if elapsed < ttl:
            use_cache = True
            
    if use_cache:
        # Build cached output
        out = {
            "snaps": _MARKET_DATA_CACHE["snaps"].copy(),
            "spy_h": _MARKET_DATA_CACHE["spy_h"].copy(),
            "vix": _MARKET_DATA_CACHE["vix"],
            "flashalpha": _MARKET_DATA_CACHE["flashalpha"],
        }
        t0 = time.perf_counter()
        out["portfolio"] = load_portfolio()
        
        # Mix in cached timing info + fresh portfolio time
        out["timing_ms"] = _MARKET_DATA_CACHE["timing_ms"].copy()
        out["timing_ms"]["portfolio_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out["timing_ms"]["cached"] = True
        return out

    # Otherwise fetch fresh
    timing = {}
    out = {"snaps": {}, "spy_h": pd.DataFrame(), "vix": (18.0, None), "portfolio": _default_pf(), "flashalpha": None}

    def _timed(name, fn, *args):
        t0 = time.perf_counter()
        try:
            return fn(*args)
        finally:
            timing[name] = round((time.perf_counter() - t0) * 1000, 1)

    tasks = {
        "snaps": lambda: _timed("snapshots_ms", _alpaca_snapshots, all_stocks),
        "spy_h": lambda: _timed("bars_ms", _alpaca_bars, "SPY", "5Min"),
        "vix": lambda: _timed("vix_ms", _vix_fallback),
        "flashalpha": lambda: _timed("flashalpha_ms", _flashalpha_spy_summary),
        "portfolio": lambda: _timed("portfolio_ms", load_portfolio),
    }
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fn): key for key, fn in tasks.items()}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                out[key] = fut.result()
            except Exception as exc:
                timing[f"{key}_error"] = str(exc)
                
    # Update cache
    _MARKET_DATA_CACHE["fetched_at"] = current_time
    _MARKET_DATA_CACHE["snaps"] = out["snaps"].copy() if out["snaps"] else {}
    _MARKET_DATA_CACHE["spy_h"] = out["spy_h"].copy()
    _MARKET_DATA_CACHE["vix"] = out["vix"]
    _MARKET_DATA_CACHE["flashalpha"] = out["flashalpha"]
    _MARKET_DATA_CACHE["timing_ms"] = timing.copy()
    
    out["timing_ms"] = timing
    return out


def _snap_price(snap, key="latestTrade"):
    """Extract price from an Alpaca snapshot safely."""
    try: return float(snap[key]["p"])
    except Exception: return 0.0

def _snap_prev_close(snap):
    try: return float(snap["prevDailyBar"]["c"])
    except Exception: return 0.0

def _pct(price, prev):
    return ((price / prev) - 1) * 100 if prev > 0 else 0.0


# ── ES Futures Order Flow Simulator ───────────────────────────────

def _calculate_es_order_flow(spy_price, vix_price, normalized_score, direction_bias):
    """Generate ES Futures Depth of Market (DOM) simulation.
    Produces 11 tick levels around current price at 0.25pt resolution.
    VIX-adaptive liquidity depth + Cumulative Volume Delta."""
    if spy_price is None or spy_price <= 0:
        return None
    
    vix = vix_price if vix_price and vix_price > 0 else 18.0
    
    # Center price to nearest ES tick (0.25 pt)
    center = round(spy_price * 4) / 4.0
    
    # VIX-adaptive base liquidity (higher VIX = thinner books)
    if vix >= 30:
        base_size = 40
    elif vix >= 25:
        base_size = 80
    elif vix >= 20:
        base_size = 120
    else:
        base_size = 200
    
    levels = []
    for i in range(-5, 6):
        price = round(center + i * 0.25, 2)
        # Distance from center affects depth (closer = thicker)
        dist = abs(i)
        depth_factor = max(0.3, 1.0 - dist * 0.12)
        
        bid_size = max(5, int(base_size * depth_factor * (0.7 + random.random() * 0.6)))
        ask_size = max(5, int(base_size * depth_factor * (0.7 + random.random() * 0.6)))
        
        # Bias: stronger signal shifts liquidity
        if direction_bias == "CALL":  # Bullish
            if i < 0:  # Below center: more bids (support)
                bid_size = int(bid_size * 1.3)
            else:  # Above center: thinner asks (less resistance)
                ask_size = int(ask_size * 0.8)
        elif direction_bias == "PUT":  # Bearish
            if i > 0:  # Above center: more asks (resistance)
                ask_size = int(ask_size * 1.3)
            else:  # Below center: thinner bids (less support)
                bid_size = int(bid_size * 0.8)
        
        is_current = (i == 0)
        levels.append({
            "price": price,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "is_current": is_current,
        })
    
    # CVD (Cumulative Volume Delta): net buying - selling pressure
    total_bids = sum(l["bid_size"] for l in levels)
    total_asks = sum(l["ask_size"] for l in levels)
    cvd_raw = total_bids - total_asks
    cvd_pct = round((cvd_raw / max(total_bids + total_asks, 1)) * 100, 1)
    
    return {
        "levels": levels,
        "center_price": center,
        "total_bids": total_bids,
        "total_asks": total_asks,
        "cvd": cvd_raw,
        "cvd_pct": cvd_pct,
        "max_depth": max(max(l["bid_size"] for l in levels), max(l["ask_size"] for l in levels)),
        "bias": direction_bias,
        "vix_regime": "HIGH" if vix >= 25 else ("ELEVATED" if vix >= 20 else "NORMAL"),
    }






PORTFOLIO_STORAGE_KEY = os.getenv("PORTFOLIO_STORAGE_KEY", "arungun_portfolio")
MAX_TRADE_HISTORY = 250

def _default_pf():
    return {"cash": STARTING_BALANCE, "positions": {}, "history": [], "trade_log": [], "recent_trades": [], "initial_balance": STARTING_BALANCE, "current_value": STARTING_BALANCE, "total_return_pct": 0.0}

def _normalize_pf(pf):
    base = _default_pf()
    if not isinstance(pf, dict):
        return base
    base.update(pf)
    # ── ES Futures Migration: reset old options-era portfolios ──
    old_bal = base.get("initial_balance", 0)
    if old_bal < STARTING_BALANCE and old_bal > 0:
        # Old options portfolio ($500) → reset to ES futures ($10,000)
        base = _default_pf()
        base["_migrated_from_options"] = True
    base["positions"] = base.get("positions") or {}
    base["history"] = base.get("history") or []
    base["trade_log"] = base.get("trade_log") or []

    # Auto-recover: rebuild trade_log from history when trade_log is empty
    if not base["trade_log"] and base["history"]:
        recovered = []
        for h in base["history"]:
            if not isinstance(h, dict):
                continue
            h = _ensure_trade_id(h)
            is_closed = _is_closed_record(h)
            evt = {
                "event": "CLOSE" if is_closed else "OPEN",
                "trade_id": h.get("trade_id"),
                "date": h.get("date"),
                "entry_time": h.get("entry_time") or h.get("time"),
                "exit_time": h.get("exit_time"),
                "direction": h.get("direction"),
                "es_direction": h.get("es_direction"),
                "instrument": h.get("instrument", "MES"),
                "entry_price": h.get("entry_price", h.get("entry_spy")),
                "exit_price": h.get("exit_price", h.get("exit_spy")),
                "contracts": h.get("contracts"),
                "sl_price": h.get("sl_price"),
                "tp_price": h.get("tp_price"),
                "margin_locked": h.get("margin_locked"),
                "exit_type": h.get("exit_type"),
                "pnl": h.get("pnl"),
                "realized_pnl": h.get("realized_pnl"),
                "pnl_pct": h.get("pnl_pct"),
                "win": h.get("win"),
                "logged_at": h.get("exit_ts") or h.get("entry_ts"),
            }
            recovered.append(evt)
        if recovered:
            base["trade_log"] = recovered
    base["recent_trades"] = base.get("recent_trades") or _build_recent_trades(base)
    return base

def _kv_credentials():
    """Resolve Upstash KV credentials from either naming convention."""
    url = (os.getenv("UPSTASH_REDIS_REST_URL")
           or os.getenv("KV_REST_API_URL", "")).rstrip("/")
    token = (os.getenv("UPSTASH_REDIS_REST_TOKEN")
             or os.getenv("KV_REST_API_TOKEN", ""))
    return (url, token) if url and token else (None, None)

def _storage_backend():
    url, token = _kv_credentials()
    if url and token:
        return "upstash"
    return "local"


# Use /tmp on Vercel serverless (read-only project filesystem)
LOCAL_PORTFOLIO_FILE = os.path.join("/tmp" if os.getenv("VERCEL") else ".", "portfolio.json")

def _fetch_local_portfolio():
    if os.path.exists(LOCAL_PORTFOLIO_FILE):
        try:
            with open(LOCAL_PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "cash" in data:
                    return data
        except Exception:
            pass
    return None

def _write_local_portfolio(pf):
    payload_pf = {
        k: v for k, v in pf.items()
        if not str(k).startswith("_") and k != "recent_trades"
    }
    try:
        with open(LOCAL_PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(payload_pf, f, cls=SafeEncoder, indent=2)
        return True
    except Exception:
        return False

def _fetch_raw_portfolio(retries=3):
    """Load portfolio JSON from persistent storage with retries, falling back to local file."""
    for attempt in range(retries):
        try:
            if _storage_backend() == "upstash":
                base, token = _kv_credentials()
                r = requests.get(
                    f"{base}/get/{PORTFOLIO_STORAGE_KEY}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=6,
                )
                if r.status_code == 200:
                    raw = r.json().get("result")
                    if raw:
                        data = json.loads(raw) if isinstance(raw, str) else raw
                        if isinstance(data, dict) and "cash" in data:
                            data["_storage"] = "upstash"
                            _write_local_portfolio(data)
                            return data
        except Exception:
            pass
        time.sleep(0.15 * (attempt + 1))

    # Fallback to local copy
    local_data = _fetch_local_portfolio()
    if local_data:
        local_data["_storage"] = "local_file"
        return local_data

    return None


def _write_raw_portfolio(pf):
    payload_pf = {
        k: v for k, v in pf.items()
        if not str(k).startswith("_") and k != "recent_trades"
    }
    # Always write locally first to guarantee persistence
    local_ok = _write_local_portfolio(pf)

    remote_ok = False
    last_err = None

    if _storage_backend() == "upstash":
        try:
            base, token = _kv_credentials()
            raw = json.dumps(payload_pf, cls=SafeEncoder)
            r = requests.post(
                base,
                json=["SET", PORTFOLIO_STORAGE_KEY, raw],
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            r.raise_for_status()
            remote_ok = True
        except Exception as e:
            print(f"Remote portfolio save failed: {e}. Local copy is safe.")
            last_err = e
    else:
        # Local mode requires no remote write
        remote_ok = True

    if local_ok:
        if not remote_ok and last_err:
            pf["_save_error"] = f"Remote failed ({last_err}), but local copy saved successfully."
        pf["_remote_ok"] = remote_ok
        return True

    # Remote saved but local failed (e.g., Vercel read-only filesystem)
    if remote_ok:
        pf["_remote_ok"] = True
        return True

    if last_err:
        raise last_err
    raise IOError("Failed to save portfolio to any backend.")



def load_portfolio():
    raw = _fetch_raw_portfolio()
    if raw:
        pf = _normalize_pf(raw)
        pf["_storage"] = raw.get("_storage", _storage_backend())
        pf["_load_ok"] = True
        return pf
    pf = _default_pf()
    pf["_load_ok"] = False
    pf["_storage"] = _storage_backend()
    return pf

def _trade_recency_key(item):
    return item.get("exit_ts") or item.get("entry_ts") or item.get("logged_at") or f"{item.get('date', '')} {item.get('exit_time') or item.get('entry_time') or item.get('time', '')}"


def _is_closed_record(item):
    return item.get("status") == "CLOSED" or item.get("event") == "CLOSE" or item.get("pnl_locked")


def _trade_signature(record):
    if not isinstance(record, dict):
        return None
    tid = record.get("trade_id")
    if tid:
        return str(tid)
    entry = record.get("entry_time") or record.get("time")
    return f"{record.get('date')}-{entry}-{record.get('direction')}-{record.get('entry_price', record.get('K_buy'))}-{record.get('es_direction', '')}"


def _ensure_trade_id(record):
    if not isinstance(record, dict):
        return record
    if not record.get("entry_time") and record.get("time"):
        record["entry_time"] = record["time"]
    if not record.get("trade_id"):
        entry = record.get("entry_time") or record.get("time") or "00:00"
        record["trade_id"] = f"{record.get('date')}-{entry}-ES-{record.get('es_direction', record.get('direction', 'UNK'))}"
    return record


def _finalize_closed_record(record):
    """Freeze realized P&L — closed trades must never be mark-to-market again."""
    row = _ensure_trade_id(dict(record))
    row["status"] = "CLOSED"
    row["pnl_locked"] = True
    if row.get("pnl") is not None:
        row["realized_pnl"] = row["pnl"]
    for key in ("unrealized_pnl", "unrealized_pnl_pct", "mark_spy", "mark_time", "current_val"):
        row.pop(key, None)
    return row


def _trade_already_closed(pf, trade_id=None, record=None):
    sig = trade_id or _trade_signature(record)
    if not sig:
        return False
    for h in pf.get("history") or []:
        if _is_closed_record(h) and (_trade_signature(h) == sig or (trade_id and h.get("trade_id") == trade_id)):
            return True
    for e in pf.get("trade_log") or []:
        if e.get("event") == "CLOSE" and (_trade_signature(e) == sig or (trade_id and e.get("trade_id") == trade_id)):
            return True
    return False


def _merge_trade_records(items):
    """Merge closed trades by trade_id; first locked close wins (P&L never changes)."""
    merged = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item = _ensure_trade_id(item)
        tid = _trade_signature(item)
        item = _finalize_closed_record(item) if _is_closed_record(item) else dict(item)
        prev = merged.get(tid)
        if not prev:
            merged[tid] = item
            continue
        if prev.get("pnl_locked"):
            continue
        if item.get("pnl_locked"):
            merged[tid] = item
            continue
        if _trade_recency_key(item) >= _trade_recency_key(prev):
            merged[tid] = item
    return sorted(merged.values(), key=_trade_recency_key, reverse=True)


def _merge_trade_events(items):
    """Keep every OPEN/CLOSE ledger event (do not collapse by trade_id)."""
    merged = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = f"{item.get('trade_id')}-{item.get('event')}-{item.get('logged_at')}"
        merged[key] = item
    return sorted(merged.values(), key=_trade_recency_key, reverse=True)


def _cap_list(items, limit=MAX_TRADE_HISTORY):
    return list(items)[:limit] if items else []


def _row_completeness(row):
    score = 0
    if row.get("entry_time") or row.get("time"):
        score += 4
    if row.get("exit_time"):
        score += 2
    if row.get("pnl") is not None:
        score += 2
    if row.get("trade_id"):
        score += 1
    return score


def _ledger_dedupe_key(row):
    st = row.get("display_status") or row.get("event") or "?"
    tid = row.get("trade_id")
    if tid:
        return f"{st}:{tid}"
    return (
        f"{st}:{row.get('date')}:{row.get('direction')}:"
        f"{row.get('entry_price', row.get('K_buy'))}:{row.get('es_direction', '')}:{row.get('exit_time') or row.get('logged_at', '')}"
    )


def _dedupe_ledger_rows(rows):
    best = {}
    for row in rows:
        key = _ledger_dedupe_key(row)
        prev = best.get(key)
        if not prev or _row_completeness(row) >= _row_completeness(prev):
            best[key] = row
    return sorted(best.values(), key=_trade_recency_key, reverse=True)


def _has_close_row(rows, record):
    tid = record.get("trade_id")
    if tid and any(r.get("trade_id") == tid and r.get("display_status") == "CLOSE" for r in rows):
        return True
    sig = (record.get("date"), record.get("direction"), record.get("entry_price", record.get("K_buy")), record.get("es_direction", ""), record.get("exit_time"))
    for r in rows:
        if r.get("display_status") != "CLOSE":
            continue
        if (r.get("date"), r.get("direction"), r.get("entry_price", r.get("K_buy")), r.get("es_direction", ""), r.get("exit_time")) == sig:
            return True
    return False


def _append_history(pf, record):
    if _is_closed_record(record):
        record = _ensure_trade_id(record)
        if _trade_already_closed(pf, record.get("trade_id"), record):
            return
        record = _finalize_closed_record(record)
    hist = pf.setdefault("history", [])
    hist.insert(0, record)
    pf["history"] = _cap_list(hist)


def _append_trade_event(pf, event):
    if event.get("event") == "CLOSE":
        event = _ensure_trade_id(event)
        if _trade_already_closed(pf, event.get("trade_id"), event):
            return
        event = _finalize_closed_record(event)
    log = pf.setdefault("trade_log", [])
    event["logged_at"] = datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S")
    log.insert(0, event)
    pf["trade_log"] = _cap_list(log)


def _build_recent_trades(pf):
    """Ledger-style feed: each OPEN and CLOSE event is its own row."""
    rows = []
    open_logged = set()

    for e in pf.get("trade_log") or []:
        ev = e.get("event")
        if ev not in ("OPEN", "CLOSE"):
            continue
        row = _finalize_closed_record(e) if ev == "CLOSE" else dict(e)
        row["display_status"] = ev
        if ev == "OPEN" and row.get("trade_id"):
            open_logged.add(row["trade_id"])
        rows.append(row)

    for pos in (pf.get("positions") or {}).values():
        tid = pos.get("trade_id")
        if tid and tid in open_logged:
            continue
        row = dict(pos)
        row["display_status"] = "OPEN"
        row["event"] = "OPEN"
        row["status"] = "OPEN"
        rows.append(row)

    for h in pf.get("history") or []:
        if _has_close_row(rows, h):
            continue
        row = _finalize_closed_record(h)
        row["display_status"] = "CLOSE"
        row["event"] = "CLOSE"
        rows.append(row)

    return _cap_list(_dedupe_ledger_rows(rows))

def _close_trade(pos, now, spy_p, exit_val, exit_type):
    """Close an ES futures position (used for stale cleanup)."""
    contracts = int(pos.get("contracts", 0) or 0)
    entry_price = pos.get("entry_price", 0)
    es_dir = pos.get("es_direction", "LONG")
    
    if es_dir == "LONG":
        point_pnl = (spy_p - entry_price) if spy_p else 0
    else:
        point_pnl = (entry_price - spy_p) if spy_p else 0
    
    pnl = round(point_pnl * ES_MULTIPLIER * contracts - ES_COMMISSION_RT * contracts, 2)
    margin_locked = pos.get("margin_locked", ES_DAY_MARGIN * contracts)
    pnl_pct = round((pnl / margin_locked) * 100, 1) if margin_locked > 0 else 0.0
    
    pos.update({
        "status": "CLOSED",
        "exit_time": now.strftime("%H:%M"),
        "exit_ts": now.strftime("%Y-%m-%d %H:%M:%S"),
        "exit_price": round(spy_p, 2) if spy_p else None,
        "exit_val": round(float(exit_val), 2) if exit_val else 0,
        "exit_type": exit_type,
        "revenue": round(margin_locked + pnl, 2),
        "pnl": pnl,
        "realized_pnl": pnl,
        "pnl_pct": pnl_pct,
        "pnl_locked": True,
        "win": pnl > 0,
    })
    for key in ("unrealized_pnl", "unrealized_pnl_pct", "mark_spy", "mark_time", "current_val", "current_price", "point_pnl"):
        pos.pop(key, None)
    return pos



def _entry_criteria_met(grade, direction_bias, score_result, portfolio=None, now=None):
    """MES Futures entry criteria — no PDT restriction for futures."""
    layers = score_result.get("layers", {})

    # Risk lock or locked signal
    if layers.get("risk", {}).get("passed") is False or grade == "LOCKED":
        return False

    # Must be STRONG signal
    if grade != "STRONG":
        return False

    # Must be in PRIME or GAMMA time window (score 20 = prime)
    if layers.get("time_window", {}).get("score", 0) < 20:
        return False

    # Must have clear directional bias
    if direction_bias not in ("CALL", "PUT"):
        return False

    # Daily loss limit halt — stop trading if down > 6% today
    if portfolio:
        initial = float(portfolio.get("initial_balance", STARTING_BALANCE) or STARTING_BALANCE)
        current = float(portfolio.get("current_value", initial) or initial)
        daily_dd = (initial - current) / initial if initial > 0 else 0
        if daily_dd >= DAILY_LOSS_LIMIT:
            return False

    # Max 1 open position
    if portfolio and len(portfolio.get("positions", {})) >= MAX_OPEN_TRADES:
        return False

    return True







def _position_invalid_reason(open_pos, grade, direction_bias, score_result):
    """
    Exit rules use hysteresis — looser than entry.
    Entry needs STRONG; exit on WEAK/NONE, direction flip, risk lock, or weak time window.
  """
    layers = score_result.get("layers", {})
    if layers.get("risk", {}).get("passed") is False or grade == "LOCKED":
        return "RISK"
    if direction_bias not in ("CALL", "PUT"):
        return "DIRECTION"
    if open_pos.get("direction") != direction_bias:
        return "DIRECTION"
    if grade in ("NONE", "WEAK"):
        return "SIGNAL"
    if layers.get("time_window", {}).get("score", 0) < 10:
        return "TIME_WINDOW"
    return None


def _record_position_close(portfolio, open_pos, today_str, now, spy_p, exit_val, exit_type):
    open_pos = _ensure_trade_id(open_pos)
    if _trade_already_closed(portfolio, open_pos.get("trade_id"), open_pos):
        if today_str in portfolio.get("positions", {}):
            del portfolio["positions"][today_str]
        return
    open_pos = _close_trade(open_pos, now, spy_p, exit_val, exit_type)
    portfolio["cash"] += open_pos["revenue"]
    _append_history(portfolio, open_pos.copy())
    _append_trade_event(portfolio, {
        "event": "CLOSE",
        "trade_id": open_pos.get("trade_id"),
        "date": open_pos.get("date"),
        "entry_time": open_pos.get("entry_time"),
        "exit_time": open_pos.get("exit_time"),
        "direction": open_pos.get("direction"),
        "es_direction": open_pos.get("es_direction"),
        "instrument": open_pos.get("instrument", "MES"),
        "contracts": open_pos.get("contracts"),
        "entry_price": open_pos.get("entry_price"),
        "exit_price": open_pos.get("exit_price"),
        "exit_type": open_pos.get("exit_type"),
        "margin_locked": open_pos.get("margin_locked"),
        "revenue": open_pos.get("revenue"),
        "pnl": open_pos.get("pnl"),
        "pnl_pct": open_pos.get("pnl_pct"),
        "win": open_pos.get("win"),
    })
    del portfolio["positions"][today_str]

def save_portfolio(pf):
    try:
        remote = _fetch_raw_portfolio() or {}
        remote_pf = _normalize_pf(remote) if remote else _default_pf()

        merged_history = _merge_trade_records((remote_pf.get("history") or []) + (pf.get("history") or []))
        merged_log = _merge_trade_events((remote_pf.get("trade_log") or []) + (pf.get("trade_log") or []))

        # Merge history — never let merge produce fewer records than local
        local_hist = pf.get("history") or []
        finalized_hist = _cap_list([
            _finalize_closed_record(h) if _is_closed_record(h) else h for h in merged_history
        ])
        pf["history"] = finalized_hist if len(finalized_hist) >= len(local_hist) else local_hist

        # Merge trade_log — never let merge produce fewer records than local
        local_log = pf.get("trade_log") or []
        finalized_log = _cap_list(merged_log)
        pf["trade_log"] = finalized_log if len(finalized_log) >= len(local_log) else local_log

        if remote_pf.get("initial_balance") and not pf.get("_seeded_balance"):
            pf.setdefault("initial_balance", remote_pf["initial_balance"])

        pf["recent_trades"] = _build_recent_trades(pf)
        pf["current_value"] = pf["cash"]
        for p in pf.get("positions", {}).values():
            pf["current_value"] += float(p.get("cost", 0) or 0) + float(p.get("unrealized_pnl", 0) or 0)
        pf["total_return_pct"] = round(((pf["current_value"] / pf["initial_balance"]) - 1) * 100, 2)
        pf["revision"] = int(remote_pf.get("revision", 0) or 0) + 1
        pf["last_saved"] = datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S")

        # Only skip save if load failed AND we have zero local data AND
        # remote already had data (avoid wiping an existing remote store).
        # For fresh Upstash, _load_ok=False is expected — allow save.
        if (not pf.get("_load_ok")
                and not pf.get("history")
                and not pf.get("trade_log")
                and remote_pf.get("history")):
            pf["_save_skipped"] = "load_failed_empty_local"
            return False

        _write_raw_portfolio(pf)
        pf["_save_ok"] = True
        # Keep _save_error — provides partial failure info to frontend UI
        return True
    except Exception as e:
        pf["_save_error"] = str(e)
        pf["_save_ok"] = False
        return False


# ── Handler ─────────────────────────────────────────────────────────

_IN_MEM_LIMITS = {}

def check_rate_limit(ip, limit=15):
    """Client IP Rate Limiter. 15 requests per minute."""
    global _IN_MEM_LIMITS
    minute_str = datetime.now(NY).strftime("%Y%m%d%H%M")
    
    # 1. Try Upstash Redis Rate Limiting if available
    base, token = _kv_credentials()
    if base and token:
        try:
            key = f"rate_limit:{ip}:{minute_str}"
            url = f"{base}/pipeline"
            commands = [
                ["INCR", key],
                ["EXPIRE", key, "60"]
            ]
            r = requests.post(url, json=commands, headers={"Authorization": f"Bearer {token}"}, timeout=2)
            if r.status_code == 200:
                res = r.json()
                if isinstance(res, list) and len(res) > 0:
                    count = res[0].get("result", 1)
                    if isinstance(count, int) and count > limit:
                        return False
                    return True
        except Exception as e:
            print(f"[Rate Limit KV Error] {e}")
            # Fall back to in-memory on KV error
            
    # 2. In-Memory Fallback Rate Limiting
    # Clean up old keys from memory
    for client in list(_IN_MEM_LIMITS.keys()):
        _IN_MEM_LIMITS[client] = {k: v for k, v in _IN_MEM_LIMITS[client].items() if k == minute_str}
        if not _IN_MEM_LIMITS[client]:
            del _IN_MEM_LIMITS[client]
            
    if ip not in _IN_MEM_LIMITS:
        _IN_MEM_LIMITS[ip] = {}
        
    current_count = _IN_MEM_LIMITS[ip].get(minute_str, 0)
    if current_count >= limit:
        return False
        
    _IN_MEM_LIMITS[ip][minute_str] = current_count + 1
    return True

ALLOWED_ORIGINS = {
    "https://hannaealgo.vercel.app",
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}

def _verify_google_token(id_token):
    """Verify Google Sign-In JWT via Google tokeninfo endpoint."""
    if not id_token:
        return None
    try:
        r = requests.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={"id_token": id_token},
            timeout=8
        )
        if r.status_code != 200:
            print(f"[Google Token] tokeninfo HTTP {r.status_code}: {r.text[:200]}")
            return None
        payload = r.json()
        # Must be email-verified
        email_ok = payload.get("email_verified") in ("true", True)
        if not email_ok:
            print("[Google Token] email not verified")
            return None
        # Must be issued for our client_id
        expected_aud = os.getenv("GOOGLE_CLIENT_ID", "729700534302-3eaf1oulfa91mt75ootm5m2lohvibk5p.apps.googleusercontent.com")
        if payload.get("aud") != expected_aud:
            print(f"[Google Token] aud mismatch: {payload.get('aud')} vs {expected_aud}")
            return None
        return payload
    except Exception as e:
        print(f"[Google Token Error] {e}")
    return None

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        origin = self.headers.get("Origin", "")
        self.send_response(200)
        if origin in ALLOWED_ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
        else:
            self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-API-Key, x-api-key, Cookie')
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()

    def do_POST(self):
        path = self.path.split("?")[0]

        # Handle logout (clears the session cookie)
        if path == "/api/logout":
            origin = self.headers.get("Origin", "")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            if origin in ALLOWED_ORIGINS:
                self.send_header('Access-Control-Allow-Origin', origin)
            else:
                self.send_header('Access-Control-Allow-Origin', '*')
            # Clear the cookie immediately
            self.send_header('Set-Cookie', 'access_token=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict')
            self.send_header('Access-Control-Allow-Credentials', 'true')
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "message": "Logged out"}).encode('utf-8'))
            return

        # We only handle /api/unlock otherwise
        if path != "/api/unlock":
            self.send_response(404)
            self.end_headers()
            return

        origin = self.headers.get("Origin", "")
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            payload = json.loads(post_data) if post_data else {}
            
            credential = payload.get("credential")

            authorized = False

            # Google Credential Auth ONLY
            if credential:
                google_payload = _verify_google_token(credential)
                if google_payload:
                    email = google_payload.get("email")
                    allowed_emails_str = os.getenv("ALLOWED_EMAILS", "yoongun64@gmail.com")
                    allowed_emails = {e.strip().lower() for e in allowed_emails_str.split(",") if e.strip()}
                    if email and email.lower() in allowed_emails:
                        authorized = True
                        print(f"[Google Auth Success] Authorized {email}")
                    else:
                        print(f"[Google Auth Blocked] Email {email} not in ALLOWED_EMAILS")
                else:
                    print("[Google Auth] Token verification failed")
            else:
                print("[Unlock] No credential provided - password auth removed")
            
            if authorized:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                if origin in ALLOWED_ORIGINS:
                    self.send_header('Access-Control-Allow-Origin', origin)
                else:
                    self.send_header('Access-Control-Allow-Origin', '*')
                
                # Issue secure httpOnly session cookie (no Max-Age = cleared when browser fully closed)
                # + client-side 3-hour inactivity protection for "no activity" case.
                cookie_str = "access_token=valid; Path=/; HttpOnly; Secure; SameSite=Strict"
                self.send_header('Set-Cookie', cookie_str)
                self.send_header('Access-Control-Allow-Credentials', 'true')
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
                return
            else:
                self.send_response(401)
                self.send_header('Content-Type', 'application/json')
                if origin in ALLOWED_ORIGINS:
                    self.send_header('Access-Control-Allow-Origin', origin)
                else:
                    self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Credentials', 'true')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Unauthorized"}).encode('utf-8'))
                return
                
        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"[Error in POST /api/unlock] {err_msg}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            if origin in ALLOWED_ORIGINS:
                self.send_header('Access-Control-Allow-Origin', origin)
            else:
                self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Internal Server Error"}).encode('utf-8'))

    def do_GET(self):
        # Resolve client IP & Check Rate limit
        ip = self.headers.get("x-forwarded-for", "127.0.0.1").split(',')[0].strip()
        if not check_rate_limit(ip, 15):
            origin = self.headers.get("Origin", "")
            self.send_response(429)
            self.send_header('Content-Type', 'application/json')
            if origin in ALLOWED_ORIGINS:
                self.send_header('Access-Control-Allow-Origin', origin)
            else:
                self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Too Many Requests", "message": "Rate limit exceeded. Max 15 requests per minute."}).encode('utf-8'))
            return

        # API Authentication — Google SSO cookie ONLY (legacy unlock key removed)
        cookie_header = self.headers.get("Cookie", "")
        has_cookie = "access_token=valid" in cookie_header

        is_authed = has_cookie
        if not is_authed:
                origin = self.headers.get("Origin", "")
                self.send_response(401)
                self.send_header('Content-Type', 'application/json')
                if origin in ALLOWED_ORIGINS:
                    self.send_header('Access-Control-Allow-Origin', origin)
                else:
                    self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Credentials', 'true')
                self.end_headers()
                
                google_client_id = os.getenv("GOOGLE_CLIENT_ID", "729700534302-3eaf1oulfa91mt75ootm5m2lohvibk5p.apps.googleusercontent.com")
                self.wfile.write(json.dumps({
                    "error": "Unauthorized", 
                    "message": "Please sign in with Google",
                    "google_client_id": google_client_id
                }).encode('utf-8'))
                return

        start_time = time.perf_counter()
        now = datetime.now(NY)
        ts = now.strftime("%Y-%m-%d %H:%M:%S")

        STOCK_SYMS = ["SPY", "QQQ", "DIA", "IWM"]
        MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
        GME_STOCK = ["GME"]
        ALL_STOCKS = STOCK_SYMS + MAG7 + GME_STOCK

        try:
            bundle = _fetch_market_bundle(ALL_STOCKS)
            snaps = bundle["snaps"] or {}
            spy_h = bundle["spy_h"]
            vix_p, vix3m_p = bundle["vix"]
            portfolio = bundle["portfolio"]
            flashalpha_spy = bundle.get("flashalpha")
            fetch_timing = bundle.get("timing_ms", {})

            # Compute market status for response + frontend optimization
            status = get_market_status(now)

            spy_p = _snap_price(snaps.get("SPY", {}))
            spy_prev = _snap_prev_close(snaps.get("SPY", {})) or spy_p
            
            # FlashAlpha VWAP override if available
            vwap = spy_p  # default
            if flashalpha_spy and flashalpha_spy.get("vwap"):
                vwap = flashalpha_spy.get("vwap")

            # ── Percentage changes from snapshots ──
            pcts_data = {}
            for sym in STOCK_SYMS:
                s = snaps.get(sym, {})
                pcts_data[sym] = _pct(_snap_price(s), _snap_prev_close(s) or 1)

            # ── Compute VWAP / volume / range from bars ──
            vol_r, d_range = 0.0, 0.0
            if not spy_h.empty and not flashalpha_spy:
                # Only compute VWAP from bars if FlashAlpha data not available
                tp = (spy_h["High"] + spy_h["Low"] + spy_h["Close"]) / 3.0
                cum_vol = spy_h["Volume"].cumsum().replace(0, pd.NA)
                vwap_s = (spy_h["Volume"] * tp).cumsum() / cum_vol
                if not vwap_s.empty and pd.notna(vwap_s.iloc[-1]):
                    vwap = float(vwap_s.iloc[-1])
            
            if not spy_h.empty:
                vol_sma = spy_h["Volume"].rolling(window=20).mean()
                if not vol_sma.empty and pd.notna(vol_sma.iloc[-1]) and vol_sma.iloc[-1] > 0:
                    vol_r = float(spy_h["Volume"].iloc[-1] / vol_sma.iloc[-1])
                d_range = float(spy_h["High"].max() - spy_h["Low"].min())

            # ── CALL SCORE ENGINE (Modular 140-point system) ──
            t_min = now.hour * 60 + now.minute
            is_regular = 570 <= t_min <= 960
            score_result = run_score_engine(
                now_et=now,
                spy_price=spy_p,
                vix_price=vix_p,
                vix3m_price=vix3m_p,
                prev_close=spy_prev,
                vwap=vwap,
                vol_ratio=vol_r,
                range_value=d_range,
                pcts=pcts_data,
                spy_history=spy_h,
                portfolio=portfolio,
                session_name="REGULAR" if is_regular else "CLOSED",
            )
            
            # Extract variables for paper trading & output
            normalized = score_result["total_score"]
            raw_total = score_result["raw_score"]
            active_max = score_result["max_score"]
            direction_bias = score_result["direction_bias"]
            
            # Map new 140-point grade to legacy signal dict
            signal = score_result["signal"]
            grade = signal["grade"]

            # Daily drawdown tracking (for halt logic + response)
            initial_bal  = float(portfolio.get("initial_balance", STARTING_BALANCE) or STARTING_BALANCE)
            current_val  = float(portfolio.get("current_value", initial_bal) or initial_bal)
            daily_dd_pct = round((initial_bal - current_val) / initial_bal * 100, 2) if initial_bal > 0 else 0.0
            if daily_dd_pct >= DAILY_LOSS_LIMIT * 100:
                signal["action"] = f"⛔ DAILY LOSS LIMIT ({daily_dd_pct:.1f}%) — TRADING HALTED"
                signal["halted"] = True
            if grade == "STRONG": signal = {"grade": "STRONG", "label": "STRONG SIGNAL", "emoji": "🟢", "action": "Full position", "color": "#3dd68c"}
            elif grade == "MODERATE": signal = {"grade": "MODERATE", "label": "MODERATE SIGNAL", "emoji": "🟡", "action": "Half position", "color": "#f5c451"}
            elif grade == "WEAK": signal = {"grade": "WEAK", "label": "STANDBY", "emoji": "🟠", "action": "Monitor only", "color": "#f5a623"}
            else: signal = {"grade": "NONE", "label": "NO SIGNAL", "emoji": "🔴", "action": "No entry", "color": "#f07178"}

            # t_min/is_regular already computed above (L802-803)
            if not is_regular:
                signal["label"] = "MARKET CLOSED"
                signal["action"] = "Market not in session"

            # ── PAPER TRADING EXECUTION ──
            today_str = now.strftime("%Y-%m-%d")
            vix_val = vix_p if vix_p and vix_p > 0 else 18.0
            
            # 1. Clean up stale positions from previous days
            to_remove = []
            for date_key, pos in list(portfolio.get("positions", {}).items()):
                if date_key == today_str:
                    continue
                tid = pos.get("trade_id")
                pos = _ensure_trade_id(pos)
                tid = pos.get("trade_id")
                if _trade_already_closed(portfolio, tid, pos):
                    to_remove.append(date_key)
                    continue
                pos = _close_trade(pos, now, spy_p, 0, "STALE_EOD")
                portfolio["cash"] += pos["revenue"]
                _append_history(portfolio, pos)
                _append_trade_event(portfolio, {
                    "event": "CLOSE",
                    "trade_id": tid,
                    "date": pos.get("date"),
                    "entry_time": pos.get("entry_time"),
                    "exit_time": pos.get("exit_time"),
                    "direction": pos.get("direction"),
                    "es_direction": pos.get("es_direction"),
                    "entry_price": pos.get("entry_price"),
                    "exit_price": pos.get("exit_price"),
                    "exit_type": pos.get("exit_type"),
                    "margin_locked": pos.get("margin_locked"),
                    "revenue": pos.get("revenue"),
                    "pnl": pos.get("pnl"),
                    "realized_pnl": pos.get("realized_pnl"),
                    "pnl_pct": pos.get("pnl_pct"),
                    "win": pos.get("win"),
                })
                to_remove.append(date_key)
            if to_remove:
                for k in to_remove: del portfolio["positions"][k]

            # 2. Check for entry — ES Futures (direct contract, no spreads)
            open_pos = portfolio.get("positions", {}).get(today_str)
            if not open_pos and is_regular and _entry_criteria_met(grade, direction_bias, score_result, portfolio, now):
                cash = portfolio["cash"]
                es_direction = "LONG" if direction_bias == "CALL" else "SHORT"
                
                # ATR-based SL (use range as proxy)
                atr_proxy = max(d_range, 2.0) if d_range > 0 else 4.0
                sl_points = max(ATR_SL_MULT * atr_proxy, 2.0)
                sl_points = min(sl_points, 15.0)
                tp_points = sl_points * 2.0  # 2:1 R/R
                
                # Position sizing
                risk_per_contract = sl_points * ES_MULTIPLIER + ES_COMMISSION_RT
                max_risk = cash * RISK_PCT
                max_by_margin = int(cash * 0.95 / ES_DAY_MARGIN)
                contracts = min(max(1, int(max_risk / risk_per_contract)), max_by_margin)
                
                if contracts > 0 and contracts * ES_DAY_MARGIN <= cash:
                    margin_locked = contracts * ES_DAY_MARGIN
                    entry_price = round(spy_p, 2)
                    if es_direction == "LONG":
                        sl_price = round(entry_price - sl_points, 2)
                        tp_price = round(entry_price + tp_points, 2)
                    else:
                        sl_price = round(entry_price + sl_points, 2)
                        tp_price = round(entry_price - tp_points, 2)
                    
                    trade_id = f"{today_str}-{now.strftime('%H%M%S')}-MES-{es_direction}"
                    new_pos = _ensure_trade_id({
                        "trade_id": trade_id, "date": today_str, "status": "OPEN",
                        "instrument": "MES", "direction": direction_bias,
                        "es_direction": es_direction,
                        "entry_price": entry_price, "contracts": contracts,
                        "sl_price": sl_price, "tp_price": tp_price,
                        "sl_points": round(sl_points, 2), "tp_points": round(tp_points, 2),
                        "margin_locked": margin_locked,
                        "entry_time": now.strftime("%H:%M"),
                        "entry_ts": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "time": now.strftime("%H:%M"),
                        "score": normalized, "grade": signal["grade"],
                        "unrealized_pnl": 0.0, "unrealized_pnl_pct": 0.0,
                    })
                    portfolio["positions"][today_str] = new_pos
                    portfolio["cash"] -= margin_locked
                    _append_trade_event(portfolio, {
                        "event": "OPEN",
                        "trade_id": trade_id, "date": today_str,
                        "time": now.strftime("%H:%M"),
                        "direction": direction_bias, "es_direction": es_direction,
                        "grade": signal["grade"], "score": normalized,
                        "entry_price": entry_price, "contracts": contracts,
                        "sl_price": sl_price, "tp_price": tp_price,
                        "margin_locked": margin_locked,
                    })

            # 3. Manage open ES futures position
            open_pos = portfolio.get("positions", {}).get(today_str)
            if open_pos:
                entry_price = open_pos.get("entry_price", spy_p)
                contracts = open_pos.get("contracts", 1)
                es_dir = open_pos.get("es_direction", "LONG")
                
                if es_dir == "LONG":
                    point_pnl = spy_p - entry_price
                else:
                    point_pnl = entry_price - spy_p
                
                unrealized_pnl = round(point_pnl * ES_MULTIPLIER * contracts - ES_COMMISSION_RT * contracts, 2)
                margin_locked = open_pos.get("margin_locked", ES_DAY_MARGIN * contracts)
                unrealized_pnl_pct = round((unrealized_pnl / margin_locked) * 100, 1) if margin_locked > 0 else 0.0
                
                open_pos["current_price"] = round(spy_p, 2)
                open_pos["mark_time"] = now.strftime("%H:%M")
                open_pos["unrealized_pnl"] = unrealized_pnl
                open_pos["unrealized_pnl_pct"] = unrealized_pnl_pct
                open_pos["point_pnl"] = round(point_pnl, 2)
                
                exit_type = None
                sl_price = open_pos.get("sl_price")
                tp_price = open_pos.get("tp_price")
                
                # ── Trailing Stop + Breakeven management ──────────────
                entry_p  = open_pos.get("entry_price", spy_p)
                sl_pts   = open_pos.get("sl_points", 4.0)
                tp_pts   = open_pos.get("tp_points", 8.0)

                # Breakeven: once +1×SL in profit, move SL to entry
                if es_dir == "LONG" and point_pnl >= sl_pts and not open_pos.get("be_activated"):
                    open_pos["sl_price"] = entry_p + ES_TICK_SIZE
                    open_pos["be_activated"] = True
                    sl_price = open_pos["sl_price"]
                elif es_dir == "SHORT" and point_pnl >= sl_pts and not open_pos.get("be_activated"):
                    open_pos["sl_price"] = entry_p - ES_TICK_SIZE
                    open_pos["be_activated"] = True
                    sl_price = open_pos["sl_price"]

                # Trailing stop: once +1.5×SL in profit, trail by 1×SL
                if es_dir == "LONG" and point_pnl >= sl_pts * 1.5:
                    trail_sl = spy_p - sl_pts
                    if trail_sl > open_pos.get("sl_price", 0):
                        open_pos["sl_price"] = round(trail_sl, 2)
                        sl_price = open_pos["sl_price"]
                elif es_dir == "SHORT" and point_pnl >= sl_pts * 1.5:
                    trail_sl = spy_p + sl_pts
                    if trail_sl < open_pos.get("sl_price", 9999):
                        open_pos["sl_price"] = round(trail_sl, 2)
                        sl_price = open_pos["sl_price"]

                # ── Exit conditions ───────────────────────────────────
                invalid_reason = _position_invalid_reason(open_pos, grade, direction_bias, score_result)
                if invalid_reason:
                    exit_type = invalid_reason
                elif es_dir == "LONG" and sl_price and spy_p <= sl_price:
                    exit_type = "SL" if not open_pos.get("be_activated") else "BE_SL"
                elif es_dir == "SHORT" and sl_price and spy_p >= sl_price:
                    exit_type = "SL" if not open_pos.get("be_activated") else "BE_SL"
                elif es_dir == "LONG" and tp_price and spy_p >= tp_price:
                    exit_type = "TP"
                elif es_dir == "SHORT" and tp_price and spy_p <= tp_price:
                    exit_type = "TP"
                elif not is_regular or (now.hour >= 15 and now.minute >= 30):
                    exit_type = "EOD"
                
                if exit_type:
                    realized_pnl = unrealized_pnl
                    open_pos.update({
                        "status": "CLOSED", "exit_price": round(spy_p, 2),
                        "exit_time": now.strftime("%H:%M"),
                        "exit_ts": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_type": exit_type,
                        "pnl": realized_pnl, "realized_pnl": realized_pnl,
                        "pnl_pct": unrealized_pnl_pct, "pnl_locked": True,
                        "win": realized_pnl > 0,
                    })
                    for key in ("unrealized_pnl", "unrealized_pnl_pct", "current_price", "mark_time", "point_pnl"):
                        open_pos.pop(key, None)
                    portfolio["cash"] += margin_locked + realized_pnl
                    _append_history(portfolio, open_pos.copy())
                    _append_trade_event(portfolio, {
                        "event": "CLOSE", "trade_id": open_pos.get("trade_id"),
                        "date": open_pos.get("date"),
                        "entry_time": open_pos.get("entry_time"),
                        "exit_time": open_pos.get("exit_time"),
                        "direction": open_pos.get("direction"),
                        "es_direction": es_dir,
                        "entry_price": entry_price, "exit_price": round(spy_p, 2),
                        "contracts": contracts, "exit_type": exit_type,
                        "pnl": realized_pnl, "pnl_pct": open_pos.get("pnl_pct"),
                        "win": open_pos.get("win"),
                    })
                    del portfolio["positions"][today_str]
                else:
                    portfolio["positions"][today_str] = open_pos


            portfolio["recent_trades"] = _build_recent_trades(portfolio)

            save_ok = save_portfolio(portfolio)

            portfolio["storage_type"] = portfolio.get("_storage", "unknown")
            portfolio["trade_count"] = len(portfolio.get("history") or [])

            rules = {
                "vix": {"val": f"{vix_p:.2f}", "ok": vix_p >= 14},
                "range": {"val": f"${d_range:.2f}", "ok": d_range >= 3.0},
                "window": {"val": now.strftime("%H:%M"), "ok": is_regular},
                "vwap": {"val": f"${spy_p - vwap:+.2f}", "ok": spy_p > vwap},
                "vol": {"val": f"{vol_r:.2f}x", "ok": vol_r >= 1.5},
                "sector": {"val": "SYNC" if score_result["layers"].get("correlation", {}).get("sector_sync") else "DIFF", "ok": score_result["layers"].get("correlation", {}).get("sector_sync")}
            }

            latency = round((time.perf_counter() - start_time) * 1000, 1)

            # Build indices/mag7/gme from snapshots
            INDICES_MAP = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Russell 2000", "^VIX": "VIX"}
            indices_out = {}
            for sym in INDICES_MAP:
                if sym == "^VIX":
                    indices_out[sym] = {"price": vix_p, "pct": 0.0}
                else:
                    s = snaps.get(sym, {})
                    indices_out[sym] = {"price": _snap_price(s), "pct": _pct(_snap_price(s), _snap_prev_close(s) or 1)}

            mag7_out = {}
            for sym in MAG7:
                s = snaps.get(sym, {})
                mag7_out[sym] = {"price": _snap_price(s), "pct": _pct(_snap_price(s), _snap_prev_close(s) or 1)}

            gme_data = {}
            for sym in GME_STOCK:
                s = snaps.get(sym, {})
                gme_data[sym] = {"price": _snap_price(s), "pct": _pct(_snap_price(s), _snap_prev_close(s) or 1)}

            order_flow = _calculate_es_order_flow(
                spy_price=spy_p, vix_price=vix_p,
                normalized_score=normalized, direction_bias=direction_bias,
            )

            # Day counters (Day 1 = 2026-05-22)
            start_dt = datetime.strptime(TRADING_START_DATE, "%Y-%m-%d").date()
            today_dt = now.date()
            calendar_day = max(1, (today_dt - start_dt).days + 1)
            # NYSE holidays 2026
            nyse_holidays_2026 = {
                datetime(2026,1,1).date(),    # New Year's Day
                datetime(2026,1,19).date(),   # MLK Day
                datetime(2026,2,16).date(),   # Presidents' Day
                datetime(2026,4,3).date(),    # Good Friday
                datetime(2026,5,25).date(),   # Memorial Day
                datetime(2026,6,19).date(),   # Juneteenth
                datetime(2026,7,3).date(),    # Independence Day (observed)
                datetime(2026,9,7).date(),    # Labor Day
                datetime(2026,11,26).date(),  # Thanksgiving
                datetime(2026,12,25).date(),  # Christmas
            }
            # Trading day = weekdays (Mon-Fri) minus NYSE holidays
            trading_day_count = 0
            d = start_dt
            while d <= today_dt:
                if d.weekday() < 5 and d not in nyse_holidays_2026:
                    trading_day_count += 1
                d += timedelta(days=1)
            trading_day = max(1, trading_day_count)

            # Live Readiness: compute paper trading phase
            pt_history = portfolio.get("history", [])
            pt_wins    = sum(1 for t in pt_history if t.get("win"))
            pt_losses  = sum(1 for t in pt_history if not t.get("win") and t.get("pnl") is not None)
            pt_total   = pt_wins + pt_losses
            pt_wr      = round(pt_wins / pt_total * 100, 1) if pt_total > 0 else None
            pt_pf      = None
            if pt_losses > 0:
                gross_win  = sum(t.get("pnl", 0) for t in pt_history if t.get("win"))
                gross_loss = abs(sum(t.get("pnl", 0) for t in pt_history if not t.get("win") and t.get("pnl") is not None))
                pt_pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else None
            pt_max_dd  = 0.0
            if pt_total > 0:
                peak = STARTING_BALANCE
                running = STARTING_BALANCE
                for t in pt_history:
                    running += float(t.get("pnl", 0) or 0)
                    peak = max(peak, running)
                    dd = (peak - running) / peak * 100 if peak > 0 else 0
                    pt_max_dd = max(pt_max_dd, dd)

            # Phase determination
            if pt_total < 30:
                pt_phase = 1
                pt_phase_label = f"Phase 1: Validation ({pt_total}/30 trades)"
            elif pt_wr and pt_wr >= 55 and pt_pf and pt_pf >= 1.3 and pt_max_dd < 20:
                pt_phase = 3
                pt_phase_label = "Phase 3: LIVE READY ✅"
            else:
                pt_phase = 2
                pt_phase_label = f"Phase 2: Accumulating ({pt_total} trades)"

            paper_trading_stats = {
                "total_trades": pt_total,
                "wins": pt_wins,
                "losses": pt_losses,
                "win_rate": pt_wr,
                "profit_factor": pt_pf,
                "max_drawdown_pct": round(pt_max_dd, 1),
                "phase": pt_phase,
                "phase_label": pt_phase_label,
                "live_ready": pt_phase == 3,
                "criteria": {
                    "trades_needed": max(0, 30 - pt_total),
                    "win_rate_ok": pt_wr >= 55 if pt_wr else False,
                    "profit_factor_ok": pt_pf >= 1.3 if pt_pf else False,
                    "drawdown_ok": pt_max_dd < 20,
                }
            }

            final = {
                "last_updated": ts, "fetch_status": "SUCCESS", "latency_ms": latency,
                "timing_ms": fetch_timing,
                "data_source": "ALPACA + FlashAlpha" if flashalpha_spy else "ALPACA",
                "flashalpha": flashalpha_spy,
                "session": "REGULAR" if is_regular else "CLOSED",
                "market_status": status,                    # NEW: regular / pre_market / after_hours / closed
                "trading_day": trading_day,
                "calendar_day": calendar_day,
                "trading_start_date": TRADING_START_DATE,
                "briefing": f"{score_result['layers']['time_window']['emoji']} [{score_result['layers']['time_window']['window']}] Regime: {score_result['layers']['regime']['regime']} | Bias: {direction_bias} | Score: {normalized}/100",
                "total_score": normalized, "max_score": active_max, "raw_score": raw_total,
                "signal": signal, "direction_bias": direction_bias,
                "layers": score_result["layers"],
                "order_flow": order_flow,
                "verdict": signal["label"], "confidence": normalized, "reason": signal["action"],
                "rules": rules, "alert_mode": "ON SIGNAL CHANGE",
                "indices": indices_out, "mag7": mag7_out,
                "gme_data": gme_data, "special_watch": gme_data,
                "paper_trading": portfolio,
                "backtest_summary": BACKTEST_SUMMARY,
                "paper_trading_stats": paper_trading_stats,
                "mes_specs": {
                    "instrument": "MES",
                    "multiplier": ES_MULTIPLIER,
                    "commission_rt": ES_COMMISSION_RT,
                    "day_margin": ES_DAY_MARGIN,
                    "tick_size": ES_TICK_SIZE,
                    "risk_pct": RISK_PCT,
                    "daily_loss_limit_pct": DAILY_LOSS_LIMIT * 100,
                    "trailing_stop": True,
                    "breakeven_activation": "1×SL in profit",
                },
                "daily_halt": daily_dd_pct >= DAILY_LOSS_LIMIT * 100,
                "daily_drawdown_pct": round(daily_dd_pct, 2),
            }

            origin = self.headers.get("Origin", "")
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            if origin in ALLOWED_ORIGINS:
                self.send_header('Access-Control-Allow-Origin', origin)
            else:
                self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(json.dumps(final, cls=SafeEncoder).encode('utf-8'))

        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"[Error in API] {err_msg}")
            
            origin = self.headers.get("Origin", "")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            if origin in ALLOWED_ORIGINS:
                self.send_header('Access-Control-Allow-Origin', origin)
            else:
                self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Hide detailed traceback in production/Vercel
            is_prod = os.getenv("VERCEL") is not None
            err_resp = {
                "error": "Internal Server Error",
                "fetch_status": "ERROR",
                "message": str(e) if not is_prod else "An error occurred on the server."
            }
            self.wfile.write(json.dumps(err_resp).encode('utf-8'))

