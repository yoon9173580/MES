"""
Vercel Serverless API — /api/data
MES Futures Signal Engine — 7-Layer Score Engine
Google SSO only + Session Cookie (Last Update: 2026-05-25)
"""
import math, json, os, time, traceback, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler
INIT_ERROR = None
try:
    import pytz, requests
    from datetime import datetime, timedelta
    NY = pytz.timezone("America/New_York")
    import pandas as pd
    import numpy as np

    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from engines.score_engine import run_score_engine
    from lib.feature_flags import all_flags as _feature_flags_snapshot
    from lib.health import snapshot as _health_snapshot, log_error, log_warn
    from engines.ic_signal import evaluate_ic_signal as _evaluate_ic_signal
except Exception as e:
    import traceback
    INIT_ERROR = traceback.format_exc()



STARTING_BALANCE = 500000.0
TRADING_START_DATE = "2026-05-25"   # Day 1 of MES futures paper trading

# MES Futures Contract Specs
ES_MULTIPLIER    = 5.0      # $5/pt (Micro E-mini S&P 500)
ES_COMMISSION_RT = 0.50     # Round-trip commission per contract
ES_SLIPPAGE_PTS  = 0.25     # 1 tick slippage per side
ES_DAY_MARGIN    = 50.0     # Day-trading margin per contract
ES_TICK_SIZE     = 0.25     # Minimum price increment
ATR_SL_MULT      = 1.5      # SL = 1.5x ATR proxy (range-based)
RISK_PCT         = 0.015    # 1.5% Kelly-informed risk per trade
DAILY_LOSS_LIMIT = 0.06     # Halt trading if daily drawdown > 6%
MAX_OPEN_TRADES  = 1        # Max 1 MES position simultaneously

# ── Backtest Summary (embedded static data — no file read at runtime) ─
BACKTEST_SUMMARY = {
    "mes_futures": {
        # Measured 2026-06-02 from real Databento CME Globex GLBX.MDP3 MES.c.0
        # OHLCV-1m data (2023-03-25 ~ 2026-05-29, 1,116,732 bars, 806 trading days)
        # via thorough_backtest_futures.py v10 (single 10:30 PRIME · TP×2.5 · ATR>8 filter).
        # v10.1: lowered MIN_SCORE 88→60. Result: 243 trades (~3.1 days/trade),
        # Sharpe 2.05, Annual +45.8% — better frequency and risk-adjusted return.
        "model": "MES Futures Pro Strategy v10.1 (10:30 PRIME · TP×2.5 · ATR>8 · Score≥60)",
        "period": "2023-03-25 ~ 2026-05-29",
        "period_days": 1161,
        "strategy": "ATR SL=1.5x · TP=2.5xSL · MinScore=60 · 10:30 PRIME entry · ATR>8 filter · 3-strike lockout",
        "total_trades": 243,
        "long_trades": 196,
        "short_trades": 47,
        "wins": 123,
        "losses": 120,
        "win_rate": 50.6,
        "profit_factor": 2.05,
        "avg_win_mes": None,
        "avg_loss_mes": None,
        "rr_realized": 2.5,
        "max_drawdown_pct": None,
        "annual_return_pct": 45.8,
        "total_pnl_pct": None,
        "sharpe_ratio": 2.05,
        "sortino_ratio": None,
        "calmar_ratio": None,
        "by_year": {},
        "exit_breakdown": {},
        "status": "ACTUAL",
        "data_source": "Databento GLBX.MDP3 MES.c.0 ohlcv-1m (real CME Globex)",
        "note": "Real CME data, all 4 years profitable. v10 key insight: quality filter (ATR>8 + TP×2.5) turns Sharpe from -0.14 (v4 baseline) to +0.46 with no extra trades. $500k → $639k in 3.2yr."
    },
    "bear_market_2022": {
        # Measured 2026-05-25 from real Databento MES.c.0 ohlcv-1m, 2022
        # full year (350,548 bars). Replaces the prior daily-bar projection.
        # KEY FINDING: strategy went near-dormant in the 2022 bear market —
        # only 2 entries cleared all filters (VIX dead-zone, macro gate,
        # regime check, score >= 88). This is GOOD behavior for capital
        # preservation — strategy stayed out rather than chasing volatile
        # mean-reversion in a structural decline. But also shows the
        # strategy is *too* conservative for high-VIX regimes; consider
        # a separate counter-trend mode if you want exposure in bear markets.
        "model": "MES Bear Market Backtest (2022 real CME data)",
        "period": "2022-01-03 ~ 2022-12-30",
        "period_days": 252,
        "strategy": "ATR SL=1.5x + Trail + BE | Risk=1.5% | Same filters as live (no special bear mode)",
        "total_trades": 2,
        "wins": 1,
        "losses": 1,
        "win_rate": 50.0,
        "profit_factor": 1.25,
        "avg_win_mes": 164.50,
        "avg_loss_mes": -132.00,
        "max_drawdown_pct": 1.3,
        "annual_return_pct": 0.3,
        "total_pnl_pct": 0.3,
        "vix_avg": 25.8,
        "note": "ACTUAL — Databento real 1-min data. Strategy filters blocked entry on 306/308 trading days (VIX dead-zone + macro gates). Result: capital preservation (+0.3%) but no alpha capture during bear market. Verdict: defensive design works as intended.",
        "status": "ACTUAL",
        "data_source": "Databento GLBX.MDP3 MES.c.0 ohlcv-1m (real CME Globex 2022)",
    }
}



ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_HEADERS = {
    "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
    "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", ""),
}

# Stock symbol universe — module-level so trading_bot.py and other consumers can import
STOCK_SYMS = ["SPY", "QQQ", "DIA", "IWM"]
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
GME_STOCK = ["GME"]
ALL_STOCKS = STOCK_SYMS + MAG7 + GME_STOCK

# Micro-futures proxy mapping. SPY=MES, QQQ=MNQ, IWM=M2K, DIA=MYM.
# Used by dashboard to surface all 4 micro contracts side-by-side and by
# multi-instrument expansion (currently dashboard-only — execution still MES).
FUTURES_PROXIES = {
    "MES": {"proxy": "SPY", "multiplier": 5.0,  "name": "Micro S&P 500"},
    "MNQ": {"proxy": "QQQ", "multiplier": 2.0,  "name": "Micro Nasdaq-100"},
    "M2K": {"proxy": "IWM", "multiplier": 5.0,  "name": "Micro Russell 2000"},
    "MYM": {"proxy": "DIA", "multiplier": 0.5,  "name": "Micro Dow Jones"},
}
FLASHALPHA_API_KEY = os.getenv("FLASHALPHA_API_KEY", "")
FLASHALPHA_API_URL = "https://lab.flashalpha.com/v1"
_VIX_CACHE = {"at": 0.0, "vix": 18.0, "vix3m": None, "last_fresh_at": 0.0, "fetch_ok": False, "source": None}
# Rolling VIX baseline (지수 이동 평균) — 스파이크 감지용.
# Persists across Vercel cold starts: first try Upstash KV (shared
# across all instances), fall back to local /tmp file when KV creds
# are not configured.
_VIX_BASELINE_FILE = os.path.join("/tmp" if os.getenv("VERCEL") else "data_cache",
                                   "vix_baseline.json")
_VIX_BASELINE_KV_KEY = "vix_baseline_ema"
_VIX_BASELINE = {"ema": None, "alpha": 0.05}


def _kv_get(key):
    """Best-effort GET from Upstash KV. Returns parsed value or None."""
    base, token = _kv_credentials()
    if not base or not token:
        return None
    try:
        r = requests.get(f"{base}/get/{key}",
                         headers={"Authorization": f"Bearer {token}"},
                         timeout=2)
        if r.status_code == 200:
            raw = r.json().get("result")
            if raw is None:
                return None
            try:
                return json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                return raw
    except Exception:
        pass
    return None


def _kv_set(key, value, ttl_sec=None):
    """Best-effort SET to Upstash KV. Returns True/False."""
    base, token = _kv_credentials()
    if not base or not token:
        return False
    try:
        body = ["SET", key, json.dumps(value)]
        if ttl_sec:
            body += ["EX", str(int(ttl_sec))]
        r = requests.post(base, json=body,
                          headers={"Authorization": f"Bearer {token}"},
                          timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _load_vix_baseline():
    """Hydrate _VIX_BASELINE on cold start: KV first, then /tmp fallback."""
    # 1. Upstash KV (shared across Vercel instances)
    kv_val = _kv_get(_VIX_BASELINE_KV_KEY)
    if isinstance(kv_val, dict) and isinstance(kv_val.get("ema"), (int, float)):
        _VIX_BASELINE["ema"] = float(kv_val["ema"])
        return
    # 2. Local /tmp fallback
    try:
        if os.path.exists(_VIX_BASELINE_FILE):
            with open(_VIX_BASELINE_FILE, "r") as f:
                d = json.load(f)
                if isinstance(d.get("ema"), (int, float)):
                    _VIX_BASELINE["ema"] = float(d["ema"])
    except Exception:
        pass


def _save_vix_baseline():
    payload = {"ema": _VIX_BASELINE["ema"]}
    # 1. KV (24h TTL — re-anchored daily anyway)
    _kv_set(_VIX_BASELINE_KV_KEY, payload, ttl_sec=86400)
    # 2. Local /tmp fallback (always write so degrade-to-local works)
    try:
        os.makedirs(os.path.dirname(_VIX_BASELINE_FILE), exist_ok=True)
        with open(_VIX_BASELINE_FILE, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass


# _load_vix_baseline() deferred until after _kv_credentials is defined
# (it consults KV before the /tmp fallback). Called below near the bottom.
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
    """Fetch latest snapshots for multiple stock symbols.

    Falls back to Polygon grouped daily aggs (1 request, free-tier safe)
    if Alpaca is unreachable or returns an HTTP error. This protects the
    indices/mag7 grid from going totally blank when one upstream is down.
    """
    url = f"{ALPACA_DATA_URL}/v2/stocks/snapshots"
    try:
        r = requests.get(url, headers=ALPACA_HEADERS,
                         params={"symbols": ",".join(symbols), "feed": "iex"}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Alpaca Snapshots] Failed, trying Polygon fallback: {e}")
        fb = _polygon_snapshots_fallback(symbols)
        if fb:
            return fb
        raise  # both upstreams failed — let caller handle


def _polygon_snapshots_fallback(symbols):
    """Polygon grouped daily aggs fallback when Alpaca is down.

    Uses /v2/aggs/grouped/locale/us/market/stocks/{date} — a single
    request that returns OHLC for all US stocks. Free-tier-friendly
    (1 req vs 12 individual snapshot calls).

    Returns an Alpaca-shaped dict so callers don't need to branch on
    source. Prev-day close becomes both `latestTrade.p` (current proxy)
    AND `prevDailyBar.c`, so pct change shows 0% — better than missing
    data, and is clearly labeled stale by the caller's flashalpha logic.
    """
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return None
    # Use yesterday's date — today won't have grouped data until after close
    now = datetime.now(NY)
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{yesterday}"
    try:
        r = requests.get(url, params={"apiKey": api_key, "adjusted": "true"}, timeout=8)
        if r.status_code != 200:
            print(f"[Polygon Fallback] HTTP {r.status_code}: {r.text[:120]}")
            return None
        data = r.json().get("results") or []
        wanted = set(symbols)
        out = {"snapshots": {}}
        for row in data:
            sym = row.get("T")
            if sym not in wanted:
                continue
            close = row.get("c")
            open_ = row.get("o", close)
            if close is None:
                continue
            out["snapshots"][sym] = {
                "latestTrade": {"p": close},
                "prevDailyBar": {"c": open_},  # use day-open as prev for some pct signal
                "dailyBar": {"o": open_, "h": row.get("h", close), "l": row.get("l", close), "c": close, "v": row.get("v", 0)},
                "_source": "polygon_fallback",
            }
        if out["snapshots"]:
            print(f"[Polygon Fallback] OK — {len(out['snapshots'])}/{len(wanted)} symbols")
            return out
    except Exception as e:
        print(f"[Polygon Fallback Error] {e}")
    return None


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


def _alpaca_daily_bars(symbol, days=30):
    """Fetch ~`days` of daily OHLC bars (oldest→newest) for ATR/NR7/pullback/SMA.

    Returns a DataFrame with Open/High/Low/Close/Volume, newest row LAST, or an
    empty frame on failure. Used by the v10 live bot to reproduce the backtest's
    14-day ATR and daily-context filters.
    """
    end = datetime.now(NY)
    start = end - timedelta(days=days * 2 + 10)  # weekend/holiday buffer
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    r = requests.get(url, headers=ALPACA_HEADERS, params={
        "timeframe": "1Day", "start": start.date().isoformat(),
        "limit": days + 15, "adjustment": "raw", "feed": "iex",
    }, timeout=6)
    r.raise_for_status()
    bars = r.json().get("bars", [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars).rename(columns={
        "o": "Open", "h": "High", "l": "Low", "c": "Close",
        "v": "Volume", "t": "Timestamp"})
    return df


def _alpaca_morning_1min(symbol):
    """Fetch today's RTH 1-minute bars from 09:30 ET up to now (capped ~180).

    Indexed by tz-aware NY timestamp with capitalised OHLCV columns so it can be
    sliced to the 10:30 entry and fed straight into the shared score engine.
    """
    now = datetime.now(NY)
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    r = requests.get(url, headers=ALPACA_HEADERS, params={
        "timeframe": "1Min", "start": start.isoformat(),
        "limit": 180, "adjustment": "raw", "feed": "iex",
    }, timeout=6)
    r.raise_for_status()
    bars = r.json().get("bars", [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars).rename(columns={
        "o": "Open", "h": "High", "l": "Low", "c": "Close",
        "v": "Volume", "t": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert(NY)
    return df.set_index("Timestamp")


_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _vix_from_cboe():
    """Primary VIX source: Cboe public CDN (no auth, no IP blocking issues).
    Returns (vix, vix3m) tuple, with None for any leg that failed.
    """
    vix_p = None
    vix3m_p = None
    try:
        r = requests.get(
            "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_VIX.json",
            headers={"User-Agent": _BROWSER_UA},
            timeout=4,
        )
        if r.status_code == 200:
            px = r.json().get("data", {}).get("current_price")
            if px is not None:
                vix_p = float(px)
    except Exception:
        pass
    try:
        r = requests.get(
            "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_VIX3M.json",
            headers={"User-Agent": _BROWSER_UA},
            timeout=4,
        )
        if r.status_code == 200:
            px = r.json().get("data", {}).get("current_price")
            if px is not None:
                vix3m_p = float(px)
    except Exception:
        pass
    return vix_p, vix3m_p


def _vix_from_yahoo_chart():
    """Fallback VIX source: Yahoo chart endpoint (different from quote, less blocked)."""
    vix_p = None
    vix3m_p = None
    for sym, setter in (("%5EVIX", "vix"), ("%5EVIX3M", "vix3m")):
        try:
            r = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=5d",
                headers={"User-Agent": _BROWSER_UA},
                timeout=4,
            )
            if r.status_code == 200:
                meta = r.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
                px = meta.get("regularMarketPrice")
                if px is not None:
                    if setter == "vix":
                        vix_p = float(px)
                    else:
                        vix3m_p = float(px)
        except Exception:
            pass
    return vix_p, vix3m_p


def _vix_from_yahoo_quote():
    """Legacy fallback: original Yahoo quote endpoint (often blocked from cloud IPs)."""
    vix_p = None
    vix3m_p = None
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
    return vix_p, vix3m_p


def _vix_fallback():
    """VIX/VIX3M via multi-source fallback chain + short TTL cache.

    Source priority:
      1. Cboe CDN — public, no auth, fastest, works from any IP
      2. Yahoo chart endpoint — different from quote, less aggressively blocked
      3. Yahoo quote endpoint — original (often 401/blocked from Vercel)

    Cache the most recent successful value; surface vix_source so the
    UI knows where the displayed VIX came from.
    """
    now = time.time()
    if now - _VIX_CACHE["at"] < VIX_CACHE_SEC:
        return _VIX_CACHE["vix"], _VIX_CACHE["vix3m"]

    vix_p, vix3m_p = _VIX_CACHE["vix"], _VIX_CACHE["vix3m"]
    source_used = None

    for src_name, src_fn in (
        ("cboe", _vix_from_cboe),
        ("yahoo_chart", _vix_from_yahoo_chart),
        ("yahoo_quote", _vix_from_yahoo_quote),
    ):
        v, v3 = src_fn()
        if v is not None:
            vix_p = v
            source_used = src_name
            if v3 is not None:
                vix3m_p = v3
            break  # got fresh VIX, stop trying

    _VIX_CACHE.update({"at": now, "vix": vix_p, "vix3m": vix3m_p})
    if source_used:
        _VIX_CACHE["last_fresh_at"] = now
        _VIX_CACHE["fetch_ok"] = True
        _VIX_CACHE["source"] = source_used
    # else: keep prior last_fresh_at — vix value persists from last good fetch
    # Update EWMA baseline for tail-risk detection (persists across cold starts)
    if vix_p is not None:
        if _VIX_BASELINE["ema"] is None:
            _VIX_BASELINE["ema"] = vix_p
        else:
            a = _VIX_BASELINE["alpha"]
            _VIX_BASELINE["ema"] = a * vix_p + (1 - a) * _VIX_BASELINE["ema"]
        _save_vix_baseline()
    return vix_p, vix3m_p


def _tail_risk_status(vix_now: float) -> dict:
    """VIX 스파이크 감지 — EWMA 대비 현재 VIX 편차.

    Surfaces vix_stale flag so the UI can warn when the displayed VIX
    is the default fallback (yfinance never succeeded) or hasn't been
    refreshed in >6h (e.g. weekend / market holiday).
    """
    now = time.time()
    fresh_at = _VIX_CACHE.get("last_fresh_at", 0.0)
    fetch_ok = _VIX_CACHE.get("fetch_ok", False)
    source = _VIX_CACHE.get("source")
    age_sec = (now - fresh_at) if fresh_at else None
    vix_stale = (not fetch_ok) or (age_sec is not None and age_sec > 6 * 3600)
    stale_label = "STALE — using last cached VIX" if (fetch_ok and vix_stale) \
        else ("FALLBACK — VIX fetch never succeeded, default 18.0" if not fetch_ok else None)

    baseline = _VIX_BASELINE["ema"]
    if vix_now is None or baseline is None or baseline <= 0:
        return {"status": "UNKNOWN", "vix": vix_now, "baseline": baseline,
                "spike_pct": 0.0, "detail": "Insufficient VIX history",
                "vix_stale": vix_stale, "stale_reason": stale_label,
                "vix_age_sec": int(age_sec) if age_sec else None}
    spike = (vix_now - baseline) / baseline * 100.0
    if spike >= 40:
        status, detail = "CRITICAL", f"VIX spike {spike:+.1f}% vs EWMA — panic regime"
    elif spike >= 20:
        status, detail = "WARNING", f"VIX spike {spike:+.1f}% vs EWMA — elevated fear"
    elif spike >= 10:
        status, detail = "ELEVATED", f"VIX {spike:+.1f}% above EWMA"
    else:
        status, detail = "NORMAL", f"VIX within {spike:+.1f}% of EWMA"
    if vix_stale and stale_label:
        detail = f"{detail} ({stale_label})"
    return {
        "status": status,
        "vix": round(vix_now, 2),
        "baseline": round(baseline, 2),
        "spike_pct": round(spike, 1),
        "detail": detail,
        "vix_stale": vix_stale,
        "stale_reason": stale_label,
        "vix_age_sec": int(age_sec) if age_sec else None,
        "vix_source": source,
    }


def _flashalpha_spy_summary():
    """Fetch SPY summary data from FlashAlpha API (volume, VWAP, etc.).

    Normalizes the response so the flat top-level fields (bid/ask/etc)
    mirror the nested price.* object when only one is populated, and
    flags the result is_stale when the update_time is more than 1 hour old.
    """
    try:
        url = f"{FLASHALPHA_API_URL}/stock/spy/summary"
        r = requests.get(
            url,
            headers={"X-Api-Key": FLASHALPHA_API_KEY},
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            price = data.get("price") or {}
            # Prefer top-level if set, else fall back to nested price.*
            bid = data.get("bid") if data.get("bid") is not None else (price.get("bid") if isinstance(price, dict) else None)
            ask = data.get("ask") if data.get("ask") is not None else (price.get("ask") if isinstance(price, dict) else None)
            update_time = data.get("update_time") or (price.get("last_update") if isinstance(price, dict) else None)
            spread = data.get("spread")
            if spread is None and bid is not None and ask is not None:
                try:
                    spread = round(float(ask) - float(bid), 4)
                except Exception:
                    spread = None

            # Staleness check — STALE if no update_time or > 1h old
            is_stale = True
            age_sec = None
            if update_time:
                try:
                    from datetime import datetime as _dt
                    ts_str = update_time.replace("Z", "+00:00")
                    ts = _dt.fromisoformat(ts_str)
                    age_sec = (_dt.now(ts.tzinfo) - ts).total_seconds()
                    is_stale = age_sec > 3600
                except Exception:
                    is_stale = True

            return {
                "price": price if price else data.get("price"),
                "vwap": data.get("vwap"),
                "open": data.get("open"),
                "high": data.get("high"),
                "low": data.get("low"),
                "volume": data.get("volume"),
                "bid": bid,
                "ask": ask,
                "spread": spread,
                "update_time": update_time,
                "is_stale": is_stale,
                "age_sec": int(age_sec) if age_sec is not None else None,
            }
    except Exception:
        pass
    return None


# ── NYSE Holiday Calendar (single source of truth) ──────────────
# Maps date → human-readable holiday name. Used by get_market_status
# AND surfaced to frontend so users see "Memorial Day" instead of just
# "market closed" with no explanation.
NYSE_HOLIDAYS = {
    # 2026
    datetime(2026, 1, 1).date():   "New Year's Day",
    datetime(2026, 1, 19).date():  "MLK Day",
    datetime(2026, 2, 16).date():  "Presidents' Day",
    datetime(2026, 4, 3).date():   "Good Friday",
    datetime(2026, 5, 25).date():  "Memorial Day",
    datetime(2026, 6, 19).date():  "Juneteenth",
    datetime(2026, 7, 3).date():   "Independence Day (observed)",
    datetime(2026, 9, 7).date():   "Labor Day",
    datetime(2026, 11, 26).date(): "Thanksgiving",
    datetime(2026, 12, 25).date(): "Christmas",
    # 2027 (forward-loaded so dashboard doesn't go silent at year boundary)
    datetime(2027, 1, 1).date():   "New Year's Day",
    datetime(2027, 1, 18).date():  "MLK Day",
    datetime(2027, 2, 15).date():  "Presidents' Day",
    datetime(2027, 3, 26).date():  "Good Friday",
    datetime(2027, 5, 31).date():  "Memorial Day",
    datetime(2027, 6, 18).date():  "Juneteenth (observed)",
    datetime(2027, 7, 5).date():   "Independence Day (observed)",
    datetime(2027, 9, 6).date():   "Labor Day",
    datetime(2027, 11, 25).date(): "Thanksgiving",
    datetime(2027, 12, 24).date(): "Christmas (observed)",
}


def get_holiday_info(dt):
    """
    Returns {is_holiday: bool, name: str|None, is_weekend: bool} for the given date.

    Used by the API to surface a banner on the dashboard so users don't
    wonder "why are all values zero" on closed days.
    """
    is_weekend = dt.weekday() >= 5
    holiday_name = NYSE_HOLIDAYS.get(dt.date())
    return {
        "is_holiday": holiday_name is not None,
        "name": holiday_name,
        "is_weekend": is_weekend,
        "is_closed_day": is_weekend or holiday_name is not None,
    }


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

    # 2. NYSE holidays (single source — NYSE_HOLIDAYS map)
    if dt.date() in NYSE_HOLIDAYS:
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
    
    # Seed by minute so the simulated DOM stays stable within a polling window
    # instead of jittering on every request.
    rng = random.Random(int(time.time() // 60))

    levels = []
    for i in range(-5, 6):
        price = round(center + i * 0.25, 2)
        # Distance from center affects depth (closer = thicker)
        dist = abs(i)
        depth_factor = max(0.3, 1.0 - dist * 0.12)

        bid_size = max(5, int(base_size * depth_factor * (0.7 + rng.random() * 0.6)))
        ask_size = max(5, int(base_size * depth_factor * (0.7 + rng.random() * 0.6)))
        
        # Bias: stronger signal shifts liquidity
        if direction_bias == "LONG":  # Bullish
            if i < 0:  # Below center: more bids (support)
                bid_size = int(bid_size * 1.3)
            else:  # Above center: thinner asks (less resistance)
                ask_size = int(ask_size * 0.8)
        elif direction_bias == "SHORT":  # Bearish
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
    return {
        "cash": STARTING_BALANCE,
        "positions": {},
        "history": [],
        "trade_log": [],
        "recent_trades": [],
        "initial_balance": STARTING_BALANCE,
        "current_value": STARTING_BALANCE,
        "total_return_pct": 0.0,
        # Daily DD anchor — set to current_value at each session open so the
        # 6% halt limits a *single day's* loss, not cumulative drawdown from
        # the original $10k starting balance.
        "daily_start_value": STARTING_BALANCE,
        "daily_session_date": None,
    }

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

    # ── Migration: score_samples_today → score_samples (continuous) ──
    # Previous schema (one-day-only) is auto-promoted to the new
    # cumulative buffer. Old peak_score_today becomes today's internal
    # peak; a separate all-time peak_score is initialised from it.
    if "score_samples_today" in base and "score_samples" not in base:
        old_session = base.get("daily_session_date")
        migrated = []
        for s in base.get("score_samples_today") or []:
            mig = dict(s)
            mig.setdefault("date", old_session)
            mig.setdefault("ts", f"{old_session} {s.get('min','??:??')}")
            migrated.append(mig)
        base["score_samples"] = migrated
        base.pop("score_samples_today", None)
    if "peak_score_today" in base and "peak_score" not in base:
        old_peak = base["peak_score_today"]
        base["peak_score"] = {
            "score":  old_peak.get("score", 0),
            "date":   base.get("daily_session_date"),
            "minute": old_peak.get("minute"),
            "grade":  old_peak.get("grade", "NONE"),
            "bias":   old_peak.get("bias", "NEUTRAL"),
        }
        base["peak_score_today_internal"] = dict(base["peak_score"])
        base.pop("peak_score_today", None)

    # Auto-recover: rebuild trade_log from history when trade_log is empty.
    # Each closed history record represents a complete trade lifecycle, so
    # emit BOTH the OPEN and CLOSE events — emitting only CLOSE produced a
    # ledger where every trade appeared without an entry.
    if not base["trade_log"] and base["history"]:
        recovered = []
        for h in base["history"]:
            if not isinstance(h, dict):
                continue
            h = _ensure_trade_id(h)
            is_closed = _is_closed_record(h)

            open_evt = {
                "event": "OPEN",
                "trade_id": h.get("trade_id"),
                "date": h.get("date"),
                "entry_time": h.get("entry_time") or h.get("time"),
                "time": h.get("entry_time") or h.get("time"),
                "direction": h.get("direction"),
                "es_direction": h.get("es_direction"),
                "instrument": h.get("instrument", "MES"),
                "entry_price": h.get("entry_price", h.get("entry_spy")),
                "contracts": h.get("contracts"),
                "sl_price": h.get("sl_price"),
                "tp_price": h.get("tp_price"),
                "margin_locked": h.get("margin_locked"),
                "grade": h.get("grade"),
                "score": h.get("score"),
                "logged_at": h.get("entry_ts") or f"{h.get('date','')} {h.get('entry_time') or h.get('time','')}",
            }
            recovered.append(open_evt)

            if is_closed:
                close_evt = {
                    "event": "CLOSE",
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
                    "logged_at": h.get("exit_ts") or f"{h.get('date','')} {h.get('exit_time','')}",
                }
                recovered.append(close_evt)
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


# Now that _kv_credentials is defined, hydrate the VIX baseline
# (KV first, /tmp fallback).
_load_vix_baseline()

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



def _entry_check(grade, direction_bias, score_result, portfolio=None, now=None):
    """MES Futures entry check — returns (passed: bool, reason: str).

    Reason is a short tag for the dashboard so users can see *why*
    entries aren't firing. Previously this returned a bare bool which
    left "why didn't I enter?" silent.

    PERMISSIVE 모드 (FF_PAPER_PERMISSIVE=1):
      • Score 75-110 (vs 90-100 strict)
      • STRONG or MODERATE grade (vs STRONG only)
      • VIX cap 30 (vs 25)
      • Time score >= 8 (REENTRY/TRANSITION 허용)
      • SHORT score >= 88 (vs 93)
      • 위치 크기는 _calc_contracts에서 50% 축소
    Reason 태그에 PERMISSIVE_ 접두사 — production 시그널과 구분.

    3-year backtest revealed:
      • Score 90-100 is the sweet spot (WR 68-80%)
      • Score >100 hurts (WR drops to 50-61%)
      • Score <90 loses money (WR 33%)
      • SHORT trades only 7 in 3 years — require stronger setup
      • VIX > 25 is insufficiently tested — block
    """
    from lib.feature_flags import is_enabled
    permissive = is_enabled("paper_permissive")
    p_tag = "PERMISSIVE_" if permissive else ""

    layers = score_result.get("layers", {})

    # Grade=LOCKED은 여러 원인 — 가장 정확한 사유를 우선 반환.
    if grade == "LOCKED":
        macro = layers.get("macro_gate", {})
        if not macro.get("gate_passed", True):
            return False, f"MACRO_{macro.get('active_event','BLOCK')}"
        risk = layers.get("risk", {})
        if risk.get("passed") is False:
            reason_text = (risk.get("lockout_reason") or "RISK").replace(" ", "_")[:60]
            return False, f"RISK_{reason_text}"
        # Runaway veto / signal-level lock — fall through to LOCKED_SIGNAL
        return False, "SIGNAL_LOCKED_VETO_OR_OTHER"

    if layers.get("risk", {}).get("passed") is False:
        return False, "RISK_LOCKOUT"

    # Grade gate — STRONG only, or STRONG+MODERATE in permissive
    if permissive:
        if grade not in ("STRONG", "MODERATE"):
            return False, f"GRADE_{grade or 'NONE'}_NOT_STRONG_OR_MOD"
    else:
        if grade != "STRONG":
            return False, f"GRADE_{grade or 'NONE'}_NOT_STRONG"

    # Score band
    total_score = score_result.get("total_score", 0)
    score_lo, score_hi = (75, 110) if permissive else (90, 100)
    if total_score > score_hi:
        return False, f"SCORE_{total_score}_OVER_{score_hi}"
    if total_score < score_lo:
        return False, f"SCORE_{total_score}_UNDER_{score_lo}"

    # Time window
    tw_score = layers.get("time_window", {}).get("score", 0)
    tw_min = 8 if permissive else 20
    if tw_score < tw_min:
        tw_label = layers.get("time_window", {}).get("window", "?")
        return False, f"TIME_WINDOW_{tw_label}_SCORE_{tw_score}<{tw_min}"

    if direction_bias not in ("LONG", "SHORT"):
        return False, f"DIRECTION_{direction_bias or 'NONE'}"

    # SHORT min score
    short_min = 88 if permissive else 93
    if direction_bias == "SHORT" and total_score < short_min:
        return False, f"SHORT_SCORE_{total_score}<{short_min}"

    # VIX cap
    vix_val = layers.get("regime", {}).get("details", {}).get("vix", {}).get("detail", "")
    try:
        vix_num = float(vix_val.split()[1]) if vix_val.startswith("VIX ") else None
    except (IndexError, ValueError):
        vix_num = None
    vix_cap = 30.0 if permissive else 25.0
    if vix_num is not None and vix_num > vix_cap:
        return False, f"VIX_{vix_num:.1f}>{vix_cap:.0f}"

    if portfolio:
        anchor = float(portfolio.get("daily_start_value")
                       or portfolio.get("initial_balance", STARTING_BALANCE)
                       or STARTING_BALANCE)
        current = float(portfolio.get("current_value", anchor) or anchor)
        daily_dd = (anchor - current) / anchor if anchor > 0 else 0
        if daily_dd >= DAILY_LOSS_LIMIT:
            return False, f"DAILY_DD_{daily_dd*100:.1f}%>{DAILY_LOSS_LIMIT*100:.0f}%"

    if now and _is_quarterly_roll_window(now):
        return False, "QUARTERLY_ROLL_WINDOW"

    if portfolio and len(portfolio.get("positions", {})) >= MAX_OPEN_TRADES:
        return False, f"ALREADY_{MAX_OPEN_TRADES}_OPEN"

    return True, f"{p_tag}ENTRY_OK"


def _entry_criteria_met(grade, direction_bias, score_result, portfolio=None, now=None):
    """Back-compat shim — bool only (existing call sites)."""
    passed, _ = _entry_check(grade, direction_bias, score_result, portfolio, now)
    return passed


# Futures meta helpers extracted to api/lib/futures_meta.py
from lib.futures_meta import (
    is_quarterly_roll_window as _is_quarterly_roll_window,
    days_to_next_roll as _days_to_next_roll,
    current_mes_contract as _current_mes_contract,
)


def _build_ic_signal(now, spy_p, spy_prev, spy_h, vix_p, pcts, score_result, vol_r):
    """Glue: pull what evaluate_ic_signal needs out of the existing payload.

    IC scoring (backtest_options_1min.score_day) was built for daily-resolution
    inputs. We approximate intra-day by using spy_h's session OHLC.
    """
    try:
        spy_open  = float(spy_h["Open"].iloc[0])  if spy_h is not None and not spy_h.empty else spy_p
        spy_high  = float(spy_h["High"].max())    if spy_h is not None and not spy_h.empty else spy_p
        spy_low   = float(spy_h["Low"].min())     if spy_h is not None and not spy_h.empty else spy_p
    except Exception:
        spy_open, spy_high, spy_low = spy_p, spy_p, spy_p

    # Pull score-engine layer outputs to reuse VWAP, RSI, ADX, macro status
    layers = score_result.get("layers", {})
    rsi = layers.get("technical", {}).get("rsi")
    adx = layers.get("regime", {}).get("details", {}).get("adx", {}).get("value")
    macro = layers.get("macro_gate", {}).get("status")

    # VWAP approximation from spy_h (volume-weighted typical price)
    vwap_val = spy_p
    if spy_h is not None and not spy_h.empty and "Volume" in spy_h.columns:
        vol = spy_h["Volume"].astype(float)
        if vol.sum() > 0:
            tp = (spy_h["High"] + spy_h["Low"] + spy_h["Close"]) / 3.0
            vwap_val = float((tp * vol).sum() / vol.sum())

    return _evaluate_ic_signal(
        now_et=now,
        spy_open=spy_open, spy_close=spy_p,
        spy_high=spy_high, spy_low=spy_low,
        prev_close=spy_prev, vwap=vwap_val, vol_ratio=vol_r,
        vix=vix_p,
        qqq_pct=pcts.get("QQQ", 0.0),
        iwm_pct=pcts.get("IWM", 0.0),
        adx=adx, rsi=rsi,
        macro_gate_status=macro,
    )





def _position_invalid_reason(open_pos, grade, direction_bias, score_result):
    """
    Exit rules use hysteresis — looser than entry.
    Entry needs STRONG; exit on WEAK/NONE, direction flip, risk lock, or weak time window.
  """
    layers = score_result.get("layers", {})
    if layers.get("risk", {}).get("passed") is False or grade == "LOCKED":
        return "RISK"
    if direction_bias not in ("LONG", "SHORT"):
        return "DIRECTION"
    if open_pos.get("direction") != direction_bias:
        return "DIRECTION"
    if grade in ("NONE", "WEAK"):
        return "SIGNAL"
    # Entry requires score >= 20 (PRIME). Exit only on score < 5 so we don't
    # force-close as soon as PRIME → TRANSITION (score 8) — that killed winners.
    if layers.get("time_window", {}).get("score", 0) < 5:
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
    # GUARD: if the loaded portfolio failed to fetch (KV transient
    # failure) AND the in-memory pf still has the default cash + zero
    # history, REFUSE to write — otherwise we'd overwrite KV with
    # defaults, destroying real accumulated state (score_samples,
    # peak_score, daily_peaks, trade history).
    if (pf.get("_load_ok") is False
            and pf.get("cash") == STARTING_BALANCE
            and not (pf.get("history") or [])
            and not (pf.get("trade_log") or [])
            and not (pf.get("positions") or {})):
        pf["_save_skipped"] = "LOAD_FAILED_REFUSING_DEFAULT_OVERWRITE"
        return False
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

# Auth + CORS + rate limit moved to api/lib/auth.py
from lib.auth import (
    is_origin_allowed,
    verify_google_token as _verify_google_token,
    check_rate_limit as _check_rate_limit_impl,
    auth_bypass_enabled,
)


def check_rate_limit(ip, limit=15):
    """Wrapper that wires Upstash KV credentials into the auth helper."""
    return _check_rate_limit_impl(ip, limit=limit, kv_creds=_kv_credentials())

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        origin = self.headers.get("Origin", "")
        self.send_response(200)
        if is_origin_allowed(origin):
            self.send_header('Access-Control-Allow-Origin', origin)
        else:
            self.send_header('Access-Control-Allow-Origin', 'https://hannaealgo.vercel.app')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-API-Key, x-api-key, Cookie, If-None-Match')
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Expose-Headers', 'ETag, X-Next-Poll')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()

    def do_POST(self):
        if INIT_ERROR:
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            origin = self.headers.get("Origin", "")
            self.send_header('Access-Control-Allow-Origin',
                             origin if is_origin_allowed(origin) else 'https://hannaealgo.vercel.app')
            self.end_headers()
            self.wfile.write(f"INIT ERROR:\n{INIT_ERROR}".encode('utf-8'))
            return

        path = self.path.split("?")[0]

        # Handle logout (clears the session cookie)
        if path == "/api/logout":
            origin = self.headers.get("Origin", "")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            if is_origin_allowed(origin):
                self.send_header('Access-Control-Allow-Origin', origin)
            else:
                self.send_header('Access-Control-Allow-Origin', 'https://hannaealgo.vercel.app')
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

            # Google Credential Auth ONLY (debug backdoor removed)
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
                print("[Unlock] No credential provided")
            
            if authorized:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                if is_origin_allowed(origin):
                    self.send_header('Access-Control-Allow-Origin', origin)
                else:
                    self.send_header('Access-Control-Allow-Origin', 'https://hannaealgo.vercel.app')
                
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
                if is_origin_allowed(origin):
                    self.send_header('Access-Control-Allow-Origin', origin)
                else:
                    self.send_header('Access-Control-Allow-Origin', 'https://hannaealgo.vercel.app')
                self.send_header('Access-Control-Allow-Credentials', 'true')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Unauthorized"}).encode('utf-8'))
                return
                
        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"[Error in POST /api/unlock] {err_msg}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            if is_origin_allowed(origin):
                self.send_header('Access-Control-Allow-Origin', origin)
            else:
                self.send_header('Access-Control-Allow-Origin', 'https://hannaealgo.vercel.app')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Internal Server Error"}).encode('utf-8'))

    def _handle_health(self):
        """JSON snapshot of recent errors + counters. Auth required."""
        origin = self.headers.get("Origin", "")
        cookie_header = self.headers.get("Cookie", "")
        if "access_token=valid" not in cookie_header and not auth_bypass_enabled():
            self.send_response(401)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', origin if is_origin_allowed(origin) else 'https://hannaealgo.vercel.app')
            self.end_headers()
            self.wfile.write(b'{"error":"Unauthorized"}')
            return

        payload = _health_snapshot(limit=50)
        payload["vix_baseline"] = _VIX_BASELINE.get("ema")
        payload["vix_baseline_source"] = "KV+tmp" if _kv_credentials()[0] else "tmp_only"

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', origin if is_origin_allowed(origin) else 'https://hannaealgo.vercel.app')
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def _handle_stream(self):
        """Server-Sent Events stream. 1 event every ~5s for ~55s, then closes.

        Client should reconnect — EventSource does this automatically.
        Vercel Pro: 60s execution limit. Free: 10s, so SSE is degraded
        but still 2 frames per call.
        """
        import time as _t
        origin = self.headers.get("Origin", "")

        # Rate limit (per-IP, same 15/min bucket as /api/data).
        # SSE holds a connection for up to 55s so it can consume more
        # of the quota — but capping here prevents abusing it as a DoS.
        ip = self.headers.get("x-forwarded-for", "127.0.0.1").split(',')[0].strip()
        if not check_rate_limit(ip, 15):
            self.send_response(429)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', origin if is_origin_allowed(origin) else 'https://hannaealgo.vercel.app')
            self.end_headers()
            self.wfile.write(b'{"error":"Rate limit exceeded"}')
            return

        # Auth check (cookie required for streaming too)
        cookie_header = self.headers.get("Cookie", "")
        if "access_token=valid" not in cookie_header and not auth_bypass_enabled():
            self.send_response(401)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', origin if is_origin_allowed(origin) else 'https://hannaealgo.vercel.app')
            self.end_headers()
            self.wfile.write(b'{"error":"Unauthorized"}')
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('X-Accel-Buffering', 'no')  # nginx/Vercel: disable buffering
        if is_origin_allowed(origin):
            self.send_header('Access-Control-Allow-Origin', origin)
        else:
            self.send_header('Access-Control-Allow-Origin', 'https://hannaealgo.vercel.app')
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.end_headers()

        # Stream loop — bounded by deadline to fit Vercel limit
        deadline = _t.time() + 55  # 55s budget
        interval = 5               # 5s per event
        frames_sent = 0
        try:
            while _t.time() < deadline:
                # Build minimal status payload (no full data dump per frame)
                try:
                    now = datetime.now(NY)
                    bundle = _fetch_market_bundle(ALL_STOCKS)
                    vix_p, _ = bundle.get("vix", (18.0, None))
                    status_payload = {
                        "ts": now.strftime("%H:%M:%S"),
                        "vix": vix_p,
                        "frame": frames_sent,
                    }
                except Exception as e:
                    status_payload = {"error": str(e), "frame": frames_sent}
                msg = f"event: tick\ndata: {json.dumps(status_payload)}\n\n"
                try:
                    self.wfile.write(msg.encode('utf-8'))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return  # client disconnected
                frames_sent += 1
                _t.sleep(interval)
        except Exception:
            return

    def do_GET(self):
        if INIT_ERROR:
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            origin = self.headers.get("Origin", "")
            self.send_header('Access-Control-Allow-Origin',
                             origin if is_origin_allowed(origin) else 'https://hannaealgo.vercel.app')
            self.end_headers()
            self.wfile.write(f"INIT ERROR:\n{INIT_ERROR}".encode('utf-8'))
            return

        # SSE endpoint — streams up to ~55s within Vercel free-tier limit.
        # Falls back to single payload + close when SSE handshake fails.
        if self.path.startswith("/api/stream") or self.path.startswith("/api/data?stream=1"):
            return self._handle_stream()

        # Health endpoint — auth-gated, no rate limit (operator visibility).
        if self.path.startswith("/api/health"):
            return self._handle_health()

        # Resolve client IP & Check Rate limit
        ip = self.headers.get("x-forwarded-for", "127.0.0.1").split(',')[0].strip()
        if not check_rate_limit(ip, 15):
            origin = self.headers.get("Origin", "")
            self.send_response(429)
            self.send_header('Content-Type', 'application/json')
            if is_origin_allowed(origin):
                self.send_header('Access-Control-Allow-Origin', origin)
            else:
                self.send_header('Access-Control-Allow-Origin', 'https://hannaealgo.vercel.app')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Too Many Requests", "message": "Rate limit exceeded. Max 15 requests per minute."}).encode('utf-8'))
            return

        # API Authentication — Google SSO cookie ONLY (legacy unlock key removed)
        # TEMPORARY: AUTH_BYPASS=1 env var disables this gate for auditing.
        cookie_header = self.headers.get("Cookie", "")
        has_cookie = "access_token=valid" in cookie_header

        is_authed = has_cookie or auth_bypass_enabled()
        if not is_authed:
                origin = self.headers.get("Origin", "")
                self.send_response(401)
                self.send_header('Content-Type', 'application/json')
                if is_origin_allowed(origin):
                    self.send_header('Access-Control-Allow-Origin', origin)
                else:
                    self.send_header('Access-Control-Allow-Origin', 'https://hannaealgo.vercel.app')
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
                # Use the trailing 5-bar mean (~25min) instead of the single
                # last bar — a one-bar spike was inflating the ratio and the
                # technical layer's volume score on noise.
                vol_recent = spy_h["Volume"].tail(5).mean()
                vol_sma = spy_h["Volume"].rolling(window=20).mean()
                if (not pd.isna(vol_recent) and not vol_sma.empty
                        and pd.notna(vol_sma.iloc[-1]) and vol_sma.iloc[-1] > 0):
                    vol_r = float(vol_recent / vol_sma.iloc[-1])
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
            # score_engine already produced a complete signal (including LOCKED for risk/veto).
            # Only override with halt signal if daily loss limit hit AND we aren't already locked.
            # NOTE: anchor is the day-open equity, refreshed once per trading
            # day, so the 6% limit governs today's loss — not cumulative
            # drawdown from the original starting balance.
            initial_bal  = float(portfolio.get("initial_balance", STARTING_BALANCE) or STARTING_BALANCE)
            current_val  = float(portfolio.get("current_value", initial_bal) or initial_bal)
            session_date_today = now.strftime("%Y-%m-%d")
            if portfolio.get("daily_session_date") != session_date_today:
                portfolio["daily_start_value"] = current_val
                portfolio["daily_session_date"] = session_date_today
                # On day rollover, snapshot yesterday's peak into daily_peaks
                # (no buffer wipe — score_samples + peak_score accumulate).
                yday_peak = portfolio.get("peak_score_today_internal")
                if yday_peak and yday_peak.get("minute"):
                    dp = portfolio.setdefault("daily_peaks", [])
                    dp.append(yday_peak)
                    # Keep last 252 days (1 trading year) of peaks
                    if len(dp) > 252:
                        portfolio["daily_peaks"] = dp[-252:]
                # Reset only the *internal* today-tracker (used to compute
                # daily snapshot). score_samples + peak_score persist.
                portfolio["peak_score_today_internal"] = {
                    "date": session_date_today,
                    "score": 0, "minute": None, "grade": "NONE", "bias": "NEUTRAL",
                }
            daily_anchor = float(portfolio.get("daily_start_value", current_val) or current_val)
            daily_dd_pct = round((daily_anchor - current_val) / daily_anchor * 100, 2) if daily_anchor > 0 else 0.0
            if daily_dd_pct >= DAILY_LOSS_LIMIT * 100 and grade != "LOCKED":
                signal = {
                    "grade": "LOCKED",
                    "label": "DAILY LIMIT HALT",
                    "emoji": "⛔",
                    "action": f"Daily loss {daily_dd_pct:.1f}% — Trading halted",
                    "color": "#f07178",
                    "halted": True,
                }
                grade = "LOCKED"

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
            # Compute entry decision + diagnostic ONCE, surface in response
            if open_pos:
                entry_passed, entry_reason = False, "POSITION_ALREADY_OPEN"
            elif not is_regular:
                entry_passed, entry_reason = False, f"SESSION_{('CLOSED' if not is_regular else 'REGULAR')}"
            else:
                entry_passed, entry_reason = _entry_check(grade, direction_bias, score_result, portfolio, now)

            # ── Score samples + peak tracker (continuous, no daily wipe) ──
            # score_samples persists across days as a rolling 2000-entry
            # buffer (~5 trading days). peak_score is all-time. A separate
            # peak_score_today_internal tracks the running daily max which
            # gets archived to daily_peaks on date rollover (above).
            if is_regular:
                cur_min_str = now.strftime("%H:%M")
                cur_dt_str  = f"{session_date_today} {cur_min_str}"
                samples = portfolio.setdefault("score_samples", [])
                # Dedupe by (day, minute) — overwrite if same minute polls twice
                if not samples or samples[-1].get("ts") != cur_dt_str:
                    samples.append({
                        "ts":     cur_dt_str,
                        "date":   session_date_today,
                        "min":    cur_min_str,
                        "score":  normalized,
                        "grade":  grade,
                        "bias":   direction_bias,
                        "reason": entry_reason if not entry_passed else "ENTRY_OK",
                    })
                    # Cap at 5000 entries (~13 trading days), rolling
                    if len(samples) > 5000:
                        portfolio["score_samples"] = samples[-5000:]

                # All-time peak tracker
                peak = portfolio.setdefault("peak_score",
                                             {"score": 0, "date": None, "minute": None,
                                              "grade": "NONE", "bias": "NEUTRAL"})
                if normalized > peak.get("score", 0):
                    peak.update({
                        "score":  normalized,
                        "date":   session_date_today,
                        "minute": cur_min_str,
                        "grade":  grade,
                        "bias":   direction_bias,
                    })

                # Today's intra-day peak (used for daily_peaks archival)
                today_peak = portfolio.setdefault("peak_score_today_internal",
                                                   {"date": session_date_today, "score": 0,
                                                    "minute": None, "grade": "NONE", "bias": "NEUTRAL"})
                if normalized > today_peak.get("score", 0):
                    today_peak.update({
                        "date":   session_date_today,
                        "score":  normalized,
                        "minute": cur_min_str,
                        "grade":  grade,
                        "bias":   direction_bias,
                    })

            if entry_passed:
                cash = portfolio["cash"]
                es_direction = direction_bias
                
                # ATR-based SL (use range as proxy)
                atr_proxy = max(d_range, 2.0) if d_range > 0 else 4.0
                sl_points = max(ATR_SL_MULT * atr_proxy, 2.0)
                sl_points = min(sl_points, 15.0)

                # Regime-aware reward/risk ratio — trending markets get
                # wider targets; choppy/uncertain regimes take quicker
                # profits to avoid round-tripping.
                regime_label = score_result["layers"].get("regime", {}).get("regime", "UNKNOWN")
                if regime_label in ("TRENDING", "BREAKOUT"):
                    rr_ratio = 3.0
                elif regime_label == "CHOPPY":
                    rr_ratio = 1.5
                else:
                    rr_ratio = 2.0
                tp_points = sl_points * rr_ratio
                
                # Position sizing
                from lib.feature_flags import is_enabled as _ff
                # PERMISSIVE 모드: 정상 STRONG 신호 대비 50% 크기로 축소
                # (validation 목적의 진입이므로 자본 보호)
                risk_mult = 0.5 if _ff("paper_permissive") else 1.0
                risk_per_contract = sl_points * ES_MULTIPLIER + ES_COMMISSION_RT
                max_risk = cash * RISK_PCT * risk_mult
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
                    
                    is_permissive = _ff("paper_permissive")
                    mode_tag = "PERMISSIVE" if is_permissive else "STRICT"
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
                        "mode": mode_tag,
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
                
                # ── Trailing Stop only — BE disabled by backtest ──────
                # 3-year backtest: BE exit fired 4 times, all 4 were
                # losses (winners got stopped out at entry instead of
                # running to TP/EOD). Premature BE protection killed
                # exactly the trades that needed runway. Trailing stop
                # (activated later at +1.5×SL) replaces it.
                entry_p  = open_pos.get("entry_price", spy_p)
                sl_pts   = open_pos.get("sl_points", 4.0)
                tp_pts   = open_pos.get("tp_points", 8.0)

                # Trailing stop: once +1.5×SL in profit, trail by 1×SL
                if es_dir == "LONG" and point_pnl >= sl_pts * 1.5:
                    trail_sl = spy_p - sl_pts
                    if trail_sl > open_pos.get("sl_price", 0):
                        open_pos["sl_price"] = round(trail_sl, 2)
                        open_pos["trail_activated"] = True
                        sl_price = open_pos["sl_price"]
                elif es_dir == "SHORT" and point_pnl >= sl_pts * 1.5:
                    trail_sl = spy_p + sl_pts
                    if trail_sl < open_pos.get("sl_price", 9999):
                        open_pos["sl_price"] = round(trail_sl, 2)
                        open_pos["trail_activated"] = True
                        sl_price = open_pos["sl_price"]

                # ── Exit conditions ───────────────────────────────────
                invalid_reason = _position_invalid_reason(open_pos, grade, direction_bias, score_result)
                if invalid_reason:
                    exit_type = invalid_reason
                elif es_dir == "LONG" and sl_price and spy_p <= sl_price:
                    exit_type = "TRAIL" if open_pos.get("trail_activated") else "SL"
                elif es_dir == "SHORT" and sl_price and spy_p >= sl_price:
                    exit_type = "TRAIL" if open_pos.get("trail_activated") else "SL"
                elif es_dir == "LONG" and tp_price and spy_p >= tp_price:
                    exit_type = "TP"
                elif es_dir == "SHORT" and tp_price and spy_p <= tp_price:
                    exit_type = "TP"
                elif not is_regular or (now.hour >= 15 and now.minute >= 0):
                    # Futures settlement is 15:00 ET — exit earlier than
                    # equity 15:30 cushion to avoid settlement pin risk.
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

            # Multi-instrument futures dashboard (data-only; execution still MES)
            futures_out = {}
            for fut, meta in FUTURES_PROXIES.items():
                s = snaps.get(meta["proxy"], {})
                px = _snap_price(s)
                pct = _pct(px, _snap_prev_close(s) or 1)
                futures_out[fut] = {
                    "proxy_etf": meta["proxy"],
                    "name": meta["name"],
                    "multiplier": meta["multiplier"],
                    "proxy_price": px,
                    "proxy_pct": pct,
                }

            gme_data = {}
            for sym in GME_STOCK:
                s = snaps.get(sym, {})
                gme_data[sym] = {"price": _snap_price(s), "pct": _pct(_snap_price(s), _snap_prev_close(s) or 1)}

            order_flow = _calculate_es_order_flow(
                spy_price=spy_p, vix_price=vix_p,
                normalized_score=normalized, direction_bias=direction_bias,
            )

            # Day counters (Day 1 = TRADING_START_DATE, currently 2026-05-25)
            start_dt = datetime.strptime(TRADING_START_DATE, "%Y-%m-%d").date()
            today_dt = now.date()
            calendar_day = max(1, (today_dt - start_dt).days + 1)
            # Trading day = weekdays (Mon-Fri) minus NYSE_HOLIDAYS
            trading_day_count = 0
            d = start_dt
            while d <= today_dt:
                if d.weekday() < 5 and d not in NYSE_HOLIDAYS:
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

            # ── Portfolio Heat (총 노출 위험) ──────────────────────
            # 현재 오픈된 모든 포지션의 잠재 손실(SL까지) 합 / 계좌 잔고.
            # 다중 포지션 환경에서 단일 트레이드 1.5% × N으로 누적 위험이
            # 6% 일간 한도를 넘는 일을 막는다.
            open_positions = list((portfolio.get("positions") or {}).values())
            heat_risk_usd = 0.0
            for p in open_positions:
                entry = p.get("entry_price") or 0
                sl    = p.get("sl_price") or entry
                contracts = p.get("contracts", 0) or 0
                heat_risk_usd += abs(entry - sl) * ES_MULTIPLIER * contracts
            heat_pct = (heat_risk_usd / current_val * 100) if current_val > 0 else 0.0
            heat_status = "OK" if heat_pct < 3.0 else ("WARNING" if heat_pct < 5.0 else "DANGER")
            portfolio_heat = {
                "open_positions": len(open_positions),
                "total_risk_usd": round(heat_risk_usd, 2),
                "heat_pct": round(heat_pct, 2),
                "status": heat_status,
                "limit_pct": 5.0,
            }

            # ── Data Health snapshot ─────────────────────────────────
            # Surfaces which upstream sources are actually delivering vs
            # which are degraded/down. Lets the frontend show a status
            # indicator instead of users wondering why a card is blank.
            # CLOSED status distinguishes "no data because market shut"
            # from "no data because API broke" — avoids false alarms.
            #
            # FlashAlpha STALE logic:
            #   - Market closed → CLOSED (stale data is expected off-hours)
            #   - Market open + Alpaca bars available → OK (Alpaca VWAP/vol fallback works)
            #   - Market open + NO Alpaca bars → STALE (genuine degradation)
            market_closed = status == "closed"
            spy_h = bundle.get("spy_h")
            has_bars = spy_h is not None and not spy_h.empty

            # Determine FlashAlpha effective status
            if flashalpha_spy and not flashalpha_spy.get("is_stale"):
                fa_status = "OK"
            elif flashalpha_spy and flashalpha_spy.get("is_stale"):
                if market_closed:
                    fa_status = "CLOSED"  # expected: no updates off-hours
                elif has_bars:
                    fa_status = "OK"      # Alpaca bars provide VWAP/volume fallback
                else:
                    fa_status = "STALE"   # genuine issue: no fallback available
            else:
                fa_status = "CLOSED" if market_closed else "DOWN"

            data_health = {
                "alpaca_snapshots": "OK" if bundle.get("snaps") else ("CLOSED" if market_closed else "DOWN"),
                "alpaca_bars": "OK" if has_bars else ("CLOSED" if market_closed else "DOWN"),
                "vix": _VIX_CACHE.get("source", "UNKNOWN"),
                "vix_fetch_ok": bool(_VIX_CACHE.get("fetch_ok", False)),
                "flashalpha": fa_status,
                "polygon_fallback_active": any(
                    (s or {}).get("_source") == "polygon_fallback"
                    for s in (bundle.get("snaps", {}).get("snapshots") or {}).values()
                ),
                "market_status": status,  # convenience — saves frontend a lookup
            }

            # ── ML Stats snapshot (sample count + confidence) ────────
            try:
                from engines.ml_weights import get_ml_stats
                ml_stats = get_ml_stats()
            except Exception as e:
                ml_stats = {"error": str(e)[:120], "confidence": "ERROR"}

            # Adaptive next-poll hint — frontend uses this instead of a
            # hardcoded 10s interval. Vercel Python serverless can't hold a
            # WebSocket, so we lean on shorter polls during active windows
            # and longer polls when nothing can change.
            if status == "regular":
                next_poll_sec = 3 if grade == "STRONG" else 5
            elif status in ("pre_market", "after_hours"):
                next_poll_sec = 15
            else:
                next_poll_sec = 60

            # Tail-risk snapshot — used in ETag basis AND response body, so
            # compute once before either consumer (cheap; reads in-mem EWMA).
            tail_risk = _tail_risk_status(vix_p)

            final = {
                "last_updated": ts, "fetch_status": "SUCCESS", "latency_ms": latency,
                "next_poll_sec": next_poll_sec,
                "timing_ms": fetch_timing,
                "data_source": "ALPACA + FlashAlpha" if flashalpha_spy else "ALPACA",
                "flashalpha": flashalpha_spy,
                "session": "REGULAR" if is_regular else "CLOSED",
                "market_status": status,                    # NEW: regular / pre_market / after_hours / closed
                "holiday_info": get_holiday_info(now),       # NEW: {is_holiday, name, is_weekend, is_closed_day}
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
                "gme_data": gme_data,
                "futures_multi": futures_out,
                # Heavy diagnostic fields (score_samples, daily_peaks) are
                # excluded — their summarized stats are already in
                # entry_diagnostic. Saves ~256 KB per response when buffer
                # is full (5000 samples × ~130 bytes/sample).
                "paper_trading": {k: v for k, v in portfolio.items()
                                  if not k.startswith("_")
                                  and k not in ("storage_type", "revision", "last_saved",
                                                 "score_samples", "daily_peaks",
                                                 "peak_score_today_internal")},
                "backtest_summary": BACKTEST_SUMMARY,
                "paper_trading_stats": paper_trading_stats,
                "portfolio_heat": portfolio_heat,
                "tail_risk": tail_risk,
                "entry_diagnostic": {
                    "passed": entry_passed,
                    "reason": entry_reason,
                    "human": "✅ Entry criteria met — order submitted" if entry_passed
                             else f"⏸ Blocked: {entry_reason}",
                    "peak_today": portfolio.get("peak_score_today_internal"),
                    "peak_all_time": portfolio.get("peak_score"),
                    "samples_count": len(portfolio.get("score_samples") or []),
                    "samples_today_count": sum(
                        1 for s in (portfolio.get("score_samples") or [])
                        if s.get("date") == session_date_today
                    ),
                    "daily_peaks_count": len(portfolio.get("daily_peaks") or []),
                    "storage_backend": portfolio.get("_storage") or _storage_backend(),
                    # KV "persisted" only when (a) configured AND (b) the most
                    # recent save actually reached the remote — bare config
                    # presence (kv_configured) is a separate field.
                    "kv_configured": _storage_backend() == "upstash",
                    "kv_persisted": (_storage_backend() == "upstash"
                                     and portfolio.get("_remote_ok") is True),
                    "save_skipped": portfolio.get("_save_skipped"),
                },
                "ic_signal": _build_ic_signal(now, spy_p, spy_prev, spy_h, vix_p,
                                              pcts_data, score_result, vol_r),
                "feature_flags": _feature_flags_snapshot(),
                "mes_specs": {
                    "instrument": "MES",
                    "contract_month": _current_mes_contract(now),
                    "multiplier": ES_MULTIPLIER,
                    "commission_rt": ES_COMMISSION_RT,
                    "day_margin": ES_DAY_MARGIN,
                    "tick_size": ES_TICK_SIZE,
                    "risk_pct": RISK_PCT,
                    "daily_loss_limit_pct": DAILY_LOSS_LIMIT * 100,
                    "trailing_stop": True,
                    "breakeven_disabled": True,
                    "eod_close_time": "15:00 ET (futures settlement)",
                    "settlement_time": "15:00 ET",
                    "roll_window_active": _is_quarterly_roll_window(now),
                    "days_to_roll": _days_to_next_roll(now),
                },
                "daily_halt": daily_dd_pct >= DAILY_LOSS_LIMIT * 100,
                "daily_drawdown_pct": round(daily_dd_pct, 2),
                "data_health": data_health,
                "ml_stats": ml_stats,
            }

            # ETag from signal-relevant subset (so unchanged signals don't
            # ship a full payload — the client just polls and gets 304).
            # Bucket by 5-minute window so identical signals across short
            # spans keep returning 304 (was per-minute → every minute the
            # ETag flipped and forced a 200, wasting bandwidth).
            of_layer = score_result["layers"].get("options_flow", {})
            etag_basis = {
                "grade": signal.get("grade"),
                "score": normalized,
                "bias": direction_bias,
                # Include date so identical signals on different days
                # never collide (was: hour:minute/5 only → 00:01 day-1
                # and 00:01 day-2 had same ETag when score stayed at 0).
                "date": session_date_today,
                "bucket": f"{now.hour}:{now.minute // 5}",
                "open_pos": today_str in (portfolio.get("positions") or {}),
                "dd": daily_dd_pct,
                "tail": tail_risk.get("status"),
                # Track OF transitions so a flip in options sentiment
                # invalidates the ETag even if normalized score didn't move.
                "of_dir": of_layer.get("direction"),
                "of_status": of_layer.get("status"),
                "macro": score_result["layers"].get("macro_gate", {}).get("status"),
            }
            import hashlib
            etag = '"' + hashlib.md5(json.dumps(etag_basis, sort_keys=True).encode()).hexdigest()[:16] + '"'
            client_etag = self.headers.get("If-None-Match", "")

            origin = self.headers.get("Origin", "")

            if client_etag and client_etag == etag:
                # Nothing material changed since the last poll.
                self.send_response(304)
                self.send_header('ETag', etag)
                if is_origin_allowed(origin):
                    self.send_header('Access-Control-Allow-Origin', origin)
                else:
                    self.send_header('Access-Control-Allow-Origin', 'https://hannaealgo.vercel.app')
                self.send_header('Access-Control-Allow-Credentials', 'true')
                self.send_header('Access-Control-Expose-Headers', 'ETag, X-Next-Poll')
                self.send_header('X-Next-Poll', str(next_poll_sec))
                self.end_headers()
                return

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            if is_origin_allowed(origin):
                self.send_header('Access-Control-Allow-Origin', origin)
            else:
                self.send_header('Access-Control-Allow-Origin', 'https://hannaealgo.vercel.app')
            self.send_header('Access-Control-Allow-Credentials', 'true')
            self.send_header('Access-Control-Expose-Headers', 'ETag, X-Next-Poll')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('ETag', etag)
            self.send_header('X-Next-Poll', str(next_poll_sec))
            self.end_headers()
            self.wfile.write(json.dumps(final, cls=SafeEncoder).encode('utf-8'))

        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"[Error in API] {err_msg}")
            
            origin = self.headers.get("Origin", "")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            if is_origin_allowed(origin):
                self.send_header('Access-Control-Allow-Origin', origin)
            else:
                self.send_header('Access-Control-Allow-Origin', 'https://hannaealgo.vercel.app')
            self.end_headers()
            
            # Hide detailed traceback in production/Vercel
            is_prod = os.getenv("VERCEL") is not None
            err_resp = {
                "error": "Internal Server Error",
                "fetch_status": "ERROR",
                "message": str(e) if not is_prod else "An error occurred on the server."
            }
            self.wfile.write(json.dumps(err_resp).encode('utf-8'))

