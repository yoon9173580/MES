"""
Vercel Serverless API — /api/data
SPY 0DTE Signal Machine — 7-Layer Score Engine
Google SSO only (replaced legacy UNLOCK key/password) + Session Cookie (Last Update: 2026-05-23)
"""
import math, json, os, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler
from datetime import datetime, timedelta
import pytz, requests

NY = pytz.timezone("America/New_York")
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from engines.score_engine import run_score_engine
STARTING_BALANCE = 500.0
TRADING_START_DATE = "2026-05-22"   # Day 1 of paper trading
PDT_THRESHOLD = 2000.0      # accounts below this are subject to PDT
PDT_MAX_DAY_TRADES = 3      # max day trades per rolling 5 business days
PDT_WINDOW_DAYS = 5         # rolling business-day window

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bs_price(S, K, T, r, sigma, opt="call"):
    if T <= 0: return max(S - K, 0) if opt == "call" else max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def dynamic_slippage(vix):
    """VIX-adaptive slippage per contract side (one-way).
    Base  : $0.03 (VIX < 20, calm market)
    Mid   : $0.05 (VIX 20-25)
    Elevated: $0.06 (VIX 25-30)
    High  : $0.08 (VIX 30+)
    """
    if vix >= 30:
        return 0.08
    if vix >= 25:
        return 0.06
    if vix >= 20:
        return 0.05
    return 0.03

ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_HEADERS = {
    "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
    "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", ""),
}
FLASHALPHA_API_KEY = os.getenv("FLASHALPHA_API_KEY", "")
FLASHALPHA_API_URL = "https://lab.flashalpha.com/v1"
_VIX_CACHE = {"at": 0.0, "vix": 18.0, "vix3m": None}
VIX_CACHE_SEC = int(os.getenv("VIX_CACHE_SEC", "45"))
YAHOO_UA = "Mozilla/5.0 (compatible; SPY0DTE/1.0)"


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


def is_market_open(dt):
    """Detect if US stock market is currently open (9:30 AM to 4:00 PM Eastern Time, weekdays only, no NYSE holidays)."""
    # 1. Weekdays only (Monday=0 to Sunday=6)
    if dt.weekday() >= 5:
        return False
    # 2. NYSE holidays 2026
    nyse_holidays_2026 = {
        datetime(2026, 1, 1).date(),    # New Year's Day
        datetime(2026, 1, 19).date(),   # MLK Day
        datetime(2026, 2, 16).date(),   # Presidents' Day
        datetime(2026, 4, 3).date(),    # Good Friday
        datetime(2026, 5, 25).date(),   # Memorial Day
        datetime(2026, 6, 19).date(),   # Juneteenth
        datetime(2026, 7, 3).date(),    # Independence Day (observed)
        datetime(2026, 9, 7).date(),    # Labor Day
        datetime(2026, 11, 26).date(),  # Thanksgiving
        datetime(2026, 12, 25).date(),  # Christmas
    }
    if dt.date() in nyse_holidays_2026:
        return False
    # 3. Regular hours: 9:30 AM to 4:00 PM Eastern
    t_min = dt.hour * 60 + dt.minute
    return 570 <= t_min <= 960

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
    m_open = is_market_open(now)
    ttl = 10 if m_open else 1800  # 10 seconds if open, 30 minutes if closed
    
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


# ── Scoring Engine (imported) ──────────────────────────────────────

def _get_0dte_option_chain(spy_price, vix_price):
    """
    Fetch 0DTE SPY Option Chain from Alpaca Option Market Data API.
    Returns:
      dict: A map of {strike: {"call": contract_snapshot, "put": contract_snapshot}}
      or None if API fails or lacks permission.
    """
    if not spy_price:
        return None
        
    # Optimization: Skip API calls and let it fallback to BS directly when market is closed
    now_ny = datetime.now(NY)
    if not is_market_open(now_ny):
        return None
    api_key = ALPACA_HEADERS.get("APCA-API-KEY-ID", "")
    api_secret = ALPACA_HEADERS.get("APCA-API-SECRET-KEY", "")
    if not api_key or not api_secret:
        return None

    # 1. Get today's date in NY Time
    now_ny = datetime.now(NY)
    today_str = now_ny.strftime("%Y-%m-%d")

    try:
        # 2. Get active SPY contracts for today or closest next date from Trading API
        is_paper = api_key.startswith("PK")
        trading_url = "https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets"
        contracts_url = f"{trading_url}/v2/options/contracts"
        
        params = {
            "underlying_symbol": "SPY",
            "status": "active",
            "expiration_date_gte": today_str,
            "limit": 100
        }
        r = requests.get(contracts_url, headers=ALPACA_HEADERS, params=params, timeout=5)
        if r.status_code != 200:
            print(f"[Alpaca Options] Error fetching contracts from {contracts_url} (HTTP {r.status_code}): {r.text}")
            return None

        contracts_data = r.json().get("option_contracts", [])
        if not contracts_data:
            return None

        # Sort and find the closest expiration date
        expirations = sorted(list(set(c["expiration_date"] for c in contracts_data)))
        target_exp = expirations[0] if expirations else today_str

        # 3. Filter contracts by closest expiration and strikes around spy_price (+/- 4 strikes)
        strike_center = int(round(spy_price))
        strikes_to_fetch = set(range(strike_center - 4, strike_center + 5))

        filtered_contracts = []
        for c in contracts_data:
            if c.get("expiration_date") == target_exp:
                try:
                    strike = int(float(c["strike_price"]))
                    if strike in strikes_to_fetch:
                        filtered_contracts.append(c)
                except (ValueError, TypeError):
                    continue

        if not filtered_contracts:
            return None

        # 4. Get snapshots for filtered contracts using Market Data API v1beta1
        symbols = [c["symbol"] for c in filtered_contracts]
        symbols_str = ",".join(symbols)

        snapshots_url = f"{ALPACA_DATA_URL}/v1beta1/options/snapshots"
        snap_params = {"symbols": symbols_str}
        snap_r = requests.get(snapshots_url, headers=ALPACA_HEADERS, params=snap_params, timeout=5)
        
        # If forbidden or subscription error, automatically fallback to indicative feed
        if snap_r.status_code == 403 or "subscription" in snap_r.text.lower():
            print(f"[Alpaca Options] 403/Subscription restriction on default feed. Retrying with indicative feed...")
            snap_params["feed"] = "indicative"
            snap_r = requests.get(snapshots_url, headers=ALPACA_HEADERS, params=snap_params, timeout=5)

        if snap_r.status_code != 200:
            print(f"[Alpaca Options] Error fetching snapshots from {snapshots_url} (HTTP {snap_r.status_code}): {snap_r.text}")
            return None

        snapshots = snap_r.json().get("snapshots", {})
        if not snapshots:
            return None

        # 5. Build structured chain map
        chain_map = {}
        for strike in strikes_to_fetch:
            chain_map[strike] = {"call": None, "put": None}

        for c in filtered_contracts:
            sym = c["symbol"]
            strike = int(float(c["strike_price"]))
            opt_type = c["type"].lower()  # "call" or "put"

            if strike not in chain_map:
                continue

            snap = snapshots.get(sym, {})
            quote = snap.get("latestQuote", {})
            trade = snap.get("latestTrade", {})

            bid = quote.get("bp")
            ask = quote.get("ap")
            last = trade.get("p")

            chain_map[strike][opt_type] = {
                "symbol": sym,
                "bid": float(bid) if bid is not None else None,
                "ask": float(ask) if ask is not None else None,
                "last": float(last) if last is not None else None,
                "bid_size": quote.get("bs"),
                "ask_size": quote.get("as"),
            }

        return chain_map

    except Exception as e:
        print(f"Error fetching Alpaca option chain: {e}")
        return None


def _calculate_strike_recommendation(spy_price, direction_bias, signal_grade,
                                      vix_price, vwap, normalized_score,
                                      portfolio_cash, now_et):
    if spy_price is None or direction_bias == "NEUTRAL" or signal_grade in ("NONE",):
        # We still want to return a default Option Chain even if no active recommendation,
        # so users can see the Option Chain live on the dashboard anytime.
        pass

    atm_strike = round(spy_price) if spy_price is not None else 500
    if signal_grade == "STRONG": otm_offset, strike_reasoning = 0, "ATM — high conviction"
    elif signal_grade == "MODERATE": otm_offset, strike_reasoning = 1, "OTM-1 — balanced cost/probability"
    else: otm_offset, strike_reasoning = 2, "OTM-2 — monitor zone"

    if direction_bias == "CALL":
        recommended_strike = atm_strike + otm_offset
        otm_1, otm_2 = atm_strike + 1, atm_strike + 2
        contract_type, type_label = "C", "CALL"
    else:
        recommended_strike = atm_strike - otm_offset
        otm_1, otm_2 = atm_strike - 1, atm_strike - 2
        contract_type, type_label = "P", "PUT"

    contract_label = f"SPY ${recommended_strike}{contract_type} 0DTE"

    # 1. Try fetching real option chain
    vix_val = vix_price if vix_price else 18.0
    iv = vix_val / 100.0
    T = max(1.0 / (252.0 * 6.5), 1e-4)

    chain_data = _get_0dte_option_chain(spy_price, vix_price)
    
    # 2. Extract values for recommended strike
    real_bid, real_ask, real_last = None, None, None
    data_source = "BS_ESTIMATE"
    mid_premium = 0.0

    if chain_data and recommended_strike in chain_data:
        opt_key = "call" if contract_type == "C" else "put"
        opt_snap = chain_data[recommended_strike][opt_key]
        if opt_snap and opt_snap.get("bid") is not None and opt_snap.get("ask") is not None:
            real_bid = opt_snap["bid"]
            real_ask = opt_snap["ask"]
            real_last = opt_snap["last"]
            mid_premium = max(round((real_bid + real_ask) / 2.0, 2), 0.01)
            data_source = "LIVE"

    if data_source == "BS_ESTIMATE":
        opt = "call" if contract_type == "C" else "put"
        atm_est = bs_price(spy_price, recommended_strike, T, 0.05, iv, opt)
        otm_discount = max(0.35, 1.0 - (otm_offset * 0.2))
        mid_premium = max(round(atm_est * otm_discount, 2), 0.05)

    est_premium_low = real_bid if real_bid is not None else max(round(mid_premium * 0.85, 2), 0.01)
    est_premium_high = real_ask if real_ask is not None else max(round(mid_premium * 1.15, 2), 0.10)

    # 3. Dynamic target and stop prices
    target_pct, stop_pct = 50, 30
    target_price = round(mid_premium * 1.5, 2)
    stop_price = round(mid_premium * 0.7, 2)
    risk_per = round(mid_premium * (stop_pct / 100.0), 2)
    reward_per = round(mid_premium * (target_pct / 100.0), 2)
    rr_ratio = round(reward_per / risk_per, 2) if risk_per > 0 else 0

    max_risk_pct = 10.0
    max_risk_dollars = round(portfolio_cash * (max_risk_pct / 100.0), 2)
    cost_per_contract = round(mid_premium * 100, 2)
    max_contracts = max(1, int(max_risk_dollars / cost_per_contract)) if cost_per_contract > 0 else 0

    # 4. Construct Option Chain payload for the frontend
    option_chain_list = []
    strike_center = int(round(spy_price)) if spy_price else 500
    strikes_range = range(strike_center - 3, strike_center + 4)

    for strike in strikes_range:
        call_info = {"bid": None, "ask": None, "last": None, "source": "ESTIMATE"}
        put_info = {"bid": None, "ask": None, "last": None, "source": "ESTIMATE"}

        if chain_data and strike in chain_data:
            # Live Call
            c_snap = chain_data[strike]["call"]
            if c_snap and c_snap.get("bid") is not None:
                call_info = {"bid": c_snap["bid"], "ask": c_snap["ask"], "last": c_snap["last"], "source": "LIVE"}
            # Live Put
            p_snap = chain_data[strike]["put"]
            if p_snap and p_snap.get("bid") is not None:
                put_info = {"bid": p_snap["bid"], "ask": p_snap["ask"], "last": p_snap["last"], "source": "LIVE"}

        # Fallback to BS if no live data
        if call_info["source"] == "ESTIMATE":
            c_est = bs_price(spy_price, strike, T, 0.05, iv, "call")
            call_info["bid"] = max(round(c_est * 0.95, 2), 0.01)
            call_info["ask"] = max(round(c_est * 1.05, 2), 0.02)
            call_info["last"] = max(round(c_est, 2), 0.01)

        if put_info["source"] == "ESTIMATE":
            p_est = bs_price(spy_price, strike, T, 0.05, iv, "put")
            put_info["bid"] = max(round(p_est * 0.95, 2), 0.01)
            put_info["ask"] = max(round(p_est * 1.05, 2), 0.02)
            put_info["last"] = max(round(p_est, 2), 0.01)

        is_recommended = (strike == recommended_strike) and (direction_bias != "NEUTRAL" and signal_grade != "NONE")

        option_chain_list.append({
            "strike": strike,
            "is_recommended": is_recommended,
            "call": call_info,
            "put": put_info
        })

    active_rec = (spy_price is not None) and (direction_bias != "NEUTRAL") and (signal_grade not in ("NONE",))

    return {
        "active": active_rec, "direction": type_label if active_rec else "NEUTRAL", "atm_strike": atm_strike,
        "recommended_strike": recommended_strike, "otm_offset": otm_offset,
        "contract_label": contract_label if active_rec else "SPY 0DTE", "contract_type": contract_type if active_rec else "C",
        "strikes": [
            {"label": "ATM", "strike": atm_strike, "recommended": otm_offset == 0 and active_rec},
            {"label": "OTM-1", "strike": otm_1, "recommended": otm_offset == 1 and active_rec},
            {"label": "OTM-2", "strike": otm_2, "recommended": otm_offset == 2 and active_rec},
        ],
        "est_premium_low": est_premium_low, "est_premium_high": est_premium_high,
        "mid_premium": mid_premium, "data_source": data_source,
        "real_bid": real_bid, "real_ask": real_ask, "real_last": real_last,
        "target_pct": target_pct, "stop_pct": stop_pct,
        "target_price": target_price, "stop_price": stop_price,
        "risk_reward": f"1:{rr_ratio}", "max_contracts": max_contracts,
        "cost_per_contract": cost_per_contract, "max_risk_dollars": max_risk_dollars,
        "max_risk_pct": max_risk_pct, "reasoning": strike_reasoning,
        "option_chain": option_chain_list
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
                "K_buy": h.get("K_buy"),
                "K_sell": h.get("K_sell"),
                "contracts": h.get("contracts"),
                "entry_spy": h.get("entry_spy"),
                "exit_spy": h.get("exit_spy"),
                "net_debit": h.get("net_debit"),
                "exit_val": h.get("exit_val"),
                "exit_type": h.get("exit_type"),
                "cost": h.get("cost"),
                "revenue": h.get("revenue"),
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
    return f"{record.get('date')}-{entry}-{record.get('direction')}-{record.get('K_buy')}-{record.get('K_sell')}"


def _ensure_trade_id(record):
    if not isinstance(record, dict):
        return record
    if not record.get("entry_time") and record.get("time"):
        record["entry_time"] = record["time"]
    if not record.get("trade_id"):
        entry = record.get("entry_time") or record.get("time") or "00:00"
        record["trade_id"] = f"{record.get('date')}-{entry}-{record.get('direction')}-{record.get('K_buy')}"
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
        f"{row.get('K_buy')}:{row.get('K_sell')}:{row.get('exit_time') or row.get('logged_at', '')}"
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
    sig = (record.get("date"), record.get("direction"), record.get("K_buy"), record.get("K_sell"), record.get("exit_time"))
    for r in rows:
        if r.get("display_status") != "CLOSE":
            continue
        if (r.get("date"), r.get("direction"), r.get("K_buy"), r.get("K_sell"), r.get("exit_time")) == sig:
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
    contracts = int(pos.get("contracts", 0) or 0)
    cost = float(pos.get("cost", 0) or 0)
    revenue = round(float(exit_val) * 100 * contracts, 2)
    pnl = round(revenue - cost, 2)
    pnl_pct = round((pnl / cost) * 100, 1) if cost > 0 else 0.0
    pos.update({
        "status": "CLOSED",
        "action": "SELL",
        "exit_time": now.strftime("%H:%M"),
        "exit_ts": now.strftime("%Y-%m-%d %H:%M:%S"),
        "exit_spy": round(spy_p, 2) if spy_p else None,
        "exit_val": round(float(exit_val), 2),
        "exit_type": exit_type,
        "revenue": revenue,
        "pnl": pnl,
        "realized_pnl": pnl,
        "pnl_pct": pnl_pct,
        "pnl_locked": True,
        "win": pnl > 0,
    })
    for key in ("unrealized_pnl", "unrealized_pnl_pct", "mark_spy", "mark_time"):
        pos.pop(key, None)
    return pos


def _count_day_trades(portfolio, now):
    """Count day trades in the rolling 5-business-day window.
    A day trade = open and close on the same calendar day."""
    history = portfolio.get("history") or []
    # Build list of the last N business days (exclude weekends)
    biz_days = []
    d = now.date() if hasattr(now, 'date') else now
    while len(biz_days) < PDT_WINDOW_DAYS:
        if d.weekday() < 5:  # Mon-Fri
            biz_days.append(d.strftime("%Y-%m-%d"))
        d -= timedelta(days=1)
    count = 0
    for h in history:
        if not isinstance(h, dict):
            continue
        trade_date = h.get("date", "")
        if trade_date not in biz_days:
            continue
        # A day trade: opened and closed on the same day
        if h.get("status") == "CLOSED" or h.get("pnl_locked"):
            count += 1
    return count, biz_days


def _pdt_status(portfolio, now):
    """Return PDT enforcement info dict."""
    account_value = portfolio.get("cash", 0) + sum(
        float(p.get("cost", 0)) for p in (portfolio.get("positions") or {}).values()
    )
    subject = account_value < PDT_THRESHOLD
    used, window = _count_day_trades(portfolio, now)
    remaining = max(0, PDT_MAX_DAY_TRADES - used) if subject else 999
    blocked = subject and used >= PDT_MAX_DAY_TRADES
    return {
        "subject": subject,
        "threshold": PDT_THRESHOLD,
        "account_value": round(account_value, 2),
        "day_trades_used": used,
        "day_trades_remaining": remaining,
        "blocked": blocked,
        "window_days": biz_days_label(window),
    }


def biz_days_label(days):
    """Return first and last date of the window for display."""
    if not days:
        return ""
    return f"{days[-1]} ~ {days[0]}"


def _entry_criteria_met(grade, direction_bias, score_result, portfolio=None, now=None):
    """Same rules used to open a debit spread."""
    layers = score_result.get("layers", {})
    if layers.get("risk", {}).get("passed") is False or grade == "LOCKED":
        return False
    if grade != "STRONG":
        return False
    if layers.get("time_window", {}).get("score", 0) < 20:
        return False
    if direction_bias not in ("CALL", "PUT"):
        return False
    # ── PDT Rule ──
    if portfolio and now:
        pdt = _pdt_status(portfolio, now)
        if pdt["blocked"]:
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
        "K_buy": open_pos.get("K_buy"),
        "K_sell": open_pos.get("K_sell"),
        "contracts": open_pos.get("contracts"),
        "entry_spy": open_pos.get("entry_spy"),
        "exit_spy": open_pos.get("exit_spy"),
        "net_debit": open_pos.get("net_debit"),
        "exit_val": open_pos.get("exit_val"),
        "exit_type": open_pos.get("exit_type"),
        "cost": open_pos.get("cost"),
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
    """Verify Google Sign-In JWT token using Google OAuth2 tokeninfo endpoint."""
    if not id_token:
        return None
    try:
        r = requests.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}", timeout=5)
        if r.status_code == 200:
            payload = r.json()
            # Basic validation
            if payload.get("email_verified") == "true" or payload.get("email_verified") is True:
                return payload
    except Exception as e:
        print(f"[Google ID Token Verification Error] {e}")
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
        # We only handle /api/unlock
        if self.path.split("?")[0] != "/api/unlock":
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
            
            # Google Credential Auth ONLY (replaced legacy unlock key/password)
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
                
                # Issue secure httpOnly cookie access_token=valid
                # Max-Age: 7 days (604800)
                # Secure; SameSite=Strict; Path=/; HttpOnly
                # For local development we can also support it. SameSite=Strict ensures no CSRF.
                cookie_str = "access_token=valid; Path=/; Max-Age=604800; HttpOnly; Secure; SameSite=Strict"
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
                
                google_client_id = os.getenv("GOOGLE_CLIENT_ID", "")
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
                    "K_buy": pos.get("K_buy"),
                    "K_sell": pos.get("K_sell"),
                    "exit_type": pos.get("exit_type"),
                    "cost": pos.get("cost"),
                    "revenue": pos.get("revenue"),
                    "pnl": pos.get("pnl"),
                    "realized_pnl": pos.get("realized_pnl"),
                    "pnl_pct": pos.get("pnl_pct"),
                    "win": pos.get("win"),
                })
                to_remove.append(date_key)
            if to_remove:
                for k in to_remove: del portfolio["positions"][k]

            # 2. Check for entry (only while entry criteria hold)
            open_pos = portfolio.get("positions", {}).get(today_str)
            if not open_pos and is_regular and _entry_criteria_met(grade, direction_bias, score_result, portfolio, now):
                # Open Debit Spread — dynamic width & sizing by account tier
                iv = vix_val / 100.0
                opt = "call" if direction_bias == "CALL" else "put"
                cash = portfolio["cash"]

                # Spread width scales with account size
                if cash < 1000:
                    spread_w = 2      # $500-tier: narrow $2-wide spreads
                elif cash < 5000:
                    spread_w = 3      # $1k-tier: $3-wide spreads
                else:
                    spread_w = 5      # $5k+: full $5-wide spreads

                K_buy = round(spy_p)
                K_sell = K_buy + spread_w if opt == "call" else K_buy - spread_w

                # Try to use real option chain quotes for entry debit
                real_entry_ok = False
                net_debit = 0.0
                chain_entry = _get_0dte_option_chain(spy_p, vix_val)
                if chain_entry and K_buy in chain_entry and K_sell in chain_entry:
                    buy_snap = chain_entry[K_buy][opt]
                    sell_snap = chain_entry[K_sell][opt]
                    if buy_snap and sell_snap and buy_snap.get("ask") is not None and sell_snap.get("bid") is not None:
                        # Buy long at Ask, Sell short at Bid (Debit Spread debit cost)
                        net_debit = buy_snap["ask"] - sell_snap["bid"]
                        if net_debit > 0.02:
                            real_entry_ok = True
                
                if not real_entry_ok:
                    slip = dynamic_slippage(vix_val)
                    T_entry = 5.5 / (252 * 6.5)
                    lp = bs_price(spy_p, K_buy, T_entry, 0.05, iv, opt)
                    sp = bs_price(spy_p, K_sell, T_entry, 0.05, iv, opt)
                    net_debit = (lp - sp) * 1.03 + slip * 2

                if net_debit > 0.05:
                    # Risk % scales with account tier & signal strength
                    if cash < 1000:
                        risk_pct = 0.20 if normalized >= 95 else (0.18 if vix_val >= 25 else (0.15 if vix_val >= 20 else 0.12))
                    elif cash < 5000:
                        risk_pct = 0.15 if normalized >= 95 else (0.12 if vix_val >= 25 else (0.10 if vix_val >= 20 else 0.08))
                    else:
                        risk_pct = 0.10 if normalized >= 95 else (0.08 if vix_val >= 25 else (0.06 if vix_val >= 20 else 0.05))

                    max_risk = cash * risk_pct
                    max_position = cash * 0.35          # never >35% of cash in one trade
                    cost_per_contract = net_debit * 100
                    contracts = int(min(max_risk, max_position) / cost_per_contract)
                    if contracts == 0 and cost_per_contract <= cash:
                        contracts = 1

                    if contracts > 0 and cost_per_contract * contracts <= cash:
                        cost = round(cost_per_contract * contracts, 2)
                        trade_id = f"{today_str}-{now.strftime('%H%M%S')}-{direction_bias}"
                        new_pos = _ensure_trade_id({
                            "trade_id": trade_id, "date": today_str, "status": "OPEN", "action": "BUY",
                            "score": normalized, "grade": signal["grade"],
                            "direction": direction_bias, "K_buy": K_buy, "K_sell": K_sell,
                            "spread_width": spread_w,
                            "net_debit": round(net_debit, 2), "contracts": contracts, "cost": cost,
                            "entry_spy": round(spy_p, 2), "entry_time": now.strftime("%H:%M"),
                            "entry_ts": now.strftime("%Y-%m-%d %H:%M:%S"), "time": now.strftime("%H:%M"),
                            "current_val": round(net_debit, 2), "unrealized_pnl": 0.0, "unrealized_pnl_pct": 0.0
                        })
                        portfolio["positions"][today_str] = new_pos
                        portfolio["cash"] -= cost
                        _append_trade_event(portfolio, {
                            "event": "OPEN",
                            "trade_id": trade_id,
                            "date": today_str,
                            "time": now.strftime("%H:%M"),
                            "direction": direction_bias,
                            "grade": signal["grade"],
                            "score": normalized,
                            "K_buy": K_buy,
                            "K_sell": K_sell,
                            "spread_width": spread_w,
                            "contracts": contracts,
                            "entry_spy": round(spy_p, 2),
                            "net_debit": round(net_debit, 2),
                            "cost": cost,
                        })

            # 3. Manage open position — mark, exit when signal invalid / SL / TP / EOD
            open_pos = portfolio.get("positions", {}).get(today_str)
            if open_pos:
                iv = vix_val / 100.0
                opt = "call" if open_pos["direction"] == "CALL" else "put"
                hours_rem = max(0.1, 16.0 - (now.hour + now.minute/60.0))
                T_rem = hours_rem / (252 * 6.5)
                
                # Try to use real option chain quotes for exit valuation
                real_exit_ok = False
                current_val = 0.0
                chain_exit = _get_0dte_option_chain(spy_p, vix_val)
                if chain_exit and open_pos["K_buy"] in chain_exit and open_pos["K_sell"] in chain_exit:
                    buy_snap = chain_exit[open_pos["K_buy"]][opt]
                    sell_snap = chain_exit[open_pos["K_sell"]][opt]
                    if buy_snap and sell_snap and buy_snap.get("bid") is not None and sell_snap.get("ask") is not None:
                        # Sell long at Bid, Buy short at Ask (closing credit revenue)
                        current_val = buy_snap["bid"] - sell_snap["ask"]
                        real_exit_ok = True
                
                if not real_exit_ok:
                    slip_exit = dynamic_slippage(vix_val)
                    lp = bs_price(spy_p, open_pos["K_buy"], T_rem, 0.05, iv, opt)
                    sp = bs_price(spy_p, open_pos["K_sell"], T_rem, 0.05, iv, opt)
                    current_val = max(0, lp - sp) * 0.97 - slip_exit * 2

                current_val = max(0, current_val)
                mark_value = round(max(0, current_val), 2)
                mark_revenue = round(mark_value * 100 * open_pos["contracts"], 2)
                open_pos["current_val"] = mark_value
                open_pos["mark_spy"] = round(spy_p, 2)
                open_pos["mark_time"] = now.strftime("%H:%M")
                open_pos["unrealized_pnl"] = round(mark_revenue - open_pos["cost"], 2)
                open_pos["unrealized_pnl_pct"] = round((open_pos["unrealized_pnl"] / open_pos["cost"]) * 100, 1) if open_pos.get("cost", 0) > 0 else 0.0
                
                tp_price = open_pos["net_debit"] * 2.0
                exit_val = mark_value
                exit_type = None

                invalid_reason = _position_invalid_reason(open_pos, grade, direction_bias, score_result)
                if invalid_reason:
                    exit_type = invalid_reason
                elif open_pos["unrealized_pnl_pct"] <= -50.0:
                    exit_type = "SL"
                elif current_val >= tp_price:
                    exit_val = tp_price
                    exit_type = "TP"
                elif not is_regular or now.hour >= 16:
                    exit_type = "EOD"

                if exit_type:
                    _record_position_close(portfolio, open_pos, today_str, now, spy_p, exit_val, exit_type)
                else:
                    portfolio["positions"][today_str] = open_pos

            portfolio["recent_trades"] = _build_recent_trades(portfolio)
            portfolio["pdt"] = _pdt_status(portfolio, now)
            save_ok = save_portfolio(portfolio)
            portfolio["persist_ok"] = save_ok
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

            strike_rec = _calculate_strike_recommendation(
                spy_price=spy_p, direction_bias=direction_bias, signal_grade=signal["grade"],
                vix_price=vix_p, vwap=vwap, normalized_score=normalized,
                portfolio_cash=portfolio.get("cash", STARTING_BALANCE), now_et=now,
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

            final = {
                "last_updated": ts, "fetch_status": "SUCCESS", "latency_ms": latency,
                "timing_ms": fetch_timing,
                "data_source": "ALPACA + FlashAlpha" if flashalpha_spy else "ALPACA",
                "flashalpha": flashalpha_spy,
                "session": "REGULAR" if is_regular else "CLOSED",
                "trading_day": trading_day,
                "calendar_day": calendar_day,
                "trading_start_date": TRADING_START_DATE,
                "briefing": f"{score_result['layers']['time_window']['emoji']} [{score_result['layers']['time_window']['window']}] Regime: {score_result['layers']['regime']['regime']} | Bias: {direction_bias} | Score: {normalized}/100",
                "total_score": normalized, "max_score": active_max, "raw_score": raw_total,
                "signal": signal, "direction_bias": direction_bias,
                "layers": score_result["layers"],
                "strike_recommendation": strike_rec,
                "verdict": signal["label"], "confidence": normalized, "reason": signal["action"],
                "rules": rules, "alert_mode": "ON SIGNAL CHANGE",
                "indices": indices_out, "mag7": mag7_out,
                "gme_data": gme_data, "special_watch": gme_data,
                "paper_trading": portfolio,
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
