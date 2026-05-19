"""
Vercel Serverless API — /api/data
SPY 0DTE Signal Machine — 7-Layer Score Engine
Hybrid: Alpaca (stocks) + yfinance (VIX fallback)
"""
import math, json, os, time, traceback
from http.server import BaseHTTPRequestHandler
from datetime import datetime, timedelta
import pytz, requests

NY = pytz.timezone("America/New_York")
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from engines.score_engine import run_score_engine
STARTING_BALANCE = 2000.0

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bs_price(S, K, T, r, sigma, opt="call"):
    if T <= 0: return max(S - K, 0) if opt == "call" else max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_HEADERS = {
    "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
    "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", ""),
}


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
                     params={"symbols": ",".join(symbols), "feed": "iex"}, timeout=10)
    r.raise_for_status()
    return r.json()


def _alpaca_bars(symbol, timeframe="5Min"):
    """Fetch intraday bars for technical analysis."""
    now = datetime.now(NY)
    start = now.replace(hour=4, minute=0, second=0, microsecond=0)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    r = requests.get(url, headers=ALPACA_HEADERS, params={
        "timeframe": timeframe, "start": start.isoformat(),
        "limit": 1000, "adjustment": "raw", "feed": "iex",
    }, timeout=10)
    r.raise_for_status()
    bars = r.json().get("bars", [])
    if not bars: return pd.DataFrame()
    df = pd.DataFrame(bars)
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                             "c": "Close", "v": "Volume", "t": "Timestamp"})
    return df


def _vix_fallback():
    """Fetch VIX/VIX3M from yfinance (Alpaca doesn't provide CBOE indices)."""
    import yfinance as yf
    vix_p, vix3m_p = 18.0, None
    try:
        v = yf.Ticker("^VIX").fast_info.last_price
        if v is not None and not pd.isna(v): vix_p = float(v)
    except: pass
    try:
        v = yf.Ticker("^VIX3M").fast_info.last_price
        if v is not None and not pd.isna(v): vix3m_p = float(v)
    except: pass
    return vix_p, vix3m_p


def _snap_price(snap, key="latestTrade"):
    """Extract price from an Alpaca snapshot safely."""
    try: return float(snap[key]["p"])
    except: return 0.0

def _snap_prev_close(snap):
    try: return float(snap["prevDailyBar"]["c"])
    except: return 0.0

def _pct(price, prev):
    return ((price / prev) - 1) * 100 if prev > 0 else 0.0


# ── Scoring Engine (imported) ──────────────────────────────────────

def _signal_grade(score):
    if score >= 90: return {"grade": "STRONG", "label": "STRONG SIGNAL", "emoji": "🟢", "action": "Full position", "color": "#3dd68c"}
    elif score >= 75: return {"grade": "MODERATE", "label": "MODERATE SIGNAL", "emoji": "🟡", "action": "Half position", "color": "#f5c451"}
    elif score >= 60: return {"grade": "WEAK", "label": "STANDBY", "emoji": "🟠", "action": "Monitor only", "color": "#f5a623"}
    else: return {"grade": "NONE", "label": "NO SIGNAL", "emoji": "🔴", "action": "No entry", "color": "#f07178"}

def _calculate_strike_recommendation(spy_price, direction_bias, signal_grade,
                                      vix_price, vwap, normalized_score,
                                      portfolio_cash, now_et):
    if spy_price is None or direction_bias == "NEUTRAL" or signal_grade in ("NONE",):
        return {"active": False, "reason": "No actionable signal"}
    atm_strike = round(spy_price)
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
    # Estimate premium from VIX
    vix_val = vix_price if vix_price else 18.0
    time_factor = math.sqrt(1.0 / 365.0)
    atm_est = spy_price * (vix_val / 100.0) * time_factor * 0.5
    otm_discount = max(0.3, 1.0 - (otm_offset * 0.25))
    mid_premium = max(round(atm_est * otm_discount, 2), 0.05)
    est_premium_low = max(round(mid_premium * 0.85, 2), 0.01)
    est_premium_high = max(round(mid_premium * 1.15, 2), 0.10)
    data_source = "ESTIMATED"
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
    return {
        "active": True, "direction": type_label, "atm_strike": atm_strike,
        "recommended_strike": recommended_strike, "otm_offset": otm_offset,
        "contract_label": contract_label, "contract_type": contract_type,
        "strikes": [
            {"label": "ATM", "strike": atm_strike, "recommended": otm_offset == 0},
            {"label": "OTM-1", "strike": otm_1, "recommended": otm_offset == 1},
            {"label": "OTM-2", "strike": otm_2, "recommended": otm_offset == 2},
        ],
        "est_premium_low": est_premium_low, "est_premium_high": est_premium_high,
        "mid_premium": mid_premium, "data_source": data_source,
        "real_bid": None, "real_ask": None, "real_last": None,
        "target_pct": target_pct, "stop_pct": stop_pct,
        "target_price": target_price, "stop_price": stop_price,
        "risk_reward": f"1:{rr_ratio}", "max_contracts": max_contracts,
        "cost_per_contract": cost_per_contract, "max_risk_dollars": max_risk_dollars,
        "max_risk_pct": max_risk_pct, "reasoning": strike_reasoning,
    }

RESTFUL_KV_URL = os.getenv(
    "PORTFOLIO_REST_URL",
    "https://api.restful-api.dev/objects/ff8081819d82fab6019e405b84415410",
)
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
    base["recent_trades"] = base.get("recent_trades") or _build_recent_trades(base)
    return base

def _storage_backend():
    if os.getenv("KV_REST_API_URL") and os.getenv("KV_REST_API_TOKEN"):
        return "upstash"
    return "restful"


def _fetch_raw_portfolio(retries=3):
    """Load portfolio JSON from persistent storage with retries."""
    for attempt in range(retries):
        try:
            if _storage_backend() == "upstash":
                base = os.getenv("KV_REST_API_URL", "").rstrip("/")
                token = os.getenv("KV_REST_API_TOKEN", "")
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
                            return data
            else:
                r = requests.get(RESTFUL_KV_URL, timeout=6)
                if r.status_code == 200:
                    data = r.json().get("data", {})
                    if isinstance(data, dict) and "cash" in data:
                        data["_storage"] = "restful"
                        return data
        except Exception:
            pass
        time.sleep(0.15 * (attempt + 1))
    return None


def _write_raw_portfolio(pf):
    payload_pf = {k: v for k, v in pf.items() if not str(k).startswith("_")}
    body = json.loads(json.dumps({"name": PORTFOLIO_STORAGE_KEY, "data": payload_pf}, cls=SafeEncoder))

    if _storage_backend() == "upstash":
        base = os.getenv("KV_REST_API_URL", "").rstrip("/")
        token = os.getenv("KV_REST_API_TOKEN", "")
        raw = json.dumps(payload_pf, cls=SafeEncoder)
        r = requests.post(
            f"{base}/set/{PORTFOLIO_STORAGE_KEY}",
            data=raw,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=10,
        )
        r.raise_for_status()
        return True

    r = requests.put(RESTFUL_KV_URL, json=body, timeout=10)
    r.raise_for_status()
    return True


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


def _finalize_closed_record(record):
    """Freeze realized P&L — closed trades must never be mark-to-market again."""
    row = dict(record)
    row["status"] = "CLOSED"
    row["pnl_locked"] = True
    if row.get("pnl") is not None:
        row["realized_pnl"] = row["pnl"]
    for key in ("unrealized_pnl", "unrealized_pnl_pct", "mark_spy", "mark_time"):
        row.pop(key, None)
    return row


def _trade_already_closed(pf, trade_id):
    if not trade_id:
        return False
    for h in pf.get("history") or []:
        if h.get("trade_id") == trade_id and _is_closed_record(h):
            return True
    for e in pf.get("trade_log") or []:
        if e.get("trade_id") == trade_id and e.get("event") == "CLOSE":
            return True
    return False


def _merge_trade_records(items):
    """Merge closed trades by trade_id; first locked close wins (P&L never changes)."""
    merged = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        tid = item.get("trade_id") or f"{item.get('date')}-{item.get('entry_time')}-{item.get('direction')}-{item.get('K_buy')}"
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
        if _trade_already_closed(pf, record.get("trade_id")):
            return
        record = _finalize_closed_record(record)
    hist = pf.setdefault("history", [])
    hist.insert(0, record)
    pf["history"] = _cap_list(hist)


def _append_trade_event(pf, event):
    if event.get("event") == "CLOSE":
        if _trade_already_closed(pf, event.get("trade_id")):
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


def _entry_criteria_met(grade, direction_bias, score_result):
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
    if _trade_already_closed(portfolio, open_pos.get("trade_id")):
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

        # Never wipe stored trades when this request failed to load remote state.
        if not pf.get("history") and remote_pf.get("history"):
            pf["history"] = remote_pf["history"]
        else:
            pf["history"] = _cap_list([
                _finalize_closed_record(h) if _is_closed_record(h) else h for h in merged_history
            ])
        if not pf.get("trade_log") and remote_pf.get("trade_log"):
            pf["trade_log"] = remote_pf["trade_log"]
        else:
            pf["trade_log"] = _cap_list(merged_log)

        if remote_pf.get("initial_balance") and not pf.get("_seeded_balance"):
            pf.setdefault("initial_balance", remote_pf["initial_balance"])

        pf["recent_trades"] = _build_recent_trades(pf)
        pf["current_value"] = pf["cash"]
        for p in pf.get("positions", {}).values():
            pf["current_value"] += float(p.get("cost", 0) or 0) + float(p.get("unrealized_pnl", 0) or 0)
        pf["total_return_pct"] = round(((pf["current_value"] / pf["initial_balance"]) - 1) * 100, 2)
        pf["revision"] = int(remote_pf.get("revision", 0) or 0) + 1
        pf["last_saved"] = datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S")

        if not pf.get("_load_ok") and not pf.get("history") and not pf.get("trade_log"):
            pf["_save_skipped"] = "load_failed_empty_local"
            return False

        _write_raw_portfolio(pf)
        pf["_save_ok"] = True
        pf.pop("_save_error", None)
        return True
    except Exception as e:
        pf["_save_error"] = str(e)
        pf["_save_ok"] = False
        return False


# ── Handler ─────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        start_time = time.perf_counter()
        now = datetime.now(NY)
        ts = now.strftime("%Y-%m-%d %H:%M:%S")

        STOCK_SYMS = ["SPY", "QQQ", "DIA", "IWM"]
        MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
        GME_STOCK = ["GME"]
        ALL_STOCKS = STOCK_SYMS + MAG7 + GME_STOCK

        try:
            # ── FETCH: Alpaca snapshots (all stocks in one call) ──
            snaps = _alpaca_snapshots(ALL_STOCKS)

            spy_p = _snap_price(snaps.get("SPY", {}))
            spy_prev = _snap_prev_close(snaps.get("SPY", {})) or spy_p

            # ── FETCH: Alpaca 5-min bars for SPY ──
            spy_h = _alpaca_bars("SPY", "5Min")

            # ── FETCH: VIX from yfinance fallback ──
            vix_p, vix3m_p = _vix_fallback()

            # ── Percentage changes from snapshots ──
            pcts_data = {}
            for sym in STOCK_SYMS:
                s = snaps.get(sym, {})
                pcts_data[sym] = _pct(_snap_price(s), _snap_prev_close(s) or 1)

            # ── Compute VWAP / volume / range from bars ──
            vwap, vol_r, d_range = 0.0, 0.0, 0.0
            if not spy_h.empty:
                tp = (spy_h["High"] + spy_h["Low"] + spy_h["Close"]) / 3.0
                cum_vol = spy_h["Volume"].cumsum().replace(0, pd.NA)
                vwap_s = (spy_h["Volume"] * tp).cumsum() / cum_vol
                if not vwap_s.empty and pd.notna(vwap_s.iloc[-1]):
                    vwap = float(vwap_s.iloc[-1])
                vol_sma = spy_h["Volume"].rolling(window=20).mean()
                if not vol_sma.empty and pd.notna(vol_sma.iloc[-1]) and vol_sma.iloc[-1] > 0:
                    vol_r = float(spy_h["Volume"].iloc[-1] / vol_sma.iloc[-1])
                d_range = float(spy_h["High"].max() - spy_h["Low"].min())

            # ── CALL SCORE ENGINE (Modular 140-point system) ──
            t_min = now.hour * 60 + now.minute
            is_regular = 570 <= t_min <= 960
            portfolio = load_portfolio()
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

            t_min = now.hour * 60 + now.minute
            is_regular = 570 <= t_min <= 960
            if not is_regular:
                signal["label"] = "MARKET CLOSED"
                signal["action"] = "Market not in session"

            # ── PAPER TRADING EXECUTION ──
            today_str = now.strftime("%Y-%m-%d")
            vix_val = vix_p if vix_p > 0 else 18.0
            
            # 1. Clean up stale positions from previous days
            to_remove = []
            for date_key, pos in list(portfolio.get("positions", {}).items()):
                if date_key == today_str:
                    continue
                tid = pos.get("trade_id")
                if _trade_already_closed(portfolio, tid):
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
            if not open_pos and is_regular and _entry_criteria_met(grade, direction_bias, score_result):
                # Open Debit Spread
                iv = vix_val / 100.0
                opt = "call" if direction_bias == "CALL" else "put"
                K_buy = round(spy_p)
                K_sell = K_buy + 5 if opt == "call" else K_buy - 5
                
                T_entry = 5.5 / (252 * 6.5)
                lp = bs_price(spy_p, K_buy, T_entry, 0.05, iv, opt)
                sp = bs_price(spy_p, K_sell, T_entry, 0.05, iv, opt)
                net_debit = (lp - sp) * 1.03 + 0.04
                
                if net_debit > 0.05:
                    risk_pct = 0.10 if normalized >= 95 else (0.08 if vix_val >= 25 else (0.06 if vix_val >= 20 else 0.05))
                    max_risk = portfolio["cash"] * risk_pct
                    contracts = max(1, int(max_risk / (net_debit * 100)))
                    
                    if contracts > 0:
                        cost = round(net_debit * 100 * contracts, 2)
                        trade_id = f"{today_str}-{now.strftime('%H%M%S')}-{direction_bias}"
                        new_pos = {
                            "trade_id": trade_id, "date": today_str, "status": "OPEN", "action": "BUY",
                            "score": normalized, "grade": signal["grade"],
                            "direction": direction_bias, "K_buy": K_buy, "K_sell": K_sell,
                            "net_debit": round(net_debit, 2), "contracts": contracts, "cost": cost,
                            "entry_spy": round(spy_p, 2), "entry_time": now.strftime("%H:%M"),
                            "entry_ts": now.strftime("%Y-%m-%d %H:%M:%S"), "time": now.strftime("%H:%M"),
                            "current_val": round(net_debit, 2), "unrealized_pnl": 0.0, "unrealized_pnl_pct": 0.0
                        }
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
                
                lp = bs_price(spy_p, open_pos["K_buy"], T_rem, 0.05, iv, opt)
                sp = bs_price(spy_p, open_pos["K_sell"], T_rem, 0.05, iv, opt)
                current_val = max(0, lp - sp) * 0.97 - 0.04
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
            save_ok = save_portfolio(portfolio)
            portfolio["persist_ok"] = save_ok
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

            final = {
                "last_updated": ts, "fetch_status": "SUCCESS", "latency_ms": latency,
                "data_source": "ALPACA",
                "session": "REGULAR" if is_regular else "CLOSED",
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

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(json.dumps(final, cls=SafeEncoder).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e), "traceback": traceback.format_exc()}).encode('utf-8'))
