"""
SPY 0DTE Signal Machine — Main Trigger Engine
═══════════════════════════════════════════════
7-Layer Scoring System for SPY 0DTE Options
Replaces binary Pass/Fail with layered score (0–110+)
Signal fires at normalized score ≥ 75

Layers:
  1. Macro Gate       [Future — API needed]
  2. Market Regime    [ACTIVE — VIX, VIX3M, ADX]
  3. Options Flow     [Future — Unusual Whales]
  4. Correlation      [ACTIVE — QQQ, IWM, DIA sync]
  5. Time Window      [ACTIVE — 0DTE optimal periods]
  6. Technical Trigger [ACTIVE — VWAP, Volume, RSI]
  7. Risk Management  [ACTIVE — 3-strike, drawdown]
"""

import json, os, time, sys, requests
from datetime import datetime, time as dtime
import pandas as pd
import pytz
import yfinance as yf

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

try:
    import pandas_market_calendars as mcal
    HAS_MCAL = True
except ImportError:
    HAS_MCAL = False

from engines.score_engine import run_score_engine
from engines.risk_manager import calculate_position_size

NY = pytz.timezone("America/New_York")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "").strip()
TWELVE_KEY = os.getenv("TWELVE_DATA_KEY", "").strip()

INDICES = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Russell 2000"}
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
SPECIAL_WATCH = ["GME"]


# === 유틸리티 ===
def now_ny(): return datetime.now(NY)

def safe_float(value, default=None):
    try: return float(value) if not pd.isna(float(value)) else default
    except: return default

def pct_change(p, prev):
    return ((p / prev) - 1.0) * 100.0 if p and prev else None


# === 장 운영 시간 ===
def get_nyse_session(current_dt):
    weekday = current_dt.weekday()
    if weekday >= 5: return "WEEKEND", True
    if HAS_MCAL:
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=current_dt.date(), end_date=current_dt.date())
        if schedule.empty: return "HOLIDAY", True
        open_ts = schedule.iloc[0]["market_open"].tz_convert(NY)
        close_ts = schedule.iloc[0]["market_close"].tz_convert(NY)
        if current_dt < open_ts: return "PRE-MARKET", False
        if open_ts <= current_dt <= close_ts: return "REGULAR", False
        return "AFTER-HOURS", False
    c_time = current_dt.time()
    if c_time < dtime(9, 30): return "PRE-MARKET", False
    if dtime(9, 30) <= c_time <= dtime(16, 0): return "REGULAR", False
    return "AFTER-HOURS", False


# === 상태 저장 & 웹훅 ===
def load_state():
    try:
        with open("state.json", "r") as f: return json.load(f)
    except: return {}

def save_state(state):
    with open("state.json", "w") as f: json.dump(state, f, indent=2)

def send_alert_if_state_changed(prev_state, curr_signal):
    prev_grade = (prev_state or {}).get("last_grade")
    curr_grade = curr_signal.get("grade")
    if prev_grade is not None and prev_grade != curr_grade:
        if WEBHOOK_URL:
            try: requests.post(WEBHOOK_URL, json={
                "event": "SIGNAL_CHANGE",
                "previous": prev_grade,
                "current": curr_grade,
                "label": curr_signal.get("label", ""),
                "score": curr_signal.get("total_score", 0),
            }, timeout=5)
            except: pass
        return True
    return False


# === Paper Trading ===
PAPER_PORTFOLIO_FILE = "paper_portfolio.json"
STARTING_BALANCE = 2000.0

def load_portfolio():
    if os.path.exists(PAPER_PORTFOLIO_FILE):
        try:
            with open(PAPER_PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "cash": STARTING_BALANCE,
        "positions": {},
        "history": [],
        "initial_balance": STARTING_BALANCE,
        "current_value": STARTING_BALANCE,
        "total_return_pct": 0.0,
    }

def save_portfolio(pf):
    with open(PAPER_PORTFOLIO_FILE, "w") as f:
        json.dump(pf, f, indent=2)


def execute_paper_trade(pf, signal_grade, total_score, ts, prices):
    """
    Execute paper trade based on signal grade.
    STRONG → full position buy
    MODERATE → half position buy
    WEAK/NONE/LOCKED → sell existing positions
    """
    target_symbol = "SPY"
    if target_symbol not in prices: return pf
    current_price = prices[target_symbol]
    if current_price is None or current_price <= 0: return pf

    if signal_grade in ("STRONG", "MODERATE"):
        # Buy condition — position size by grade
        if pf["cash"] >= current_price:
            sizing = calculate_position_size(pf, signal_grade, current_price)
            shares_to_buy = sizing["max_shares"]

            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                pf["cash"] -= cost

                if target_symbol not in pf["positions"]:
                    pf["positions"][target_symbol] = {"shares": 0, "avg_price": 0.0}

                old_shares = pf["positions"][target_symbol]["shares"]
                old_cost = old_shares * pf["positions"][target_symbol]["avg_price"]
                new_shares = old_shares + shares_to_buy
                new_avg = (old_cost + cost) / new_shares

                pf["positions"][target_symbol] = {"shares": new_shares, "avg_price": new_avg}
                pf["history"].append({
                    "time": ts, "action": "BUY", "symbol": target_symbol,
                    "shares": shares_to_buy, "price": current_price, "cost": cost,
                    "signal_grade": signal_grade, "score": total_score,
                })
                print(f"PAPER TRADE: BOUGHT {shares_to_buy} {target_symbol} @ ${current_price:.2f} [{signal_grade}]")

    elif signal_grade in ("NONE", "LOCKED"):
        # Sell condition
        if target_symbol in pf["positions"] and pf["positions"][target_symbol]["shares"] > 0:
            shares_to_sell = pf["positions"][target_symbol]["shares"]
            avg_price = pf["positions"][target_symbol]["avg_price"]
            revenue = shares_to_sell * current_price
            pnl = revenue - (shares_to_sell * avg_price)
            pf["cash"] += revenue
            pf["positions"][target_symbol]["shares"] = 0
            pf["history"].append({
                "time": ts, "action": "SELL", "symbol": target_symbol,
                "shares": shares_to_sell, "price": current_price,
                "revenue": revenue, "pnl": pnl,
                "signal_grade": signal_grade, "score": total_score,
            })
            print(f"PAPER TRADE: SOLD {shares_to_sell} {target_symbol} @ ${current_price:.2f} (PnL: ${pnl:+.2f})")

    # Calculate current value
    total_value = pf["cash"]
    for sym, pos in pf["positions"].items():
        if pos["shares"] > 0 and sym in prices and prices[sym]:
            total_value += pos["shares"] * prices[sym]
    pf["current_value"] = total_value
    pf["total_return_pct"] = ((total_value / pf["initial_balance"]) - 1.0) * 100.0

    return pf


# === API 데이터 수집기 ===
def get_api_data():
    prices, pcts = {}, {}
    symbols = list(INDICES.keys()) + MAG7 + SPECIAL_WATCH
    if FINNHUB_KEY:
        for sym in symbols:
            if sym.startswith("^"): continue
            try:
                r = requests.get(f"https://finnhub.io/api/v1/quote?symbol={sym}&token={FINNHUB_KEY}", timeout=5).json()
                prices[sym] = float(r.get('c', 0))
                pcts[sym] = float(r.get('dp', 0))
            except: pass
    return prices, pcts


# === MAIN ===
def main():
    start = time.perf_counter()
    now = now_ny()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    prev_state = load_state()
    portfolio = load_portfolio()

    try:
        session_name, is_closed = get_nyse_session(now)

        # 1. 시세 데이터 병합 (Finnhub + YFinance fallback)
        prices, pcts = get_api_data()

        # YFinance for missing data + VIX + VIX3M
        yf_symbols = [s for s in list(INDICES.keys()) + MAG7 + SPECIAL_WATCH + ["^VIX", "^VIX3M"]
                       if s not in prices]
        if yf_symbols:
            tickers = yf.Tickers(" ".join(yf_symbols))
            for sym in tickers.tickers:
                t = tickers.tickers[sym]
                p = safe_float(getattr(t.fast_info, 'last_price', None))
                prev = safe_float(getattr(t.fast_info, 'previous_close', None))
                prices[sym] = p
                pcts[sym] = pct_change(p, prev)

        spy_price = prices.get("SPY")
        vix_price = prices.get("^VIX")
        vix3m_price = prices.get("^VIX3M")

        # Get SPY previous close
        spy_prev_close = None
        try:
            spy_ticker = yf.Ticker("SPY")
            spy_prev_close = safe_float(getattr(spy_ticker.fast_info, 'previous_close', None))
        except:
            pass

        # 2. VWAP & Technical Data
        vwap, range_val, vol_ratio = None, None, None
        spy_h = None
        try:
            spy_h = yf.Ticker("SPY").history(period="1d", interval="5m", prepost=True)
            if spy_h is not None and not spy_h.empty:
                tp = (spy_h["High"] + spy_h["Low"] + spy_h["Close"]) / 3.0
                valid = spy_h["Volume"].cumsum().replace(0, pd.NA)
                vwap_s = (spy_h["Volume"] * tp).cumsum() / valid
                if not vwap_s.empty: vwap = safe_float(vwap_s.iloc[-1])
                range_val = safe_float(spy_h["High"].max() - spy_h["Low"].min())
                vol_sma = spy_h["Volume"].rolling(window=20).mean()
                if not vol_sma.empty and pd.notna(vol_sma.iloc[-1]) and vol_sma.iloc[-1] > 0:
                    vol_ratio = safe_float(spy_h["Volume"].iloc[-1] / vol_sma.iloc[-1])
        except Exception as e:
            print(f"VWAP calculation warning: {e}")

        # 3. Run 7-Layer Score Engine
        engine_result = run_score_engine(
            now_et=now,
            spy_price=spy_price,
            vix_price=vix_price,
            vix3m_price=vix3m_price,
            prev_close=spy_prev_close,
            vwap=vwap,
            vol_ratio=vol_ratio,
            range_value=range_val,
            pcts=pcts,
            spy_history=spy_h,
            portfolio=portfolio,
            session_name=session_name,
        )

        signal = engine_result["signal"]
        total_score = engine_result["total_score"]

        # 4. Generate briefing
        vix_str = f"{vix_price:.2f}" if vix_price is not None else "N/A"
        if is_closed:
            briefing = f"🌙 [{session_name}] Market closed. System on standby."
        elif now.time() < dtime(9, 30):
            briefing = f"⚠️ [PRE-MARKET] Pre-market scan. VIX: {vix_str}"
        else:
            window_info = engine_result["layers"]["time_window"]
            regime_info = engine_result["layers"]["regime"]
            briefing = (
                f"{window_info['emoji']} [{window_info['window']}] "
                f"Regime: {regime_info['regime']} | "
                f"Bias: {engine_result['direction_bias']} | "
                f"Score: {total_score}/100"
            )

        # 5. Paper trading with score-based logic
        if is_closed:
            signal_grade = "NONE"
        else:
            signal_grade = signal["grade"]

        portfolio = execute_paper_trade(portfolio, signal_grade, total_score, ts, prices)
        save_portfolio(portfolio)

        # 6. Build output
        latency = round((time.perf_counter() - start) * 1000, 1)

        def build_snap(syms):
            return {s: {"price": prices.get(s), "pct": pcts.get(s)} for s in syms if s in prices}

        # Legacy rules for backward compatibility
        legacy_rules = {
            "vix": {"val": f"{vix_price:.2f}" if vix_price else "--", "ok": vix_price is not None and vix_price >= 14},
            "range": {"val": f"${range_val:.2f}" if range_val else "--", "ok": range_val is not None and range_val >= 3.0},
            "window": {"val": now.strftime("%H:%M"), "ok": session_name == "REGULAR"},
            "vwap": {"val": f"${(spy_price - vwap):+.2f}" if spy_price and vwap else "--", "ok": spy_price is not None and vwap is not None and spy_price > vwap},
            "vol": {"val": f"{vol_ratio:.2f}x" if vol_ratio else "--", "ok": vol_ratio is not None and vol_ratio >= 1.5},
            "sector": {"val": "SYNC" if engine_result["layers"]["correlation"]["sector_sync"] else "DIFF", "ok": engine_result["layers"]["correlation"]["sector_sync"]},
        }

        result = {
            "last_updated": ts,
            "fetch_status": "SUCCESS",
            "session": session_name,
            "latency_ms": latency,
            "briefing": briefing,

            # === NEW: Score Engine Output ===
            "total_score": total_score,
            "max_score": engine_result["max_score"],
            "raw_score": engine_result["raw_score"],
            "signal": signal,
            "direction_bias": engine_result["direction_bias"],
            "layers": {
                "regime": engine_result["layers"]["regime"],
                "options_flow": engine_result["layers"]["options_flow"],
                "correlation": engine_result["layers"]["correlation"],
                "time_window": engine_result["layers"]["time_window"],
                "technical": engine_result["layers"]["technical"],
                "risk": engine_result["layers"]["risk"],
            },

            # === Legacy fields (backward compat) ===
            "verdict": signal["label"],
            "confidence": total_score,
            "reason": signal["action"],
            "rules": legacy_rules,
            "alert_mode": "ON SIGNAL CHANGE",

            # === Market data ===
            "indices": build_snap([s for s in INDICES if not s.startswith("^")]),
            "mag7": build_snap(MAG7),
            "special_watch": build_snap(SPECIAL_WATCH),
            "paper_trading": portfolio,
        }

        result["alert_fired"] = send_alert_if_state_changed(prev_state, {
            "grade": signal["grade"],
            "total_score": total_score,
            "label": signal["label"],
        })

        with open("data.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        save_state({
            "last_grade": signal["grade"],
            "last_score": total_score,
            "last_label": signal["label"],
            "last_updated": ts,
            "last_session": session_name,
            "last_latency_ms": latency,
        })

        print(f"✅ SYNC: {ts} | Score: {total_score}/100 | {signal['emoji']} {signal['label']} | {engine_result['direction_bias']}")

    except Exception as e:
        end = time.perf_counter()
        latency_ms = round((end - start) * 1000, 1)

        error_result = {
            "last_updated": ts,
            "fetch_status": "ERROR",
            "verdict": "SYSTEM ERROR",
            "confidence": 0,
            "total_score": 0,
            "signal": {"grade": "NONE", "label": "SYSTEM ERROR", "emoji": "❌",
                       "action": f"Error: {str(e)}", "color": "#f07178"},
            "reason": f"Fetch error: {str(e)}",
            "session": get_nyse_session(now)[0],
            "latency_ms": latency_ms,
            "briefing": f"❌ System error. Check logs.",
            "rules": {}, "layers": {},
            "indices": {}, "mag7": {}, "special_watch": {},
        }

        with open("data.json", "w", encoding="utf-8") as f:
            json.dump(error_result, f, indent=2)

        print(f"❌ ERROR: {e}")

if __name__ == "__main__": main()
