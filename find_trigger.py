import json, os, time, sys, requests
from datetime import datetime, time as dtime
import pandas as pd
import pytz
import yfinance as yf

# 건님의 원본 라이브러리 설정 복구
try:
    import pandas_market_calendars as mcal
    HAS_MCAL = True
except ImportError:
    HAS_MCAL = False

NY = pytz.timezone("America/New_York")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "").strip()
TWELVE_KEY = os.getenv("TWELVE_DATA_KEY", "").strip()

INDICES = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Russell 2000"}
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
SPECIAL_WATCH = ["GME"]

# === 안전한 데이터 변환 유틸리티 (원본 복구) ===
def now_ny(): return datetime.now(NY)

def safe_float(value, default=None):
    try: return float(value) if not pd.isna(float(value)) else default
    except: return default

def pct_change(p, prev): 
    return ((p / prev) - 1.0) * 100.0 if p and prev else None

# === 장 운영 시간 계산 (원본 복구) ===
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

# === 룰 및 신뢰도 생성 ===
def build_rules(now, session_name, spy_price, vix_price, vwap, range_value, vol_ratio, sector_sync):
    return {
        "vix": {"val": f"{vix_price:.2f}" if vix_price is not None else "--", "ok": vix_price is not None and vix_price >= 14},
        "range": {"val": f"${range_value:.2f}" if range_value is not None else "--", "ok": range_value is not None and range_value >= 3.0},
        "window": {"val": now.strftime("%H:%M"), "ok": session_name == "REGULAR"},
        "vwap": {"val": f"${(spy_price - vwap):+.2f}" if spy_price and vwap else "--", "ok": spy_price is not None and vwap is not None and spy_price > vwap},
        "vol": {"val": f"{vol_ratio:.2f}x" if vol_ratio is not None else "--", "ok": vol_ratio is not None and vol_ratio >= 1.5},
        "sector": {"val": "SYNC" if sector_sync else "DIFF", "ok": bool(sector_sync)}
    }

def summarize_rules(rules):
    failed = [name for name, r in rules.items() if not r["ok"]]
    passed = len(rules) - len(failed)
    confidence = int((passed / max(len(rules), 1)) * 100)
    if confidence == 100: return "STRONG GO", confidence, "All conditions satisfied", failed
    elif confidence >= 60: return "WAITING", confidence, "Waiting: " + ", ".join(failed), failed
    else: return "STOP", confidence, "Weak conditions: " + ", ".join(failed), failed

# === 상태 저장 및 웹훅 ===
def load_state():
    try:
        with open("state.json", "r") as f: return json.load(f)
    except: return {}

def save_state(state):
    with open("state.json", "w") as f: json.dump(state, f, indent=2)

def send_alert_if_state_changed(prev_state, curr):
    prev_v = (prev_state or {}).get("last_verdict")
    curr_v = curr["verdict"]
    if prev_v is not None and prev_v != curr_v:
        if WEBHOOK_URL:
            try: requests.post(WEBHOOK_URL, json={"event": "STATE_CHANGE", "previous": prev_v, "current": curr_v, "reason": curr.get("reason","")}, timeout=5)
            except: pass
        return True
    return False

# === PAPER TRADING ===
PAPER_PORTFOLIO_FILE = "paper_portfolio.json"
STARTING_BALANCE = 2000.0

def load_portfolio():
    if os.path.exists(PAPER_PORTFOLIO_FILE):
        try:
            with open(PAPER_PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    # Initialize new portfolio
    return {
        "cash": STARTING_BALANCE,
        "positions": {},
        "history": [],
        "initial_balance": STARTING_BALANCE
    }

def save_portfolio(pf):
    with open(PAPER_PORTFOLIO_FILE, "w") as f:
        json.dump(pf, f, indent=2)

def execute_paper_trade(pf, verdict, ts, prices):
    target_symbol = "SPY" # Default to SPY for VWAP strat
    if target_symbol not in prices: return pf
    
    current_price = prices[target_symbol]
    if current_price is None or current_price <= 0: return pf

    if verdict == "STRONG GO":
        # Buy condition
        if pf["cash"] >= current_price:
            # We buy maximum whole shares
            shares_to_buy = int(pf["cash"] // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                pf["cash"] -= cost
                
                if target_symbol not in pf["positions"]:
                    pf["positions"][target_symbol] = {"shares": 0, "avg_price": 0.0}
                
                # Update avg price
                old_shares = pf["positions"][target_symbol]["shares"]
                old_cost = old_shares * pf["positions"][target_symbol]["avg_price"]
                new_shares = old_shares + shares_to_buy
                new_avg = (old_cost + cost) / new_shares
                
                pf["positions"][target_symbol] = {"shares": new_shares, "avg_price": new_avg}
                pf["history"].append({"time": ts, "action": "BUY", "symbol": target_symbol, "shares": shares_to_buy, "price": current_price, "cost": cost})
                print(f"PAPER TRADE: BOUGHT {shares_to_buy} {target_symbol} @ ${current_price:.2f}")
    
    elif verdict in ["STOP", "WAITING"]:
        # Sell condition
        if target_symbol in pf["positions"] and pf["positions"][target_symbol]["shares"] > 0:
            shares_to_sell = pf["positions"][target_symbol]["shares"]
            revenue = shares_to_sell * current_price
            pf["cash"] += revenue
            pf["positions"][target_symbol]["shares"] = 0 # Sell all
            pf["history"].append({"time": ts, "action": "SELL", "symbol": target_symbol, "shares": shares_to_sell, "price": current_price, "revenue": revenue})
            print(f"PAPER TRADE: SOLD {shares_to_sell} {target_symbol} @ ${current_price:.2f}")

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
            if sym == "^VIX": continue
            try:
                r = requests.get(f"https://finnhub.io/api/v1/quote?symbol={sym}&token={FINNHUB_KEY}", timeout=5).json()
                prices[sym] = float(r.get('c', 0))
                pcts[sym] = float(r.get('dp', 0))
            except: pass
    return prices, pcts

def main():
    start = time.perf_counter()
    now = now_ny()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    prev_state = load_state()
    portfolio = load_portfolio()
    
    try:
        session_name, is_closed = get_nyse_session(now)
        
        # 1. 시세 데이터 병합 (Finnhub + YFinance Fallback)
        prices, pcts = get_api_data()
        
        tickers = yf.Tickers(" ".join([s for s in list(INDICES.keys()) + MAG7 + SPECIAL_WATCH if s not in prices or s == "^VIX"]))
        for sym in tickers.tickers:
            t = tickers.tickers[sym]
            p = safe_float(getattr(t.fast_info, 'last_price', None))
            prev = safe_float(getattr(t.fast_info, 'previous_close', None))
            prices[sym] = p
            pcts[sym] = pct_change(p, prev)
            
        spy_price = prices.get("SPY")
        vix_price = prices.get("^VIX")

        # 2. VWAP 계산 (원본 로직)
        vwap, range_val, vol_ratio = None, None, None
        spy_h = tickers.tickers["SPY"].history(period="1d", interval="5m", prepost=True) if "SPY" in tickers.tickers else yf.Ticker("SPY").history(period="1d", interval="5m", prepost=True)
        if not spy_h.empty:
            tp = (spy_h["High"] + spy_h["Low"] + spy_h["Close"]) / 3.0
            valid = spy_h["Volume"].cumsum().replace(0, pd.NA)
            vwap_s = (spy_h["Volume"] * tp).cumsum() / valid
            if not vwap_s.empty: vwap = safe_float(vwap_s.iloc[-1])
            range_val = safe_float(spy_h["High"].max() - spy_h["Low"].min())
            vol_sma = spy_h["Volume"].rolling(window=20).mean()
            if not vol_sma.empty and pd.notna(vol_sma.iloc[-1]) and vol_sma.iloc[-1] > 0: 
                vol_ratio = safe_float(spy_h["Volume"].iloc[-1] / vol_sma.iloc[-1])

        # 3. 브리핑 문자열 안전 생성 (VIX None 충돌 방지)
        sector_sync = (pcts.get("SPY", 0) >= 0 and pcts.get("QQQ", 0) >= 0 and pcts.get("IWM", 0) >= 0) or (pcts.get("SPY", 0) <= 0 and pcts.get("QQQ", 0) <= 0 and pcts.get("IWM", 0) <= 0)
        vix_str = f"{vix_price:.2f}" if vix_price is not None else "조회불가"

        if is_closed:
            briefing = f"🌙 [{session_name}] 시장이 닫혀 있습니다. 데이터를 정비 중입니다."
        else:
            if now.time() < dtime(9, 30): briefing = f"⚠️ [PRE-MARKET] 개장 전입니다. VIX: {vix_str}. GME 프리마켓 주시 요망."
            else: briefing = "🔥 [REGULAR MARKET] 본장 진행 중. 섹터 동기화 및 VWAP 집중 감시."

        rules = build_rules(now, session_name, spy_price, vix_price, vwap, range_val, vol_ratio, sector_sync)
        verdict, confidence, reason, _ = summarize_rules(rules)

        if is_closed:
            verdict, reason = "STOP", f"Market Closed ({session_name})"
            
        portfolio = execute_paper_trade(portfolio, verdict, ts, prices)
        save_portfolio(portfolio)

        latency = round((time.perf_counter() - start) * 1000, 1)

        def build_snap(syms): return {s: {"price": prices.get(s), "pct": pcts.get(s)} for s in syms if s in prices}

        result = {
            "last_updated": ts, "fetch_status": "SUCCESS", "session": session_name,
            "verdict": verdict, "confidence": confidence, "reason": reason,
            "latency_ms": latency, "alert_mode": "ON STATE CHANGE", "briefing": briefing,
            "rules": rules,
            "indices": build_snap([s for s in INDICES if s != "^VIX"]),
            "mag7": build_snap(MAG7),
            "special_watch": build_snap(SPECIAL_WATCH),
            "paper_trading": portfolio
        }

        result["alert_fired"] = send_alert_if_state_changed(prev_state, result)

        with open("data.json", "w", encoding="utf-8") as f: json.dump(result, f, indent=2)
        save_state({"last_verdict": verdict, "last_confidence": confidence, "last_updated": ts, "last_session": session_name, "last_latency_ms": latency})
        print(f"SYNC SUCCESS: {ts}")

    except Exception as e:
        # 에러 발생 시 강제 종료(sys.exit)하지 않고 대시보드에 에러 상태를 전달합니다!
        end = time.perf_counter()
        latency_ms = round((end - start) * 1000, 1)

        error_result = {
            "last_updated": ts,
            "fetch_status": "ERROR",
            "verdict": "STOP",
            "confidence": 0,
            "reason": f"Fetch error: {str(e)}",
            "session": get_nyse_session(now)[0],
            "latency_ms": latency_ms,
            "alert_mode": "ON STATE CHANGE",
            "briefing": f"❌ 시스템 에러 발생. 로그 확인 요망.",
            "rules": {}, "indices": {}, "mag7": {}, "special_watch": {}
        }

        with open("data.json", "w", encoding="utf-8") as f:
            json.dump(error_result, f, indent=2)

        print(f"ERROR: {e}")

if __name__ == "__main__": main()
