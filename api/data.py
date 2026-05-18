"""
Vercel Serverless API — /api/data
SPY 0DTE Signal Machine — 7-Layer Score Engine
"""
import math
from http.server import BaseHTTPRequestHandler
import json
import os
import time
from datetime import datetime
import pytz
import yfinance as yf
import pandas as pd

NY = pytz.timezone("America/New_York")
STARTING_BALANCE = 2000.0


# ── Inline Engine (Vercel can't import local modules easily) ────────
# Simplified versions of the engine layers for serverless deployment

def _score_vix(vix_price):
    if vix_price is None: return 0, "VIX N/A"
    if 14 <= vix_price <= 20: return 15, f"VIX {vix_price:.1f} — Normal"
    elif 20 < vix_price <= 30: return 0, f"VIX {vix_price:.1f} — Elevated"
    elif vix_price > 30: return -20, f"VIX {vix_price:.1f} — FEAR"
    else: return -5, f"VIX {vix_price:.1f} — Low"


def _score_vix_term(vix, vix3m):
    if vix is None or vix3m is None: return 0, 0, "Term N/A"
    spread = vix - vix3m
    if spread < 0: return 10, spread, f"Contango ({spread:+.2f})"
    else: return -15, spread, f"Backwardation ({spread:+.2f})"


def _calc_adx(hist, period=14):
    if hist is None or len(hist) < period + 1: return None
    try:
        h, l, c = hist["High"], hist["Low"], hist["Close"]
        plus_dm = h.diff().where((h.diff() > l.diff().abs()) & (h.diff() > 0), 0.0)
        minus_dm = l.diff().abs().where((l.diff().abs() > h.diff()) & (l.diff().abs() > 0), 0.0)
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        adx = dx.rolling(period).mean().dropna()
        return float(adx.iloc[-1]) if not adx.empty else None
    except: return None


def _calc_rsi(hist, period=14):
    if hist is None or len(hist) < period + 1: return None
    try:
        delta = hist["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        last = rsi.dropna()
        return float(last.iloc[-1]) if not last.empty else None
    except: return None


def _time_window(now_et):
    h, m = now_et.hour, now_et.minute
    t = h * 60 + m
    windows = [
        (570, 600, 0, "OPEN_CHAOS", "❌", "Gap chaos — avoid"),
        (600, 630, 5, "FORMING", "⚠️", "Direction forming"),
        (630, 690, 20, "PRIME", "🟢", "Best window"),
        (690, 720, 8, "TRANSITION", "⚠️", "Pre-lunch transition"),
        (720, 780, 0, "LUNCH_LULL", "❌", "Lunch lull — avoid"),
        (780, 840, 8, "REENTRY", "⚠️", "Afternoon re-entry"),
        (840, 885, 15, "GAMMA", "🟡", "Gamma window"),
        (885, 960, 0, "GAMMA_BOMB", "❌", "Gamma explosion"),
    ]
    for start, end, score, label, emoji, desc in windows:
        if start <= t < end:
            nxt = None
            if score < 15:
                for s2, e2, sc2, lb2, em2, _ in windows:
                    if sc2 >= 15 and s2 > t:
                        mins = s2 - t
                        nxt = {"window": lb2, "emoji": em2, "minutes_until": mins,
                               "countdown": f"{mins//60}h {mins%60}m", "starts_at": f"{s2//60:02d}:{s2%60:02d}"}
                        break
            return {"score": score, "max": 20, "window": label, "emoji": emoji,
                    "description": desc, "next_window": nxt, "is_blocked": score == 0 and t >= 570,
                    "current_time": now_et.strftime("%H:%M")}
    return {"score": 0, "max": 20, "window": "CLOSED", "emoji": "⏸️",
            "description": "Market closed", "next_window": None, "is_blocked": False,
            "current_time": now_et.strftime("%H:%M")}


def _signal_grade(score):
    if score >= 90: return {"grade": "STRONG", "label": "STRONG SIGNAL", "emoji": "🟢", "action": "Full position", "color": "#3dd68c"}
    elif score >= 75: return {"grade": "MODERATE", "label": "MODERATE SIGNAL", "emoji": "🟡", "action": "Half position", "color": "#f5c451"}
    elif score >= 60: return {"grade": "WEAK", "label": "STANDBY", "emoji": "🟠", "action": "Monitor only", "color": "#f5a623"}
    else: return {"grade": "NONE", "label": "NO SIGNAL", "emoji": "🔴", "action": "No entry", "color": "#f07178"}


def _calculate_strike_recommendation(spy_price, direction_bias, signal_grade,
                                      vix_price, vwap, normalized_score,
                                      portfolio_cash, now_et):
    """
    Calculate the recommended option strike to trade.
    Returns a dict with strike details, premium estimates, and risk params.
    """
    if spy_price is None or direction_bias == "NEUTRAL" or signal_grade in ("NONE",):
        return {"active": False, "reason": "No actionable signal"}

    atm_strike = round(spy_price)

    if signal_grade == "STRONG":
        otm_offset = 0
        strike_reasoning = "ATM — high conviction, all conditions met"
    elif signal_grade == "MODERATE":
        otm_offset = 1
        strike_reasoning = "OTM-1 — balanced cost/probability"
    else:
        otm_offset = 2
        strike_reasoning = "OTM-2 — monitor zone, low cost entry"

    if direction_bias == "CALL":
        recommended_strike = atm_strike + otm_offset
        otm_1 = atm_strike + 1
        otm_2 = atm_strike + 2
        contract_type = "C"
        type_label = "CALL"
    else:
        recommended_strike = atm_strike - otm_offset
        otm_1 = atm_strike - 1
        otm_2 = atm_strike - 2
        contract_type = "P"
        type_label = "PUT"

    contract_label = f"SPY ${recommended_strike}{contract_type} 0DTE"

    # Try real option chain from yfinance
    real_bid, real_ask, real_last, chain_success = None, None, None, False
    try:
        spy_ticker = yf.Ticker("SPY")
        exp_dates = spy_ticker.options
        today_str = now_et.strftime("%Y-%m-%d")
        target_exp = None
        for exp in exp_dates:
            if exp >= today_str:
                target_exp = exp
                break
        if target_exp:
            chain = spy_ticker.option_chain(target_exp)
            opt_df = chain.calls if direction_bias == "CALL" else chain.puts
            match = opt_df[opt_df["strike"] == float(recommended_strike)]
            if not match.empty:
                row = match.iloc[0]
                real_bid = float(row.get("bid", 0))
                real_ask = float(row.get("ask", 0))
                real_last = float(row.get("lastPrice", 0))
                chain_success = True
    except:
        pass

    if chain_success and real_bid and real_ask and real_bid > 0:
        est_premium_low = round(real_bid, 2)
        est_premium_high = round(real_ask, 2)
        mid_premium = round((real_bid + real_ask) / 2, 2)
        data_source = "LIVE"
    else:
        vix_val = vix_price if vix_price else 18.0
        time_factor = math.sqrt(1.0 / 365.0)
        atm_est = spy_price * (vix_val / 100.0) * time_factor * 0.5
        otm_discount = max(0.3, 1.0 - (otm_offset * 0.25))
        mid_premium = round(atm_est * otm_discount, 2)
        est_premium_low = round(mid_premium * 0.85, 2)
        est_premium_high = round(mid_premium * 1.15, 2)
        data_source = "ESTIMATED"

    mid_premium = max(mid_premium, 0.05)
    est_premium_low = max(est_premium_low, 0.01)
    est_premium_high = max(est_premium_high, 0.10)

    target_pct = 50
    stop_pct = 30
    target_price = round(mid_premium * 1.5, 2)
    stop_price = round(mid_premium * 0.7, 2)

    risk_per = round(mid_premium * (stop_pct / 100.0), 2)
    reward_per = round(mid_premium * (target_pct / 100.0), 2)
    rr_ratio = round(reward_per / risk_per, 2) if risk_per > 0 else 0

    max_risk_pct = 10.0
    max_risk_dollars = round(portfolio_cash * (max_risk_pct / 100.0), 2)
    cost_per_contract = round(mid_premium * 100, 2)
    max_contracts = max(1, int(max_risk_dollars / cost_per_contract)) if cost_per_contract > 0 else 0

    strikes = [
        {"label": "ATM", "strike": atm_strike, "recommended": otm_offset == 0},
        {"label": "OTM-1", "strike": otm_1, "recommended": otm_offset == 1},
        {"label": "OTM-2", "strike": otm_2, "recommended": otm_offset == 2},
    ]

    return {
        "active": True,
        "direction": type_label,
        "atm_strike": atm_strike,
        "recommended_strike": recommended_strike,
        "otm_offset": otm_offset,
        "contract_label": contract_label,
        "contract_type": contract_type,
        "strikes": strikes,
        "est_premium_low": est_premium_low,
        "est_premium_high": est_premium_high,
        "mid_premium": mid_premium,
        "data_source": data_source,
        "real_bid": real_bid,
        "real_ask": real_ask,
        "real_last": real_last,
        "target_pct": target_pct,
        "stop_pct": stop_pct,
        "target_price": target_price,
        "stop_price": stop_price,
        "risk_reward": f"1:{rr_ratio}",
        "max_contracts": max_contracts,
        "cost_per_contract": cost_per_contract,
        "max_risk_dollars": max_risk_dollars,
        "max_risk_pct": max_risk_pct,
        "reasoning": strike_reasoning,
    }


def load_portfolio():
    try:
        raw = os.getenv("PAPER_PORTFOLIO", "")
        if raw: return json.loads(raw)
    except: pass
    return {"cash": STARTING_BALANCE, "positions": {}, "history": [],
            "initial_balance": STARTING_BALANCE, "current_value": STARTING_BALANCE, "total_return_pct": 0.0}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        start_time = time.perf_counter()
        now = datetime.now(NY)
        ts = now.strftime("%Y-%m-%d %H:%M:%S")

        INDICES = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Russell 2000", "^VIX": "VIX"}
        MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
        GME_STOCK = ["GME"]

        try:
            all_syms = list(INDICES.keys()) + MAG7 + GME_STOCK + ["^VIX3M"]
            tickers = yf.Tickers(" ".join(all_syms))

            spy = tickers.tickers["SPY"]
            spy_p = float(spy.fast_info.last_price)
            spy_prev = float(spy.fast_info.previous_close)
            vix_p = float(tickers.tickers["^VIX"].fast_info.last_price)

            # VIX3M
            vix3m_p = None
            try: vix3m_p = float(tickers.tickers["^VIX3M"].fast_info.last_price)
            except: pass

            # SPY 5-min history
            spy_h = spy.history(period="1d", interval="5m", prepost=True)
            vwap, vol_r, d_range = 0.0, 0.0, 0.0
            if not spy_h.empty:
                tp = (spy_h["High"] + spy_h["Low"] + spy_h["Close"]) / 3.0
                cum_vol = spy_h["Volume"].cumsum().replace(0, pd.NA)
                vwap = float(((spy_h["Volume"] * tp).cumsum() / cum_vol).iloc[-1])
                vol_sma = spy_h["Volume"].rolling(window=20).mean()
                if not vol_sma.empty and pd.notna(vol_sma.iloc[-1]) and vol_sma.iloc[-1] > 0:
                    vol_r = float(spy_h["Volume"].iloc[-1] / vol_sma.iloc[-1])
                d_range = float(spy_h["High"].max() - spy_h["Low"].min())

            # ── Percentage Changes ──
            pcts_data = {}
            for sym in ["SPY", "QQQ", "IWM", "DIA"]:
                try:
                    t = tickers.tickers[sym]
                    pcts_data[sym] = (float(t.fast_info.last_price) / float(t.fast_info.previous_close) - 1) * 100
                except: pcts_data[sym] = 0

            # ── LAYER 2: Regime ──
            vix_score, vix_detail = _score_vix(vix_p)
            term_score, vix_spread, term_detail = _score_vix_term(vix_p, vix3m_p)
            adx_val = _calc_adx(spy_h)
            adx_score = 15 if adx_val and adx_val >= 25 else (5 if adx_val and adx_val >= 20 else 0)
            gap_pct = ((spy_p / spy_prev) - 1) * 100 if spy_prev else 0
            gap_score = 5 if abs(gap_pct) > 0.5 else 0
            regime_total = vix_score + term_score + adx_score + gap_score

            # Classify regime
            if adx_val and adx_val >= 25: regime_label = "TRENDING"
            elif adx_val and adx_val < 20: regime_label = "CHOPPY"
            else: regime_label = "UNKNOWN"

            regime = {
                "score": regime_total, "max": 40, "regime": regime_label,
                "vix_spread": vix_spread if vix3m_p else None,
                "details": {
                    "vix": {"score": vix_score, "detail": vix_detail},
                    "vix_term": {"score": term_score, "detail": term_detail},
                    "adx": {"score": adx_score, "detail": f"ADX {adx_val:.1f}" if adx_val else "ADX N/A", "value": adx_val},
                    "gap": {"score": gap_score, "detail": f"Gap {gap_pct:+.2f}%"},
                }
            }

            # ── LAYER 4: Correlation ──
            qqq_aligned = (pcts_data.get("SPY", 0) >= 0 and pcts_data.get("QQQ", 0) >= 0) or \
                          (pcts_data.get("SPY", 0) < 0 and pcts_data.get("QQQ", 0) < 0)
            iwm_risk = pcts_data.get("IWM", 0)
            sector_sync = all(v >= 0 for v in [pcts_data.get("SPY", 0), pcts_data.get("QQQ", 0), pcts_data.get("IWM", 0)]) or \
                          all(v < 0 for v in [pcts_data.get("SPY", 0), pcts_data.get("QQQ", 0), pcts_data.get("IWM", 0)])
            corr_score = (10 if qqq_aligned else -5) + (5 if iwm_risk > 0.3 else (-3 if iwm_risk < -0.3 else 0)) + (5 if sector_sync else 0)
            corr_score = max(0, min(20, corr_score))
            correlation = {"score": corr_score, "max": 20, "sector_sync": sector_sync,
                           "details": {"qqq_alignment": {"score": 10 if qqq_aligned else -5, "detail": f"QQQ {'aligned' if qqq_aligned else 'diverged'}"},
                                       "iwm_risk": {"score": 5 if iwm_risk > 0.3 else 0, "detail": f"IWM {iwm_risk:+.2f}%"},
                                       "sector_sync": {"score": 5 if sector_sync else 0, "detail": "Synced" if sector_sync else "Diverged"}}}

            # ── LAYER 5: Time Window ──
            time_win = _time_window(now)

            # ── LAYER 6: Technical ──
            vwap_score = 10 if spy_p > vwap else 10  # Both directions get score
            vwap_dir = "CALL" if spy_p > vwap else "PUT"
            vol_score = 10 if vol_r >= 2.0 else (7 if vol_r >= 1.5 else (3 if vol_r >= 1.0 else 0))
            range_score = 10 if d_range >= 3.0 else (5 if d_range >= 2.0 else 0)
            rsi_val = _calc_rsi(spy_h)
            rsi_score, rsi_dir = 0, "NEUTRAL"
            if rsi_val:
                if rsi_val >= 60: rsi_score, rsi_dir = 10, "CALL"
                elif rsi_val <= 40: rsi_score, rsi_dir = 10, "PUT"
            tech_total = min(30, vwap_score + vol_score + range_score + rsi_score)
            direction_bias = vwap_dir  # Primary bias from VWAP
            technical = {"score": tech_total, "max": 30, "direction_bias": direction_bias,
                         "rsi": rsi_val,
                         "details": {
                             "vwap_position": {"score": vwap_score, "detail": f"{'Above' if spy_p > vwap else 'Below'} VWAP by ${spy_p - vwap:+.2f}"},
                             "volume": {"score": vol_score, "detail": f"Volume {vol_r:.2f}x"},
                             "range": {"score": range_score, "detail": f"Range ${d_range:.2f}"},
                             "momentum": {"score": rsi_score, "detail": f"RSI {rsi_val:.1f}" if rsi_val else "RSI N/A"}
                         }}

            # ── LAYER 7: Risk (simplified for serverless) ──
            portfolio = load_portfolio()
            risk = {"passed": True, "score": 0, "lockout": False, "lockout_reason": None,
                    "strikes_remaining": 3, "trades_remaining": 3, "daily_drawdown": 0,
                    "details": {}}

            # ── TOTAL SCORE ──
            raw_total = regime_total + corr_score + time_win["score"] + tech_total
            active_max = 40 + 20 + 20 + 30  # 110
            normalized = max(0, int((raw_total / active_max) * 100)) if active_max > 0 else 0

            signal = _signal_grade(normalized)

            # Market session check
            t_min = now.hour * 60 + now.minute
            is_regular = 570 <= t_min <= 960

            if not is_regular:
                signal["label"] = "MARKET CLOSED"
                signal["action"] = "Market not in session"

            # Legacy rules
            rules = {
                "vix": {"val": f"{vix_p:.2f}", "ok": vix_p >= 14},
                "range": {"val": f"${d_range:.2f}", "ok": d_range >= 3.0},
                "window": {"val": now.strftime("%H:%M"), "ok": is_regular},
                "vwap": {"val": f"${spy_p - vwap:+.2f}", "ok": spy_p > vwap},
                "vol": {"val": f"{vol_r:.2f}x", "ok": vol_r >= 1.5},
                "sector": {"val": "SYNC" if sector_sync else "DIFF", "ok": sector_sync}
            }

            latency = round((time.perf_counter() - start_time) * 1000, 1)

            gme_data = {}
            for s in GME_STOCK:
                try:
                    t = tickers.tickers[s]
                    gme_data[s] = {"price": float(t.fast_info.last_price),
                                   "pct": (float(t.fast_info.last_price) / float(t.fast_info.previous_close) - 1) * 100}
                except: pass

            # ── STRIKE RECOMMENDATION ──
            strike_rec = _calculate_strike_recommendation(
                spy_price=spy_p,
                direction_bias=direction_bias,
                signal_grade=signal["grade"],
                vix_price=vix_p,
                vwap=vwap,
                normalized_score=normalized,
                portfolio_cash=portfolio.get("cash", STARTING_BALANCE),
                now_et=now,
            )

            final = {
                "last_updated": ts, "fetch_status": "SUCCESS", "latency_ms": latency,
                "session": "REGULAR" if is_regular else "CLOSED",
                "briefing": f"{time_win['emoji']} [{time_win['window']}] Regime: {regime_label} | Bias: {direction_bias} | Score: {normalized}/100",

                # Score engine output
                "total_score": normalized, "max_score": active_max, "raw_score": raw_total,
                "signal": signal, "direction_bias": direction_bias,
                "layers": {
                    "regime": regime,
                    "options_flow": {"score": 0, "max": 30, "status": "NOT_IMPLEMENTED", "detail": "Coming soon"},
                    "correlation": correlation,
                    "time_window": time_win,
                    "technical": technical,
                    "risk": risk,
                },

                # Strike recommendation
                "strike_recommendation": strike_rec,

                # Legacy
                "verdict": signal["label"], "confidence": normalized, "reason": signal["action"],
                "rules": rules, "alert_mode": "ON SIGNAL CHANGE",

                # Market data
                "indices": {s: {"price": float(tickers.tickers[s].fast_info.last_price),
                                "pct": (float(tickers.tickers[s].fast_info.last_price) / float(tickers.tickers[s].fast_info.previous_close) - 1) * 100}
                            for s in INDICES},
                "mag7": {s: {"price": float(tickers.tickers[s].fast_info.last_price),
                             "pct": (float(tickers.tickers[s].fast_info.last_price) / float(tickers.tickers[s].fast_info.previous_close) - 1) * 100}
                         for s in MAG7},
                "gme_data": gme_data, "special_watch": gme_data,
                "paper_trading": portfolio,
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(json.dumps(final).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
