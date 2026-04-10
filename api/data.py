from http.server import BaseHTTPRequestHandler
import json
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import pytz

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        start_time = time.perf_counter()
        NY = pytz.timezone("America/New_York")
        now = datetime.now(NY)
        ts = now.strftime("%Y-%m-%d %H:%M:%S")

        INDICES = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Russell 2000", "^VIX": "VIX"}
        MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
        GME_STOCK = ["GME"]

        try:
            all_syms = list(INDICES.keys()) + MAG7 + GME_STOCK
            tickers = yf.Tickers(" ".join(all_syms))
            spy = tickers.tickers["SPY"]
            spy_p = float(spy.fast_info.last_price)
            vix_p = float(tickers.tickers["^VIX"].fast_info.last_price)

            spy_h = spy.history(period="1d", interval="5m", prepost=True)
            vwap, vol_r, d_range = 0.0, 0.0, 0.0
            if not spy_h.empty:
                tp = (spy_h["High"] + spy_h["Low"] + spy_h["Close"]) / 3.0
                vwap = float(((spy_h["Volume"] * tp).cumsum() / spy_h["Volume"].cumsum()).iloc[-1])
                vol_sma = spy_h["Volume"].rolling(window=20).mean()
                vol_r = float(spy_h["Volume"].iloc[-1] / vol_sma.iloc[-1]) if not vol_sma.empty else 0.0
                d_range = float(spy_h["High"].max() - spy_h["Low"].min())

            t_min = now.hour * 60 + now.minute
            win_ok = (570 <= t_min <= 630) or (840 <= t_min <= 930)

            spy_pct = (spy_p / spy.fast_info.previous_close - 1) * 100
            qqq_pct = (tickers.tickers["QQQ"].fast_info.last_price / tickers.tickers["QQQ"].fast_info.previous_close - 1) * 100
            iwm_pct = (tickers.tickers["IWM"].fast_info.last_price / tickers.tickers["IWM"].fast_info.previous_close - 1) * 100
            sector_sync = (spy_pct >= 0 and qqq_pct >= 0 and iwm_pct >= 0) or (spy_pct <= 0 and qqq_pct <= 0 and iwm_pct <= 0)

            rules = {
                "vix": {"val": f"{vix_p:.2f}", "ok": vix_p >= 14},
                "range": {"val": f"${d_range:.2f}", "ok": d_range >= 3.0},
                "window": {"val": now.strftime("%H:%M"), "ok": win_ok},
                "vwap": {"val": f"${spy_p - vwap:+.2f}", "ok": spy_p > vwap},
                "vol": {"val": f"{vol_r:.2f}x", "ok": vol_r >= 1.5},
                "sector": {"val": "SYNC" if sector_sync else "DIFF", "ok": sector_sync}
            }

            latency = round((time.perf_counter() - start_time) * 1000, 1)
            final = {
                "last_updated": ts, "fetch_status": "SUCCESS", "latency_ms": latency,
                "verdict": "STRONG GO" if all(r["ok"] for r in rules.values()) else "WAITING",
                "rules": rules,
                "indices": {s: {"price": float(tickers.tickers[s].fast_info.last_price), "pct": (float(tickers.tickers[s].fast_info.last_price / tickers.tickers[s].fast_info.previous_close)-1)*100} for s in INDICES},
                "mag7": {s: {"price": float(tickers.tickers[s].fast_info.last_price), "pct": (float(tickers.tickers[s].fast_info.last_price / tickers.tickers[s].fast_info.previous_close)-1)*100} for s in MAG7},
                "gme_data": {s: {"price": float(tickers.tickers[s].fast_info.last_price), "pct": (float(tickers.tickers[s].fast_info.last_price / tickers.tickers[s].fast_info.previous_close)-1)*100} for s in GME_STOCK}
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(final).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
