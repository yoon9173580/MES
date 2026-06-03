"""
MILLI-V3 Auto-Trading Bot — paper SPY proxy for MES futures signal.

Reuses the same market-data fetch and 7-layer score engine that powers /api/data,
so the live bot and dashboard never disagree on what the model said.
"""
import sys
import os
import time
import logging
import json
from datetime import datetime

import requests
import pytz

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

# Make the api package importable regardless of CWD.
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "api"))

from api.data import (
    _fetch_market_bundle,
    _snap_price,
    _snap_prev_close,
    _pct,
    load_portfolio,
    ALL_STOCKS,
    STOCK_SYMS,
)
from engines.score_engine import run_score_engine
from engines.ml_weights import feedback_trade_result
from lib.brokers import get_broker

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("AutoTrader")

load_dotenv()

# Broker is selected by env var BROKER (alpaca / tradovate / dryrun).
# Defaults to Alpaca paper for back-compat. Switch to tradovate for real
# MES/ES futures once TRADOVATE_* env vars are set.
BROKER = get_broker()

STATE_FILE = "data_cache/bot_state.json"
NY = pytz.timezone("America/New_York")


def get_bot_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"in_position": False, "last_trade_id": None, "last_dominant": None, "entry_equity": None}


def save_bot_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_open_positions():
    """Open positions from the active broker (empty on failure)."""
    return BROKER.get_open_positions()


def get_account_equity():
    return BROKER.get_account_equity()


def place_bracket_order(symbol, qty, side, take_profit, stop_loss):
    logger.info(f"[{BROKER.name}] Submitting Order: {symbol} {side} {qty} TP={take_profit} SL={stop_loss}")
    res = BROKER.place_bracket_order(symbol, qty, side, take_profit, stop_loss)
    if res:
        logger.info(f"Order placed: id={res.get('id')}")
    else:
        logger.error(f"Order failed via {BROKER.name}")
    return res


def _dominant_layer(score_result):
    """Pick the layer that contributed the largest share of the total score.

    Includes 'flow' (options_flow) when LIVE — previously the flow weight
    in ml_weights was set up to be tuned but never received feedback
    because the bot only voted between {regime, correlation, technical}.
    """
    layers = score_result.get("layers", {})
    candidates = {
        "regime":      layers.get("regime", {}).get("score", 0) or 0,
        "correlation": layers.get("correlation", {}).get("score", 0) or 0,
        "technical":   layers.get("technical", {}).get("score", 0) or 0,
    }
    of_layer = layers.get("options_flow", {})
    if of_layer.get("status") == "LIVE":
        candidates["flow"] = of_layer.get("score", 0) or 0
    if not any(candidates.values()):
        return "technical"
    return max(candidates, key=candidates.get)


def main_loop():
    logger.info(f"MILLI-V3 Auto-Trading Bot starting (broker={BROKER.name})")
    ok, reason = BROKER.is_ready()
    if not ok:
        logger.error(f"Broker {BROKER.name} not ready: {reason}")
        return

    if BROKER.name == "alpaca_paper":
        logger.warning("⚠️  Alpaca paper supports SPY equity only — for real MES futures set BROKER=tradovate")
    elif BROKER.name == "tradovate":
        # For futures, the bot should trade the active MES contract code,
        # not "SPY". Override the symbol used below.
        logger.info("Tradovate active — symbol will be MES front-month contract")

    while True:
        try:
            now = datetime.now(NY)

            # 1. Pull the same market bundle the dashboard uses.
            bundle = _fetch_market_bundle(ALL_STOCKS)
            snaps = bundle.get("snaps") or {}
            spy_snap = snaps.get("SPY", {})
            spy_price = _snap_price(spy_snap)
            if not spy_price:
                logger.warning("No SPY snapshot — skipping tick")
                time.sleep(10)
                continue

            spy_prev = _snap_prev_close(spy_snap) or spy_price
            vix_p, vix3m_p = bundle.get("vix", (18.0, None))
            spy_h = bundle.get("spy_h")

            # 2. Build the same pcts dict /api/data feeds the score engine.
            pcts = {}
            for sym in STOCK_SYMS:
                s = snaps.get(sym, {})
                pcts[sym] = _pct(_snap_price(s), _snap_prev_close(s) or 1)

            # VWAP / volume / range — keep simple, the engine tolerates approximations.
            vwap = spy_price
            vol_r = 1.0
            d_range = abs(spy_price - spy_prev)

            # 3. Score engine expects the paper-portfolio shape (history/trade_log/positions).
            portfolio = load_portfolio()

            t_min = now.hour * 60 + now.minute
            is_regular = 570 <= t_min <= 960

            score_result = run_score_engine(
                now_et=now,
                spy_price=spy_price,
                vix_price=vix_p,
                vix3m_price=vix3m_p,
                prev_close=spy_prev,
                vwap=vwap,
                vol_ratio=vol_r,
                range_value=d_range,
                pcts=pcts,
                spy_history=spy_h,
                portfolio=portfolio,
                session_name="REGULAR" if is_regular else "CLOSED",
            )

            signal = score_result["signal"]
            total_score = score_result["total_score"]
            bias = score_result["direction_bias"]

            logger.info(f"Signal: {total_score}/100 [{signal['grade']}] Bias: {bias}")

            # 4. Execution logic.
            state = get_bot_state()
            open_positions = get_open_positions()
            has_open = len(open_positions) > 0

            if has_open:
                if not state.get("in_position"):
                    state["in_position"] = True
                    state["entry_equity"] = get_account_equity()
                    save_bot_state(state)
            else:
                if state.get("in_position"):
                    # Position just closed — compute realized PnL and feed it back to ML.
                    new_equity = get_account_equity()
                    entry_eq = state.get("entry_equity")
                    if new_equity is not None and entry_eq is not None:
                        realized = new_equity - entry_eq
                    else:
                        realized = 0.0
                    dominant = state.get("last_dominant") or "technical"
                    feedback_trade_result({"pnl": realized, "dominant_layer": dominant})
                    logger.info(f"Position closed. PnL={realized:.2f} → feedback layer={dominant}")
                    state.update({"in_position": False, "entry_equity": None})
                    save_bot_state(state)

                # Entry criteria — match the dashboard's STRONG-only rule.
                # Trust the grade computed by the score engine (GRADE_STRONG
                # may be env-overridden) instead of hard-coding a score floor
                # that could disagree with the dashboard's classification.
                # SHORT trades require >=93 (matches api/data.py
                # _entry_criteria_met) — backtest had only 7 SHORT in 78
                # so the SHORT setup needs stronger conviction.
                if (
                    is_regular
                    and signal.get("grade") == "STRONG"
                    and bias in ("LONG", "SHORT")
                    and not (bias == "SHORT" and total_score < 93)
                ):
                    logger.info("STRONG SIGNAL DETECTED — submitting bracket order")
                    side = "buy" if bias == "LONG" else "sell"

                    # ATR-based SL/TP (regime-aware RR computed in score_engine)
                    atr = max(d_range, 2.0)
                    regime_label = score_result["layers"].get("regime", {}).get("regime", "UNKNOWN")
                    if regime_label in ("TRENDING", "BREAKOUT"):
                        rr = 3.0
                    elif regime_label == "CHOPPY":
                        rr = 1.5
                    else:
                        rr = 2.0
                    if bias == "LONG":
                        tp = spy_price + atr * rr
                        sl = spy_price - atr * 1.5
                    else:
                        tp = spy_price - atr * rr
                        sl = spy_price + atr * 1.5

                    # Symbol + qty differ by broker type.
                    if BROKER.supports_futures:
                        # MES front-month contract code, 1.5% account risk → ~1-3 contracts
                        from lib.futures_meta import current_mes_contract
                        symbol = current_mes_contract(now)
                        equity = get_account_equity() or 10000.0
                        risk_amount = equity * 0.015
                        risk_per_contract = abs(spy_price - sl) * 5.0 + 0.50  # $5/pt + commission
                        qty = max(1, int(risk_amount / risk_per_contract))
                    else:
                        symbol = "SPY"
                        qty = 100  # equity proxy

                    res = place_bracket_order(symbol, qty, side, tp, sl)
                    if res:
                        state.update({
                            "in_position": True,
                            "last_trade_id": res.get("id"),
                            "last_dominant": _dominant_layer(score_result),
                            "entry_equity": get_account_equity(),
                        })
                        save_bot_state(state)

        except Exception as e:
            logger.exception(f"Error in main loop: {e}")

        time.sleep(60)



# ─────────────────────────────────────────────────────────────────────────────
# v10 single-tick mode — logic lives in api/v10_runner.py (shared with Vercel
# cron at api/cron_v10.py). Re-exported here so `python trading_bot.py --once`
# keeps working for the GitHub Actions path.
# ─────────────────────────────────────────────────────────────────────────────
from api.v10_runner import (  # noqa: E402
    FileStore, KVStore, run_once_entry, run_once_flatten, run_once_monitor,
    v10_worker, V10_MIN_SCORE, V10_TP_MULT, V10_SL_ATR_MULT,
)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MES v10 trading bot")
    p.add_argument("--once", choices=["entry", "flatten", "monitor"],
                   help="Run a single tick (for cron) then exit. "
                        "Omit to run --worker or the legacy main_loop.")
    p.add_argument("--worker", action="store_true",
                   help="Run the persistent v10 worker: minute-by-minute entry + "
                        "trail/BE exit management (most faithful to the backtest).")
    p.add_argument("--poll", type=int, default=60,
                   help="Worker poll interval in seconds (default 60).")
    p.add_argument("--store", choices=["file", "kv"], default="file",
                   help="State backend: file (default) or kv (Upstash).")
    a = p.parse_args()
    _store = KVStore() if a.store == "kv" else FileStore()
    if a.once == "entry":
        run_once_entry(store=_store)
    elif a.once == "flatten":
        run_once_flatten(store=_store)
    elif a.once == "monitor":
        run_once_monitor(store=_store)
    elif a.worker:
        v10_worker(poll_sec=a.poll, store=_store)
    else:
        main_loop()
