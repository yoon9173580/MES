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
# v10 single-tick mode (for GitHub Actions scheduling)
# ─────────────────────────────────────────────────────────────────────────────
# Persistent state/log live at the repo root (NOT data_cache/, which is
# gitignored) so the scheduled workflow can commit them back between the
# ephemeral runs.
V10_STATE_FILE = "v10_state.json"
V10_LOG_FILE = "v10_paper_log.json"

# v10 strategy constants (mirror thorough_backtest_futures.py v10 profile)
V10_MIN_SCORE = 88            # entry score floor (MIN_SCORE, not grade STRONG)
V10_SHORT_MIN_SCORE = 88      # backtest v10 used same floor for SHORT
V10_SL_ATR_MULT = 1.5         # SL = 1.5 × ATR
V10_TP_MULT = 2.5             # TP = 2.5 × SL  (the robust v10 lever)
V10_ATR_PCT_FLOOR = 0.3       # skip "dead market" (robust stand-in for ATR>8)
ES_PER_SPY = 10.0             # S&P 500 index ≈ SPY ETF × 10


def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _append_log(entry):
    log = _load_json(V10_LOG_FILE, [])
    if not isinstance(log, list):
        log = []
    log.append(entry)
    _save_json(V10_LOG_FILE, log)


def _evaluate(now):
    """Run the same score engine the dashboard uses; return (score_result, ctx)."""
    bundle = _fetch_market_bundle(ALL_STOCKS)
    snaps = bundle.get("snaps") or {}
    spy_snap = snaps.get("SPY", {})
    spy_price = _snap_price(spy_snap)
    if not spy_price:
        return None, {"reason": "no SPY snapshot"}
    spy_prev = _snap_prev_close(spy_snap) or spy_price
    vix_p, vix3m_p = bundle.get("vix", (18.0, None))
    spy_h = bundle.get("spy_h")

    pcts = {}
    for sym in STOCK_SYMS:
        s = snaps.get(sym, {})
        pcts[sym] = _pct(_snap_price(s), _snap_prev_close(s) or 1)

    t_min = now.hour * 60 + now.minute
    is_regular = 570 <= t_min <= 960
    score_result = run_score_engine(
        now_et=now, spy_price=spy_price, vix_price=vix_p, vix3m_price=vix3m_p,
        prev_close=spy_prev, vwap=spy_price, vol_ratio=1.0,
        range_value=abs(spy_price - spy_prev), pcts=pcts, spy_history=spy_h,
        portfolio=load_portfolio(),
        session_name="REGULAR" if is_regular else "CLOSED",
    )
    # ATR estimate in ES index points from intraday session range (×10 SPY→ES).
    atr_es = 8.0
    try:
        if spy_h is not None and not spy_h.empty:
            sess_range = float(spy_h["High"].max() - spy_h["Low"].min())
            atr_es = max(min(sess_range * ES_PER_SPY, 15.0), 4.0)
    except Exception:
        pass
    ctx = {
        "spy_price": spy_price, "es_price": spy_price * ES_PER_SPY,
        "atr_es": atr_es, "is_regular": is_regular,
        "atr_pct": (score_result.get("layers", {}).get("regime", {}) or {}).get("atr_pct"),
    }
    return score_result, ctx


def run_once_entry(now=None):
    """Evaluate v10 entry once and place at most one paper order for the day."""
    now = now or datetime.now(NY)
    today = now.strftime("%Y-%m-%d")
    state = _load_json(V10_STATE_FILE, {})

    # PRIME window gate (tolerates GitHub Actions cron delay up to ~60 min)
    t_min = now.hour * 60 + now.minute
    if not (10 * 60 + 25 <= t_min <= 11 * 60 + 30):
        logger.info(f"Outside v10 PRIME window ({now:%H:%M} ET) — no entry.")
        return
    if state.get("last_trade_date") == today:
        logger.info(f"Already evaluated/traded today ({today}) — skipping.")
        return

    score_result, ctx = _evaluate(now)
    if score_result is None:
        logger.warning(f"Eval failed: {ctx.get('reason')}")
        # Don't mark the day done — a later cron tick may have data.
        _append_log({"date": today, "ts": now.isoformat(),
                     "action": "NO_DATA", "reason": ctx.get("reason")})
        return

    total_score = score_result["total_score"]
    bias = score_result["direction_bias"]
    atr_pct = ctx.get("atr_pct")

    log_base = {
        "date": today, "ts": now.isoformat(), "score": total_score,
        "bias": bias, "atr_pct": atr_pct, "spy": ctx["spy_price"],
    }

    # ── v10 entry gate ──
    reasons = []
    if not ctx["is_regular"]:
        reasons.append("not RTH")
    if total_score < V10_MIN_SCORE:
        reasons.append(f"score {total_score} < {V10_MIN_SCORE}")
    if bias not in ("LONG", "SHORT"):
        reasons.append(f"bias {bias}")
    if bias == "SHORT" and total_score < V10_SHORT_MIN_SCORE:
        reasons.append("SHORT conviction")
    if atr_pct is not None and atr_pct < V10_ATR_PCT_FLOOR:
        reasons.append(f"dead market atr%={atr_pct:.2f}")

    if reasons:
        logger.info(f"No entry — {'; '.join(reasons)}")
        _append_log({**log_base, "action": "NO_ENTRY", "reasons": reasons})
        state["last_trade_date"] = today
        _save_json(V10_STATE_FILE, state)
        return

    # ── place v10 bracket order ──
    sl_es = round(max(V10_SL_ATR_MULT * ctx["atr_es"], 2.0), 2)
    tp_es = round(V10_TP_MULT * sl_es, 2)
    es_price = ctx["es_price"]
    side = "buy" if bias == "LONG" else "sell"
    if bias == "LONG":
        tp_price, sl_price = es_price + tp_es, es_price - sl_es
    else:
        tp_price, sl_price = es_price - tp_es, es_price + sl_es

    broker = get_broker()
    if broker.supports_futures:
        from lib.futures_meta import current_mes_contract
        symbol = current_mes_contract(now)
        equity = broker.get_account_equity() or 500000.0
        risk_per_contract = sl_es * 5.0 + 0.50  # $5/pt + commission
        qty = max(1, int((equity * 0.015) / risk_per_contract))
    else:
        # equity-proxy broker (Alpaca SPY): convert ES levels back to SPY
        symbol = "SPY"
        qty = 100
        tp_price = round(tp_price / ES_PER_SPY, 2)
        sl_price = round(sl_price / ES_PER_SPY, 2)

    logger.info(f"v10 ENTRY: {symbol} {side} score={total_score} SL={sl_es}pt TP={tp_es}pt")
    res = place_bracket_order(symbol, qty, side, tp_price, sl_price)
    _append_log({
        **log_base, "action": "ENTRY", "symbol": symbol, "side": side,
        "qty": qty, "sl_pts": sl_es, "tp_pts": tp_es,
        "tp_price": tp_price, "sl_price": sl_price,
        "broker": broker.name, "order_id": (res or {}).get("id"),
    })
    state.update({"last_trade_date": today, "in_position": bool(res),
                  "last_order_id": (res or {}).get("id")})
    _save_json(V10_STATE_FILE, state)


def run_once_flatten(now=None):
    """EOD flatten: close any open paper positions (bracket TP/SL handles intraday)."""
    now = now or datetime.now(NY)
    broker = get_broker()
    positions = broker.get_open_positions()
    if not positions:
        logger.info("No open positions to flatten.")
    else:
        logger.info(f"Flattening {len(positions)} position(s) at EOD.")
        closer = getattr(broker, "close_all_positions", None)
        if callable(closer):
            closer()
    state = _load_json(V10_STATE_FILE, {})
    state["in_position"] = False
    _save_json(V10_STATE_FILE, state)
    _append_log({"date": now.strftime("%Y-%m-%d"), "ts": now.isoformat(),
                 "action": "FLATTEN", "positions": len(positions)})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MES v10 trading bot")
    p.add_argument("--once", choices=["entry", "flatten"],
                   help="Run a single tick (for GitHub Actions cron) then exit. "
                        "Omit to run the continuous main_loop.")
    a = p.parse_args()
    if a.once == "entry":
        run_once_entry()
    elif a.once == "flatten":
        run_once_flatten()
    else:
        main_loop()
