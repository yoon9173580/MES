"""
v10 single-tick runner — shared by trading_bot.py (CLI / GitHub Actions) and
api/cron_v10.py (Vercel Cron). Lives under api/ so Vercel's Python builder
bundles it together with data.py, engines/ and lib/.

Strategy mirrors thorough_backtest_futures.py v10 profile:
  • single 10:30 ET PRIME entry (live window widened to [10:00, 12:00] for cron slack)
  • MIN_SCORE 88 (not grade STRONG)
  • SL = 1.5 × ATR, TP = 2.5 × SL  (the robust v10 lever)
  • dead-market guard (regime ATR% ≥ 0.3) instead of the overfit backtest ATR>8 filter
"""
import os
import sys
import json
import logging
from datetime import datetime

import pytz

# Make this module importable whether api/ is on sys.path (Vercel / trading_bot)
# or only the repo root is. Try bare names first, fall back to the api.* package.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    from data import (
        _fetch_market_bundle, _snap_price, _snap_prev_close, _pct,
        load_portfolio, ALL_STOCKS, STOCK_SYMS, _kv_get, _kv_set,
    )
    from engines.score_engine import run_score_engine
    from lib.brokers import get_broker
except ImportError:
    from api.data import (
        _fetch_market_bundle, _snap_price, _snap_prev_close, _pct,
        load_portfolio, ALL_STOCKS, STOCK_SYMS, _kv_get, _kv_set,
    )
    from api.engines.score_engine import run_score_engine
    from api.lib.brokers import get_broker

logger = logging.getLogger("v10")
NY = pytz.timezone("America/New_York")

V10_STATE_FILE = "v10_state.json"
V10_LOG_FILE = "v10_paper_log.json"
V10_LOG_CAP = 1000

V10_MIN_SCORE = 88
V10_SHORT_MIN_SCORE = 88
V10_SL_ATR_MULT = 1.5
V10_TP_MULT = 2.5
V10_ATR_PCT_FLOOR = 0.3
ES_PER_SPY = 10.0


# ── Pluggable persistence ────────────────────────────────────────────────────
class FileStore:
    """State/log on local filesystem — committed back by the GH Actions workflow."""
    def get_state(self):
        if os.path.exists(V10_STATE_FILE):
            try:
                with open(V10_STATE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_state(self, state):
        with open(V10_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def append_log(self, entry):
        log = []
        if os.path.exists(V10_LOG_FILE):
            try:
                with open(V10_LOG_FILE) as f:
                    log = json.load(f)
            except Exception:
                log = []
        if not isinstance(log, list):
            log = []
        log.append(entry)
        with open(V10_LOG_FILE, "w") as f:
            json.dump(log[-V10_LOG_CAP:], f, indent=2)


class KVStore:
    """State/log in Upstash KV — persists across ephemeral Vercel invocations."""
    STATE_KEY = "v10:state"
    LOG_KEY = "v10:log"

    def get_state(self):
        return _kv_get(self.STATE_KEY) or {}

    def save_state(self, state):
        _kv_set(self.STATE_KEY, state)

    def append_log(self, entry):
        log = _kv_get(self.LOG_KEY) or []
        if not isinstance(log, list):
            log = []
        log.append(entry)
        _kv_set(self.LOG_KEY, log[-V10_LOG_CAP:])


def _place_bracket_order(broker, symbol, qty, side, tp, sl):
    logger.info(f"[{broker.name}] Order: {symbol} {side} {qty} TP={tp} SL={sl}")
    res = broker.place_bracket_order(symbol, qty, side, tp, sl)
    if res:
        logger.info(f"Order placed: id={res.get('id')}")
    else:
        logger.error(f"Order failed via {broker.name}")
    return res


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


def run_once_entry(now=None, store=None):
    """Evaluate v10 entry once and place at most one paper order for the day."""
    now = now or datetime.now(NY)
    store = store or FileStore()
    today = now.strftime("%Y-%m-%d")
    state = store.get_state()

    t_min = now.hour * 60 + now.minute
    if not (10 * 60 <= t_min <= 12 * 60):
        logger.info(f"Outside v10 PRIME window ({now:%H:%M} ET) — no entry.")
        return {"action": "SKIP_WINDOW", "et": now.strftime("%H:%M")}
    if state.get("last_trade_date") == today:
        logger.info(f"Already evaluated/traded today ({today}) — skipping.")
        return {"action": "SKIP_DONE", "date": today}

    score_result, ctx = _evaluate(now)
    if score_result is None:
        logger.warning(f"Eval failed: {ctx.get('reason')}")
        store.append_log({"date": today, "ts": now.isoformat(),
                          "action": "NO_DATA", "reason": ctx.get("reason")})
        return {"action": "NO_DATA", "reason": ctx.get("reason")}

    total_score = score_result["total_score"]
    bias = score_result["direction_bias"]
    atr_pct = ctx.get("atr_pct")
    log_base = {"date": today, "ts": now.isoformat(), "score": total_score,
                "bias": bias, "atr_pct": atr_pct, "spy": ctx["spy_price"]}

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
        store.append_log({**log_base, "action": "NO_ENTRY", "reasons": reasons})
        state["last_trade_date"] = today
        store.save_state(state)
        return {"action": "NO_ENTRY", "reasons": reasons, "score": total_score}

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
        risk_per_contract = sl_es * 5.0 + 0.50
        qty = max(1, int((equity * 0.015) / risk_per_contract))
    else:
        symbol = "SPY"
        qty = 100
        tp_price = round(tp_price / ES_PER_SPY, 2)
        sl_price = round(sl_price / ES_PER_SPY, 2)

    logger.info(f"v10 ENTRY: {symbol} {side} score={total_score} SL={sl_es}pt TP={tp_es}pt")
    res = _place_bracket_order(broker, symbol, qty, side, tp_price, sl_price)
    entry = {**log_base, "action": "ENTRY", "symbol": symbol, "side": side,
             "qty": qty, "sl_pts": sl_es, "tp_pts": tp_es,
             "tp_price": tp_price, "sl_price": sl_price,
             "broker": broker.name, "order_id": (res or {}).get("id")}
    store.append_log(entry)
    state.update({"last_trade_date": today, "in_position": bool(res),
                  "last_order_id": (res or {}).get("id")})
    store.save_state(state)
    return entry


def run_once_flatten(now=None, store=None):
    """EOD flatten: close any open paper positions (bracket TP/SL handles intraday)."""
    now = now or datetime.now(NY)
    store = store or FileStore()
    broker = get_broker()
    positions = broker.get_open_positions()
    if not positions:
        logger.info("No open positions to flatten.")
    else:
        logger.info(f"Flattening {len(positions)} position(s) at EOD.")
        closer = getattr(broker, "close_all_positions", None)
        if callable(closer):
            closer()
    state = store.get_state()
    state["in_position"] = False
    store.save_state(state)
    result = {"date": now.strftime("%Y-%m-%d"), "ts": now.isoformat(),
              "action": "FLATTEN", "positions": len(positions)}
    store.append_log(result)
    return result
