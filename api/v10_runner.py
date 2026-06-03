"""
v10 single-tick runner — shared by trading_bot.py (CLI / GitHub Actions) and
api/cron_v10.py (Vercel Cron). Lives under api/ so Vercel's Python builder
bundles it together with data.py, engines/ and lib/.

The entry decision is NOT re-implemented here — it is delegated to
api/v10_strategy.evaluate_entry, the SAME pure function the backtest
(thorough_backtest_futures.py) calls. That guarantees the live paper bot trades
the exact strategy that produced the published metrics:
  • single 10:30 ET PRIME entry, 14-day daily ATR (ES points)
  • SL = min(max(1.5×ATR, 2), 15), TP = 2.5×SL
  • MIN_SCORE 60, ATR floor 8, NR7/pullback boosts, runaway veto, daily-bias,
    VIX-20 mean-reversion switch — all identical to the backtest.
This module only handles data plumbing (SPY→ES scale), order sizing, bracket
placement, persistence and the cron entry/flatten gating.
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
        _fetch_market_bundle, _snap_price,
        _alpaca_daily_bars, _alpaca_morning_1min, _kv_get, _kv_set,
    )
    from v10_strategy import evaluate_entry, ENTRY_TIME
    from lib.brokers import get_broker
except ImportError:
    from api.data import (
        _fetch_market_bundle, _snap_price,
        _alpaca_daily_bars, _alpaca_morning_1min, _kv_get, _kv_set,
    )
    from api.v10_strategy import evaluate_entry, ENTRY_TIME
    from api.lib.brokers import get_broker

logger = logging.getLogger("v10")
NY = pytz.timezone("America/New_York")

V10_STATE_FILE = "v10_state.json"
V10_LOG_FILE = "v10_paper_log.json"
V10_LOG_CAP = 1000

# Entry-decision parameters live in api/v10_strategy.py (shared with the
# backtest). These are kept only for the order-sizing math below, which must
# match thorough_backtest_futures.py's contract sizing.
ES_PER_SPY = 10.0
ES_MULTIPLIER = 5.0        # $5 per point (MES — Micro E-mini)
ES_COMMISSION_RT = 0.50
ES_SLIPPAGE_PTS = 0.25
ES_DAY_MARGIN = 50.0
RISK_PCT = 0.015
MARGIN_UTIL = 0.95
# Back-compat re-exports (trading_bot.py imports these names)
V10_MIN_SCORE = 60
V10_TP_MULT = 2.5
V10_SL_ATR_MULT = 1.5


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


def _gather_inputs(now):
    """Assemble the exact inputs the shared v10 decision needs, in ES price space.

    Backtest replays index/ES-scale prices (~5300). Live snapshots are SPY ETF
    (~530), so every price is multiplied by ES_PER_SPY (10) to land on the same
    scale the backtest (and its ATR floor of 8 points) was tuned on.
    Returns (inputs_dict, ctx) or (None, {"reason": ...}) on failure.
    """
    bundle = _fetch_market_bundle(["SPY"])
    snaps = bundle.get("snaps") or {}
    spy_price = _snap_price(snaps.get("SPY", {}))
    if not spy_price:
        return None, {"reason": "no SPY snapshot"}
    vix_p, _ = bundle.get("vix", (18.0, None))

    # Daily history (newest LAST) for ATR / NR7 / pullback / 20-SMA bias.
    try:
        daily = _alpaca_daily_bars("SPY", days=30)
    except Exception as e:
        return None, {"reason": f"daily bars failed: {e}"}
    if daily is None or daily.empty or len(daily) < 11:
        return None, {"reason": "insufficient daily history"}
    dh = (daily["High"] * ES_PER_SPY).tolist()
    dl = (daily["Low"] * ES_PER_SPY).tolist()
    dc = (daily["Close"] * ES_PER_SPY).tolist()
    day_open = float(daily["Open"].iloc[-1]) * ES_PER_SPY

    # Morning 1-min slice up to the 10:30 entry, in ES scale.
    try:
        morning = _alpaca_morning_1min("SPY")
    except Exception as e:
        return None, {"reason": f"morning bars failed: {e}"}
    if morning is None or morning.empty:
        return None, {"reason": "no morning bars"}
    morning = morning[morning.index.time <= ENTRY_TIME].copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in morning.columns:
            morning[col] = morning[col] * ES_PER_SPY
    if len(morning) < 5:
        return None, {"reason": f"morning slice {len(morning)} bars < 5"}

    entry_price = float(morning["Close"].iloc[-1])
    entry_ts = morning.index[-1].to_pydatetime()
    inputs = {
        "daily_highs": dh, "daily_lows": dl, "daily_closes": dc,
        "day_open": day_open, "entry_price": entry_price, "entry_ts": entry_ts,
        "vix_val": vix_p, "morning_df": morning,
    }
    ctx = {"spy_price": spy_price, "es_price": entry_price, "vix": vix_p}
    return inputs, ctx


def run_once_entry(now=None, store=None):
    """Evaluate v10 entry once and place at most one paper order for the day.

    Delegates the whole decision (ATR, scoring, direction, SL/TP) to
    api/v10_strategy.evaluate_entry — the *identical* function the backtest uses
    — so the live paper bot trades the published strategy bar-for-bar.
    """
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

    inputs, ctx = _gather_inputs(now)
    if inputs is None:
        logger.warning(f"Eval failed: {ctx.get('reason')}")
        store.append_log({"date": today, "ts": now.isoformat(),
                          "action": "NO_DATA", "reason": ctx.get("reason")})
        return {"action": "NO_DATA", "reason": ctx.get("reason")}

    decision = evaluate_entry(**inputs)
    log_base = {"date": today, "ts": now.isoformat(), "score": decision["score"],
                "atr": decision["atr"], "spy": ctx["spy_price"], "vix": ctx["vix"]}

    if not decision["enter"]:
        reasons = decision["reasons"]
        logger.info(f"No entry — {'; '.join(reasons)}")
        store.append_log({**log_base, "action": "NO_ENTRY", "reasons": reasons,
                          "direction": decision.get("direction")})
        state["last_trade_date"] = today
        store.save_state(state)
        return {"action": "NO_ENTRY", "reasons": reasons, "score": decision["score"]}

    sl_es = round(decision["sl_points"], 2)
    tp_es = round(decision["tp_points"], 2)
    trade_dir = decision["direction"]
    es_price = ctx["es_price"]
    side = "buy" if trade_dir == "LONG" else "sell"
    if trade_dir == "LONG":
        tp_price, sl_price = es_price + tp_es, es_price - sl_es
    else:
        tp_price, sl_price = es_price - tp_es, es_price + sl_es

    broker = get_broker()
    # Sizing mirrors thorough_backtest_futures.py: risk% with slippage + margin cap.
    equity = broker.get_account_equity() or 10000.0
    risk_per_contract = (sl_es + ES_SLIPPAGE_PTS * 2) * ES_MULTIPLIER + ES_COMMISSION_RT
    qty = max(1, int((equity * RISK_PCT) / risk_per_contract))
    qty = min(qty, max(1, int((equity * MARGIN_UTIL) / ES_DAY_MARGIN)))

    if broker.supports_futures:
        from lib.futures_meta import current_mes_contract
        symbol = current_mes_contract(now)
    else:
        symbol = "SPY"
        tp_price = round(tp_price / ES_PER_SPY, 2)
        sl_price = round(sl_price / ES_PER_SPY, 2)

    logger.info(f"v10 ENTRY: {symbol} {side} score={decision['score']} "
                f"dir={trade_dir} SL={sl_es}pt TP={tp_es}pt qty={qty}")
    res = _place_bracket_order(broker, symbol, qty, side, tp_price, sl_price)
    entry = {**log_base, "action": "ENTRY", "symbol": symbol, "side": side,
             "direction": trade_dir, "strategy": decision["strategy"],
             "boosted_score": decision["boosted_score"], "qty": qty,
             "sl_pts": sl_es, "tp_pts": tp_es,
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
