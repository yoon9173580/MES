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
    from v10_strategy import (evaluate_entry, init_position, manage_bar,
                              ENTRY_TIME, EXIT_TIME, vix_risk_pct)
    from lib.brokers import get_broker
except ImportError:
    from api.data import (
        _fetch_market_bundle, _snap_price,
        _alpaca_daily_bars, _alpaca_morning_1min, _kv_get, _kv_set,
    )
    from api.v10_strategy import (evaluate_entry, init_position, manage_bar,
                                  ENTRY_TIME, EXIT_TIME, vix_risk_pct)
    from api.lib.brokers import get_broker

logger = logging.getLogger("v10")
NY = pytz.timezone("America/New_York")

V10_STATE_FILE = "v10_state.json"
V10_LOG_FILE = "v10_paper_log.json"
V10_LOG_CAP = 1000

# All authoritative entry & sizing parameters come from v10_constants (via v10_strategy).
# These local names are kept only for back-compat with older callers (trading_bot.py etc).
try:
    from .v10_constants import (
        ES_PER_SPY, ES_MULTIPLIER, ES_COMMISSION_RT, ES_SLIPPAGE_PTS, ES_DAY_MARGIN,
        TP_MULT as V10_TP_MULT,
        MIN_SCORE as V10_MIN_SCORE,
        ATR_SL_MULT as V10_SL_ATR_MULT,
    )
except ImportError:
    from api.v10_constants import (
        ES_PER_SPY, ES_MULTIPLIER, ES_COMMISSION_RT, ES_SLIPPAGE_PTS, ES_DAY_MARGIN,
        TP_MULT as V10_TP_MULT,
        MIN_SCORE as V10_MIN_SCORE,
        ATR_SL_MULT as V10_SL_ATR_MULT,
    )
MARGIN_UTIL = 0.95

# Note: Live uses vix-scaled risk (RISK_PCT_FULL etc) inside vix_risk_pct + runner sizing.
# The old flat RISK_PCT=0.015 is no longer the primary live value.


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

    # Record ML context (adaptive weights). Hard-skip classifier is backtest-only.
    try:
        try:
            from .v10_strategy import get_live_ml_context, should_apply_live_ml_skip
        except ImportError:
            from api.v10_strategy import get_live_ml_context, should_apply_live_ml_skip
        ml_ctx = get_live_ml_context()
        skip, skip_reason = should_apply_live_ml_skip(decision, ml_ctx)
        if skip:
            decision["enter"] = False
            decision.setdefault("reasons", []).append(f"ml_hard_skip: {skip_reason}")
        decision["ml_context"] = {
            "confidence": ml_ctx.get("confidence"),
            "sample_count": ml_ctx.get("sample_count"),
            "skip_checked": True,
            "skip_reason": skip_reason,
        }
    except Exception:
        decision["ml_context"] = {"confidence": "ERROR", "skip_checked": False}

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
    # Sizing: VIX-scaled risk% (Option C) + slippage + margin cap.
    equity = broker.get_account_equity() or 10000.0
    risk_pct = vix_risk_pct(ctx["vix"])
    risk_per_contract = (sl_es + ES_SLIPPAGE_PTS * 2) * ES_MULTIPLIER + ES_COMMISSION_RT
    qty = max(1, int((equity * risk_pct) / risk_per_contract))
    qty = min(qty, max(1, int((equity * MARGIN_UTIL) / ES_DAY_MARGIN)))

    if broker.supports_futures:
        from lib.futures_meta import current_mes_contract
        symbol = current_mes_contract(now)
    else:
        symbol = "SPY"
        tp_price = round(tp_price / ES_PER_SPY, 2)
        sl_price = round(sl_price / ES_PER_SPY, 2)

    logger.info(f"v10 ENTRY: {symbol} {side} score={decision['score']} "
                f"dir={trade_dir} strategy={decision['strategy']} "
                f"SL={sl_es}pt TP={tp_es}pt qty={qty} risk%={risk_pct*100:.1f} vix={ctx['vix']:.1f}")
    res = _place_bracket_order(broker, symbol, qty, side, tp_price, sl_price)
    entry = {**log_base, "action": "ENTRY", "symbol": symbol, "side": side,
             "direction": trade_dir, "strategy": decision["strategy"],
             "boosted_score": decision["boosted_score"], "qty": qty,
             "sl_pts": sl_es, "tp_pts": tp_es,
             "tp_price": tp_price, "sl_price": sl_price,
             "broker": broker.name, "order_id": (res or {}).get("id")}
    store.append_log(entry)
    # Position state for the worker's intraday trail/BE management. init_position
    # is the SAME builder the backtest uses, so manage_bar() trails identically.
    pos = init_position(trade_dir, es_price, decision["atr"], sl_es, V10_TP_MULT)
    pos.update({"open": True, "qty": qty, "symbol": symbol, "side": side,
                "date": today, "entry_ts": now.isoformat(),
                "score": decision["score"], "strategy": decision["strategy"]})
    state.update({"last_trade_date": today, "in_position": bool(res),
                  "last_order_id": (res or {}).get("id"), "position": pos})
    store.save_state(state)
    return entry


def _latest_bar_es():
    """Most recent completed 1-min (High, Low) in ES scale, or None."""
    try:
        m = _alpaca_morning_1min("SPY")
    except Exception:
        return None
    if m is None or m.empty:
        return None
    last = m.iloc[-1]
    return float(last["High"]) * ES_PER_SPY, float(last["Low"]) * ES_PER_SPY


def _close_paper(store, state, pos, exit_price, exit_type, now):
    """Record a paper exit (simulated fill, exactly like the backtest) + close."""
    entry_price = pos["entry_price"]
    qty = pos.get("qty", 1)
    if pos["trade_dir"] == "LONG":
        pts = exit_price - entry_price
    else:
        pts = entry_price - exit_price
    net_pts = pts - ES_SLIPPAGE_PTS * 2
    pnl = net_pts * ES_MULTIPLIER * qty - ES_COMMISSION_RT * qty
    result = {"date": pos.get("date"), "ts": now.isoformat(), "action": "EXIT",
              "exit_type": exit_type, "direction": pos["trade_dir"],
              "entry_price": round(entry_price, 2), "exit_price": round(exit_price, 2),
              "qty": qty, "point_pnl": round(net_pts, 2), "pnl": round(pnl, 2),
              "be": pos["breakeven_activated"], "trail": pos["trailing_activated"]}
    store.append_log(result)
    pos["open"] = False
    state["position"] = pos
    state["in_position"] = False
    store.save_state(state)
    logger.info(f"v10 EXIT [{exit_type}] {pos['trade_dir']} pnl=${pnl:.0f} "
                f"({net_pts:+.2f}pt × {qty})")
    return result


def run_once_monitor(now=None, store=None):
    """One intraday management tick: trail/BE the open position or exit it.

    Mirrors one iteration of the backtest's minute loop via manage_bar(). The
    worker calls this every minute; it is also safe to call from a cron.
    """
    now = now or datetime.now(NY)
    store = store or FileStore()
    state = store.get_state()
    pos = state.get("position")
    if not pos or not pos.get("open"):
        return {"action": "NO_POSITION"}

    # EOD flatten — close at the latest price.
    if now.time() >= EXIT_TIME:
        bar = _latest_bar_es()
        last_px = (bar[0] + bar[1]) / 2 if bar else pos["entry_price"]
        return _close_paper(store, state, pos, last_px, "EOD", now)

    bar = _latest_bar_es()
    if bar is None:
        return {"action": "NO_BAR"}
    high, low = bar
    exit_price, exit_type = manage_bar(pos, high, low)
    if exit_type is not None:
        return _close_paper(store, state, pos, exit_price, exit_type, now)

    state["position"] = pos
    store.save_state(state)
    return {"action": "HOLD", "sl_target": round(pos["sl_target"], 2),
            "best": round(pos["best_price"], 2),
            "be": pos["breakeven_activated"], "trail": pos["trailing_activated"]}


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


def v10_worker(poll_sec=60, store=None):
    """Persistent worker — reproduces the backtest minute-by-minute on live data.

    Run this on an always-on host (Oracle Cloud free VM, Fly.io, a small VPS,
    etc.). It is the most faithful deployment of the v10 strategy because it
    performs true intraday trail/breakeven management via manage_bar(), which a
    2-cron-per-day Vercel schedule cannot do.

    Loop per minute during RTH:
      • 10:30-12:00 ET, flat → run_once_entry (one trade/day, KV-locked)
      • position open       → run_once_monitor (trail/BE, or exit)
      • ≥15:30 ET, open      → EOD flatten via run_once_monitor
    Idles outside RTH. State persists to `store` (use KVStore to share with the
    dashboard/cron, or FileStore for a self-contained box).
    """
    import time as _time
    store = store or FileStore()
    logger.info(f"v10 worker starting (broker={get_broker().name}, poll={poll_sec}s)")
    while True:
        try:
            now = datetime.now(NY)
            t_min = now.hour * 60 + now.minute
            dow = now.weekday()  # 0=Mon … 4=Fri
            state = store.get_state()
            pos = state.get("position") or {}
            has_pos = bool(pos.get("open"))

            if dow < 5 and 570 <= t_min <= 960:        # weekday RTH 09:30-16:00
                if has_pos:
                    r = run_once_monitor(now=now, store=store)
                    if r.get("action") in ("EXIT", "HOLD"):
                        logger.info(f"monitor: {r}")
                elif 10 * 60 <= t_min <= 12 * 60 and state.get("last_trade_date") != now.strftime("%Y-%m-%d"):
                    r = run_once_entry(now=now, store=store)
                    logger.info(f"entry: {r.get('action')}")
            _time.sleep(poll_sec)
        except KeyboardInterrupt:
            logger.info("v10 worker stopped.")
            break
        except Exception as e:
            logger.exception(f"worker loop error: {e}")
            _time.sleep(poll_sec)
