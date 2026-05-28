"""Regression tests for modules added in the recent upgrade batch:
macro_gate, options_flow, futures_meta, brokers, feature_flags, health.
"""
from datetime import datetime
import pytz

NY = pytz.timezone("America/New_York")


# ── futures_meta ─────────────────────────────────────────────────
class TestFuturesMeta:
    def test_third_friday(self):
        from lib.futures_meta import third_friday
        # March 2026: 1st=Sun, first Fri=6, third=20
        assert third_friday(2026, 3) == 20
        # June 2026: 1st=Mon, first Fri=5, third=19
        assert third_friday(2026, 6) == 19

    def test_quarterly_roll_window(self):
        from lib.futures_meta import is_quarterly_roll_window
        # 3rd Friday March 2026 = 20. Roll window = Tue/Wed/Thu = 17,18,19
        assert is_quarterly_roll_window(datetime(2026, 3, 18))
        assert is_quarterly_roll_window(datetime(2026, 3, 19))
        assert not is_quarterly_roll_window(datetime(2026, 3, 20))  # 3rd Fri itself
        assert not is_quarterly_roll_window(datetime(2026, 3, 16))  # Mon before
        assert not is_quarterly_roll_window(datetime(2026, 4, 10))  # non-quarter month

    def test_current_mes_contract(self):
        from lib.futures_meta import current_mes_contract
        # Before March 2026 expiry: front month = MESH26
        assert current_mes_contract(datetime(2026, 3, 10)) == "MESH26"
        # In roll window of March 2026: front month rolls to MESM26
        assert current_mes_contract(datetime(2026, 3, 18)) == "MESM26"
        # December 2026 roll → MESH27
        assert current_mes_contract(datetime(2026, 12, 16)) == "MESH27"


# ── macro_gate ───────────────────────────────────────────────────
class TestMacroGate:
    def test_fomc_release_window(self):
        from engines.macro_gate import calculate_macro_gate
        # FOMC 2026-01-28 14:00 ET. Expect BLOCKED 30min before & after.
        dt_before = NY.localize(datetime(2026, 1, 28, 13, 30))
        dt_after = NY.localize(datetime(2026, 1, 28, 14, 30))
        assert calculate_macro_gate(dt_before)["status"] == "BLOCKED"
        assert calculate_macro_gate(dt_after)["status"] == "BLOCKED"
        # 4h after FOMC → CLEAR (windows ±3h)
        dt_clear = NY.localize(datetime(2026, 1, 28, 18, 30))
        assert calculate_macro_gate(dt_clear)["status"] == "CLEAR"

    def test_random_weekday_clear(self):
        from engines.macro_gate import calculate_macro_gate
        # Tuesday 2026-05-26 10:30 — no major release
        dt = NY.localize(datetime(2026, 5, 26, 10, 30))
        r = calculate_macro_gate(dt)
        assert r["gate_passed"] is True
        assert r["status"] in ("CLEAR", "WARNING")

    def test_nfp_first_friday(self):
        """NFP releases first Friday at 8:30 ET. First Fri of June 2026 = 5th."""
        from engines.macro_gate import calculate_macro_gate
        dt = NY.localize(datetime(2026, 6, 5, 8, 30))
        r = calculate_macro_gate(dt)
        assert r["status"] == "BLOCKED"
        assert r["active_event"] == "NFP"


# ── options_flow ─────────────────────────────────────────────────
class TestOptionsFlow:
    def test_strong_call_demand_returns_long(self):
        from engines.options_flow import _score_options_flow
        m = {"pc_vol_ratio": 0.5, "pc_oi_ratio": 0.8,
             "unusual_count": 2, "unusual_top": []}
        score, direction, _ = _score_options_flow(m)
        assert direction == "LONG"
        assert score >= 20

    def test_strong_put_demand_returns_short(self):
        from engines.options_flow import _score_options_flow
        m = {"pc_vol_ratio": 1.5, "pc_oi_ratio": 1.2,
             "unusual_count": 0, "unusual_top": []}
        score, direction, _ = _score_options_flow(m)
        assert direction == "SHORT"
        assert score >= 15

    def test_neutral_no_score(self):
        from engines.options_flow import _score_options_flow
        m = {"pc_vol_ratio": 1.0, "pc_oi_ratio": 1.0,
             "unusual_count": 0, "unusual_top": []}
        score, direction, _ = _score_options_flow(m)
        assert direction == "NEUTRAL"
        assert score == 0

    def test_no_data_safe_default(self):
        from engines.options_flow import _score_options_flow
        score, direction, detail = _score_options_flow(None)
        assert score == 0
        assert direction == "NEUTRAL"


# ── brokers ──────────────────────────────────────────────────────
class TestBrokers:
    def test_dryrun_always_ready(self):
        from lib.brokers import DryRunAdapter
        b = DryRunAdapter()
        ok, _ = b.is_ready()
        assert ok is True
        assert b.supports_futures and b.supports_equity

    def test_dryrun_places_order_locally(self):
        from lib.brokers import DryRunAdapter
        b = DryRunAdapter()
        res = b.place_bracket_order("MESM26", 1, "buy", 5005, 4995)
        assert res is not None
        assert res["symbol"] == "MESM26"
        assert res["mode"] == "DRYRUN"

    def test_alpaca_reports_missing_keys(self, monkeypatch):
        from lib.brokers import AlpacaPaperAdapter
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
        b = AlpacaPaperAdapter()
        ok, reason = b.is_ready()
        assert ok is False
        assert "ALPACA" in reason

    def test_factory_returns_dryrun_for_unknown(self):
        from lib.brokers import get_broker
        b = get_broker("nonexistent")
        assert b.name == "dryrun"


# ── feature_flags ────────────────────────────────────────────────
class TestFeatureFlags:
    def test_default_enabled(self, monkeypatch):
        from lib.feature_flags import is_enabled
        # Clear any env override
        monkeypatch.delenv("FF_VIX_CAP_25", raising=False)
        assert is_enabled("vix_cap_25") is True

    def test_env_override_disables(self, monkeypatch):
        from lib.feature_flags import is_enabled
        monkeypatch.setenv("FF_VIX_CAP_25", "false")
        assert is_enabled("vix_cap_25") is False

    def test_unknown_flag_returns_false(self):
        from lib.feature_flags import is_enabled
        assert is_enabled("totally_made_up_flag") is False

    def test_all_flags_snapshot(self):
        from lib.feature_flags import all_flags, DEFAULTS
        snap = all_flags()
        assert isinstance(snap, dict)
        assert set(snap.keys()) == set(DEFAULTS.keys())


# ── health (log buffer) ──────────────────────────────────────────
class TestHealth:
    def test_log_and_snapshot(self):
        from lib.health import log_error, log_warn, snapshot
        # Add a couple of entries
        log_error("test_source", "synthetic error")
        log_warn("test_source", "synthetic warning")
        snap = snapshot(limit=10)
        assert snap["counters"]["error"] >= 1
        assert snap["counters"]["warn"] >= 1
        # Recent list should contain our messages
        recent_messages = [e["message"] for e in snap["recent"]]
        assert any("synthetic error" in m for m in recent_messages)
        assert any("synthetic warning" in m for m in recent_messages)

    def test_uptime_increases(self):
        import time
        from lib.health import snapshot
        s1 = snapshot()
        time.sleep(0.01)
        s2 = snapshot()
        assert s2["process_uptime_sec"] >= s1["process_uptime_sec"]


# ── ml_weights enhanced update path ──────────────────────────────
class TestMLWeights:
    def test_magnitude_aware_update(self):
        from engines.ml_weights import AdaptiveWeightEngine
        eng = AdaptiveWeightEngine()
        eng.weights["technical"] = 1.0
        # +2R win should move 2× a +1R win
        eng.update_weights({"pnl": 100, "dominant_layer": "technical", "magnitude_r": 1.0})
        after_1r = eng.weights["technical"]
        eng.weights["technical"] = 1.0
        eng.update_weights({"pnl": 200, "dominant_layer": "technical", "magnitude_r": 2.0})
        after_2r = eng.weights["technical"]
        # Both move up. 2R should move more.
        assert after_1r > 1.0
        assert after_2r > after_1r

    def test_decay_pulls_toward_one(self):
        from engines.ml_weights import AdaptiveWeightEngine
        eng = AdaptiveWeightEngine()
        eng.weights["regime"] = 1.5  # max
        # A neutral feedback (pnl=0) still decays
        eng.update_weights({"pnl": 0, "dominant_layer": "regime", "magnitude_r": 1.0})
        # Decay alpha=0.995, so 1.5 → 1.0 + 0.5*0.995 = 1.4975
        assert eng.weights["regime"] < 1.5
        assert eng.weights["regime"] > 1.0
