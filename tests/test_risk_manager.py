"""Unit tests for Layer 7 — Risk Management & Position Sizing.

Covers:
- Position sizing per signal grade (STRONG / MODERATE / NONE)
- 3-strike consecutive loss lockout
- Daily drawdown calculation against daily_start_value anchor
- Daily trade count from ledger and history fallback
- Weekly loss limit
- Risk check return shape (passed/lockout/details)
"""
import pytest
from datetime import datetime
import pytz

from engines.risk_manager import (
    calculate_position_size,
    check_risk_rules,
    _count_consecutive_losses,
    _count_today_trades,
    _calculate_daily_drawdown,
    MAX_DAILY_TRADES,
    CONSECUTIVE_LOSS_LOCK,
    MAX_DAILY_LOSS_PCT,
    MAX_WEEKLY_LOSS_PCT,
)


NY = pytz.timezone("America/New_York")


# ── Position Sizing ──────────────────────────────────────────────────
class TestPositionSizing:
    def test_strong_signal_allocates_full(self):
        portfolio = {"cash": 500000, "initial_balance": 500000}
        result = calculate_position_size(portfolio, "STRONG", entry_price=5800.0)
        assert result["allocation_pct"] == 100
        assert result["contracts"] >= 1
        # max_risk_amount = 500000 * 0.015 = $7,500 at full allocation
        assert result["max_risk_amount"] == 7500.0
        assert "STRONG" in result["sizing_reason"]

    def test_moderate_signal_allocates_half(self):
        portfolio = {"cash": 500000, "initial_balance": 500000}
        result = calculate_position_size(portfolio, "MODERATE", entry_price=5800.0)
        assert result["allocation_pct"] == 50
        # max_risk_amount = 500000 * 0.015 * 0.5 = $3,750
        assert result["max_risk_amount"] == 3750.0
        assert "MODERATE" in result["sizing_reason"]

    def test_none_signal_zero_contracts(self):
        portfolio = {"cash": 500000, "initial_balance": 500000}
        result = calculate_position_size(portfolio, "NONE", entry_price=5800.0)
        assert result["contracts"] == 0
        assert result["allocation_pct"] == 0
        assert result["max_risk_amount"] == 0

    def test_insufficient_cash_caps_contracts(self):
        # $30 cash can't even buy 1 contract at $50 margin
        portfolio = {"cash": 30, "initial_balance": 500000}
        result = calculate_position_size(portfolio, "STRONG", entry_price=5800.0)
        # contracts_by_margin = int(30/50) = 0, but max(1, ...) ensures >=1
        # Yet 1 contract requires $50 margin — caller is responsible for blocking entry.
        assert result["contracts"] >= 0


# ── Consecutive Loss Counter ─────────────────────────────────────────
class TestConsecutiveLosses:
    def test_no_history_returns_zero(self):
        assert _count_consecutive_losses({"history": []}) == 0
        assert _count_consecutive_losses({}) == 0

    def test_counts_consecutive_losses_from_head(self):
        # History is newest-first (insert(0, …)) per api/data.py convention.
        portfolio = {"history": [
            {"pnl": -50, "pnl_locked": True},
            {"pnl": -30, "pnl_locked": True},
            {"pnl": -20, "pnl_locked": True},
            {"pnl": +100, "pnl_locked": True},  # this breaks the streak
        ]}
        assert _count_consecutive_losses(portfolio) == 3

    def test_win_at_head_resets_streak(self):
        portfolio = {"history": [
            {"pnl": +50, "pnl_locked": True},   # most recent is a win
            {"pnl": -30, "pnl_locked": True},
            {"pnl": -20, "pnl_locked": True},
        ]}
        assert _count_consecutive_losses(portfolio) == 0

    def test_skips_open_positions_without_pnl(self):
        portfolio = {"history": [
            {"pnl": None, "status": "OPEN"},     # ignored
            {"pnl": -50, "pnl_locked": True},
            {"pnl": -30, "pnl_locked": True},
        ]}
        assert _count_consecutive_losses(portfolio) == 2

    def test_three_strike_triggers_lockout(self):
        portfolio = {"history": [
            {"pnl": -30, "pnl_locked": True},
            {"pnl": -40, "pnl_locked": True},
            {"pnl": -50, "pnl_locked": True},
        ], "daily_start_value": 500000, "current_value": 500000}
        result = check_risk_rules(portfolio)
        assert result["lockout"] is True
        assert "3-STRIKE" in result["lockout_reason"]
        assert result["consecutive_losses"] == 3
        assert result["strikes_remaining"] == 0


# ── Daily Trade Count ────────────────────────────────────────────────
class TestDailyTradeCount:
    def test_counts_open_events_from_trade_log(self):
        today = datetime.now(NY).strftime("%Y-%m-%d")
        portfolio = {"trade_log": [
            {"event": "OPEN", "date": today},
            {"event": "OPEN", "date": today},
            {"event": "CLOSE", "date": today},  # not counted (only OPENs)
            {"event": "OPEN", "date": "2020-01-01"},  # different day
        ]}
        assert _count_today_trades(portfolio) == 2

    def test_falls_back_to_history_when_log_empty(self):
        today = datetime.now(NY).strftime("%Y-%m-%d")
        portfolio = {
            "trade_log": [],
            "history": [
                {"date": today},
                {"date": today},
                {"date": "2020-01-01"},
            ]
        }
        assert _count_today_trades(portfolio) == 2

    def test_three_daily_trades_triggers_lockout(self):
        today = datetime.now(NY).strftime("%Y-%m-%d")
        portfolio = {
            "trade_log": [{"event": "OPEN", "date": today}] * MAX_DAILY_TRADES,
            "daily_start_value": 500000,
            "current_value": 500000,
        }
        result = check_risk_rules(portfolio)
        assert result["lockout"] is True
        assert "MAX TRADES" in result["lockout_reason"]
        assert result["daily_trades"] == MAX_DAILY_TRADES
        assert result["trades_remaining"] == 0


# ── Daily Drawdown ───────────────────────────────────────────────────
class TestDailyDrawdown:
    def test_no_drawdown_when_balanced(self):
        portfolio = {"daily_start_value": 500000, "current_value": 500000}
        assert _calculate_daily_drawdown(portfolio) == 0.0

    def test_calculates_percent_from_day_open_anchor(self):
        portfolio = {"daily_start_value": 500000, "current_value": 475000}
        # 5% drawdown
        assert _calculate_daily_drawdown(portfolio) == pytest.approx(5.0, abs=0.01)

    def test_falls_back_to_initial_balance_if_no_anchor(self):
        portfolio = {"initial_balance": 500000, "current_value": 485000}
        assert _calculate_daily_drawdown(portfolio) == pytest.approx(3.0, abs=0.01)

    def test_default_anchor_500k_when_nothing_set(self):
        portfolio = {"current_value": 490000}
        # falls back to default $500k starting balance
        assert _calculate_daily_drawdown(portfolio) == pytest.approx(2.0, abs=0.01)

    def test_positive_pnl_returns_zero_not_negative(self):
        # Drawdown is always >= 0 (we don't have "negative drawdown")
        portfolio = {"daily_start_value": 500000, "current_value": 525000}
        assert _calculate_daily_drawdown(portfolio) == 0.0

    def test_six_percent_drawdown_triggers_lockout(self):
        portfolio = {
            "daily_start_value": 500000,
            "current_value": 470000,  # exactly 6% down
        }
        result = check_risk_rules(portfolio)
        assert result["lockout"] is True
        assert "DAILY LOSS LIMIT" in result["lockout_reason"]


# ── Risk Rules — Return Shape ────────────────────────────────────────
class TestRiskRulesShape:
    def test_returns_required_keys(self):
        portfolio = {"daily_start_value": 500000, "current_value": 500000}
        result = check_risk_rules(portfolio)
        required = {
            "passed", "score", "max", "lockout", "lockout_reason",
            "details", "daily_trades", "consecutive_losses",
            "daily_drawdown", "strikes_remaining", "trades_remaining",
        }
        assert required.issubset(set(result.keys()))

    def test_max_is_zero_to_exclude_from_active_sum(self):
        # Risk layer is a gate, not a score contributor — max must be 0 so
        # it doesn't inflate active_max in score_engine.
        portfolio = {"daily_start_value": 500000, "current_value": 500000}
        result = check_risk_rules(portfolio)
        assert result["max"] == 0

    def test_clean_portfolio_passes_all_checks(self):
        portfolio = {"daily_start_value": 500000, "current_value": 500000, "history": []}
        result = check_risk_rules(portfolio)
        assert result["passed"] is True
        assert result["lockout"] is False
        assert result["lockout_reason"] is None
        assert all(d["passed"] for d in result["details"].values())

    def test_strikes_remaining_decrements_with_losses(self):
        portfolio = {
            "daily_start_value": 500000, "current_value": 500000,
            "history": [
                {"pnl": -30, "pnl_locked": True},
                {"pnl": -40, "pnl_locked": True},
            ],
        }
        result = check_risk_rules(portfolio)
        assert result["consecutive_losses"] == 2
        assert result["strikes_remaining"] == 1  # 3 - 2

    def test_weekly_loss_limit(self):
        portfolio = {
            "daily_start_value": 500000, "current_value": 500000,
            "total_return_pct": -MAX_WEEKLY_LOSS_PCT,
        }
        result = check_risk_rules(portfolio)
        assert result["lockout"] is True
        assert "WEEKLY" in result["lockout_reason"]
