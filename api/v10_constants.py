"""
v10 Single Source of Truth — Production MES Futures Strategy Parameters.

This module is the authoritative definition for the live bot (cron + runner)
and the primary backtest (thorough_backtest_futures.py).

Dashboard / rich scoring (score_engine) uses a superset of layers for UI visibility
but should reference these values for core gates (MIN_SCORE, ATR_MIN, etc.) where
they overlap with the production entry decision.

ML classifier hard-skip (P(win) threshold) is implemented inside backtest walk-forward
only. Live uses adaptive layer weights (ml_weights.py) + the gates below.
"""

from datetime import time as dtime

# =============================================================================
# MES / ES Contract Specifications (shared)
# =============================================================================
ES_MULTIPLIER = 5.0        # $5 per point (Micro E-mini S&P 500)
ES_COMMISSION_RT = 0.50    # Round-trip per contract
ES_SLIPPAGE_PTS = 0.25     # 1 tick
ES_DAY_MARGIN = 50.0       # Day trading margin per contract (MES)
ES_TICK_SIZE = 0.25
ES_PER_SPY = 10.0          # Live approximates ES via SPY * 10

# =============================================================================
# v10 Core Decision Parameters (live + matching backtest)
# =============================================================================
ATR_SL_MULT = 1.5
TP_MULT = 2.5               # Key v10 lever: 2.5x SL
ATR_MIN = 8.0               # Dead market guard
MIN_SCORE = 65              # Entry threshold (after boosts). ML hard-skip (P(win)) is backtest validation only.
SL_CAP_PTS = 22.0
SL_MIN_PTS = 2.0

# VIX regime & sizing
VIX_THRESHOLD = 25.0        # <25 trend-follow, else mean-reversion (with crisis override)
VIX_SHORT_FILTER = 20.0     # Skip SHORT when daily bias bull + VIX low
VIX_CRISIS = 30.0
VIX_SIZE_25 = 25.0
VIX_SIZE_35 = 35.0

RISK_PCT_FULL = 0.025       # VIX < 25
RISK_PCT_BEAR = 0.010
RISK_PCT_CRISIS = 0.007

# Filters / vetoes
RSI_UPPER = 90.0
RSI_LOWER = 10.0
ADX_RUNAWAY = 40.0
SECTOR_THRESHOLD = 1.8

# Boosts (applied before gate)
NR7_SCORE_BOOST = 5
PULLBACK_SCORE_BOOST = 5

# Timing (ET)
ENTRY_TIME = dtime(10, 30)  # PRIME bar evaluation
EXIT_TIME = dtime(15, 30)   # EOD flatten

# Intraday management (manage_bar)
TRAILING_ACTIVATION = 0.5
TRAILING_STEP = 0.25
BREAKEVEN_AT = 0.25

# =============================================================================
# ML (classifier hard-skip is backtest-only)
# =============================================================================
# WalkForwardML P(win) hard filter improved backtest numbers.
# Live deliberately does not load sklearn/lightgbm models.
ML_SKIP_AFTER_N = 30
ML_SKIP_THRESH = 0.35

# =============================================================================
# Risk (dashboard simulation layer uses its own; live bot uses vix_risk_pct + above)
# =============================================================================
# Kept here for reference / possible future unification.
MAX_DAILY_LOSS_PCT = 6.0
MAX_WEEKLY_LOSS_PCT = 10.0
CONSECUTIVE_LOSS_LOCK = 3
MAX_DAILY_TRADES = 3
