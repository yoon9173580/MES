# GUN_SPY_MILLI — MES Futures Signal Engine

Pro trader-style signal engine for **MES (Micro E-mini S&P 500 futures)**.
Deployed at https://hannaealgo.vercel.app (Google SSO required).

> Low-frequency, high-conviction strategy. Quality over quantity.
> Goal: catch the few setups per month that pay big, skip the noise.

> 📋 **New here?** Read **[HANDOFF.md](HANDOFF.md)** — a thorough guide to the
> environment, v10.3 strategy, architecture, deployment, and gotchas for anyone
> (human or AI) picking up this project.

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Polygon API     │    │  Alpaca Markets  │    │  Cboe public CDN │
│  (MES Databento) │    │  (stock data)    │    │  (VIX)           │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         └───────────┬───────────┴───────────┬───────────┘
                     │                       │
              ┌──────▼───────────────────────▼──────┐
              │  api/data.py — main orchestrator    │
              │  • parallel data fetch (ThreadPool) │
              │  • cache (Upstash Redis if avail.)  │
              │  • adaptive polling cadence         │
              └──────────────┬──────────────────────┘
                             │
              ┌──────────────▼──────────────────────┐
              │  engines/ — 7-layer scoring system  │
              │   1 Regime (VIX + ADX + ATR)        │
              │   2 Options Flow (NO_DATA on free)  │
              │   3 Correlation (sector sync)       │
              │   4 Time Window (PRIME/GAMMA/lunch) │
              │   5 Technical (VWAP + RSI + EMA)    │
              │   6 Macro Gate (FOMC/CPI/NFP/PPI)   │
              │   7 Risk Manager (3-strike + DD)    │
              └──────────────┬──────────────────────┘
                             │ JSON
                             ▼
                    ┌────────────────┐
                    │  index.html    │
                    │  (Vanilla JS)  │
                    └────────────────┘
```

---

## Strategy

| Parameter           | Value      | Note                                    |
|---------------------|------------|-----------------------------------------|
| Instrument          | MES        | $5 per point (Micro E-mini)             |
| Per-trade risk      | 1.5%       | Kelly-informed                          |
| Daily loss limit    | 6%         | Hard halt                               |
| Weekly loss limit   | 10%        | Hard halt                               |
| Consecutive losses  | 3          | 3-strike lockout                        |
| Max daily trades    | 3          | Quality > quantity                      |
| Entry signal min    | 88 / 120   | Score gate                              |
| Entry window        | 10:30 ET   | After OPEN_CHAOS, before LUNCH_LULL     |
| Exit                | EOD 15:00  | Trail stop + breakeven move             |
| SL distance         | 1.5 × ATR  | Dynamic; bar-close trigger              |

---

## Backtest Results (v10.4 — Real Databento CME Data)

### 3.1-Year Window (2023-03-25 ~ 2026-05-28, $10k acct)

| Metric                  | Value      | Note                        |
|-------------------------|-----------:|-----------------------------|
| Total trades            | 156        | ~49/yr                      |
| Win rate                | 53.2%      |                             |
| Profit factor           | **2.21**   |                             |
| R:R realized            | 1.95       | TP=2.5×SL asymmetry         |
| **Annual return (CAGR)**| **31.6%**  | ⚠️ in-sample optimized + 2.5% risk |
| **Max drawdown**        | **6.0%**   |                             |
| **Sharpe ratio**        | **1.44**   |                             |
| Calmar ratio            | 5.23       |                             |

All 4 calendar years profitable (P&L on $10k acct):
2023 +$2,594 · 2024 +$3,372 · 2025 +$5,835 · 2026 +$1,904

> ⚠️ **Overfitting caveat — read this before trusting the headline.**
> `MIN_SCORE=68` and `SL_CAP=22` were **grid-searched on this same 2023–2026
> dataset**, and the headline uses **2.5% risk-per-trade** (pure leverage). So
> 31.6% / Sharpe 1.44 is an **in-sample-optimized, leveraged** figure — expect
> less live. The walk-forward robustness check (`walk_forward_backtest.py`) is
> the honest picture: every test year stays profitable but **Sharpe degrades
> from 2.24 (2023 train) to ~1.1–1.6 (2024–26)**:
>
> | Split | Trades | WR | Annual | Sharpe | PF |
> |---|--:|--:|--:|--:|--:|
> | 2023 (train) | 32 | 62.5% | 43.8% | 2.24 | 3.97 |
> | 2024 (test)  | 41 | 51.2% | 19.6% | 1.10 | 2.26 |
> | 2025 (test)  | 43 | 58.1% | 28.9% | 1.28 | 2.00 |
> | 2026 (test)  | 16 | 43.8% | 29.1% | 1.56 | 3.56 |
>
> The older, more conservative v10 baseline (MIN_SCORE 88, 1.5% risk, SL_CAP 15)
> measured **8.8% CAGR / Sharpe 0.46 / 34 trades** on the same data — a useful
> lower bound on what's robust vs. tuned.

### v10.x Key Changes

| Change | v10 baseline | v10.4 |
|---|---|---|
| TP target | 1.5×SL | **2.5×SL** |
| ATR filter | none | **ATR > 8 pts/day** |
| ML skip | on (SKIP_N=25) | **off** |
| Entry window | PRIME only | PRIME only (same) |
| Score threshold | 88 | **68** (grid-searched; 74→68 for +18% frequency) |
| SL cap | 15 pt | **22 pt** (grid-searched) |
| Risk per trade | 1.5% | **2.5%** (leverage) |
| VIX threshold | 20 | **25** |

### Bear Market 2022

2022년 Databento 데이터 미보유 (CSV는 2023-03-27부터). 재백테스트를 위해
`MES_1min_data_2022_et_rth.csv` (Databento GLBX.MDP3 ohlcv-1m RTH) 필요.
v10.1 결과 (score≥88): 2거래, 자본 보존. v10.4 (score≥68 + TREND_BEAR)에서는
더 많은 SHORT 진입 예상.

---

## Development

### Setup
```bash
pip install -r requirements.txt   # if requirements.txt exists
pip install pytest pytz requests pandas numpy databento
cp .env.example .env              # add your API keys
```

### Required env vars
```
POLYGON_API_KEY=...          # for backtest data + Alpaca fallback
DATABENTO_API_KEY=...        # for CME MES OHLCV download
APCA_API_KEY_ID=...          # for live stock snapshots
APCA_API_SECRET_KEY=...
FLASHALPHA_API_KEY=...       # optional, for SPY VWAP
GRADE_STRONG=95              # signal grade thresholds (defaults: 95/85/70)
GRADE_MODERATE=85
GRADE_WEAK=70
```

### Run tests
```bash
pytest                        # 108 unit tests, ~10 seconds
```

CI runs the same suite on every push to main (`.github/workflows/test.yml`).

### Pre-push hook (run tests automatically before every push)
A versioned git hook runs `pytest` before each push and aborts if anything fails —
local protection that doesn't depend on GitHub Actions. Enable once per clone:
```bash
git config core.hooksPath .githooks
```
Override a single push (skip tests): `git push --no-verify`.

### Run backtest
```bash
# v10 (recommended default — real CME data, RTH-filtered)
python thorough_backtest_futures.py --csv MES_1min_data_et_rth.csv --balance 500000

# Different balance
python thorough_backtest_futures.py --csv MES_1min_data_et_rth.csv --balance 50000

# Explicit profile selection
python thorough_backtest_futures.py --csv MES_1min_data_et_rth.csv --profile v10
python thorough_backtest_futures.py --csv MES_1min_data_et_rth.csv --profile v4

# Custom tuning flags
python thorough_backtest_futures.py --csv MES_1min_data_et_rth.csv --tp-mult 3.0 --atr-min 10

# Walk-forward OOS validation
python walk_forward_backtest.py --csv MES_1min_data_et_rth.csv

# Download fresh data from Databento
python download_mes_data.py --start 2022-01-03 --end 2026-12-31
```

### Live v10 Paper Bot

The v10 single-tick logic lives in `api/v10_runner.py` (`run_once_entry` /
`run_once_flatten`) and is driven by **two interchangeable schedulers**. Both
share the same code; pick whichever fits your infra.

**v10 entry gate:** RTH + score ≥ 88 + bias LONG/SHORT + regime ATR% ≥ 0.3
("dead-market" guard). SL = 1.5×ATR, **TP = 2.5×SL** (the robust v10 lever).
A one-trade-per-day lock prevents duplicate entries, and every tick appends an
audit record (ENTRY / NO_ENTRY+reasons / NO_DATA / FLATTEN).

#### Option 1 — Vercel Cron (default; no extra infra)

`api/cron_v10.py` is hit on schedule by `vercel.json` `crons`. State + audit log
persist in **Upstash KV** (`v10:state` / `v10:log`), since Vercel functions are
ephemeral.

| Phase   | ET time            | UTC cron      | Notes                                  |
|---------|--------------------|---------------|----------------------------------------|
| Entry   | 11:00 EDT/10:00 EST| `0 15 * * 1-5`| both land in the [10:00, 12:00] window |
| Flatten | 16:30 EDT/15:30 EST| `30 20 * * 1-5`| at/after the 15:35 ET EOD             |

Two crons fits the Vercel Hobby limit. Required env (Vercel project settings):
`UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN` (state), `APCA_API_KEY_ID`
+ `APCA_API_SECRET_KEY` (+ optional `POLYGON_API_KEY`) for market data, and
`CRON_SECRET` to authenticate cron calls. `BROKER` defaults to `dryrun`.

```bash
# Manually trigger a tick (when CRON_SECRET is set)
curl -H "Authorization: Bearer $CRON_SECRET" \
     "https://hannaealgo.vercel.app/api/cron_v10?mode=entry"
```

#### Option 2 — GitHub Actions

`.github/workflows/v10_bot.yml` runs `python trading_bot.py --once entry|flatten`
on dual DST-safe crons, committing `v10_state.json` / `v10_paper_log.json` back
to the repo. (Currently blocked by an account-level Actions billing issue — see
Vercel Cron above as the working path.)

```bash
# Manual single tick (local)
BROKER=dryrun python trading_bot.py --once entry
BROKER=dryrun python trading_bot.py --once flatten
```

**Broker:** `dryrun` logs intended orders (no real fills — safe for signal
validation). For real paper futures fills set `BROKER=tradovate` + `TRADOVATE_*`
creds (demo by default).

> Note: the live bot uses a "dead-market" volatility guard (regime ATR% ≥ 0.3)
> instead of the backtest's exact `ATR>8` filter — that filter removed only 1 of
> 35 backtest trades and is too overfit to trust live. The robust lever
> (TP=2.5×SL) is fully replicated.

---

## File Structure

```
GUN_SPY_MILLI-V2/
├── api/
│   ├── data.py                # main API endpoint (Vercel serverless)
│   ├── lib/auth.py            # Google SSO + CORS + rate limit
│   └── engines/
│       ├── score_engine.py    # grade orchestrator + thresholds
│       ├── regime.py          # Layer 2: VIX/ADX/ATR regime
│       ├── options_flow.py    # Layer 3: gamma exposure (NO_DATA on free)
│       ├── correlation.py     # Layer 4: sector sync (SPY+QQQ+IWM+DIA)
│       ├── time_window.py     # Layer 5: PRIME/GAMMA windows
│       ├── technical.py       # Layer 6: VWAP/RSI/EMA scoring
│       ├── macro_gate.py      # Layer 7: FOMC/CPI/NFP blackouts
│       ├── risk_manager.py    # Layer 8: 3-strike + DD + position sizing
│       └── ml_weights.py      # adaptive feedback weights
├── tests/                     # pytest unit tests (74 tests)
│   ├── test_risk_manager.py
│   ├── test_score_engine.py
│   ├── test_time_window.py
│   └── test_correlation.py
├── thorough_backtest_futures.py   # MES backtest (live params)
├── walk_forward_backtest.py       # OOS validation across years
├── download_mes_data.py           # Databento data downloader
├── index.html                     # frontend (single-file)
└── vercel.json                    # Vercel build + routing config
```

---

## API Response Schema (Important Fields)

```json
{
  "last_updated": "2026-05-25 11:30:36",
  "market_status": "closed",
  "holiday_info": {
    "is_holiday": true,
    "name": "Memorial Day",
    "is_weekend": false,
    "is_closed_day": true
  },
  "total_score": 37,         // normalized 0-100 (gauge value)
  "max_score": 120,          // raw active denominator
  "signal": {
    "grade": "NONE",         // STRONG | MODERATE | WEAK | NONE
    "label": "NO SIGNAL",
    "action": "No entry — conditions insufficient",
    "color": "#f07178"
  },
  "direction_bias": "NEUTRAL",   // LONG | SHORT | NEUTRAL
  "layers": { ... },             // per-layer score breakdown
  "backtest_summary": {
    "mes_futures": { ... },      // real Databento measurement
    "bear_market_2022": { ... }  // real 2022 measurement
  },
  "data_health": {
    "alpaca_snapshots": "OK",
    "vix": "cboe",
    "polygon_fallback_active": false
  },
  "ml_stats": {
    "confidence": "COLD_START",
    "sample_count": 0,
    "weights": { "technical": 1.0, "regime": 1.0, ... }
  }
}
```

---

## Deployment

Pushed to `main` → Vercel auto-deploys.

`vercel.json` uses legacy `version: 2` builds+routes config (security
headers must be inline in each route, NOT at top level — Vercel
silently ignores top-level headers with this config).

---

## Security

- Google SSO via OAuth tokeninfo verification
- `AUTH_BYPASS=1` env var disables auth gate (REMOVE after auditing)
- CORS whitelist for hannaealgo.vercel.app + *.vercel.app + localhost
- IP rate limit: 15 req/min (Upstash KV with in-memory fallback)
- All API keys in `.env` (gitignored) or Vercel env vars

---

## Score Calibration

Score is normalized to 0-100. Grade thresholds (env-overridable):

| Grade    | Default | Action          |
|----------|--------:|-----------------|
| STRONG   | 95+     | Full position   |
| MODERATE | 85+     | Half position   |
| WEAK     | 70+     | Monitor only    |
| NONE     | <70     | No entry        |

Override via env:
```
GRADE_STRONG=90
GRADE_MODERATE=80
GRADE_WEAK=65
```
