# GUN_SPY_MILLI — MES Futures Signal Engine

Pro trader-style signal engine for **MES (Micro E-mini S&P 500 futures)**.
Deployed at https://hannaealgo.vercel.app (Google SSO required).

> Low-frequency, high-conviction strategy. Quality over quantity.
> Goal: catch the few setups per month that pay big, skip the noise.

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

## Backtest Results (Live Params — Real Databento CME Data)

### 3-Year Forward Window (2023-03-25 ~ 2026-03-25)

| Metric                  | Value      |
|-------------------------|-----------:|
| Total trades            | 55         |
| Win rate                | 60.0%      |
| Profit factor           | 2.58       |
| R:R realized            | 1.72       |
| **Annual return (CAGR)**| **12.6%**  |
| **Max drawdown**        | **3.6%**   |
| Avg win                 | $212       |
| Avg loss                | -$124      |

### Walk-Forward OOS Validation

| Split  | Trades | WR    | PF    | P&L     |
|--------|-------:|------:|------:|--------:|
| TRAIN 2023 | 11 | 63.6% | 3.91  | +8.5%   |
| TEST  2024 | 16 | 56.2% | 2.33  | +7.4%   |
| TEST  2025 | 22 | 59.1% | 3.31  | +15.4%  |
| TEST  2026 | 5  | 40.0% | 1.06  | +0.1%   |

Aggregate OOS: 43 trades, 55.8% WR, +$2,289 P&L. Strategy holds.

### Bear Market 2022 (Real Data)

| Metric                  | Value      |
|-------------------------|-----------:|
| Total trades            | 2          |
| Annual return           | +0.3%      |
| Max drawdown            | 1.3%       |
| **Verdict**             | **DORMANT — defensive design worked** |

Filters blocked entry on 306/308 days. Capital preservation worked but
no alpha capture — would need a separate counter-trend mode to capture
bear-market mean reversion.

### Scenario Comparison (Same Strategy, Different Starting Balance)

| Balance | End      | Net      | Annual | Avg Contracts |
|--------:|---------:|---------:|-------:|--------------:|
| $10k    | $14,293  | +$4,293  | 12.6%  | 1.9           |
| $50k    | $77,112  | +$27,112 | 15.5%  | 11.7          |
| $500k   | $783,239 | +$283k   | 16.1%  | 123.2         |

MDD stays at 3.6-3.7% across all scales. $50k = best efficiency.

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
# Single run on full 3yr dataset (default $10k balance)
python thorough_backtest_futures.py --csv MES_1min_data.csv

# Different balance
python thorough_backtest_futures.py --csv MES_1min_data.csv --balance 50000

# Walk-forward OOS validation
python walk_forward_backtest.py --csv MES_1min_data.csv

# Download fresh data from Databento
python download_mes_data.py --start 2022-01-03 --end 2026-12-31
```

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
