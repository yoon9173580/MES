#!/usr/bin/env python3
"""
Walk-Forward Validation: MIN_SCORE Overfitting Test (v10.5)

Data range: 2023–2026 (~3 years, 156 trades at score=65)

Tests:
  1. Year-by-year Sharpe at score=65  → consistency check
  2. Score sweep by year              → IS-optimal score stable across years?
  3. IS/OOS split (IS=2023-24, OOS=2025-26) → IS vs OOS efficiency ratio

Runtime: ~7 min (42 backtest runs × ~10s each)
"""
import sys, os, io, time
from contextlib import redirect_stdout, redirect_stderr
from datetime import time as dtime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thorough_backtest_futures as bt

CSV = "SPY_1min_synthetic.csv"
START_BAL = 10_000.0
SCORES = [62, 64, 65, 66, 68, 70, 74]
YEARS = {
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2026": ("2026-01-01", "2026-12-31"),
}
IS_PERIOD  = ("2023-01-01", "2024-12-31")
OOS_PERIOD = ("2025-01-01", "2026-12-31")


def _setup(score: int):
    bt.ENTRY_WINDOWS     = [dtime(10, 30)]
    bt.MAX_TRADES_PER_DAY = 1
    bt.LOCKOUT_STRIKES   = 3
    bt.LOCKOUT_DAYS      = 1
    bt.WalkForwardML.SKIP_AFTER_N = 9999
    bt.WalkForwardML.SKIP_THRESH  = 0.43
    bt.MIN_SCORE         = score


def run_one(score: int, start: str, end: str) -> dict:
    _setup(score)
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        r = bt.run_futures_backtest(
            csv_path=CSV,
            start_str=start,
            end_str=end,
            start_balance=START_BAL,
            out_path="/tmp/wf_tmp.json",
            atr_min=8.0,
        )
    return {
        "trades":  r.get("total_trades", 0),
        "wr":      r.get("win_rate", 0),
        "cagr":    r.get("annual_return", 0),
        "sharpe":  float(r.get("sharpe_ratio") or 0.0),
        "max_dd":  r.get("max_drawdown", 0),
    }


# ─── Pre-compute all score × year results ────────────────────────────────────
total_runs = len(SCORES) * len(YEARS) + len(SCORES) * 2
run_n = 0
t_total = time.time()

print(f"\nRunning {total_runs} backtest windows …")
print("(each ~10 s; total ≈ 7 min)\n")

# year_data[score][year] = metrics dict
year_data: dict[int, dict[str, dict]] = {}
for score in SCORES:
    year_data[score] = {}
    for yr, (s, e) in YEARS.items():
        run_n += 1
        print(f"  [{run_n:>2}/{total_runs}] score={score} year={yr}  ", end="", flush=True)
        t0 = time.time()
        year_data[score][yr] = run_one(score, s, e)
        print(f"trades={year_data[score][yr]['trades']} sharpe={year_data[score][yr]['sharpe']:.2f}  ({time.time()-t0:.0f}s)")

# IS / OOS
is_data:  dict[int, dict] = {}
oos_data: dict[int, dict] = {}
for score in SCORES:
    run_n += 1
    print(f"  [{run_n:>2}/{total_runs}] score={score} IS=2023-24  ", end="", flush=True)
    t0 = time.time()
    is_data[score] = run_one(score, *IS_PERIOD)
    print(f"trades={is_data[score]['trades']} sharpe={is_data[score]['sharpe']:.2f}  ({time.time()-t0:.0f}s)")

    run_n += 1
    print(f"  [{run_n:>2}/{total_runs}] score={score} OOS=2025-26 ", end="", flush=True)
    t0 = time.time()
    oos_data[score] = run_one(score, *OOS_PERIOD)
    print(f"trades={oos_data[score]['trades']} sharpe={oos_data[score]['sharpe']:.2f}  ({time.time()-t0:.0f}s)")

elapsed = time.time() - t_total
print(f"\nAll runs complete in {elapsed/60:.1f} min\n")


# ─── Report ──────────────────────────────────────────────────────────────────
SEP = "─" * 68

print("\n" + "=" * 68)
print("  TEST 1: Year-by-Year Consistency at MIN_SCORE=65 (v10.5)")
print("=" * 68)
print(f"{'Year':<8} {'Trades':>7} {'WR%':>6} {'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8}")
print(SEP)
sharpe_by_year = []
for yr in YEARS:
    m = year_data[65][yr]
    sharpe_by_year.append(m["sharpe"])
    print(f"{yr:<8} {m['trades']:>7} {m['wr']:>6.1f} {m['cagr']:>8.1f} {m['sharpe']:>8.2f} {m['max_dd']:>8.1f}")

arr = np.array(sharpe_by_year)
mean_s, std_s = float(np.mean(arr)), float(np.std(arr))
cv = (std_s / mean_s * 100) if mean_s > 0 else 999
consistency = "LOW — stable" if cv < 25 else "MEDIUM" if cv < 45 else "HIGH — unstable"
print(f"\n  Sharpe  mean={mean_s:.2f}  std={std_s:.2f}  CV={cv:.0f}%  →  {consistency}")


print("\n" + "=" * 68)
print("  TEST 2: Score Sensitivity — IS-optimal stable across years?")
print("=" * 68)
yr_list = list(YEARS.keys())
hdr = f"{'Score':<7}" + "".join(f"  {y:>6}" for y in yr_list) + f"  {'Avg':>6}"
print(hdr)
print(SEP)

per_year_optimal: dict[str, int] = {yr: 0 for yr in yr_list}
for score in SCORES:
    row = [year_data[score][yr]["sharpe"] for yr in yr_list]
    avg = float(np.mean(row))
    marker = "  ← v10.5" if score == 65 else ""
    print(f"{score:<7}" + "".join(f"  {s:>6.2f}" for s in row) + f"  {avg:>6.2f}{marker}")

print(f"\n  Per-year IS optimal (highest Sharpe):")
for yr in yr_list:
    best = max(SCORES, key=lambda s: year_data[s][yr]["sharpe"])
    print(f"    {yr}: score={best}  (Sharpe {year_data[best][yr]['sharpe']:.2f})")

n_match = sum(1 for yr in yr_list if max(SCORES, key=lambda s: year_data[s][yr]["sharpe"]) == 65)
print(f"\n  score=65 is per-year optimal in {n_match}/{len(yr_list)} years")


print("\n" + "=" * 68)
print("  TEST 3: IS/OOS Split  (IS=2023-2024 | OOS=2025-2026)")
print("=" * 68)
print(f"{'Score':<7} {'IS Sh':>8} {'IS Tr':>6}  {'OOS Sh':>8} {'OOS Tr':>7}  {'Ratio':>6}")
print(SEP)
for score in SCORES:
    i = is_data[score]
    o = oos_data[score]
    ratio = (o["sharpe"] / i["sharpe"]) if i["sharpe"] > 0 else 0.0
    marker = "  ← v10.5" if score == 65 else ""
    print(f"{score:<7} {i['sharpe']:>8.2f} {i['trades']:>6}  {o['sharpe']:>8.2f} {o['trades']:>7}  {ratio:>6.2f}{marker}")

is_opt = max(SCORES, key=lambda s: is_data[s]["sharpe"])
oos_at_is_opt = oos_data[is_opt]["sharpe"]
oos_at_65    = oos_data[65]["sharpe"]
is_at_65     = is_data[65]["sharpe"]
ratio_65     = oos_at_65  / is_at_65     if is_at_65               > 0 else 0
ratio_opt    = oos_at_is_opt / is_data[is_opt]["sharpe"] if is_data[is_opt]["sharpe"] > 0 else 0

print(f"\n  IS-optimal: score={is_opt}  (IS Sharpe {is_data[is_opt]['sharpe']:.2f})")
print(f"  OOS at IS-optimal ({is_opt}): Sharpe {oos_at_is_opt:.2f}  (ratio {ratio_opt:.2f})")
print(f"  v10.5 score=65: IS={is_at_65:.2f} → OOS={oos_at_65:.2f}  (ratio {ratio_65:.2f})")


# ─── Overall verdict ─────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  VERDICT")
print("=" * 68)

flags = []
if cv > 35:
    flags.append(f"year-to-year Sharpe CV={cv:.0f}% (>35% threshold)")
if is_opt != 65:
    flags.append(f"IS-optimal score={is_opt} ≠ 65 (score not globally optimal in-sample)")
if ratio_65 < 0.70:
    flags.append(f"OOS/IS ratio={ratio_65:.2f} (<0.70 threshold)")

if len(flags) == 0:
    level = "LOW"
    note  = "score=65 is robust — consistent across years and IS/OOS split."
elif len(flags) == 1:
    level = "MODERATE"
    note  = "some data dependency, but within acceptable range for this sample size."
else:
    level = "HIGH"
    note  = "likely overfit — performance may degrade out-of-sample."

print(f"\n  Overfitting risk: {level}")
print(f"  {note}")
if flags:
    print(f"\n  Warning signals:")
    for f in flags:
        print(f"    • {f}")
print(f"\n  Data note: only 3 years (2023-2026) with {sum(year_data[65][y]['trades'] for y in yr_list)} trades.")
print(f"  Extending to 10+ years would substantially lower uncertainty.\n")
