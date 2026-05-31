"""
Walk-Forward Out-of-Sample Validation for MES Futures Backtest

Splits the 3-year Databento MES dataset into:
  - TRAIN window  (2023-03-25 ~ 2023-12-31)   ~9 months
  - TEST window 1 (2024-01-01 ~ 2024-12-31)   12 months — pure OOS
  - TEST window 2 (2025-01-01 ~ 2025-12-31)   12 months — pure OOS
  - TEST window 3 (2026-01-01 ~ 2026-03-25)   ~3 months — most recent OOS

Strategy has no fitted parameters (MIN_SCORE, ATR multiplier, risk %
are all hardcoded), so this is pure OOS validation rather than
parameter retuning. The point is to confirm WR / PF / MDD do NOT
collapse on data the strategy has never been calibrated against.

If TEST metrics degrade > 30% vs TRAIN, the engine is overfit to
2023 conditions and the live system should not use these settings.

Outputs walk_forward_results.json with split-by-split breakdown.
"""
import json
import os
import sys
import argparse

# Reuse the main backtest engine — single source of truth
from thorough_backtest_futures import run_futures_backtest


SPLITS = [
    ("TRAIN",  "2023-03-25", "2023-12-31"),  # in-sample
    ("TEST_2024", "2024-01-01", "2024-12-31"),
    ("TEST_2025", "2025-01-01", "2025-12-31"),
    ("TEST_2026", "2026-01-01", "2026-03-25"),
]


def _pct_change(new, old):
    if old == 0 or old is None:
        return None
    return ((new - old) / abs(old)) * 100


def main(csv_path: str, balance: float):
    print("=" * 80)
    print("  WALK-FORWARD OUT-OF-SAMPLE VALIDATION — MES Futures")
    print("=" * 80)

    results = {}
    for label, start, end in SPLITS:
        print(f"\n>>> Running split: {label} ({start} ~ {end})")
        r = run_futures_backtest(
            csv_path=csv_path,
            start_str=start,
            end_str=end,
            start_balance=balance,
        )
        results[label] = {
            "period": f"{start} ~ {end}",
            "trades": r["total_trades"],
            "wins": r["wins"],
            "losses": r["losses"],
            "win_rate": r["win_rate"],
            "profit_factor": r["profit_factor"],
            "annual_return": r["annual_return"],
            "max_drawdown": r["max_drawdown"],
            "avg_win": r["avg_win"],
            "avg_loss": r["avg_loss"],
            "total_pnl_pct": r["pnl_pct"],
            "end_balance": r["end_balance"],
        }

    # ── Comparison Report ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  WALK-FORWARD COMPARISON")
    print("=" * 80)

    train = results["TRAIN"]
    test_keys = ["TEST_2024", "TEST_2025", "TEST_2026"]

    print(f"\n  {'Metric':<22} {'TRAIN':>10} " +
          " ".join(f"{k.replace('TEST_',''):>10}" for k in test_keys))
    print("  " + "-" * 78)

    for metric, fmt in [
        ("trades", "{:d}"),
        ("win_rate", "{:.1f}%"),
        ("profit_factor", "{:.2f}"),
        ("annual_return", "{:+.1f}%"),
        ("max_drawdown", "{:.1f}%"),
        ("total_pnl_pct", "{:+.1f}%"),
    ]:
        row = f"  {metric:<22} {fmt.format(train[metric]):>10} "
        for k in test_keys:
            row += f"{fmt.format(results[k][metric]):>10} "
        print(row)

    # ── Overfit Detection ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  OVERFIT DETECTION (TEST vs TRAIN degradation)")
    print("=" * 80)

    train_wr = train["win_rate"]
    train_pf = train["profit_factor"]

    overfit_flags = []
    for k in test_keys:
        t = results[k]
        wr_delta = _pct_change(t["win_rate"], train_wr) if train_wr else 0
        pf_delta = _pct_change(t["profit_factor"], train_pf) if train_pf else 0

        flag = ""
        if wr_delta is not None and wr_delta < -30:
            flag += " WR-COLLAPSE"
            overfit_flags.append(f"{k}: WR drop {wr_delta:.1f}%")
        if pf_delta is not None and pf_delta < -30:
            flag += " PF-COLLAPSE"
            overfit_flags.append(f"{k}: PF drop {pf_delta:.1f}%")

        wr_str = f"{wr_delta:+.1f}%" if wr_delta is not None else "n/a"
        pf_str = f"{pf_delta:+.1f}%" if pf_delta is not None else "n/a"
        print(f"  {k:<14} WR {wr_str:>8} | PF {pf_str:>8} {flag}")

    print()
    if overfit_flags:
        print("  [!] OVERFIT WARNING:")
        for f in overfit_flags:
            print(f"    - {f}")
        verdict = "FAIL_OVERFIT"
    else:
        print("  [OK] OOS performance holds - no major degradation detected.")
        verdict = "PASS_OOS"

    # ── Aggregate OOS Period ───────────────────────────────────────
    oos_trades = sum(results[k]["trades"] for k in test_keys)
    oos_wins = sum(results[k]["wins"] for k in test_keys)
    oos_losses = sum(results[k]["losses"] for k in test_keys)
    oos_wr = (oos_wins / oos_trades * 100) if oos_trades else 0
    oos_pnl = sum(results[k]["end_balance"] - balance for k in test_keys)

    print("\n" + "=" * 80)
    print("  AGGREGATE OUT-OF-SAMPLE (2024 + 2025 + 2026)")
    print("=" * 80)
    print(f"  Total trades:    {oos_trades}")
    print(f"  Win rate:        {oos_wr:.1f}% ({oos_wins}W / {oos_losses}L)")
    print(f"  Combined P&L:    ${oos_pnl:+.2f}")
    print(f"  Verdict:         {verdict}")

    # ── Save JSON ──────────────────────────────────────────────────
    output = {
        "model": "MES Walk-Forward OOS Validation",
        "splits": results,
        "aggregate_oos": {
            "trades": oos_trades,
            "wins": oos_wins,
            "losses": oos_losses,
            "win_rate": round(oos_wr, 1),
            "combined_pnl_usd": round(oos_pnl, 2),
        },
        "overfit_flags": overfit_flags,
        "verdict": verdict,
    }
    with open("walk_forward_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[*] Saved: walk_forward_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward OOS backtest")
    parser.add_argument("--csv", type=str, default="MES_1min_data.csv")
    parser.add_argument("--balance", type=float, default=500000.0)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: Could not find CSV file at {args.csv}")
        sys.exit(1)

    main(args.csv, args.balance)
