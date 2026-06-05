"""
Trade-Shuffling Monte Carlo for MES Futures Backtest (v10.4)

Robustness test that answers: "Was the observed equity curve lucky, and how bad
could the drawdown realistically get if the SAME trades arrived in a different
order?"

Method (the standard serious-retail-quant approach):
  1. Take the realized trade list from the v10.4 backtest.
  2. Convert each trade to a RETURN on equity (pnl / balance_before_trade) so the
     shuffle is position-sizing-correct (a +$500 win on a $10k balance is a
     different *return* than the same $500 on a $20k balance — shuffling raw
     dollars would corrupt that; shuffling returns does not).
  3. Resample the trade ORDER N times (default 10,000):
       - "shuffle"   = permutation, no replacement (path-dependency test:
                       identical trade set, different sequence).
       - "bootstrap" = sampling WITH replacement (also tests "what if the edge
                       were slightly different" — wider, more pessimistic).
  4. Compound each path from the start balance, record final return and the
     path's max drawdown.
  5. Report the DISTRIBUTION of outcomes and locate the ACTUAL observed result
     inside it (percentile rank).

Interpretation:
  - If the observed Max DD sits in the LOW percentiles of the distribution, the
    live system got a *lucky* ordering — plan for the p95/p99 DD, not the
    backtest's 6%.
  - A fat right tail on Max DD (p99 >> observed) is the real risk-of-ruin signal
    for a low-frequency strategy like this (~49 trades/yr).

Usage:
  python3 monte_carlo_backtest.py                       # run v10.4 backtest, then MC
  python3 monte_carlo_backtest.py --from-json backtest_futures.json
  python3 monte_carlo_backtest.py --iters 20000 --method bootstrap --plot
"""
import argparse
import json
import os
import sys

import numpy as np


def load_trades_from_json(path):
    with open(path) as f:
        data = json.load(f)
    return data["trades"], data.get("start_balance", 10000.0)


def run_v104_backtest(csv_path, start_balance):
    """Run the live v10.4 profile (single 10:30 PRIME entry, score>=68) and
    return its trade list — single source of truth with the live bot."""
    from datetime import time as dtime
    import thorough_backtest_futures as _bt

    _bt.MIN_SCORE = 68
    _bt.MAX_TRADES_PER_DAY = 1
    _bt.ENTRY_WINDOWS = [dtime(10, 30)]
    _bt.WalkForwardML.SKIP_AFTER_N = 9999
    r = _bt.run_futures_backtest(
        csv_path=csv_path, start_str="2023-03-25", end_str=None,
        start_balance=start_balance, out_path="backtest_futures.json",
    )
    return r["trades"], r["start_balance"]


def trades_to_returns(trades):
    """Convert each trade to a fractional return on the equity it was sized
    against: ret_i = pnl_i / balance_before_i, where balance_before = balance - pnl."""
    rets = []
    for t in trades:
        pnl = float(t["pnl"])
        bal_after = float(t["balance"])
        bal_before = bal_after - pnl
        if bal_before <= 0:
            continue
        rets.append(pnl / bal_before)
    return np.asarray(rets, dtype=float)


def equity_curve_and_maxdd(returns, start_balance):
    """Compound a return sequence into an equity curve; return (final_balance,
    max_drawdown_pct over the path including the start point)."""
    equity = start_balance * np.cumprod(1.0 + returns)
    equity = np.concatenate(([start_balance], equity))
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    return equity[-1], float(drawdowns.max() * 100.0)


def monte_carlo(returns, start_balance, iters=10000, method="shuffle", seed=42):
    rng = np.random.default_rng(seed)
    n = len(returns)
    final_returns = np.empty(iters)
    max_dds = np.empty(iters)
    for i in range(iters):
        if method == "bootstrap":
            sample = rng.choice(returns, size=n, replace=True)
        else:  # shuffle / permutation
            sample = returns[rng.permutation(n)]
        final_bal, mdd = equity_curve_and_maxdd(sample, start_balance)
        final_returns[i] = (final_bal - start_balance) / start_balance * 100.0
        max_dds[i] = mdd
    return final_returns, max_dds


def pct(arr, q):
    return float(np.percentile(arr, q))


def main():
    ap = argparse.ArgumentParser(description="Trade-shuffling Monte Carlo (v10.4)")
    ap.add_argument("--csv", default="MES_1min_data_et_rth.csv")
    ap.add_argument("--from-json", default=None,
                    help="Skip backtest; load trades from this results JSON")
    ap.add_argument("--balance", type=float, default=10000.0)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--method", choices=["shuffle", "bootstrap"], default="shuffle")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true", help="Save PNG charts")
    ap.add_argument("--out", default="monte_carlo_results.json")
    args = ap.parse_args()

    # ── Get trades ────────────────────────────────────────────────────
    if args.from_json:
        if not os.path.exists(args.from_json):
            print(f"ERROR: {args.from_json} not found")
            sys.exit(1)
        trades, start_balance = load_trades_from_json(args.from_json)
        print(f"[*] Loaded {len(trades)} trades from {args.from_json}")
    else:
        if not os.path.exists(args.csv):
            print(f"ERROR: {args.csv} not found")
            sys.exit(1)
        print(f"[*] Running v10.4 backtest on {args.csv} ...")
        trades, start_balance = run_v104_backtest(args.csv, args.balance)
        print(f"[*] Backtest produced {len(trades)} trades")

    returns = trades_to_returns(trades)
    n = len(returns)
    if n < 10:
        print(f"ERROR: only {n} usable trades — too few for Monte Carlo")
        sys.exit(1)

    # ── Observed (actual) path ────────────────────────────────────────
    obs_final_bal, obs_maxdd = equity_curve_and_maxdd(returns, start_balance)
    obs_return = (obs_final_bal - start_balance) / start_balance * 100.0

    # ── Monte Carlo ───────────────────────────────────────────────────
    print(f"[*] Running {args.iters:,} '{args.method}' iterations over {n} trades ...")
    final_returns, max_dds = monte_carlo(
        returns, start_balance, iters=args.iters, method=args.method, seed=args.seed)

    # Percentile rank of the OBSERVED result inside the simulated distribution
    obs_dd_rank = float((max_dds < obs_maxdd).mean() * 100.0)   # % of sims with smaller DD
    obs_ret_rank = float((final_returns < obs_return).mean() * 100.0)
    prob_loss = float((final_returns < 0).mean() * 100.0)
    prob_dd_gt_15 = float((max_dds > 15).mean() * 100.0)
    prob_dd_gt_20 = float((max_dds > 20).mean() * 100.0)

    # ── Report ────────────────────────────────────────────────────────
    line = "=" * 74
    print("\n" + line)
    print("  TRADE-SHUFFLING MONTE CARLO — MES v10.4")
    print(line)
    print(f"  Trades:            {n}   (~{n/3.18:.0f}/yr)")
    print(f"  Iterations:        {args.iters:,}  ({args.method})")
    print(f"  Start balance:     ${start_balance:,.0f}")
    print("  " + "-" * 70)
    print("  OBSERVED (actual backtest order):")
    print(f"    Total return:    {obs_return:+.1f}%")
    print(f"    Max drawdown:    {obs_maxdd:.1f}%")
    print("  " + "-" * 70)
    print("  SIMULATED TOTAL RETURN distribution:")
    print(f"    p5  / p25 / p50 / p75 / p95 :  "
          f"{pct(final_returns,5):+.0f}% / {pct(final_returns,25):+.0f}% / "
          f"{pct(final_returns,50):+.0f}% / {pct(final_returns,75):+.0f}% / "
          f"{pct(final_returns,95):+.0f}%")
    print(f"    mean:            {final_returns.mean():+.1f}%")
    print(f"    P(total loss < 0):           {prob_loss:.2f}%")
    print("  " + "-" * 70)
    print("  SIMULATED MAX DRAWDOWN distribution (the key robustness output):")
    print(f"    p5  / p25 / p50 / p75 / p95 :  "
          f"{pct(max_dds,5):.1f}% / {pct(max_dds,25):.1f}% / "
          f"{pct(max_dds,50):.1f}% / {pct(max_dds,75):.1f}% / {pct(max_dds,95):.1f}%")
    print(f"    p99:             {pct(max_dds,99):.1f}%")
    print(f"    worst seen:      {max_dds.max():.1f}%")
    print(f"    P(MaxDD > 15%):  {prob_dd_gt_15:.1f}%")
    print(f"    P(MaxDD > 20%):  {prob_dd_gt_20:.1f}%")
    print("  " + "-" * 70)
    print("  WHERE THE OBSERVED RESULT SITS:")
    print(f"    Observed MaxDD {obs_maxdd:.1f}% is at the {obs_dd_rank:.0f}th percentile "
          f"of simulated DDs")
    if obs_dd_rank < 25:
        print("      → LUCKY ordering: live DD will very likely exceed the backtest's.")
    elif obs_dd_rank > 75:
        print("      → UNLUCKY ordering: backtest DD overstates typical risk.")
    else:
        print("      → TYPICAL ordering: backtest DD is representative.")
    print(f"    Observed return {obs_return:+.0f}% is at the {obs_ret_rank:.0f}th percentile")
    print(line)

    out = {
        "model": "MES v10.4 trade-shuffling Monte Carlo",
        "method": args.method,
        "iterations": args.iters,
        "trades": n,
        "start_balance": start_balance,
        "observed": {"total_return_pct": round(obs_return, 1),
                     "max_drawdown_pct": round(obs_maxdd, 1),
                     "maxdd_percentile_rank": round(obs_dd_rank, 1),
                     "return_percentile_rank": round(obs_ret_rank, 1)},
        "sim_total_return_pct": {q: round(pct(final_returns, q), 1)
                                  for q in (5, 25, 50, 75, 95)},
        "sim_total_return_mean": round(float(final_returns.mean()), 1),
        "prob_total_loss_pct": round(prob_loss, 2),
        "sim_max_drawdown_pct": {q: round(pct(max_dds, q), 1)
                                  for q in (5, 25, 50, 75, 95, 99)},
        "sim_max_drawdown_worst": round(float(max_dds.max()), 1),
        "prob_maxdd_gt_15_pct": round(prob_dd_gt_15, 1),
        "prob_maxdd_gt_20_pct": round(prob_dd_gt_20, 1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[*] Saved: {args.out}")

    # ── Optional charts ───────────────────────────────────────────────
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(args.seed)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # (1) Equity-curve fan: 200 sample paths + observed
        ax = axes[0]
        for _ in range(200):
            sample = (rng.choice(returns, size=n, replace=True)
                      if args.method == "bootstrap"
                      else returns[rng.permutation(n)])
            eq = start_balance * np.cumprod(1.0 + sample)
            eq = np.concatenate(([start_balance], eq))
            ax.plot(eq, color="steelblue", alpha=0.06, linewidth=0.8)
        obs_eq = start_balance * np.cumprod(1.0 + returns)
        obs_eq = np.concatenate(([start_balance], obs_eq))
        ax.plot(obs_eq, color="crimson", linewidth=2.0, label="Observed (actual)")
        ax.set_title(f"Equity-curve fan — {args.iters:,} {args.method} paths (200 shown)")
        ax.set_xlabel("Trade #"); ax.set_ylabel("Balance ($)")
        ax.legend(); ax.grid(alpha=0.3)

        # (2) Max-DD histogram with observed + p95/p99 markers
        ax = axes[1]
        ax.hist(max_dds, bins=60, color="slategray", alpha=0.8)
        ax.axvline(obs_maxdd, color="crimson", linewidth=2,
                   label=f"Observed {obs_maxdd:.1f}%")
        ax.axvline(pct(max_dds, 95), color="orange", linestyle="--",
                   label=f"p95 {pct(max_dds,95):.1f}%")
        ax.axvline(pct(max_dds, 99), color="darkred", linestyle="--",
                   label=f"p99 {pct(max_dds,99):.1f}%")
        ax.set_title("Max-drawdown distribution")
        ax.set_xlabel("Max drawdown (%)"); ax.set_ylabel("Frequency")
        ax.legend(); ax.grid(alpha=0.3)

        fig.suptitle("MES v10.4 — Trade-Shuffling Monte Carlo", fontsize=13, fontweight="bold")
        fig.tight_layout()
        png = "monte_carlo_results.png"
        fig.savefig(png, dpi=110, bbox_inches="tight")
        print(f"[*] Saved chart: {png}")


if __name__ == "__main__":
    main()
