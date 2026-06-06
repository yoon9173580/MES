"""
Trade-Shuffling Monte Carlo for MES Futures Backtest (v10.5)

Robustness test that answers: "Was the observed equity curve lucky, and how bad
could the drawdown realistically get if the SAME trades arrived in a different
order?"

Method (the standard serious-retail-quant approach):
  1. Take the realized trade list from the v10.5 backtest.
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
  python3 monte_carlo_backtest.py                       # run v10.5 backtest, then MC
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
    """Run the live v10.5 profile (single 10:30 PRIME entry, score>=65) and
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


def equity_curve(returns, start_balance):
    """Compound a return sequence into a full equity curve including the start
    point (length = len(returns) + 1)."""
    eq = start_balance * np.cumprod(1.0 + returns)
    return np.concatenate(([start_balance], eq))


def equity_curve_and_maxdd(returns, start_balance):
    """Compound a return sequence into an equity curve; return (final_balance,
    max_drawdown_pct over the path including the start point)."""
    equity = equity_curve(returns, start_balance)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    return equity[-1], float(drawdowns.max() * 100.0)


def longest_underwater_trades(equity):
    """Recovery-time metric: the longest stretch (in trades) the equity curve
    spends BELOW a prior peak before making a new high. This is the 'pain
    duration' — how long you'd wait, in trades, to get back to even after the
    worst drawdown of the path.

    Returns (longest_underwater, ended_underwater):
      longest_underwater — max consecutive points strictly below the running max
      ended_underwater   — True if the path never recovered its final peak
                           (the run extends to the last trade = censored)."""
    running_max = np.maximum.accumulate(equity)
    underwater = equity < running_max               # bool per point
    max_run = cur = 0
    for uw in underwater:
        if uw:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    ended_underwater = bool(underwater[-1])
    return max_run, ended_underwater


def monte_carlo(returns, start_balance, iters=10000, method="shuffle", seed=42,
                collect_equity=False):
    """Run the resampling loop. Always returns final_returns, max_dds,
    recovery_trades (longest underwater stretch per path) and never_recovered
    (bool per path). If collect_equity, also returns the full equity matrix
    (iters × n+1) for the fan chart — otherwise that slot is None."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    final_returns = np.empty(iters)
    max_dds = np.empty(iters)
    recovery_trades = np.empty(iters, dtype=int)
    never_recovered = np.zeros(iters, dtype=bool)
    eq_matrix = np.empty((iters, n + 1)) if collect_equity else None
    for i in range(iters):
        if method == "bootstrap":
            sample = rng.choice(returns, size=n, replace=True)
        else:  # shuffle / permutation
            sample = returns[rng.permutation(n)]
        eq = equity_curve(sample, start_balance)
        running_max = np.maximum.accumulate(eq)
        dd = (running_max - eq) / running_max
        final_returns[i] = (eq[-1] - start_balance) / start_balance * 100.0
        max_dds[i] = float(dd.max() * 100.0)
        uw, ended = longest_underwater_trades(eq)
        recovery_trades[i] = uw
        never_recovered[i] = ended
        if collect_equity:
            eq_matrix[i] = eq
    return final_returns, max_dds, recovery_trades, never_recovered, eq_matrix


def pct(arr, q):
    return float(np.percentile(arr, q))


def main():
    ap = argparse.ArgumentParser(description="Trade-shuffling Monte Carlo (v10.5)")
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
        print(f"[*] Running v10.5 backtest on {args.csv} ...")
        trades, start_balance = run_v104_backtest(args.csv, args.balance)
        print(f"[*] Backtest produced {len(trades)} trades")

    returns = trades_to_returns(trades)
    n = len(returns)
    if n < 10:
        print(f"ERROR: only {n} usable trades — too few for Monte Carlo")
        sys.exit(1)

    # ── Observed (actual) path ────────────────────────────────────────
    obs_eq = equity_curve(returns, start_balance)
    obs_final_bal, obs_maxdd = equity_curve_and_maxdd(returns, start_balance)
    obs_return = (obs_final_bal - start_balance) / start_balance * 100.0
    obs_recovery, obs_never = longest_underwater_trades(obs_eq)

    # ── Monte Carlo ───────────────────────────────────────────────────
    print(f"[*] Running {args.iters:,} '{args.method}' iterations over {n} trades ...")
    final_returns, max_dds, recovery_trades, never_recovered, eq_matrix = monte_carlo(
        returns, start_balance, iters=args.iters, method=args.method, seed=args.seed,
        collect_equity=args.plot)
    trades_per_year = n / 3.18   # observed sample spans ~3.18 years

    # Percentile rank of the OBSERVED result inside the simulated distribution
    obs_dd_rank = float((max_dds < obs_maxdd).mean() * 100.0)   # % of sims with smaller DD
    obs_ret_rank = float((final_returns < obs_return).mean() * 100.0)
    prob_loss = float((final_returns < 0).mean() * 100.0)
    prob_dd_gt_15 = float((max_dds > 15).mean() * 100.0)
    prob_dd_gt_20 = float((max_dds > 20).mean() * 100.0)

    # Recovery-time stats (longest underwater stretch, in trades → approx days)
    def trades_to_days(t):
        # ~252 trading days/yr; observed ~trades_per_year trades/yr
        return t * (252.0 / trades_per_year)
    prob_never = float(never_recovered.mean() * 100.0)
    obs_rec_rank = float((recovery_trades < obs_recovery).mean() * 100.0)

    # ── Report ────────────────────────────────────────────────────────
    line = "=" * 74
    print("\n" + line)
    print("  TRADE-SHUFFLING MONTE CARLO — MES v10.5")
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
    print("  RECOVERY TIME — longest underwater stretch (trades → ~days):")
    print(f"    p50 / p75 / p95 / p99 :  "
          f"{pct(recovery_trades,50):.0f} / {pct(recovery_trades,75):.0f} / "
          f"{pct(recovery_trades,95):.0f} / {pct(recovery_trades,99):.0f} trades")
    print(f"                          (~{trades_to_days(pct(recovery_trades,50)):.0f} / "
          f"{trades_to_days(pct(recovery_trades,75)):.0f} / "
          f"{trades_to_days(pct(recovery_trades,95)):.0f} / "
          f"{trades_to_days(pct(recovery_trades,99)):.0f} calendar days)")
    print(f"    worst:           {recovery_trades.max():.0f} trades "
          f"(~{trades_to_days(recovery_trades.max()):.0f} days)")
    print(f"    observed:        {obs_recovery} trades "
          f"(~{trades_to_days(obs_recovery):.0f} days), {obs_rec_rank:.0f}th pct")
    print(f"    P(never recovered by end of sample): {prob_never:.1f}%")
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
        "model": "MES v10.5 trade-shuffling Monte Carlo",
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
        "recovery_trades": {q: int(pct(recovery_trades, q)) for q in (50, 75, 95, 99)},
        "recovery_days_approx": {q: round(trades_to_days(pct(recovery_trades, q)), 0)
                                  for q in (50, 75, 95, 99)},
        "recovery_worst_trades": int(recovery_trades.max()),
        "recovery_observed_trades": int(obs_recovery),
        "prob_never_recovered_pct": round(prob_never, 1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[*] Saved: {args.out}")

    # ── Optional charts ───────────────────────────────────────────────
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(n + 1)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # (A) FAN CHART — percentile bands of equity across all paths ------
        ax = axes[0, 0]
        # eq_matrix: (iters × n+1). Percentiles along the path axis at each step.
        p5, p25, p50, p75, p95 = (np.percentile(eq_matrix, q, axis=0)
                                  for q in (5, 25, 50, 75, 95))
        ax.fill_between(x, p5, p95, color="steelblue", alpha=0.18, label="p5–p95")
        ax.fill_between(x, p25, p75, color="steelblue", alpha=0.35, label="p25–p75")
        ax.plot(x, p50, color="navy", linewidth=1.6, label="median (p50)")
        ax.plot(x, obs_eq, color="crimson", linewidth=2.0, label="Observed (actual)")
        ax.set_title(f"Fan chart — equity percentile bands ({args.iters:,} {args.method} paths)")
        ax.set_xlabel("Trade #"); ax.set_ylabel("Balance ($)")
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)

        # (B) RECOVERY-TIME histogram — longest underwater stretch ---------
        ax = axes[0, 1]
        ax.hist(recovery_trades, bins=40, color="darkseagreen", alpha=0.85)
        ax.axvline(obs_recovery, color="crimson", linewidth=2,
                   label=f"Observed {obs_recovery} tr (~{trades_to_days(obs_recovery):.0f}d)")
        ax.axvline(pct(recovery_trades, 95), color="orange", linestyle="--",
                   label=f"p95 {pct(recovery_trades,95):.0f} tr "
                         f"(~{trades_to_days(pct(recovery_trades,95)):.0f}d)")
        ax.axvline(pct(recovery_trades, 99), color="darkred", linestyle="--",
                   label=f"p99 {pct(recovery_trades,99):.0f} tr")
        ax.set_title("Recovery time — longest underwater stretch (trades)")
        ax.set_xlabel("Trades spent below prior peak"); ax.set_ylabel("Frequency")
        ax.legend(); ax.grid(alpha=0.3)

        # (2) Max-DD histogram --------------------------------------------
        ax = axes[1, 0]
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

        # (3) Total-return histogram --------------------------------------
        ax = axes[1, 1]
        ax.hist(final_returns, bins=60, color="mediumpurple", alpha=0.8)
        ax.axvline(obs_return, color="crimson", linewidth=2,
                   label=f"Observed {obs_return:+.0f}%")
        ax.axvline(pct(final_returns, 5), color="orange", linestyle="--",
                   label=f"p5 {pct(final_returns,5):+.0f}%")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("Total-return distribution")
        ax.set_xlabel("Total return (%)"); ax.set_ylabel("Frequency")
        ax.legend(); ax.grid(alpha=0.3)

        fig.suptitle(f"MES v10.5 — Trade-Shuffling Monte Carlo ({args.method})",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        png = "monte_carlo_results.png"
        fig.savefig(png, dpi=110, bbox_inches="tight")
        print(f"[*] Saved chart: {png}")


if __name__ == "__main__":
    main()
