"""Stage 7: live target portfolio + charts + the self-contained HTML report (reports/battle_ready_report.html)."""
import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.battle.context import CORE3, build_context
from src.battle.data import BATTLE_DIR
from src.battle.engine import metrics, run_backtest, signal_days
from src.battle.report_text import render
from src.config import REPORTS_DIR

BLUE, RED, GREY, GREEN = "#4c78a8", "#e45756", "#8c8c8c", "#54a24b"


def img(fig, caption):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=125, bbox_inches="tight")
    plt.close(fig)
    b = base64.b64encode(buf.getvalue()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def target_portfolio(ctx) -> pd.DataFrame:
    mkt = ctx.mkt
    a, b = ctx.oos
    r = run_backtest(mkt, ctx.signal("core3", "Q"), "Q", a, b, record=True)
    last_date, held = r.holdings[-1]
    score = ctx.signal("core3", "Q")[-1]
    ok = ctx.elig[-1] & ~np.isnan(score)
    rank = pd.Series(np.where(ok, score, np.nan), index=mkt.symbols).rank(ascending=False)
    rows = []
    for sym in held:
        i = mkt.symbols.index(sym)
        row = {"Symbol": sym, "Weight": 1 / len(held), "HeldSince": str(last_date.date()),
               "ReturnSinceEntry": mkt.close[-1, i] / mkt.open[mkt.dates.get_loc(last_date), i] - 1,
               "CurrentRank_core3": rank.get(sym), "LastClose": mkt.close[-1, i]}
        for s in CORE3:
            sc = ctx.signal(s, "Q")[-1]
            r_ = pd.Series(np.where(ctx.elig[-1] & ~np.isnan(sc), sc, np.nan), index=mkt.symbols).rank(ascending=False)
            row[f"Rank_{s}"] = r_.get(sym)
        rows.append(row)
    cur = pd.DataFrame(rows).sort_values("CurrentRank_core3")
    top_now = rank.sort_values().head(10).index.tolist()
    cur["InTop10Today"] = cur["Symbol"].isin(top_now)
    cur.to_csv(BATTLE_DIR / "target_portfolio_current.csv", index=False)
    fresh = pd.DataFrame({"Symbol": top_now, "Rank": range(1, 11), "AlphaScore": [float(score[mkt.symbols.index(s)]) for s in top_now]})
    fresh.to_csv(BATTLE_DIR / "target_portfolio_fresh_ranking.csv", index=False)
    return cur, fresh


def make_charts(ctx):
    mkt, d = ctx.mkt, ctx.mkt.dates
    oos = ctx.oos
    out = {}
    curves = {}
    for label, sig, f in [("Final: core3, quarterly (pre-registered)", "core3", "Q"), ("core3, half-yearly (post-hoc)", "core3", "S"),
                          ("Original composite, static", "composite_quantitative", "N"), ("core3, daily", "core3", "D")]:
        r = run_backtest(mkt, ctx.signal(sig, f), f, *oos)
        curves[label] = pd.Series(r.nav, index=r.dates)
    bench = pd.Series(np.cumprod(1 + mkt.bench_ret[oos[0] : oos[1] + 1]), index=d[oos[0] : oos[1] + 1])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for (k, v), c in zip(curves.items(), [BLUE, GREEN, "#f58518", RED]):
        ax.plot(v.index, v, label=k, color=c, lw=1.6 if "Final" in k else 1.1)
    ax.plot(bench.index, bench, color="k", lw=1.6, ls="--", label="Benchmark: equal-weight universe")
    ax.axvline(d[ctx.hold[0]], color=GREY, ls=":")
    ax.text(d[ctx.hold[0]], ax.get_ylim()[1] * 0.97, " holdout starts", fontsize=8, va="top", color=GREY)
    ax.set_title("Growth of 1.0, walk-forward out-of-sample (2022-01 to 2026-10, 0.1% cost per side)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out["equity"] = img(fig, "Walk-forward equity curves. Validation: left of the dotted line. Holdout: right.")

    fin = curves["Final: core3, quarterly (pre-registered)"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.fill_between(fin.index, fin / fin.cummax() - 1, color=BLUE, alpha=0.5, label="Final strategy")
    ax.plot(bench.index, bench / bench.cummax() - 1, color="k", lw=1.0, label="Benchmark")
    ax.set_title("Drawdown")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out["drawdown"] = img(fig, "Drawdowns of the final strategy and the benchmark")

    s = pd.read_csv(BATTLE_DIR / "strategy_summary.csv")
    fr = [("core3 | Daily", "Daily"), ("core3 | Weekly", "Weekly"), ("core3 | Monthly", "Monthly"),
          ("FINAL core3 | Quarterly", "Quarterly"), ("core3 | Half-yearly", "Half-yearly"), ("core3 | Static (no rebalance)", "Static")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(fr))
    for ax, col, ttl in zip(axes, ["Sharpe", "Turnover"], ["Sharpe by rebalance frequency", "Annual one-way turnover (x NAV)"]):
        if col == "Sharpe":
            v = [s[(s.Strategy == k) & (s.Window == "Val")][col].iloc[0] for k, _ in fr]
            h = [s[(s.Strategy == k) & (s.Window == "Hold")][col].iloc[0] for k, _ in fr]
            ax.bar(x - 0.2, v, 0.4, label="Validation", color=BLUE)
            ax.bar(x + 0.2, h, 0.4, label="Holdout", color=RED)
            ax.axhline(0, color="k", lw=0.6)
            ax.legend(fontsize=8)
        else:
            ax.bar(x, [s[(s.Strategy == k) & (s.Window == "Val")][col].iloc[0] for k, _ in fr], color=GREY)
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in fr], rotation=30, fontsize=8)
        ax.set_title(ttl)
        ax.grid(alpha=0.3, axis="y")
    out["freq"] = img(fig, "Higher turnover hurts: daily and weekly rebalancing are the worst after costs")

    ic = pd.read_csv(BATTLE_DIR / "signal_ic_decay.csv")
    ic = ic[ic.Window == "Val"].set_index("Signal")
    hs = [1, 5, 21, 63, 126]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for sig, c in [("composite_quantitative", BLUE), ("momentum_12_1", GREEN), ("momentum_6m", "#f58518"), ("ridge", "#b279a2"), ("lgbm", RED), ("mlp", GREY)]:
        ax.plot(range(5), [ic.loc[sig, f"IC_{h}d"] for h in hs], marker="o", label=sig, color=c)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{h}d" for h in hs])
    ax.set_title("Rank IC vs forward-return horizon (validation window)")
    ax.set_ylabel("Mean Spearman IC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out["ic"] = img(fig, "Predictive power builds over months, not days: weekly or daily trading has almost nothing to harvest")

    g = pd.read_csv(BATTLE_DIR / "grid_signal_frequency.csv")
    rp = pd.read_csv(BATTLE_DIR / "replicability.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(g.Val_Sharpe, g.Hold_Sharpe, c=BLUE, alpha=0.6)
    ax.axhline(rp.BenchHoldSharpe, color="k", ls="--", lw=1, label="Benchmark holdout Sharpe")
    ax.set_xlabel("Validation Sharpe (2022-01 to 2024-06)")
    ax.set_ylabel("Holdout Sharpe (2024-07 to 2026-10)")
    ax.set_title(f"130 signal x frequency x buffer cells\nSpearman rho = {rp.SpearmanValHoldSharpe:.2f} (p = {rp.p:.2f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out["rep"] = img(fig, "A good validation result barely predicts a good holdout result")

    b = pd.read_csv(BATTLE_DIR / "halfyear_blocks.csv")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    xx = np.arange(len(b))
    ax.bar(xx - 0.2, b.Strategy, 0.4, color=BLUE, label="Final strategy")
    ax.bar(xx + 0.2, b.Benchmark, 0.4, color=GREY, label="Benchmark")
    ax.set_xticks(xx)
    ax.set_xticklabels([str(x)[:7] for x in b.BlockEnd], rotation=30, fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Return per half-year block (fresh out-of-sample period each time)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    out["blocks"] = img(fig, "Half-year out-of-sample blocks (the first and last blocks are short)")

    pw = pd.read_csv(BATTLE_DIR / "procedure_walkforward_blocks.csv")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    xx = np.arange(len(pw))
    ax.bar(xx - 0.2, pw.Procedure, 0.4, color=BLUE, label="Re-selected strategy")
    ax.bar(xx + 0.2, pw.Benchmark, 0.4, color=GREY, label="Benchmark")
    ax.set_xticks(xx)
    ax.set_xticklabels([f"{o[:7]}\n{f}" for o, f in zip(pw.Origin, pw.ChosenFreq)], fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Walk-forward of the selection procedure: choose on the past, trade the next 6 months")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    out["procedure"] = img(fig, "Eight origins; label shows the rebalance frequency the rule picked at each origin")

    pl_ = pd.read_csv(BATTLE_DIR / "placebo_random.csv")
    st = pd.read_csv(BATTLE_DIR / "strategy_summary.csv")
    fm = st[(st.Strategy == "FINAL core3 | Quarterly") & (st.Window == "Full")].iloc[0]
    roll = pd.read_csv(BATTLE_DIR / "rolling_windows.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(pl_.Sharpe, bins=30, color=GREY, alpha=0.8)
    axes[0].axvline(fm.Sharpe, color=RED, lw=2, label=f"Final strategy {fm.Sharpe:.2f}")
    axes[0].set_title("300 random 10-stock quarterly portfolios: Sharpe")
    axes[0].legend(fontsize=8)
    r12 = roll[roll.Months == 12]
    axes[1].hist(r12.Excess, bins=20, color=BLUE, alpha=0.8)
    axes[1].axvline(0, color="k")
    axes[1].set_title("Rolling 12-month excess return vs benchmark")
    for ax in axes:
        ax.grid(alpha=0.3)
    out["placebo"] = img(fig, "Left: random stock-picking in this universe already earns a high Sharpe. Right: 12-month excess is centred near zero.")

    rd = pd.read_csv(BATTLE_DIR / "ridge_h21_feature_weights.csv").set_index("TestStartDate")
    top = rd.abs().mean().sort_values(ascending=False).head(10).index
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in top:
        ax.plot(range(len(rd)), rd[c], marker=".", label=c)
    ax.set_xticks(range(len(rd)))
    ax.set_xticklabels([x[:7] for x in rd.index], rotation=60, fontsize=7)
    ax.set_title("Ridge coefficients re-learned at each quarterly retrain (top 10 by size)")
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(alpha=0.3)
    out["ridge"] = img(fig, "Learned factor weights are unstable over time, a symptom of regime dependence")

    se = pd.read_csv(BATTLE_DIR / "sensitivity.csv")
    h = se[se.Test == "Holdings"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(h.Value, h.Sharpe, marker="o", color=BLUE)
    axes[0].set_title("Sharpe vs number of holdings")
    axes[1].plot(h.Value, h.MaxDD, marker="o", color=RED)
    axes[1].set_title("Max drawdown vs number of holdings")
    for ax in axes:
        ax.grid(alpha=0.3)
    out["holdings"] = img(fig, "More names help risk-adjusted performance (post-hoc; the 10-name limit comes from config.py)")
    return out


def main():
    ctx = build_context()
    cur, fresh = target_portfolio(ctx)
    charts = make_charts(ctx)
    html = render(charts, cur, fresh)
    out = REPORTS_DIR / "battle_ready_report.html"
    out.write_text(html)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
