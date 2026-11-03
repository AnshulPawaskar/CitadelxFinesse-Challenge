"""Stage 4: robustness of the frozen strategy - holdout, half-year blocks, rolling windows, bootstrap,
random-portfolio placebo, cost / size sensitivity, validation->holdout replicability, deflated Sharpe."""
import numpy as np
import pandas as pd
from scipy import stats

from src.battle.context import FREQ_NAME, HANDSET, build_context
from src.battle.data import BATTLE_DIR
from src.battle.design import trend_overlay
from src.battle.engine import metrics, run_backtest

FINAL = dict(signal="core3", freq="S")
N_TRIALS = 80 + 8  # factor grid cells (8 signals x 5 frequencies x 2 buffers) + design variants


def nav_series(ctx, signal, freq, window, **kw):
    r = run_backtest(ctx.mkt, ctx.signal(signal, freq), freq, window[0], window[1], **kw)
    return pd.Series(r.returns, index=r.dates), r


def summary_rows(ctx, label, ret_full: pd.Series, bench: pd.Series, turnover=np.nan):
    rows = []
    for tag, (a, b) in (("Val", ctx.val), ("Hold", ctx.hold), ("Full", ctx.oos)):
        s = ret_full.loc[ctx.mkt.dates[a] : ctx.mkt.dates[b]]
        bm = bench.loc[s.index]
        m = metrics(s.to_numpy(), s.index)
        mb = metrics(bm.to_numpy(), bm.index)
        rows.append({"Strategy": label, "Window": tag, **m, "BenchCAGR": mb["CAGR"], "BenchSharpe": mb["Sharpe"],
                     "ExcessCAGR": m["CAGR"] - mb["CAGR"], "Turnover": turnover})
    return rows


def block_returns(ret: pd.Series, bench: pd.Series, freq="6ME") -> pd.DataFrame:
    rows = []
    for key, g in ret.groupby(pd.Grouper(freq=freq)):
        if len(g) < 20:
            continue
        b = bench.loc[g.index]
        rows.append({"BlockEnd": key.date(), "Days": len(g), "Strategy": (1 + g).prod() - 1, "Benchmark": (1 + b).prod() - 1,
                     "StrategySharpe": metrics(g.to_numpy(), g.index)["Sharpe"]})
    df = pd.DataFrame(rows)
    df["Excess"] = df["Strategy"] - df["Benchmark"]
    return df


def rolling_windows(ret: pd.Series, bench: pd.Series, months: int) -> pd.DataFrame:
    starts = pd.date_range(ret.index[0], ret.index[-1], freq="MS")
    rows = []
    for s in starts:
        e = s + pd.DateOffset(months=months)
        if e > ret.index[-1] + pd.Timedelta(days=1):
            break
        g = ret.loc[s:e - pd.Timedelta(days=1)]
        if len(g) < months * 15:
            continue
        b = bench.loc[g.index]
        rows.append({"Start": s.date(), "Months": months, "Strategy": (1 + g).prod() - 1, "Benchmark": (1 + b).prod() - 1})
    df = pd.DataFrame(rows)
    df["Excess"] = df["Strategy"] - df["Benchmark"]
    return df


def stationary_bootstrap(x: np.ndarray, n_boot=2000, block=21, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty((n_boot, n))
    for i in range(n_boot):
        idx = np.empty(n, dtype=int)
        pos = rng.integers(n)
        for t in range(n):
            idx[t] = pos
            pos = rng.integers(n) if rng.random() < 1 / block else (pos + 1) % n
        out[i] = x[idx]
    return out


def deflated_sharpe(ret: np.ndarray, n_trials: int, sr_var_trials: float) -> tuple[float, float]:
    """Probabilistic Sharpe vs 0 and Deflated Sharpe (Bailey & Lopez de Prado) using daily returns."""
    n = len(ret)
    sr = ret.mean() / ret.std(ddof=1)
    sk, ku = stats.skew(ret), stats.kurtosis(ret, fisher=False)
    denom = np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    psr = stats.norm.cdf(sr * np.sqrt(n - 1) / denom)
    emc = 0.5772156649
    sr0 = np.sqrt(sr_var_trials) * ((1 - emc) * stats.norm.ppf(1 - 1 / n_trials) + emc * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom)
    return psr, dsr


def run(ctx):
    mkt = ctx.mkt
    bench = pd.Series(mkt.bench_ret, index=mkt.dates)
    oos = ctx.oos
    out = {}

    # --- comparators over the whole walk-forward OOS window (continuous NAV) -----------------------------
    runs = {}
    for label, sig, f, kw in [
        ("FINAL core3 | Half-yearly", "core3", "S", {}),
        ("core3 | Quarterly", "core3", "Q", {}),
        ("core3 | Daily", "core3", "D", {}), ("core3 | Weekly", "core3", "W", {}),
        ("core3 | Monthly", "core3", "M", {}),
        ("core3 | Static (no rebalance)", "core3", "N", {}),
        ("Original composite_quantitative | Static", "composite_quantitative", "N", {}),
        ("composite_quantitative | Quarterly", "composite_quantitative", "Q", {}),
        ("momentum_12_1 | Half-yearly", "momentum_12_1", "S", {}),
        ("momentum_6m | Half-yearly", "momentum_6m", "S", {}),
    ]:
        s, r = nav_series(ctx, sig, f, oos, **kw)
        runs[label] = (s, r)
    rows = []
    for label, (s, r) in runs.items():
        rows += summary_rows(ctx, label, s, bench, r.turnover)
    rows += summary_rows(ctx, "Benchmark (equal-weight universe, daily rebal)", bench.loc[mkt.dates[oos[0]] : mkt.dates[oos[1]]].reindex(mkt.dates).fillna(0), bench)
    summ = pd.DataFrame(rows)
    summ.to_csv(BATTLE_DIR / "strategy_summary.csv", index=False)

    final_ret, final_run = runs["FINAL core3 | Half-yearly"]
    pd.DataFrame({"Strategy": final_ret, "Benchmark": bench.loc[final_ret.index]}).to_csv(BATTLE_DIR / "final_daily_returns.csv")

    # --- own out-of-sample periods -------------------------------------------------------------------
    blocks = block_returns(final_ret, bench.loc[final_ret.index])
    blocks.to_csv(BATTLE_DIR / "halfyear_blocks.csv", index=False)
    yearly = block_returns(final_ret, bench.loc[final_ret.index], "YE")
    yearly.to_csv(BATTLE_DIR / "calendar_years.csv", index=False)
    roll = pd.concat([rolling_windows(final_ret, bench.loc[final_ret.index], m) for m in (6, 12)])
    roll.to_csv(BATTLE_DIR / "rolling_windows.csv", index=False)
    rsum = roll.groupby("Months").agg(Windows=("Excess", "size"), BeatBenchmark=("Excess", lambda x: (x > 0).mean()),
                                      MedianStrategy=("Strategy", "median"), MedianBenchmark=("Benchmark", "median"),
                                      MedianExcess=("Excess", "median"), WorstExcess=("Excess", "min"),
                                      PositiveReturn=("Strategy", lambda x: (x > 0).mean()))
    rsum.to_csv(BATTLE_DIR / "rolling_summary.csv")

    # --- bootstrap on daily excess returns ----------------------------------------------------------------
    ex = (final_ret - bench.loc[final_ret.index]).to_numpy()
    boot = stationary_bootstrap(ex)
    ann = boot.mean(1) * 252
    bs = {"MeanAnnExcess": ex.mean() * 252, "CI5": np.percentile(ann, 5), "CI95": np.percentile(ann, 95),
          "ProbExcessPositive": (ann > 0).mean()}
    # strategy Sharpe CI via stationary bootstrap of the strategy's own returns
    sb = stationary_bootstrap(final_ret.to_numpy(), n_boot=1000, seed=1)
    shs = sb.mean(1) / sb.std(1, ddof=1) * np.sqrt(252)
    bs.update({"SharpeCI5": np.percentile(shs, 5), "SharpeCI95": np.percentile(shs, 95), "Sharpe": metrics(final_ret.to_numpy())["Sharpe"]})
    pd.DataFrame([bs]).to_csv(BATTLE_DIR / "bootstrap.csv", index=False)

    # --- random-portfolio placebo (same frequency, size, costs, universe) --------------------------------
    rng = np.random.default_rng(42)
    pl_rows = []
    for i in range(300):
        rnd = rng.random(mkt.close.shape)
        r = run_backtest(mkt, rnd, FINAL["freq"], oos[0], oos[1])
        m = metrics(r.returns, r.dates)
        pl_rows.append({"Sharpe": m["Sharpe"], "CAGR": m["CAGR"], "MaxDD": m["MaxDD"]})
    placebo = pd.DataFrame(pl_rows)
    placebo.to_csv(BATTLE_DIR / "placebo_random.csv", index=False)
    fm = metrics(final_ret.to_numpy(), final_ret.index)

    # --- sensitivities --------------------------------------------------------------------------------------
    sens = []
    for cost in (0.001, 0.002, 0.003, 0.005):
        r = run_backtest(mkt, ctx.signal("core3", "S"), "S", oos[0], oos[1], cost=cost)
        sens.append({"Test": "Cost per side", "Value": f"{cost:.1%}", **metrics(r.returns, r.dates)})
    for n in (5, 10, 15, 20, 30):
        r = run_backtest(mkt, ctx.signal("core3", "S"), "S", oos[0], oos[1], n=n)
        sens.append({"Test": "Holdings", "Value": str(n), **metrics(r.returns, r.dates)})
    for adv in (2e7, 5e7, 1e8):
        r = run_backtest(mkt, ctx.signal("core3", "S"), "S", oos[0], oos[1], eligible=mkt.eligible(adv))
        sens.append({"Test": "Min avg daily value (Rs)", "Value": f"{adv / 1e7:.0f} cr", **metrics(r.returns, r.dates)})
    # shifted rebalance phase: same half-yearly frequency, signal computed earlier each period
    score = ctx.signal("core3", "S")
    for shift in (0, 21, 42):
        sh = np.full_like(score, np.nan)
        sh[shift:] = score[: len(score) - shift] if shift else score
        r = run_backtest(mkt, sh, "S", oos[0] + shift, oos[1])
        sens.append({"Test": "Stale signal (days)", "Value": str(shift), **metrics(r.returns, r.dates)})
    pd.DataFrame(sens).to_csv(BATTLE_DIR / "sensitivity.csv", index=False)

    # --- replicability: does validation performance predict holdout across the whole grid? -------------------------
    g = pd.read_csv(BATTLE_DIR / "grid_signal_frequency.csv")
    g = g[g["Signal"].isin(HANDSET)]
    rho, p = stats.spearmanr(g["Val_Sharpe"], g["Hold_Sharpe"])
    bh = metrics(bench.loc[mkt.dates[ctx.hold[0]] : mkt.dates[ctx.hold[1]]].to_numpy())["Sharpe"]
    rep = {"SpearmanValHoldSharpe": rho, "p": p, "CellsBeatBenchHoldSharpe": (g["Hold_Sharpe"] > bh).mean(),
           "CellsPositiveHoldReturn": (g["Hold_TotalReturn"] > 0).mean(), "BenchHoldSharpe": bh,
           "MedianHoldSharpe": g["Hold_Sharpe"].median(), "MedianValSharpe": g["Val_Sharpe"].median()}
    pd.DataFrame([rep]).to_csv(BATTLE_DIR / "replicability.csv", index=False)

    psr, dsr = deflated_sharpe(final_ret.to_numpy(), N_TRIALS, np.nanvar(g["Val_Sharpe"] / np.sqrt(252)))
    stat = {"PSR_vs_0": psr, "DSR": dsr, "N_trials": N_TRIALS,
            "PlaceboSharpePctile": (placebo["Sharpe"] < fm["Sharpe"]).mean(), "PlaceboCAGRPctile": (placebo["CAGR"] < fm["CAGR"]).mean(),
            "PlaceboMedianSharpe": placebo["Sharpe"].median()}
    pd.DataFrame([stat]).to_csv(BATTLE_DIR / "statistical_tests.csv", index=False)
    return summ, blocks, yearly, rsum, bs, rep, stat, sens, placebo


if __name__ == "__main__":
    ctx = build_context()
    summ, blocks, yearly, rsum, bs, rep, stat, sens, placebo = run(ctx)
    pd.set_option("display.width", 250)
    print(summ[summ.Window != "Full"][["Strategy", "Window", "CAGR", "Sharpe", "MaxDD", "BenchCAGR", "BenchSharpe", "Turnover"]].round(3).to_string(index=False))
    print(summ[summ.Window == "Full"][["Strategy", "CAGR", "Sharpe", "MaxDD", "BenchCAGR", "BenchSharpe"]].round(3).to_string(index=False))
    print(blocks.round(3).to_string(index=False)); print(yearly.round(3).to_string(index=False)); print(rsum.round(3))
    print(bs); print(rep); print(stat); print(pd.DataFrame(sens).round(3).to_string(index=False))
