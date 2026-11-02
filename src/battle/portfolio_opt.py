"""Weighting challengers on the frozen core3 selection: equal vs inverse-vol, shrinkage min-variance,
risk parity, mean-variance and CAPM-implied mean-variance. Rule fixed beforehand: adopt a scheme only if it
beats equal weight on VALIDATION Sharpe and max drawdown; the holdout is then reported separately."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.battle.context import FREQ_NAME, build_context
from src.battle.data import BATTLE_DIR
from src.battle.engine import metrics, run_backtest

LOOKBACK, CAP, LAMBDA, ANN = 252, 0.20, 3.0, 252
SCHEMES = ["equal", "invvol", "minvar_lw", "risk_parity_lw", "meanvar_lw", "capm_meanvar_lw"]


def _returns(mkt):
    c = mkt.close
    prev = np.vstack([c[:1], c[:-1]])
    ok = (c > 0) & (prev > 0) & mkt.obs
    return np.where(ok, c / np.where(prev > 0, prev, 1) - 1, 0.0)


def _solve(obj, n):
    res = minimize(obj, np.full(n, 1 / n), method="SLSQP", bounds=[(1e-4, CAP)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 200})
    w = np.clip(res.x if res.success else np.full(n, 1 / n), 1e-4, CAP)
    return w / w.sum()


def make_weight_fn(mkt, scheme):
    ret = _returns(mkt)
    bench = mkt.bench_ret

    def fn(t, chosen):
        n = len(chosen)
        if scheme == "equal":
            return np.full(n, 1 / n)
        R = ret[max(0, t - LOOKBACK + 1): t + 1][:, chosen]
        if scheme == "invvol":
            iv = 1 / np.maximum(R.std(0), 1e-6)
            w = iv / iv.sum()
            for _ in range(5):
                w = np.minimum(w, CAP)
                w /= w.sum()
            return w
        cov = LedoitWolf().fit(R).covariance_ * ANN
        if scheme == "minvar_lw":
            return _solve(lambda w: w @ cov @ w, n)
        if scheme == "risk_parity_lw":
            return _solve(lambda w: 0.5 * w @ cov @ w - np.log(w).sum() / n, n)
        if scheme == "meanvar_lw":
            m = R.mean(0) * ANN
            mu = 0.5 * m + 0.5 * m.mean()  # shrink noisy individual means to the cross-sectional mean
        else:  # CAPM: expected return = beta x market premium
            mb = bench[max(0, t - LOOKBACK + 1): t + 1]
            beta = np.array([np.cov(R[:, j], mb)[0, 1] for j in range(n)]) / max(mb.var(ddof=1), 1e-12)
            mu = beta * max(mb.mean() * ANN, 0.05)
        return _solve(lambda w: -(mu @ w) + 0.5 * LAMBDA * w @ cov @ w, n)

    return fn


def run(ctx, freqs=("Q", "M", "S")):
    mkt = ctx.mkt
    wins = {"Val": ctx.val, "Hold": ctx.hold, "Full": ctx.oos}
    rows, daily = [], {}
    for f in freqs:
        score = ctx.signal("core3", f)
        for sch in SCHEMES:
            fn = make_weight_fn(mkt, sch)
            for tag, (a, b) in wins.items():
                r = run_backtest(mkt, score, f, a, b, weight_fn=fn)
                m = metrics(r.returns, r.dates)
                rows.append({"Freq": FREQ_NAME[f], "Scheme": sch, "Window": tag, **m, "Turnover": r.turnover})
                if tag == "Full":
                    daily[(f, sch)] = pd.Series(r.returns, index=r.dates)
    summ = pd.DataFrame(rows)
    summ.to_csv(BATTLE_DIR / "portfolio_opt_summary.csv", index=False)

    # per-period comparison with equal weight (the "better every rebalance?" question)
    blocks = []
    for (f, sch), s in daily.items():
        eq = daily[(f, "equal")]
        for label, key in (("Year", s.index.year), ("Quarter", s.index.to_period("Q").astype(str))):
            g = s.groupby(key)
            tot, vol = g.apply(lambda x: (1 + x).prod() - 1), g.std() * np.sqrt(ANN)
            etot, evol = eq.groupby(key).apply(lambda x: (1 + x).prod() - 1), eq.groupby(key).std() * np.sqrt(ANN)
            for k in tot.index:
                blocks.append({"Freq": FREQ_NAME[f], "Scheme": sch, "Block": label, "Period": str(k), "Return": tot[k],
                               "EqualReturn": etot[k], "Vol": vol[k], "EqualVol": evol[k]})
    bl = pd.DataFrame(blocks)
    bl.to_csv(BATTLE_DIR / "portfolio_opt_blocks.csv", index=False)
    win = (bl.assign(RetWin=bl.Return > bl.EqualReturn, VolWin=bl.Vol < bl.EqualVol)
             .groupby(["Freq", "Scheme", "Block"]).agg(Periods=("Period", "count"), BeatEqualReturn=("RetWin", "mean"),
                                                        LowerVolThanEqual=("VolWin", "mean"),
                                                        MeanExcessReturn=("Return", lambda x: 0)).reset_index())
    ex = bl.assign(Ex=bl.Return - bl.EqualReturn).groupby(["Freq", "Scheme", "Block"]).Ex.mean().values
    win["MeanExcessReturn"] = ex
    win.to_csv(BATTLE_DIR / "portfolio_opt_winrates.csv", index=False)
    return summ, win


if __name__ == "__main__":
    ctx = build_context()
    summ, win = run(ctx)
    cols = ["Freq", "Scheme", "Window", "TotalReturn", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]
    pd.set_option("display.width", 200)
    for w in ("Val", "Hold", "Full"):
        print(f"\n== {w}")
        print(summ[summ.Window == w][cols].drop(columns="Window").round(3).to_string(index=False))
    print(win[win.Block == "Quarter"].round(3).to_string(index=False))
