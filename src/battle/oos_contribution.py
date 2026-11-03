"""Profit contribution by stock for the competition window (1 Jan to 30 Jun 2026): factor blend (core3), half-yearly, equal weight.
One portfolio is formed from the signal on the last trading day before 1 Jan 2026 and held to 30 Jun 2026."""
import numpy as np
import pandas as pd

from src.battle.context import build_context
from src.battle.engine import run_backtest

CAPITAL = 1e7
COST = 0.001


def main():
    ctx = build_context()
    m = ctx.mkt
    a = int(m.dates.searchsorted(pd.Timestamp("2026-01-01")))
    b = int(m.dates.searchsorted(pd.Timestamp("2026-06-30"), side="right")) - 1
    score = ctx.signal("core3", "S")
    r = run_backtest(m, score, "S", a, b, record=True)
    names = r.holdings[0][1]
    idx = [m.symbols.index(s) for s in names]
    # shares bought at the open on day a, after the entry cost
    shares = (1 - COST) * 0.1 / m.open[a, idx]
    pnl = (shares * m.close[b, idx] - 0.1) * CAPITAL
    comp = {}
    for k in ("momentum_6m", "risk_adjusted_momentum", "momentum_12_1", "core3"):
        sc = ctx.signal(k, "S")[a - 1]
        valid = ~np.isnan(sc)
        rank = pd.Series(sc).where(valid).rank(ascending=False)
        comp[k] = [int(rank.iloc[i]) if valid[i] else None for i in idx]
    df = pd.DataFrame({"Symbol": names, "PnL_Rs": pnl, "Weight": 0.1, "RankMomentum6m": comp["momentum_6m"],
                       "RankRiskAdjMomentum": comp["risk_adjusted_momentum"], "RankMomentum12_1": comp["momentum_12_1"], "RankFinal": comp["core3"]})
    df["ContribPctOfStartCapital"] = df.PnL_Rs / CAPITAL * 100
    df["PctOfNetProfit"] = df.PnL_Rs / df.PnL_Rs.sum() * 100
    df = df.sort_values("RankFinal")
    df.to_csv("results/oos_2026H1_contribution.csv", index=False)
    print(df.round(2).to_string(index=False), "\nnet", df.PnL_Rs.sum(), "nav", (r.nav[-1] - 1) * CAPITAL)


if __name__ == "__main__":
    main()
