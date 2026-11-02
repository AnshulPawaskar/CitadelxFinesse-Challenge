"""Stage 5: walk-forward test of the SELECTION PROCEDURE itself.

At each origin the frozen rule is re-run using only data before the origin (expanding validation window from
2022-01-03): top-3 signals by mean Sharpe across frequencies -> rank-average -> frequency with the highest mean
Sharpe across signals. The chosen strategy is then traded for the next 6 months (fresh capital, entry costs charged).
This repeats the 'select on past, test on future' experiment many times instead of once."""
import numpy as np
import pandas as pd

from src.battle.context import SIGNALS, build_context, combo, idx_on_or_after
from src.battle.data import BATTLE_DIR
from src.battle.engine import metrics, run_backtest

ORIGINS = ["2023-01-02", "2023-07-03", "2024-01-02", "2024-07-01", "2025-01-02", "2025-07-01", "2026-01-02", "2026-07-01"]
FREQS = ["D", "W", "M", "Q", "S"]


def val_sharpe_table(ctx, a, b) -> pd.DataFrame:
    rows = {}
    for sig in SIGNALS:
        for f in FREQS:
            r = run_backtest(ctx.mkt, ctx.signal(sig, f), f, a, b)
            rows[(sig, f)] = metrics(r.returns, r.dates)["Sharpe"]
    return pd.Series(rows).unstack()[FREQS]


def run(ctx):
    d = ctx.mkt.dates
    bounds = [idx_on_or_after(d, o) for o in ORIGINS] + [len(d)]
    bench = pd.Series(ctx.mkt.bench_ret, index=d)
    rows, series = [], {"Procedure": [], "Fixed core3 Q": [], "Fixed core3 S": [], "Fixed composite S": [], "Benchmark": []}
    for k, o in enumerate(ORIGINS):
        a, b = bounds[k], bounds[k + 1] - 1
        tab = val_sharpe_table(ctx, ctx.first_test, a - 1)
        top3 = list(tab.mean(axis=1).sort_values(ascending=False).index[:3])
        freq = tab.mean(axis=0).idxmax()
        picks = {
            "Procedure": (combo(ctx, top3, freq), freq),
            "Fixed core3 Q": (ctx.signal("core3", "Q"), "Q"),
            "Fixed core3 S": (ctx.signal("core3", "S"), "S"),
            "Fixed composite S": (ctx.signal("composite_quantitative", "S"), "S"),
        }
        row = {"Origin": o, "TestEnd": str(d[b].date()), "ChosenSignals": ", ".join(top3), "ChosenFreq": freq}
        for name, (score, f) in picks.items():
            r = run_backtest(ctx.mkt, score, f, a, b)
            series[name].append(pd.Series(r.returns, index=r.dates))
            row[name] = (1 + r.returns).prod() - 1
        bs = bench.iloc[a : b + 1]
        series["Benchmark"].append(bs)
        row["Benchmark"] = (1 + bs).prod() - 1
        row["ProcedureExcess"] = row["Procedure"] - row["Benchmark"]
        rows.append(row)
        print(o, top3, freq, round(row["Procedure"], 3), round(row["Benchmark"], 3), flush=True)
    blocks = pd.DataFrame(rows)
    blocks.to_csv(BATTLE_DIR / "procedure_walkforward_blocks.csv", index=False)
    summ = []
    for name, parts in series.items():
        s = pd.concat(parts)
        summ.append({"Strategy": name, **metrics(s.to_numpy(), s.index), "BlocksBeatingBenchmark": (blocks[name] > blocks["Benchmark"]).mean() if name != "Benchmark" else np.nan})
    summ = pd.DataFrame(summ)
    summ.to_csv(BATTLE_DIR / "procedure_walkforward_summary.csv", index=False)
    pd.concat({k: pd.concat(v) for k, v in series.items()}, axis=1).to_csv(BATTLE_DIR / "procedure_walkforward_returns.csv")
    return blocks, summ


if __name__ == "__main__":
    blocks, summ = run(build_context())
    pd.set_option("display.width", 250)
    print(blocks.round(3).to_string(index=False)); print(summ.round(3).to_string(index=False))
