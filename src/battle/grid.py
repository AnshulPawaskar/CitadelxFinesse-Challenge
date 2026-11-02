"""Stage 2: signal x rebalance-frequency x hold-buffer grid, scored separately on validation and holdout."""
import time

import numpy as np
import pandas as pd

from src.battle.context import FREQ_NAME, SIGNALS, build_context
from src.battle.data import BATTLE_DIR
from src.battle.engine import metrics, run_backtest

FREQS = ["D", "W", "M", "Q", "S"]
BUFFERS = [None, 20]


def bench_metrics(ctx, window):
    a, b = window
    return metrics(ctx.mkt.bench_ret[a : b + 1], ctx.mkt.dates[a : b + 1])


def run_grid(ctx, signals=SIGNALS, freqs=FREQS, buffers=BUFFERS, weighting="equal", cost=0.001) -> pd.DataFrame:
    rows = []
    for sig in signals:
        for f in freqs:
            score = ctx.signal(sig, f)
            for buf in buffers:
                row = {"Signal": sig, "Freq": f, "Buffer": buf or 0, "Weighting": weighting}
                for tag, (a, b) in (("Val", ctx.val), ("Hold", ctx.hold)):
                    r = run_backtest(ctx.mkt, score, f, a, b, buffer=buf, cost=cost, weighting=weighting)
                    m = metrics(r.returns, r.dates)
                    row.update({f"{tag}_{k}": v for k, v in m.items()})
                    row[f"{tag}_Turnover"] = r.turnover
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = time.time()
    ctx = build_context()
    df = run_grid(ctx)
    df.to_csv(BATTLE_DIR / "grid_signal_frequency.csv", index=False)
    bv, bh = bench_metrics(ctx, ctx.val), bench_metrics(ctx, ctx.hold)
    pd.DataFrame([{"Window": "Val", **bv}, {"Window": "Hold", **bh}]).to_csv(BATTLE_DIR / "benchmark_metrics.csv", index=False)
    print(f"{len(df)} runs in {time.time() - t:.0f}s")
    print("bench val", {k: round(v, 3) for k, v in bv.items()})
    # Selection uses validation columns only.
    show = df.sort_values("Val_Sharpe", ascending=False).head(12)[["Signal", "Freq", "Buffer", "Val_CAGR", "Val_Sharpe", "Val_MaxDD", "Val_Turnover"]]
    print(show.round(3).to_string(index=False))
