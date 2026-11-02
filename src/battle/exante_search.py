"""Ex-ante search: choose a configuration using ONLY data up to 2025-12-31, then evaluate 2026-01-01..2026-06-30 once.
Selection rules are fixed in code before the target window is read."""
import itertools
import numpy as np
import pandas as pd

from src.battle.context import FREQ_NAME, HANDSET, build_context
from src.battle.data import BATTLE_DIR
from src.battle.engine import metrics, run_backtest

SIGNALS = HANDSET + ["ridge", "lgbm", "mlp", "ml_ensemble", "core3"]
FREQS = ["W", "M", "Q", "S"]
BUFFERS = [None, 20]
WINDOWS = {"Train": ("2022-01-03", "2023-12-29"), "Select": ("2024-01-02", "2025-12-31"), "Target": ("2026-01-01", "2026-06-30")}


def idx(m, day, right=False):
    return int(m.dates.searchsorted(pd.Timestamp(day), side="right" if right else "left")) - (1 if right else 0)


def main():
    ctx = build_context()
    m = ctx.mkt
    w = {k: (idx(m, a), idx(m, b, True)) for k, (a, b) in WINDOWS.items()}
    rows = []
    for sig, f, buf in itertools.product(SIGNALS, FREQS, BUFFERS):
        s = ctx.signal(sig, f)
        row = {"Signal": sig, "Freq": FREQ_NAME[f], "Buffer": buf or 0}
        for k, (a, b) in w.items():
            r = run_backtest(m, s, f, a, b, buffer=buf)
            mt = metrics(r.returns, r.dates)
            row[f"{k}_Return"], row[f"{k}_Sharpe"], row[f"{k}_MaxDD"] = mt["TotalReturn"], mt["Sharpe"], mt["MaxDD"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["Score"] = (df.Train_Sharpe + df.Select_Sharpe) / 2  # selection uses pre-2026 data only
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.to_csv(BATTLE_DIR / "exante_search.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["Signal", "Freq", "Buffer", "Train_Sharpe", "Select_Sharpe", "Score", "Target_Return"]
    print("Top 10 by pre-2026 score and their 2026H1 return:\n", df[cols].head(10).round(3).to_string())
    top = df.head(10)
    print("\nTop-1: %.1f%%   mean of top-5: %.1f%%   mean of top-10: %.1f%%   all-%d median: %.1f%%" % (
        top.Target_Return.iloc[0] * 100, df.head(5).Target_Return.mean() * 100, top.Target_Return.mean() * 100, len(df), df.Target_Return.median() * 100))
    print("rank corr (pre-2026 score vs H1-2026 return): %.2f" % df[["Score", "Target_Return"]].corr(method="spearman").iloc[0, 1])
    print("share of configs >=22%%: %.1f%%   max: %.1f%%" % ((df.Target_Return >= .22).mean() * 100, df.Target_Return.max() * 100))


if __name__ == "__main__":
    main()
