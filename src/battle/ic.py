"""Stage 1: signal predictive power (rank IC) by horizon and rank persistence - the basis for choosing a rebalance frequency."""
import numpy as np
import pandas as pd

from src.battle.context import ML, HANDSET, build_context
from src.battle.data import BATTLE_DIR
from src.battle.models import cs_percentile

HORIZONS = [1, 5, 21, 63, 126]
LAGS = [1, 5, 21, 63]


def row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = ~np.isnan(a) & ~np.isnan(b)
    a, b = np.where(m, a, 0.0), np.where(m, b, 0.0)
    n = m.sum(1)
    ma, mb = a.sum(1) / np.maximum(n, 1), b.sum(1) / np.maximum(n, 1)
    da, db = np.where(m, a - ma[:, None], 0), np.where(m, b - mb[:, None], 0)
    with np.errstate(all="ignore"):
        c = (da * db).sum(1) / np.sqrt((da ** 2).sum(1) * (db ** 2).sum(1))
    return np.where(n > 30, c, np.nan)


def run(ctx) -> pd.DataFrame:
    rows = []
    sigs = [(s, "M") for s in HANDSET + ML + ["hybrid"]]
    for name, f in sigs:
        score = ctx.signal(name, f)
        rk = cs_percentile(score, ctx.elig & ~np.isnan(score))
        for tag, (a, b) in (("Val", ctx.val), ("Hold", ctx.hold)):
            sl = slice(a, b + 1)
            row = {"Signal": name, "Window": tag}
            for h in HORIZONS:
                ic = row_corr(rk[sl], cs_percentile(ctx.mkt.forward_return(h), ctx.elig)[sl])
                row[f"IC_{h}d"] = np.nanmean(ic)
                row[f"ICIR_{h}d"] = np.nanmean(ic) / np.nanstd(ic) if np.nanstd(ic) > 0 else np.nan
            for lag in LAGS:
                sh = np.full_like(rk, np.nan)
                sh[lag:] = rk[:-lag]
                row[f"RankAutocorr_{lag}d"] = np.nanmean(row_corr(rk[sl], sh[sl]))
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run(build_context())
    df.to_csv(BATTLE_DIR / "signal_ic_decay.csv", index=False)
    v = df[df.Window == "Val"].set_index("Signal")
    print(v[[f"IC_{h}d" for h in HORIZONS]].round(4))
    print(v[[f"RankAutocorr_{l}d" for l in LAGS]].round(3))
