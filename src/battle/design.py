"""Stage 3: design choices for the frozen core signal, decided on VALIDATION only."""
import numpy as np
import pandas as pd

from src.battle.context import build_context
from src.battle.data import BATTLE_DIR
from src.battle.engine import metrics, run_backtest


def trend_overlay(mkt, ma: int = 200, floor: float = 0.5) -> np.ndarray:
    """Exposure 1.0 while the equal-weight universe index is above its 200d average, else `floor` (rest in cash)."""
    idx = pd.Series(mkt.bench_index)
    return np.where(idx > idx.rolling(ma).mean(), 1.0, floor)


def run(ctx, signal="core3", freq="Q") -> pd.DataFrame:
    score = ctx.signal(signal, freq)
    overlay = trend_overlay(ctx.mkt)
    rows = []
    for buf in (None, 20):
        for wt in ("equal", "invvol"):
            for ov in ("none", "trend"):
                row = {"Signal": signal, "Freq": freq, "Buffer": buf or 0, "Weighting": wt, "Overlay": ov}
                for tag, (a, b) in (("Val", ctx.val), ("Hold", ctx.hold)):
                    r = run_backtest(ctx.mkt, score, freq, a, b, buffer=buf, weighting=wt,
                                     exposure=overlay if ov == "trend" else None)
                    row.update({f"{tag}_{k}": v for k, v in metrics(r.returns, r.dates).items()})
                    row[f"{tag}_Turnover"] = r.turnover
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ctx = build_context()
    df = run(ctx)
    df.to_csv(BATTLE_DIR / "design_variants.csv", index=False)
    print(df[["Buffer", "Weighting", "Overlay", "Val_CAGR", "Val_Sharpe", "Val_MaxDD", "Val_Turnover"]].round(3).to_string(index=False))
