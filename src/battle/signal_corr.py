"""Average cross-sectional rank correlation between signals (sampled every 21 trading days, validation period)."""
import numpy as np
import pandas as pd

from src.battle.context import build_context
from src.battle.data import BATTLE_DIR

SIGNALS = ["momentum_6m", "risk_adjusted_momentum", "momentum_12_1", "momentum_lowvol", "relative_strength", "trend_quality", "technical_composite"]


def main():
    ctx = build_context()
    a, b = ctx.val
    days = range(a, b + 1, 21)
    mats = []
    for d in days:
        df = pd.DataFrame({k: ctx.signal(k, "Q")[d] for k in SIGNALS})
        mats.append(df.rank().corr().to_numpy())
    avg = pd.DataFrame(np.nanmean(mats, axis=0), index=SIGNALS, columns=SIGNALS)
    avg.to_csv(BATTLE_DIR / "signal_correlation.csv")
    print(avg.round(2))


if __name__ == "__main__":
    main()
