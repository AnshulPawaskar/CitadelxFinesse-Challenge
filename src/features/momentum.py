"""Momentum return features. All windows use trailing (past-only) data — no look-ahead."""
import polars as pl

RETURN_WINDOWS = {"R21": 21, "R63": 63, "R126": 126, "R252": 252}

# Multi-horizon momentum skips the most recent month (21d) to avoid short-term mean-reversion noise.
SKIP_MONTH_MOMENTUM = {
    "Momentum3_1": 63,
    "Momentum6_1": 126,
    "Momentum9_1": 189,
    "Momentum12_1": 252,
}


def add_momentum_features(df: pl.DataFrame) -> pl.DataFrame:
    exprs = [(pl.col("Close") / pl.col("Close").shift(w) - 1).alias(name) for name, w in RETURN_WINDOWS.items()]
    exprs += [
        (pl.col("Close").shift(21) / pl.col("Close").shift(w) - 1).alias(name)
        for name, w in SKIP_MONTH_MOMENTUM.items()
    ]
    return df.with_columns(exprs)
