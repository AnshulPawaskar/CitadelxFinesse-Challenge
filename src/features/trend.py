"""Trend-quality, 52-week high/low, and breakout features. Highs/lows use prior-day data only
(shift(1) before rolling) so a breakout signal never uses today's own high/low to confirm itself."""
import polars as pl
import talib

from src.features.utils import rolling_ols_trend, talib_series

TREND_WINDOWS = [60, 90, 126, 180]
HIGH_LOW_WINDOWS = {"52W": 252, "126D": 126}
BREAKOUT_WINDOWS = [20, 55, 126, 252]


def add_trend_regression_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("Close").log().alias("_log_close"))
    for window in TREND_WINDOWS:
        df = rolling_ols_trend(df, "_log_close", window, prefix=f"Trend{window}")
    return df.drop("_log_close")


def add_high_low_features(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for label, window in HIGH_LOW_WINDOWS.items():
        prior_high = pl.col("High").shift(1).rolling_max(window_size=window)
        prior_low = pl.col("Low").shift(1).rolling_min(window_size=window)
        exprs += [
            prior_high.alias(f"High_{label}"),
            prior_low.alias(f"Low_{label}"),
            (pl.col("Close") / prior_high).alias(f"CloseTo{label}High"),
            (pl.col("Close") / prior_low).alias(f"CloseTo{label}Low"),
            (pl.col("Close") / prior_high - 1).alias(f"DistanceFrom{label}High"),
        ]
    return df.with_columns(exprs)


def add_breakout_features(df: pl.DataFrame) -> pl.DataFrame:
    high = df["High"].to_numpy().astype("float64")
    low = df["Low"].to_numpy().astype("float64")
    close = df["Close"].to_numpy().astype("float64")
    atr14 = talib.ATR(high, low, close, timeperiod=14)

    exprs = []
    for window in BREAKOUT_WINDOWS:
        prior_high = pl.col("High").shift(1).rolling_max(window_size=window)
        distance = pl.col("Close") - prior_high
        exprs += [
            (pl.col("Close") > prior_high).alias(f"Breakout{window}D"),
            distance.alias(f"BreakoutDistance{window}D"),
        ]
    df = df.with_columns(exprs)

    df = df.with_columns(talib_series("_atr14", atr14))
    df = df.with_columns([
        pl.when(pl.col("_atr14") > 0).then(pl.col(f"BreakoutDistance{w}D") / pl.col("_atr14")).otherwise(None)
        .alias(f"BreakoutDistance{w}D_ATR")
        for w in BREAKOUT_WINDOWS
    ])
    return df.drop("_atr14")


def add_trend_features(df: pl.DataFrame) -> pl.DataFrame:
    df = add_trend_regression_features(df)
    df = add_high_low_features(df)
    df = add_breakout_features(df)
    return df
