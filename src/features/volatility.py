"""Volatility and drawdown risk features."""
import numpy as np
import polars as pl
import talib

from src.features.utils import talib_series

TRADING_DAYS_PER_YEAR = 252
VOL_WINDOWS = {"Vol20": 20, "Vol60": 60, "Vol126": 126}
DRAWDOWN_WINDOWS = [60, 126, 252]


def _rolling_max_drawdown(close: np.ndarray, window: int) -> np.ndarray:
    """Worst peak-to-trough drawdown observed within each trailing window (vectorized, no per-row loop)."""
    n = len(close)
    result = np.full(n, np.nan)
    if n < window:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(close, window)
    running_peak = np.maximum.accumulate(windows, axis=1)
    drawdowns = windows / running_peak - 1
    result[window - 1:] = drawdowns.min(axis=1)
    return result


def add_volatility_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns((pl.col("Close") / pl.col("Close").shift(1) - 1).alias("_return"))

    vol_exprs = [
        (pl.col("_return").rolling_std(window_size=w) * TRADING_DAYS_PER_YEAR**0.5).alias(name)
        for name, w in VOL_WINDOWS.items()
    ]
    df = df.with_columns(vol_exprs)

    # Downside (semi-)deviation: only negative-return days contribute, annualized like the volatility above.
    df = df.with_columns(
        pl.when(pl.col("_return") < 0).then(pl.col("_return")).otherwise(0.0).alias("_downside_return")
    )
    df = df.with_columns(
        ((pl.col("_downside_return") ** 2).rolling_mean(window_size=60) ** 0.5 * TRADING_DAYS_PER_YEAR**0.5)
        .alias("DownsideVolatility60")
    )

    high = df["High"].to_numpy().astype("float64")
    low = df["Low"].to_numpy().astype("float64")
    close = df["Close"].to_numpy().astype("float64")
    atr14 = talib.ATR(high, low, close, timeperiod=14)
    df = df.with_columns(talib_series("ATR14", atr14))
    df = df.with_columns((pl.col("ATR14") / pl.col("Close")).alias("ATR14_Close"))

    for window in DRAWDOWN_WINDOWS:
        df = df.with_columns(pl.Series(f"MaxDrawdown{window}D", _rolling_max_drawdown(close, window)))

    return df.drop(["_return", "_downside_return"])
