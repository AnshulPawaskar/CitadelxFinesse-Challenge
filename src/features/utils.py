"""Shared rolling-window math not natively available in Polars (closed-form, no per-row Python loops)."""
import numpy as np
import polars as pl


def talib_series(name: str, values: np.ndarray) -> pl.Series:
    """Wrap a TA-Lib output array as a Polars Series, converting its NaN warm-up values to proper
    nulls (TA-Lib uses NaN, but Polars' `is_null`/`is_not_null` filters don't treat NaN as null —
    without this, NaN scores could silently leak past null-filters used during signal selection)."""
    return pl.Series(name, values).fill_nan(None)


def rolling_ols_trend(df: pl.DataFrame, y_col: str, window: int, prefix: str) -> pl.DataFrame:
    """Rolling OLS slope/R^2 of y_col against a 0..window-1 time index, using closed-form
    rolling-sum formulas (fully vectorized, causal — each row only uses its own trailing window)."""
    n = df.height
    idx = pl.arange(0, n, eager=True).cast(pl.Float64)
    y = df[y_col]
    iy = idx * y

    sum_y = y.rolling_sum(window_size=window)
    sum_iy = iy.rolling_sum(window_size=window)
    sum_y2 = (y * y).rolling_sum(window_size=window)

    start_idx = idx - (window - 1)
    mean_x = (window - 1) / 2
    var_x = (window**2 - 1) / 12

    mean_y = sum_y / window
    cov_xy = (sum_iy - start_idx * sum_y) / window - mean_x * mean_y
    var_y = sum_y2 / window - mean_y**2

    slope = cov_xy / var_x
    r2 = pl.when(var_y > 0).then((cov_xy**2) / (var_x * var_y)).otherwise(0.0)

    return df.with_columns([
        slope.alias(f"{prefix}_Slope"),
        r2.alias(f"{prefix}_R2"),
        (slope * r2).alias(f"{prefix}_TrendStrength"),
    ])
