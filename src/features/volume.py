"""Volume-based confirmation features."""
import polars as pl
import talib

from src.features.utils import rolling_ols_trend, talib_series

VOLUME_MA_WINDOWS = [20, 60]


def add_volume_features(df: pl.DataFrame) -> pl.DataFrame:
    close = df["Close"].to_numpy().astype("float64")
    volume = df["Volume"].to_numpy().astype("float64")
    obv = talib.OBV(close, volume)
    df = df.with_columns(talib_series("OBV", obv))
    df = rolling_ols_trend(df, "OBV", window=20, prefix="OBV20")
    df = df.rename({"OBV20_Slope": "OBV_Slope"}).drop(["OBV20_R2", "OBV20_TrendStrength"])

    vol_ma_exprs = [pl.col("Volume").rolling_mean(window_size=w).alias(f"VolMA{w}") for w in VOLUME_MA_WINDOWS]
    df = df.with_columns(vol_ma_exprs)
    df = df.with_columns([(pl.col("Volume") / pl.col(f"VolMA{w}")).alias(f"VolumeToMA{w}") for w in VOLUME_MA_WINDOWS])

    df = df.with_columns(
        ((pl.col("Volume") - pl.col("VolMA60")) / pl.col("Volume").rolling_std(window_size=60)).alias("VolumeZScore")
    )

    df = df.with_columns((pl.col("Close") / pl.col("Close").shift(1) - 1).alias("_return"))
    df = df.with_columns(
        pl.rolling_corr(pl.col("_return"), pl.col("Volume").diff(), window_size=20).alias("PriceVolumeCorr20")
    )
    return df.drop("_return")
