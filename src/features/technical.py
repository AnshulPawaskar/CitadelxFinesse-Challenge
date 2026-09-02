"""Moving-average, oscillator and trend-strength indicators. Uses TA-Lib (C-compiled) for the
standard indicators — far faster than a hand-rolled Polars/Python rolling implementation."""
import polars as pl
import talib

from src.features.utils import talib_series

SMA_WINDOWS = [20, 50, 100, 200]
RSI_WINDOWS = {"RSI7": 7, "RSI14": 14, "RSI21": 21}


def add_moving_average_features(df: pl.DataFrame) -> pl.DataFrame:
    close = df["Close"].to_numpy().astype("float64")

    sma_cols = {f"MA{w}": talib.SMA(close, timeperiod=w) for w in SMA_WINDOWS}
    ema12 = talib.EMA(close, timeperiod=12)
    ema26 = talib.EMA(close, timeperiod=26)
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    df = df.with_columns([talib_series(name, values) for name, values in sma_cols.items()])
    df = df.with_columns([
        talib_series("EMA12", ema12),
        talib_series("EMA26", ema26),
        talib_series("MACD", macd),
        talib_series("MACD_Signal", macd_signal),
        talib_series("MACD_Hist", macd_hist),
    ])

    df = df.with_columns([(pl.col("Close") / pl.col(f"MA{w}")).alias(f"CloseToMA{w}") for w in SMA_WINDOWS])
    df = df.with_columns([
        (pl.col("MA20") / pl.col("MA50")).alias("MA20_MA50"),
        (pl.col("MA50") / pl.col("MA200")).alias("MA50_MA200"),
    ])
    return df


def add_rsi_features(df: pl.DataFrame) -> pl.DataFrame:
    close = df["Close"].to_numpy().astype("float64")
    rsi_cols = {name: talib.RSI(close, timeperiod=w) for name, w in RSI_WINDOWS.items()}
    return df.with_columns([talib_series(name, values) for name, values in rsi_cols.items()])


def add_adx_features(df: pl.DataFrame) -> pl.DataFrame:
    high = df["High"].to_numpy().astype("float64")
    low = df["Low"].to_numpy().astype("float64")
    close = df["Close"].to_numpy().astype("float64")
    adx14 = talib.ADX(high, low, close, timeperiod=14)
    plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
    minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)

    df = df.with_columns([
        talib_series("ADX14", adx14),
        talib_series("DI_Plus", plus_di),
        talib_series("DI_Minus", minus_di),
    ])
    return df.with_columns((pl.col("DI_Plus") - pl.col("DI_Minus")).alias("DI_Diff"))


def add_technical_features(df: pl.DataFrame) -> pl.DataFrame:
    df = add_moving_average_features(df)
    df = add_rsi_features(df)
    df = add_adx_features(df)
    return df
