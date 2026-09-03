"""Relative strength vs. a benchmark return index.

No local Nifty index-level OHLCV series exists in this repo, only constituent lists, so
build_synthetic_benchmark() uses an equal-weighted universe proxy until real index data is sourced.
"""
import polars as pl

RS_WINDOWS = {"RS21": 21, "RS63": 63, "RS126": 126, "RS252": 252}


def build_synthetic_benchmark(history: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Equal-weighted average daily return across the universe -> a benchmark index level."""
    frames = [df.select(["Datetime", "Close"]).rename({"Close": symbol}) for symbol, df in history.items()]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.join(frame, on="Datetime", how="full", coalesce=True)
    merged = merged.sort("Datetime")

    price_cols = [c for c in merged.columns if c != "Datetime"]
    daily_returns = merged.select(
        [pl.col("Datetime")]
        + [(pl.col(c) / pl.col(c).shift(1) - 1).alias(c) for c in price_cols]
    )
    avg_return = daily_returns.select(
        pl.col("Datetime"),
        pl.mean_horizontal(price_cols).fill_null(0.0).alias("_avg_return"),
    )
    return avg_return.with_columns(
        (pl.col("_avg_return") + 1).cum_prod().alias("BenchmarkIndex")
    ).select(["Datetime", "BenchmarkIndex"])


def add_relative_strength_features(df: pl.DataFrame, benchmark: pl.DataFrame) -> pl.DataFrame:
    """benchmark must have columns [Datetime, BenchmarkIndex]."""
    df = df.join(benchmark, on="Datetime", how="left")
    exprs = []
    for name, w in RS_WINDOWS.items():
        stock_ret = pl.col("Close") / pl.col("Close").shift(w) - 1
        bench_ret = pl.col("BenchmarkIndex") / pl.col("BenchmarkIndex").shift(w) - 1
        exprs.append((stock_ret - bench_ret).alias(name))
        exprs.append(
            pl.when(bench_ret != 0).then(stock_ret / bench_ret).otherwise(None).alias(f"{name}_Ratio")
        )
    return df.with_columns(exprs)
