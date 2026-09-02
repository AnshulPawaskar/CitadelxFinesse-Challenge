"""Orchestrates per-symbol feature computation across the universe, in parallel."""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

from src.config import RESULTS_DIR
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data
from src.features.momentum import add_momentum_features
from src.features.relative_strength import add_relative_strength_features, build_synthetic_benchmark
from src.features.technical import add_technical_features
from src.features.trend import add_trend_features
from src.features.volatility import add_volatility_features
from src.features.volume import add_volume_features


def compute_symbol_features(symbol: str, df: pl.DataFrame, benchmark: pl.DataFrame) -> pl.DataFrame:
    """Apply every feature module to one symbol's OHLCV history and tag the result with its symbol."""
    df = add_momentum_features(df)
    df = add_relative_strength_features(df, benchmark)
    df = add_trend_features(df)
    df = add_technical_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = df.with_columns(
        (pl.col("R126") * (pl.col("Volume") / pl.col("VolMA60"))).alias("VolumeConfirmedMomentum")
    )
    return df.with_columns(pl.lit(symbol).alias("Symbol"))


def _compute_worker(args: tuple[str, pl.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    symbol, df, benchmark = args
    return compute_symbol_features(symbol, df, benchmark)


def compute_universe_features(
    history: dict[str, pl.DataFrame],
    benchmark: pl.DataFrame,
    parallel: bool = True,
    max_workers: int | None = None,
) -> pl.DataFrame:
    """Compute features for every symbol, in parallel (CPU-bound TA-Lib/Polars work benefits from
    process-based parallelism since it releases the GIL only partially)."""
    tasks = [(symbol, df, benchmark) for symbol, df in history.items()]
    results = []

    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_compute_worker, task): task[0] for task in tasks}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error computing features for {symbol}: {e}")
    else:
        for task in tasks:
            results.append(_compute_worker(task))

    return pl.concat(results, how="vertical")


def run_feature_pipeline(write_output: bool = True) -> pl.DataFrame:
    symbol_to_scrip_data = get_symbol_to_scrip_data()
    history = load_universe_history(symbol_to_scrip_data)

    print(f"Building synthetic benchmark from {len(history)} symbols...")
    benchmark = build_synthetic_benchmark(history)

    start = time.time()
    features = compute_universe_features(history, benchmark)
    print(f"Computed features for {features['Symbol'].n_unique()} symbols in {time.time() - start:.1f}s")

    if write_output:
        features.write_parquet(RESULTS_DIR / "features.parquet")

    return features


if __name__ == "__main__":
    run_feature_pipeline()
