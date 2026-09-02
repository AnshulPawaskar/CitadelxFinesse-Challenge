"""Load locally stored OHLCV parquet data into continuous per-symbol time series."""
import polars as pl

from src.config import DATA_DIR, TIMEFRAME


def list_available_periods(timeframe: str = TIMEFRAME) -> list[tuple[int, int]]:
    """Return sorted (year, half) tuples for which a data folder exists on disk."""
    root = DATA_DIR / timeframe
    periods = []
    for year_dir in root.iterdir():
        if not year_dir.is_dir():
            continue
        for half_dir in year_dir.iterdir():
            if half_dir.is_dir():
                periods.append((int(year_dir.name), int(half_dir.name)))
    return sorted(periods)


def load_symbol(scrip_data: str, timeframe: str = TIMEFRAME, periods: list[tuple[int, int]] | None = None) -> pl.DataFrame | None:
    """Concatenate one symbol's half-year parquet files (e.g. scrip_data='RELIANCE_EQ') into a sorted daily series."""
    periods = periods if periods is not None else list_available_periods(timeframe)
    frames = []
    for year, half in periods:
        file_path = DATA_DIR / timeframe / str(year) / str(half) / f"{scrip_data}.parquet"
        if file_path.exists():
            df = pl.read_parquet(file_path)
            if df.shape[0] > 0:
                frames.append(df)

    if not frames:
        return None

    full = pl.concat(frames, how="vertical")
    full = full.with_columns(pl.col("Datetime").str.to_datetime())
    return full.sort("Datetime").unique(subset=["Datetime"], keep="first")


def load_universe_history(symbol_to_scrip_data: dict[str, str], timeframe: str = TIMEFRAME) -> dict[str, pl.DataFrame]:
    """Load full history for every symbol in the mapping; reports symbols with no usable data."""
    periods = list_available_periods(timeframe)
    history = {}
    missing = []
    for symbol, scrip_data in symbol_to_scrip_data.items():
        df = load_symbol(scrip_data, timeframe, periods)
        if df is None:
            missing.append(symbol)
        else:
            history[symbol] = df

    if missing:
        print(f"Warning: no usable data found for {len(missing)} symbols: {missing}")

    return history
