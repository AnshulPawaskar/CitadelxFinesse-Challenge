"""Validate locally stored OHLCV data quality and produce a per-symbol audit report."""
import polars as pl

from src.config import RESULTS_DIR, TIMEFRAME
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data

# Single-day |return| beyond this is flagged as a possible unadjusted corporate action / bad tick.
LARGE_MOVE_THRESHOLD = 0.15


def validate_symbol(symbol: str, df: pl.DataFrame | None) -> dict:
    """Run data-quality checks on one symbol's full history and return a summary row."""
    if df is None or df.shape[0] == 0:
        return {
            "Symbol": symbol, "Status": "MISSING", "Rows": 0,
            "StartDate": None, "EndDate": None,
            "NullValues": None, "DuplicateDates": None,
            "InvalidOHLCRows": None, "NonPositivePrices": None,
            "NonPositiveVolumeRows": None, "LargeSingleDayMoves": None,
        }

    null_counts = df.null_count()
    total_nulls = int(null_counts.select(pl.sum_horizontal(pl.all())).item())

    invalid_ohlc = df.filter(
        (pl.col("High") < pl.col("Low"))
        | (pl.col("Close") > pl.col("High")) | (pl.col("Close") < pl.col("Low"))
        | (pl.col("Open") > pl.col("High")) | (pl.col("Open") < pl.col("Low"))
    )

    non_positive_prices = df.filter(
        (pl.col("Open") <= 0) | (pl.col("High") <= 0) | (pl.col("Low") <= 0) | (pl.col("Close") <= 0)
    )
    non_positive_volume = df.filter(pl.col("Volume") < 0)

    returns = df["Close"].pct_change()
    large_moves = returns.filter(returns.abs() > LARGE_MOVE_THRESHOLD)

    return {
        "Symbol": symbol,
        "Status": "OK",
        "Rows": df.shape[0],
        "StartDate": df["Datetime"].min(),
        "EndDate": df["Datetime"].max(),
        "NullValues": total_nulls,
        "DuplicateDates": int(df["Datetime"].is_duplicated().sum()),
        "InvalidOHLCRows": invalid_ohlc.shape[0],
        "NonPositivePrices": non_positive_prices.shape[0],
        "NonPositiveVolumeRows": non_positive_volume.shape[0],
        "LargeSingleDayMoves": large_moves.len(),
    }


def run_full_audit(timeframe: str = TIMEFRAME, write_report: bool = True) -> pl.DataFrame:
    """Validate every universe symbol's local history and optionally write results/data_audit_report.csv."""
    symbol_to_scrip_data = get_symbol_to_scrip_data()
    history = load_universe_history(symbol_to_scrip_data, timeframe)

    rows = [validate_symbol(symbol, history.get(symbol)) for symbol in symbol_to_scrip_data]
    report = pl.DataFrame(rows)

    if write_report:
        report.write_csv(RESULTS_DIR / "data_audit_report.csv")

    n_missing = report.filter(pl.col("Status") == "MISSING").shape[0]
    n_flagged = report.filter(
        (pl.col("InvalidOHLCRows") > 0) | (pl.col("NonPositivePrices") > 0) | (pl.col("NullValues") > 0)
    ).shape[0]
    print(f"Audit complete: {report.shape[0]} symbols checked, {n_missing} missing, {n_flagged} with data-quality flags.")

    return report


if __name__ == "__main__":
    run_full_audit()
