"""MODEL 1 — classic 12-1 momentum signal, read straight from the precomputed feature panel."""
import polars as pl


def get_momentum_signal(features_df: pl.DataFrame, formation_date) -> pl.DataFrame:
    """Cross-sectional 12-1 momentum score as of the last available trading day <= formation_date
    (never looks past formation_date, so this is safe to use for portfolio selection)."""
    available = features_df.filter(pl.col("Datetime") <= formation_date)
    if available.height == 0:
        raise ValueError(f"No feature data available on or before {formation_date}.")

    last_date = available["Datetime"].max()
    cross_section = available.filter(
        (pl.col("Datetime") == last_date) & pl.col("Momentum12_1").is_not_null()
    )
    return cross_section.select(["Symbol", "Datetime", "Momentum12_1"]).rename({"Momentum12_1": "AlphaScore"})
