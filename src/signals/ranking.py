"""Cross-sectional rank/percentile utilities for turning raw scores into comparable [0, 1] scores."""
import polars as pl


def to_percentile(df: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """Convert a raw score column (single cross-section, one row per symbol) into a 0..1 percentile
    rank where 1.0 = best. Ties are averaged."""
    n = df.height
    if n <= 1:
        return df.with_columns(pl.lit(1.0).alias(f"{score_col}_Percentile"))
    return df.with_columns(
        ((pl.col(score_col).rank(method="average") - 1) / (n - 1)).alias(f"{score_col}_Percentile")
    )
