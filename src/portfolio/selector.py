"""Select the final holdings from a ranked signal cross-section."""
import polars as pl

from src.config import MAX_HOLDINGS


def select_top_n(signal_df: pl.DataFrame, n: int = MAX_HOLDINGS, score_col: str = "AlphaScore") -> pl.DataFrame:
    """Simplest selection method — take the top n symbols by score (Method A)."""
    if signal_df.height < n:
        print(f"Warning: only {signal_df.height} eligible symbols available, fewer than requested {n}.")
    return signal_df.sort(score_col, descending=True).head(n)
