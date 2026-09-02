"""Trade execution convention: buy at the next available trading day's open after the signal date."""
import polars as pl


def get_entry_price(symbol_history: pl.DataFrame, formation_date) -> tuple | None:
    """Returns (entry_date, entry_open_price) for the first trading day strictly after formation_date,
    or None if no such day exists in the local data."""
    future = symbol_history.filter(pl.col("Datetime") > formation_date).sort("Datetime")
    if future.height == 0:
        return None
    row = future.row(0, named=True)
    return row["Datetime"], row["Open"]
