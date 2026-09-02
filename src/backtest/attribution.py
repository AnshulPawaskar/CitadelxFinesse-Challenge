"""Per-stock performance attribution for a completed static portfolio backtest."""
import polars as pl


def compute_stock_contributions(result: dict, history: dict[str, pl.DataFrame], end_date) -> pl.DataFrame:
    """For every held position, compute its individual return and $ contribution to total portfolio P&L."""
    rows = []
    for row in result["trade_log"].iter_rows(named=True):
        symbol = row["Symbol"]
        symbol_history = history[symbol].filter(pl.col("Datetime") <= end_date).sort("Datetime")
        exit_price = symbol_history["Close"][-1]
        exit_date = symbol_history["Datetime"][-1]

        net_entry_value = row["GrossValue"] - row["TransactionCost"]
        exit_value = row["Shares"] * exit_price
        pnl = exit_value - net_entry_value

        rows.append({
            "Symbol": symbol, "Weight": row["Weight"],
            "EntryDate": row["EntryDate"], "EntryPrice": row["EntryPrice"],
            "ExitDate": exit_date, "ExitPrice": exit_price,
            "Return": exit_price / row["EntryPrice"] - 1,
            "PnL": pnl,
        })

    contributions = pl.DataFrame(rows).sort("PnL", descending=True)
    total_pnl = contributions["PnL"].sum()
    return contributions.with_columns((pl.col("PnL") / total_pnl).alias("ContributionShare"))


def summarize_best_worst(contributions: pl.DataFrame) -> dict:
    best = contributions.row(0, named=True)
    worst = contributions.row(-1, named=True)
    return {
        "BestPerformer": best["Symbol"], "BestReturn": best["Return"], "BestPnL": best["PnL"],
        "WorstPerformer": worst["Symbol"], "WorstReturn": worst["Return"], "WorstPnL": worst["PnL"],
    }
