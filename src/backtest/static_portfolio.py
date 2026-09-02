"""Static (buy-once, hold, zero rebalancing) portfolio construction and NAV tracking."""
import polars as pl

from src.backtest.costs import transaction_cost
from src.backtest.execution import get_entry_price
from src.backtest.validation import validate_no_rebalancing, count_rebalances


def build_static_portfolio(
    selected: pl.DataFrame,
    history: dict[str, pl.DataFrame],
    formation_date,
    end_date,
    initial_capital: float,
) -> dict:
    """Buy the selected {Symbol, Weight} holdings once at the next trading day's open after
    formation_date, hold unchanged through end_date, and return the trade log + daily NAV."""
    positions = []
    for row in selected.iter_rows(named=True):
        symbol, weight = row["Symbol"], row["Weight"]
        entry = get_entry_price(history[symbol], formation_date)
        if entry is None:
            print(f"Warning: no tradable entry price for {symbol} after {formation_date}, skipping.")
            continue

        entry_date, entry_price = entry
        position_value = initial_capital * weight
        cost = transaction_cost(position_value)
        shares = (position_value - cost) / entry_price

        positions.append({
            "Symbol": symbol, "Weight": weight, "EntryDate": entry_date, "EntryPrice": entry_price,
            "Shares": shares, "GrossValue": position_value, "TransactionCost": cost,
        })

    if not positions:
        raise ValueError("No positions could be established for this formation date.")

    trade_log = pl.DataFrame(positions)
    validate_no_rebalancing(trade_log)
    portfolio_start = trade_log["EntryDate"].max()  # NAV begins once every position is filled

    nav_frames = []
    for pos in positions:
        window = history[pos["Symbol"]].filter(
            (pl.col("Datetime") >= portfolio_start) & (pl.col("Datetime") <= end_date)
        ).select(["Datetime", "Close"])
        window = window.with_columns((pos["Shares"] * pl.col("Close")).alias(pos["Symbol"]))
        nav_frames.append(window.select(["Datetime", pos["Symbol"]]))

    nav = nav_frames[0]
    for frame in nav_frames[1:]:
        nav = nav.join(frame, on="Datetime", how="inner")

    value_cols = [c for c in nav.columns if c != "Datetime"]
    nav = nav.with_columns(pl.sum_horizontal(value_cols).alias("PortfolioValue")).sort("Datetime")

    gross_invested = trade_log["GrossValue"].sum()
    total_cost = trade_log["TransactionCost"].sum()

    return {
        "trade_log": trade_log,
        "daily_nav": nav.select(["Datetime", "PortfolioValue"]),
        "formation_date": formation_date,
        "portfolio_start": portfolio_start,
        "initial_capital": initial_capital,
        "gross_invested": gross_invested,
        "transaction_costs": total_cost,
        "net_invested": gross_invested - total_cost,
        "num_rebalances": count_rebalances(trade_log),
    }
