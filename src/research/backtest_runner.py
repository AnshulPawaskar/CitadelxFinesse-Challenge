"""Shared helper to take a scored cross-section from formation through a completed backtest."""
import polars as pl

from src.config import INITIAL_CAPITAL, MAX_HOLDINGS
from src.portfolio.selector import select_top_n
from src.portfolio.weighting import equal_weight
from src.backtest.static_portfolio import build_static_portfolio
from src.backtest.metrics import compute_metrics


def run_backtest_from_selection(
    selected: pl.DataFrame,
    history: dict[str, pl.DataFrame],
    formation_date,
    end_date,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """selected must already have columns [Symbol, Weight] (post selection + weighting)."""
    result = build_static_portfolio(selected, history, formation_date, end_date, initial_capital)
    result["selected"] = selected
    result["metrics"] = compute_metrics(result["daily_nav"], initial_capital)
    return result


def run_static_backtest(
    signal_df: pl.DataFrame,
    history: dict[str, pl.DataFrame],
    end_date,
    initial_capital: float = INITIAL_CAPITAL,
    max_holdings: int = MAX_HOLDINGS,
) -> dict:
    """signal_df must have columns [Symbol, Datetime, AlphaScore] for a single formation cross-section."""
    formation_date = signal_df["Datetime"][0]
    selected = equal_weight(select_top_n(signal_df, n=max_holdings))
    return run_backtest_from_selection(selected, history, formation_date, end_date, initial_capital)
