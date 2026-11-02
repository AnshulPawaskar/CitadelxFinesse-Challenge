"""Compare static, quarterly-rebalanced, and Markowitz-style portfolio strategies.

This script is intentionally reusable so the same comparison can be rerun with a different
model or formation date without retyping the one-off benchmark logic.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from src.backtest.costs import transaction_cost
from src.backtest.metrics import compute_metrics
from src.config import INITIAL_CAPITAL, MAX_HOLDINGS, RESULTS_DIR
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data
from src.portfolio.correlation import build_trailing_returns
from src.portfolio.selector import select_top_n
from src.portfolio.weighting import equal_weight
from src.research.backtest_runner import run_backtest_from_selection
from src.signals.composite import build_composite_score

DEFAULT_FORMATION_DATE = datetime(2021, 1, 1)
DEFAULT_END_DATE = datetime(2025, 12, 31)
DEFAULT_MODEL_NAME = "momentum_3m"
REBALANCE_PERIODS = ["1D", "1W", "1M", "3M", "6M"]


def _add_months(dt: datetime, months: int) -> datetime:
    """Add a signed number of calendar months without using pandas."""
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    return datetime(year, month, 1)


def generate_rebalance_dates(formation_date, end_date, rebalance_period: str = "3M") -> list[datetime]:
    """Generate rebalancing dates for a period such as 1D, 1W, 1M, 3M, or 6M."""
    period = rebalance_period.upper().replace(" ", "")
    if period == "1D":
        step = lambda d: d + __import__("datetime").timedelta(days=1)
    elif period == "1W":
        step = lambda d: d + __import__("datetime").timedelta(weeks=1)
    elif period == "1M":
        step = lambda d: _add_months(d, 1)
    elif period == "3M":
        step = lambda d: _add_months(d, 3)
    elif period == "6M":
        step = lambda d: _add_months(d, 6)
    else:
        raise ValueError(f"Unsupported rebalance_period '{rebalance_period}'. Use one of {REBALANCE_PERIODS}.")

    dates: list[datetime] = []
    current = formation_date
    while current <= end_date:
        dates.append(current)
        current = step(current)
    return dates


def _sorted_daily_dates(history: dict[str, pl.DataFrame]) -> list[datetime]:
    all_dates = {
        d.to_pydatetime() if hasattr(d, "to_pydatetime") else d
        for df in history.values()
        for d in df["Datetime"].to_list()
    }
    return sorted(all_dates)


def run_static_strategy(
    signal_df: pl.DataFrame,
    history: dict[str, pl.DataFrame],
    formation_date,
    end_date,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    selected = equal_weight(select_top_n(signal_df, n=MAX_HOLDINGS))
    return run_backtest_from_selection(selected, history, formation_date, end_date, initial_capital)


def run_quarterly_rebalance_strategy(
    features_df: pl.DataFrame,
    history: dict[str, pl.DataFrame],
    formation_date,
    end_date,
    model_name: str = DEFAULT_MODEL_NAME,
    initial_capital: float = INITIAL_CAPITAL,
    rebalance_period: str = "3M",
) -> dict:
    """Equal-weight portfolio that reselects on a configurable cadence using the latest signal cross-section."""
    rebalance_dates = generate_rebalance_dates(formation_date, end_date, rebalance_period)
    positions: dict[str, float] = {}
    nav_rows: list[tuple[datetime, float]] = []

    for d in _sorted_daily_dates(history):
        if d < formation_date or d > end_date:
            continue

        if d in rebalance_dates or not positions:
            signal_now = build_composite_score(features_df, d, model_name)
            candidates = equal_weight(select_top_n(signal_now, n=MAX_HOLDINGS))

            if not positions:
                market_value = initial_capital
            else:
                market_value = 0.0
                for sym, shares in positions.items():
                    close = history[sym].filter((pl.col("Datetime") <= d)).sort("Datetime")["Close"].to_list()[-1]
                    market_value += shares * close

            new_positions: dict[str, float] = {}
            for row in candidates.iter_rows(named=True):
                sym = row["Symbol"]
                price_series = history[sym].filter(pl.col("Datetime") <= d).sort("Datetime")
                if price_series.height == 0:
                    continue
                price = price_series["Close"].to_list()[-1]
                target = market_value * row["Weight"]
                target_after_cost = max(target - transaction_cost(target), 0.0)
                shares = target_after_cost / price if price else 0.0
                if shares > 0:
                    new_positions[sym] = shares
            positions = new_positions

        portfolio_value = 0.0
        for sym, shares in positions.items():
            close = history[sym].filter((pl.col("Datetime") <= d)).sort("Datetime")["Close"].to_list()[-1]
            portfolio_value += shares * close
        nav_rows.append((d, portfolio_value))

    nav = pl.DataFrame(nav_rows, schema=["Datetime", "PortfolioValue"], orient="row").sort("Datetime")
    metrics = compute_metrics(nav, initial_capital)
    return {
        "daily_nav": nav,
        "initial_capital": initial_capital,
        "metrics": metrics,
        "positions": positions,
    }


def build_min_variance_weights(
    symbol_list: list[str],
    history: dict[str, pl.DataFrame],
    as_of_date,
    max_weight: float = 0.20,
    window: int = 120,
) -> list[float]:
    """Minimum-variance weight vector from trailing daily-return covariance matrix."""
    returns = build_trailing_returns(history, symbol_list, as_of_date, window=window)
    if returns.is_empty() or returns.width == 0:
        return [1.0 / len(symbol_list)] * len(symbol_list)

    matrix = returns.to_numpy()
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.size == 0:
        return [1.0 / len(symbol_list)] * len(symbol_list)

    try:
        sigma = np.cov(matrix, rowvar=False)
    except Exception:
        sigma = np.eye(len(symbol_list))

    if sigma.shape == (0,) or sigma.shape == (1, 1):
        return [1.0 / len(symbol_list)] * len(symbol_list)

    inv_sigma = np.linalg.pinv(sigma)
    raw = inv_sigma @ np.ones(len(symbol_list))
    raw = np.where(np.isfinite(raw), raw, 0.0)
    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 0 or not np.isfinite(raw.sum()):
        return [1.0 / len(symbol_list)] * len(symbol_list)

    weights = raw / raw.sum()
    weights = np.clip(weights, 0.0, max_weight)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        return [1.0 / len(symbol_list)] * len(symbol_list)
    weights = weights / total
    return [float(w) for w in weights]


def run_markowitz_strategy(
    signal_df: pl.DataFrame,
    history: dict[str, pl.DataFrame],
    formation_date,
    end_date,
    initial_capital: float = INITIAL_CAPITAL,
    max_weight: float = 0.20,
) -> dict:
    """Static buy-and-hold portfolio using minimum-variance weights from the formation cross-section."""
    selected = equal_weight(select_top_n(signal_df, n=MAX_HOLDINGS))
    symbol_list = selected["Symbol"].to_list()
    weights = build_min_variance_weights(symbol_list, history, formation_date, max_weight=max_weight)

    trade_rows: list[dict] = []
    for symbol, weight in zip(symbol_list, weights):
        symbol_history = history[symbol].filter(pl.col("Datetime") >= formation_date).sort("Datetime")
        if symbol_history.height == 0:
            continue
        entry_date = symbol_history["Datetime"][0]
        entry_price = symbol_history["Open"][0]
        position_value = initial_capital * weight
        position_after_cost = max(position_value - transaction_cost(position_value), 0.0)
        shares = position_after_cost / entry_price if entry_price else 0.0
        trade_rows.append(
            {"Symbol": symbol, "Weight": weight, "EntryDate": entry_date, "EntryPrice": entry_price, "Shares": shares}
        )

    if not trade_rows:
        raise ValueError("No valid symbols available to build a Markowitz portfolio.")

    nav_rows: list[tuple[datetime, float]] = []
    for symbol in symbol_list:
        sym_history = history[symbol].filter(pl.col("Datetime") >= formation_date).sort("Datetime")
        if sym_history.height == 0:
            continue
        pos_map = {row["Symbol"]: row for row in trade_rows}
        # Build a daily portfolio value from the fixed-position buy-and-hold profile.
        # Each symbol's position is constant from its own entry date to the end of the sample.
        symbol_trade = pos_map[symbol]
        if symbol_trade["Shares"] <= 0:
            continue

        daily = sym_history.select(["Datetime", "Close"])
        daily = daily.with_columns((pl.lit(symbol_trade["Shares"]) * pl.col("Close")).alias(symbol))
        nav_rows.append(daily.select(["Datetime", symbol]))

    if not nav_rows:
        raise ValueError("Unable to construct daily Markowitz portfolio values.")

    nav = nav_rows[0]
    for frame in nav_rows[1:]:
        nav = nav.join(frame, on="Datetime", how="inner")
    nav = nav.with_columns(pl.sum_horizontal([c for c in nav.columns if c != "Datetime"]).alias("PortfolioValue")).sort("Datetime")
    metrics = compute_metrics(nav.select(["Datetime", "PortfolioValue"]), initial_capital)
    return {
        "daily_nav": nav.select(["Datetime", "PortfolioValue"]),
        "initial_capital": initial_capital,
        "metrics": metrics,
        "weights": {row["Symbol"]: row["Weight"] for row in trade_rows},
    }


def run_comparison(
    formation_date=DEFAULT_FORMATION_DATE,
    end_date=DEFAULT_END_DATE,
    model_name: str = DEFAULT_MODEL_NAME,
    output_csv: str | None = None,
    rebalance_periods: list[str] | None = None,
) -> pl.DataFrame:
    """Compare static, Markowitz, and rebalanced strategies across the configured rebalance cadence list."""
    features = pl.read_parquet(RESULTS_DIR / "features.parquet")
    history = load_universe_history(get_symbol_to_scrip_data())
    signal = build_composite_score(features, formation_date, model_name)

    periods = rebalance_periods or REBALANCE_PERIODS
    rows = []

    static_result = run_static_strategy(signal, history, formation_date, end_date)
    rows.append({
        "Strategy": "static",
        **_metrics_to_row(static_result["metrics"]),
    })

    markowitz_result = run_markowitz_strategy(signal, history, formation_date, end_date)
    rows.append({
        "Strategy": "markowitz",
        **_metrics_to_row(markowitz_result["metrics"]),
    })

    for period in periods:
        rebalance_result = run_quarterly_rebalance_strategy(
            features,
            history,
            formation_date,
            end_date,
            model_name=model_name,
            rebalance_period=period,
        )
        rows.append({
            "Strategy": f"rebalance_{period.lower()}",
            **_metrics_to_row(rebalance_result["metrics"]),
        })

    comparison = pl.DataFrame(rows).sort("Sharpe", descending=True)
    output_path = RESULTS_DIR / (output_csv or "strategy_comparison.csv")
    comparison.write_csv(output_path)

    print("\nStrategy comparison")
    print(comparison)
    print(f"\nSaved to {output_path}")
    return comparison


def _metrics_to_row(metrics: dict) -> dict:
    return {
        "InitialCapital": metrics["InitialCapital"],
        "FinalValue": metrics["FinalValue"],
        "NetPnL": metrics["NetPnL"],
        "TotalReturn": metrics["TotalReturn"],
        "CAGR": metrics["CAGR"],
        "AnnualizedVolatility": metrics["AnnualizedVolatility"],
        "MaxDrawdown": metrics["MaxDrawdown"],
        "Sharpe": metrics["Sharpe"],
        "Sortino": metrics["Sortino"],
        "GainLossRatio": metrics["GainLossRatio"],
        "WinRate": metrics["WinRate"],
    }


if __name__ == "__main__":
    run_comparison(rebalance_periods=REBALANCE_PERIODS)
