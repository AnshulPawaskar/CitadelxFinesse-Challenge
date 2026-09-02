"""Standard performance metrics computed from a daily NAV series."""
import numpy as np
import polars as pl

from src.config import RISK_FREE_RATE

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(daily_nav: pl.DataFrame, initial_capital: float, risk_free_rate: float = RISK_FREE_RATE) -> dict:
    nav = daily_nav.sort("Datetime")
    values = nav["PortfolioValue"].to_numpy()
    dates = nav["Datetime"].to_numpy()

    total_return = values[-1] / values[0] - 1
    n_days = (dates[-1] - dates[0]).astype("timedelta64[D]").astype(int)
    years = max(n_days / 365.25, 1e-9)
    cagr = (values[-1] / values[0]) ** (1 / years) - 1

    daily_returns = values[1:] / values[:-1] - 1
    ann_vol = daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    running_max = np.maximum.accumulate(values)
    max_drawdown = (values / running_max - 1).min()

    excess_daily = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_std = excess_daily.std(ddof=1)
    sharpe = (excess_daily.mean() / excess_std) * np.sqrt(TRADING_DAYS_PER_YEAR) if excess_std > 0 else float("nan")

    downside = excess_daily[excess_daily < 0]
    sortino = (
        (excess_daily.mean() / downside.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if len(downside) > 1 and downside.std(ddof=1) > 0 else float("nan")
    )

    gains, losses = daily_returns[daily_returns > 0], daily_returns[daily_returns < 0]
    gain_loss_ratio = (gains.mean() / abs(losses.mean())) if len(gains) > 0 and len(losses) > 0 else float("nan")
    win_rate = (daily_returns > 0).mean()

    return {
        "InitialCapital": initial_capital,
        "FinalValue": float(values[-1]),
        "NetPnL": float(values[-1] - initial_capital),
        "TotalReturn": float(total_return),
        "CAGR": float(cagr),
        "AnnualizedVolatility": float(ann_vol),
        "MaxDrawdown": float(max_drawdown),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "GainLossRatio": float(gain_loss_ratio),
        "WinRate": float(win_rate),
    }
