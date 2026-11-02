from datetime import datetime

import polars as pl

from src.config import INITIAL_CAPITAL, RESULTS_DIR
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data
from src.research.strategy_comparison import (
    run_markowitz_strategy,
    run_quarterly_rebalance_strategy,
    run_static_strategy,
)
from src.signals.composite import build_composite_score

PERIODS = [
    ("2021-01-01", "2021-01-01", "2021-06-30"),
    ("2021-07-01", "2021-07-01", "2021-12-31"),
    ("2022-01-01", "2022-01-01", "2022-06-30"),
    ("2022-07-01", "2022-07-01", "2022-12-31"),
    ("2023-01-01", "2023-01-01", "2023-06-30"),
    ("2023-07-01", "2023-07-01", "2023-12-31"),
    ("2024-01-01", "2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-07-01", "2025-12-31"),
    ("2026-01-01", "2026-01-01", "2026-06-30"),
    ("2026-07-01", "2026-07-01", "2026-10-31"),
]

STRATEGIES = ["static", "rebalance_1m", "rebalance_3m", "rebalance_6m", "markowitz"]


def run_matrix() -> pl.DataFrame:
    features = pl.read_parquet(RESULTS_DIR / "features.parquet")
    history = load_universe_history(get_symbol_to_scrip_data())
    rows = []

    for portfolio_creation_date, investment_date, investment_end_date in PERIODS:
        creation_dt = datetime.strptime(portfolio_creation_date, "%Y-%m-%d")
        end_dt = datetime.strptime(investment_end_date, "%Y-%m-%d")
        signal = build_composite_score(features, creation_dt, "momentum_3m")

        for strategy_name in STRATEGIES:
            if strategy_name == "static":
                result = run_static_strategy(signal, history, creation_dt, end_dt, INITIAL_CAPITAL)
            elif strategy_name == "markowitz":
                result = run_markowitz_strategy(signal, history, creation_dt, end_dt, INITIAL_CAPITAL)
            else:
                period_code = strategy_name.split("_")[1].upper()
                result = run_quarterly_rebalance_strategy(
                    features,
                    history,
                    creation_dt,
                    end_dt,
                    model_name="momentum_3m",
                    initial_capital=INITIAL_CAPITAL,
                    rebalance_period=period_code,
                )

            metrics = result["metrics"]
            rows.append({
                "PortfolioCreationDate": portfolio_creation_date,
                "InvestmentDate": investment_date,
                "InvestmentEndDate": investment_end_date,
                "Strategy": strategy_name,
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
            })

    df = pl.DataFrame(rows)
    output_path = RESULTS_DIR / "period_strategy_matrix.csv"
    df.write_csv(output_path)
    print(df)
    print(f"\nSaved to {output_path}")
    return df


if __name__ == "__main__":
    run_matrix()
