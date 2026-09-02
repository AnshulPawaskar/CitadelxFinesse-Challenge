"""A single static portfolio using the classic 12-1 momentum signal only (MODEL 1), equal-weighted.

Formation date: 2021-01-01 (configurable below). Hold through 2025-12-31 with zero rebalancing.
"""
from datetime import datetime

import polars as pl

from src.config import RESULTS_DIR
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data
from src.signals.momentum import get_momentum_signal
from src.research.backtest_runner import run_static_backtest
from src.backtest.attribution import compute_stock_contributions, summarize_best_worst

FORMATION_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2025, 12, 31)


def run(formation_date=FORMATION_DATE, end_date=END_DATE):
    features = pl.read_parquet(RESULTS_DIR / "features.parquet")
    symbol_to_scrip_data = get_symbol_to_scrip_data()
    history = load_universe_history(symbol_to_scrip_data)

    signal = get_momentum_signal(features, formation_date)
    print(f"Requested formation date: {formation_date}, actual (last available) trading day: {signal['Datetime'][0]}")

    result = run_static_backtest(signal, history, end_date)
    print(result["selected"])

    print("\n--- Performance ---")
    for key, value in result["metrics"].items():
        print(f"{key}: {value}")
    print(f"num_rebalances: {result['num_rebalances']}")

    contributions = compute_stock_contributions(result, history, end_date)
    print("\n--- Stock Contributions ---")
    print(contributions)
    print(summarize_best_worst(contributions))

    result["trade_log"].write_csv(RESULTS_DIR / "trade_log.csv")
    result["daily_nav"].write_csv(RESULTS_DIR / "daily_nav.csv")
    contributions.write_csv(RESULTS_DIR / "stock_contributions.csv")

    return result


if __name__ == "__main__":
    run()
