"""Compares our official portfolio (same picks as src/competition/final_submission.py) against a
hindsight "oracle" that picks the 10 best-performing stocks of 2026 H1 using their actual future
returns. The oracle isn't a valid strategy, it's a theoretical ceiling for comparison only.
"""
from datetime import datetime

import polars as pl

from src.config import RESULTS_DIR, MAX_HOLDINGS, INITIAL_CAPITAL
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data
from src.signals.composite import build_composite_score
from src.portfolio.selector import select_top_n
from src.portfolio.weighting import equal_weight
from src.backtest.static_portfolio import build_static_portfolio
from src.backtest.metrics import compute_metrics
from src.backtest.execution import get_entry_price
from src.competition.final_submission import FORMATION_DATE as OFFICIAL_FORMATION_DATE, MODEL_NAME

ENTRY_DATE = datetime(2025, 12, 31)  # same OOS entry boundary used in final_submission.py
END_DATE = datetime(2026, 6, 30)


def build_oracle_signal(history: dict[str, pl.DataFrame], formation_date, end_date) -> pl.DataFrame:
    """Ranks every symbol by its actual realized return over [formation_date, end_date] using
    future data. Only for comparison, never for real selection."""
    rows = []
    for symbol, df in history.items():
        entry = get_entry_price(df, formation_date)
        if entry is None:
            continue
        entry_date, entry_price = entry
        window = df.filter((pl.col("Datetime") >= entry_date) & (pl.col("Datetime") <= end_date)).sort("Datetime")
        if window.height == 0:
            continue
        exit_price = window["Close"][-1]
        rows.append({"Symbol": symbol, "Datetime": formation_date, "AlphaScore": exit_price / entry_price - 1})

    return pl.DataFrame(rows)


def run():
    features = pl.read_parquet(RESULTS_DIR / "features.parquet")
    history = load_universe_history(get_symbol_to_scrip_data())

    # our official portfolio, re-entered with fresh capital at the OOS boundary
    real_signal = build_composite_score(features, OFFICIAL_FORMATION_DATE, MODEL_NAME)
    real_selected = equal_weight(select_top_n(real_signal, n=MAX_HOLDINGS))
    real_result = build_static_portfolio(real_selected, history, ENTRY_DATE, END_DATE, INITIAL_CAPITAL)
    real_metrics = compute_metrics(real_result["daily_nav"], INITIAL_CAPITAL)

    # hindsight oracle selection, same entry timing/capital
    oracle_signal = build_oracle_signal(history, ENTRY_DATE, END_DATE)
    oracle_selected = equal_weight(select_top_n(oracle_signal, n=MAX_HOLDINGS))
    oracle_result = build_static_portfolio(oracle_selected, history, ENTRY_DATE, END_DATE, INITIAL_CAPITAL)
    oracle_metrics = compute_metrics(oracle_result["daily_nav"], INITIAL_CAPITAL)

    print(f"Our official portfolio (formed {OFFICIAL_FORMATION_DATE.date()}): {real_selected['Symbol'].to_list()}")
    print(f"Oracle (hindsight-best for 2026 H1): {oracle_selected['Symbol'].to_list()}")

    comparison = pl.DataFrame([
        {"Portfolio": "Our official portfolio", **real_metrics},
        {"Portfolio": "Oracle (hindsight best)", **oracle_metrics},
    ])
    print("\n--- 2026 H1 Comparison (both: fresh Rs 1,00,00,000, entered 2026-01-01) ---")
    print(comparison.select(["Portfolio", "FinalValue", "NetPnL", "TotalReturn", "CAGR", "MaxDrawdown", "Sharpe"]))

    profit_gap = oracle_metrics["NetPnL"] - real_metrics["NetPnL"]
    print(f"\nProfit left on the table vs. a perfect-hindsight pick: Rs {profit_gap:,.0f} "
          f"({oracle_metrics['TotalReturn'] - real_metrics['TotalReturn']:.2%} extra return)")

    comparison.write_csv(RESULTS_DIR / "oracle_comparison_2026h1.csv")
    return comparison


if __name__ == "__main__":
    run()
