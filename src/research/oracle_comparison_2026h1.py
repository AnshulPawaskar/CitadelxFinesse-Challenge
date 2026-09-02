"""Standalone comparison: our actual (real, no-look-ahead) momentum_3m portfolio vs. a hindsight
"oracle" portfolio that cherry-picks the 10 best-performing stocks of 2026 H1 using their ACTUAL
future returns. This is NOT a valid trading strategy (it uses information that wasn't available on
the formation date) — it only exists to show the gap between our model and a theoretical best case.
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

FORMATION_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 6, 30)
MODEL_NAME = "momentum_3m"


def build_oracle_signal(history: dict[str, pl.DataFrame], formation_date, end_date) -> pl.DataFrame:
    """Rank every symbol by its ACTUAL realized return over [formation_date, end_date] using future
    data — a cheat only meant for comparison, never for real portfolio selection."""
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

    # --- Our real, no-look-ahead momentum_3m selection ---
    real_signal = build_composite_score(features, FORMATION_DATE, MODEL_NAME)
    real_formation_date = real_signal["Datetime"][0]
    real_selected = equal_weight(select_top_n(real_signal, n=MAX_HOLDINGS))
    real_result = build_static_portfolio(real_selected, history, real_formation_date, END_DATE, INITIAL_CAPITAL)
    real_metrics = compute_metrics(real_result["daily_nav"], INITIAL_CAPITAL)

    # --- Hindsight "oracle" selection (cheats using future returns) ---
    oracle_signal = build_oracle_signal(history, FORMATION_DATE, END_DATE)
    oracle_selected = equal_weight(select_top_n(oracle_signal, n=MAX_HOLDINGS))
    oracle_result = build_static_portfolio(oracle_selected, history, FORMATION_DATE, END_DATE, INITIAL_CAPITAL)
    oracle_metrics = compute_metrics(oracle_result["daily_nav"], INITIAL_CAPITAL)

    print(f"Our (real) momentum_3m picks: {real_selected['Symbol'].to_list()}")
    print(f"Oracle (hindsight-best) picks: {oracle_selected['Symbol'].to_list()}")

    comparison = pl.DataFrame([
        {"Portfolio": "Our momentum_3m model", **real_metrics},
        {"Portfolio": "Oracle (hindsight best)", **oracle_metrics},
    ])
    print("\n--- 2026 H1 Comparison ---")
    print(comparison.select(["Portfolio", "FinalValue", "NetPnL", "TotalReturn", "CAGR", "MaxDrawdown", "Sharpe"]))

    profit_gap = oracle_metrics["NetPnL"] - real_metrics["NetPnL"]
    print(f"\nProfit left on the table vs. a perfect-hindsight pick: Rs {profit_gap:,.0f} "
          f"({oracle_metrics['TotalReturn'] - real_metrics['TotalReturn']:.2%} extra return)")

    comparison.write_csv(RESULTS_DIR / "oracle_comparison_2026h1.csv")
    return comparison


if __name__ == "__main__":
    run()
