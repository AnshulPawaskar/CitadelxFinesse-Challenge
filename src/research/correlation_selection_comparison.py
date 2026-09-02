"""Compare Method A (simple Top-10) vs. Method B (Top-30 -> correlation filter -> Top-10) across
several correlation thresholds, holding the alpha model fixed."""
from datetime import datetime

import polars as pl

from src.config import RESULTS_DIR, MAX_HOLDINGS
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data
from src.signals.composite import build_composite_score
from src.portfolio.correlation import select_with_correlation_filter, CORRELATION_THRESHOLDS
from src.portfolio.weighting import equal_weight
from src.research.backtest_runner import run_backtest_from_selection

FORMATION_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2025, 12, 31)
MODEL_NAME = "composite_quantitative"


def run(formation_date=FORMATION_DATE, end_date=END_DATE, model_name=MODEL_NAME, write_output: bool = True) -> pl.DataFrame:
    features = pl.read_parquet(RESULTS_DIR / "features.parquet")
    symbol_to_scrip_data = get_symbol_to_scrip_data()
    history = load_universe_history(symbol_to_scrip_data)

    signal = build_composite_score(features, formation_date, model_name)
    actual_formation_date = signal["Datetime"][0]

    rows = []
    for threshold in CORRELATION_THRESHOLDS:
        selected = equal_weight(select_with_correlation_filter(
            signal, history, actual_formation_date, n=MAX_HOLDINGS, correlation_threshold=threshold
        ))
        result = run_backtest_from_selection(selected, history, actual_formation_date, end_date)

        label = "No constraint" if threshold is None else str(threshold)
        rows.append({
            "CorrelationThreshold": label,
            "Symbols": ", ".join(selected["Symbol"].to_list()),
            **result["metrics"],
        })
        print(f"threshold={label}: CAGR={result['metrics']['CAGR']:.2%}, Sharpe={result['metrics']['Sharpe']:.2f}, "
              f"MaxDrawdown={result['metrics']['MaxDrawdown']:.2%}")

    comparison = pl.DataFrame(rows)
    if write_output:
        comparison.write_csv(RESULTS_DIR / "correlation_threshold_comparison.csv")

    return comparison


if __name__ == "__main__":
    run()
