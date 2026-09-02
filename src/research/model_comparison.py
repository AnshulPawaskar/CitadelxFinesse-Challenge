"""Backtest every candidate alpha model (MODEL 1-8) at a single formation date and compare
performance — this doubles as the feature-ablation test ("which information actually helps?").
"""
from datetime import datetime

import polars as pl

from src.config import RESULTS_DIR
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data
from src.signals.composite import build_composite_score, MODEL_DEFINITIONS
from src.research.backtest_runner import run_static_backtest

FORMATION_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2025, 12, 31)


def run(formation_date=FORMATION_DATE, end_date=END_DATE, write_output: bool = True) -> pl.DataFrame:
    features = pl.read_parquet(RESULTS_DIR / "features.parquet")
    symbol_to_scrip_data = get_symbol_to_scrip_data()
    history = load_universe_history(symbol_to_scrip_data)

    rows = []
    for model_name in MODEL_DEFINITIONS:
        signal = build_composite_score(features, formation_date, model_name)
        result = run_static_backtest(signal, history, end_date)
        rows.append({"Model": model_name, "Symbols": ", ".join(result["selected"]["Symbol"].to_list()), **result["metrics"]})
        print(f"{model_name}: CAGR={result['metrics']['CAGR']:.2%}, Sharpe={result['metrics']['Sharpe']:.2f}, "
              f"MaxDrawdown={result['metrics']['MaxDrawdown']:.2%}")

    comparison = pl.DataFrame(rows).sort("NetPnL", descending=True)
    if write_output:
        comparison.write_csv(RESULTS_DIR / "model_comparison.csv")

    return comparison


if __name__ == "__main__":
    run()
