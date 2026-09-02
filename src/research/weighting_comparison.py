"""Compare all 5 portfolio weighting methods, holding the alpha model and stock selection fixed."""
from datetime import datetime

import polars as pl

from src.config import RESULTS_DIR, MAX_HOLDINGS
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data
from src.signals.composite import build_composite_score
from src.portfolio.selector import select_top_n
from src.portfolio.weighting import WEIGHTING_METHODS, attach_volatility
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
    candidates = select_top_n(signal, n=MAX_HOLDINGS)
    candidates = attach_volatility(candidates, features, actual_formation_date)

    rows = []
    for method_name, weighting_fn in WEIGHTING_METHODS.items():
        selected = weighting_fn(candidates)
        result = run_backtest_from_selection(selected, history, actual_formation_date, end_date)

        weights = dict(zip(selected["Symbol"].to_list(), selected["Weight"].to_list()))
        rows.append({
            "WeightingMethod": method_name,
            "Weights": ", ".join(f"{s}:{w:.1%}" for s, w in weights.items()),
            "MaxWeight": max(weights.values()),
            **result["metrics"],
        })
        print(f"{method_name}: CAGR={result['metrics']['CAGR']:.2%}, Sharpe={result['metrics']['Sharpe']:.2f}, "
              f"MaxDrawdown={result['metrics']['MaxDrawdown']:.2%}, MaxWeight={max(weights.values()):.1%}")

    comparison = pl.DataFrame(rows)
    if write_output:
        comparison.write_csv(RESULTS_DIR / "weighting_method_comparison.csv")

    return comparison


if __name__ == "__main__":
    run()
