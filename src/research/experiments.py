"""Full experiment matrix: signals x selection method x correlation threshold x weighting method,
run across multiple formation dates (spec section 36) to see which methodology is actually robust."""
from datetime import datetime

import polars as pl

from src.config import RESULTS_DIR, MAX_HOLDINGS
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data
from src.signals.composite import build_composite_score
from src.portfolio.selector import select_top_n
from src.portfolio.correlation import select_with_correlation_filter, TOP_N_CANDIDATES
from src.portfolio.weighting import (
    equal_weight, alpha_weight, alpha_volatility_weight, inverse_volatility_weight, attach_volatility,
)
from src.research.backtest_runner import run_backtest_from_selection

FORMATION_DATES = [
    datetime(2021, 1, 1), datetime(2021, 7, 1),
    datetime(2022, 1, 1), datetime(2022, 7, 1),
    datetime(2023, 1, 1), datetime(2023, 7, 1),
    datetime(2024, 1, 1), datetime(2024, 7, 1),
    datetime(2025, 1, 1),
]

# The 10 signal families from the spec's experiment matrix.
EXPERIMENT_SIGNALS = [
    "momentum_12_1", "momentum_6m", "momentum_3m", "multi_horizon_momentum",
    "relative_strength", "trend_strength", "high_52w", "breakout",
    "volume_confirmation", "technical_composite",
]

SELECTION_METHODS = ["top_n", "correlation_filtered"]
CORRELATION_THRESHOLDS = [None, 0.70, 0.75, 0.80, 0.85]  # only used when selection_method == "correlation_filtered"
WEIGHTING_METHODS = {
    "equal": equal_weight,
    "alpha": alpha_weight,
    "alpha_volatility": alpha_volatility_weight,
    "inverse_volatility": inverse_volatility_weight,
}


def run_experiment_matrix(formation_dates: list = FORMATION_DATES, write_output: bool = True) -> pl.DataFrame:
    features = pl.read_parquet(RESULTS_DIR / "features.parquet")
    history = load_universe_history(get_symbol_to_scrip_data())
    end_date = features["Datetime"].max()

    rows = []
    for formation_date in formation_dates:
        for model_name in EXPERIMENT_SIGNALS:
            try:
                signal = build_composite_score(features, formation_date, model_name)
            except ValueError as e:
                print(f"Skipping {formation_date.date()} / {model_name}: {e}")
                continue
            actual_formation_date = signal["Datetime"][0]

            for selection_method in SELECTION_METHODS:
                thresholds = CORRELATION_THRESHOLDS if selection_method == "correlation_filtered" else [None]
                for correlation_threshold in thresholds:
                    if selection_method == "top_n":
                        candidates = select_top_n(signal, n=MAX_HOLDINGS)
                    else:
                        candidates = select_with_correlation_filter(
                            signal, history, actual_formation_date, n=MAX_HOLDINGS,
                            top_n_candidates=TOP_N_CANDIDATES, correlation_threshold=correlation_threshold,
                        )
                    candidates = attach_volatility(candidates, features, actual_formation_date)

                    for weighting_name, weighting_fn in WEIGHTING_METHODS.items():
                        selected = weighting_fn(candidates)
                        try:
                            result = run_backtest_from_selection(selected, history, actual_formation_date, end_date)
                        except ValueError as e:
                            print(f"Skipping backtest {formation_date.date()}/{model_name}/{selection_method}/{weighting_name}: {e}")
                            continue

                        rows.append({
                            "FormationDate": formation_date,
                            "ActualFormationDate": actual_formation_date,
                            "Model": model_name,
                            "SelectionMethod": selection_method,
                            "CorrelationThreshold": "None" if correlation_threshold is None else correlation_threshold,
                            "WeightingMethod": weighting_name,
                            **result["metrics"],
                        })

    experiment_results = pl.DataFrame(rows)
    print(f"Completed {experiment_results.height} backtests across {len(formation_dates)} formation dates.")

    if write_output:
        experiment_results.write_csv(RESULTS_DIR / "experiment_results.csv")
        experiment_results.write_parquet(RESULTS_DIR / "experiment_results.parquet")

    return experiment_results


if __name__ == "__main__":
    run_experiment_matrix()
