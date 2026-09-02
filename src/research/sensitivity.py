"""Sensitivity analysis: sweep momentum/trend lookback windows around commonly used periods to see
whether performance is a robust plateau or a fragile single-parameter peak (spec section 37/38)."""
from datetime import datetime

import polars as pl

from src.config import RESULTS_DIR, MAX_HOLDINGS
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data
from src.portfolio.selector import select_top_n
from src.portfolio.weighting import equal_weight
from src.research.backtest_runner import run_backtest_from_selection
from src.signals.composite import get_cross_section
from src.signals.ranking import to_percentile

FORMATION_DATES = [
    datetime(2021, 1, 1), datetime(2021, 7, 1),
    datetime(2022, 1, 1), datetime(2022, 7, 1),
    datetime(2023, 1, 1), datetime(2023, 7, 1),
    datetime(2024, 1, 1), datetime(2024, 7, 1),
    datetime(2025, 1, 1),
]

MOMENTUM_LOOKBACK_GRID = {"3M": [50, 63, 75], "6M": [100, 126, 150], "12M": [200, 252, 300]}
TREND_WINDOWS = [60, 90, 126, 180]  # already precomputed in the feature panel
SKIP_MONTH = 21


def momentum_signal_for_window(history: dict[str, pl.DataFrame], formation_date, window: int, skip: int = SKIP_MONTH) -> pl.DataFrame:
    """Skip-month momentum score (Close.shift(skip)/Close.shift(skip+window)-1) computed directly
    from raw price history for an arbitrary lookback — used to sweep windows the precomputed feature
    panel doesn't already cover (it only stores 63/126/189/252)."""
    rows = []
    for symbol, df in history.items():
        past = df.filter(pl.col("Datetime") <= formation_date).sort("Datetime")
        if past.height < skip + window + 1:
            continue
        close = past["Close"]
        score = close[-1 - skip] / close[-1 - skip - window] - 1
        rows.append({"Symbol": symbol, "Datetime": past["Datetime"][-1], "RawScore": score})

    if not rows:
        raise ValueError(f"No symbols have enough history for a {window}-day lookback as of {formation_date}.")

    signal = to_percentile(pl.DataFrame(rows), "RawScore")
    return signal.rename({"RawScore_Percentile": "AlphaScore"}).select(["Symbol", "Datetime", "AlphaScore"])


def trend_signal_for_window(features_df: pl.DataFrame, formation_date, window: int) -> pl.DataFrame:
    """Reuse the precomputed Trend{window}_TrendStrength feature already in the feature panel."""
    cross_section, _ = get_cross_section(features_df, formation_date)
    col = f"Trend{window}_TrendStrength"
    cross_section = cross_section.filter(pl.col(col).is_not_null())
    if cross_section.height == 0:
        raise ValueError(f"No symbols have a non-null {col} as of {formation_date}.")
    cross_section = to_percentile(cross_section, col)
    return cross_section.rename({f"{col}_Percentile": "AlphaScore"}).select(["Symbol", "Datetime", "AlphaScore"])


def run_momentum_sensitivity(formation_dates: list = FORMATION_DATES, write_output: bool = True) -> pl.DataFrame:
    history = load_universe_history(get_symbol_to_scrip_data())
    end_date = max(df["Datetime"].max() for df in history.values())

    rows = []
    for label, windows in MOMENTUM_LOOKBACK_GRID.items():
        for window in windows:
            for formation_date in formation_dates:
                try:
                    signal = momentum_signal_for_window(history, formation_date, window)
                except ValueError as e:
                    print(f"Skipping {label}/{window}d @ {formation_date.date()}: {e}")
                    continue
                result = run_backtest_from_selection(
                    equal_weight(select_top_n(signal, n=MAX_HOLDINGS)), history, signal["Datetime"][0], end_date,
                )
                rows.append({"SignalFamily": label, "LookbackDays": window, "FormationDate": formation_date, **result["metrics"]})

    results = pl.DataFrame(rows)
    if write_output:
        results.write_csv(RESULTS_DIR / "momentum_lookback_sensitivity.csv")
    return results


def run_trend_sensitivity(formation_dates: list = FORMATION_DATES, write_output: bool = True) -> pl.DataFrame:
    features = pl.read_parquet(RESULTS_DIR / "features.parquet")
    history = load_universe_history(get_symbol_to_scrip_data())
    end_date = features["Datetime"].max()

    rows = []
    for window in TREND_WINDOWS:
        for formation_date in formation_dates:
            try:
                signal = trend_signal_for_window(features, formation_date, window)
            except ValueError as e:
                print(f"Skipping trend {window}d @ {formation_date.date()}: {e}")
                continue
            result = run_backtest_from_selection(
                equal_weight(select_top_n(signal, n=MAX_HOLDINGS)), history, signal["Datetime"][0], end_date,
            )
            rows.append({"LookbackDays": window, "FormationDate": formation_date, **result["metrics"]})

    results = pl.DataFrame(rows)
    if write_output:
        results.write_csv(RESULTS_DIR / "trend_window_sensitivity.csv")
    return results


def summarize_stability(results: pl.DataFrame, group_col: str) -> pl.DataFrame:
    """Aggregate across formation dates per parameter value, flagging unstable ones (any negative
    Net P&L period, or a low hit-rate) rather than just picking the single highest-average value."""
    return results.group_by(group_col).agg(
        pl.col("NetPnL").mean().alias("AvgNetPnL"),
        pl.col("NetPnL").std().alias("StdNetPnL"),
        pl.col("CAGR").mean().alias("AvgCAGR"),
        pl.col("Sharpe").mean().alias("AvgSharpe"),
        (pl.col("NetPnL") < 0).any().alias("HadNegativePeriod"),
        (pl.col("TotalReturn") > 0).mean().alias("PctPositivePeriods"),
    ).sort(group_col)


if __name__ == "__main__":
    momentum_results = run_momentum_sensitivity()
    print("\n--- Momentum lookback stability ---")
    print(summarize_stability(momentum_results, "LookbackDays"))

    trend_results = run_trend_sensitivity()
    print("\n--- Trend window stability ---")
    print(summarize_stability(trend_results, "LookbackDays"))
