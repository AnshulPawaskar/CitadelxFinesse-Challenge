"""Orchestrates chart, Excel, and HTML report generation for the final submission portfolio."""
from datetime import datetime

import polars as pl

from src.config import RESULTS_DIR
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data
from src.signals.composite import build_composite_score, get_cross_section
from src.portfolio.selector import select_top_n
from src.portfolio.weighting import equal_weight
from src.research.backtest_runner import run_backtest_from_selection
from src.backtest.metrics import compute_metrics
from src.backtest.attribution import compute_stock_contributions
from src.features.relative_strength import build_synthetic_benchmark
from src.research.feature_correlation import run as run_feature_correlation
from src.reporting.charts import generate_all_charts
from src.reporting.excel import build_submission_workbook
from src.reporting.reports import build_html_report

FORMATION_DATE = datetime(2021, 1, 1)
BACKTEST_END_DATE = datetime(2025, 12, 31)
MODEL_NAME = "composite_quantitative"


def _benchmark_comparison(benchmark, portfolio_start, oos_start, full_end_date) -> dict:
    window = benchmark.filter((pl.col("Datetime") >= portfolio_start) & (pl.col("Datetime") <= full_end_date)).sort("Datetime")
    backtest_window = window.filter(pl.col("Datetime") <= BACKTEST_END_DATE)
    oos_window = window.filter(pl.col("Datetime") >= oos_start)
    return {
        "BenchmarkReturn_Backtest": backtest_window["BenchmarkIndex"][-1] / backtest_window["BenchmarkIndex"][0] - 1,
        "BenchmarkReturn_OOS2026H1": (
            oos_window["BenchmarkIndex"][-1] / oos_window["BenchmarkIndex"][0] - 1 if oos_window.height > 1 else None
        ),
    }


def run():
    features = pl.read_parquet(RESULTS_DIR / "features.parquet")
    history = load_universe_history(get_symbol_to_scrip_data())
    full_end_date = features["Datetime"].max()
    oos_start = datetime(2026, 1, 1)

    signal = build_composite_score(features, FORMATION_DATE, MODEL_NAME)
    actual_formation_date = signal["Datetime"][0]

    selected = equal_weight(select_top_n(signal))
    result = run_backtest_from_selection(selected, history, actual_formation_date, full_end_date)

    daily_nav = result["daily_nav"]
    metrics = compute_metrics(daily_nav.filter(pl.col("Datetime") <= BACKTEST_END_DATE), result["initial_capital"])
    oos_nav = daily_nav.filter(pl.col("Datetime") >= oos_start)
    oos_metrics = (
        compute_metrics(oos_nav, oos_nav["PortfolioValue"][0]) if oos_nav.height > 1
        else {"Note": "Insufficient OOS 2026 H1 data in daily_nav for this formation date."}
    )

    contributions = compute_stock_contributions(result, history, full_end_date)

    stock_scores, _ = get_cross_section(features, FORMATION_DATE)
    stock_scores = stock_scores.select(["Symbol", "Momentum12_1", "RS126", "Trend126_TrendStrength", "Vol126"]).sort(
        "Momentum12_1", descending=True
    )

    benchmark = build_synthetic_benchmark(history)
    benchmark_comparison = _benchmark_comparison(benchmark, result["portfolio_start"], oos_start, full_end_date)
    benchmark_comparison["PortfolioReturn_Backtest"] = metrics["TotalReturn"]
    benchmark_comparison["ExcessReturn_Backtest"] = metrics["TotalReturn"] - benchmark_comparison["BenchmarkReturn_Backtest"]

    corr_matrix, _ = run_feature_correlation(write_output=False)
    experiment_results = pl.read_parquet(RESULTS_DIR / "experiment_results.parquet")

    chart_paths = generate_all_charts(daily_nav, corr_matrix, experiment_results, contributions)

    excel_path = build_submission_workbook(
        metrics=metrics, trade_log=result["trade_log"], daily_nav=daily_nav, contributions=contributions,
        stock_scores=stock_scores, benchmark_comparison=benchmark_comparison, oos_metrics=oos_metrics,
        experiment_results=experiment_results,
    )
    html_path = build_html_report(metrics, oos_metrics, chart_paths)

    print(f"Excel workbook: {excel_path}")
    print(f"HTML report: {html_path}")
    print("Charts:", {name: str(path) for name, path in chart_paths.items()})

    return {"metrics": metrics, "oos_metrics": oos_metrics, "excel_path": excel_path, "html_path": html_path}


if __name__ == "__main__":
    run()
