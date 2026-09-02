"""The official competition submission backtest — deliberately kept separate from src/research/
per the spec's instruction not to mix research experiments with the final competition result.

Chosen model: momentum_3m (63-day skip-month momentum, single factor). Selected based on the
Phase 9/10 evidence: it had the best average Net P&L and CAGR across all 9 formation dates AND
0 negative-P&L periods (the most parameter-stable signal tested), matching the spec's ranking
priority (Net P&L primary, stability/robustness secondary) better than the heavier composite model.

Selection: simple Top-10 (Method A) — correlation filtering (Method B) did not change the top-10
in any comparison we ran, so the extra complexity isn't justified here.
Weighting: equal weight — performed as well as alpha-weighting with full transparency.
"""
from datetime import datetime

import polars as pl

from src.config import RESULTS_DIR, MAX_HOLDINGS, INITIAL_CAPITAL
from src.data.loader import load_universe_history, truncate_history
from src.data.universe import get_symbol_to_scrip_data, build_universe
from src.signals.composite import build_composite_score, get_cross_section
from src.portfolio.selector import select_top_n
from src.portfolio.weighting import equal_weight
from src.backtest.static_portfolio import build_static_portfolio
from src.backtest.metrics import compute_metrics
from src.backtest.attribution import compute_stock_contributions, summarize_best_worst
from src.backtest.validation import validate_no_rebalancing
from src.features.relative_strength import build_synthetic_benchmark
from src.signals.ranking import to_percentile
from src.portfolio.correlation import build_trailing_returns

FORMATION_DATE = datetime(2021, 1, 1)
COMPETITION_END_DATE = datetime(2025, 12, 31)
MODEL_NAME = "momentum_3m"


def _score_columns(features_df: pl.DataFrame, formation_date) -> pl.DataFrame:
    """Per-symbol component scores (percentile-ranked) for the Final Portfolio report."""
    cross_section, _ = get_cross_section(features_df, formation_date)
    for col in ["Momentum3_1", "RS126", "Trend126_TrendStrength", "CloseTo52WHigh", "VolumeConfirmedMomentum"]:
        cross_section = cross_section.filter(pl.col(col).is_not_null())
        cross_section = to_percentile(cross_section, col)
    return cross_section.select([
        "Symbol", "Vol126",
        pl.col("Momentum3_1_Percentile").alias("MomentumScore"),
        pl.col("RS126_Percentile").alias("RelativeStrengthScore"),
        pl.col("Trend126_TrendStrength_Percentile").alias("TrendScore"),
        pl.col("CloseTo52WHigh_Percentile").alias("52WHighScore"),
        pl.col("VolumeConfirmedMomentum_Percentile").alias("VolumeScore"),
    ])


def _correlation_to_portfolio(history: dict[str, pl.DataFrame], symbols: list[str], as_of_date) -> dict[str, float]:
    returns = build_trailing_returns(history, symbols, as_of_date)
    if returns.is_empty():
        return {s: None for s in symbols}
    corr = returns.corr()
    avg_corr = {}
    for i, symbol in enumerate(returns.columns):
        others = [c for c in returns.columns if c != symbol]
        avg_corr[symbol] = float(sum(abs(corr[o][i]) for o in others) / len(others)) if others else None
    return avg_corr


def _role_label(rank: int) -> str:
    if rank <= 3:
        return "Core Holding"
    if rank <= 7:
        return "Growth Contributor"
    return "Diversifier"


def build_final_portfolio_report(selected: pl.DataFrame, features_df: pl.DataFrame, history: dict, formation_date, universe: pl.DataFrame) -> pl.DataFrame:
    scores = _score_columns(features_df, formation_date)
    symbols = selected["Symbol"].to_list()
    correlations = _correlation_to_portfolio(history, symbols, formation_date)
    universe_lookup = dict(zip(universe["Symbol"].to_list(), universe["Index"].to_list()))

    report = selected.join(scores, on="Symbol", how="left").sort("AlphaScore", descending=True, maintain_order=True)
    report = report.with_columns(pl.Series("Rank", range(1, report.height + 1)))
    report = report.with_columns([
        pl.col("Symbol").replace_strict(universe_lookup, default="Unknown").alias("Universe"),
        pl.col("Symbol").replace_strict(correlations, default=None).alias("CorrelationToPortfolio"),
        pl.col("Rank").map_elements(_role_label, return_dtype=pl.Utf8).alias("ExpectedRole"),
    ])
    return report.select([
        "Rank", "Symbol", "Universe", "Weight", "AlphaScore", "MomentumScore", "RelativeStrengthScore",
        "TrendScore", "52WHighScore", "VolumeScore", "Vol126", "CorrelationToPortfolio", "ExpectedRole",
    ]).rename({"AlphaScore": "CompositeScore", "Vol126": "Volatility"})


def run():
    features = pl.read_parquet(RESULTS_DIR / "features.parquet")
    full_history = load_universe_history(get_symbol_to_scrip_data())
    universe = build_universe()

    # Strategy generation (signal + backtest) is hard-restricted to 2020-01-01..2025-12-31 — the
    # 2026 rows are simply not present in this dict, so they can't leak into selection or pricing.
    backtest_history = truncate_history(full_history, COMPETITION_END_DATE)

    signal = build_composite_score(features, FORMATION_DATE, MODEL_NAME)
    actual_formation_date = signal["Datetime"][0]
    selected = equal_weight(select_top_n(signal, n=MAX_HOLDINGS))

    # --- Backtest: fresh Rs 1,00,00,000, formed 2021-01-01, held through 2025-12-31 ---
    backtest_result = build_static_portfolio(
        selected, backtest_history, actual_formation_date, COMPETITION_END_DATE, initial_capital=INITIAL_CAPITAL
    )
    validate_no_rebalancing(backtest_result["trade_log"])
    backtest_metrics = compute_metrics(backtest_result["daily_nav"], backtest_result["initial_capital"])

    # --- OOS 2026 H1: a SEPARATE fresh Rs 1,00,00,000 invested in the SAME picks/weights, entered
    # at 2026 prices — never a continuation of the grown 2021-2025 capital.
    oos_result = build_static_portfolio(
        selected, full_history, COMPETITION_END_DATE, full_history[selected["Symbol"][0]]["Datetime"].max(),
        initial_capital=INITIAL_CAPITAL,
    )
    validate_no_rebalancing(oos_result["trade_log"])
    oos_metrics = compute_metrics(oos_result["daily_nav"], oos_result["initial_capital"])

    benchmark = build_synthetic_benchmark(backtest_history)
    bench_window = benchmark.filter(
        (pl.col("Datetime") >= backtest_result["portfolio_start"]) & (pl.col("Datetime") <= COMPETITION_END_DATE)
    ).sort("Datetime")
    benchmark_return = bench_window["BenchmarkIndex"][-1] / bench_window["BenchmarkIndex"][0] - 1

    backtest_contributions = compute_stock_contributions(backtest_result, backtest_history, COMPETITION_END_DATE)
    oos_contributions = compute_stock_contributions(oos_result, full_history, oos_result["daily_nav"]["Datetime"].max())
    final_portfolio = build_final_portfolio_report(selected, features, backtest_history, actual_formation_date, universe)

    print(f"Formation date: {actual_formation_date}")
    print(final_portfolio)
    print("\n--- Competition Backtest Performance (2021-01-01 to 2025-12-31, Rs 1,00,00,000) ---")
    for key, value in backtest_metrics.items():
        print(f"{key}: {value}")
    print(f"BenchmarkReturn: {benchmark_return:.2%}  |  ExcessReturn: {backtest_metrics['TotalReturn'] - benchmark_return:.2%}")
    print(f"num_rebalances: {backtest_result['num_rebalances']}")

    print("\n--- Out-of-Sample Performance (2026 H1, SEPARATE fresh Rs 1,00,00,000, same picks) ---")
    for key, value in oos_metrics.items():
        print(f"{key}: {value}")
    print(f"num_rebalances: {oos_result['num_rebalances']}")

    print("\n--- Backtest Best/Worst Contributors ---")
    print(summarize_best_worst(backtest_contributions))
    print("\n--- OOS Best/Worst Contributors ---")
    print(summarize_best_worst(oos_contributions))

    final_portfolio.write_csv(RESULTS_DIR / "final_portfolio.csv")
    backtest_result["trade_log"].write_csv(RESULTS_DIR / "competition_trade_log.csv")
    backtest_result["daily_nav"].write_csv(RESULTS_DIR / "competition_daily_nav.csv")
    backtest_contributions.write_csv(RESULTS_DIR / "competition_stock_contributions.csv")
    oos_result["trade_log"].write_csv(RESULTS_DIR / "oos_2026h1_trade_log.csv")
    oos_result["daily_nav"].write_csv(RESULTS_DIR / "oos_2026h1_daily_nav.csv")
    oos_contributions.write_csv(RESULTS_DIR / "oos_2026h1_stock_contributions.csv")

    return {
        "final_portfolio": final_portfolio, "backtest_metrics": backtest_metrics, "oos_metrics": oos_metrics,
        "benchmark_return": benchmark_return,
        "num_rebalances": backtest_result["num_rebalances"] + oos_result["num_rebalances"],
    }


    return {
        "final_portfolio": final_portfolio, "backtest_metrics": backtest_metrics, "oos_metrics": oos_metrics,
        "benchmark_return": benchmark_return, "num_rebalances": result["num_rebalances"],
    }


if __name__ == "__main__":
    run()
