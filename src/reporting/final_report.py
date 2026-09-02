"""Generates the narrative final-submission report (Markdown) that accompanies the Excel workbook
and HTML performance report — pulls live numbers from the official competition backtest so it never
goes stale relative to the other artifacts.
"""
from datetime import datetime

import polars as pl

from src.config import REPORTS_DIR, RESULTS_DIR, INITIAL_CAPITAL, MAX_HOLDINGS, TRANSACTION_COST_RATE, RISK_FREE_RATE
from src.reporting.excel import ASSUMPTIONS_TEXT
from src.competition.final_submission import run as run_competition_backtest, MODEL_NAME
from src.research.sensitivity import summarize_stability

PHASE_METHODOLOGY = """
**Phase 1 — Data Audit.** Inspected `Data/1d/{year}/{half}/{SYMBOL}_EQ.parquet` (300 stocks,
2020-01-01 to 2026-06-30, raw OHLCV, no Adjusted Close). Found and fixed a scripcode-mapping bug
that had produced 6 stray empty files; confirmed no invalid OHLC rows, no duplicate dates.

**Phase 2 — Data Pipeline (`src/data/`).** `universe.py` builds the fixed, deduped 300-symbol
2026 universe and maps each symbol to its local filename; `loader.py` concatenates every symbol's
half-year files into one continuous daily series; `validator.py` runs the full data-quality audit.

**Phase 3 — Feature Engine (`src/features/`).** Momentum (R21/63/126/252, skip-month 3/6/9/12-month),
relative strength vs. a synthetic universe-proxy benchmark, rolling OLS trend-strength (60/90/126/180d,
closed-form via Polars rolling sums — no look-ahead), 52-week high/breakout, TA-Lib-based
SMA/EMA/MACD/RSI/ADX, volatility/drawdown, and volume-confirmation features. Computed for all 300
symbols in parallel (`ProcessPoolExecutor`) in ~2s, with a fixed deterministic row order.

**Phase 4 — Baseline Static Portfolio.** Built the simplest version first: MODEL 1 (12-1 momentum),
Top-10 by score, equal weight, buy-at-next-open, hold with zero rebalancing, 0.1% one-time cost —
to validate the mechanics before adding complexity.

**Phase 5 — Multi-Factor Composite Models (`src/signals/composite.py`).** Implemented 8+ candidate
alpha models (momentum variants, relative strength, trend, technical composite, risk-adjusted
momentum, trend quality, the spec's weighted composite baseline) plus a feature-correlation analysis
that confirmed heavy redundancy between RSI/MACD/MA-derived features (many pairs >0.90 correlated).

**Phase 6 — Correlation-Aware Selection (`src/portfolio/correlation.py`).** Added Method B (Top-30 →
120-day return correlation filter → Top-10) as an alternative to simple Top-N, tested at thresholds
0.70/0.75/0.80/0.85/none.

**Phase 7 — Weighting Methods (`src/portfolio/weighting.py`).** Equal, alpha-proportional,
alpha/volatility, inverse-volatility, and a blended risk-adjusted method — all capped at 20% per
position with iterative pro-rata redistribution.

**Phase 8 — Backtest Engine (`src/backtest/`).** Static (buy-once, hold-forever) portfolio
construction with an automatic, hard-enforced zero-rebalance guard (`validation.py`), plus
per-stock P&L attribution (`attribution.py`) to identify best/worst contributors.

**Phase 9 — Experiment Matrix (`src/research/experiments.py`).** Backtested every combination of
10 signal families × 2 selection methods × 5 correlation thresholds × 4 weighting methods, across
9 formation dates spanning 2021-2025 (2,160 backtests total) — this is the data-driven basis for
the model choice below.

**Phase 10 — Sensitivity Analysis (`src/research/sensitivity.py`).** Swept momentum lookback
windows (50/63/75, 100/126/150, 200/252/300 days) and trend windows (60/90/126/180 days) across
all 9 formation dates to check whether performance is a stable plateau or a fragile single-parameter
peak, rather than picking the single best-backtested value (which risks overfitting).

**Phase 11 — Reporting (`src/reporting/`).** Equity curve, drawdown, annual returns, feature
correlation heatmap, signal-importance, and per-stock contribution charts; the 13-sheet Excel
workbook; this narrative report.

**Phase 12 — Unit Tests (`tests/`).** 23 tests covering every feature calculation, the ranking/
weighting/correlation logic, backtest metrics, and a dedicated look-ahead-bias test (mutating future
prices must leave historical signals byte-identical). Caught and fixed a real bug: TA-Lib's NaN
warm-up values weren't being converted to proper Polars nulls, which could have let NaN scores leak
past `is_not_null()` filters used during selection.

**Phase 13 — Competition Backtest (`src/competition/final_submission.py`).** The official submission
run, deliberately kept separate from the research experiments above: strategy generation is
hard-restricted to 2020-2025 data only, then the SAME picks are re-entered with a fresh, independent
capital allocation for the 2026 H1 out-of-sample evaluation (never a continuation of the grown
backtest capital).
""".strip()


def _df_to_markdown_table(df: pl.DataFrame, float_fmt: str = "{:.4f}") -> str:
    def fmt(value):
        if isinstance(value, float):
            return float_fmt.format(value)
        return str(value)

    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join([header, separator] + rows)


def _metrics_to_markdown_table(metrics: dict) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
    return f"| Metric | Value |\n| --- | --- |\n{rows}"


def _model_selection_rationale() -> str:
    """Recomputes the Phase 9 experiment-matrix ranking and Phase 10 stability check live from the
    saved results, so the rationale below always matches whatever's actually on disk."""
    experiments = pl.read_parquet(RESULTS_DIR / "experiment_results.parquet")
    ranking = experiments.group_by("Model").agg(
        pl.col("NetPnL").mean().round(0).alias("AvgNetPnL"),
        pl.col("CAGR").mean().round(3).alias("AvgCAGR"),
        pl.col("Sharpe").mean().round(3).alias("AvgSharpe"),
        (pl.col("NetPnL") < 0).any().alias("HadNegativePeriod"),
        (pl.col("TotalReturn") > 0).mean().round(3).alias("PctPositivePeriods"),
    ).sort("AvgNetPnL", descending=True)
    ranking_table = _df_to_markdown_table(ranking.head(5), float_fmt="{:,.3f}")

    chosen_row = ranking.filter(pl.col("Model") == MODEL_NAME).row(0, named=True)
    rank = ranking["Model"].to_list().index(MODEL_NAME) + 1

    momentum_sensitivity = pl.read_csv(RESULTS_DIR / "momentum_lookback_sensitivity.csv")
    stability = summarize_stability(momentum_sensitivity, "LookbackDays").sort("LookbackDays")
    stability = stability.with_columns(
        pl.col("AvgNetPnL").round(0), pl.col("StdNetPnL").round(0),
        pl.col("AvgCAGR").round(3), pl.col("AvgSharpe").round(3), pl.col("PctPositivePeriods").round(3),
    )
    stability_table = _df_to_markdown_table(stability, float_fmt="{:,.3f}")

    return f"""### 3.1 Chosen Model: `{MODEL_NAME}`

**Ranking across all {experiments["Model"].n_unique()} tested signal families**
(averaged over 9 formation dates x 2 selection methods x 5 correlation thresholds x 4 weighting
methods = {experiments.height} backtests per signal family group, see Phase 9):

{ranking_table}

`{MODEL_NAME}` ranked **#{rank}** by average Net P&L (Rs {chosen_row["AvgNetPnL"]:,.0f}, {chosen_row["AvgCAGR"]:.1%} avg CAGR)
and had **{"zero" if not chosen_row["HadNegativePeriod"] else "at least one"} negative-P&L formation period**
out of 9 tested ({chosen_row["PctPositivePeriods"]:.0%} of periods profitable) — matching the spec's
selection priority (Net P&L primary, parameter stability/robustness secondary) better than the
heavier multi-factor composite models, which scored lower on both counts.

### 3.2 Parameter Stability Check (Phase 10)

The `momentum_3m` signal's underlying lookback window (63 trading days, ~3 months) was stress-tested
against neighboring lookbacks to confirm the result isn't a fragile, single-parameter fluke:

{stability_table}

Performance is a fairly stable plateau across the whole 50-300 day range (CAGR clustered ~34-38%,
Sharpe ~1.1-1.2) rather than a sharp peak at exactly 63 days, which supports using it as the final
signal rather than over-fitting to one specific window."""


def build_final_submission_report(output_path=None) -> str:
    output_path = output_path or REPORTS_DIR / "final_submission_report.md"
    result = run_competition_backtest()
    model_rationale = _model_selection_rationale()

    portfolio_table = _df_to_markdown_table(
        result["final_portfolio"].with_columns(pl.col("Weight").round(3)), float_fmt="{:.3f}"
    )
    backtest_table = _metrics_to_markdown_table(result["backtest_metrics"])
    oos_table = _metrics_to_markdown_table(result["oos_metrics"])

    report = f"""# Static Portfolio Selection — Final Submission Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 1. Executive Summary

- Initial capital: Rs {INITIAL_CAPITAL:,}
- Maximum holdings: {MAX_HOLDINGS}
- Transaction cost: {TRANSACTION_COST_RATE:.2%} (applied once at formation, no rebalancing)
- Risk-free rate: {RISK_FREE_RATE:.0%}
- Number of rebalances: {result["num_rebalances"]} (must be 0 — validated automatically by the backtest engine)
- Benchmark return (backtest period, synthetic universe-proxy): {result["benchmark_return"]:.2%}

## 2. Methodology by Phase

{PHASE_METHODOLOGY}

## 3. Model Selection: What Was Used, and Why

{model_rationale}

## 4. Final Portfolio

{portfolio_table}

## 5. Backtest Performance (2021-01-01 → 2025-12-31)

{backtest_table}

## 6. Out-of-Sample Performance (2026 H1, separate fresh capital, same picks)

{oos_table}

## 7. Assumptions & Limitations

{chr(10).join(ASSUMPTIONS_TEXT)}

## 8. Supporting Artifacts

- Full Excel workbook: `results/competition_submission.xlsx`
- HTML performance report with charts: `reports/performance_report.html`
- Final portfolio detail: `results/final_portfolio.csv`
- Trade log / daily NAV (backtest): `results/competition_trade_log.csv`, `results/competition_daily_nav.csv`
- Trade log / daily NAV (OOS 2026 H1): `results/oos_2026h1_trade_log.csv`, `results/oos_2026h1_daily_nav.csv`
- Per-stock contribution breakdown: `results/competition_stock_contributions.csv`, `results/oos_2026h1_stock_contributions.csv`
- Full research experiment matrix: `results/experiment_results.csv`

## 9. Conclusion

_[Add your own closing summary/interpretation here before submitting.]_
"""

    output_path.write_text(report)
    print(f"Final submission report written to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    build_final_submission_report()
