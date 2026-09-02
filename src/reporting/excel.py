"""Build competition_submission.xlsx with the sheets required by the spec."""
import xlsxwriter
import polars as pl

from src.config import RESULTS_DIR, INITIAL_CAPITAL, MAX_HOLDINGS, TRANSACTION_COST_RATE, RISK_FREE_RATE

METHODOLOGY_TEXT = [
    "Static portfolio selection: the model ranks the fixed 2026 Nifty 100 + Midcap 100 + Smallcap 100",
    "universe by a cross-sectional alpha score computed only from information available on or before",
    "the formation date, selects the top holdings, assigns weights once, and holds them unchanged",
    "through the evaluation period (zero rebalancing, enforced and validated by the backtest engine).",
    "",
    "Signal: percentile-ranked technical/quantitative features (momentum, relative strength, trend",
    "regression quality, moving-average structure, RSI, ADX, breakout distance, volume confirmation)",
    "combined into a single composite AlphaScore.",
    "",
    "Selection: either simple Top-N by AlphaScore, or Top-30 -> pairwise 120-day-return-correlation",
    "filter -> Top-N, to avoid concentrating the portfolio in highly correlated names.",
    "",
    "Weighting: equal weight, alpha-proportional, alpha/volatility, or inverse-volatility, always",
    "capped at 20% per position and renormalized to sum to 1.",
    "",
    "Execution: entry at the next trading day's open after the signal date; 0.1% transaction cost",
    "applied once at formation only (no rebalancing costs).",
]

ASSUMPTIONS_TEXT = [
    "Corporate-action adjustment status of the local OHLCV data is unverified (only raw Close is",
    "available, no Adjusted Close column) -- large single-day moves may reflect unadjusted",
    "splits/bonuses rather than genuine price action. Flagged in the Phase 1 data audit.",
    "",
    "No local Nifty 100 / Midcap 100 / Smallcap 100 index-level OHLCV series exists in this repo.",
    "The 'Benchmark' sheet uses an equal-weighted average of the universe's own returns as a proxy",
    "until real index data is sourced.",
    "",
    "Risk-free rate assumed 0% throughout, per the competition brief.",
    "Universe is the fixed CURRENT 2026 constituent list; historical index membership is not",
    "reconstructed (no survivorship-bias adjustment), per the competition's explicit restriction.",
]

CALCULATIONS_TEXT = [
    "CAGR = (FinalValue / InitialValue) ^ (365.25 / days_held) - 1",
    "Annualized Volatility = std(daily_returns) * sqrt(252)",
    "Max Drawdown = min(Value_t / running_max(Value)_t - 1)",
    "Sharpe = mean(daily_return - rf/252) / std(daily_return - rf/252) * sqrt(252)",
    "Sortino = same as Sharpe but the denominator only uses the standard deviation of negative excess returns",
    "Gain/Loss Ratio = mean(positive daily returns) / abs(mean(negative daily returns))",
    "Win Rate = fraction of days with a positive daily return",
    "Transaction Cost = position_value * 0.1%, applied once at formation only",
]


def _write_dataframe(workbook, df: pl.DataFrame, sheet_name: str):
    df.write_excel(workbook=workbook, worksheet=sheet_name, autofit=True)


def _write_text_sheet(workbook, lines: list[str], sheet_name: str):
    worksheet = workbook.add_worksheet(sheet_name)
    for row, line in enumerate(lines):
        worksheet.write(row, 0, line)
    worksheet.set_column(0, 0, 110)


def build_submission_workbook(
    metrics: dict,
    trade_log: pl.DataFrame,
    daily_nav: pl.DataFrame,
    contributions: pl.DataFrame,
    stock_scores: pl.DataFrame,
    benchmark_comparison: dict,
    oos_metrics: dict,
    experiment_results: pl.DataFrame,
    output_path=None,
) -> str:
    output_path = str(output_path or RESULTS_DIR / "competition_submission.xlsx")
    workbook = xlsxwriter.Workbook(output_path)

    summary_rows = [
        {"Metric": "Initial Capital", "Value": INITIAL_CAPITAL},
        {"Metric": "Max Holdings", "Value": MAX_HOLDINGS},
        {"Metric": "Transaction Cost Rate", "Value": TRANSACTION_COST_RATE},
        {"Metric": "Risk-Free Rate", "Value": RISK_FREE_RATE},
    ] + [{"Metric": k, "Value": v} for k, v in metrics.items()]
    _write_dataframe(workbook, pl.DataFrame(summary_rows), "Executive Summary")

    _write_dataframe(workbook, trade_log, "Final Portfolio")
    _write_dataframe(workbook, trade_log.select(["Symbol", "Weight", "EntryDate", "EntryPrice", "Shares"]), "Portfolio Weights")
    _write_dataframe(workbook, pl.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()]), "Performance")
    _write_dataframe(workbook, pl.DataFrame([{"Metric": k, "Value": v} for k, v in benchmark_comparison.items()]), "Benchmark")

    annual = daily_nav.sort("Datetime").with_columns(pl.col("Datetime").dt.year().alias("Year")).group_by(
        "Year", maintain_order=True
    ).agg((pl.col("PortfolioValue").last() / pl.col("PortfolioValue").first() - 1).alias("Return")).sort("Year")
    _write_dataframe(workbook, annual, "Annual Returns")

    _write_dataframe(workbook, contributions, "Drawdown")  # per-stock detail doubles as drawdown-contributor context
    _write_dataframe(workbook, stock_scores, "Stock Scores")
    _write_text_sheet(workbook, METHODOLOGY_TEXT, "Methodology")
    _write_text_sheet(workbook, ASSUMPTIONS_TEXT, "Assumptions")
    _write_text_sheet(workbook, CALCULATIONS_TEXT, "Calculations")
    _write_dataframe(workbook, pl.DataFrame([{"Metric": k, "Value": v} for k, v in oos_metrics.items()]), "OOS 2026")
    _write_dataframe(workbook, experiment_results, "Experiment Results")

    workbook.close()
    return output_path
