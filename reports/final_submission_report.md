# Static Portfolio Selection — Final Submission Report

Generated: 2026-09-02 21:10

## 1. Executive Summary

- Initial capital: Rs 10,000,000
- Maximum holdings: 10
- Transaction cost: 0.10% (applied once at formation, no rebalancing)
- Risk-free rate: 0%
- Number of rebalances: 0 (must be 0 — validated automatically by the backtest engine)
- Benchmark return (backtest period, synthetic universe-proxy): 329.81%

## 2. Methodology by Phase

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

## 3. Model Selection: What Was Used, and Why

### 3.1 Chosen Model: `momentum_3m`

**Ranking across all 10 tested signal families**
(averaged over 9 formation dates x 2 selection methods x 5 correlation thresholds x 4 weighting
methods = 2160 backtests per signal family group, see Phase 9):

| Model | AvgNetPnL | AvgCAGR | AvgSharpe | HadNegativePeriod | PctPositivePeriods |
| --- | --- | --- | --- | --- | --- |
| momentum_3m | 30,613,270.000 | 0.402 | 1.228 | False | 1.000 |
| momentum_6m | 29,078,549.000 | 0.367 | 1.196 | False | 1.000 |
| relative_strength | 26,373,493.000 | 0.343 | 1.118 | True | 0.889 |
| momentum_12_1 | 26,007,164.000 | 0.356 | 1.198 | False | 1.000 |
| multi_horizon_momentum | 25,798,654.000 | 0.343 | 1.090 | True | 0.889 |

`momentum_3m` ranked **#1** by average Net P&L (Rs 30,613,270, 40.2% avg CAGR)
and had **zero negative-P&L formation period**
out of 9 tested (100% of periods profitable) — matching the spec's
selection priority (Net P&L primary, parameter stability/robustness secondary) better than the
heavier multi-factor composite models, which scored lower on both counts.

### 3.2 Parameter Stability Check (Phase 10)

The `momentum_3m` signal's underlying lookback window (63 trading days, ~3 months) was stress-tested
against neighboring lookbacks to confirm the result isn't a fragile, single-parameter fluke:

| LookbackDays | AvgNetPnL | StdNetPnL | AvgCAGR | AvgSharpe | HadNegativePeriod | PctPositivePeriods |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 30,717,836.000 | 24,135,237.000 | 0.381 | 1.176 | False | 1.000 |
| 63 | 27,409,396.000 | 23,070,829.000 | 0.359 | 1.173 | False | 1.000 |
| 75 | 25,021,608.000 | 22,287,213.000 | 0.339 | 1.124 | False | 1.000 |
| 100 | 28,550,358.000 | 27,019,707.000 | 0.366 | 1.205 | False | 1.000 |
| 126 | 27,359,941.000 | 31,394,752.000 | 0.310 | 1.054 | True | 0.889 |
| 150 | 28,943,476.000 | 27,577,624.000 | 0.347 | 1.146 | True | 0.889 |
| 200 | 25,928,067.000 | 24,127,373.000 | 0.357 | 1.192 | False | 1.000 |
| 252 | 19,521,186.000 | 14,075,143.000 | 0.344 | 1.179 | False | 1.000 |
| 300 | 17,687,661.000 | 10,407,493.000 | 0.339 | 1.147 | False | 1.000 |

Performance is a fairly stable plateau across the whole 50-300 day range (CAGR clustered ~34-38%,
Sharpe ~1.1-1.2) rather than a sharp peak at exactly 63 days, which supports using it as the final
signal rather than over-fitting to one specific window.

## 4. Final Portfolio

| Rank | Symbol | Universe | Weight | CompositeScore | MomentumScore | RelativeStrengthScore | TrendScore | 52WHighScore | VolumeScore | Volatility | CorrelationToPortfolio | ExpectedRole |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ATGL | nse_midcap_100 | 0.100 | 1.000 | 1.000 | 0.959 | 0.979 | 0.752 | 0.697 | 0.588 | 0.293 | Core Holding |
| 2 | CGPOWER | nse_100 | 0.100 | 0.996 | 0.996 | 1.000 | 1.000 | 0.349 | 0.941 | 0.578 | 0.141 | Core Holding |
| 3 | PGEL | nse_smallcap_100 | 0.100 | 0.992 | 0.992 | 0.992 | 0.996 | 0.248 | 0.950 | 0.607 | 0.099 | Core Holding |
| 4 | ADANIENSOL | nse_100 | 0.100 | 0.988 | 0.988 | 0.798 | 0.917 | 0.651 | 0.559 | 0.469 | 0.286 | Growth Contributor |
| 5 | TATASTEEL | nse_100 | 0.100 | 0.984 | 0.984 | 0.905 | 0.831 | 0.853 | 0.790 | 0.360 | 0.316 | Growth Contributor |
| 6 | SHRIRAMFIN | nse_100 | 0.100 | 0.979 | 0.979 | 0.740 | 0.694 | 0.244 | 0.706 | 0.499 | 0.229 | Growth Contributor |
| 7 | ADANIGREEN | nse_100 | 0.100 | 0.975 | 0.975 | 0.979 | 0.992 | 0.387 | 0.626 | 0.525 | 0.164 | Growth Contributor |
| 8 | INDUSINDBK | nse_midcap_100 | 0.100 | 0.971 | 0.971 | 0.880 | 0.921 | 0.017 | 0.576 | 0.529 | 0.262 | Diversifier |
| 9 | SAIL | nse_midcap_100 | 0.100 | 0.967 | 0.967 | 0.955 | 0.566 | 0.941 | 0.987 | 0.473 | 0.282 | Diversifier |
| 10 | WOCKPHARMA | nse_smallcap_100 | 0.100 | 0.963 | 0.963 | 0.917 | 0.773 | 0.660 | 0.937 | 0.585 | 0.184 | Diversifier |

## 5. Backtest Performance (2021-01-01 → 2025-12-31)

| Metric | Value |
| --- | --- |
| InitialCapital | 10000000 |
| FinalValue | 72841063.28546068 |
| NetPnL | 62841063.28546068 |
| TotalReturn | 6.312171184133742 |
| CAGR | 0.4900895368262892 |
| AnnualizedVolatility | 0.32382112869428864 |
| MaxDrawdown | -0.380303147431557 |
| Sharpe | 1.4154583605934627 |
| Sortino | 1.8562499180314687 |
| GainLossRatio | 1.026455486905903 |
| WinRate | 0.5561843168957155 |

## 6. Out-of-Sample Performance (2026 H1, separate fresh capital, same picks)

| Metric | Value |
| --- | --- |
| InitialCapital | 10000000 |
| FinalValue | 12262511.12920688 |
| NetPnL | 2262511.129206881 |
| TotalReturn | 0.21396143712831583 |
| CAGR | 0.48205990487995676 |
| AnnualizedVolatility | 0.29613030719557676 |
| MaxDrawdown | -0.14068174854199045 |
| Sharpe | 1.5345230171025621 |
| Sortino | 2.1651823370830945 |
| GainLossRatio | 0.8770916422695965 |
| WinRate | 0.5966386554621849 |

## 7. Assumptions & Limitations

Corporate-action adjustment status of the local OHLCV data is unverified (only raw Close is
available, no Adjusted Close column) -- large single-day moves may reflect unadjusted
splits/bonuses rather than genuine price action. Flagged in the Phase 1 data audit.

No local Nifty 100 / Midcap 100 / Smallcap 100 index-level OHLCV series exists in this repo.
The 'Benchmark' sheet uses an equal-weighted average of the universe's own returns as a proxy
until real index data is sourced.

Risk-free rate assumed 0% throughout, per the competition brief.
Universe is the fixed CURRENT 2026 constituent list; historical index membership is not
reconstructed (no survivorship-bias adjustment), per the competition's explicit restriction.

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
