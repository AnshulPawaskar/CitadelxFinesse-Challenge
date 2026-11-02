# CitadelxFinesse Challenge — Full Methodology Report

**Team:** Quantify · **Task:** build a portfolio of at most 10 stocks from the Nifty 100 + Midcap 100 + Smallcap 100 universe, backtest it, and trade it out-of-sample.

All figures below come from the files in `results/` and were reproduced from the code in `src/`. Where a conclusion is a judgement rather than something the code proves, it says so.

---

## 1. Executive summary

| Item | Value |
|---|---|
| Universe | 300 NSE stocks (current constituents of Nifty 100, Midcap 100, Smallcap 100) |
| Data | Daily OHLCV, 2020-01-01 → 2026-10-30, from 5Paisa, stored as half-year parquet files |
| Final strategy | `composite_quantitative` multi-factor score → top 10 → equal weight (10% each) → buy and hold |
| Formation date | 2021-01-01 (orders executed at the next trading day's open, 2021-01-04) |
| In-sample backtest (2021-01-04 → 2025-12-31) | CAGR **53.7%**, Sharpe **1.50**, max drawdown **−34.7%**, total return **+754%** |
| Out-of-sample (2026-01-01 → 2026-10-30) | Total return **+7.6%**, Sharpe **0.49**, max drawdown **−16.6%**; the equal-weighted universe proxy returned **−4.8%** |
| Final holdings | GPIL, ADANIENT, ATGL, CGPOWER, TATAELXSI, DIXON, AFFLE, PGEL, ADANIGREEN, LAURUSLABS |

**Headline caveat:** the in-sample result is very strong, but the out-of-sample result is modest (Sharpe 0.49 over ten months) and the strategy was chosen using in-sample information. Section 10 logs every decision and Section 11 lists the limitations that should temper any conclusion.

---

## 2. Pipeline overview

`run_pipeline.py` runs ten stages in this order:

1. **data_audit** – validates the stored OHLCV data (`src/data/validator.py`).
2. **features** – computes 98 columns per stock-day for 300 stocks (`src/features/`).
3. **momentum_baseline** – first simple strategy, 12-1 momentum.
4. **model_comparison** – backtests every alpha model at one formation date.
5. **feature_correlation** – checks which features are redundant.
6. **correlation_selection** – compares plain top-N with correlation-filtered selection.
7. **weighting_comparison** – compares five weighting methods.
8. **experiments** – full grid across signals, selection, weights and formation dates (2,160 backtests).
9. **sensitivity** – sweeps lookback windows.
10. **reports** – charts, Excel workbook and HTML report for the final portfolio.

The logic flows from raw prices to a decision like this:

```
raw OHLCV → features → factor percentiles → AlphaScore (per model)
          → top-N selection (optional correlation filter)
          → weights (capped at 20%) → static backtest → metrics
```

---

## 3. Data

### 3.1 Universe and mapping

- Constituent lists are in `Stocks/` (three CSVs). They are combined and de-duplicated by `Symbol` (`src/data/universe.py`).
- Each symbol is joined to the 5Paisa script master (`ScriptBook/`), restricted to NSE cash equity (`Exch = N`, `ExchType = C`, `Series = EQ`), to obtain the `ScripCode` and `ScripData` used to request and store data.
- Prices are fetched in half-year chunks, giving `Data/1d/{year}/{half}/{SCRIP}_EQ.parquet`.

### 3.2 Data quality audit

`results/data_audit_report.csv` (regenerated through 2026-10-31) reports for each of the 300 stocks: row count, date range, nulls, duplicate dates, invalid OHLC rows (high < low, or open/close outside the high–low range), non-positive prices, negative volumes, and single-day moves above 15%.

Findings:
- All 300 stocks have status OK, zero nulls, zero duplicates, zero invalid OHLC rows.
- Every stock has data through **2026-10-30**.
- 239 stocks have data from 2020-01-01; the rest start later because they listed later (for example AFCONS from 2024-11-04). Empty early half-year files for these are expected.
- **Gaps exist.** Against the union of trading dates seen in the data, 311 stock-date observations are missing across 225 stocks. Most (224 stocks) are one date, **2024-07-19**, which 50 other stocks have. FORCEMOT is missing 77 dates between 2023-10-26 and 2024-07-19. These are not fixed yet; the suggested remedy is to re-fetch just those stocks and dates from 5Paisa and merge without overwriting existing rows.
- Corporate-action (split/bonus) adjustment is unverified. Only raw `Close` is stored, with no adjusted close. Large moves may be unadjusted actions rather than real price changes.

### 3.3 Loading

`src/data/loader.py` concatenates a stock's half-year files, parses `Datetime`, sorts, and drops duplicate dates. `truncate_history` hard-cuts history at a chosen end date so later data cannot leak into a backtest.

---

## 4. Feature engineering

Features are computed per stock by `src/features/feature_pipeline.py` (parallelised) and saved to `results/features.parquet`. Every feature uses only past data (rolling or shifted windows); `tests/test_lookahead.py` verifies that changing future prices does not change earlier momentum or trend values.

### 4.1 Feature families

| Family | Features (examples) | Definition |
|---|---|---|
| **Momentum** | `R21/R63/R126/R252`; `Momentum3_1`, `Momentum6_1`, `Momentum9_1`, `Momentum12_1` | `R_w = Close / Close[t−w] − 1`. Skip-month momentum: `Close[t−21] / Close[t−w] − 1` with w = 63, 126, 189, 252. The most recent 21 days are excluded to avoid short-term reversal. |
| **Relative strength** | `RS21/63/126/252` and ratio forms | Stock return over w days minus benchmark return over w days. Benchmark is a synthetic index: cumulative product of (1 + equal-weighted mean daily return of all universe stocks). |
| **Trend quality** | `Trend{60,90,126,180}_Slope/R2/TrendStrength` | Rolling OLS of `ln(Close)` on time. `TrendStrength = Slope × R²`, so a steep, clean trend scores high and a steep, noisy one is discounted. Computed in closed form with rolling sums. |
| **52-week high / breakout** | `CloseTo52WHigh`, `CloseTo126DHigh`, `Breakout{20,55,126,252}D`, `BreakoutDistance*_ATR` | Prior-high uses `High.shift(1).rolling_max(w)` so today's bar never confirms itself. Breakout distance `(Close − prior high) / ATR14`. |
| **Moving averages** | SMA 20/50/100/200, EMA 12/26, MACD, `CloseToMA*`, `MA20_MA50`, `MA50_MA200` | TA-Lib implementations. |
| **Oscillators / direction** | RSI 7/14/21, ADX14, +DI, −DI, `DI_Diff` | TA-Lib implementations. |
| **Volatility / risk** | `Vol20/60/126`, `DownsideVolatility60`, `ATR14`, `ATR14_Close`, rolling max drawdown 60/126/252 | `Vol_w = std(daily return, w) × √252`; ATR via TA-Lib. |
| **Volume** | OBV, `OBV_Slope`, `VolMA20/60`, `VolumeToMA*`, `VolumeZScore`, `PriceVolumeCorr20` | OBV from TA-Lib; slope from rolling OLS. |
| **Derived** | `VolumeConfirmedMomentum` | `R126 × Volume / VolMA60`. |

### 4.2 How features were screened

`src/research/feature_correlation.py` computes the Pearson correlation between 37 candidate features across the whole stock-day panel and lists pairs with |ρ| ≥ 0.80. **41 pairs** exceed that level (`results/highly_correlated_feature_pairs.csv`). Examples:

| Pair | ρ |
|---|---|
| RSI14 – RSI21 | 0.979 |
| Trend180 – MA50/MA200 | 0.945 |
| RSI7 – RSI14 | 0.936 |
| RSI14 – DI_Diff | 0.929 |
| Trend126 – MA50/MA200 | 0.923 |
| Momentum6_1 – MA50/MA200 | 0.908 |
| Momentum6_1 – Trend126 | 0.886 |

What this means for model design:
- **Redundant variants were not stacked.** Each model takes one representative per idea, for example RSI14 only (not RSI7/21), Trend126 only, and `MAStructure` as one average of two MA ratios, instead of every MA/EMA/MACD variant.
- Momentum, trend and MA structure are correlated (0.8–0.9), so a composite of them is largely one underlying "uptrend" signal. The composite therefore diversifies across measurement methods more than across independent information. This is an honest limit of the design.
- The 126-day trend window was chosen partly from the sensitivity test (Section 7.5), and momentum windows from the standard 3/6/12-month convention.

**Not done:** there was no statistical feature selection (for example information coefficients, regularised regression, or tree-based importance) fitted to forward returns. Selection is by economic rationale plus correlation screening plus backtest comparison. `plots/feature_importance.png` is a report chart and does not drive the model.

---

## 5. From features to a model (AlphaScore)

Implemented in `src/signals/composite.py` and `src/signals/ranking.py`.

### 5.1 Cross-sectional percentile ranking

For one formation date, every stock's raw feature value is converted to a percentile rank among all eligible stocks:

```
percentile = (rank_average − 1) / (N − 1)        # 1.0 = best, 0.0 = worst
```

Reasons:
- **Comparable scales.** Momentum is a fraction, RSI is 0–100, volume ratios are unbounded. Percentiles put them all on [0, 1].
- **Robust to outliers.** A stock up 340% and a stock up 120% differ by one rank step, so one extreme value cannot dominate the blend.
- **Cross-sectional by construction.** The task is "which stocks are best relative to the others on this day", which is exactly a rank.

A stock is eligible only if every feature that the model needs is non-null on the formation date (for example `Momentum12_1` needs about a year of history). This is why models have 239–244 eligible stocks at 2021-01-01.

### 5.2 Combining factors

```
AlphaScore = Σ_i  percentile_i × (weight_i / Σ weights)
```

Weights are fixed by hand in `MODEL_DEFINITIONS`, not fitted. The stock with the highest AlphaScore ranks 1.

### 5.3 The 15 models

Single-factor models test one idea in isolation; composite models blend ideas. (Full formulas are in `results/strategy_catalog.csv`.)

| # | Model | Idea |
|---|---|---|
| 1 | `momentum_12_1` | 12-month return, skipping latest month |
| 2 | `momentum_6m` | 6-month return, skipping latest month |
| 3 | `momentum_3m` | 3-month return, skipping latest month |
| 4 | `multi_horizon_momentum` | Equal blend of 3/6/12-month momentum |
| 5 | `momentum_relative_strength` | Momentum12_1 + RS126 |
| 6 | `momentum_trend` | Momentum12_1 + Trend126 strength |
| 7 | `relative_strength` | RS126 only |
| 8 | `trend_strength` | Trend126 strength only |
| 9 | `trend_quality` | Trend126 + CloseTo52WHigh + MAStructure |
| 10 | `high_52w` | CloseTo52WHigh only |
| 11 | `breakout` | ATR-scaled distance above 252-day high |
| 12 | `volume_confirmation` | `R126 × Volume/VolMA60` |
| 13 | `risk_adjusted_momentum` | `Momentum12_1 / Vol126` |
| 14 | `technical_composite` | Equal blend of 8 factors incl. RSI14, ADX14 |
| 15 | `composite_quantitative` | Weighted blend of 9 factors (final model) |

### 5.4 The final model's weights

```
Momentum12_1 0.25 | Momentum6_1 0.15 | Momentum3_1 0.10   → momentum block 0.50
RS126 0.15                                                 → relative strength 0.15
Trend126_TrendStrength 0.10 | MAStructure 0.05             → trend 0.15
CloseTo52WHigh 0.10                                        → price position 0.10
VolumeConfirmedMomentum 0.05 | RiskAdjMomentum 0.05        → confirmation/risk 0.10
```

The design logic (a judgement, not an optimisation):
- Half the weight goes to momentum, the factor with the longest empirical record, spread over three horizons so no single lookback decides the result.
- Relative strength, trend quality and 52-week-high proximity each add a different view of the same trend.
- Volume confirmation and risk-adjusted momentum get small weights as filters, because a high-volume move is more credible and a smooth gain is preferable to a volatile one.
- These weights were set by hand and were **not tuned** against backtest results.

---

## 6. Backtest engine

`src/backtest/` and `src/research/backtest_runner.py`.

| Rule | Implementation |
|---|---|
| Strategy type | Static: buy once, hold, never rebalance (a design choice). `validate_no_rebalancing` raises an error if any symbol has more than one trade. |
| Capital | ₹1,00,00,000 |
| Entry | Open of the first trading day **strictly after** the formation date (`get_entry_price`). No same-day execution. |
| Transaction cost | 0.1% of each position value, charged once at entry (`shares = (position_value − cost) / entry_price`) |
| Holding | Unchanged through the end date; no exit trade or exit cost is charged |
| NAV | Sum of `shares × Close` across holdings, starting once every position has been filled |
| Risk-free rate | 0% |
| Max single weight | 20%, enforced iteratively with redistribution (`cap_and_renormalize`) |

### 6.1 Metrics (`src/backtest/metrics.py`)

```
CAGR            = (FinalValue / InitialValue)^(365.25 / calendar_days) − 1
Annual vol      = std(daily returns, ddof=1) × √252
Max drawdown    = min(NAV_t / running_max(NAV)_t − 1)
Sharpe          = mean(excess daily return) / std(excess daily return) × √252
Sortino         = same, with std of negative excess returns only
Gain/Loss ratio = mean(positive daily returns) / |mean(negative daily returns)|
Win rate        = share of days with a positive return
```

---

## 7. How the strategy was chosen — the experiments

Each stage held most choices fixed and varied one, so the effect can be isolated. All use top 10, 2021-01-01 formation unless stated.

### 7.1 Stage A — Single-model baseline

`momentum_baseline.py` backtests plain 12-1 momentum to create a reference: CAGR 55.2%, Sharpe 1.52, max drawdown −33.6%.

### 7.2 Stage B — Model comparison (15 alpha models, one formation date)

Results from `results/model_comparison.csv` (backtest end 2025-12-31):

| Model | CAGR | Sharpe | Max DD |
|---|---|---|---|
| momentum_6m | 59.9% | 1.65 | −29.6% |
| momentum_relative_strength | 57.5% | 1.57 | −32.9% |
| relative_strength | 56.1% | 1.59 | −33.3% |
| momentum_12_1 | 55.2% | 1.52 | −33.6% |
| trend_strength | 53.6% | 1.51 | −34.8% |
| momentum_trend | 53.7% | 1.50 | −34.7% |
| **composite_quantitative** | **53.7%** | **1.50** | **−34.7%** |
| multi_horizon_momentum | 53.8% | 1.48 | −35.0% |
| risk_adjusted_momentum | 49.1% | 1.55 | −37.2% |
| momentum_3m | 49.0% | 1.42 | −38.0% |
| volume_confirmation | 46.4% | 1.64 | −29.5% |
| technical_composite | 45.9% | **1.67** | **−26.1%** |
| trend_quality | 34.6% | 1.37 | −37.6% |
| breakout | 34.3% | 0.88 | −50.0% |
| high_52w | 34.2% | 0.87 | −50.3% |

Observations:
- Momentum and relative-strength families lead on return.
- Pure 52-week-high and breakout signals are the weakest, with roughly 50% drawdowns (they pick stocks already at highs that can be fragile).
- `technical_composite` has the best Sharpe and smallest drawdown but a lower CAGR.
- `momentum_trend` and `composite_quantitative` give **identical** metrics, meaning they selected the same ten stocks at this date.
- This comparison uses a single formation date, so rankings among close models are not statistically meaningful.

### 7.3 Stage C — Is the ranking robust? (Experiment grid)

`experiments.py` runs **2,160 backtests**:

```
10 signals × (top-N + correlation-filtered × 5 thresholds) × 4 weighting methods × 9 formation dates
(2021-01-01 … 2025-01-01, every six months), each held to the end of the data
```

Average across all runs, by signal:

| Signal | Avg Sharpe | Avg CAGR | Avg max DD |
|---|---|---|---|
| momentum_3m | 1.23 | 40.2% | −33.3% |
| momentum_12_1 | 1.20 | 35.6% | −32.1% |
| momentum_6m | 1.20 | 36.7% | −32.3% |
| trend_strength | 1.19 | 34.1% | −35.2% |
| relative_strength | 1.12 | 34.3% | −31.7% |
| multi_horizon_momentum | 1.09 | 34.3% | −34.3% |
| high_52w | 0.95 | 22.0% | −30.3% |
| breakout | 0.95 | 23.1% | −30.4% |
| technical_composite | 0.94 | 23.6% | −29.8% |
| volume_confirmation | 0.83 | 20.7% | −31.7% |

By formation date (average across all runs): Sharpe is highest for 2021-01-01 (1.31), 2022-07-01 (1.38), 2023-07-01 (1.37) and 2024-01-01 (1.32), and falls for later starts: 2024-07-01 (0.70) and 2025-01-01 (0.13), which have much less time to compound. Only 96 of 2,160 runs had a negative total return.

Takeaways:
- **Momentum families are the most consistent.** They lead on average across dates, not just at 2021-01-01.
- **Rank order changes between single-date and multi-date views.** For instance, `technical_composite` leads at one date but sits near the bottom on average, so a single-date win is a weak reason to choose it. This is a main reason a diversified, momentum-anchored composite was preferred over the single-date winner.
- The grid's best single run (Sharpe 2.07, `trend_strength`, July 2023, alpha-weighted) is an outlier-maximum of 2,160 trials and should not be read as a recommendation.
- The grid does not include `composite_quantitative`; it was evaluated in the Stage B/D/E comparisons.

### 7.4 Stage D — Does correlation-aware selection help?

Method: take the top 30 by AlphaScore, then walk down the list keeping a stock only if its absolute correlation of trailing 120-day returns with every already-selected stock is below a threshold (0.70/0.75/0.80/0.85/none); if fewer than 10 pass, fill from the next-best candidates.

| Selection | Avg Sharpe (grid) | Avg CAGR | Avg max DD |
|---|---|---|---|
| top_n | 1.068 | 30.5% | −32.2% |
| correlation_filtered | 1.069 | 30.4% | −32.1% |

- Paired against plain top-N, a 0.70 threshold changes the result in only 18% of cells (82% are identical); the average Sharpe difference is +0.003.
- For `composite_quantitative` at 2021-01-01, all five thresholds produced identical metrics (CAGR 53.7%, Sharpe 1.50), meaning the filter removed no names.
- **Conclusion:** no measurable benefit, so the simpler top-N rule was kept (fewer moving parts, fewer parameters to overfit).

### 7.5 Stage E — Sensitivity of lookback windows

`sensitivity.py` sweeps windows across the same nine formation dates and reports the average Sharpe and the fraction of positive-return periods:

| Family | Window (days) | Avg Sharpe | Positive periods |
|---|---|---|---|
| 3M momentum | 50 / 63 / 75 | 1.18 / 1.17 / 1.12 | 100% / 100% / 100% |
| 6M momentum | 100 / 126 / 150 | 1.20 / 1.05 / 1.15 | 100% / 89% / 89% |
| 12M momentum | 200 / 252 / 300 | 1.19 / 1.18 / 1.15 | 100% / 100% / 100% |
| Trend strength | 60 / 90 / 126 / 180 | 1.00 / 1.16 / 1.19 / 1.09 | 89% / 100% / 100% / 89% |

- 3M and 12M momentum are stable plateaus: Sharpe barely moves across windows.
- 6M momentum at exactly 126 days is the weakest of its neighbours (1.05 vs 1.20 at 100 days), a sign that the 6M result is somewhat sensitive to the window and the standard value is not special.
- Trend strength peaks around 90–126 days; 126 was kept.
- This supports using **several horizons together** (the final model blends 3/6/12-month momentum) instead of betting on one window.

---

## 8. Why equal weights, and not alpha- or risk-based weights?

The final portfolio is equal-weighted (10% each, the 20% cap never binds). Five methods were implemented in `src/portfolio/weighting.py`:

| Method | Formula |
|---|---|
| A. Equal | `w = 1/N` |
| B. Alpha | `w ∝ AlphaScore` |
| C. Alpha/Volatility | `w ∝ AlphaScore / Vol126` |
| D. Inverse volatility | `w ∝ 1 / Vol126` |
| E. Risk-adjusted alpha | average of C and D |

All are capped at 20% and renormalised.

### 8.1 The evidence

On the final model (`weighting_method_comparison.csv`):

| Method | CAGR | Sharpe | Max DD |
|---|---|---|---|
| Equal | 53.7% | 1.498 | −34.7% |
| Alpha | 53.7% | 1.501 | −34.5% |
| Alpha/Vol | 51.3% | 1.486 | −33.0% |
| Inverse vol | 51.4% | 1.484 | −33.2% |
| Risk-adjusted alpha | 51.4% | 1.485 | −33.1% |

Across the full 2,160-run grid (540 runs per method): average Sharpe is 1.071 (alpha/vol), 1.069 (inverse vol), 1.069 (alpha), 1.068 (equal). In paired top-N comparisons (same model and formation date), mean Sharpe differences between equal weight and the others were within ±0.004.

### 8.2 The reasoning

1. **No meaningful improvement from alternatives.** Sharpe differences are at most about 0.01 on the final model and 0.003 across the grid, well inside noise for a sample of nine start dates.
2. **Alpha weighting is nearly equal weighting anyway.** Percentile scores of the top 10 lie in a narrow band (0.905–0.957 at 2021-01-01), so alpha weights run only from about 9.8% to 10.3%. A rank-based score doesn't contain enough information to justify bigger bets on rank 1 than rank 10.
3. **Volatility weighting trades return for a small drawdown reduction.** It cuts max drawdown by about 1.5 points (−34.7% → −33.0%) but costs roughly 2.4 points of CAGR on the final model. That is a real, repeatable trade-off, but not an improvement in risk-adjusted terms (Sharpe slightly lower).
4. **Fewer estimated parameters.** Equal weighting uses no volatility or covariance estimates, and for a portfolio held for years without rebalancing, a 126-day volatility estimate taken at formation goes stale. The weights are fixed once and never touched again.
5. **Markowitz check.** A minimum-variance variant (trailing 120-day covariance, 20% cap) on a different signal (`momentum_3m`) gave Sharpe 1.42 vs 1.42 for equal weight static, so optimisation did not add value (note this is not a perfectly like-for-like test: its entry convention differs slightly).
6. **Diversification and transparency.** Ten equal positions cap any single failure at 10% of capital and make the portfolio easy to explain.

This is a considered judgement based on the evidence above. It does not prove equal weighting is optimal; a larger or longer sample might separate the methods.

### 8.3 Rebalancing was also tested and rejected

Using `momentum_3m` formed on 2021-01-01 (`results/strategy_comparison.csv`):

| Strategy | CAGR | Sharpe | Max DD |
|---|---|---|---|
| Rebalance every 6M | 46.7% | 1.55 | −28.9% |
| Rebalance every 3M | 41.3% | 1.48 | −35.3% |
| Markowitz (static) | 48.7% | 1.42 | −37.7% |
| Static (buy and hold) | 49.0% | 1.42 | −38.0% |
| Rebalance 1M | 37.5% | 1.30 | −28.9% |
| Rebalance 1W | 33.7% | 1.19 | −37.2% |
| Rebalance 1D | 23.1% | 0.78 | −48.5% |

Frequent rebalancing hurts, mostly because turnover costs and signal noise accumulate. Slow rebalancing (six months) has a better Sharpe and drawdown than static buy and hold but lower CAGR. Static was our own design choice, not a verified brief requirement, so rebalancing was a legitimate candidate; it was set aside here because of the cost and noise evidence above, and was re-examined with a leak-free walk-forward protocol in `reports/battle_ready_report.html`. Note that the rebalancing engine in `strategy_comparison.py` charges transaction costs only on the portfolio value at each rebalance; treat its figures as indicative.

---

## 9. The final portfolio and out-of-sample result

### 9.1 Why `composite_quantitative`?

It was chosen over higher single-date performers (`momentum_6m` CAGR 59.9%, `technical_composite` Sharpe 1.67) because:
- it blends nine factors across momentum, relative strength, trend, price position, volume, and risk, so it doesn't depend on one lookback or one indicator;
- it is momentum-anchored, matching the factor family that was consistently strong across all nine formation dates (Section 7.3);
- the single-date winners were not consistent winners across the grid;
- its in-sample result (Sharpe 1.50, CAGR 53.7%) is in the same range as the best momentum models, so little is given up.

This was a judgement call. The code (`build_report.py`) fixes `MODEL_NAME = "composite_quantitative"`. It was not selected by an automated search, and the evidence above does not prove it is the best model.

### 9.2 Selected stocks (formed 2021-01-01)

From `results/final_selection_explained.csv`:

| Rank | Stock | AlphaScore | Top-10 in N of 15 models | OOS return (2026-01-01 → 2026-10-30) |
|---|---|---|---|---|
| 1 | GPIL | 0.957 | 12 | −26.3% |
| 2 | ADANIENT | 0.950 | 10 | +14.9% |
| 3 | ATGL | 0.931 | 8 | −3.8% |
| 4 | CGPOWER | 0.929 | 10 | +25.8% |
| 5 | TATAELXSI | 0.922 | 6 | −27.2% |
| 6 | DIXON | 0.920 | 8 | −1.4% |
| 7 | AFFLE | 0.918 | 8 | −20.8% |
| 8 | PGEL | 0.913 | 10 | −6.0% |
| 9 | ADANIGREEN | 0.911 | 10 | +29.7% |
| 10 | LAURUSLABS | 0.905 | 9 | +114.6% |

Each stock gets 10%. Stocks ranked 11 onward (LTM at 0.898, APLAPOLLO at 0.892, …) fell short of the cutoff; `ScoreGapToCutoff` shows by how much. Strong agreement across models (several are top-10 in 10 or more of the 15) indicates these were not artefacts of one factor mix. `results/stock_rankings.csv` and `results/final_selection_explained.csv` show the factor-by-factor breakdown.

### 9.3 Performance

| Period | Total return | CAGR | Vol | Sharpe | Sortino | Max DD |
|---|---|---|---|---|---|---|
| In-sample 2021-01-04 → 2025-12-31 | +754.1% | 53.7% | 32.8% | 1.50 | 1.87 | −34.7% |
| Out-of-sample 2026-01-01 → 2026-10-30 | +7.6% | 9.3% (annualised) | 24.3% | 0.49 | 0.73 | −16.6% |

- Equal-weighted universe proxy over the same periods: +329.7% in-sample and −4.8% out-of-sample, so the portfolio beat the proxy in both, though in 2026 the margin is much smaller in absolute terms.
- The out-of-sample result re-enters the same ten stocks at the open after 2025-12-31 with fresh capital and the same weights, as `build_report.py` does. The existing HTML/Excel report covers 2026 H1 only (+12.9%, Sharpe 1.07), because `features.parquet` ends on 2026-06-30. The numbers above extend this to 2026-10-30 using raw price history. Re-run `python run_pipeline.py` (after the missing-data issue is resolved) to refresh the report files.
- Ten months is a short window and ten stocks is a small, concentrated sample: LAURUSLABS alone (+115%) contributes most of the gain, while five of the ten holdings lost money.

---

## 10. Decision log

Each row records what had to be decided, how it was decided, and what was finally done. "Evidence" means a backtest or data check in this repo; "Judgement" means a reasoned choice without a test proving it; "Given" means set by the competition brief or `src/config.py`.

| # | Stage | Decision to make | How it was decided | Result |
|---|---|---|---|---|
| 1 | Setup | Capital, number of holdings, cost, risk-free rate | Given: `INITIAL_CAPITAL` ₹1 cr, `MAX_HOLDINGS` 10, `TRANSACTION_COST_RATE` 0.1%, `RISK_FREE_RATE` 0% in `src/config.py` | Fixed for all backtests |
| 2 | Setup | Static portfolio or periodic rebalancing | Judgement: static was our own design choice (the repo only records it as a design note; no brief text requiring it was verified). Rebalancing was tested as a diagnostic (Section 8.3) and fully revisited in `reports/battle_ready_report.html` | Static buy-and-hold; `validate_no_rebalancing` enforces it |
| 3 | Data | Which universe | Given: Nifty 100, Midcap 100, Smallcap 100 lists in `Stocks/` | 300 stocks, NSE EQ series only |
| 4 | Data | Timeframe | Judgement: daily bars are enough for a hold-for-years strategy, and keep download and storage small | `TIMEFRAME = "1d"` |
| 5 | Data | Is the data usable as downloaded? | Evidence: audit for nulls, duplicates, invalid OHLC, date range (Section 3.2) | All 300 pass; gaps found (mainly 2024-07-19, FORCEMOT) |
| 6 | Data | Extend to end of October 2026 and handle stray rows | Evidence: all stocks end 2026-10-30 (Oct 31 is a Saturday); rows dated after 2026-10-31 (stray 2026-11-02) were excluded | Audit cut off at 2026-10-31 |
| 7 | Data | Fill the missing dates by re-downloading from 5Paisa? | Judgement: gaps are about 311 observations, so retry only the affected stocks and dates, sample first, merge without overwriting | Recommended, **not yet done** |
| 8 | Data | Benchmark | Constraint: no index series stored locally; chose an equal-weighted universe index as a stand-in | Synthetic benchmark, flagged as a limitation |
| 9 | Features | Which feature families to build | Judgement: standard momentum, relative strength, trend, breakout, technical, volatility and volume families | 98 columns per stock-day |
| 10 | Features | Skip the most recent month in momentum | Judgement: the one-month skip avoids short-term reversal | `Momentum3_1`, `6_1`, `9_1`, `12_1` use `Close[t−21]` |
| 11 | Features | Avoid look-ahead | Design rule: rolling windows only use past data, and the high/low windows are shifted by one day; `tests/test_lookahead.py` covers momentum and trend only | Applied everywhere; the test covers a subset |
| 12 | Features | Plain slope or quality-adjusted trend | Judgement: slope × R² rewards clean trends | `TrendStrength = Slope × R²` |
| 13 | Feature selection | Which of the correlated features to keep | Evidence: pairwise correlation over 37 features, 41 pairs at |ρ| ≥ 0.80 | One representative per idea in each model (for example RSI14 only, Trend126 only) |
| 14 | Modelling | How to put different-unit features on one scale | Judgement: cross-sectional percentile rank, robust to outliers | `(rank − 1)/(N − 1)` |
| 15 | Modelling | How to combine factors | Judgement: weighted average of percentiles, with fixed hand-set weights, not fitted | `AlphaScore` |
| 16 | Modelling | Which stocks are eligible | Rule: all required factors must be non-null at the formation date | 239–244 eligible at 2021-01-01 |
| 17 | Strategy search | Which ideas to test | Judgement: 15 models covering momentum, relative strength, trend, price position/breakout, volume, risk-adjusted and composites | 15 models in `MODEL_DEFINITIONS` |
| 18 | Strategy search | Compare models fairly | Evidence: same formation date, same top 10, same equal weights, same costs | `model_comparison.csv` |
| 19 | Selection rule | Plain top-N or correlation-filtered | Evidence: 5 thresholds (0.70–0.85, none) across the grid; filter changed only about 18% of cells, with an average Sharpe gain of about 0.003 | Kept plain top-N (simpler, fewer parameters) |
| 20 | Weighting | Equal, alpha, alpha/vol, inverse-vol or risk-adjusted | Evidence: weighting test on the final model plus the 2,160-run grid (Section 8); differences within about 0.01 Sharpe | Equal weight, 10% each (cap of 20% never binds) |
| 21 | Weighting | Use Markowitz optimisation? | Evidence: min-variance test on `momentum_3m`, Sharpe 1.42, the same as static | Not used |
| 22 | Rebalancing | Rebalance at any frequency? | Evidence: 1D/1W/1M/3M/6M versus static; frequent rebalancing was clearly worse and 6M slightly better in Sharpe but lower in CAGR | Not used in this report's static design; superseded by the walk-forward study in `reports/battle_ready_report.html` |
| 23 | Robustness | Do rankings hold across start dates? | Evidence: 9 formation dates, 2021-01 to 2025-01 | Momentum families led on average, and some single-date winners dropped (for example `technical_composite`) |
| 24 | Robustness | Are lookback windows sensitive? | Evidence: sensitivity sweeps over momentum and trend windows (Section 7.5) | Broad plateau, so blend 3/6/12-month horizons in the final model |
| 25 | Final model | Which model to submit | Judgement informed by evidence: diversified, momentum-anchored, Sharpe 1.50, not dependent on a single lookback; single-date winners were not consistent across dates | `composite_quantitative` |
| 26 | Final model | Factor weights inside the composite | Judgement: 50% momentum across three horizons, 15% relative strength, 15% trend, 10% price position, 10% confirmation/risk; not optimised | 0.25/0.15/0.10/0.15/0.10/0.10/0.05/0.05/0.05 |
| 27 | Execution | When to trade after the signal | Rule: next trading day's open after the formation date, to avoid same-bar execution | Entry at open 2021-01-04 |
| 28 | Execution | How to charge costs | Given rate; simplification: 0.1% once at entry, no exit cost, no slippage | Flat entry cost only |
| 29 | Formation date | Which date to form the portfolio | Judgement: 2021-01-01 gives 5 years of in-sample history and 2026 as a trading window; it is also one of the best dates in the grid, so this is a risk of favourable timing | 2021-01-01 |
| 30 | Out-of-sample | How to evaluate 2026 | Rule: keep the same picks and weights, re-enter at the first 2026 price; no changes after seeing results | +7.6% to 2026-10-30, Sharpe 0.49 |
| 31 | Reporting | Which results to export for explainability | User request: strategy catalog, stock rankings, and a trace of the final selection | Three CSVs in `results/` |
| 32 | Reporting | Refresh Excel/HTML to October 2026? | Not decided: the report build was not re-run; features end 2026-06-30 | Open; existing reports cover 2026 H1 only |

### 10.1 Which decisions carried the most weight

- **#25 (model), #26 (factor weights) and #29 (formation date)** are judgement calls and drive most of the headline result. They are the most exposed to in-sample bias.
- **#19, #20, #21 and #22** were settled by testing, and in each case the simpler option was kept because the alternatives did not clearly help.
- **#7 and #32** remain open and are listed under next steps in Section 11.

---

## 11. Limitations and risks

1. **In-sample selection.** Model comparison, weighting, and correlation tests were all run on data that includes the backtest period. The 2026 period is the only out-of-sample window, and it is short.
2. **Formation-date dependence.** Results vary strongly with the start date (average Sharpe 1.31 at 2021-01-01 vs 0.13 at 2025-01-01 in the grid). The 2021-01-01 start is one of the strongest dates.
3. **Survivorship and look-ahead in the universe.** The universe is today's constituent list applied to 2021. Stocks that were later added to indices (often winners) are included, while those that dropped out are not. This likely inflates in-sample results. Several holdings (for example the Adani group names) have idiosyncratic histories.
4. **Concentration.** Ten stocks, several in related sectors (Adani group: ADANIENT, ATGL, ADANIGREEN; capital goods; consumer durables), so diversification is limited.
5. **Benchmark is a proxy.** There is no real Nifty index series in the repo; the comparison uses an equal-weighted average of the universe.
6. **Unverified corporate-action adjustment.** Unadjusted splits or bonuses could create false momentum or drawdowns.
7. **Missing data.** About 311 stock-date observations are missing (mostly 2024-07-19 and FORCEMOT). Gaps affect rolling features and the date alignment used when joining stocks for the NAV (an inner join on dates).
8. **Stale feature file.** `features.parquet` ends 2026-06-30, so experiment backtests that read its latest date stop there. The final 2021 formation signal is unaffected.
9. **Backtest simplifications.** Cost is a flat 0.1% at entry only; no slippage, taxes, liquidity limits or exit cost. Small caps such as PGEL may be hard to trade at size.
10. **Multiple testing.** 2,160 grid runs and 15 models create a risk that the best-looking result is partly luck.
11. **Hand-set factor weights.** The 9-factor weights were not optimised or validated out-of-sample.

### Suggested next steps

- Re-fetch the missing dates from 5Paisa, then rebuild features and re-run the pipeline.
- Replace the synthetic benchmark with real Nifty index data.
- Test a walk-forward design: choose the model on data to one date, then evaluate strictly afterwards.
- Check corporate-action adjustment for stocks with large single-day moves (flagged in the audit).
- Add a sector or group concentration limit.

---

## 12. Output files

| File | Contents |
|---|---|
| `results/data_audit_report.csv` | Per-stock data-quality audit through 2026-10-31 |
| `results/strategy_catalog.csv` | The 15 models: category, indicators, formulas, weights, results |
| `results/stock_rankings.csv` | Rank, score and factor breakdown for every stock under every model at 2021-01-01 |
| `results/final_selection_explained.csv` | How the final ten were chosen, with cross-strategy ranks and 2026 out-of-sample returns |
| `results/model_comparison.csv` | Single-date backtest of all 15 models |
| `results/experiment_results.csv` | 2,160-run experiment grid |
| `results/weighting_method_comparison.csv` | Five weighting methods |
| `results/correlation_threshold_comparison.csv` | Correlation filter thresholds |
| `results/strategy_comparison.csv` | Static vs rebalanced vs Markowitz |
| `results/momentum_lookback_sensitivity.csv`, `results/trend_window_sensitivity.csv` | Window sensitivity |
| `results/feature_correlation_matrix.csv`, `results/highly_correlated_feature_pairs.csv` | Feature redundancy |
| `results/competition_submission.xlsx`, `reports/performance_report.html` | Submission workbook and HTML report (currently 2026 H1 only) |

Reproduce with `python run_pipeline.py`, or run a single stage, for example `python -m src.research.final_selection`.
