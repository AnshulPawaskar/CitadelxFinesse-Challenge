"""Explain how the final submission portfolio was reached: factor weights -> factor ranks -> composite
AlphaScore -> Top-N -> portfolio weights -> out-of-sample (2026) outcome, with every other strategy's
rank for the same stock shown alongside for cross-strategy context.

The final strategy mirrors src/reporting/build_report.py (composite_quantitative, formed 2021-01-01,
top 10, equal weight, held unchanged into the 2026 out-of-sample period).
"""
from datetime import datetime

import polars as pl

from src.config import RESULTS_DIR, MAX_HOLDINGS
from src.data.loader import load_universe_history
from src.data.universe import get_symbol_to_scrip_data
from src.portfolio.selector import select_top_n
from src.portfolio.weighting import equal_weight
from src.reporting.build_report import FORMATION_DATE, BACKTEST_END_DATE, MODEL_NAME
from src.research.stock_rankings import build_model_ranking
from src.signals.composite import MODEL_DEFINITIONS, build_composite_score, get_cross_section
from src.signals.ranking import to_percentile

OOS_START = datetime(2026, 1, 1)


def _factor_table(features: pl.DataFrame, formation_date, model_name: str) -> pl.DataFrame:
    """Per-stock raw value, rank (1 = best), percentile and weighted contribution for each factor of the final model."""
    weights = MODEL_DEFINITIONS[model_name]
    total_weight = sum(weights.values())
    cross_section, _ = get_cross_section(features, formation_date)
    cross_section = cross_section.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in weights]))

    exprs = []
    for factor, weight in weights.items():
        cross_section = to_percentile(cross_section, factor)
        exprs += [
            pl.col(factor).alias(f"Raw_{factor}"),
            pl.col(factor).rank(method="min", descending=True).cast(pl.Int64).alias(f"FactorRank_{factor}"),
            pl.col(f"{factor}_Percentile").alias(f"Percentile_{factor}"),
            (pl.col(f"{factor}_Percentile") * weight / total_weight).alias(f"Contribution_{factor}"),
        ]
    return cross_section.select(["Symbol"] + exprs)


def _oos_outcome(history: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Buy at the next open after the backtest end (same rule as the pipeline), mark at the last available close."""
    rows = []
    for symbol, df in history.items():
        future = df.filter(pl.col("Datetime") > BACKTEST_END_DATE).sort("Datetime")
        if future.height == 0:
            continue
        entry_price, last_close = future["Open"][0], future["Close"][-1]
        rows.append({
            "Symbol": symbol,
            "OOS_EntryDate": future["Datetime"][0],
            "OOS_EntryPrice": entry_price,
            "OOS_LastDate": future["Datetime"][-1],
            "OOS_LastClose": last_close,
            "OOS_Return": last_close / entry_price - 1,
        })
    return pl.DataFrame(rows)


def run(write_output: bool = True) -> pl.DataFrame:
    features = pl.read_parquet(RESULTS_DIR / "features.parquet")
    history = load_universe_history(get_symbol_to_scrip_data())

    signal = build_composite_score(features, FORMATION_DATE, MODEL_NAME)
    selected = equal_weight(select_top_n(signal, n=MAX_HOLDINGS))
    weights = dict(zip(selected["Symbol"].to_list(), selected["Weight"].to_list()))

    base = build_model_ranking(features, FORMATION_DATE, MODEL_NAME).select(
        "Symbol", "Rank", "EligibleStocks", "AlphaScore", "ScoreGapToCutoff", "Selected"
    ).rename({"Rank": "FinalRank", "AlphaScore": "FinalAlphaScore"})

    factors = _factor_table(features, FORMATION_DATE, MODEL_NAME)

    # Rank of the same stock under every strategy (null = stock not eligible for that strategy at the formation date).
    all_ranks = {m: build_model_ranking(features, FORMATION_DATE, m).select("Symbol", pl.col("Rank").alias(f"StrategyRank_{m}"))
                 for m in MODEL_DEFINITIONS}
    strategy_rank_cols = [f"StrategyRank_{m}" for m in MODEL_DEFINITIONS]
    ranks = None
    for frame in all_ranks.values():
        ranks = frame if ranks is None else ranks.join(frame, on="Symbol", how="full", coalesce=True)

    ranks = ranks.with_columns([
        pl.sum_horizontal([(pl.col(c) <= MAX_HOLDINGS).fill_null(False).cast(pl.Int64) for c in strategy_rank_cols]).alias("StrategiesSelectingTop10"),
        pl.mean_horizontal(strategy_rank_cols).alias("AvgRankAcrossStrategies"),
    ])

    out = (
        base.join(factors, on="Symbol", how="left")
        .join(ranks, on="Symbol", how="left")
        .join(_oos_outcome(history), on="Symbol", how="left")
        .with_columns(
            pl.col("Symbol").replace_strict(weights, default=0.0, return_dtype=pl.Float64).alias("PortfolioWeight"),
            pl.when(pl.col("Selected")).then(pl.lit("TRADED in 2026 out-of-sample period")).otherwise(pl.lit("Not traded")).alias("FinalDecision"),
        )
        .with_columns(
            pl.when(pl.col("Selected"))
            .then(pl.format(
                "Ranked {} of {} on the {} composite (AlphaScore {}); inside the top {} cutoff, so it receives {}% of capital (equal weight, held unchanged). "
                "Also in the top 10 of {} of {} individual strategies.",
                pl.col("FinalRank"), pl.col("EligibleStocks"), pl.lit(MODEL_NAME), pl.col("FinalAlphaScore").round(3),
                pl.lit(MAX_HOLDINGS), (pl.col("PortfolioWeight") * 100).round(1),
                pl.col("StrategiesSelectingTop10"), pl.lit(len(MODEL_DEFINITIONS)),
            ))
            .otherwise(pl.format(
                "Ranked {} of {} (AlphaScore {}); {} below the cutoff score of the 10th-ranked stock, so it is not traded.",
                pl.col("FinalRank"), pl.col("EligibleStocks"), pl.col("FinalAlphaScore").round(3), pl.col("ScoreGapToCutoff").abs().round(3),
            ))
            .alias("SelectionReason")
        )
        .sort("FinalRank")
    )

    lead = ["FinalRank", "Symbol", "FinalDecision", "PortfolioWeight", "FinalAlphaScore", "ScoreGapToCutoff",
            "StrategiesSelectingTop10", "AvgRankAcrossStrategies", "SelectionReason"]
    out = out.select(lead + [c for c in out.columns if c not in lead and c not in ("Selected", "EligibleStocks")])

    if write_output:
        out.write_csv(RESULTS_DIR / "final_selection_explained.csv")
        print(f"Wrote {out.height} rows to {RESULTS_DIR / 'final_selection_explained.csv'}")
    return out


if __name__ == "__main__":
    run()
