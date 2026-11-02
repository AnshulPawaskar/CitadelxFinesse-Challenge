"""Export the full per-stock ranking for every alpha model so it's clear why each stock was (or wasn't) selected.

One row per (Model, Symbol). For every factor in the model we store the raw value, its cross-sectional
percentile (1.0 = best) and its weighted contribution to AlphaScore, so AlphaScore = sum of the contributions.
"""
from datetime import datetime

import polars as pl

from src.config import RESULTS_DIR, MAX_HOLDINGS
from src.signals.composite import MODEL_DEFINITIONS, get_cross_section
from src.signals.ranking import to_percentile

FORMATION_DATE = datetime(2021, 1, 1)


def build_model_ranking(features: pl.DataFrame, formation_date, model_name: str, n_selected: int = MAX_HOLDINGS) -> pl.DataFrame:
    weights = MODEL_DEFINITIONS[model_name]
    total_weight = sum(weights.values())
    cross_section, last_date = get_cross_section(features, formation_date)
    cross_section = cross_section.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in weights]))

    contribution_cols = []
    for factor, weight in weights.items():
        cross_section = to_percentile(cross_section, factor)
        cross_section = cross_section.with_columns(
            (pl.col(f"{factor}_Percentile") * weight / total_weight).alias(f"{factor}_Contribution")
        )
        contribution_cols.append(f"{factor}_Contribution")

    cross_section = cross_section.with_columns(pl.sum_horizontal(contribution_cols).alias("AlphaScore"))
    # Same ordering as select_top_n (stable sort), so Selected matches the backtested portfolios.
    ranked = cross_section.sort("AlphaScore", descending=True, maintain_order=True)
    ranked = ranked.with_row_index("Rank", offset=1)

    rows = []
    for row in ranked.iter_rows(named=True):
        ordered = sorted(weights, key=lambda f: row[f"{f}_Contribution"], reverse=True)
        top = ordered[0]
        reason = (
            f"Rank {row['Rank']} of {ranked.height}; AlphaScore {row['AlphaScore']:.3f}. "
            f"Largest driver: {top} (raw {row[top]:.4g}, percentile {row[f'{top}_Percentile']:.2f})."
        )
        selected = row["Rank"] <= n_selected
        rows.append({
            "Model": model_name,
            "FormationDate": formation_date,
            "DataDate": last_date,
            "Symbol": row["Symbol"],
            "Rank": row["Rank"],
            "EligibleStocks": ranked.height,
            "AlphaScore": row["AlphaScore"],
            "Selected": selected,
            "Decision": "SELECTED (top " + str(n_selected) + ")" if selected else "Not selected",
            "ScoreGapToCutoff": row["AlphaScore"] - ranked["AlphaScore"][n_selected - 1],
            "TopDriver": top,
            "Explanation": reason,
            "FactorBreakdown": "; ".join(
                f"{f}: raw={row[f]:.4g}, pct={row[f'{f}_Percentile']:.2f}, weight={weights[f] / total_weight:.2f}, "
                f"contrib={row[f'{f}_Contribution']:.3f}"
                for f in ordered
            ),
        })
    return pl.DataFrame(rows)


def run(formation_date=FORMATION_DATE, write_output: bool = True) -> pl.DataFrame:
    features = pl.read_parquet(RESULTS_DIR / "features.parquet")
    rankings = pl.concat([build_model_ranking(features, formation_date, m) for m in MODEL_DEFINITIONS], how="vertical")
    if write_output:
        rankings.write_csv(RESULTS_DIR / "stock_rankings.csv")
        print(f"Wrote {rankings.height} rows ({rankings['Model'].n_unique()} models) to {RESULTS_DIR / 'stock_rankings.csv'}")
    return rankings


if __name__ == "__main__":
    run()
