"""Multi-factor composite alpha scoring models (MODEL 1-8). Every component is percentile-ranked
cross-sectionally before combining so raw units/scales never distort the blended score."""
import polars as pl

from src.signals.ranking import to_percentile

# Momentum/vol ratio and MA-structure aren't stored columns — derive them once per cross-section.
_DERIVED_COLUMNS = {
    "RiskAdjMomentum": (pl.col("Momentum12_1") / pl.col("Vol126")),
    "MAStructure": pl.mean_horizontal(["MA20_MA50", "MA50_MA200"]),
}

# name -> {feature_column: weight}. Weights need not sum to 1; they're normalized at combine time.
MODEL_DEFINITIONS = {
    "momentum_12_1": {"Momentum12_1": 1.0},
    "multi_horizon_momentum": {"Momentum12_1": 1.0, "Momentum6_1": 1.0, "Momentum3_1": 1.0},
    "momentum_relative_strength": {"Momentum12_1": 1.0, "RS126": 1.0},
    "momentum_trend": {"Momentum12_1": 1.0, "Trend126_TrendStrength": 1.0},
    "technical_composite": {
        "Momentum12_1": 1.0, "RS126": 1.0, "Trend126_TrendStrength": 1.0, "MAStructure": 1.0,
        "RSI14": 1.0, "ADX14": 1.0, "BreakoutDistance252D_ATR": 1.0, "VolumeConfirmedMomentum": 1.0,
    },
    "risk_adjusted_momentum": {"RiskAdjMomentum": 1.0},
    "trend_quality": {"Trend126_TrendStrength": 1.0, "CloseTo52WHigh": 1.0, "MAStructure": 1.0},
    "composite_quantitative": {
        "Momentum12_1": 0.25, "Momentum6_1": 0.15, "Momentum3_1": 0.10, "RS126": 0.15,
        "Trend126_TrendStrength": 0.10, "CloseTo52WHigh": 0.10, "VolumeConfirmedMomentum": 0.05,
        "RiskAdjMomentum": 0.05, "MAStructure": 0.05,
    },
    # Single-factor models, used by the experiment matrix to test each signal family in isolation.
    "momentum_6m": {"Momentum6_1": 1.0},
    "momentum_3m": {"Momentum3_1": 1.0},
    "relative_strength": {"RS126": 1.0},
    "trend_strength": {"Trend126_TrendStrength": 1.0},
    "high_52w": {"CloseTo52WHigh": 1.0},
    "breakout": {"BreakoutDistance252D_ATR": 1.0},
    "volume_confirmation": {"VolumeConfirmedMomentum": 1.0},
}


def get_cross_section(features_df: pl.DataFrame, formation_date) -> tuple[pl.DataFrame, object]:
    """Return the last available trading-day cross-section on or before formation_date, with the
    derived (non-stored) feature columns added."""
    available = features_df.filter(pl.col("Datetime") <= formation_date)
    if available.height == 0:
        raise ValueError(f"No feature data available on or before {formation_date}.")

    last_date = available["Datetime"].max()
    cross_section = available.filter(pl.col("Datetime") == last_date)
    cross_section = cross_section.with_columns([expr.alias(name) for name, expr in _DERIVED_COLUMNS.items()])
    return cross_section, last_date


def build_composite_score(features_df: pl.DataFrame, formation_date, model_name: str) -> pl.DataFrame:
    """Score every eligible symbol using the named model and return [Symbol, Datetime, AlphaScore]."""
    if model_name not in MODEL_DEFINITIONS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_DEFINITIONS)}")

    weights = MODEL_DEFINITIONS[model_name]
    cross_section, last_date = get_cross_section(features_df, formation_date)

    required_cols = list(weights)
    cross_section = cross_section.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in required_cols]))

    total_weight = sum(weights.values())
    score_expr = pl.lit(0.0)
    for col, weight in weights.items():
        cross_section = to_percentile(cross_section, col)
        score_expr = score_expr + pl.col(f"{col}_Percentile") * (weight / total_weight)

    cross_section = cross_section.with_columns(score_expr.alias("AlphaScore"))
    return cross_section.select(["Symbol", "Datetime", "AlphaScore"])
