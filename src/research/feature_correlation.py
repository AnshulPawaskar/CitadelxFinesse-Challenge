"""Correlation analysis across candidate features, to spot redundant signals before combining them
into a composite score (e.g. RSI/MACD/MA ratios often all just proxy for the same trend information)."""
import polars as pl

from src.config import RESULTS_DIR

# Representative numeric features spanning momentum, RS, trend, technical, volatility and volume.
CANDIDATE_FEATURES = [
    "Momentum3_1", "Momentum6_1", "Momentum9_1", "Momentum12_1",
    "RS21", "RS63", "RS126", "RS252",
    "Trend60_TrendStrength", "Trend90_TrendStrength", "Trend126_TrendStrength", "Trend180_TrendStrength",
    "CloseToMA20", "CloseToMA50", "CloseToMA100", "CloseToMA200", "MA20_MA50", "MA50_MA200",
    "RSI7", "RSI14", "RSI21", "MACD_Hist", "ADX14", "DI_Diff",
    "CloseTo52WHigh", "CloseTo126DHigh", "BreakoutDistance252D_ATR",
    "Vol20", "Vol60", "Vol126", "ATR14_Close", "MaxDrawdown252D",
    "VolumeToMA20", "VolumeToMA60", "VolumeZScore", "OBV_Slope", "PriceVolumeCorr20",
]

HIGH_CORRELATION_THRESHOLD = 0.80


def compute_feature_correlation(features_df: pl.DataFrame, columns: list[str] = CANDIDATE_FEATURES) -> pl.DataFrame:
    """Pearson correlation matrix across the full panel (all symbols, all dates), ignoring nulls pairwise."""
    available = [c for c in columns if c in features_df.columns]
    numeric = features_df.select(available).drop_nulls()

    corr = numeric.corr()
    return corr.with_columns(pl.Series("Feature", available)).select(["Feature"] + available)


def find_highly_correlated_pairs(corr_matrix: pl.DataFrame, threshold: float = HIGH_CORRELATION_THRESHOLD) -> pl.DataFrame:
    """Long-format list of feature pairs whose |correlation| exceeds the threshold (excluding self-pairs)."""
    features = corr_matrix["Feature"].to_list()
    rows = []
    for i, feature_a in enumerate(features):
        for feature_b in features[i + 1:]:
            value = corr_matrix.filter(pl.col("Feature") == feature_a)[feature_b][0]
            if value is not None and abs(value) >= threshold:
                rows.append({"FeatureA": feature_a, "FeatureB": feature_b, "Correlation": value})
    return pl.DataFrame(rows).sort("Correlation", descending=True) if rows else pl.DataFrame(
        schema={"FeatureA": pl.Utf8, "FeatureB": pl.Utf8, "Correlation": pl.Float64}
    )


def run(write_output: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    features_df = pl.read_parquet(RESULTS_DIR / "features.parquet")
    corr_matrix = compute_feature_correlation(features_df)
    high_corr_pairs = find_highly_correlated_pairs(corr_matrix)

    if write_output:
        corr_matrix.write_csv(RESULTS_DIR / "feature_correlation_matrix.csv")
        high_corr_pairs.write_csv(RESULTS_DIR / "highly_correlated_feature_pairs.csv")

    print(f"{high_corr_pairs.height} feature pairs with |correlation| >= {HIGH_CORRELATION_THRESHOLD}:")
    print(high_corr_pairs)
    return corr_matrix, high_corr_pairs


if __name__ == "__main__":
    run()
