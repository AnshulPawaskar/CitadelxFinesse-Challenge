"""Portfolio weighting methods (Methods A-E), assigned once at formation — never touched again
since there is no rebalancing. Every method enforces a 20% maximum individual weight."""
import polars as pl

MAX_INDIVIDUAL_WEIGHT = 0.20


def _drop_missing_volatility(selected_df: pl.DataFrame, vol_col: str) -> pl.DataFrame:
    filtered = selected_df.filter(pl.col(vol_col).is_not_null())
    dropped = selected_df.height - filtered.height
    if dropped > 0:
        print(f"Warning: dropped {dropped} symbol(s) with missing {vol_col} before risk-based weighting.")
    return filtered


def attach_volatility(selected_df: pl.DataFrame, features_df: pl.DataFrame, formation_date, vol_col: str = "Vol126") -> pl.DataFrame:
    """Join each selected symbol's trailing volatility (as of formation_date) onto the selection —
    required by the volatility-based weighting methods below."""
    available = features_df.filter(pl.col("Datetime") <= formation_date)
    last_date = available["Datetime"].max()
    cross_section = available.filter(pl.col("Datetime") == last_date).select(["Symbol", vol_col]).rename({vol_col: "Volatility"})
    return selected_df.join(cross_section, on="Symbol", how="left")


def cap_and_renormalize(df: pl.DataFrame, max_weight: float = MAX_INDIVIDUAL_WEIGHT, weight_col: str = "Weight") -> pl.DataFrame:
    """Iteratively cap any weight above max_weight and redistribute the excess pro-rata among the
    remaining uncapped positions, until every weight satisfies the cap and weights still sum to 1."""
    n = df.height
    if n * max_weight < 1.0:
        print(f"Warning: {max_weight:.0%} cap is infeasible for {n} position(s) (max possible sum "
              f"{n * max_weight:.0%}); skipping cap enforcement for this selection.")
        weights = dict(zip(df["Symbol"].to_list(), df[weight_col].to_list()))
        total = sum(weights.values())
        return df.with_columns(pl.Series(weight_col, [weights[s] / total for s in df["Symbol"].to_list()]))

    weights = dict(zip(df["Symbol"].to_list(), df[weight_col].to_list()))
    capped: set[str] = set()

    for _ in range(len(weights)):
        total = sum(weights.values())
        weights = {s: w / total for s, w in weights.items()}

        over_cap = {s for s, w in weights.items() if w > max_weight and s not in capped}
        if not over_cap:
            break

        for s in over_cap:
            weights[s] = max_weight
            capped.add(s)

        uncapped = [s for s in weights if s not in capped]
        if not uncapped:
            break
        remaining = 1 - sum(weights[s] for s in capped)
        uncapped_total = sum(weights[s] for s in uncapped)
        if uncapped_total > 0:
            for s in uncapped:
                weights[s] = weights[s] / uncapped_total * remaining

    return df.with_columns(pl.Series(weight_col, [weights[s] for s in df["Symbol"].to_list()]))


def equal_weight(selected_df: pl.DataFrame) -> pl.DataFrame:
    """Method A — 1/N weight for each selected symbol."""
    n = selected_df.height
    return cap_and_renormalize(selected_df.with_columns(pl.lit(1.0 / n).alias("Weight")))


def alpha_weight(selected_df: pl.DataFrame, score_col: str = "AlphaScore") -> pl.DataFrame:
    """Method B — w_i = Alpha_i / sum(Alpha)."""
    total = selected_df[score_col].sum()
    df = selected_df.with_columns((pl.col(score_col) / total).alias("Weight"))
    return cap_and_renormalize(df)


def alpha_volatility_weight(selected_df: pl.DataFrame, score_col: str = "AlphaScore", vol_col: str = "Volatility") -> pl.DataFrame:
    """Method C — w_i proportional to Alpha_i / Volatility_i."""
    selected_df = _drop_missing_volatility(selected_df, vol_col)
    raw = selected_df.with_columns((pl.col(score_col) / pl.col(vol_col)).alias("_raw_weight"))
    total = raw["_raw_weight"].sum()
    df = raw.with_columns((pl.col("_raw_weight") / total).alias("Weight")).drop("_raw_weight")
    return cap_and_renormalize(df)


def inverse_volatility_weight(selected_df: pl.DataFrame, vol_col: str = "Volatility") -> pl.DataFrame:
    """Method D — w_i proportional to 1 / Volatility_i."""
    selected_df = _drop_missing_volatility(selected_df, vol_col)
    raw = selected_df.with_columns((1.0 / pl.col(vol_col)).alias("_raw_weight"))
    total = raw["_raw_weight"].sum()
    df = raw.with_columns((pl.col("_raw_weight") / total).alias("Weight")).drop("_raw_weight")
    return cap_and_renormalize(df)


def risk_adjusted_alpha_weight(selected_df: pl.DataFrame, score_col: str = "AlphaScore", vol_col: str = "Volatility") -> pl.DataFrame:
    """Method E — equal blend of alpha/volatility and inverse-volatility weighting, capped at 20%."""
    df = alpha_volatility_weight(selected_df, score_col, vol_col).rename({"Weight": "_alpha_vol_weight"})
    df = inverse_volatility_weight(df, vol_col).rename({"Weight": "_inv_vol_weight"})
    df = df.with_columns(((pl.col("_alpha_vol_weight") + pl.col("_inv_vol_weight")) / 2).alias("Weight"))
    df = df.drop(["_alpha_vol_weight", "_inv_vol_weight"])
    return cap_and_renormalize(df)


WEIGHTING_METHODS = {
    "equal": equal_weight,
    "alpha": alpha_weight,
    "alpha_volatility": alpha_volatility_weight,
    "inverse_volatility": inverse_volatility_weight,
    "risk_adjusted_alpha": risk_adjusted_alpha_weight,
}
