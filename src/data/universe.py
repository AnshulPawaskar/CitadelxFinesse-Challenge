"""Build the fixed 2026 stock universe and map each symbol to its local ScripData filename."""
from polars import DataFrame, read_csv, read_parquet, concat, col, lit

from src.config import STOCKS_DIR, SCRIPTBOOK_DIR, INDICES


def load_index_constituents(indices: list[str] = INDICES) -> DataFrame:
    """Concatenate the index constituent CSVs, tagging each row with its source index."""
    df = None
    for name in indices:
        chunk = read_csv(STOCKS_DIR / f"{name}.csv").with_columns(lit(name).alias("Index"))
        df = chunk if df is None else concat([df, chunk], how="vertical")

    dup_symbols = df.filter(df["Symbol"].is_duplicated())["Symbol"].unique().to_list()
    if dup_symbols:
        print(f"Warning: duplicate symbols across indices, keeping first occurrence: {dup_symbols}")

    return df.unique(subset=["Symbol"], keep="first").sort("Symbol")


def load_script_mapping() -> DataFrame:
    """Load the locally cached 5Paisa script master, restricted to NSE cash-equity rows."""
    scripts = read_parquet(SCRIPTBOOK_DIR / "script_details.parquet")
    return scripts.filter(
        (col("Exch") == "N") & (col("ExchType") == "C") & (col("Series") == "EQ")
    ).select(["Name", "Exch", "ExchType", "ScripCode", "ScripData"])


def build_universe() -> DataFrame:
    """Join the index constituents to their ScripData mapping (case/whitespace-normalized)."""
    stocks = load_index_constituents()
    scripts = load_script_mapping()

    stocks = stocks.with_columns(col("Symbol").str.strip_chars().str.to_uppercase().alias("_join_key"))
    scripts = scripts.with_columns(col("Name").str.strip_chars().str.to_uppercase().alias("_join_key"))
    df = stocks.join(scripts, on="_join_key", how="left").drop("_join_key")

    missing = df.filter(col("ScripCode").is_null())
    if missing.shape[0] > 0:
        print(f"Warning: {missing.shape[0]} symbols had no matching ScripCode: {missing['Symbol'].to_list()}")

    dup_counts = df.group_by("Symbol").len().filter(col("len") > 1)
    if dup_counts.shape[0] > 0:
        print(f"Warning: {dup_counts.shape[0]} symbols matched multiple ScripCodes (ambiguous, first is kept): {dup_counts['Symbol'].to_list()}")

    return df.unique(subset=["Symbol"], keep="first").sort("Symbol")


def get_universe_symbols() -> list[str]:
    """Return the deduped list of tradable symbols (e.g. 'RELIANCE')."""
    return build_universe()["Symbol"].to_list()


def get_symbol_to_scrip_data() -> dict[str, str]:
    """Return a mapping of {Symbol: ScripData} (e.g. {'RELIANCE': 'RELIANCE_EQ'}) used to locate parquet files."""
    df = build_universe().filter(col("ScripData").is_not_null())
    return dict(zip(df["Symbol"].to_list(), df["ScripData"].to_list()))
