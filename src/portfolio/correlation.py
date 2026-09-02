"""Correlation-aware selection (Method B): Top-30 by alpha -> greedy pairwise-correlation filter
-> final Top-N. An alternative to simply taking the raw Top-N by score (Method A, selector.py)."""
import polars as pl

from src.config import MAX_HOLDINGS

RETURN_WINDOW = 120
TOP_N_CANDIDATES = 30
CORRELATION_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, None]  # None = no correlation constraint


def build_trailing_returns(history: dict[str, pl.DataFrame], symbols: list[str], as_of_date, window: int = RETURN_WINDOW) -> pl.DataFrame:
    """Wide table of each symbol's daily returns over the trailing `window` days ending on/before as_of_date
    (symbols with insufficient history are simply absent from the result)."""
    frames = []
    for symbol in symbols:
        history_to_date = history[symbol].filter(pl.col("Datetime") <= as_of_date).sort("Datetime").tail(window + 1)
        if history_to_date.height < window + 1:
            continue
        returns = history_to_date.select(
            pl.col("Datetime"),
            (pl.col("Close") / pl.col("Close").shift(1) - 1).alias(symbol),
        ).drop_nulls()
        frames.append(returns)

    if not frames:
        return pl.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.join(frame, on="Datetime", how="inner")
    return merged.drop("Datetime")


def select_with_correlation_filter(
    signal_df: pl.DataFrame,
    history: dict[str, pl.DataFrame],
    as_of_date,
    n: int = MAX_HOLDINGS,
    top_n_candidates: int = TOP_N_CANDIDATES,
    correlation_threshold: float | None = 0.75,
    window: int = RETURN_WINDOW,
) -> pl.DataFrame:
    """Walk candidates in AlphaScore order, keeping a stock only if its trailing-return correlation
    with every already-selected stock stays below the threshold. Falls back to the next-highest-alpha
    candidates (ignoring correlation) if the filter can't fill all n slots."""
    candidates = signal_df.sort("AlphaScore", descending=True).head(top_n_candidates)
    symbols = candidates["Symbol"].to_list()

    returns = build_trailing_returns(history, symbols, as_of_date, window)
    if returns.is_empty():
        print("Warning: insufficient trailing return history for correlation filtering, falling back to Method A.")
        return candidates.head(n)

    available_symbols = returns.columns
    corr_matrix = returns.corr()

    def pairwise_corr(a: str, b: str) -> float:
        return corr_matrix[b][available_symbols.index(a)]

    selected_symbols: list[str] = []
    selected_rows = []
    for row in candidates.iter_rows(named=True):
        symbol = row["Symbol"]
        if symbol not in available_symbols:
            continue
        passes = (
            correlation_threshold is None
            or not selected_symbols
            or all(abs(pairwise_corr(symbol, s)) < correlation_threshold for s in selected_symbols)
        )
        if passes:
            selected_rows.append(row)
            selected_symbols.append(symbol)
        if len(selected_rows) == n:
            break

    if len(selected_rows) < n:
        print(f"Warning: only {len(selected_rows)} symbols passed the correlation filter "
              f"(threshold={correlation_threshold}); filling remaining slots with next-highest-alpha candidates.")
        chosen = set(selected_symbols)
        for row in candidates.iter_rows(named=True):
            if row["Symbol"] not in chosen:
                selected_rows.append(row)
                chosen.add(row["Symbol"])
            if len(selected_rows) == n:
                break

    return pl.DataFrame(selected_rows)
