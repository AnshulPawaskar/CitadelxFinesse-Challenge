"""Structural guard confirming the backtest engine never rebalances a static portfolio."""
import polars as pl


def count_rebalances(trade_log: pl.DataFrame) -> int:
    """A rebalance would show up as more than one trade row for the same symbol. The static engine
    only ever writes one entry row per symbol, so this should always return 0 — call it as a
    regression guard in case rebalancing logic is ever accidentally introduced."""
    return int(trade_log.group_by("Symbol").len().filter(pl.col("len") > 1).height)


def validate_no_rebalancing(trade_log: pl.DataFrame) -> None:
    n = count_rebalances(trade_log)
    if n != 0:
        raise AssertionError(
            f"Rebalancing detected: {n} symbols have more than one trade entry, "
            "which violates the no-rebalance requirement for a static portfolio."
        )
