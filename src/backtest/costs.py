"""Transaction cost model. Applied once at formation since there is no rebalancing."""
from src.config import TRANSACTION_COST_RATE


def transaction_cost(position_value: float, rate: float = TRANSACTION_COST_RATE) -> float:
    return position_value * rate
