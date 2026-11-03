"""Shared research context: market data, features, all signals (hand-set + walk-forward ML) and the split protocol.

Protocol (fixed before any results were inspected):
  * Walk-forward OOS scoring starts 2022-01-03 (ML needs ~1y of purged history first).
  * VALIDATION  2022-01-03 .. 2024-06-28  -> the only window used to choose signal / frequency / design.
  * HOLDOUT     2024-07-01 .. 2026-10-30  -> evaluated after the choice is frozen; ML keeps retraining
                                            walk-forward, so it never sees future labels.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.battle.data import Market
from src.battle.models import (FEATURES, build_X, build_y, cached_scores, cs_percentile, ensemble,
                               handset_score)

HORIZON = {"D": 5, "W": 5, "M": 21, "Q": 63, "S": 63, "N": 63}
FREQ_NAME = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly", "S": "Half-yearly"}
HANDSET = ["composite_quantitative", "momentum_12_1", "momentum_6m", "relative_strength",
           "risk_adjusted_momentum", "technical_composite", "trend_quality", "momentum_lowvol"]
ML = ["ridge", "lgbm", "mlp", "ml_ensemble"]
SIGNALS = HANDSET + ML + ["hybrid"]
# Top 3 signals by mean VALIDATION Sharpe across the five frequencies (rule frozen before holdout was inspected).
CORE3 = ["momentum_6m", "risk_adjusted_momentum", "momentum_12_1"]


def idx_on_or_after(dates: pd.DatetimeIndex, day: str) -> int:
    return int(np.searchsorted(dates, pd.Timestamp(day)))


@dataclass
class Context:
    mkt: Market
    elig: np.ndarray
    X: np.ndarray
    first_test: int
    val: tuple[int, int]
    hold: tuple[int, int]
    _cache: dict = field(default_factory=dict)

    @property
    def oos(self) -> tuple[int, int]:
        return self.first_test, len(self.mkt.dates) - 1

    def signal(self, name: str, freq: str) -> np.ndarray:
        h = HORIZON[freq]
        key = (name, h if (name in ML or name in ("hybrid", "core3")) else 0)
        if key in self._cache:
            return self._cache[key]
        if name in HANDSET:
            s = handset_score(self.X, name)
        elif name in ("ridge", "lgbm", "mlp"):
            y = build_y(self.mkt, self.elig, h)
            s = cached_scores(self.mkt, self.X, y, self.elig, name, h, self.first_test)
        elif name == "ml_ensemble":
            s = ensemble([self.signal(k, freq) for k in ("ridge", "lgbm", "mlp")], self.elig)
        elif name == "core3":
            s = ensemble([self.signal(k, freq) for k in CORE3], self.elig)
        elif name == "hybrid":
            s = ensemble([self.signal("composite_quantitative", freq), self.signal("ml_ensemble", freq)], self.elig)
        else:
            raise ValueError(name)
        self._cache[key] = s
        return s


def combo(ctx, names, freq):
    """Rank-average of several signals at the horizon belonging to `freq`."""
    key = ("combo", tuple(sorted(names)), HORIZON[freq])
    if key not in ctx._cache:
        ctx._cache[key] = ensemble([ctx.signal(n, freq) for n in names], ctx.elig)
    return ctx._cache[key]


def build_context() -> Context:
    mkt = Market()
    elig = mkt.eligible()
    X = build_X(mkt, elig)
    d = mkt.dates
    return Context(
        mkt, elig, X,
        first_test=idx_on_or_after(d, "2022-01-03"),
        val=(idx_on_or_after(d, "2022-01-03"), idx_on_or_after(d, "2024-06-29") - 1),
        hold=(idx_on_or_after(d, "2024-07-01"), len(d) - 1),
    )
