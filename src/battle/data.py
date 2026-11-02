"""Wide (date x symbol) market and feature matrices used by the walk-forward research framework."""
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

from src.config import RESULTS_DIR
from src.data.loader import load_universe_history, truncate_history
from src.data.universe import get_symbol_to_scrip_data
from src.features.feature_pipeline import compute_universe_features
from src.features.relative_strength import build_synthetic_benchmark

BATTLE_DIR = RESULTS_DIR / "battle"
BATTLE_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_FILE = BATTLE_DIR / "features_full.parquet"
DATA_END = datetime(2026, 10, 31)  # excludes the stray partial 2026-11-02 bar

# One representative per idea; near-duplicates (RSI7/21, Trend60/90/180 variants, EMAs) are dropped.
ML_FEATURES = [
    "R21", "R63", "Momentum3_1", "Momentum6_1", "Momentum9_1", "Momentum12_1",
    "RS63", "RS126", "RS252",
    "Trend60_TrendStrength", "Trend126_TrendStrength", "Trend126_R2",
    "CloseTo52WHigh", "CloseTo126DHigh", "CloseTo52WLow",
    "BreakoutDistance55D_ATR", "BreakoutDistance252D_ATR",
    "CloseToMA20", "CloseToMA50", "CloseToMA200", "MA20_MA50", "MA50_MA200",
    "RSI14", "ADX14", "DI_Diff", "MACD_Hist",
    "Vol20", "Vol60", "Vol126", "DownsideVolatility60", "ATR14_Close", "MaxDrawdown126D",
    "VolumeToMA20", "VolumeZScore", "OBV_Slope", "PriceVolumeCorr20", "VolumeConfirmedMomentum",
]
DERIVED = ["RiskAdjMomentum", "MAStructure"]


def build_feature_panel() -> pl.DataFrame:
    """Recompute the feature panel through DATA_END (the stored features.parquet stops at 2026-06-30)."""
    history = truncate_history(load_universe_history(get_symbol_to_scrip_data()), DATA_END)
    benchmark = build_synthetic_benchmark(history)
    features = compute_universe_features(history, benchmark)
    features.write_parquet(FEATURE_FILE)
    return features


def _wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="Datetime", columns="Symbol", values=col).sort_index()


class Market:
    """Aligned wide matrices. `obs` marks stock-days with a real bar; prices are forward-filled
    only for valuing existing holdings across data gaps (never for eligibility)."""

    def __init__(self, features: pl.DataFrame | None = None):
        if features is None:
            features = pl.read_parquet(FEATURE_FILE)
        features = features.filter(pl.col("Datetime") < DATA_END).with_columns(
            (pl.col("Open") * 0 + 1).alias("_obs"),
            pl.mean_horizontal(["MA50_MA200", "MA20_MA50"]).alias("MAStructure"),
            (pl.col("Momentum12_1") / pl.col("Vol126")).alias("RiskAdjMomentum"),
        )
        cols = ["Datetime", "Symbol", "Open", "Close", "Volume", "_obs"] + ML_FEATURES + DERIVED
        pdf = features.select(cols).to_pandas()
        self.close_raw = _wide(pdf, "Close")
        self.dates = self.close_raw.index
        self.symbols = list(self.close_raw.columns)
        self.obs = _wide(pdf, "_obs").reindex(self.dates).notna().to_numpy()
        self.close = self.close_raw.ffill().to_numpy()
        self.open = _wide(pdf, "Open").reindex(self.dates)[self.symbols].ffill().to_numpy()
        self.open = np.where(np.isnan(self.open), self.close, self.open)
        self.open, self.close = np.nan_to_num(self.open), np.nan_to_num(self.close)  # pre-listing -> 0, never held
        volume = _wide(pdf, "Volume").reindex(self.dates)[self.symbols]
        self.adv20 = (volume * self.close_raw.reindex(self.dates)[self.symbols]).rolling(20, min_periods=15).mean().to_numpy()
        self.feat = {f: _wide(pdf, f).reindex(self.dates)[self.symbols].to_numpy() for f in ML_FEATURES + DERIVED}
        # Benchmark: equal-weight daily rebalanced universe, using only stocks with bars on both days.
        ret = self.close_raw.pct_change(fill_method=None).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bench_ret = np.nanmean(np.where(self.obs, ret, np.nan), axis=1)
        self.bench_ret = np.nan_to_num(self.bench_ret)
        self.bench_index = np.cumprod(1 + self.bench_ret)
        self.next_ret_cache: dict[int, np.ndarray] = {}

    def forward_return(self, h: int) -> np.ndarray:
        """Return from next open (t+1) to open at t+1+h: what a trade signalled at close t actually earns."""
        if h not in self.next_ret_cache:
            o = self.open
            fwd = np.full_like(o, np.nan)
            fwd[: len(o) - 1 - h] = o[1 + h:] / o[1 : len(o) - h] - 1
            self.next_ret_cache[h] = np.where(self.obs, fwd, np.nan)
        return self.next_ret_cache[h]

    def eligible(self, min_adv: float = 2e7) -> np.ndarray:
        """A stock is investable on a day if it has a bar, a full feature set and enough traded value (Rs 2 cr/day)."""
        need = ["Momentum12_1", "RS126", "Trend126_TrendStrength", "Vol126", "CloseTo52WHigh"]
        ok = self.obs.copy()
        for f in need:
            ok &= ~np.isnan(self.feat[f])
        return ok & (self.adv20 >= min_adv)


if __name__ == "__main__":
    import time
    t = time.time()
    panel = build_feature_panel()
    print(panel.shape, panel["Datetime"].max(), f"{time.time() - t:.0f}s")
