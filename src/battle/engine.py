"""Vectorised rebalancing backtester: signals at close t -> trades at open t+1, costs on actual turnover."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def signal_days(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Index of the last trading day of each period (the day the signal is computed)."""
    if freq == "N":
        return np.array([], dtype=int)
    if freq == "D":
        return np.arange(len(dates))
    key = {
        "W": dates.to_period("W-FRI"),
        "M": dates.to_period("M"),
        "Q": dates.to_period("Q"),
        "S": pd.Index([(d.year, 1 if d.month <= 6 else 2) for d in dates]),
    }[freq]
    s = pd.Series(np.arange(len(dates)), index=key)
    return s.groupby(level=0).last().sort_values().to_numpy()


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    nav: np.ndarray
    turnover: float  # annualised one-way turnover, fraction of NAV
    n_rebalances: int
    holdings: list = field(default_factory=list)  # (execution date, list of symbols)

    @property
    def returns(self) -> np.ndarray:
        return np.diff(self.nav, prepend=1.0) / np.concatenate([[1.0], self.nav[:-1]])


def run_backtest(mkt, score: np.ndarray, freq: str, start: int, end: int, n: int = 10,
                 buffer: int | None = None, cost: float = 0.001, exposure: np.ndarray | None = None,
                 eligible: np.ndarray | None = None, record: bool = False,
                 weighting: str = "equal") -> BacktestResult:
    """score[T,N]: higher is better, NaN = not investable. Start/end are inclusive date indices of the
    NAV window; the first trade is at the open of `start`, signalled at start-1."""
    elig = mkt.eligible() if eligible is None else eligible
    sig = set(signal_days(mkt.dates, freq).tolist())
    nstk = score.shape[1]
    shares = np.zeros(nstk)
    cash = 1.0
    nav = np.empty(end - start + 1)
    traded_total = 0.0
    n_reb = 0
    holdings = []
    for d in range(start, end + 1):
        if (d - 1) in sig or d == start:
            s = np.where(elig[d - 1], score[d - 1], np.nan)
            ok = ~np.isnan(s)
            if ok.sum() >= n:
                order = np.argsort(-np.where(ok, s, -np.inf))
                rank = np.empty(nstk, dtype=int)
                rank[order] = np.arange(nstk)
                held = np.flatnonzero(shares > 0)
                if buffer is not None and len(held):
                    keep = [i for i in held if ok[i] and rank[i] < buffer]
                    keep = sorted(keep, key=lambda i: rank[i])[:n]
                else:
                    keep = []
                chosen = list(keep) + [i for i in order[: n + len(keep)] if i not in keep][: n - len(keep)]
                expo = 1.0 if exposure is None else float(exposure[d - 1])
                px = mkt.open[d]
                wt = np.full(len(chosen), 1.0 / n)
                if weighting == "invvol":
                    inv = 1.0 / mkt.feat["Vol126"][d - 1, chosen]
                    wt = inv / inv.sum()
                    for _ in range(5):  # cap single names at 20% and redistribute
                        wt = np.minimum(wt, 0.20)
                        wt = wt / wt.sum()
                val_open = cash + shares @ px
                target = np.zeros(nstk)
                target[chosen] = val_open * expo * wt / px[chosen]
                c = cost * np.abs(target - shares) @ px
                scale = (val_open - c) / val_open
                target *= scale
                traded_total += np.abs(target - shares) @ px / val_open
                cash = val_open - c - target @ px
                shares = target
                n_reb += 1
                if record:
                    holdings.append((mkt.dates[d], [mkt.symbols[i] for i in chosen]))
        nav[d - start] = cash + shares @ mkt.close[d]
    years = (mkt.dates[end] - mkt.dates[start]).days / 365.25
    return BacktestResult(mkt.dates[start : end + 1], nav, traded_total / 2 / max(years, 1e-9), n_reb, holdings)


def metrics(returns: np.ndarray, dates: pd.DatetimeIndex | None = None) -> dict:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    nav = np.cumprod(1 + r)
    years = (dates[-1] - dates[0]).days / 365.25 if dates is not None and len(dates) > 1 else len(r) / TRADING_DAYS
    total = nav[-1] - 1
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    down = r[r < 0]
    dd = nav / np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:] - 1
    cagr = nav[-1] ** (1 / years) - 1 if years > 0 and nav[-1] > 0 else np.nan
    return {
        "TotalReturn": total, "CAGR": cagr, "Vol": sd * np.sqrt(TRADING_DAYS),
        "Sharpe": r.mean() / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else np.nan,
        "Sortino": r.mean() / down.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(down) > 1 else np.nan,
        "MaxDD": dd.min(), "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
    }
