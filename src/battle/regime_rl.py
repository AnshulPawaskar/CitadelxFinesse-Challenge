"""Stage 6: regime diagnostics (why / when momentum works) and an RL exposure overlay (tabular Q-learning, gamma=0).

An exposure decision does not move the market, so the next state is independent of the action: the problem is a
contextual bandit. We still train it with epsilon-greedy tabular Q-learning replayed over history, strictly
walk-forward: the policy at decision date t uses only rewards realised before t."""
import numpy as np
import pandas as pd

from src.battle.context import build_context
from src.battle.data import BATTLE_DIR
from src.battle.engine import metrics, run_backtest

ACTIONS = np.array([0.0, 0.5, 1.0])
HOLD = 21


def market_state(mkt) -> pd.DataFrame:
    idx = pd.Series(mkt.bench_index, index=mkt.dates)
    ret = pd.Series(mkt.bench_ret, index=mkt.dates)
    vol = ret.rolling(60).std() * np.sqrt(252)
    return pd.DataFrame({
        "Trend": (idx > idx.rolling(200).mean()).astype(int),
        "HighVol": (vol > vol.expanding(120).median()).astype(int),
        "Ret63": idx.pct_change(63),
    }).dropna()


def regime_table(ctx, strat: pd.Series, bench: pd.Series) -> pd.DataFrame:
    st = market_state(ctx.mkt).reindex(strat.index).shift(1)  # state known before the day
    rows = []
    for name, mask in {
        "Uptrend (index > 200d MA)": st.Trend == 1, "Downtrend (index < 200d MA)": st.Trend == 0,
        "Low volatility": st.HighVol == 0, "High volatility": st.HighVol == 1,
        "Index 63d return > +10%": st.Ret63 > 0.10, "Index 63d return < -5%": st.Ret63 < -0.05,
    }.items():
        m = mask.fillna(False)
        s, b = strat[m], bench.loc[strat.index][m]
        rows.append({"Regime": name, "Days": int(m.sum()), "ShareOfDays": m.mean(),
                     "StrategyAnnReturn": s.mean() * 252, "BenchAnnReturn": b.mean() * 252,
                     "AnnExcess": (s.mean() - b.mean()) * 252, "StrategySharpe": s.mean() / s.std() * np.sqrt(252)})
    return pd.DataFrame(rows)


def train_q(states, rewards_by_action, epochs=60, alpha=0.1, eps=0.2, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((4, len(ACTIONS)))
    for _ in range(epochs):
        for s, ra in zip(states, rewards_by_action):
            a = rng.integers(len(ACTIONS)) if rng.random() < eps else int(Q[s].argmax())
            Q[s, a] += alpha * (ra[a] - Q[s, a])
    return Q


def rl_overlay(ctx, train_strat: pd.Series, target: pd.Series, cost=0.001, risk_aversion=3.0):
    mkt = ctx.mkt
    st = market_state(mkt)
    dates = train_strat.index
    pos = {d: i for i, d in enumerate(dates)}
    dec = [d for d in dates[::HOLD] if d in st.index]
    S = {d: int(st.loc[d, "Trend"] * 2 + st.loc[d, "HighVol"]) for d in dec}
    R = {}
    for d in dec:
        i = pos[d]
        if i + HOLD < len(dates):
            R[d] = (1 + train_strat.iloc[i + 1 : i + 1 + HOLD]).prod() - 1
    expo = pd.Series(1.0, index=target.index)
    policy_log, prev = [], 1.0
    first = pd.Timestamp("2023-02-01")
    for d in dec:
        if d < first:
            continue
        past = [x for x in R if x + pd.Timedelta(days=HOLD * 1.5) <= d]  # reward fully realised before d
        if len(past) < 18:
            continue
        rewards = [np.array([a * R[x] - 0.5 * risk_aversion * (a * R[x]) ** 2 for a in ACTIONS]) for x in past]
        Q = train_q([S[x] for x in past], rewards)
        a = ACTIONS[int(Q[S[d]].argmax())]
        policy_log.append({"Date": d.date(), "State": S[d], "Exposure": a})
        expo.loc[expo.index > d] = a
    expo = expo.shift(1).fillna(1.0)
    chg = expo.diff().abs().fillna(0)
    ret = target * expo - cost * chg
    return ret, expo, pd.DataFrame(policy_log)


if __name__ == "__main__":
    ctx = build_context()
    mkt = ctx.mkt
    bench = pd.Series(mkt.bench_ret, index=mkt.dates)
    final = pd.read_csv(BATTLE_DIR / "final_daily_returns.csv", index_col=0, parse_dates=True)["Strategy"]
    reg = regime_table(ctx, final, bench)
    reg.to_csv(BATTLE_DIR / "regime_analysis.csv", index=False)
    print(reg.round(3).to_string(index=False))

    start = int(np.argmax(ctx.elig.sum(1) >= 100)) + 1
    r = run_backtest(mkt, ctx.signal("composite_quantitative", "Q"), "Q", start, len(mkt.dates) - 1)
    train = pd.Series(r.returns, index=r.dates)
    ov, expo, pol = rl_overlay(ctx, train, final)
    cmp_rows = []
    sl = final.loc["2023-02-01":]
    for name, s in (("Core3 Q (no overlay)", sl), ("Core3 Q + RL exposure overlay", ov.loc["2023-02-01":]),
                    ("Benchmark", bench.loc[sl.index])):
        cmp_rows.append({"Strategy": name, **metrics(s.to_numpy(), s.index)})
    cmp = pd.DataFrame(cmp_rows)
    cmp["AvgExposure"] = [1.0, expo.loc["2023-02-01":].mean(), np.nan]
    cmp.to_csv(BATTLE_DIR / "rl_overlay.csv", index=False)
    pol.to_csv(BATTLE_DIR / "rl_policy_log.csv", index=False)
    print(cmp.round(3).to_string(index=False)); print(pol["Exposure"].value_counts())
