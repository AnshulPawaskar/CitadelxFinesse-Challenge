"""Signal generators: hand-set factor composites and walk-forward ML/DL rank predictors.

Every model outputs score[T, N]. ML scores at date t come from a model trained only on rows whose
forward-return label was fully realised before the test block began (purged expanding window)."""
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from src.battle.data import DERIVED, ML_FEATURES, BATTLE_DIR
from src.signals.composite import MODEL_DEFINITIONS

FEATURES = ML_FEATURES + DERIVED
torch.set_num_threads(4)


def cs_percentile(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile in [-0.5, 0.5] per row over masked entries; NaN elsewhere."""
    df = pd.DataFrame(np.where(mask, a, np.nan))
    r = df.rank(axis=1, method="average")
    n = df.notna().sum(axis=1).to_numpy()[:, None]
    return ((r.to_numpy() - 1) / np.maximum(n - 1, 1) - 0.5).astype(np.float32)


def build_X(mkt, elig: np.ndarray) -> np.ndarray:
    return np.stack([cs_percentile(mkt.feat[f], elig & ~np.isnan(mkt.feat[f])) for f in FEATURES], axis=2)


def build_y(mkt, elig: np.ndarray, h: int) -> np.ndarray:
    return cs_percentile(mkt.forward_return(h), elig)


def handset_score(X: np.ndarray, model_name: str) -> np.ndarray:
    """Weighted mean of factor percentiles. Negative weights reverse a factor (e.g. low volatility is good)."""
    w = CUSTOM_DEFINITIONS.get(model_name) or MODEL_DEFINITIONS[model_name]
    idx = {f: i for i, f in enumerate(FEATURES)}
    num = sum(X[:, :, idx[f]] * wt for f, wt in w.items())
    return num / sum(abs(v) for v in w.values())


# Pre-specified before any validation results were seen: momentum with a classic low-risk tilt.
CUSTOM_DEFINITIONS = {
    "momentum_lowvol": {"Momentum12_1": 0.30, "Momentum6_1": 0.20, "RS126": 0.20,
                        "Vol126": -0.20, "MaxDrawdown126D": 0.10},
}


class MLP:
    def __init__(self, n_in: int, seed: int):
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(32, 1))

    def fit(self, X, y, epochs=12):
        Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        opt = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-2)
        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 1024):
                b = perm[i : i + 1024]
                opt.zero_grad()
                torch.nn.functional.huber_loss(self.net(Xt[b]).squeeze(1), yt[b], delta=0.3).backward()
                opt.step()
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()


def _fit_predict(kind: str, Xtr, ytr, Xte):
    if kind == "ridge":
        m = Ridge(alpha=200.0).fit(Xtr, ytr)
        return m.predict(Xte), m.coef_
    if kind == "lgbm":
        m = LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=300,
                          subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
                          n_jobs=8, random_state=0, verbose=-1).fit(Xtr, ytr)
        return m.predict(Xte), m.feature_importances_
    if kind == "mlp":
        preds = [MLP(Xtr.shape[1], seed).fit(Xtr, ytr).predict(Xte) for seed in range(3)]
        return np.mean(preds, axis=0), None
    raise ValueError(kind)


def walk_forward_scores(X, y, kind: str, h: int, first_test: int, retrain_every: int = 63,
                        step: int = 3, min_rows: int = 5000):
    """Expanding-window, purged walk-forward. Returns (score[T,N], [(date_idx, importance_or_coef)])."""
    T, N, F = X.shape
    score = np.full((T, N), np.nan, dtype=np.float32)
    log = []
    for b in range(first_test, T, retrain_every):
        last_train = b - 2 - h  # label of day t uses the open at t+1+h, which must precede the test block
        tr = np.arange(0, last_train + 1, step)
        Xtr, ytr = X[tr].reshape(-1, F), y[tr].reshape(-1)
        keep = ~np.isnan(ytr) & ~np.isnan(Xtr).any(axis=1)
        if keep.sum() < min_rows:
            continue
        te = np.arange(b, min(b + retrain_every, T))
        Xte = X[te].reshape(-1, F)
        ok = ~np.isnan(Xte).any(axis=1)
        pred = np.full(len(Xte), np.nan, dtype=np.float32)
        pred[ok], info = _fit_predict(kind, Xtr[keep], ytr[keep], Xte[ok])
        score[te] = pred.reshape(len(te), N)
        log.append((b, info))
    return score, log


def ensemble(scores: list[np.ndarray], elig: np.ndarray) -> np.ndarray:
    """Average cross-sectional percentiles of several scores (rank averaging)."""
    return np.nanmean([cs_percentile(s, elig & ~np.isnan(s)) for s in scores], axis=0)


def cached_scores(mkt, X, y, elig, kind: str, h: int, first_test: int):
    path = BATTLE_DIR / f"score_{kind}_h{h}.npy"
    if path.exists():
        return np.load(path)
    score, log = walk_forward_scores(X, y, kind, h, first_test)
    np.save(path, score)
    if kind in ("ridge", "lgbm"):
        rows = [dict(zip(FEATURES, info), TestStartDate=str(mkt.dates[b].date())) for b, info in log]
        pd.DataFrame(rows).to_csv(BATTLE_DIR / f"{kind}_h{h}_feature_weights.csv", index=False)
    return score
