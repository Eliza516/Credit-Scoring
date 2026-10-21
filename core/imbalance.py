"""Xử lý mất cân bằng – quyết định của người dùng, chỉ áp dụng cho TRAIN."""
from __future__ import annotations

import numpy as np
import pandas as pd


def resample(X: pd.DataFrame, y: pd.Series, method: str, target_bad_rate: float | None = None, seed: int = 42):
    """Trả (X', y', sample_weight|None). method: none | oversample_random | oversample_smote | undersample_random | sample_weight."""
    if method in (None, "none"):
        return X, y, None
    rng = np.random.default_rng(seed)
    pos = y[y == 1].index
    neg = y[y == 0].index
    if method == "sample_weight":
        rate = target_bad_rate or 0.5
        w_pos = rate / (len(pos) / len(y))
        w_neg = (1 - rate) / (len(neg) / len(y))
        w = pd.Series(np.where(y == 1, w_pos, w_neg), index=y.index)
        return X, y, w
    r = target_bad_rate or 0.5
    if method == "oversample_random":
        n_pos_new = int(round(r / (1 - r) * len(neg)))
        extra = rng.choice(pos.to_numpy(), max(n_pos_new - len(pos), 0), replace=True)
        idx = np.concatenate([X.index.to_numpy(), extra])
        Xn = X.loc[idx].reset_index(drop=True)
        yn = y.loc[idx].reset_index(drop=True)
        return Xn, yn, None
    if method == "undersample_random":
        n_neg_new = int(round((1 - r) / r * len(pos)))
        keep_neg = rng.choice(neg.to_numpy(), min(n_neg_new, len(neg)), replace=False)
        idx = np.concatenate([pos.to_numpy(), keep_neg])
        idx = rng.permutation(idx)
        return X.loc[idx], y.loc[idx], None
    if method == "oversample_smote":
        from imblearn.over_sampling import SMOTE
        strat = min(max(r / (1 - r), len(pos) / len(neg) + 1e-6), 1.0)
        sm = SMOTE(sampling_strategy=strat, random_state=seed, k_neighbors=min(5, len(pos) - 1))
        Xr, yr = sm.fit_resample(X.fillna(0), y)
        return pd.DataFrame(Xr, columns=X.columns), pd.Series(yr, name=y.name), None
    raise ValueError(f"Phương án không hợp lệ: {method}")


def prior_offset(bad_rate_dev: float, bad_rate_true: float) -> float:
    """Offset hiệu chỉnh intercept: ln[(bad_true/(1-bad_true)) / (bad_dev/(1-bad_dev))] (đưa PD về bad rate thật)."""
    odds_true = bad_rate_true / (1 - bad_rate_true)
    odds_dev = bad_rate_dev / (1 - bad_rate_dev)
    return float(np.log(odds_true / odds_dev))
