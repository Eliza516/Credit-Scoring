"""Chia Train/Test (không K-fold). Tuỳ chọn OOT theo thời gian."""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_indices(df: pd.DataFrame, y: pd.Series, cfg: dict, date_col: str | None = None) -> dict[str, pd.Index]:
    """Chia chỉ số theo cấu hình. y: Series 0/1 (NaN đã loại bởi người gọi). Trả {'train','test'[,'oot']}."""
    sp = cfg["split"]
    seed = cfg.get("seed", 42)
    idx = y.dropna().index
    oot_idx = pd.Index([])
    if sp["oot"]["enabled"]:
        if not date_col:
            raise ValueError("OOT cần cột ngày.")
        d = pd.to_datetime(df.loc[idx, date_col])
        if sp["oot"].get("pct"):                               # OOT = phần trăm cuối theo thời gian
            order = d.sort_values(kind="stable").index
            n_oot = int(round(len(order) * sp["oot"]["pct"]))
            oot_idx = pd.Index(order[len(order) - n_oot:])
        else:
            cut = d.max() - pd.DateOffset(months=sp["oot"]["months"])
            oot_idx = idx[(d > cut).to_numpy()]
        idx = idx.difference(oot_idx)
    if sp["method"] == "time":
        if not date_col:
            raise ValueError("Chia theo thời gian cần cột ngày (open_date).")
        order = pd.to_datetime(df.loc[idx, date_col]).sort_values(kind="stable").index
        n_train = int(round(len(order) * sp["train_pct"]))
        train, test = order[:n_train], order[n_train:]
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(idx))
        n_train = int(round(len(idx) * sp["train_pct"]))
        train, test = idx[perm[:n_train]], idx[perm[n_train:]]
    out = {"train": pd.Index(train), "test": pd.Index(test)}
    if len(oot_idx):
        out["oot"] = oot_idx
    return out


def check_sizes(y: pd.Series, parts: dict[str, pd.Index], cfg: dict) -> pd.DataFrame:
    """Bảng cỡ mẫu + cảnh báo theo min_goods/min_bads (áp dụng cho train)."""
    sp = cfg["split"]
    rows = []
    for name, ix in parts.items():
        yy = y.loc[ix]
        bads, goods = int((yy == 1).sum()), int((yy == 0).sum())
        warn = ""
        if name == "train":
            if goods < sp["min_goods"]:
                warn += f"good {goods} < {sp['min_goods']}; "
            if bads < sp["min_bads"]:
                warn += f"bad {bads} < {sp['min_bads']}; "
        rows.append({"set": name, "rows": len(yy), "goods": goods, "bads": bads,
                     "bad_rate": round(bads / max(len(yy), 1), 4), "warning": warn.strip("; ")})
    return pd.DataFrame(rows)
