"""Tính toán cho dashboard từng bước (bảng số liệu thuần pandas; phần vẽ nằm ở app/dash.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_LABEL = "Thiếu"
OTHER_LABEL = "Khác"


def wilson(bad: pd.Series | np.ndarray, n: pd.Series | np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Khoảng tin cậy Wilson 95% cho tỷ lệ bad."""
    bad, n = np.asarray(bad, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, bad / n, np.nan)
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


# ----------------------------------------------------------------- tổng quan
def overview(df: pd.DataFrame, id_col: str | None = None) -> dict:
    kinds = {"số": 0, "chữ": 0, "ngày": 0, "khác": 0}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            kinds["ngày"] += 1
        elif pd.api.types.is_bool_dtype(s):
            kinds["khác"] += 1
        elif pd.api.types.is_numeric_dtype(s):
            kinds["số"] += 1
        elif pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            kinds["chữ"] += 1
        else:
            kinds["khác"] += 1
    out = {"rows": len(df), "cols": df.shape[1], "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
           "missing_cells_pct": float(df.isna().to_numpy().mean()) if df.size else 0.0, "kinds": kinds}
    if id_col and id_col in df.columns:
        out["duplicate_ids"] = int(df[id_col].duplicated().sum())
    return out


def missing_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or list(df.columns)
    t = pd.DataFrame({"column": cols, "missing_pct": [float(df[c].isna().mean()) for c in cols],
                      "n_unique": [int(df[c].nunique(dropna=True)) for c in cols], "dtype": [str(df[c].dtype) for c in cols]})
    return t.sort_values("missing_pct", ascending=False).reset_index(drop=True)


def value_counts_table(s: pd.Series, top: int = 20) -> pd.DataFrame:
    vc = s.astype("object").where(s.notna(), MISSING_LABEL).value_counts()
    t = pd.DataFrame({"value": vc.index.astype(str), "count": vc.to_numpy()})
    t["share"] = t["count"] / max(t["count"].sum(), 1)
    return t.head(top)


# ------------------------------------------------------------------- thời gian
def to_period(d: pd.Series, freq: str = "M") -> pd.Series:
    return pd.to_datetime(d, errors="coerce").dt.to_period(freq).dt.to_timestamp()


def volume_by_period(df: pd.DataFrame, date_col: str, y: pd.Series | None = None, freq: str = "M") -> pd.DataFrame:
    """Theo kỳ: số khoản vay, (nếu có y) số có nhãn, số bad, bad rate."""
    per = to_period(df[date_col], freq)
    t = pd.DataFrame({"period": per})
    if y is not None:
        yy = y.reindex(df.index)
        t["labeled"] = yy.notna()
        t["bad"] = (yy == 1)
        g = t.groupby("period").agg(n=("labeled", "size"), labeled=("labeled", "sum"), bads=("bad", "sum")).reset_index()
        g["bad_rate"] = np.where(g["labeled"] > 0, g["bads"] / g["labeled"].where(g["labeled"] > 0, 1), np.nan)
        return g
    g = t.groupby("period").size().rename("n").reset_index()
    return g


# --------------------------------------------------------------- bad rate theo biến
def group_series(s: pd.Series, n_bins: int = 10, max_categories: int = 15) -> pd.Series:
    """Chia biến thành nhóm hiển thị: biến số ít giá trị → từng giá trị; biến số nhiều giá trị → phân vị; biến chữ → top k + 'Khác'.
    Giá trị thiếu thành nhóm 'Thiếu'. Trả Categorical có thứ tự."""
    miss = s.isna()
    if pd.api.types.is_datetime64_any_dtype(s):
        s = s.dt.to_period("Y").astype(str).where(~miss)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        nun = s.nunique(dropna=True)
        if nun <= max(n_bins, 2):
            labels = s.astype("object").map(lambda v: f"{v:g}" if pd.notna(v) else None)
            order = [f"{v:g}" for v in sorted(s.dropna().unique())]
        else:
            q = pd.qcut(s, n_bins, duplicates="drop")
            labels = q.astype(object).map(lambda iv: f"({iv.left:.4g}, {iv.right:.4g}]" if pd.notna(iv) else None)
            order = [f"({iv.left:.4g}, {iv.right:.4g}]" for iv in q.cat.categories]
    else:
        vc = s.astype(object).where(~miss).value_counts()
        keep = list(vc.index[:max_categories])
        labels = s.astype(object).where(s.astype(object).isin(keep), OTHER_LABEL).where(~miss)
        order = [str(k) for k in keep] + ([OTHER_LABEL] if len(vc) > max_categories else [])
        labels = labels.map(lambda v: str(v) if pd.notna(v) else None)
    out = pd.Series(labels, index=s.index, dtype=object).where(~miss, MISSING_LABEL)
    cats = order + ([MISSING_LABEL] if miss.any() else [])
    return pd.Series(pd.Categorical(out, categories=cats, ordered=True), index=s.index)


def badrate_by_feature(df: pd.DataFrame, y: pd.Series, feature: str, n_bins: int = 10, max_categories: int = 15,
                       min_count: int = 30) -> pd.DataFrame:
    """Bảng theo nhóm của biến: n, tỷ trọng, bads, bad_rate, lift so với trung bình, khoảng tin cậy, cờ nhóm quá nhỏ."""
    yy = y.reindex(df.index)
    lab = yy.notna()
    g = group_series(df.loc[lab, feature], n_bins, max_categories)
    t = pd.DataFrame({"group": g, "y": yy[lab].to_numpy()})
    agg = t.groupby("group", observed=True)["y"].agg(["size", "sum"]).rename(columns={"size": "n", "sum": "bads"}).reset_index()
    total = max(int(agg["n"].sum()), 1)
    base = float(yy[lab].mean()) if lab.any() else np.nan
    agg["share"] = agg["n"] / total
    agg["bad_rate"] = agg["bads"] / agg["n"]
    agg["lift"] = agg["bad_rate"] / base if base and base > 0 else np.nan
    lo, hi = wilson(agg["bads"], agg["n"])
    agg["ci_low"], agg["ci_high"] = lo, hi
    agg["small"] = agg["n"] < min_count
    agg["group"] = agg["group"].astype(str)
    agg.attrs["base_rate"] = base
    return agg


def badrate_by_feature_time(df: pd.DataFrame, y: pd.Series, feature: str, date_col: str, freq: str = "Y", n_bins: int = 5,
                            max_categories: int = 6) -> pd.DataFrame:
    """Bad rate theo (kỳ, nhóm) – xem một nhóm có ổn định theo thời gian hay không."""
    yy = y.reindex(df.index)
    lab = yy.notna()
    g = group_series(df.loc[lab, feature], n_bins, max_categories)
    t = pd.DataFrame({"period": to_period(df.loc[lab, date_col], freq).to_numpy(), "group": g.astype(str).to_numpy(), "y": yy[lab].to_numpy()})
    out = t.groupby(["period", "group"])["y"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "bad_rate"}).reset_index()
    return out.dropna(subset=["period"])


# ------------------------------------------------------------------ phân phối
def distribution(s: pd.Series, bins: int = 30, p_lo: float = 1, p_hi: float = 99) -> dict:
    """Histogram + phân vị + tỷ lệ ngoại lai (ngoài [Q1-1.5·IQR, Q3+1.5·IQR] và ngoài [P_lo, P_hi])."""
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return {"hist": pd.DataFrame(columns=["left", "right", "count"]), "quantiles": {}, "outlier_iqr_pct": np.nan, "outlier_pct_pct": np.nan}
    cnt, edges = np.histogram(v, bins=bins)
    q = v.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    iqr = q[0.75] - q[0.25]
    lo, hi = np.percentile(v, [p_lo, p_hi])
    return {"hist": pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": cnt}),
            "quantiles": {f"P{int(k*100)}": float(x) for k, x in q.items()},
            "outlier_iqr_pct": float(((v < q[0.25] - 1.5 * iqr) | (v > q[0.75] + 1.5 * iqr)).mean()),
            "outlier_pct_pct": float(((v < lo) | (v > hi)).mean())}


def numeric_by_class(df: pd.DataFrame, y: pd.Series, feature: str, bins: int = 30) -> pd.DataFrame:
    """Histogram chuẩn hoá của biến số cho good và bad (so sánh hai phân phối)."""
    yy = y.reindex(df.index)
    v = pd.to_numeric(df[feature], errors="coerce")
    ok = yy.notna() & v.notna()
    if not ok.any():
        return pd.DataFrame(columns=["left", "right", "class", "density"])
    lo, hi = np.percentile(v[ok], [0.5, 99.5])
    edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
    rows = []
    for cls, name in ((0, "good"), (1, "bad")):
        c, _ = np.histogram(v[ok & (yy == cls)].clip(lo, hi), bins=edges)
        d = c / max(c.sum(), 1)
        rows += [{"left": edges[i], "right": edges[i + 1], "class": name, "density": d[i]} for i in range(bins)]
    return pd.DataFrame(rows)


def corr_matrix(df: pd.DataFrame, columns: list[str], method: str = "spearman", max_rows: int = 100_000) -> pd.DataFrame:
    d = df[columns]
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=0)
    return d.corr(method=method)


def split_summary(parts: dict[str, pd.Index], df: pd.DataFrame, date_col: str | None, y: pd.Series) -> pd.DataFrame:
    """Mỗi tập: số dòng, khoảng thời gian, bad rate."""
    rows = []
    for k, ix in parts.items():
        yy = y.loc[ix]
        row = {"set": k, "rows": len(ix), "bads": int((yy == 1).sum()), "bad_rate": float((yy == 1).mean()) if len(ix) else np.nan}
        if date_col:
            d = pd.to_datetime(df.loc[ix, date_col], errors="coerce")
            row["from"], row["to"] = d.min(), d.max()
        rows.append(row)
    return pd.DataFrame(rows)
