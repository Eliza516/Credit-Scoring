"""Vintage analysis: bad rate tích lũy theo cohort tháng mở × MOB."""
from __future__ import annotations

import numpy as np
import pandas as pd


def vintage_table(df: pd.DataFrame, mapping: dict, cfg: dict, status: pd.Series | None = None) -> pd.DataFrame:
    """Bảng cohort × MOB: bad rate tích lũy trong các hợp đồng đã quan sát tới MOB đó.

    Cần cột open_date, first_bad_mob (rỗng nếu chưa bad), months_observed.
    status (tuỳ chọn): loại các dòng excluded khỏi mẫu số.
    """
    for k in ("open_date", "first_bad_mob", "months_observed"):
        if not mapping.get(k):
            raise ValueError(f"Vintage cần ánh xạ cột '{k}'.")
    v = cfg["vintage"]
    max_mob = v["max_mob"]
    d = pd.DataFrame({
        "cohort": pd.to_datetime(df[mapping["open_date"]]).dt.to_period("M").astype(str),
        "fb": df[mapping["first_bad_mob"]],
        "obs": df[mapping["months_observed"]],
    })
    if status is not None:
        d = d[status.reindex(df.index).ne("excluded").to_numpy()]
    rows = {}
    for cohort, g in d.groupby("cohort"):
        res = {}
        for m in range(1, max_mob + 1):
            seen = g[g["obs"] >= m]
            if len(seen) == 0:
                res[m] = np.nan
                continue
            res[m] = float(((seen["fb"].notna()) & (seen["fb"] <= m)).mean())
        res["size"] = len(g)
        rows[cohort] = res
    tab = pd.DataFrame(rows).T.sort_index()
    return tab


def suggest_window(tab: pd.DataFrame, cfg: dict) -> dict:
    """MOB đầu tiên mà mức tăng bad rate trung bình giữa 2 MOB liên tiếp < delta trong N MOB liền nhau."""
    v = cfg["vintage"]["maturity"]
    mob_cols = [c for c in tab.columns if c != "size"]
    w = tab["size"].to_numpy(dtype=float)
    curve = []
    for m in mob_cols:
        col = tab[m].to_numpy(dtype=float)
        ok = ~np.isnan(col)
        curve.append(float(np.average(col[ok], weights=w[ok])) if ok.any() else np.nan)
    curve = pd.Series(curve, index=mob_cols)
    delta_pp = curve.diff() * 100
    n, thr = v["consecutive_mob"], v["delta_threshold_pp"]
    suggested = None
    vals = delta_pp.to_numpy()
    for i in range(1, len(vals) - n + 1):
        win = vals[i:i + n]
        if not np.isnan(win).any() and (win < thr).all():
            suggested = int(mob_cols[i - 1])
            break
    lo, hi = cfg["target"]["performance_window_range"]
    return {"curve": curve, "delta_pp": delta_pp, "suggested_mob": suggested,
            "within_range": suggested is not None and lo <= suggested <= hi,
            "message": (f"Gợi ý performance window = {suggested} tháng (bad rate tích lũy ngừng tăng > {thr}pp trong {n} MOB liên tiếp)."
                        if suggested else "Chưa tìm được MOB 'chín' theo ngưỡng; xem đồ thị và chọn thủ công.")}


def cohort_anomalies(tab: pd.DataFrame, cfg: dict, at_mob: int | None = None) -> pd.DataFrame:
    """Cohort có bad rate lệch > z-score ngưỡng tại MOB chỉ định (mặc định MOB lớn nhất có đủ cohort)."""
    c = cfg["vintage"]["cohort_anomaly"]
    mob_cols = [m for m in tab.columns if m != "size"]
    if at_mob is None:
        ok = [m for m in mob_cols if tab[m].notna().sum() >= 6]
        at_mob = ok[-1] if ok else mob_cols[-1]
    s = tab[at_mob]
    big = tab["size"] >= c["min_cohort_size"]
    ref = s[big].dropna()
    mu, sd = ref.mean(), ref.std(ddof=0) or np.nan
    z = (s - mu) / sd
    out = pd.DataFrame({"cohort": tab.index, "size": tab["size"].to_numpy(), "bad_rate": s.to_numpy(), "z": z.to_numpy()})
    out["flag"] = (out["z"].abs() > c["zscore_threshold"]) & big.to_numpy()
    out["at_mob"] = at_mob
    return out
