"""Vintage analysis: bad rate tích lũy (ever-bad) theo cohort tháng mở × MOB."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cohort_table(open_date: pd.Series, first_bad_mob: pd.Series, months_observed: pd.Series, max_mob: int,
                 min_cell: int = 1, weight: pd.Series | None = None) -> pd.DataFrame:
    """Bảng cohort × MOB (+ cột 'size').

    Ô (cohort, m) = tỷ lệ khoản vay ĐÃ QUAN SÁT tới MOB m mà đã chạm ngưỡng bad ở MOB ≤ m (theo số khoản, hoặc theo `weight`).
    Ô có ít hơn `min_cell` khoản quan sát → NaN (che đuôi nhiễu). Số khoản quan sát mỗi ô: xem `n_obs(tab)`.
    """
    d = pd.DataFrame({
        "c": pd.to_datetime(open_date).dt.to_period("M").astype(str).to_numpy(),
        "fb": pd.to_numeric(first_bad_mob, errors="coerce").to_numpy(),
        "obs": pd.to_numeric(months_observed, errors="coerce").fillna(0).to_numpy(),
        "w": (weight.fillna(0).to_numpy(dtype=float) if weight is not None else 1.0),
    })
    d = d[d["c"] != "NaT"]
    cols, nobs = {}, {}
    for m in range(1, max_mob + 1):
        seen = d["obs"] >= m
        bad = seen & d["fb"].notna() & (d["fb"] <= m)
        n = seen.groupby(d["c"]).sum()
        den = d["w"].where(seen, 0.0).groupby(d["c"]).sum()
        num = d["w"].where(bad, 0.0).groupby(d["c"]).sum()
        rate = (num / den.where(den > 0)).where(n >= max(min_cell, 1))
        cols[m], nobs[m] = rate, n
    tab = pd.DataFrame(cols).sort_index()
    tab["size"] = d.groupby("c").size().reindex(tab.index)
    n = pd.DataFrame(nobs).sort_index().reindex(tab.index)
    # dict thuần (không lưu DataFrame trong attrs: pandas so sánh attrs khi concat/astype và sẽ lỗi)
    tab.attrs["n_obs"] = {"index": list(n.index), "columns": list(n.columns), "values": n.to_numpy().tolist()}
    return tab


def n_obs(tab: pd.DataFrame) -> pd.DataFrame | None:
    """Số khoản quan sát ở mỗi ô cohort × MOB (None nếu bảng không do cohort_table tạo)."""
    a = tab.attrs.get("n_obs")
    return None if a is None else pd.DataFrame(a["values"], index=a["index"], columns=a["columns"])


def vintage_from_loans(loans: pd.DataFrame, cfg: dict, exclude: pd.Series | None = None, weight: bool = False) -> pd.DataFrame:
    """loans: bảng theo khoản vay có open_date, first_bad_mob, months_observed (+ weight). exclude: True = loại khỏi vintage."""
    v = cfg["vintage"]
    d = loans if exclude is None else loans[~exclude.reindex(loans.index).fillna(False).to_numpy()]
    w = d["weight"] if (weight and "weight" in d.columns) else None
    return cohort_table(d["open_date"], d["first_bad_mob"], d["months_observed"], v["max_mob"], v.get("min_loans_per_cell", 1), w)


def vintage_table(df: pd.DataFrame, mapping: dict, cfg: dict, status: pd.Series | None = None) -> pd.DataFrame:
    """Đường 'cột có sẵn': cần ánh xạ open_date, first_bad_mob, months_observed.

    status (tuỳ chọn, tương thích cũ): Series True ở dòng cần loại hoặc chuỗi 'excluded'. Khuyến nghị dùng
    target.exclusion_flags (chỉ loại fraud/đã mất/nhân viên/chính sách, KHÔNG loại khoản chưa đủ chín).
    """
    for k in ("open_date", "first_bad_mob", "months_observed"):
        if not mapping.get(k):
            raise ValueError(f"Vintage cần ánh xạ cột '{k}'.")
    loans = pd.DataFrame({"open_date": df[mapping["open_date"]], "first_bad_mob": df[mapping["first_bad_mob"]],
                          "months_observed": df[mapping["months_observed"]]}, index=df.index)
    excl = None
    if status is not None:
        s = status.reindex(df.index)
        excl = s if s.dtype == bool else s.eq("excluded")
    return vintage_from_loans(loans, cfg, excl)


def _n_obs(tab: pd.DataFrame, mob_cols: list) -> pd.DataFrame:
    n = globals()["n_obs"](tab)
    if n is not None:
        return n.reindex(index=tab.index, columns=mob_cols).fillna(0)
    # bảng không có n_obs (ví dụ tự dựng): dùng size cohort ở ô có dữ liệu
    return pd.DataFrame({m: tab["size"].where(tab[m].notna(), 0) for m in mob_cols}, index=tab.index)


def suggest_window(tab: pd.DataFrame, cfg: dict) -> dict:
    """Gợi ý performance window = MOB mà đường cong 'chín' (độ tăng < ngưỡng trong N MOB liên tiếp).

    Độ tăng từ MOB m-1 → m tính LIKE-FOR-LIKE: chỉ trên các cohort quan sát được ở cả hai MOB (trọng số = số khoản
    quan sát), rồi cộng dồn thành đường cong → không bị lệch do tập cohort thay đổi theo MOB.
    """
    v = cfg["vintage"]["maturity"]
    mob_cols = [c for c in tab.columns if c != "size"]
    R = tab[mob_cols].to_numpy(dtype=float)
    N = _n_obs(tab, mob_cols).to_numpy(dtype=float)
    inc = np.full(len(mob_cols), np.nan)
    curve = np.full(len(mob_cols), np.nan)
    for j in range(len(mob_cols)):
        if j == 0:
            ok = ~np.isnan(R[:, 0]) & (N[:, 0] > 0)
            if ok.any():
                curve[0] = np.average(R[ok, 0], weights=N[ok, 0])
            continue
        ok = ~np.isnan(R[:, j]) & ~np.isnan(R[:, j - 1]) & (N[:, j] > 0)
        if ok.any():
            inc[j] = np.average(R[ok, j] - R[ok, j - 1], weights=N[ok, j])
            curve[j] = curve[j - 1] + inc[j] if not np.isnan(curve[j - 1]) else np.nan
    curve_s = pd.Series(curve, index=mob_cols)
    delta_pp = pd.Series(inc * 100, index=mob_cols)
    n, thr = v["consecutive_mob"], v["delta_threshold_pp"]
    suggested = None
    vals = delta_pp.to_numpy()
    for i in range(1, len(vals) - n + 1):
        win = vals[i:i + n]
        if not np.isnan(win).any() and (win < thr).all():
            suggested = int(mob_cols[i - 1])
            break
    lo, hi = cfg["target"]["performance_window_range"]
    return {"curve": curve_s, "delta_pp": delta_pp, "suggested_mob": suggested,
            "within_range": suggested is not None and lo <= suggested <= hi,
            "message": (f"Gợi ý performance window = {suggested} tháng (độ tăng bad rate tích lũy < {thr}pp trong {n} MOB liên tiếp, so sánh cùng tập cohort)."
                        if suggested else "Chưa tìm được MOB 'chín' theo ngưỡng; xem đồ thị và chọn thủ công.")}


def cohort_anomalies(tab: pd.DataFrame, cfg: dict, at_mob: int | None = None) -> pd.DataFrame:
    """Cohort có bad rate lệch > z-score ngưỡng tại MOB chỉ định (mặc định MOB lớn nhất có ≥6 cohort)."""
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
