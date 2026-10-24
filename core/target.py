"""Định nghĩa target bad/good/indeterminate/exclusion."""
from __future__ import annotations

import numpy as np
import pandas as pd

BAD, GOOD, INDET, EXCL = "bad", "good", "indeterminate", "excluded"


def _flag(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col:
        return pd.Series(False, index=df.index)
    return df[col].fillna(0).astype(float).astype(bool)


def _flag_exclusions(df: pd.DataFrame, mapping: dict, cfg: dict) -> tuple[pd.Series, pd.Series]:
    """(mask, lý do) cho các loại trừ theo CỜ: fraud, đã mất, nhân viên/VIP, tài khoản chính sách (không gồm chưa đủ chín)."""
    ex = cfg["target"]["exclusion"]
    mask = pd.Series(False, index=df.index)
    reason = pd.Series("", index=df.index, dtype=object)
    for key, col, why in (("fraud", "fraud_flag", "fraud"), ("deceased", "deceased_flag", "đã mất"),
                          ("staff_vip", "staff_vip_flag", "nhân viên/VIP"), ("policy_accounts", "policy_flag", "tài khoản chính sách")):
        if ex.get(key):
            f = _flag(df, mapping.get(col))
            reason[f & ~mask] = why
            mask = mask | f
    return mask, reason


def exclusion_flags(df: pd.DataFrame, mapping: dict, cfg: dict) -> pd.Series:
    """True ở khoản vay bị loại theo cờ (fraud/đã mất/nhân viên-VIP/chính sách). KHÔNG gồm 'chưa đủ chín' –
    dùng cho vintage để không mất các cohort trẻ."""
    return _flag_exclusions(df, mapping, cfg)[0]


def build_target(df: pd.DataFrame, mapping: dict, cfg: dict) -> pd.DataFrame:
    """Trả DataFrame (cùng index) gồm: status (bad/good/indeterminate/excluded), reason, y (1 bad, 0 good, NaN còn lại)."""
    t = cfg["target"]
    n = len(df)
    status = pd.Series(INDET, index=df.index, dtype=object)
    reason = pd.Series("", index=df.index, dtype=object)

    if mapping.get("target"):
        raw = df[mapping["target"]]
        status = pd.Series(np.where(raw == 1, BAD, np.where(raw == 0, GOOD, EXCL)), index=df.index, dtype=object)
        reason[raw.isna()] = "target rỗng"
        return _finish(status, reason, df)

    if mapping.get("status_col"):
        return _build_from_status(df, mapping, cfg)

    win = t["performance_window_months"]
    dpd = df[mapping["dpd_max"]].fillna(0)
    bad_dpd = dpd >= t["bad"]["dpd_threshold"]
    ev = pd.Series(False, index=df.index)
    for key, flag in (("include_charge_off", "charge_off_flag"), ("include_write_off", "write_off_flag"),
                      ("include_bankruptcy", "bankruptcy_flag")):
        if t["bad"].get(key):
            ev |= _flag(df, mapping.get(flag))
    is_bad = bad_dpd | ev
    # bad xảy ra sau performance window thì chưa phải bad trong cửa sổ
    late = pd.Series(False, index=df.index)
    if mapping.get("first_bad_mob"):
        fb = df[mapping["first_bad_mob"]]
        late = is_bad & fb.notna() & (fb > win)
        is_bad = is_bad & ~late
    clean = (dpd <= t["good"]["max_dpd"]) & ~ev
    status[clean] = GOOD
    status[is_bad] = BAD
    status[late] = INDET
    reason[late] = f"bad sau MOB {win}"
    status[(dpd > t["good"]["max_dpd"]) & ~is_bad & ~late] = INDET
    reason[(status == INDET) & (reason == "")] = f"DPD {t['indeterminate']['dpd_range'][0]}-{t['indeterminate']['dpd_range'][1]} (chưa 91+)"

    excl, flag_reason = _flag_exclusions(df, mapping, cfg)
    reason[excl] = flag_reason[excl]
    ex = t["exclusion"]

    def mark(mask, why):
        nonlocal excl
        m = mask & ~excl
        reason[m] = why
        excl = excl | mask
    if mapping.get("n_payments") and ex.get("min_payments"):
        mark((df[mapping["n_payments"]].fillna(0) < ex["min_payments"]) & (status != BAD), f"< {ex['min_payments']} kỳ thanh toán")
    if mapping.get("months_observed"):
        # chưa đủ chín: chưa bad và quan sát chưa tới hết performance window
        mark((df[mapping["months_observed"]].fillna(0) < win) & (status != BAD), f"chưa đủ {win} tháng quan sát")
    status[excl] = EXCL
    return _finish(status, reason, df)


def provisional_y(df: pd.DataFrame, mapping: dict, cfg: dict) -> pd.Series | None:
    """Nhãn tạm (1 bad / 0 good / NaN) chỉ để dựng dashboard sớm: theo cột trạng thái, cột target, hoặc DPD tối đa ≥ ngưỡng.
    Không áp dụng loại trừ/đáo hạn. None nếu chưa có nguồn."""
    st = cfg["target"]["status"]
    if mapping.get("status_col") and mapping["status_col"] in df.columns and (st["bad"] or st["good"]):
        raw = df[mapping["status_col"]].astype("string")
        y = pd.Series(np.nan, index=df.index)
        y[raw.isin(st["good"]).to_numpy()] = 0.0
        y[raw.isin(st["bad"]).to_numpy()] = 1.0
        return y
    if mapping.get("target") and mapping["target"] in df.columns:
        v = pd.to_numeric(df[mapping["target"]], errors="coerce")
        return v.where(v.isin([0, 1]))
    if mapping.get("dpd_max") and mapping["dpd_max"] in df.columns:
        d = pd.to_numeric(df[mapping["dpd_max"]], errors="coerce").fillna(0)
        return (d >= cfg["target"]["bad"]["dpd_threshold"]).astype(float)
    return None


def term_months(s: pd.Series) -> pd.Series:
    """Kỳ hạn (tháng) từ cột số hoặc chữ kiểu '36 months'."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype("string").str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")


def data_end_period(df: pd.DataFrame, col: str) -> pd.Period | None:
    """Tháng cuối của dữ liệu = max của cột ngày (chữ hoặc datetime)."""
    d = df[col] if pd.api.types.is_datetime64_any_dtype(df[col]) else pd.to_datetime(df[col], errors="coerce")
    m = d.max()
    return None if pd.isna(m) else m.to_period("M")


def _build_from_status(df: pd.DataFrame, mapping: dict, cfg: dict) -> pd.DataFrame:
    """Nhãn từ cột trạng thái chữ (vd loan_status): bad/good theo cfg.target.status; trạng thái khác → indeterminate.

    Loại trừ: cờ (fraud…), mở trước min_open_year, chưa đáo hạn (ngày mở + kỳ hạn > tháng cuối của dữ liệu).
    """
    st = cfg["target"]["status"]
    raw = df[mapping["status_col"]].astype("string")
    status = pd.Series(INDET, index=df.index, dtype=object)
    status[raw.isin(st["bad"]).to_numpy()] = BAD
    status[raw.isin(st["good"]).to_numpy()] = GOOD
    reason = pd.Series("", index=df.index, dtype=object)
    other = (status == INDET).to_numpy()
    reason[other] = ("trạng thái " + raw.fillna("(trống)")[other]).to_numpy()
    excl, why = _flag_exclusions(df, mapping, cfg)
    labeled = status.isin([BAD, GOOD])
    reason[excl & labeled] = why[excl & labeled]
    od = None
    if mapping.get("open_date"):
        od = pd.to_datetime(df[mapping["open_date"]], errors="coerce")

    def mark(mask, text):
        nonlocal excl
        m = mask & labeled & ~excl
        reason[m] = text
        excl = excl | (mask & labeled)
    if st.get("min_open_year") and od is not None:
        mark(od.dt.year < st["min_open_year"], f"mở trước {st['min_open_year']}")
    mt = st.get("maturity") or {}
    if mt.get("enabled") and mt.get("term_col") and mt.get("asof_col") and od is not None:
        end = data_end_period(df, mt["asof_col"])
        if end is not None:
            due = od.dt.to_period("M") + term_months(df[mt["term_col"]]).fillna(10**4).astype(int)
            mark(due > end, f"chưa đáo hạn tới {end}")
    status[excl & labeled] = EXCL
    return _finish(status, reason, df)


def _finish(status, reason, df) -> pd.DataFrame:
    y = pd.Series(np.nan, index=df.index)
    y[status == BAD] = 1.0
    y[status == GOOD] = 0.0
    return pd.DataFrame({"status": status, "reason": reason, "y": y})


def summary(tgt: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    vc = tgt["status"].value_counts()
    tab = pd.DataFrame({"count": vc})
    tab["pct"] = (tab["count"] / len(tgt)).round(4)
    tab["in_development"] = [s in (BAD, GOOD) or (s == INDET and not cfg["target"]["exclude_indeterminate_from_development"]) for s in tab.index]
    return tab


def exclusion_reasons(tgt: pd.DataFrame) -> pd.DataFrame:
    m = tgt[tgt["reason"] != ""]
    return m.groupby(["status", "reason"]).size().rename("count").reset_index()


def development_mask(tgt: pd.DataFrame, cfg: dict) -> pd.Series:
    """True cho dòng đưa vào mẫu phát triển (bad/good; indeterminate tuỳ cấu hình)."""
    keep = tgt["status"].isin([BAD, GOOD])
    return keep


def bad_rate(tgt: pd.DataFrame) -> float:
    y = tgt["y"].dropna()
    return float(y.mean()) if len(y) else float("nan")
