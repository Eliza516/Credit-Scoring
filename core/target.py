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
