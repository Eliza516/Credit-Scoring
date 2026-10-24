"""Phân tích chất lượng dữ liệu + gợi ý theo ngưỡng. Không tự áp dụng; apply() nhận quyết định của người dùng."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _special_mask(s: pd.Series, specials: list) -> pd.Series:
    if not specials:
        return pd.Series(False, index=s.index)
    return s.isin(specials)


def analyze(df: pd.DataFrame, features: list[str], cfg: dict, forced_drop: dict[str, str] | None = None) -> pd.DataFrame:
    """Bảng phân tích mỗi biến + cột 'suggestion' (hành động đề xuất) và 'reason'.

    forced_drop: {cột: lý do} các cột phải loại theo hồ sơ dữ liệu (rò rỉ, định danh…).
    """
    forced_drop = forced_drop or {}
    cl = cfg["cleaning"]
    specials = cl.get("special_values") or []
    rules = cl.get("impossible_value_rules") or {}
    p_lo, p_hi = cl["outliers"]["cap_floor_percentiles"]
    rows = []
    for c in features:
        s = df[c]
        sp = _special_mask(s, specials)
        miss = float(s.isna().mean())
        nun = int(s.nunique(dropna=True))
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        row = dict(variable=c, dtype=str(s.dtype), numeric=is_num, missing_pct=round(miss, 4),
                   special_pct=round(float(sp.mean()), 4), n_unique=nun,
                   outlier_pct=np.nan, impossible_pct=np.nan)
        if is_num:
            v = s[~sp].dropna()
            if len(v) > 0:
                lo, hi = np.percentile(v, [p_lo, p_hi])
                row["outlier_pct"] = round(float(((v < lo) | (v > hi)).mean()), 4)
            if c in rules:
                a, b = rules[c]
                bad = ((v < a) | (v > b))
                row["impossible_pct"] = round(float(bad.mean()), 4) if len(v) else 0.0
        sugg, reason = "keep", ""
        if nun <= 1:
            sugg, reason = "drop", "hằng số"
        elif cl.get("drop_constant") and nun > 1 and (s.value_counts(normalize=True, dropna=False).iloc[0] >= 0.999):
            sugg, reason = "drop", "gần hằng số (>=99.9% một giá trị)"
        elif c in forced_drop:
            sugg, reason = "drop", forced_drop[c]
        elif miss > cl["max_missing_pct"]:
            sugg, reason = "drop", f"missing {miss:.0%} > ngưỡng {cl['max_missing_pct']:.0%}"
        elif (not is_num) and nun > cl.get("max_unique_categorical", 10**9) and not pd.api.types.is_datetime64_any_dtype(s):
            sugg, reason = "drop", f"{nun} giá trị chữ > ngưỡng {cl['max_unique_categorical']}"
        elif (not pd.api.types.is_float_dtype(s)) and len(s) >= 1000 and nun / max(s.notna().sum(), 1) >= cl.get("identifier_unique_ratio", 2):
            sugg, reason = "drop", "gần như định danh (hầu hết giá trị duy nhất)"
        elif row["special_pct"] > 0:
            sugg, reason = "review", f"có {row['special_pct']:.1%} giá trị đặc biệt → bin riêng"
        elif not np.isnan(row["impossible_pct"]) and row["impossible_pct"] > 0:
            sugg, reason = "review", f"{row['impossible_pct']:.2%} giá trị ngoài khoảng hợp lệ"
        rows.append({**row, "suggestion": sugg, "reason": reason})
    out = pd.DataFrame(rows)
    return out


def impossible_values(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Đếm giá trị ngoài khoảng hợp lệ theo quy tắc."""
    rules = cfg["cleaning"].get("impossible_value_rules") or {}
    out = []
    for c, (a, b) in rules.items():
        if c in df.columns:
            n = int(((df[c] < a) | (df[c] > b)).sum())
            out.append({"variable": c, "range": f"[{a}, {b}]", "n_violations": n})
    return pd.DataFrame(out, columns=["variable", "range", "n_violations"])


def apply(df: pd.DataFrame, cfg: dict, drop_vars: list[str], numeric_vars: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Áp dụng quyết định làm sạch. Trả (df mới, log).

    - Loại biến trong drop_vars.
    - Giá trị đặc biệt: giữ nguyên (binning xử lý thành bin riêng) – chỉ ghi log.
    - Giá trị bất khả theo quy tắc → NaN (sẽ thành missing bin).
    - Outlier: strategy cap_floor → clip theo P1/P99 (tính trên toàn bộ dữ liệu đưa vào, nên gọi sau khi chia train nếu muốn tránh rò rỉ).
    - missing: strategy impute → điền median/mode; drop_records → bỏ dòng thiếu nếu < ngưỡng.
    """
    cl = cfg["cleaning"]
    out = df.drop(columns=[c for c in drop_vars if c in df.columns]).copy()
    log = {"dropped": list(drop_vars), "impossible_to_nan": {}, "capped": [], "imputed": [], "rows_dropped": 0}
    specials = cl.get("special_values") or []
    for c, (a, b) in (cl.get("impossible_value_rules") or {}).items():
        if c in out.columns:
            m = (out[c] < a) | (out[c] > b)
            if m.any():
                out.loc[m, c] = np.nan
                log["impossible_to_nan"][c] = int(m.sum())
    nums = [c for c in (numeric_vars or out.columns) if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
    if cl["outliers"]["strategy"] == "cap_floor":
        lo_p, hi_p = cl["outliers"]["cap_floor_percentiles"]
        for c in nums:
            v = out[c].where(~out[c].isin(specials))
            if v.notna().sum() == 0:
                continue
            lo, hi = np.nanpercentile(v, [lo_p, hi_p])
            keep = out[c].isin(specials)
            out[c] = np.where(keep, out[c], out[c].clip(lo, hi))
            log["capped"].append(c)
    strat = cl["missing"]["strategy"]
    if strat == "impute":
        for c in out.columns:
            if out[c].isna().any():
                if c in nums:
                    out[c] = out[c].fillna(out[c].median())
                else:
                    mode = out[c].mode(dropna=True)
                    out[c] = out[c].fillna(mode.iloc[0] if len(mode) else "missing")
                log["imputed"].append(c)
    elif strat == "drop_records":
        thr = cl["missing"]["drop_records_if_missing_pct_below"]
        for c in list(out.columns):
            pct = out[c].isna().mean()
            if 0 < pct < thr:
                before = len(out)
                out = out[out[c].notna()]
                log["rows_dropped"] += before - len(out)
    return out, log
