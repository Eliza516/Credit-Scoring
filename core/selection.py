"""Phễu chọn biến theo ngưỡng + gợi ý. Người dùng duyệt/ghi đè."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import binning


def iv_band(iv: float, cfg: dict) -> str:
    t = cfg["selection"]["iv"]
    if iv < t["useless_below"]:
        return "useless"
    if iv < t["weak_below"]:
        return "weak"
    if iv < t["medium_below"]:
        return "medium"
    if iv <= t["strong_below"]:
        return "strong"
    return "suspicious"


def compliance_flags(variables: list[str], cfg: dict) -> pd.DataFrame:
    c = cfg["selection"]["compliance"]
    rows = []
    for v in variables:
        lv = v.lower()
        prot = [p for p in c["protected_attributes"] if p.lower() in lv]
        prox = [p for p in c["proxy_warning_features"] if p.lower() in lv]
        rows.append({"variable": v, "protected": bool(prot), "proxy_warning": bool(prox) and not prot,
                     "note": ("thuộc tính được bảo vệ: " + ",".join(prot)) if prot else (("biến proxy: " + ",".join(prox)) if prox else "")})
    return pd.DataFrame(rows)


def _slope_sign(table: pd.DataFrame) -> int:
    br = table.loc[table["kind"] == "normal", "bad_rate"].dropna().to_numpy()
    if len(br) < 2:
        return 0
    x = np.arange(len(br))
    s = np.polyfit(x, br, 1)[0]
    return int(np.sign(s)) if abs(s) > 1e-9 else 0


def vif_table(W: pd.DataFrame) -> pd.Series:
    if W.shape[1] < 2:
        return pd.Series(1.0, index=W.columns)
    X = np.column_stack([np.ones(len(W)), W.to_numpy()])
    return pd.Series([variance_inflation_factor(X, i + 1) for i in range(W.shape[1])], index=W.columns)


def funnel(df_train: pd.DataFrame, W_train: pd.DataFrame, tabs_train: dict, tabs_test: dict | None, cfg: dict,
           protected_override: list[str] | None = None) -> pd.DataFrame:
    """Chạy phễu theo thứ tự p.41. Mỗi biến nhận suggestion keep/drop/review + reasons + stage bị loại.

    df_train: dữ liệu thô train (để tương quan thô). W_train: WoE train. tabs_*: bảng bin train/test (cùng spec).
    """
    s = cfg["selection"]
    ivs = pd.Series({v: binning.total_iv(t) for v, t in tabs_train.items()})
    order = ivs.sort_values(ascending=False).index.tolist()
    comp = compliance_flags(order, cfg).set_index("variable")
    res = {v: {"variable": v, "iv": ivs[v], "iv_band": iv_band(ivs[v], cfg), "stage": "", "suggestion": "keep", "reason": "",
               "iv_test": np.nan, "bad_rate_trend_reversal": False, "flags": comp.loc[v, "note"]} for v in order}

    def drop(v, stage, reason, kind="drop"):
        r = res[v]
        if r["suggestion"] == "keep" or (kind == "drop" and r["suggestion"] == "review"):
            r["suggestion"], r["stage"], r["reason"] = kind, stage, reason

    must_keep, must_drop = set(s.get("must_keep") or []), set(s.get("must_drop") or [])
    for v in order:
        if comp.loc[v, "protected"]:
            drop(v, "compliance", f"{comp.loc[v, 'note']} – không được dùng")
        elif comp.loc[v, "proxy_warning"]:
            res[v]["flags"] = comp.loc[v, "note"]
        if v in must_drop:
            drop(v, "must_drop", "nằm trong danh sách bắt buộc loại")
    # 2. IV
    for v in order:
        band = res[v]["iv_band"]
        if band == "useless":
            drop(v, "iv", f"IV {ivs[v]:.3f} < {s['iv']['useless_below']}")
        elif band == "suspicious":
            drop(v, "iv", f"IV {ivs[v]:.2f} > {s['iv']['suspicious_above']} nghi ngờ rò rỉ", "review")
    # 3. đơn điệu / logic
    for v in order:
        tr = binning.trend(tabs_train[v])
        res[v]["trend"] = tr
    # 4. ổn định train vs test
    if tabs_test:
        for v in order:
            tt = tabs_test.get(v)
            if tt is None:
                continue
            ivt = binning.total_iv(tt)
            res[v]["iv_test"] = ivt
            a, b = _slope_sign(tabs_train[v]), _slope_sign(tt)
            rev = a != 0 and b != 0 and a != b
            res[v]["bad_rate_trend_reversal"] = rev
            if s["stability"]["check_bad_rate_trend_reversal"] and rev:
                drop(v, "stability", "xu hướng bad rate đảo chiều giữa train và test")
            elif ivs[v] > 0 and (ivs[v] - ivt) / ivs[v] > s["stability"]["max_iv_drop_pct"]:
                drop(v, "stability", f"IV tụt {(ivs[v]-ivt)/ivs[v]:.0%} từ train sang test", "review")
    # 5. tương quan thô (numeric) – giữ biến IV cao hơn
    kept = [v for v in order if res[v]["suggestion"] == "keep"]
    num = [v for v in kept if pd.api.types.is_numeric_dtype(df_train[v]) and not pd.api.types.is_bool_dtype(df_train[v])]
    if len(num) > 1:
        cm = df_train[num].corr(method="spearman").abs()
        done: list[str] = []
        for v in num:
            hit = [u for u in done if cm.loc[v, u] > s["correlation"]["raw_abs_threshold"]]
            if hit:
                drop(v, "corr_raw", f"|r|={cm.loc[v, hit[0]]:.2f} > {s['correlation']['raw_abs_threshold']} với {hit[0]} (IV cao hơn)")
            else:
                done.append(v)
    # 6. tương quan sau binning
    kept = [v for v in order if res[v]["suggestion"] == "keep"]
    if len(kept) > 1:
        cm = W_train[kept].corr().abs()
        done = []
        for v in kept:
            hit = [u for u in done if cm.loc[v, u] > s["correlation"]["binned_abs_threshold"]]
            if hit:
                drop(v, "corr_binned", f"|r| WoE={cm.loc[v, hit[0]]:.2f} > {s['correlation']['binned_abs_threshold']} với {hit[0]}")
            else:
                done.append(v)
    # 7. VIF
    kept = [v for v in order if res[v]["suggestion"] == "keep"]
    while len(kept) > 1:
        vif = vif_table(W_train[kept])
        worst = vif.idxmax()
        if vif[worst] <= s["vif"]["threshold"]:
            break
        drop(worst, "vif", f"VIF {vif[worst]:.1f} > {s['vif']['threshold']}")
        kept.remove(worst)
    vif = vif_table(W_train[kept]) if len(kept) > 1 else pd.Series(1.0, index=kept)
    # 8. giới hạn số biến
    hi = s["final_variable_count"][1]
    kept = [v for v in order if res[v]["suggestion"] == "keep"]
    for v in kept[hi:]:
        drop(v, "count", f"vượt tối đa {hi} biến (IV thấp hơn)", "review")
    out = pd.DataFrame(res.values())
    out["vif"] = out["variable"].map(vif)
    # must_keep ghi đè (trừ biến bị bảo vệ)
    for v in must_keep:
        if v in res and not comp.loc[v, "protected"]:
            m = out["variable"] == v
            out.loc[m, ["suggestion", "reason"]] = ["keep", "bắt buộc giữ (must_keep)"]
    return out
