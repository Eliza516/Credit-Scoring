"""Dữ liệu performance (lịch sử DPD theo tháng) cho vintage & dựng target.

Hai dạng đầu vào: dọc (mỗi dòng = khoản vay × tháng) và ngang (các cột DPD_M1…DPD_Mn).
Mọi nguồn đều quy về (a) panel dọc [id, mob, dpd, (balance)] hoặc (b) bảng theo khoản vay căn theo dòng của file chính
với các cột: open_date, first_bad_mob, months_observed (+ dpd_max, n_payments, weight khi có panel).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_SUFFIX = re.compile(r"^(?P<prefix>.*?)[\s_\-]*(?P<n>\d+)$")
_HINTS = {
    "id": r"(^|_)(id|loan|account|acct|contract|hd|ma)(_|$)|loan|account|contract|hop_dong",
    "period": r"month|date|period|snapshot|report|thang|ky_|ky$|mob|as_?of",
    "dpd": r"dpd|delq|dlq|overdue|past_?due|qua_?han|days",
    "balance": r"bal|outstanding|du_?no|principal|exposure",
}


class PerformanceError(ValueError):
    pass


# ----------------------------------------------------------------- nhận diện
def wide_groups(df: pd.DataFrame, min_cols: int = 3) -> dict[str, list[tuple[int, str]]]:
    """Nhóm các cột số có cùng tiền tố và hậu tố là số (dpd_m1, dpd_m2…). Trả {tiền_tố: [(n, cột)] sắp theo n}."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        m = _SUFFIX.match(str(c))
        if m and m.group("prefix"):
            groups.setdefault(m.group("prefix").lower().rstrip("_- "), []).append((int(m.group("n")), c))
    return {p: sorted(v) for p, v in groups.items() if len({n for n, _ in v}) >= min_cols}


def detect_layout(df: pd.DataFrame) -> str:
    """'wide' nếu có ≥3 cột DPD theo tháng cùng tiền tố, ngược lại 'long'."""
    return "wide" if wide_groups(df) else "long"


def guess_wide_cols(df: pd.DataFrame) -> list[str]:
    """Nhóm cột ngang khả dĩ nhất (ưu tiên tiền tố có chữ dpd/delq…, rồi nhóm dài nhất)."""
    g = wide_groups(df)
    if not g:
        return []
    key = max(g, key=lambda p: (bool(re.search(_HINTS["dpd"], p)), len(g[p])))
    return [c for _, c in g[key]]


def guess_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Gợi ý cột id / kỳ quan sát / dpd / dư nợ theo tên."""
    out: dict[str, str | None] = {k: None for k in _HINTS}
    taken: set[str] = set()
    for role in ("dpd", "balance", "period", "id"):
        for c in df.columns:
            if c in taken:
                continue
            if re.search(_HINTS[role], str(c), re.I):
                out[role] = c
                taken.add(c)
                break
    return out


# ------------------------------------------------------------------ chuẩn hoá
def _months_between(a: pd.Series, b: pd.Series) -> pd.Series:
    """Số tháng từ a đến b (theo năm-tháng)."""
    return (b.dt.year - a.dt.year) * 12 + (b.dt.month - a.dt.month)


def parse_period(s: pd.Series) -> pd.Series:
    """Chuyển cột kỳ sang datetime: hỗ trợ ngày, 'YYYY-MM', số YYYYMM."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna()
        if len(v) and ((v >= 190001) & (v <= 299912)).all() and (v % 100).between(1, 12).all():
            return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def wide_to_long(df: pd.DataFrame, id_col: str, dpd_cols: list[str]) -> pd.DataFrame:
    """Dạng ngang → dọc [id, mob, dpd]. MOB lấy từ hậu tố số của tên cột; ô trống = chưa quan sát (bị bỏ)."""
    if not dpd_cols:
        raise PerformanceError("Chưa chọn cột DPD theo tháng.")
    mobs = {}
    for c in dpd_cols:
        m = _SUFFIX.match(str(c))
        if not m:
            raise PerformanceError(f"Cột '{c}' không có hậu tố số (vd dpd_m3) để suy ra MOB.")
        mobs[c] = int(m.group("n"))
    if len(set(mobs.values())) != len(mobs):
        raise PerformanceError("Hai cột DPD có cùng số tháng (hậu tố trùng).")
    long = df[[id_col] + dpd_cols].melt(id_vars=id_col, var_name="col", value_name="dpd")
    long["mob"] = long["col"].map(mobs)
    long = long.dropna(subset=["dpd"]).rename(columns={id_col: "id"})[["id", "mob", "dpd"]]
    long.attrs["quality"] = {"rows_in": int(len(df) * len(dpd_cols)), "nan_dpd": int(len(df) * len(dpd_cols) - len(long)),
                             "neg_dpd": int((long["dpd"] < 0).sum()), "dup_rows": 0, "neg_mob": 0, "unmatched_rows": 0}
    long["dpd"] = long["dpd"].clip(lower=0)
    return long.reset_index(drop=True)


def normalize_long(perf: pd.DataFrame, id_col: str, period_col: str, kind: str, dpd_col: str,
                   open_dates: pd.Series | None = None, balance_col: str | None = None) -> pd.DataFrame:
    """Panel dọc → [id, mob, dpd, (balance)].

    kind='mob': cột kỳ là MOB (số). kind='date': cột kỳ là tháng quan sát; MOB = số tháng kể từ tháng mở
    (open_dates: Series theo id, kiểu datetime). Dòng trùng (id, mob) lấy DPD lớn nhất. Thống kê chất lượng ở .attrs['quality'].
    """
    cols = [id_col, period_col, dpd_col] + ([balance_col] if balance_col else [])
    if len(set(cols)) != len(cols):
        raise PerformanceError("Các vai trò cột đang trùng nhau.")
    d = perf[cols].copy()
    d.columns = ["id", "period", "dpd"] + (["balance"] if balance_col else [])
    q = {"rows_in": len(d)}
    d["dpd"] = pd.to_numeric(d["dpd"], errors="coerce")
    q["nan_dpd"] = int(d["dpd"].isna().sum())
    d = d.dropna(subset=["dpd", "id", "period"])
    q["neg_dpd"] = int((d["dpd"] < 0).sum())
    d["dpd"] = d["dpd"].clip(lower=0)
    q["unmatched_rows"] = 0
    if kind == "mob":
        d["mob"] = pd.to_numeric(d["period"], errors="coerce")
    elif kind == "date":
        if open_dates is None:
            raise PerformanceError("Kỳ dạng ngày cần cột ngày mở ở file chính (open_date) và ID để ghép.")
        per = parse_period(d["period"])
        if per.isna().all():
            raise PerformanceError("Không đọc được cột kỳ quan sát thành ngày/tháng.")
        od = d["id"].map(open_dates)
        q["unmatched_rows"] = int(od.isna().sum())
        d["mob"] = _months_between(od, per)
    else:
        raise PerformanceError("kind phải là 'mob' hoặc 'date'.")
    d = d.dropna(subset=["mob"])
    d["mob"] = d["mob"].astype(int)
    q["neg_mob"] = int((d["mob"] < 0).sum())
    d = d[d["mob"] >= 0]
    q["dup_rows"] = int(d.duplicated(["id", "mob"]).sum())
    agg = {"dpd": "max"}
    if balance_col:
        agg["balance"] = "max"
    d = d.groupby(["id", "mob"], as_index=False).agg(agg)
    d.attrs["quality"] = q
    return d.sort_values(["id", "mob"]).reset_index(drop=True)


def quality_report(long: pd.DataFrame, main_ids: pd.Series) -> pd.DataFrame:
    """Bảng kiểm tra chất lượng: độ phủ ID, dòng trùng, DPD âm, khoảng trống MOB…"""
    q = long.attrs.get("quality", {})
    perf_ids = pd.Index(long["id"].unique())
    main_idx = pd.Index(main_ids.dropna().unique())
    g = long[long["mob"] >= 1].groupby("id")["mob"].agg(["min", "max", "nunique"])
    gaps = int(((g["max"] - g["min"] + 1 - g["nunique"]) > 0).sum()) if len(g) else 0
    rows = [
        ("Dòng performance đọc được", q.get("rows_in", len(long)), ""),
        ("Khoản vay có performance", len(perf_ids), ""),
        ("Khoản vay ở file chính nhưng KHÔNG có performance", int(len(main_idx.difference(perf_ids))), "sẽ bị loại “chưa đủ quan sát” khi dựng target"),
        ("ID ở performance nhưng không có ở file chính", int(len(perf_ids.difference(main_idx))), "bị bỏ qua"),
        ("Dòng trùng (ID, MOB) – đã gộp lấy DPD lớn nhất", q.get("dup_rows", 0), ""),
        ("Dòng thiếu/lỗi DPD – đã bỏ", q.get("nan_dpd", 0), ""),
        ("DPD âm – đã đặt về 0", q.get("neg_dpd", 0), ""),
        ("Dòng MOB âm (trước tháng mở) – đã bỏ", q.get("neg_mob", 0), ""),
        ("Dòng không ghép được ngày mở", q.get("unmatched_rows", 0), ""),
        ("Khoản vay có khoảng trống MOB", gaps, "tháng thiếu giữa các kỳ quan sát"),
        ("MOB lớn nhất", int(long["mob"].max()) if len(long) else 0, ""),
    ]
    return pd.DataFrame(rows, columns=["kiểm tra", "giá trị", "ghi chú"])


# ------------------------------------------------------ bảng theo khoản vay
def panel_summary(long: pd.DataFrame, bad_dpd: float) -> pd.DataFrame:
    """Theo id: dpd_max, first_bad_mob (MOB nhỏ nhất có DPD ≥ ngưỡng), months_observed, n_payments, weight (dư nợ ở MOB nhỏ nhất)."""
    g = long.groupby("id")
    out = pd.DataFrame({"dpd_max": g["dpd"].max(), "months_observed": g["mob"].max()})
    out["first_bad_mob"] = long[long["dpd"] >= bad_dpd].groupby("id")["mob"].min().reindex(out.index)
    out["n_payments"] = long[long["mob"] >= 1].groupby("id").size().reindex(out.index).fillna(0).astype(int)
    if "balance" in long.columns:
        first = long.sort_values("mob").drop_duplicates("id").set_index("id")["balance"]
        out["weight"] = first.reindex(out.index)
    return out


def loans_from_panel(long: pd.DataFrame, df: pd.DataFrame, id_col: str, open_col: str, bad_dpd: float) -> pd.DataFrame:
    """Bảng theo khoản vay CĂN THEO DÒNG của df (file chính). Khoản không có performance: months_observed=0, n_payments=0, dpd_max=0."""
    if df[id_col].duplicated().any():
        raise PerformanceError(f"Cột ID '{id_col}' ở file chính có giá trị trùng – không ghép được performance.")
    s = panel_summary(long, bad_dpd).reindex(df[id_col].to_numpy())
    s.index = df.index
    s["months_observed"] = s["months_observed"].fillna(0)
    s["n_payments"] = s["n_payments"].fillna(0)
    s["dpd_max"] = s["dpd_max"].fillna(0)
    s.insert(0, "open_date", pd.to_datetime(df[open_col]))
    return s


def loans_from_main_summary(df: pd.DataFrame, open_col: str, first_bad: str | None, first_bad_kind: str,
                            observed: str | None, observed_kind: str, asof_date=None, bad_mask: pd.Series | None = None) -> pd.DataFrame:
    """Nguồn B1: dùng cột tổng hợp có sẵn trong file chính.

    first_bad_kind: 'mob' (cột là MOB) | 'date' (cột là ngày bad đầu tiên, rỗng = chưa bad).
    observed_kind: 'mob' (cột số tháng đã quan sát) | 'date' (cột ngày quan sát) | 'asof' (một ngày cắt dữ liệu chung, asof_date).
    bad_mask: nếu có, chỉ khoản vay ở mask này mới có first_bad_mob (các khoản khác = chưa bad) – dùng khi cột ngày
    (vd ngày thanh toán cuối) chỉ là thời điểm vỡ nợ đối với khoản đã bad theo cột trạng thái.
    """
    op = pd.to_datetime(df[open_col])
    out = pd.DataFrame({"open_date": op}, index=df.index)
    if first_bad is None:
        raise PerformanceError("Cần chọn cột MOB/ngày bad đầu tiên.")
    fb = df[first_bad]
    out["first_bad_mob"] = (_months_between(op, pd.to_datetime(fb, errors="coerce")) if first_bad_kind == "date"
                            else pd.to_numeric(fb, errors="coerce")).clip(lower=0)
    if bad_mask is not None:
        out.loc[~bad_mask.reindex(df.index).fillna(False).to_numpy(), "first_bad_mob"] = np.nan
    if observed_kind == "asof":
        if asof_date is None:
            raise PerformanceError("Cần chọn ngày cắt dữ liệu.")
        out["months_observed"] = _months_between(op, pd.Series(pd.Timestamp(asof_date), index=df.index))
    elif observed_kind == "date":
        out["months_observed"] = _months_between(op, pd.to_datetime(df[observed], errors="coerce"))
    else:
        out["months_observed"] = pd.to_numeric(df[observed], errors="coerce")
    out["months_observed"] = out["months_observed"].clip(lower=0).fillna(0)
    return out


FIELD_COLS = {"dpd_max": "__v_dpd_max", "first_bad_mob": "__v_first_bad_mob", "months_observed": "__v_months_observed", "n_payments": "__v_n_payments"}


def apply_loans(df: pd.DataFrame, mapping: dict, loans: pd.DataFrame, fields: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Ghép các cột suy ra từ performance/vintage vào BẢN SAO của df và mapping hiệu lực (ghi đè ánh xạ cũ).

    Cột tạm có tiền tố '__v_' – không được thành biến mô hình (tránh rò rỉ); df gốc không bị sửa.
    """
    fields = [f for f in (fields or list(FIELD_COLS)) if f in loans.columns]
    d, m = df.copy(), dict(mapping)
    for f in fields:
        d[FIELD_COLS[f]] = loans[f].reindex(df.index)
        m[f] = FIELD_COLS[f]
    return d, m


def target_source_errors(mapping: dict, cfg: dict | None = None) -> list[str]:
    """Lỗi nếu chưa có nguồn để dựng target (cột target, cột trạng thái, hoặc dpd_max)."""
    if mapping.get("status_col"):
        st = (cfg or {}).get("target", {}).get("status", {})
        if cfg is not None and not (st.get("bad") and st.get("good")):
            return ["Đã chọn cột trạng thái nhưng chưa khai báo giá trị bad và good (trang Quy tắc → Định nghĩa target → status)."]
        return []
    if mapping.get("target") or mapping.get("dpd_max"):
        return []
    return ["Chưa có nguồn dựng target: ánh xạ cột target/dpd_max ở bước 1, hoặc nạp dữ liệu performance (DPD theo tháng) ở bước 3."]
