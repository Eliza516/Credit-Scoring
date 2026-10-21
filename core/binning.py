"""Binning WoE: khởi tạo bằng optbinning, chỉnh tay (gộp/tách bin), bảng WoE/IV + gợi ý theo ngưỡng."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

OTHER = "__OTHER__"
RARE_CAT_PCT = 0.01


# ----------------------------------------------------------------- spec
def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def numeric_labels(splits: list[float]) -> list[str]:
    if not splits:
        return ["(-inf, inf)"]
    labs = [f"(-inf, {_fmt(splits[0])})"]
    for a, b in zip(splits[:-1], splits[1:]):
        labs.append(f"[{_fmt(a)}, {_fmt(b)})")
    labs.append(f"[{_fmt(splits[-1])}, inf)")
    return labs


def group_labels(groups: list[list]) -> list[str]:
    return ["{" + ", ".join(map(str, g)) + "}" for g in groups]


def assign(x: pd.Series, spec: dict) -> pd.Series:
    """Nhãn bin cho từng giá trị: 'Special', 'Missing' hoặc nhãn bin thường."""
    sp = spec.get("specials") or []
    is_sp = x.isin(sp) if sp else pd.Series(False, index=x.index)
    if spec["kind"] == "numeric":
        labs = np.array(numeric_labels(spec["splits"]), dtype=object)
        v = pd.to_numeric(x, errors="coerce")
        idx = np.searchsorted(np.asarray(spec["splits"], dtype=float), v.to_numpy(dtype=float, na_value=np.nan), side="right")
        out = pd.Series(labs[idx], index=x.index, dtype=object)
    else:
        labs = group_labels(spec["groups"])
        cmap = {}
        for lab, g in zip(labs, spec["groups"]):
            for c in g:
                cmap[c] = lab
        other_lab = cmap.get(OTHER)
        out = x.map(cmap)
        out = out.where(out.notna() | x.isna(), other_lab if other_lab else "Unseen")
        out = out.astype(object)
    out[x.isna()] = "Missing"
    out[is_sp] = "Special"
    return out


def make_table(x: pd.Series, y: pd.Series, spec: dict) -> pd.DataFrame:
    """Bảng bin: count, %, bad, good, bad_rate, WoE=ln(%Good/%Bad), IV. Cũng ghi kind/order."""
    lab = assign(x, spec)
    normal = numeric_labels(spec["splits"]) if spec["kind"] == "numeric" else group_labels(spec["groups"])
    order = normal + ["Special", "Missing"]
    d = pd.DataFrame({"bin": lab, "y": y.to_numpy()})
    g = d.groupby("bin")["y"].agg(["count", "sum"]).reindex(order).fillna(0)
    g = g[(g["count"] > 0) | g.index.isin(normal)]
    out = pd.DataFrame({"bin": g.index, "count": g["count"].astype(int).to_numpy(), "bad": g["sum"].astype(int).to_numpy()})
    out["good"] = out["count"] - out["bad"]
    out["kind"] = np.where(out["bin"] == "Special", "special", np.where(out["bin"] == "Missing", "missing", "normal"))
    k = len(out)
    G, B = out["good"].sum(), out["bad"].sum()
    out["count_pct"] = out["count"] / max(out["count"].sum(), 1)
    out["bad_rate"] = np.where(out["count"] > 0, out["bad"] / out["count"].where(out["count"] > 0, 1), np.nan)
    pg = (out["good"] + 0.5) / (G + 0.5 * k)
    pb = (out["bad"] + 0.5) / (B + 0.5 * k)
    out["woe"] = np.log(pg / pb)
    out["iv"] = (pg - pb) * out["woe"]
    return out.reset_index(drop=True)


def total_iv(table: pd.DataFrame) -> float:
    return float(table["iv"].sum())


def trend(table: pd.DataFrame) -> str:
    """'ascending' | 'descending' | 'none' của bad rate qua các bin thường (theo thứ tự bin)."""
    br = table.loc[table["kind"] == "normal", "bad_rate"].dropna().to_numpy()
    if len(br) < 2:
        return "flat"
    d = np.diff(br)
    if (d >= 0).all():
        return "ascending"
    if (d <= 0).all():
        return "descending"
    return "none"


# --------------------------------------------------------------- fitting
def _quantile_splits(v: pd.Series, k: int) -> list[float]:
    qs = np.unique(np.nanquantile(v, np.linspace(0, 1, k + 1)[1:-1]))
    return [float(q) for q in qs]


def _collapse_rare(x: pd.Series, specials) -> pd.Series:
    vc = x.value_counts(normalize=True)
    rare = vc[vc < RARE_CAT_PCT].index
    return x.where(~x.isin(rare), OTHER)


def fit_variable(x: pd.Series, y: pd.Series, cfg: dict, specials: list | None = None, force_monotonic: str | None = None) -> dict:
    """Binning khởi tạo cho 1 biến. force_monotonic: 'auto'|'ascending'|'descending'|None (không ràng buộc)."""
    b = cfg["binning"]
    specials = specials or []
    num = is_numeric(x)
    mask = x.notna() & ~x.isin(specials)
    spec: dict = {"variable": x.name, "kind": "numeric" if num else "categorical", "specials": list(specials)}
    xv, yv = x[mask], y[mask]
    mono = force_monotonic if force_monotonic is not None else b["monotonic_trend"]
    if x.name in (b.get("allow_non_monotonic_vars") or []):
        mono = "none"
    if len(xv) == 0 or yv.nunique() < 2:
        if num:
            spec["splits"] = []
        else:
            spec.update(splits=None, groups=[[str(c)] for c in xv.astype(str).unique()])
        return spec
    try:
        from optbinning import OptimalBinning
        kw = dict(name=str(x.name), dtype="numerical" if num else "categorical", solver="cp",
                  max_n_prebins=max(b["initial_fine_bins"], 5), min_bin_size=b["min_bin_pct"],
                  max_n_bins=b["max_coarse_bins"], min_n_bins=b["min_coarse_bins"],
                  monotonic_trend=None if mono in ("none", None) else mono, time_limit=10)
        if num:
            xo = xv
        else:
            xo = _collapse_rare(xv.astype(str), specials)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ob = OptimalBinning(**kw)
            ob.fit(xo.to_numpy(), yv.to_numpy())
        if ob.status not in ("OPTIMAL", "FEASIBLE"):
            raise RuntimeError(ob.status)
        if num:
            spec["splits"] = [float(s) for s in ob.splits]
        else:
            spec["groups"] = [list(map(str, grp)) for grp in ob.splits]
    except Exception:  # noqa: BLE001 – fallback phân vị / mỗi category một nhóm
        if num:
            spec["splits"] = _quantile_splits(xv, min(b["max_coarse_bins"], 5))
        else:
            xo = _collapse_rare(xv.astype(str), specials)
            rate = yv.groupby(xo.to_numpy()).mean().sort_values()
            spec["groups"] = [[str(c)] for c in rate.index]
    if not num:
        spec["splits"] = None
        # xếp nhóm theo bad rate tăng dần để bảng dễ đọc
        tmp = _collapse_rare(xv.astype(str), specials)
        gm = {c: i for i, g in enumerate(spec["groups"]) for c in g}
        rates = [yv[tmp.map(gm).to_numpy() == i].mean() if (tmp.map(gm).to_numpy() == i).any() else 0 for i in range(len(spec["groups"]))]
        spec["groups"] = [spec["groups"][i] for i in np.argsort(rates)]
    return spec


def prepare(x: pd.Series, spec: dict) -> pd.Series:
    """Chuẩn hoá giá trị trước assign (categorical: ép về str, giữ NaN)."""
    if spec["kind"] == "categorical":
        return x.astype(object).where(x.isna(), x.astype(str))
    return x


def fit_all(df: pd.DataFrame, y: pd.Series, features: list[str], cfg: dict, specials: list | None = None, progress=None) -> dict[str, dict]:
    specials = specials if specials is not None else (cfg["cleaning"].get("special_values") or [])
    out = {}
    for i, c in enumerate(features):
        sp = specials if is_numeric(df[c]) else []
        out[c] = fit_variable(df[c], y, cfg, sp)
        if progress:
            progress(i + 1, len(features), c)
    return out


def tables(df: pd.DataFrame, y: pd.Series, specs: dict[str, dict]) -> dict[str, pd.DataFrame]:
    return {c: make_table(prepare(df[c], s), y, s) for c, s in specs.items()}


def iv_summary(tabs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [{"variable": c, "iv": total_iv(t), "n_bins": int((t["kind"] == "normal").sum()), "trend": trend(t)} for c, t in tabs.items()]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def attach_woe(spec: dict, table: pd.DataFrame) -> dict:
    """Gắn bản đồ nhãn→WoE vào spec (dùng cho transform trên mọi tập)."""
    spec = dict(spec)
    spec["woe"] = dict(zip(table["bin"], table["woe"]))
    return spec


def transform(df: pd.DataFrame, specs: dict[str, dict], variables: list[str] | None = None) -> pd.DataFrame:
    """Chuyển sang WoE theo spec đã fit (cần spec['woe']). Nhãn lạ → 0."""
    cols = {}
    for c in (variables or list(specs)):
        s = specs[c]
        lab = assign(prepare(df[c], s), s)
        cols[c] = lab.map(s["woe"]).astype(float).fillna(0.0)
    return pd.DataFrame(cols, index=df.index)


# ------------------------------------------------------------- gợi ý
def suggest(table: pd.DataFrame, cfg: dict, variable: str = "") -> list[str]:
    """Gợi ý hành động theo ngưỡng (không tự áp dụng)."""
    b = cfg["binning"]
    out = []
    normal = table[table["kind"] == "normal"].reset_index(drop=True)
    for i, r in normal.iterrows():
        if r["count_pct"] < b["min_bin_pct"]:
            nb = "bin kế tiếp" if i == 0 else "bin liền trước"
            out.append(f"Gộp bin {i+1} ({r['bin']}) vào {nb}: chỉ {r['count_pct']:.1%} < {b['min_bin_pct']:.0%} mẫu.")
        if r["bad"] < b["min_bin_bads"]:
            out.append(f"Bin {i+1} chỉ có {int(r['bad'])} bad < {b['min_bin_bads']}: kém ổn định.")
    if len(normal) > b["max_coarse_bins"]:
        out.append(f"{len(normal)} bin > tối đa {b['max_coarse_bins']}: nên gộp bớt.")
    tr = trend(table)
    allowed = variable in (b.get("allow_non_monotonic_vars") or [])
    if tr == "none" and b["monotonic_trend"] != "none" and not allowed:
        br = normal["bad_rate"].to_numpy()
        for i in range(1, len(br) - 1):
            if (br[i] - br[i - 1]) * (br[i + 1] - br[i]) < 0:
                out.append(f"Bad rate không đơn điệu tại bin {i+1}: nên gộp bin {i+1} với bin {i+2} (hoặc {i}).")
                break
    iv = total_iv(table)
    if iv > cfg["selection"]["iv"]["suspicious_above"]:
        out.append(f"IV {iv:.2f} > {cfg['selection']['iv']['suspicious_above']}: nghi ngờ rò rỉ/over-predict, kiểm tra biến.")
    return out


# -------------------------------------------------------------- chỉnh tay
def merge_bins(spec: dict, i: int, j: int | None = None) -> dict:
    """Gộp bin i với bin liền sau (numeric: bỏ split i) hoặc nhóm i với nhóm j (categorical)."""
    spec = dict(spec)
    if spec["kind"] == "numeric":
        s = list(spec["splits"])
        if not 0 <= i < len(s):
            raise IndexError("Không có bin liền sau để gộp.")
        s.pop(i)
        spec["splits"] = s
    else:
        g = [list(x) for x in spec["groups"]]
        j = i + 1 if j is None else j
        if i == j or not (0 <= i < len(g) and 0 <= j < len(g)):
            raise IndexError("Chỉ số nhóm không hợp lệ.")
        lo, hi = sorted((i, j))
        g[lo] = g[lo] + g[hi]
        g.pop(hi)
        spec["groups"] = g
    return spec


def split_numeric(spec: dict, value: float) -> dict:
    spec = dict(spec)
    s = sorted(set(list(spec["splits"]) + [float(value)]))
    spec["splits"] = s
    return spec


def split_group(spec: dict, i: int, categories: list[str]) -> dict:
    """Tách một số category khỏi nhóm i thành nhóm mới."""
    spec = dict(spec)
    g = [list(x) for x in spec["groups"]]
    moved = [c for c in g[i] if c in categories]
    if not moved or len(moved) == len(g[i]):
        raise ValueError("Chọn một phần (không phải tất cả) category trong nhóm.")
    g[i] = [c for c in g[i] if c not in moved]
    g.insert(i + 1, moved)
    spec["groups"] = g
    return spec


def set_splits(spec: dict, splits: list[float]) -> dict:
    spec = dict(spec)
    spec["splits"] = sorted(set(float(s) for s in splits))
    return spec
