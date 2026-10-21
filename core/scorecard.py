"""Quy đổi điểm (base score / odds / PDO), bảng điểm, cutoff & vùng quyết định."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import binning, evaluation


def scaling(cfg: dict) -> tuple[float, float]:
    """(factor, offset): score = offset + factor·ln(odds good/bad)."""
    sc = cfg["scorecard"]
    factor = sc["pdo"] / np.log(2)
    offset = sc["base_score"] - factor * np.log(sc["base_odds_good_to_bad"])
    return float(factor), float(offset)


def build_table(model: dict, specs: dict, tabs: dict, cfg: dict) -> pd.DataFrame:
    """Bảng điểm: mỗi bin một dòng với WoE, hệ số và điểm. Intercept chia đều cho các biến."""
    factor, offset = scaling(cfg)
    vars_ = model["variables"]
    n = len(vars_)
    b0 = model["intercept"] + model["prior_offset"]
    rows = []
    for v in vars_:
        t = tabs[v]
        for _, r in t.iterrows():
            pts = -(model["coef"][v] * r["woe"] + b0 / n) * factor + offset / n
            rows.append({"variable": v, "bin": r["bin"], "kind": r["kind"], "count_pct": r["count_pct"], "bad_rate": r["bad_rate"],
                         "woe": r["woe"], "coef": model["coef"][v], "points": pts})
    tab = pd.DataFrame(rows)
    tab["points"] = tab["points"].round(cfg["scorecard"]["rounding"])
    return tab


def score(df: pd.DataFrame, model: dict, specs: dict, points: pd.DataFrame, cfg: dict, clip: bool = True) -> pd.Series:
    """Điểm cho từng dòng bằng tra bảng điểm (nhãn lạ → điểm trung bình biến)."""
    total = pd.Series(0.0, index=df.index)
    for v in model["variables"]:
        lab = binning.assign(binning.prepare(df[v], specs[v]), specs[v])
        pv = points[points["variable"] == v].set_index("bin")["points"]
        total = total + lab.map(pv).fillna(pv.mean())
    if clip:
        lo, hi = cfg["scorecard"]["score_range"]
        total = total.clip(lo, hi)
    return total


def score_from_logit(logit_bad: np.ndarray, cfg: dict) -> np.ndarray:
    factor, offset = scaling(cfg)
    return offset - factor * np.asarray(logit_bad)


def cutoff_table(y: pd.Series, s: pd.Series, n_points: int = 40) -> pd.DataFrame:
    """Với mỗi cutoff (duyệt điểm ≥ cutoff): tỷ lệ duyệt, bad rate của nhóm duyệt/từ chối, bad bắt được."""
    y = pd.Series(np.asarray(y)); s = pd.Series(np.asarray(s))
    cuts = np.unique(np.quantile(s, np.linspace(0.01, 0.99, n_points)))
    B, G = (y == 1).sum(), (y == 0).sum()
    rows = []
    for c in cuts:
        ap = s >= c
        rows.append({"cutoff": float(c), "approval_rate": float(ap.mean()), "bad_rate_approved": float(y[ap].mean()) if ap.any() else np.nan,
                     "bad_rate_rejected": float(y[~ap].mean()) if (~ap).any() else np.nan,
                     "bads_rejected_pct": float(((~ap) & (y == 1)).sum() / B), "goods_rejected_pct": float(((~ap) & (y == 0)).sum() / G)})
    return pd.DataFrame(rows)


def suggest_cutoff(ct: pd.DataFrame, cfg: dict, y: pd.Series | None = None, s: pd.Series | None = None) -> dict:
    """Gợi ý cutoff theo chiến lược trong config. Trả {'cutoff', 'message'} (cutoff None nếu thiếu tham số)."""
    c = cfg["cutoff"]
    strat = c["strategy"]
    if strat == "target_bad_rate":
        t = c["target_bad_rate"]
        if t is None:
            return {"cutoff": None, "message": "Chưa đặt cutoff.target_bad_rate – bạn cần chốt."}
        ok = ct[ct["bad_rate_approved"] <= t]
        if ok.empty:
            return {"cutoff": None, "message": f"Không cutoff nào đạt bad rate duyệt ≤ {t:.1%}."}
        r = ok.sort_values("approval_rate", ascending=False).iloc[0]
        return {"cutoff": float(r["cutoff"]), "message": f"Cutoff {r['cutoff']:.0f}: duyệt {r['approval_rate']:.1%}, bad rate duyệt {r['bad_rate_approved']:.2%} ≤ {t:.1%}."}
    if strat == "target_approval_rate":
        t = c["target_approval_rate"]
        if t is None:
            return {"cutoff": None, "message": "Chưa đặt cutoff.target_approval_rate – bạn cần chốt."}
        r = ct.iloc[(ct["approval_rate"] - t).abs().argmin()]
        return {"cutoff": float(r["cutoff"]), "message": f"Cutoff {r['cutoff']:.0f}: duyệt {r['approval_rate']:.1%} (mục tiêu {t:.1%}), bad rate duyệt {r['bad_rate_approved']:.2%}."}
    if strat == "profit":
        pg, lb = c.get("profit_per_good"), c.get("loss_per_bad")
        if pg is None or lb is None or y is None or s is None:
            return {"cutoff": None, "message": "Cần cutoff.profit_per_good và cutoff.loss_per_bad – bạn cần chốt."}
        best = None
        for cut in ct["cutoff"]:
            ap = s >= cut
            val = pg * ((y[ap] == 0).sum()) - lb * ((y[ap] == 1).sum())
            if best is None or val > best[1]:
                best = (cut, val)
        return {"cutoff": float(best[0]), "message": f"Cutoff {best[0]:.0f} tối đa lợi nhuận kỳ vọng = {best[1]:,.0f}."}
    return {"cutoff": None, "message": f"Chiến lược '{strat}' không hợp lệ."}


def decision_zones(s: pd.Series, cutoff: float, cfg: dict) -> tuple[float, float]:
    """Vùng xét tay quanh cutoff chiếm ~manual_review_band_width_pct % dân số: trả (reject_below, approve_from)."""
    w = cfg["cutoff"]["manual_review_band_width_pct"] / 100
    p_cut = float((s < cutoff).mean())
    lo = float(np.quantile(s, max(p_cut - w / 2, 0)))
    hi = float(np.quantile(s, min(p_cut + w / 2, 1)))
    return lo, hi


def apply_zones(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return pd.Series(np.where(s >= hi, "approve", np.where(s < lo, "reject", "manual_review")), index=s.index)


def zone_summary(y: pd.Series, zone: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"zone": zone.to_numpy(), "y": np.asarray(y)})
    g = d.groupby("zone")["y"].agg(["count", "sum"]).rename(columns={"sum": "bad"})
    g["pct"] = g["count"] / g["count"].sum()
    g["bad_rate"] = g["bad"] / g["count"]
    return g.reindex(["approve", "manual_review", "reject"]).dropna(how="all").reset_index()
