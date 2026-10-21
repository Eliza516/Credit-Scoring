"""Chỉ số đánh giá: AUC/Gini/KS, chỉ số kiểu ML, tranche, PSI/CSI, xếp hạng theo ngưỡng."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)


def auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def gini(y, s) -> float:
    return 2 * auc(y, s) - 1


def ks(y, s) -> float:
    """KS giữa phân phối điểm của bad và good (s: điểm rủi ro, càng cao càng bad)."""
    d = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(s)}).sort_values("s")
    cb = (d["y"] == 1).cumsum() / max((d["y"] == 1).sum(), 1)
    cg = (d["y"] == 0).cumsum() / max((d["y"] == 0).sum(), 1)
    return float((cb - cg).abs().max())


def pr_auc(y, s) -> float:
    return float(average_precision_score(y, s))


def lift_table(y, s, bins: int = 10) -> pd.DataFrame:
    """Decile theo rủi ro giảm dần (decile 1 = rủi ro cao nhất): bad rate, lift, cumulative capture."""
    d = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(s)})
    d["rank"] = d["s"].rank(method="first", ascending=False)
    d["decile"] = pd.qcut(d["rank"], bins, labels=False) + 1
    g = d.groupby("decile")["y"].agg(["count", "sum"]).rename(columns={"sum": "bad"})
    g["bad_rate"] = g["bad"] / g["count"]
    base = d["y"].mean()
    g["lift"] = g["bad_rate"] / base if base > 0 else np.nan
    g["cum_bad_capture"] = g["bad"].cumsum() / max(g["bad"].sum(), 1)
    return g.reset_index()


def threshold_metrics(y, risk, threshold: float) -> dict:
    """Chỉ số phân loại tại ngưỡng: dự đoán bad nếu risk >= threshold."""
    pred = (np.asarray(risk) >= threshold).astype(int)
    y = np.asarray(y)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "specificity": float(tn / max(tn + fp, 1)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "mcc": float(matthews_corrcoef(y, pred))}


def best_f1_threshold(y, risk) -> float:
    qs = np.unique(np.quantile(risk, np.linspace(0.02, 0.98, 97)))
    best = max(qs, key=lambda t: f1_score(y, (risk >= t).astype(int), zero_division=0))
    return float(best)


def metrics_row(y, risk, cfg: dict, threshold: float | None = None) -> dict:
    """Bộ chỉ số tổng hợp cho 1 mô hình/tập. risk: xác suất hoặc điểm rủi ro (cao = bad)."""
    ml = cfg["evaluation"]["ml_metrics"]
    row = {"auc": auc(y, risk), "gini": gini(y, risk), "ks": ks(y, risk), "pr_auc": pr_auc(y, risk),
           "base_bad_rate": float(np.mean(y))}
    lt = lift_table(y, risk)
    row["lift_top_decile"] = float(lt.loc[0, "lift"])
    if threshold is None:
        mode = ml["decision_threshold"]["mode"]
        threshold = ml["decision_threshold"]["fixed_value"] if mode == "fixed" else best_f1_threshold(y, risk)
    row.update(threshold_metrics(y, risk, threshold))
    return row


def rate(value: float, cuts: dict, kind: str) -> str:
    """Xếp hạng theo ngưỡng cấu hình. kind: ks | gini | auc (cùng thang với cfg['evaluation'][kind])."""
    if kind == "ks":
        v = value * 100
        return "mạnh" if v > cuts["strong_above"] else "chấp nhận" if v >= cuts["acceptable_min"] else "yếu"
    v = value * 100 if kind == "gini" else value
    if v > cuts["excellent_above"]:
        return "xuất sắc"
    if v >= cuts["good_min"]:
        return "tốt"
    if v >= cuts["acceptable_min"]:
        return "chấp nhận"
    return "yếu"


def assessment(metrics_by_set: dict[str, dict], cfg: dict) -> pd.DataFrame:
    """Bảng chỉ số + xếp hạng + cảnh báo theo ngưỡng + khoảng cách train–test."""
    ev, ml = cfg["evaluation"], cfg["evaluation"]["ml_metrics"]
    rows = []
    for name, m in metrics_by_set.items():
        rows.append({"set": name, "auc": m["auc"], "auc_rating": rate(m["auc"], ev["auc"], "auc"),
                     "gini": m["gini"], "gini_rating": rate(m["gini"], ev["gini"], "gini"),
                     "ks": m["ks"], "ks_rating": rate(m["ks"], ev["ks"], "ks"),
                     "pr_auc": m["pr_auc"], "lift_top_decile": m["lift_top_decile"]})
    df = pd.DataFrame(rows)
    warns = []
    if "train" in metrics_by_set and "test" in metrics_by_set:
        a, b = metrics_by_set["train"], metrics_by_set["test"]
        if a["auc"] - b["auc"] > ml["overfit"]["max_auc_gap_train_test"]:
            warns.append(f"Chênh AUC train–test {a['auc']-b['auc']:.3f} > {ml['overfit']['max_auc_gap_train_test']} → nghi overfit.")
        if (a["gini"] - b["gini"]) * 100 > cfg["split"]["max_train_test_gap_gini_pp"]:
            warns.append(f"Chênh Gini train–test {(a['gini']-b['gini'])*100:.1f}pp > {cfg['split']['max_train_test_gap_gini_pp']}pp.")
    for name, m in metrics_by_set.items():
        if name == "train":
            continue
        if m["pr_auc"] < ml["thresholds"]["min_pr_auc_lift_over_base_rate"] * m["base_bad_rate"]:
            warns.append(f"[{name}] PR-AUC {m['pr_auc']:.3f} < {ml['thresholds']['min_pr_auc_lift_over_base_rate']}× bad rate nền.")
        if m["lift_top_decile"] < ml["lift"]["top_decile_min"]:
            warns.append(f"[{name}] Lift decile cao nhất {m['lift_top_decile']:.2f} < {ml['lift']['top_decile_min']}.")
        for key, lab in (("min_recall_at_cutoff", "recall"), ("min_precision_at_cutoff", "precision")):
            thr = ml["thresholds"].get(key)
            if thr is not None and m[lab] < thr:
                warns.append(f"[{name}] {lab} {m[lab]:.2f} < {thr}.")
    df.attrs["warnings"] = warns
    return df


def tranche_table(y, score, n: int = 10, higher_is_safer: bool = True) -> pd.DataFrame:
    """Tranche A..J: A = rủi ro cao nhất. Kiểm tra bad rate đơn điệu."""
    d = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score)})
    key = d["score"] if higher_is_safer else -d["score"]
    d["tr"] = pd.qcut(key.rank(method="first"), n, labels=False)
    g = d.groupby("tr").agg(score_min=("score", "min"), score_max=("score", "max"), count=("y", "size"), bad=("y", "sum"))
    g["good"] = g["count"] - g["bad"]
    g["bad_rate"] = g["bad"] / g["count"]
    g["tranche"] = [chr(ord("A") + i) for i in range(len(g))]
    g["cum_bad_pct"] = g["bad"].cumsum() / g["bad"].sum()
    g["cum_good_pct"] = g["good"].cumsum() / g["good"].sum()
    g["ks_gap"] = (g["cum_bad_pct"] - g["cum_good_pct"]).abs()
    out = g.reset_index(drop=True)
    out.attrs["monotonic"] = bool((out["bad_rate"].diff().dropna() <= 1e-12).all())
    return out[["tranche", "score_min", "score_max", "count", "bad", "good", "bad_rate", "cum_bad_pct", "cum_good_pct", "ks_gap"]]


def psi(expected, actual, bins: int = 10) -> float:
    e, a = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    cuts = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts[0], cuts[-1] = -np.inf, np.inf
    pe = np.histogram(e, cuts)[0] / len(e)
    pa = np.histogram(a, cuts)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def csi_table(bins_expected: pd.DataFrame, bins_actual: pd.DataFrame) -> pd.DataFrame:
    """CSI từng biến từ 2 DataFrame nhãn bin (mỗi cột = 1 biến)."""
    rows = []
    for c in bins_expected.columns:
        pe = bins_expected[c].value_counts(normalize=True)
        pa = bins_actual[c].value_counts(normalize=True)
        idx = pe.index.union(pa.index)
        pe, pa = pe.reindex(idx).fillna(0).clip(lower=1e-6), pa.reindex(idx).fillna(0).clip(lower=1e-6)
        rows.append({"variable": c, "csi": float(((pa - pe) * np.log(pa / pe)).sum())})
    return pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)


def stability_flag(value: float, green_below: float, red_above: float) -> str:
    return "xanh" if value < green_below else "đỏ" if value > red_above else "vàng"


def compare_models(rows: dict[str, dict], cfg: dict) -> pd.DataFrame:
    """Bảng so sánh nhiều mô hình (mỗi value = metrics_row trên Test) + cờ cảnh báo."""
    df = pd.DataFrame(rows).T
    cols = [c for c in ["gini", "auc", "ks", "pr_auc", "lift_top_decile", "precision", "recall", "f1", "mcc"] if c in df.columns]
    df = df[cols]
    keys = cfg["evaluation"]["ml_metrics"]["compare_models"]["rank_by"]
    df["rank"] = df[keys].rank(ascending=False).mean(axis=1).rank(method="min").astype(int)
    return df.sort_values("rank")


def ml_vs_primary_warnings(rows: dict[str, dict], primary: str, cfg: dict) -> list[str]:
    out = []
    gap_ml = cfg["modeling"]["challengers"]["max_gini_gap_vs_primary_pp"]
    gap_primary = cfg["evaluation"]["ml_metrics"]["compare_models"]["require_primary_not_worse_than_challenger_pp"]
    p = rows[primary]["gini"] * 100
    for k, v in rows.items():
        if k == primary:
            continue
        d = v["gini"] * 100 - p
        if d > min(gap_ml, gap_primary):
            out.append(f"{k} hơn mô hình chính {d:.1f}pp Gini > {min(gap_ml, gap_primary)}pp → có thể logistic bỏ sót phi tuyến/tương tác.")
    return out
