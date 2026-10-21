"""Gợi ý phân tầng từ dữ liệu: so sánh bad rate theo nhóm + cây nông + ước lượng Gini tăng thêm."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _groups(s: pd.Series, max_groups: int = 6) -> pd.Series:
    """Nhóm hoá biến ứng viên: categorical giữ nguyên (gộp hiếm), numeric chia phân vị."""
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > max_groups:
        try:
            return pd.qcut(s, 4, duplicates="drop").astype(str).where(s.notna(), "Missing")
        except ValueError:
            return s.astype(str)
    return s.astype(object).where(s.notna(), "Missing").astype(str)


def _design(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            X[c] = s.fillna(s.median())
            X[c + "__na"] = s.isna().astype(float)
        else:
            top = s.astype(str).value_counts().index[:8]
            for t in top:
                X[f"{c}={t}"] = (s.astype(str) == t).astype(float)
    return X.loc[:, X.nunique() > 1]


def _fit_gini(Xtr, ytr, Xte, yte) -> tuple[float, np.ndarray | None]:
    if ytr.nunique() < 2 or yte.nunique() < 2 or Xtr.shape[1] == 0:
        return np.nan, None
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=300).fit(sc.transform(Xtr), ytr)
    p = m.predict_proba(sc.transform(Xte))[:, 1]
    return 2 * roc_auc_score(yte, p) - 1, p


def suggest(df: pd.DataFrame, y: pd.Series, features: list[str], cfg: dict, train_idx, test_idx) -> dict:
    """Trả {'candidates': DataFrame, 'tree': DataFrame, 'baseline_gini': float}.

    Với mỗi biến ứng viên: tách nhóm, kiểm kích thước/bad tối thiểu/chênh bad rate, ước Gini toàn cục vs Gini khi mỗi nhóm
    một mô hình logistic đơn giản (đánh giá trên Test).
    """
    sg = cfg["segmentation"]
    cands = sg["candidate_features"] or [c for c in features if df[c].nunique() <= 12]
    base_feats = [c for c in features]
    Xall = _design(df.loc[:, base_feats], base_feats)
    g0, p0 = _fit_gini(Xall.loc[train_idx], y.loc[train_idx], Xall.loc[test_idx], y.loc[test_idx])
    rows = []
    for c in cands:
        if c not in df.columns:
            continue
        g = _groups(df[c])
        tr_g, te_g = g.loc[train_idx], g.loc[test_idx]
        vc = tr_g.value_counts(normalize=True)
        stats = []
        for val in vc.index:
            ix = tr_g.index[tr_g == val]
            stats.append((val, len(ix) / len(tr_g), int((y.loc[ix] == 1).sum()), float(y.loc[ix].mean())))
        sd = pd.DataFrame(stats, columns=["group", "size_pct", "bads", "bad_rate"])
        ok = sd[(sd["size_pct"] >= sg["min_segment_size_pct"]) & (sd["bads"] >= sg["min_bads_per_segment"])]
        ratio = (ok["bad_rate"].max() / ok["bad_rate"].min()) if len(ok) >= 2 and ok["bad_rate"].min() > 0 else np.nan
        gain = np.nan
        if len(ok) >= 2 and len(ok) <= sg["max_segments"]:
            preds = pd.Series(np.nan, index=test_idx)
            used = set(ok["group"])
            feats_c = [f for f in base_feats if f != c]
            Xc = _design(df.loc[:, feats_c], feats_c)
            for val in ok["group"]:
                tri = tr_g.index[tr_g == val]
                tei = te_g.index[te_g == val]
                _, p = _fit_gini(Xc.loc[tri], y.loc[tri], Xc.loc[tei], y.loc[tei]) if len(tei) else (np.nan, None)
                if p is not None:
                    preds.loc[tei] = p
            # nhóm nhỏ không đủ điều kiện dùng mô hình toàn cục
            miss = preds.index[preds.isna()]
            if len(miss) and p0 is not None:
                preds.loc[miss] = pd.Series(p0, index=test_idx).loc[miss]
            if y.loc[test_idx].nunique() == 2:
                gain = (2 * roc_auc_score(y.loc[test_idx], preds) - 1 - g0) * 100
        sugg = "không tách"
        reason = ""
        if len(ok) < 2:
            reason = "không đủ nhóm thoả cỡ/bad tối thiểu"
        elif len(ok) > sg["max_segments"]:
            reason = f"{len(ok)} nhóm > tối đa {sg['max_segments']}"
        elif not np.isnan(ratio) and ratio < sg["min_bad_rate_ratio"]:
            reason = f"chênh bad rate {ratio:.2f}x < {sg['min_bad_rate_ratio']}x"
        elif np.isnan(gain) or gain < sg["suggest_if_gini_gain_pp"]:
            reason = f"Gini tăng {gain:.1f}pp < {sg['suggest_if_gini_gain_pp']}pp" if not np.isnan(gain) else "không ước được Gini"
        else:
            sugg, reason = "nên tách", f"chênh bad rate {ratio:.2f}x, Gini +{gain:.1f}pp"
        rows.append({"feature": c, "n_groups_ok": len(ok), "bad_rate_ratio": ratio, "gini_gain_pp": gain,
                     "suggestion": sugg, "reason": reason})
    cand_df = pd.DataFrame(rows)
    if len(cand_df):
        cand_df = cand_df.sort_values(["suggestion", "gini_gain_pp"], ascending=[False, False]).reset_index(drop=True)
    return {"candidates": cand_df, "baseline_gini": g0 * 100 if not np.isnan(g0) else np.nan, "tree": tree_rules(df, y, cands, cfg, train_idx)}


def tree_rules(df: pd.DataFrame, y: pd.Series, cands: list[str], cfg: dict, train_idx) -> pd.DataFrame:
    """Cây quyết định nông trên biến ứng viên → các lá = nhóm gợi ý."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    sg = cfg["segmentation"]
    cands = [c for c in cands if c in df.columns]
    if not cands:
        return pd.DataFrame(columns=["leaf", "size_pct", "bad_rate", "bads"])
    X = _design(df.loc[train_idx, cands], cands)
    yy = y.loc[train_idx]
    dt = DecisionTreeClassifier(max_depth=sg["tree_max_depth"], min_samples_leaf=max(int(sg["min_segment_size_pct"] * len(X)), 1),
                                random_state=cfg.get("seed", 42)).fit(X, yy)
    leaves = dt.apply(X)
    rules = export_text(dt, feature_names=list(X.columns)).splitlines()
    out = []
    for lf in np.unique(leaves):
        m = leaves == lf
        out.append({"leaf": int(lf), "size_pct": float(m.mean()), "bad_rate": float(yy[m].mean()), "bads": int(yy[m].sum())})
    res = pd.DataFrame(out).sort_values("bad_rate", ascending=False).reset_index(drop=True)
    res.attrs["tree_text"] = "\n".join(rules)
    return res


def assign_segments(df: pd.DataFrame, feature: str, groups: dict[str, list[str]] | None = None) -> pd.Series:
    """Gán tầng theo feature (mỗi giá trị = 1 tầng) hoặc theo ánh xạ nhóm {tên_tầng: [giá trị...]}."""
    g = _groups(df[feature])
    if not groups:
        return g
    inv = {v: k for k, vs in groups.items() for v in vs}
    return g.map(inv).fillna("Other")
