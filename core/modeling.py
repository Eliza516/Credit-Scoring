"""Logistic scorecard (WoE) + stepwise + mô hình đối chứng DT/RF/LightGBM."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import evaluation, imbalance


def _glm(X: pd.DataFrame, y: pd.Series, w: pd.Series | None):
    Xc = sm.add_constant(X, has_constant="add")
    kw = {"freq_weights": w.to_numpy()} if w is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.GLM(y.to_numpy(), Xc, family=sm.families.Binomial(), **kw).fit()


def _crit(res, criterion: str) -> float:
    return {"aic": res.aic, "bic": res.bic_llf}.get(criterion, res.aic)


def _sign_ok(res, variables: list[str], expected: str, require: bool) -> bool:
    if not require:
        return True
    p = res.params
    for v in variables:
        if (expected == "negative" and p[v] > 0) or (expected == "positive" and p[v] < 0):
            return False
    return True


def stepwise(W: pd.DataFrame, y: pd.Series, cfg: dict, candidates: list[str] | None = None,
             forced: list[str] | None = None, sample_weight: pd.Series | None = None) -> dict:
    """Stepwise forward/backward/both theo p-value hoặc AIC/BIC. Trả {'selected': [...], 'steps': DataFrame}."""
    st = cfg["modeling"]["stepwise"]
    cand = list(candidates if candidates is not None else W.columns)
    forced = [v for v in (forced or []) if v in cand]
    direction, crit = st["direction"], st["criterion"]
    pe, pr = st["p_enter"], st["p_remove"]
    exp, req = st["expected_sign_on_woe"], st["require_expected_sign"]
    maxv = st["max_variables"]
    steps = []

    def add_row(action, var, detail):
        steps.append({"step": len(steps) + 1, "action": action, "variable": var, "detail": detail})

    selected = list(forced) if direction != "backward" else list(cand)
    if direction == "backward":
        selected = list(cand)
    cur = _glm(W[selected], y, sample_weight) if selected else None
    for _ in range(4 * max(len(cand), 1)):
        changed = False
        # --- thêm
        if direction in ("forward", "both") and len(selected) < maxv:
            best = None
            pool = [v for v in cand if v not in selected]
            for v in pool:
                res = _glm(W[selected + [v]], y, sample_weight)
                if not _sign_ok(res, selected + [v], exp, req):
                    continue
                if crit == "pvalue":
                    score = res.pvalues[v]
                    if score < pe and (best is None or score < best[1]):
                        best = (v, score, res)
                else:
                    score = _crit(res, crit)
                    base = _crit(cur, crit) if cur is not None else np.inf
                    if score < base - 1e-9 and (best is None or score < best[1]):
                        best = (v, score, res)
            if best:
                selected.append(best[0])
                cur = best[2]
                add_row("add", best[0], f"{'p' if crit=='pvalue' else crit.upper()}={best[1]:.4g}")
                changed = True
        # --- loại
        if direction in ("backward", "both") and selected:
            cur = _glm(W[selected], y, sample_weight)
            removable = [v for v in selected if v not in forced]
            drop = None
            if crit == "pvalue":
                bad_sign = [v for v in removable if req and ((exp == "negative" and cur.params[v] > 0) or (exp == "positive" and cur.params[v] < 0))]
                if bad_sign:
                    drop = (bad_sign[0], f"dấu hệ số sai ({cur.params[bad_sign[0]]:+.3f})")
                else:
                    worst = max(removable, key=lambda v: cur.pvalues[v], default=None)
                    if worst and cur.pvalues[worst] > pr:
                        drop = (worst, f"p={cur.pvalues[worst]:.4g} > {pr}")
            else:
                base = _crit(cur, crit)
                bestv = None
                for v in removable:
                    r2 = _glm(W[[u for u in selected if u != v]], y, sample_weight) if len(selected) > 1 else None
                    if r2 is not None and _crit(r2, crit) < base - 1e-9 and (bestv is None or _crit(r2, crit) < bestv[1]):
                        bestv = (v, _crit(r2, crit))
                if bestv:
                    drop = (bestv[0], f"{crit.upper()} giảm còn {bestv[1]:.2f}")
            if drop:
                selected.remove(drop[0])
                cur = _glm(W[selected], y, sample_weight) if selected else None
                add_row("remove", drop[0], drop[1])
                changed = True
        if not changed:
            break
    notes = []
    if len(selected) < st["min_variables"]:
        notes.append(f"Chỉ {len(selected)} biến < tối thiểu khuyến nghị {st['min_variables']}.")
    return {"selected": selected, "steps": pd.DataFrame(steps, columns=["step", "action", "variable", "detail"]), "notes": notes}


def fit_logistic(W: pd.DataFrame, y: pd.Series, variables: list[str], cfg: dict, sample_weight: pd.Series | None = None,
                 prior_offset: float = 0.0) -> dict:
    """Logistic trên WoE: logit(P(bad)) = b0 + Σ b_i·WoE_i (+ prior_offset vào intercept khi dự báo PD)."""
    m = cfg["modeling"]["logistic"]
    exp = cfg["modeling"]["stepwise"]["expected_sign_on_woe"]
    if m["penalty"] == "l2":
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=m["C"], max_iter=m["max_iter"]).fit(W[variables], y, sample_weight=None if sample_weight is None else sample_weight.to_numpy())
        coef = pd.Series(lr.coef_[0], index=variables)
        table = pd.DataFrame({"coef": coef, "std_err": np.nan, "z": np.nan, "p_value": np.nan})
        b0 = float(lr.intercept_[0])
    else:
        res = _glm(W[variables], y, sample_weight)
        coef = res.params.drop("const")
        table = pd.DataFrame({"coef": coef, "std_err": res.bse.drop("const"), "z": res.tvalues.drop("const"),
                              "p_value": res.pvalues.drop("const")})
        b0 = float(res.params["const"])
    table["sign_ok"] = (table["coef"] < 0) if exp == "negative" else (table["coef"] > 0)
    return {"variables": list(variables), "intercept": b0, "coef": table["coef"].to_dict(), "table": table,
            "prior_offset": float(prior_offset)}


def predict_logit(model: dict, W: pd.DataFrame, with_prior: bool = True) -> np.ndarray:
    z = model["intercept"] + (model["prior_offset"] if with_prior else 0.0)
    for v in model["variables"]:
        z = z + model["coef"][v] * W[v].to_numpy()
    return np.asarray(z)


def predict_proba(model: dict, W: pd.DataFrame, with_prior: bool = True) -> np.ndarray:
    return 1 / (1 + np.exp(-predict_logit(model, W, with_prior)))


def prior_offset_for(y_orig: pd.Series, y_used: pd.Series, w_used: pd.Series | None) -> float:
    """Offset đưa PD về bad rate gốc khi mẫu train bị cân bằng lại."""
    rate_true = float(y_orig.mean())
    rate_dev = float(np.average(y_used, weights=w_used)) if w_used is not None else float(y_used.mean())
    if abs(rate_dev - rate_true) < 1e-9:
        return 0.0
    return imbalance.prior_offset(rate_dev, rate_true)


def fit_challengers(W: pd.DataFrame, y: pd.Series, variables: list[str], cfg: dict, sample_weight: pd.Series | None = None) -> dict:
    """Huấn luyện DT/RF/LightGBM (chỉ đối chứng) trên các WoE đã chọn."""
    ch = cfg["modeling"]["challengers"]
    seed = cfg.get("seed", 42)
    sw = None if sample_weight is None else sample_weight.to_numpy()
    X = W[variables]
    out = {}
    if ch["decision_tree"]["enabled"]:
        from sklearn.tree import DecisionTreeClassifier
        out["decision_tree"] = DecisionTreeClassifier(random_state=seed, **ch["decision_tree"]["params"]).fit(X, y, sample_weight=sw)
    if ch["random_forest"]["enabled"]:
        from sklearn.ensemble import RandomForestClassifier
        out["random_forest"] = RandomForestClassifier(random_state=seed, **ch["random_forest"]["params"]).fit(X, y, sample_weight=sw)
    if ch["lightgbm"]["enabled"]:
        import lightgbm as lgb
        out["lightgbm"] = lgb.LGBMClassifier(random_state=seed, verbose=-1, **ch["lightgbm"]["params"]).fit(X, y, sample_weight=sw)
    return out


def shap_importance(model, X: pd.DataFrame, max_rows: int = 3000) -> pd.Series | None:
    try:
        import shap
        Xs = X.sample(min(len(X), max_rows), random_state=0)
        sv = shap.TreeExplainer(model).shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    except Exception:  # noqa: BLE001
        return None


def compare_imbalance(W_tr: pd.DataFrame, y_tr: pd.Series, W_te: pd.DataFrame, y_te: pd.Series, variables: list[str],
                      cfg: dict, methods: list[str] | None = None, target_bad_rate: float | None = None) -> pd.DataFrame:
    """So sánh các phương án mất cân bằng cạnh nhau (fit logistic trên train đã xử lý, đo trên Test gốc)."""
    sp = cfg["split"]["imbalance"]
    methods = methods or sp["allowed_methods"]
    r = target_bad_rate or sp.get("target_bad_rate") or 0.3
    rows = []
    for mth in methods:
        try:
            Xr, yr, w = imbalance.resample(W_tr[variables], y_tr, mth, r, cfg.get("seed", 42))
            model = fit_logistic(Xr, yr, variables, cfg, w)
            s = predict_logit(model, W_te, with_prior=False)
            m = evaluation.metrics_row(y_te.to_numpy(), s, cfg)
            rows.append({"method": mth, "train_rows": len(yr), "train_bad_rate": float(np.average(yr, weights=w) if w is not None else yr.mean()),
                         "gini_test": m["gini"], "ks_test": m["ks"], "pr_auc_test": m["pr_auc"], "recall": m["recall"], "precision": m["precision"]})
        except Exception as e:  # noqa: BLE001
            rows.append({"method": mth, "error": str(e)[:80]})
    return pd.DataFrame(rows)
