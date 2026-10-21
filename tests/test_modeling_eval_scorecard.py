import numpy as np
import pandas as pd
import pytest

from core import binning, config, evaluation, export, imbalance, modeling, scorecard, split, store


@pytest.fixture(scope="module")
def built(dev, mapping, cfg):
    d, y = dev
    parts = split.split_indices(d, y, cfg, mapping["open_date"])
    tr, te = parts["train"], parts["test"]
    feats = ["age", "income_mil", "dti", "credit_util", "employment_years", "inquiries_6m", "delinq_12m", "term_months", "channel"]
    specs = binning.fit_all(d.loc[tr], y.loc[tr], feats, cfg)
    tabs = binning.tables(d.loc[tr], y.loc[tr], specs)
    specs = {c: binning.attach_woe(specs[c], tabs[c]) for c in specs}
    Wtr, Wte = binning.transform(d.loc[tr], specs), binning.transform(d.loc[te], specs)
    return d, y, tr, te, specs, tabs, Wtr, Wte


@pytest.mark.parametrize("direction", ["forward", "backward", "both"])
@pytest.mark.parametrize("crit", ["pvalue", "aic", "bic"])
def test_stepwise(built, cfg, direction, crit):
    d, y, tr, te, specs, tabs, Wtr, Wte = built
    c = config.clone(cfg)
    c["modeling"]["stepwise"].update(direction=direction, criterion=crit)
    r = modeling.stepwise(Wtr, y.loc[tr], c)
    sel = r["selected"]
    assert 1 <= len(sel) <= c["modeling"]["stepwise"]["max_variables"]
    assert "term_months" not in sel                       # nhiễu
    m = modeling.fit_logistic(Wtr, y.loc[tr], sel, c)
    assert m["table"]["sign_ok"].all()                    # dấu đúng
    if crit == "pvalue" and direction != "forward":
        assert (m["table"]["p_value"] <= c["modeling"]["stepwise"]["p_remove"]).all()
    assert set(r["steps"]["action"]) <= {"add", "remove"}


def test_stepwise_forced_and_max(built, cfg):
    d, y, tr, te, specs, tabs, Wtr, Wte = built
    c = config.clone(cfg)
    c["modeling"]["stepwise"]["max_variables"] = 3
    r = modeling.stepwise(Wtr, y.loc[tr], c, forced=["term_months"])
    assert "term_months" in r["selected"] and len(r["selected"]) <= 3


def test_logistic_predict_and_prior(built, cfg):
    d, y, tr, te, specs, tabs, Wtr, Wte = built
    sel = ["age", "dti", "income_mil"]
    m = modeling.fit_logistic(Wtr, y.loc[tr], sel, cfg)
    p = modeling.predict_proba(m, Wte)
    assert ((p > 0) & (p < 1)).all()
    assert abs(p.mean() - y.loc[te].mean()) < 0.03
    # resample + prior correction đưa PD về bad rate gốc
    Xr, yr, w = imbalance.resample(Wtr[sel], y.loc[tr], "undersample_random", 0.5, 1)
    off = modeling.prior_offset_for(y.loc[tr], yr, w)
    m2 = modeling.fit_logistic(Xr, yr, sel, cfg, w, prior_offset=off)
    assert abs(modeling.predict_proba(m2, Wte).mean() - y.loc[te].mean()) < 0.03
    assert modeling.predict_proba(m2, Wte, with_prior=False).mean() > 0.4


def test_l2_logistic(built, cfg):
    d, y, tr, te, specs, tabs, Wtr, Wte = built
    c = config.clone(cfg); c["modeling"]["logistic"]["penalty"] = "l2"
    m = modeling.fit_logistic(Wtr, y.loc[tr], ["age", "dti"], c)
    assert m["table"]["p_value"].isna().all() and (m["table"]["coef"] < 0).all()


def test_challengers_and_compare(built, cfg):
    d, y, tr, te, specs, tabs, Wtr, Wte = built
    sel = ["age", "dti", "income_mil", "credit_util"]
    ch = modeling.fit_challengers(Wtr, y.loc[tr], sel, cfg)
    assert set(ch) == {"decision_tree", "random_forest", "lightgbm"}
    m = modeling.fit_logistic(Wtr, y.loc[tr], sel, cfg)
    rows = {"logistic": evaluation.metrics_row(y.loc[te].to_numpy(), modeling.predict_logit(m, Wte), cfg)}
    for k, v in ch.items():
        rows[k] = evaluation.metrics_row(y.loc[te].to_numpy(), v.predict_proba(Wte[sel])[:, 1], cfg)
    cmp_ = evaluation.compare_models(rows, cfg)
    assert cmp_["rank"].min() == 1 and len(cmp_) == 4
    imp = modeling.shap_importance(ch["lightgbm"], Wte[sel])
    assert imp is None or set(imp.index) == set(sel)
    assert isinstance(evaluation.ml_vs_primary_warnings(rows, "logistic", cfg), list)


def test_compare_imbalance(built, cfg):
    d, y, tr, te, specs, tabs, Wtr, Wte = built
    t = modeling.compare_imbalance(Wtr, y.loc[tr], Wte, y.loc[te], ["age", "dti"], cfg, ["none", "undersample_random", "sample_weight"], 0.4)
    assert list(t["method"]) == ["none", "undersample_random", "sample_weight"] and t["gini_test"].between(0.1, 0.9).all()


def test_evaluation_metrics_known():
    y = np.array([0, 0, 0, 1, 1, 1]); s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert evaluation.auc(y, s) == 1 and evaluation.gini(y, s) == 1 and evaluation.ks(y, s) == 1
    tm = evaluation.threshold_metrics(y, s, 0.5)
    assert tm["precision"] == 1 and tm["recall"] == 1 and tm["f1"] == 1 and tm["tp"] == 3 and tm["tn"] == 3 and tm["mcc"] == 1
    assert evaluation.gini(y, -s) == -1
    rng = np.random.default_rng(1)
    r = rng.random(1000)
    assert abs(evaluation.auc((rng.random(1000) < .2).astype(int), r) - 0.5) < 0.08


def test_rating_thresholds(cfg):
    ev = cfg["evaluation"]
    assert evaluation.rate(0.70, ev["auc"], "auc") == "chấp nhận" and evaluation.rate(0.74, ev["auc"], "auc") == "tốt"
    assert evaluation.rate(0.65, ev["gini"], "gini") == "xuất sắc"
    assert evaluation.rate(0.50, ev["gini"], "gini") == "tốt"
    assert evaluation.rate(0.40, ev["gini"], "gini") == "chấp nhận"
    assert evaluation.rate(0.30, ev["gini"], "gini") == "yếu"
    assert evaluation.rate(0.45, ev["ks"], "ks") == "mạnh" and evaluation.rate(0.35, ev["ks"], "ks") == "chấp nhận" and evaluation.rate(0.15, ev["ks"], "ks") == "yếu"
    assert evaluation.rate(0.82, ev["auc"], "auc") == "xuất sắc" and evaluation.rate(0.60, ev["auc"], "auc") == "yếu"


def test_assessment_warns_overfit(cfg):
    a = {"auc": 0.9, "gini": 0.8, "ks": 0.6, "pr_auc": 0.5, "lift_top_decile": 3.0, "base_bad_rate": 0.1}
    b = {"auc": 0.7, "gini": 0.4, "ks": 0.3, "pr_auc": 0.12, "lift_top_decile": 1.5, "base_bad_rate": 0.1}
    df = evaluation.assessment({"train": a, "test": b}, cfg)
    w = " ".join(df.attrs["warnings"])
    assert "overfit" in w and "PR-AUC" in w and "Lift" in w


def test_psi_csi_tranche_lift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert evaluation.psi(a, rng.normal(size=5000)) < 0.05
    assert evaluation.psi(a, rng.normal(1.0, 1, 5000)) > 0.25
    e = pd.DataFrame({"x": rng.choice(list("ab"), 1000, p=[.5, .5])}); s_ = pd.DataFrame({"x": rng.choice(list("ab"), 1000, p=[.9, .1])})
    assert evaluation.csi_table(e, s_)["csi"].iloc[0] > 0.25
    assert evaluation.stability_flag(0.05, .1, .25) == "xanh" and evaluation.stability_flag(.2, .1, .25) == "vàng" and evaluation.stability_flag(.3, .1, .25) == "đỏ"
    score = rng.normal(600, 50, 10000)
    y = (rng.random(10000) < 1 / (1 + np.exp((score - 560) / 25))).astype(int)
    tt = evaluation.tranche_table(y, score, 10)
    assert tt.attrs["monotonic"] or tt["bad_rate"].iloc[0] > tt["bad_rate"].iloc[-1]
    assert tt["tranche"].iloc[0] == "A" and tt["count"].sum() == 10000
    lt = evaluation.lift_table(y, -score)
    assert lt["lift"].iloc[0] > 1.5 and lt["cum_bad_capture"].iloc[-1] == pytest.approx(1)


def test_scaling_and_points(built, cfg):
    f, off = scorecard.scaling(cfg)
    sc = cfg["scorecard"]
    s_base = scorecard.score_from_logit(np.array([-np.log(sc["base_odds_good_to_bad"])]), cfg)[0]
    assert s_base == pytest.approx(sc["base_score"])
    s_pdo = scorecard.score_from_logit(np.array([-np.log(2 * sc["base_odds_good_to_bad"])]), cfg)[0]
    assert s_pdo - s_base == pytest.approx(sc["pdo"])
    d, y, tr, te, specs, tabs, Wtr, Wte = built
    sel = ["age", "dti", "income_mil", "employment_years"]
    m = modeling.fit_logistic(Wtr, y.loc[tr], sel, cfg)
    c2 = config.clone(cfg); c2["scorecard"]["rounding"] = 6; c2["scorecard"]["score_range"] = [-1e9, 1e9]
    pts = scorecard.build_table(m, specs, tabs, c2)
    s_tab = scorecard.score(d.loc[te], m, specs, pts, c2)
    s_log = scorecard.score_from_logit(modeling.predict_logit(m, Wte), c2)
    assert np.allclose(s_tab, s_log, atol=1e-3)                  # bảng điểm khớp công thức logit
    assert evaluation.gini(y.loc[te], -s_tab) > 0.3              # điểm cao = rủi ro thấp
    assert set(pts["variable"]) == set(sel)


def test_cutoff_zones(cfg):
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(600, 50, 20000))
    y = pd.Series((rng.random(20000) < 1 / (1 + np.exp((s - 560) / 25))).astype(int))
    ct = scorecard.cutoff_table(y, s)
    assert ct["approval_rate"].is_monotonic_decreasing and ct["bad_rate_approved"].iloc[-1] < ct["bad_rate_approved"].iloc[0]
    c = config.clone(cfg)
    assert scorecard.suggest_cutoff(ct, c)["cutoff"] is None       # chưa chốt tham số
    c["cutoff"]["target_bad_rate"] = 0.10
    r = scorecard.suggest_cutoff(ct, c)
    row = ct[ct.cutoff == r["cutoff"]].iloc[0]
    assert row["bad_rate_approved"] <= 0.10
    c["cutoff"].update(strategy="target_approval_rate", target_approval_rate=0.6)
    r2 = scorecard.suggest_cutoff(ct, c)
    assert abs(ct[ct.cutoff == r2["cutoff"]]["approval_rate"].iloc[0] - 0.6) < 0.05
    c["cutoff"].update(strategy="profit", profit_per_good=1, loss_per_bad=9)
    assert scorecard.suggest_cutoff(ct, c, y, s)["cutoff"] is not None
    lo, hi = scorecard.decision_zones(s, r["cutoff"], c)
    assert lo < r["cutoff"] < hi
    z = scorecard.apply_zones(s, lo, hi)
    summ = scorecard.zone_summary(y, z)
    assert abs(summ["pct"].sum() - 1) < 1e-9
    br = dict(zip(summ["zone"], summ["bad_rate"]))
    assert br["approve"] < br["manual_review"] < br["reject"]


def test_export_and_score_new(built, cfg, tmp_path):
    d, y, tr, te, specs, tabs, Wtr, Wte = built
    sel = ["age", "dti", "income_mil"]
    m = modeling.fit_logistic(Wtr, y.loc[tr], sel, cfg)
    pts = scorecard.build_table(m, specs, tabs, cfg)
    s = scorecard.score(d.loc[te], m, specs, pts, cfg)
    lo, hi = scorecard.decision_zones(s, float(s.median()), cfg)
    bundle = export.build_bundle(cfg, m, specs, pts, {"cutoff": float(s.median()), "reject_below": lo, "approve_from": hi})
    b2 = export.bundle_from_bytes(export.bundle_to_bytes(bundle))
    out = export.score_new_data(b2, d.loc[te].head(200))
    assert {"score", "pd", "zone"} <= set(out.columns) and np.allclose(out["score"], s.head(200))
    with pytest.raises(ValueError):
        export.score_new_data(b2, d[["age"]])
    xl = export.to_excel({"points": pts, "metrics": pd.DataFrame({"a": [1]})}, cfg)
    assert xl[:2] == b"PK"
    pdf = export.to_pdf("Báo cáo", [("Bảng điểm", pts)], ["ghi chú"])
    assert pdf[:4] == b"%PDF"


def test_store(tmp_path, cfg):
    st = store.Store("s1", tmp_path)
    assert st.status(cfg)[0]["state"] == "pending"
    st.save("ingest", {"a": 1}, cfg)
    st.save("cleaning", {"b": 2}, cfg)
    assert st.load("ingest")["payload"] == {"a": 1}
    s = {r["step"]: r["state"] for r in st.status(cfg)}
    assert s["ingest"] == "done" and s["cleaning"] == "done" and s["target"] == "pending"
    import time; time.sleep(0.01)
    st.save("ingest", {"a": 3}, cfg)                       # chạy lại bước trước → bước sau stale
    assert {r["step"]: r["state"] for r in st.status(cfg)}["cleaning"] == "stale"
    c2 = config.clone(cfg); config.set_(c2, "seed", 1)
    assert {r["step"]: r["state"] for r in st.status(c2)}["ingest"] == "config_changed"
    h = st.save_config_version(cfg, "test")
    assert len(st.config_history()) == 1 and h == config.config_hash(cfg)
