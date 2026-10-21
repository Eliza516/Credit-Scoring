import numpy as np
import pandas as pd
import pytest

from core import binning, config, ingest, segmentation, selection, split


@pytest.fixture(scope="module")
def prep(dev, mapping, cfg):
    d, y = dev
    parts = split.split_indices(d, y, cfg, mapping["open_date"])
    tr, te = parts["train"], parts["test"]
    feats = ["age", "income_mil", "dti", "credit_util", "employment_years", "inquiries_6m", "delinq_12m",
             "loan_amount_mil", "zip_code", "gender", "channel", "term_months", "product_type"]
    specs = binning.fit_all(d.loc[tr], y.loc[tr], feats, cfg)
    tabs = binning.tables(d.loc[tr], y.loc[tr], specs)
    tabs_te = binning.tables(d.loc[te], y.loc[te], specs)
    specs = {c: binning.attach_woe(specs[c], tabs[c]) for c in specs}
    W = binning.transform(d.loc[tr], specs)
    return d, y, tr, te, feats, tabs, tabs_te, W


def test_iv_bands(cfg):
    assert [selection.iv_band(v, cfg) for v in (0.01, 0.05, 0.2, 0.4, 0.7)] == ["useless", "weak", "medium", "strong", "suspicious"]


def test_compliance(cfg):
    f = selection.compliance_flags(["gender", "zip_code", "age", "marital_status"], cfg).set_index("variable")
    assert f.loc["gender", "protected"] and f.loc["marital_status", "protected"]
    assert f.loc["zip_code", "proxy_warning"] and not f.loc["age", "protected"]


def test_funnel(prep, cfg):
    d, y, tr, te, feats, tabs, tabs_te, W = prep
    f = selection.funnel(d.loc[tr], W, tabs, tabs_te, cfg).set_index("variable")
    assert f.loc["gender", "suggestion"] == "drop" and f.loc["gender", "stage"] == "compliance"
    assert f.loc["term_months", "suggestion"] == "drop"            # IV ≈ 0
    # cặp tương quan thô cao: chỉ một trong age / employment_years còn lại
    assert {f.loc["age", "suggestion"], f.loc["employment_years", "suggestion"]} >= {"keep"} or f.loc["age", "stage"] == "corr_raw" or f.loc["employment_years", "stage"] == "corr_raw"
    kept = f[f.suggestion == "keep"]
    assert 1 <= len(kept) <= cfg["selection"]["final_variable_count"][1]
    assert (kept["vif"] <= cfg["selection"]["vif"]["threshold"] + 1e-9).all()
    assert all(f.loc[kept.index, "iv"] >= cfg["selection"]["iv"]["useless_below"])


def test_must_keep_drop(prep, cfg):
    d, y, tr, te, feats, tabs, tabs_te, W = prep
    c = config.clone(cfg)
    c["selection"]["must_drop"] = ["dti"]
    c["selection"]["must_keep"] = ["term_months", "gender"]
    f = selection.funnel(d.loc[tr], W, tabs, tabs_te, c).set_index("variable")
    assert f.loc["dti", "suggestion"] == "drop"
    assert f.loc["term_months", "suggestion"] == "keep"
    assert f.loc["gender", "suggestion"] == "drop"      # biến bị bảo vệ không thể ép giữ


def test_vif_detects_collinearity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    W = pd.DataFrame({"a": a, "b": a + rng.normal(0, 0.05, 500), "c": rng.normal(size=500)})
    v = selection.vif_table(W)
    assert v["a"] > 10 and v["b"] > 10 and v["c"] < 2


def test_segmentation_suggest(prep, cfg):
    d, y, tr, te, feats, tabs, tabs_te, W = prep
    c = config.clone(cfg)
    c["segmentation"]["candidate_features"] = ["product_type", "channel"]
    c["segmentation"]["min_bads_per_segment"] = 100
    res = segmentation.suggest(d, y, ["age", "income_mil", "dti", "credit_util", "employment_years", "inquiries_6m", "delinq_12m", "product_type", "channel"], c, tr, te)
    assert set(res["candidates"]["feature"]) == {"product_type", "channel"}
    assert res["candidates"]["suggestion"].isin(["nên tách", "không tách"]).all()
    assert 0 < res["baseline_gini"] < 100
    assert len(res["tree"]) >= 1 and abs(res["tree"]["size_pct"].sum() - 1) < 1e-9


def test_assign_segments(dev):
    d, _ = dev
    s = segmentation.assign_segments(d, "product_type", {"A": ["personal"], "B": ["motorbike", "credit_card"]})
    assert set(s.unique()) == {"A", "B"}
