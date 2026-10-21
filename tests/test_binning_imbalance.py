import numpy as np
import pandas as pd
import pytest

from core import binning, config, imbalance


@pytest.fixture(scope="module")
def tr(dev, mapping, cfg):
    from core import split
    d, y = dev
    parts = split.split_indices(d, y, cfg, mapping["open_date"])
    return d.loc[parts["train"]], y.loc[parts["train"]]


def test_numeric_bins_table_consistent(tr, cfg):
    d, y = tr
    spec = binning.fit_variable(d["age"], y, cfg)
    t = binning.make_table(d["age"], y, spec)
    assert t["count"].sum() == len(d)
    assert t["bad"].sum() == y.sum()
    assert abs(t["count_pct"].sum() - 1) < 1e-9
    assert binning.total_iv(t) > 0.01
    assert (t["woe"].iloc[:-0 or None].notna()).all()
    # WoE đúng dấu: bad rate cao → WoE thấp
    n = t[t.kind == "normal"]
    assert np.corrcoef(n["bad_rate"], n["woe"])[0, 1] < 0
    assert (n["count_pct"] >= cfg["binning"]["min_bin_pct"] - 0.01).all()


def test_special_and_missing_bins(tr, cfg):
    d, y = tr
    spec = binning.fit_variable(d["income_mil"], y, cfg, [-999])
    t = binning.make_table(d["income_mil"], y, spec)
    assert "Special" in set(t["bin"])
    d2 = d["employment_years"]
    s2 = binning.fit_variable(d2, y, cfg)
    t2 = binning.make_table(d2, y, s2)
    assert "Missing" in set(t2["bin"]) and t2.loc[t2.bin == "Missing", "kind"].iloc[0] == "missing"


def test_categorical_bins(tr, cfg):
    d, y = tr
    spec = binning.fit_variable(d["zip_code"], y, cfg)
    assert spec["kind"] == "categorical"
    flat = [c for g in spec["groups"] for c in g]
    assert set(flat) >= set(d["zip_code"].unique()) or binning.OTHER in flat
    t = binning.make_table(binning.prepare(d["zip_code"], spec), y, spec)
    assert t["count"].sum() == len(d)


def test_manual_merge_split_numeric(tr, cfg):
    d, y = tr
    spec = binning.fit_variable(d["dti"], y, cfg)
    n0 = len(spec["splits"])
    merged = binning.merge_bins(spec, 0)
    assert len(merged["splits"]) == n0 - 1
    back = binning.split_numeric(merged, spec["splits"][0])
    assert back["splits"] == spec["splits"]
    with pytest.raises(IndexError):
        binning.merge_bins(spec, n0)
    t = binning.make_table(d["dti"], y, merged)
    assert t["count"].sum() == len(d)


def test_manual_group_ops():
    spec = {"variable": "v", "kind": "categorical", "groups": [["a"], ["b"], ["c", "d"]], "specials": []}
    m = binning.merge_bins(spec, 0, 2)
    assert m["groups"] == [["a", "c", "d"], ["b"]]
    s = binning.split_group(spec, 2, ["d"])
    assert s["groups"] == [["a"], ["b"], ["c"], ["d"]]
    with pytest.raises(ValueError):
        binning.split_group(spec, 2, ["c", "d"])


def test_transform_and_unseen(tr, cfg):
    d, y = tr
    spec = binning.fit_variable(d["product_type"], y, cfg)
    t = binning.make_table(binning.prepare(d["product_type"], spec), y, spec)
    spec = binning.attach_woe(spec, t)
    new = pd.DataFrame({"product_type": ["personal", "zzz", None]})
    W = binning.transform(new, {"product_type": spec})
    assert W.shape == (3, 1) and W["product_type"].notna().all()


def test_suggestions(cfg):
    t = pd.DataFrame({"bin": ["a", "b", "c", "d"], "kind": ["normal"] * 4, "count": [30, 500, 400, 70], "bad": [1, 50, 60, 8],
                      "count_pct": [0.03, 0.5, 0.4, 0.07], "bad_rate": [0.03, 0.1, 0.15, 0.11], "woe": [1, 0, -0.3, -0.1], "iv": [0.1, 0, 0.01, 0.0]})
    s = binning.suggest(t, cfg, "x")
    assert any("Gộp bin 1" in m for m in s) and any("không đơn điệu" in m for m in s)


def test_fit_all_and_iv_summary(tr, cfg):
    d, y = tr
    specs = binning.fit_all(d, y, ["age", "dti", "zip_code"], cfg)
    tabs = binning.tables(d, y, specs)
    s = binning.iv_summary(tabs)
    assert list(s["iv"]) == sorted(s["iv"], reverse=True)


def test_imbalance_methods():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=list("abc"))
    y = pd.Series((rng.random(1000) < 0.1).astype(int))
    for m in ["none", "oversample_random", "undersample_random", "oversample_smote", "sample_weight"]:
        Xr, yr, w = imbalance.resample(X, y, m, 0.4, 1)
        assert len(Xr) == len(yr)
        if m in ("oversample_random", "undersample_random", "oversample_smote"):
            assert abs(yr.mean() - 0.4) < 0.02
        if m == "sample_weight":
            assert abs(np.average(yr, weights=w) - 0.4) < 1e-9
        if m == "none":
            assert Xr is X and w is None
    with pytest.raises(ValueError):
        imbalance.resample(X, y, "bogus")


def test_prior_offset():
    off = imbalance.prior_offset(0.4, 0.1)
    p_dev = 0.4
    odds = p_dev / (1 - p_dev) * np.exp(off)
    assert abs(odds / (1 + odds) - 0.1) < 1e-9
