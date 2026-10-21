import numpy as np
import pandas as pd

from core import config, split, target, vintage


def test_target_statuses(raw, mapping, cfg, tgt):
    assert set(tgt["status"].unique()) <= {"bad", "good", "indeterminate", "excluded"}
    assert tgt.loc[tgt["status"] == "bad", "y"].eq(1).all()
    assert tgt.loc[tgt["status"] == "good", "y"].eq(0).all()
    assert tgt.loc[~tgt["status"].isin(["bad", "good"]), "y"].isna().all()
    assert (tgt.loc[raw["fraud_flag"] == 1, "status"] == "excluded").all()
    assert (tgt.loc[raw["staff_flag"] == 1, "status"] == "excluded").all()
    # bad: DPD>=91 và first_bad_mob <= window
    bad = raw[tgt["status"] == "bad"]
    assert (bad["max_dpd"] >= 91).all() and (bad["first_bad_mob"] <= cfg["target"]["performance_window_months"]).all()
    # indeterminate gồm DPD 10-90
    mid = raw[(raw["max_dpd"] > 0) & (raw["max_dpd"] < 91) & (raw["fraud_flag"] == 0) & (raw["staff_flag"] == 0)]
    assert (tgt.loc[mid.index, "status"] != "good").all()


def test_target_window_changes_bad_rate(raw, mapping, cfg):
    c = config.clone(cfg)
    c["target"]["performance_window_months"] = 24
    t24 = target.build_target(raw, mapping, c)
    assert (t24["status"] == "bad").sum() >= (target.build_target(raw, mapping, cfg)["status"] == "bad").sum()


def test_target_precomputed(raw, cfg):
    df = pd.DataFrame({"t": [1, 0, np.nan, 1]})
    t = target.build_target(df, {"target": "t"}, cfg)
    assert t["status"].tolist() == ["bad", "good", "excluded", "bad"]


def test_vintage_curves_monotonic_and_window(raw, mapping, cfg, tgt):
    vt = vintage.vintage_table(raw, mapping, cfg, tgt["status"])
    curves = vt.drop(columns="size")
    # bad rate tích lũy không giảm theo MOB (mẫu số thay đổi nên chỉ kiểm cohort đủ chín)
    old = curves.iloc[0].dropna().to_numpy()
    assert (np.diff(old) >= -1e-9).all()
    w = vintage.suggest_window(vt, cfg)
    assert w["suggested_mob"] is None or 1 <= w["suggested_mob"] <= cfg["vintage"]["max_mob"]
    an = vintage.cohort_anomalies(vt, cfg)
    assert {"cohort", "z", "flag"} <= set(an.columns)


def test_vintage_requires_columns(raw, cfg):
    import pytest
    with pytest.raises(ValueError):
        vintage.vintage_table(raw, {"open_date": "open_date"}, cfg)


def test_split_time_and_sizes(dev, mapping, cfg):
    d, y = dev
    parts = split.split_indices(d, y, cfg, mapping["open_date"])
    assert len(parts["train"]) + len(parts["test"]) == len(y)
    assert d.loc[parts["train"], "open_date"].max() <= d.loc[parts["test"], "open_date"].min()
    assert abs(len(parts["train"]) / len(y) - cfg["split"]["train_pct"]) < 0.01
    tab = split.check_sizes(y, parts, cfg)
    assert set(tab["set"]) == {"train", "test"}


def test_split_random_and_oot(dev, mapping, cfg):
    d, y = dev
    c = config.clone(cfg)
    c["split"]["method"] = "random"
    c["split"]["oot"]["enabled"] = True
    parts = split.split_indices(d, y, c, mapping["open_date"])
    assert "oot" in parts
    assert not set(parts["train"]) & set(parts["test"]) and not set(parts["oot"]) & set(parts["train"])
    p2 = split.split_indices(d, y, c, mapping["open_date"])
    assert parts["train"].equals(p2["train"])  # tái lập


def test_split_min_warning(dev, mapping, cfg):
    d, y = dev
    c = config.clone(cfg)
    c["split"]["min_bads"] = 10**6
    tab = split.check_sizes(y, split.split_indices(d, y, c, mapping["open_date"]), c)
    assert "bad" in tab.loc[tab["set"] == "train", "warning"].iloc[0]
