import io

import numpy as np
import pandas as pd
import pytest

from core import cleaning, ingest


def test_read_csv_and_limits():
    csv = b"a,b\n1,2\n3,4\n"
    df = ingest.read_file(io.BytesIO(csv), "x.csv", {"max_rows": 10}, size_bytes=len(csv))
    assert df.shape == (2, 2)
    with pytest.raises(ingest.IngestError):
        ingest.read_file(io.BytesIO(csv), "x.csv", {"max_rows": 1})
    with pytest.raises(ingest.IngestError):
        ingest.read_file(io.BytesIO(csv), "x.csv", {"max_upload_mb": 0.000001}, size_bytes=1000)
    with pytest.raises(ingest.IngestError):
        ingest.read_file(io.BytesIO(b"x"), "x.pdf")


def test_mapping_validation(raw, mapping):
    assert ingest.validate_mapping(raw, mapping) == []
    bad = dict(mapping, dpd_max="nope")
    assert any("nope" in e for e in ingest.validate_mapping(raw, bad))
    dup = dict(mapping, id="max_dpd")
    assert any("nhiều vai trò" in e for e in ingest.validate_mapping(raw, dup))


def test_feature_columns_excludes_roles(raw, mapping):
    f = ingest.feature_columns(raw, mapping)
    assert "max_dpd" not in f and "loan_id" not in f and "age" in f


def test_analyze_flags(raw, mapping, cfg):
    feats = ingest.feature_columns(raw, mapping, ["first_bad_mob"])
    a = cleaning.analyze(raw, feats, cfg).set_index("variable")
    assert a.loc["all_const", "suggestion"] == "drop"
    assert a.loc["mostly_missing", "suggestion"] == "drop"
    assert a.loc["age", "suggestion"] in ("keep", "review")


def test_special_values_and_impossible(raw, cfg):
    import core.config as C
    c = C.clone(cfg)
    c["cleaning"]["special_values"] = [-999]
    c["cleaning"]["impossible_value_rules"] = {"age": [18, 100]}
    a = cleaning.analyze(raw, ["income_mil", "age"], c).set_index("variable")
    assert a.loc["income_mil", "special_pct"] > 0.01
    assert a.loc["age", "impossible_pct"] > 0
    out, log = cleaning.apply(raw, c, [], ["age"])
    assert out["age"].max() <= 100 and log["impossible_to_nan"]["age"] > 0
    assert (cleaning.impossible_values(raw, c)["n_violations"] > 0).all()


def test_apply_impute_and_drop(cfg):
    import core.config as C
    c = C.clone(cfg)
    c["cleaning"]["missing"]["strategy"] = "impute"
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "x"], "c": [1, 2, 3]})
    out, log = cleaning.apply(df, c, ["c"])
    assert "c" not in out and out["a"].isna().sum() == 0 and out["b"].tolist() == ["x", "x", "x"]
