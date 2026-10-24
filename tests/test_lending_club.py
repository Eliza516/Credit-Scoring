"""Hồ sơ dữ liệu, nhãn từ cột trạng thái, nạp file lớn/nhiều file, dashboard – trên dữ liệu mô phỏng kiểu Lending Club."""
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import cleaning, config, dashboard as D, demo, ingest, profiles, split, target

ROOT = Path(__file__).resolve().parent.parent
PAGE = str(ROOT / "app" / "pages" / "1_analysis.py")


@pytest.fixture(scope="module")
def lc_raw():
    return demo.make_lending_club_like(12000, seed=3)


@pytest.fixture(scope="module")
def prof():
    return profiles.load_profile("lending_club")


@pytest.fixture(scope="module")
def lc_cfg(prof):
    return profiles.apply_to_config(config.load_default(prefer_real=False), prof)


@pytest.fixture(scope="module")
def lc_df(lc_raw, prof, lc_cfg):
    m = profiles.mapping_for(lc_raw.columns, prof)
    d = ingest.apply_mapping(lc_raw, m, lc_cfg["columns"]["date_formats"])
    d, made = profiles.apply_derived(d, prof)
    return d, m, made


# ------------------------------------------------------------------ hồ sơ
def test_profiles_listing_and_dictionary(prof):
    ps = profiles.list_profiles()
    assert {"generic", "lending_club"} <= set(ps)
    assert len(prof["dictionary"]) == 151                                  # từ điển đủ 151 cột như repo Lending Club
    ex = profiles.excluded_features(prof)
    assert {"total_pymnt", "recoveries", "out_prncp", "last_fico_range_high", "id"} <= set(ex)
    assert "loan_amnt" not in ex and "int_rate" not in ex and "dti" not in ex
    assert profiles.column_notes(prof)["dti"]


def test_apply_to_config_and_merge(prof):
    base = config.load_default(prefer_real=False)
    c = profiles.apply_to_config(base, prof)
    assert base["target"]["status"]["bad"] == [] and c["target"]["status"]["bad"] == ["Charged Off"]     # không sửa bản gốc
    assert c["split"]["oot"]["enabled"] and c["split"]["oot"]["pct"] == 0.15
    assert abs(c["split"]["train_pct"] + c["split"]["test_pct"] - 1) < 1e-6 and config.validate(c) == []
    assert c["columns"]["date_formats"]["issue_d"] == "%b-%Y"
    g = profiles.apply_to_config(base, profiles.load_profile("generic"))
    assert g["target"]["status"]["bad"] == []


def test_dates_and_derived(lc_df, lc_raw):
    d, m, made = lc_df
    assert m == {"id": "id", "open_date": "issue_d", "status_col": "loan_status"}
    assert pd.api.types.is_datetime64_any_dtype(d["issue_d"])
    assert set(made) == {"term_months", "emp_length_years", "credit_history_months", "fico_mid"}
    assert set(d["term_months"].dropna().unique()) == {36.0, 60.0}
    el = d.loc[lc_raw["emp_length"] == "< 1 year", "emp_length_years"]
    assert (el == 0).all() and d.loc[lc_raw["emp_length"] == "10+ years", "emp_length_years"].eq(10).all()
    assert d["emp_length_years"].isna().sum() == lc_raw["emp_length"].isna().sum()
    assert (d["credit_history_months"] > 0).all()
    assert "term" not in d.columns and "earliest_cr_line" not in d.columns and "fico_range_low" not in d.columns   # cột gốc bị bỏ
    assert (d["fico_mid"] == lc_raw[["fico_range_low", "fico_range_high"]].mean(axis=1)).all()


# ------------------------------------------------------- nhãn từ trạng thái
def test_status_target_labels_and_exclusions(lc_df, lc_cfg):
    d, m, _ = lc_df
    t = target.build_target(d, m, lc_cfg)
    st = d["loan_status"]
    assert set(t.loc[(t["status"] == "bad"), "y"]) == {1.0} and set(t.loc[t["status"] == "good", "y"]) == {0.0}
    assert (st[t["status"] == "bad"] == "Charged Off").all() and (st[t["status"] == "good"] == "Fully Paid").all()
    cur = st.isin(["Current", "Late (31-120 days)", "In Grace Period", "Default"])
    assert (t.loc[cur & (d["issue_d"].dt.year >= 2010), "status"] != "bad").all()
    # trước 2010 bị loại; chưa đáo hạn bị loại
    pre = (d["issue_d"].dt.year < 2010) & st.isin(["Fully Paid", "Charged Off"])
    assert (t.loc[pre, "reason"] == "mở trước 2010").all() and (t.loc[pre, "status"] == "excluded").all()
    end = target.data_end_period(d, "last_pymnt_d")
    due = d["issue_d"].dt.to_period("M") + d["term_months"].astype(int)
    unmatured = (due > end) & st.isin(["Fully Paid", "Charged Off"]) & ~pre
    assert unmatured.sum() > 0 and (t.loc[unmatured, "status"] == "excluded").all()
    keep = t["status"].isin(["bad", "good"])
    assert ((due[keep] <= end) & (d.loc[keep, "issue_d"].dt.year >= 2010)).all()
    assert 0.05 < t.loc[keep, "y"].mean() < 0.4


def test_status_target_requires_declared_values(lc_df, lc_cfg):
    from core import performance
    d, m, _ = lc_df
    empty = config.clone(lc_cfg)
    empty["target"]["status"]["bad"] = []
    assert performance.target_source_errors(m, empty) and not performance.target_source_errors(m, lc_cfg)
    y = target.provisional_y(d, m, lc_cfg)
    assert set(y.dropna().unique()) == {0.0, 1.0} and y.isna().sum() == d["loan_status"].isin(["Current", "Default", "In Grace Period", "Late (31-120 days)"]).sum()
    assert target.provisional_y(d, {}, lc_cfg) is None


# ---------------------------------------------------------------- làm sạch
def test_cleaning_forced_and_cardinality(lc_df, prof, lc_cfg):
    d, m, _ = lc_df
    feats = ingest.feature_columns(d, m)
    a = cleaning.analyze(d, feats, lc_cfg, profiles.excluded_features(prof)).set_index("variable")
    assert a.loc["total_pymnt", "suggestion"] == "drop" and "sau khi vay" in a.loc["total_pymnt", "reason"]
    assert a.loc["zip_code", "suggestion"] == "drop" and "giá trị chữ" in a.loc["zip_code", "reason"]
    assert a.loc["emp_title", "suggestion"] == "drop"
    assert a.loc["all_empty_2012", "suggestion"] == "drop"
    assert a.loc["dti", "suggestion"] == "keep" and a.loc["term_months", "suggestion"] == "keep"
    assert "giá trị chữ" not in a.loc["last_pymnt_d", "reason"]          # cột ngày không bị coi là biến chữ nhiều giá trị


def test_identifier_like_detection(cfg):
    df = pd.DataFrame({"uid": [f"u{i}" for i in range(2000)], "grp": ["a", "b"] * 1000})
    a = cleaning.analyze(df, ["uid", "grp"], cfg).set_index("variable")
    assert a.loc["uid", "suggestion"] == "drop" and a.loc["grp", "suggestion"] == "keep"


# ---------------------------------------------------- split OOT theo phần trăm
def test_split_oot_pct_time_ordered(lc_df, lc_cfg):
    d, m, _ = lc_df
    t = target.build_target(d, m, lc_cfg)
    dev = t["status"].isin(["bad", "good"])
    D0, y = d[dev], t.loc[dev, "y"]
    parts = split.split_indices(D0, y, lc_cfg, "issue_d")
    assert {"train", "test", "oot"} == set(parts)
    n = len(y)
    assert abs(len(parts["oot"]) / n - 0.15) < 0.01 and abs(len(parts["train"]) / n - 0.70) < 0.02
    dt = D0["issue_d"]
    assert dt[parts["train"]].max() <= dt[parts["test"]].min() <= dt[parts["test"]].max() <= dt[parts["oot"]].min()


# ----------------------------------------------------- file lớn & nhiều file
def _gz(df, name="x.csv.gz", footer=True):
    txt = df.to_csv(index=False)
    if footer:
        txt += '\nTotal amount funded in policy code 1: 1234\n'
    return io.BytesIO(gzip.compress(txt.encode())), name


def test_chunked_read_filter_skip_sample_and_footer(lc_raw, prof, lc_cfg):
    f, name = _gz(lc_raw)
    keep = profiles.row_filter(prof, lc_cfg)
    df, info = ingest.read_csv_chunked(f, name, skip_columns=prof["load"]["skip_columns"], chunk_rows=3000, keep=keep, max_rows=2500, seed=1)
    assert info["rows_read"] == len(lc_raw) + 1 and info["sampled"] and info["rows_kept"] == 2500 == len(df)
    assert not ({"url", "desc", "emp_title", "title", "member_id"} & set(df.columns)) and "emp_title" in lc_raw.columns
    assert set(df["loan_status"]) <= {"Fully Paid", "Charged Off"}                     # chỉ khoản có nhãn
    assert pd.to_datetime(df["issue_d"], format="%b-%Y").dt.year.min() >= 2010
    assert pd.api.types.is_numeric_dtype(df["id"])                                      # id có dòng tổng kết cuối file vẫn ép về số
    # mẫu đều: bad rate gần với toàn bộ dòng thoả bộ lọc
    full = lc_raw[lc_raw["loan_status"].isin(["Fully Paid", "Charged Off"]) & (pd.to_datetime(lc_raw["issue_d"], format="%b-%Y").dt.year >= 2010)]
    assert abs((df["loan_status"] == "Charged Off").mean() - (full["loan_status"] == "Charged Off").mean()) < 0.03
    f2, _ = _gz(lc_raw, footer=False)
    all_rows, info2 = ingest.read_csv_chunked(f2, name, chunk_rows=5000)
    assert len(all_rows) == len(lc_raw) and not info2["sampled"]
    again, _ = ingest.read_csv_chunked(_gz(lc_raw)[0], name, keep=keep, max_rows=2500, chunk_rows=3000, seed=1, skip_columns=["url", "desc", "emp_title", "title", "member_id"])
    pd.testing.assert_frame_equal(df, again)                                            # cùng seed → cùng mẫu


def test_group_combine_roles_and_duplicates(lc_raw):
    a, b = lc_raw.iloc[:6000].copy(), lc_raw.iloc[6000:].copy()
    b = b.rename(columns={"annual_inc": "Annual_Inc"})                                   # khác hoa/thường vẫn là cùng cấu trúc
    rejected = pd.DataFrame({"Amount Requested": [1000, 2000], "Application Date": ["2016-01-01"] * 2, "Risk_Score": [600, 640]})
    perf = pd.DataFrame({"loan_id": [1, 1, 2], "month": ["2016-01-31", "2016-02-29", "2016-01-31"], "dpd": [0, 31, 0]})
    groups = ingest.group_by_schema([list(a.columns), list(rejected.columns), list(b.columns), list(perf.columns)], 0.9)
    assert groups == [[0, 2], [1], [3]]
    big = ingest.combine_frames([a, b], ["p1.csv", "p2.csv"])
    assert len(big) == len(lc_raw) and "Annual_Inc" not in big.columns and big["annual_inc"].notna().all()
    assert big["_source_file"].tolist() == ["p1.csv"] * 6000 + ["p2.csv"] * 6000
    assert ingest.suggest_group_role(big, True) == "main" and ingest.suggest_group_role(rejected, False) == "ignore"
    assert ingest.suggest_group_role(perf, False) == "performance"
    dup = pd.concat([a.iloc[:100], a.iloc[50:150]])
    rep = ingest.duplicate_id_report(ingest.combine_frames([a.iloc[:100], a.iloc[50:150]], ["x", "y"]), "id")
    assert rep == {"duplicate_rows": 50, "duplicate_ids": 50, "cross_file_ids": 50}
    assert ingest.group_by_schema([["a", "b"], ["c", "d"]]) == [[0], [1]]


# ------------------------------------------------------------------ dashboard
def test_dashboard_tables(lc_df, lc_cfg):
    d, m, _ = lc_df
    y = target.provisional_y(d, m, lc_cfg)
    t = D.badrate_by_feature(d, y, "grade")
    assert t["n"].sum() == y.notna().sum() and abs(t["share"].sum() - 1) < 1e-9
    gr = t.set_index("group")["bad_rate"]
    assert gr["A"] < gr["D"] < gr["F"]                                                   # tỷ lệ bad tăng theo hạng tín dụng
    assert ((t["ci_low"] <= t["bad_rate"]) & (t["bad_rate"] <= t["ci_high"])).all() and t.attrs["base_rate"] == pytest.approx(y.mean())
    num = D.badrate_by_feature(d, y, "dti", n_bins=8)
    assert 2 <= len(num) <= 9 and num["n"].sum() == y.notna().sum()
    miss = D.badrate_by_feature(d, y, "emp_length_years")
    assert "Thiếu" in set(miss["group"]) and miss["group"].iloc[-1] == "Thiếu"
    many = D.badrate_by_feature(d, y, "zip_code", max_categories=5)
    assert len(many) <= 6 and "Khác" in set(many["group"])
    small = D.badrate_by_feature(d, y, "dti", n_bins=8, min_count=10**6)
    assert small["small"].all()
    tt = D.badrate_by_feature_time(d, y, "grade", "issue_d", "Y", 5, 4)
    assert {"period", "group", "n", "bad_rate"} <= set(tt.columns) and tt["n"].sum() == y.notna().sum()
    vol = D.volume_by_period(d, "issue_d", y, "Y")
    assert vol["n"].sum() == len(d) and vol["bads"].sum() == int((y == 1).sum()) and vol["bad_rate"].between(0, 1).all()
    ov = D.overview(d, "id")
    assert ov["rows"] == len(d) and ov["duplicate_ids"] == 0 and sum(ov["kinds"].values()) == ov["cols"]
    dist = D.distribution(d["annual_inc"])
    assert dist["hist"]["count"].sum() == d["annual_inc"].notna().sum() and dist["quantiles"]["P50"] > 0
    h = D.numeric_by_class(d, y, "dti")
    assert set(h["class"]) == {"good", "bad"} and h.groupby("class")["density"].sum().round(6).eq(1).all()
    lo, hi = D.wilson(np.array([5, 0]), np.array([10, 10]))
    assert lo[0] < 0.5 < hi[0] and lo[1] == 0 and hi[1] > 0


# ------------------------------------------------------------ giao diện
def _write_parts(tmp_path, df):
    paths = []
    half = len(df) // 2
    for i, part in enumerate([df.iloc[:half], df.iloc[half:]]):
        p = tmp_path / f"accepted_part{i + 1}.csv.gz"
        part.to_csv(p, index=False, compression="gzip")
        paths.append(str(p))
    rej = tmp_path / "rejected.csv"
    pd.DataFrame({"Amount Requested": [1000, 2000], "Application Date": ["2016-01-01"] * 2, "Risk_Score": [600, 640]}).to_csv(rej, index=False)
    paths.append(str(rej))
    return paths


def click(at, part):
    b = [b for b in at.button if part in b.label]
    assert b, f"không thấy nút '{part}'; có {[x.label for x in at.button]}"
    b[0].click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_ui_multi_file_profile_flow(tmp_path, lc_raw):
    paths = _write_parts(tmp_path, lc_raw)
    at = AppTest.from_file(PAGE, default_timeout=600)
    at.run()
    at.selectbox(key="profile_sel").set_value("lending_club").run()
    assert at.session_state["cfg"]["target"]["status"]["bad"] == ["Charged Off"]
    at.text_area(key="paths").set_value("\n".join(paths)).run()
    click(at, "Nạp các file theo đường dẫn")
    pend = at.session_state["pending"]["files"]
    assert [f["name"] for f in pend] == ["accepted_part1.csv.gz", "accepted_part2.csv.gz", "rejected.csv"]
    assert pend[0]["info"]["rows_kept"] < len(lc_raw) / 2                                # bộ lọc hồ sơ bỏ dòng chưa có nhãn / trước 2010
    click(at, "Dùng dữ liệu này")
    raw = at.session_state["raw"]
    assert len(raw) == sum(f["info"]["rows_kept"] for f in pend[:2]) and "_source_file" in raw.columns        # 2 file cùng cấu trúc được gộp; file lạ bị bỏ
    assert at.session_state["mapping_draft"] == {"id": "id", "open_date": "issue_d", "status_col": "loan_status"}
    click(at, "Chốt dữ liệu")
    assert "term_months" in at.session_state["raw"].columns and "term" not in at.session_state["raw"].columns
    assert "total_pymnt" in at.session_state["feats_all"]                                # hồ sơ chỉ gợi ý loại ở bước 2
    click(at, "Phân tích chất lượng")
    an = at.session_state["clean_an"].set_index("variable")
    assert an.loc["total_pymnt", "suggestion"] == "drop" and an.loc["zip_code", "suggestion"] == "drop"
    click(at, "Chốt làm sạch")
    assert "total_pymnt" not in at.session_state["feats"] and "int_rate" in at.session_state["feats"]
    click(at, "Dựng target")
    tgt = at.session_state["tgt"]
    assert set(tgt["status"].unique()) >= {"bad", "good"}
    click(at, "Chốt target")
    click(at, "Chia mẫu")
    assert {"train", "test", "oot"} == set(at.session_state["parts"])
    click(at, "Chốt chia mẫu")
    click(at, "Binning tự động")
    click(at, "Chốt binning")
    click(at, "Chạy phễu")
    f = at.session_state["funnel"].set_index("variable")
    assert "total_pymnt" not in f.index and f["suggestion"].eq("keep").any()


def test_score_raw_file_with_profile(lc_df, lc_cfg, prof, lc_raw):
    """Bundle mang theo hồ sơ: chấm điểm trực tiếp file thô (ngày dạng chữ, term chữ) mà không cần tự tạo cột suy ra."""
    from core import binning, export, modeling, scorecard
    d, m, _ = lc_df
    t = target.build_target(d, m, lc_cfg)
    dev = t["status"].isin(["bad", "good"])
    D0, y = d[dev], t.loc[dev, "y"]
    feats = ["int_rate", "dti", "annual_inc", "term_months", "fico_mid", "credit_history_months"]
    specs = binning.fit_all(D0, y, feats, lc_cfg)
    tabs = binning.tables(D0, y, specs)
    specs = {c: binning.attach_woe(specs[c], tabs[c]) for c in specs}
    W = binning.transform(D0, specs)
    model = modeling.fit_logistic(W, y, feats, lc_cfg)
    pts = scorecard.build_table(model, specs, tabs, lc_cfg)
    bundle = export.build_bundle(lc_cfg, model, specs, pts, None, None, None, prof)
    b2 = export.bundle_from_bytes(export.bundle_to_bytes(bundle))
    out = export.score_new_data(b2, lc_raw.head(300))                                  # lc_raw = file thô
    ref = scorecard.score(D0.loc[D0.index.intersection(out.index)], model, specs, pts, lc_cfg)
    assert {"score", "pd"} <= set(out.columns) and out["score"].notna().all()
    common = ref.index
    assert np.allclose(out.loc[common, "score"], ref, atol=1e-6)
    assert "slim" not in b2["profile"] and set(b2["profile"]) <= {"name", "date_formats", "derived", "drop_after_derive"}
