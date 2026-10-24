import numpy as np
import pandas as pd
import pytest

from core import config, demo, ingest, performance as P, target, vintage


@pytest.fixture(scope="module")
def perf(raw):
    return demo.make_demo_performance(raw)


@pytest.fixture(scope="module")
def long(raw, perf):
    od = pd.Series(raw["open_date"].to_numpy(), index=raw["loan_id"].to_numpy())
    return P.normalize_long(perf, "loan_id", "snapshot_month", "date", "dpd", od, "balance")


# ------------------------------------------------------------- nhận diện
def test_detect_layout_and_guess():
    wide = pd.DataFrame({"loan_id": [1, 2], "dpd_m1": [0, 0], "dpd_m2": [0, 30], "dpd_m3": [0, 60], "age": [30, 40]})
    assert P.detect_layout(wide) == "wide"
    assert P.guess_wide_cols(wide) == ["dpd_m1", "dpd_m2", "dpd_m3"]
    longdf = pd.DataFrame({"loan_id": [1], "snapshot_month": ["2024-01-31"], "dpd": [0], "outstanding_balance": [10.0]})
    assert P.detect_layout(longdf) == "long"
    g = P.guess_columns(longdf)
    assert g == {"id": "loan_id", "period": "snapshot_month", "dpd": "dpd", "balance": "outstanding_balance"}


def test_wide_to_long_order_and_blank():
    wide = pd.DataFrame({"id": ["a", "b"], "DPD12": [0, 91], "DPD2": [0, np.nan], "DPD3": [30, np.nan]})
    long = P.wide_to_long(wide, "id", ["DPD12", "DPD2", "DPD3"])
    # ô trống = chưa quan sát → bị bỏ; MOB lấy từ hậu tố số
    got = {(r.id, r.mob): r.dpd for r in long.itertuples()}
    assert got == {("a", 12): 0, ("a", 2): 0, ("a", 3): 30, ("b", 12): 91}
    with pytest.raises(P.PerformanceError):
        P.wide_to_long(wide, "id", ["id"])
    with pytest.raises(P.PerformanceError):
        P.wide_to_long(wide, "id", [])


# --------------------------------------------------------------- chuẩn hoá
def test_normalize_date_yyyymm_and_mob():
    opens = pd.Series(pd.to_datetime(["2024-01-15", "2024-03-02"]), index=[1, 2])
    d = pd.DataFrame({"id": [1, 1, 1, 2, 2, 9], "ym": [202402, 202403, 202312, 202404, 202404, 202402],
                      "dpd": [0, 31, 5, 0, 91, 0]})
    out = P.normalize_long(d, "id", "ym", "date", "dpd", opens)
    q = out.attrs["quality"]
    assert q["neg_mob"] == 1 and q["dup_rows"] == 1 and q["unmatched_rows"] == 1      # 202312 trước tháng mở; (2,202404) trùng; id 9 lạ
    got = {(r.id, r.mob): r.dpd for r in out.itertuples()}
    assert got == {(1, 1): 0, (1, 2): 31, (2, 1): 91}                                  # trùng lấy DPD lớn nhất
    mob = P.normalize_long(pd.DataFrame({"id": [1, 1], "m": [1, 2], "dpd": [-5, 10]}), "id", "m", "mob", "dpd")
    assert mob.attrs["quality"]["neg_dpd"] == 1 and mob["dpd"].tolist() == [0, 10]
    with pytest.raises(P.PerformanceError):
        P.normalize_long(d, "id", "ym", "date", "dpd", None)
    with pytest.raises(P.PerformanceError):
        P.normalize_long(d, "id", "id", "mob", "dpd")


def test_quality_report_counts():
    long = pd.DataFrame({"id": [1, 1, 1, 2, 7], "mob": [1, 2, 5, 1, 1], "dpd": [0] * 5})
    long.attrs["quality"] = {"rows_in": 6, "dup_rows": 1}
    rep = P.quality_report(long, pd.Series([1, 2, 3])).set_index("kiểm tra")["giá trị"]
    assert rep["Khoản vay có performance"] == 3
    assert rep["Khoản vay ở file chính nhưng KHÔNG có performance"] == 1       # id 3
    assert rep["ID ở performance nhưng không có ở file chính"] == 1            # id 7
    assert rep["Khoản vay có khoảng trống MOB"] == 1                           # id 1 thiếu MOB 3-4
    assert rep["Dòng trùng (ID, MOB) – đã gộp lấy DPD lớn nhất"] == 1


# ------------------------------------------------------ bảng theo khoản vay
def test_panel_summary_hand_calc():
    long = pd.DataFrame({"id": [1, 1, 1, 2, 2], "mob": [1, 2, 3, 1, 2], "dpd": [0, 91, 120, 0, 40], "balance": [10, 9, 8, 5, 4]})
    s = P.panel_summary(long, 91)
    assert s.loc[1, "first_bad_mob"] == 2 and s.loc[1, "dpd_max"] == 120 and s.loc[1, "months_observed"] == 3
    assert np.isnan(s.loc[2, "first_bad_mob"]) and s.loc[2, "dpd_max"] == 40 and s.loc[2, "n_payments"] == 2
    assert s.loc[1, "weight"] == 10                                             # dư nợ ở MOB nhỏ nhất
    assert P.panel_summary(long, 30).loc[2, "first_bad_mob"] == 2               # ngưỡng thấp hơn bắt thêm khoản


def test_loans_from_panel_missing_loans_and_duplicates(cfg):
    df = pd.DataFrame({"id": [1, 2, 3], "open": pd.to_datetime(["2024-01-01"] * 3)})
    long = pd.DataFrame({"id": [1, 1], "mob": [1, 2], "dpd": [0, 91]})
    loans = P.loans_from_panel(long, df, "id", "open", 91)
    assert loans["months_observed"].tolist() == [2, 0, 0] and loans["n_payments"].tolist() == [2, 0, 0]
    assert loans.index.equals(df.index)
    with pytest.raises(P.PerformanceError):
        P.loans_from_panel(long, pd.DataFrame({"id": [1, 1], "open": df["open"][:2]}), "id", "open", 91)
    # khoản không có performance bị loại "chưa đủ quan sát" khi dựng target, không bị nhầm thành good
    d2, m2 = P.apply_loans(df, {"id": "id", "open_date": "open"}, loans)
    t = target.build_target(d2, m2, cfg)
    assert t["status"].tolist()[1:] == ["excluded", "excluded"]
    assert t["status"].iloc[0] in ("bad", "indeterminate")


def test_loans_from_main_summary_kinds():
    df = pd.DataFrame({"open": pd.to_datetime(["2024-01-10", "2024-02-20", "2024-03-05"]),
                       "fb": [3, np.nan, 1], "fbd": pd.to_datetime(["2024-04-01", None, "2024-04-30"]),
                       "obs": [12, 11, 10], "obsd": pd.to_datetime(["2025-01-31", "2025-01-31", "2024-12-01"])})
    a = P.loans_from_main_summary(df, "open", "fb", "mob", "obs", "mob")
    assert a["first_bad_mob"].tolist()[0] == 3 and np.isnan(a["first_bad_mob"].iloc[1]) and a["months_observed"].tolist() == [12, 11, 10]
    b = P.loans_from_main_summary(df, "open", "fbd", "date", None, "asof", "2025-01-31")
    assert b["first_bad_mob"].tolist()[0] == 3 and b["first_bad_mob"].tolist()[2] == 1       # tháng 4 - tháng mở
    assert b["months_observed"].tolist() == [12, 11, 10]                                      # đến 2025-01
    c = P.loans_from_main_summary(df, "open", "fb", "mob", "obsd", "date")
    assert c["months_observed"].tolist() == [12, 11, 9]
    with pytest.raises(P.PerformanceError):
        P.loans_from_main_summary(df, "open", None, "mob", "obs", "mob")
    with pytest.raises(P.PerformanceError):
        P.loans_from_main_summary(df, "open", "fb", "mob", None, "asof")


def test_apply_loans_no_leak_and_override(raw, mapping):
    loans = pd.DataFrame({"first_bad_mob": 5.0, "months_observed": 30, "n_payments": 3, "dpd_max": 0}, index=raw.index)
    cols0 = list(raw.columns)
    d, m = P.apply_loans(raw, mapping, loans, ["first_bad_mob", "months_observed"])
    assert list(raw.columns) == cols0                                     # df gốc không bị sửa
    assert m["first_bad_mob"] == "__v_first_bad_mob" and m["dpd_max"] == mapping["dpd_max"]
    assert d["__v_months_observed"].eq(30).all()


def test_target_source_errors_and_relaxed_mapping(raw):
    assert P.target_source_errors({}) and not P.target_source_errors({"dpd_max": "x"}) and not P.target_source_errors({"target": "t"})
    assert ingest.validate_mapping(raw, {"id": "loan_id", "open_date": "open_date"}) == []     # bước 1 không còn bắt buộc dpd_max


# ------------------------------------------------------------------ vintage
def test_panel_vintage_equals_column_vintage(raw, mapping, cfg, long):
    """Cùng dữ liệu, đường performance và đường cột có sẵn cho cùng bảng vintage (MOB ≤ 24)."""
    loans = P.loans_from_panel(long, raw, "loan_id", "open_date", 91)
    excl = target.exclusion_flags(raw, mapping, cfg)
    a = vintage.vintage_from_loans(loans, cfg, excl)
    b = vintage.vintage_from_loans(pd.DataFrame({"open_date": raw["open_date"], "first_bad_mob": raw["first_bad_mob"],
                                                 "months_observed": raw["months_observed"]}), cfg, excl)
    pd.testing.assert_frame_equal(a.drop(columns="size"), b.drop(columns="size"), check_exact=False, atol=1e-12)
    assert a["size"].sum() == (~excl).sum()


def test_threshold_ordering_and_monotone(raw, mapping, cfg, long):
    excl = target.exclusion_flags(raw, mapping, cfg)
    tabs = {thr: vintage.vintage_from_loans(P.loans_from_panel(long, raw, "loan_id", "open_date", thr), cfg, excl) for thr in (30, 60, 91)}
    for lo, hi in ((30, 60), (60, 91)):
        a, b = tabs[lo].drop(columns="size").to_numpy(), tabs[hi].drop(columns="size").to_numpy()
        ok = ~np.isnan(a) & ~np.isnan(b)
        assert (a[ok] >= b[ok] - 1e-12).all()                        # ngưỡng DPD thấp ⇒ nhiều bad hơn ở mọi ô
    t = tabs[91]
    old = t.iloc[0].drop("size").dropna().to_numpy()                 # cohort già nhất, đã quan sát đủ → không giảm
    assert (np.diff(old) >= -1e-12).all() and old[-1] > old[0]


def test_balance_weighting_and_min_cell(cfg):
    c = config.clone(cfg); c["vintage"]["max_mob"] = 3; c["vintage"]["min_loans_per_cell"] = 1
    loans = pd.DataFrame({"open_date": pd.to_datetime(["2024-01-05"] * 4), "first_bad_mob": [2, np.nan, np.nan, np.nan],
                          "months_observed": [3, 3, 3, 3], "weight": [100.0, 10.0, 10.0, 10.0]})
    cnt = vintage.vintage_from_loans(loans, c)
    bal = vintage.vintage_from_loans(loans, c, weight=True)
    assert cnt.loc["2024-01", 2] == pytest.approx(0.25) and bal.loc["2024-01", 2] == pytest.approx(100 / 130)
    assert cnt.loc["2024-01", 1] == 0
    c["vintage"]["min_loans_per_cell"] = 5
    assert vintage.vintage_from_loans(loans, c).drop(columns="size").isna().all().all()      # 4 khoản < 5 → che hết
    assert vintage.n_obs(vintage.vintage_from_loans(loans, c)).loc["2024-01", 1] == 4


def test_regression_immature_loans_stay_in_vintage(raw, mapping, cfg):
    """Lỗi cũ: vintage loại mọi dòng 'excluded' – gồm cả khoản chưa đủ chín – nên mất cohort trẻ."""
    c = config.clone(cfg); c["target"]["performance_window_months"] = 24
    young = raw.copy(); young["months_observed"] = np.minimum(young["months_observed"], 6)
    t = target.build_target(young, mapping, c)
    assert (t["status"] == "excluded").mean() > 0.7                        # phần lớn bị loại khỏi target vì chưa đủ 24 tháng
    old_way = vintage.vintage_table(young, mapping, c, t["status"])        # kiểu cũ (qua status)
    flags = target.exclusion_flags(young, mapping, c)
    new_way = vintage.vintage_table(young, mapping, c, flags)
    assert old_way["size"].sum() < 0.3 * len(young) and new_way["size"].sum() == (~flags).sum()
    assert new_way[1].notna().sum() >= 20                                  # các cohort vẫn có số liệu MOB đầu


def _two_cohort_loans(n=1000):
    """Cohort già (obs 12) chín ở MOB 6 với 12% bad; cohort trẻ (obs 6) chỉ 3% bad."""
    rng = np.random.default_rng(0)
    def make(date, obs, rate):
        fb = np.where(rng.random(n) < rate, rng.integers(1, 7, n), np.nan)
        return pd.DataFrame({"open_date": pd.to_datetime(date), "first_bad_mob": fb, "months_observed": obs})
    return pd.concat([make("2023-01-10", 12, 0.12), make("2023-07-10", 6, 0.03)], ignore_index=True)


def test_regression_like_for_like_window(cfg):
    """Lỗi cũ: trung bình theo cohort ở mỗi MOB trộn tập cohort khác nhau → nhảy giả ở MOB 7."""
    c = config.clone(cfg); c["vintage"]["max_mob"] = 12; c["vintage"]["min_loans_per_cell"] = 1
    tab = vintage.vintage_from_loans(_two_cohort_loans(), c)
    # trung bình thô kiểu cũ: ở MOB 7 chỉ còn cohort già → nhảy giả
    naive = [np.average(tab[m].dropna(), weights=tab.loc[tab[m].notna(), "size"]) for m in range(1, 13)]
    assert (naive[6] - naive[5]) * 100 > 3
    w = vintage.suggest_window(tab, c)
    assert w["suggested_mob"] == 6                                         # like-for-like: không còn tăng sau MOB 6
    assert w["delta_pp"].loc[7:12].abs().max() < 1e-9
    assert (w["curve"].dropna().diff().dropna() >= -1e-12).all()           # đường tổng hợp không giảm
    assert w["curve"].loc[6] == pytest.approx(0.075, abs=0.02)


def test_suggest_window_constant_hazard_never_matures(cfg):
    c = config.clone(cfg); c["vintage"]["max_mob"] = 12
    rng = np.random.default_rng(1)
    n = 4000
    fb = rng.geometric(0.03, n).astype(float)
    loans = pd.DataFrame({"open_date": pd.to_datetime("2023-01-10"), "first_bad_mob": np.where(fb <= 24, fb, np.nan), "months_observed": 24})
    w = vintage.suggest_window(vintage.vintage_from_loans(loans, c), c)
    assert w["suggested_mob"] is None                                      # hazard không đổi → chưa bao giờ 'chín'


def test_exclusion_flags_only_flags(raw, mapping, cfg):
    f = target.exclusion_flags(raw, mapping, cfg)
    assert (f == ((raw["fraud_flag"] == 1) | (raw["staff_flag"] == 1))).all()
    t = target.build_target(raw, mapping, cfg)
    assert (t.loc[f, "status"] == "excluded").all() and (t.loc[f, "reason"].isin(["fraud", "nhân viên/VIP"])).all()
