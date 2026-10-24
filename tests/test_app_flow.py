"""Chạy toàn bộ giao diện bằng Streamlit AppTest (không cần trình duyệt) trên dữ liệu demo."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.steps.s3_target import SRC_MAIN, SRC_NONE, SRC_UP, B_WIDE

ROOT = Path(__file__).resolve().parent.parent
PAGE = str(ROOT / "app" / "pages" / "1_analysis.py")
RULES = str(ROOT / "app" / "pages" / "2_rules.py")
DOCS = str(ROOT / "app" / "pages" / "3_docs.py")


def click(at, label_part):
    btns = [b for b in at.button if label_part in b.label]
    assert btns, f"không thấy nút '{label_part}'; có: {[b.label for b in at.button]}"
    btns[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.fixture(scope="module")
def at():
    a = AppTest.from_file(PAGE, default_timeout=300)
    a.run()
    assert not a.exception
    a.session_state["cfg"]["split"]["min_goods"] = 500   # demo 40k dòng, giữ ngưỡng thực tế; nới nhẹ cho test nhanh nếu cần
    return a


def test_full_flow(at):
    click(at, "dữ liệu demo")
    click(at, "Chốt dữ liệu")
    assert "mapping" in at.session_state
    click(at, "Phân tích chất lượng")
    an = at.session_state["clean_an"]
    assert set(an.loc[an.suggestion == "drop", "variable"]) >= {"all_const", "mostly_missing"}
    click(at, "Chốt làm sạch")
    assert "all_const" not in at.session_state["feats"]
    at.radio(key="vsrc_kind").set_value(SRC_MAIN).run()                  # vintage từ các trường có sẵn trong file chính
    click(at, "Chuẩn hoá & kiểm tra")
    click(at, "Vẽ vintage")
    assert "vint" in at.session_state and at.session_state["vint"]["size"].sum() > 0
    click(at, "Dựng target")
    click(at, "Chốt target")
    assert at.session_state["target_ok"]
    click(at, "Chia mẫu")
    assert {"train", "test"} <= set(at.session_state["parts"])
    click(at, "Chốt chia mẫu")
    click(at, "Binning tự động")
    assert len(at.session_state["specs"]) == len(at.session_state["feats"])
    click(at, "Chốt binning")
    click(at, "Chạy phễu")
    f = at.session_state["funnel"]
    assert f.set_index("variable").loc["gender", "suggestion"] == "drop"
    click(at, "Chốt danh sách biến")
    assert at.session_state["sel_confirmed"] and "gender" not in at.session_state["final_vars"]
    # mô hình: chưa chọn mất cân bằng thì bị chặn
    assert at.session_state["cfg"]["split"]["imbalance"]["method"] is None
    assert not [b for b in at.button if "Chạy stepwise" in b.label]
    at.radio(key="imb_choice").set_value("undersample_random").run()
    at.radio(key="imb_prior").set_value("Có").run()
    click(at, "Lưu lựa chọn")
    assert at.session_state["cfg"]["split"]["imbalance"]["method"] == "undersample_random"
    click(at, "Chạy stepwise")
    click(at, "Huấn luyện logistic")
    assert at.session_state["model"]["table"]["sign_ok"].all()
    click(at, "Huấn luyện mô hình đối chứng")
    click(at, "Chốt mô hình")
    click(at, "Chạy đánh giá")
    E = at.session_state["eval"]
    assert E["metrics"]["test"]["gini"] > 0.3 and len(E["compare"]) == 4
    click(at, "Tạo bảng điểm")
    cfg = at.session_state["cfg"]
    click(at, "Chốt cutoff")
    assert "all" in at.session_state["bundles"]
    statuses = {r["step"]: r["state"] for r in at.session_state["store"].status(cfg)}
    assert statuses["scorecard"] in ("done", "config_changed")


def test_rules_page_runs():
    a = AppTest.from_file(RULES, default_timeout=120)
    a.run()
    assert not a.exception, [e.value for e in a.exception]
    assert len(a.tabs) >= 13
    btn = [b for b in a.button if "Áp dụng" in b.label][0]
    btn.click().run()
    assert not a.exception


def test_docs_page_runs():
    a = AppTest.from_file(DOCS, default_timeout=60)
    a.run()
    assert not a.exception


def _to_target_step(a):
    click(a, "dữ liệu demo")
    for role in ("dpd_max", "first_bad_mob", "months_observed", "n_payments"):    # bước 1 chỉ có ID + ngày mở + biến đặc trưng
        a.selectbox(key=f"map_{role}_demo").set_value("—")
    a.run()
    click(a, "Chốt dữ liệu")
    click(a, "Phân tích chất lượng")
    click(a, "Chốt làm sạch")


def test_step3_without_any_source_blocks_target():
    a = AppTest.from_file(PAGE, default_timeout=300)
    a.run()
    _to_target_step(a)
    assert a.session_state["mapping"]["dpd_max"] is None                       # bước 1 không còn bắt buộc dpd_max
    assert a.radio(key="vsrc_kind").value == SRC_NONE                          # vintage là tuỳ chọn
    click(a, "Dựng target")
    assert "tgt" not in a.session_state and any("Chưa có nguồn dựng target" in e.value for e in a.error)


def test_step3_performance_upload_flow_matches_columns():
    from core import config, demo, target
    a = AppTest.from_file(PAGE, default_timeout=600)
    a.run()
    _to_target_step(a)
    a.radio(key="vsrc_kind").set_value(SRC_UP).run()
    click(a, "Dùng performance demo")
    assert len(a.session_state["perf_raw"]) > 500_000
    click(a, "Chuẩn hoá & kiểm tra")
    q = a.session_state["perf_quality"].set_index("kiểm tra")["giá trị"]
    assert q["Dòng không ghép được ngày mở"] == 0 and q["ID ở performance nhưng không có ở file chính"] == 0
    a.selectbox(key="v_thr").set_value(60).run()
    click(a, "Vẽ vintage")
    assert a.session_state["vint_info"]["thr"] == 60 and a.session_state["vint_info"]["measure"] == "count"
    vt60 = a.session_state["vint"].copy()
    a.selectbox(key="v_thr").set_value(91).run()
    click(a, "Vẽ vintage")
    vt91 = a.session_state["vint"]
    assert (vt60[12].dropna() >= vt91[12].reindex(vt60[12].dropna().index) - 1e-12).all()
    a.radio(key="v_measure").set_value("balance").run()
    click(a, "Vẽ vintage")
    assert a.session_state["vint_info"]["measure"] == "balance"
    assert a.checkbox(key="use_vsrc").value is True                               # mapping thiếu cột → tự dùng nguồn vintage
    click(a, "Dựng target")
    tgt = a.session_state["tgt"]
    cfg = config.load_default(prefer_real=False)
    full = demo.make_demo()
    ref = target.build_target(full, demo.DEMO_MAPPING, cfg)
    assert int((tgt["status"] == "bad").sum()) == int((ref["status"] == "bad").sum())     # khớp đường dùng cột có sẵn
    # panel demo chỉ có tới MOB 24: khoản bad muộn hơn MOB 24 (ref: indeterminate) hiện ra là good → lệch tối đa bằng số khoản đó
    n_beyond = int(((full["first_bad_mob"] > 24) & (full["fraud_flag"] == 0) & (full["staff_flag"] == 0)).sum())
    diff = int((tgt["status"] == "good").sum()) - int((ref["status"] == "good").sum())
    assert 0 <= diff <= n_beyond


def test_step3_main_wide_dpd_columns():
    import numpy as np, pandas as pd
    from core import performance
    a = AppTest.from_file(PAGE, default_timeout=300)
    a.run()
    click(a, "dữ liệu demo")
    raw = a.session_state["raw"]
    assert performance.detect_layout(raw) == "long"                          # demo gốc không có cột DPD ngang
    wide = raw.copy()
    for m in range(1, 9):
        wide[f"dpd_m{m}"] = np.where(raw["first_bad_mob"].fillna(99) <= m, 95, 0)
    a.session_state["raw"] = wide
    for role in ("dpd_max", "first_bad_mob", "months_observed", "n_payments"):
        a.selectbox(key=f"map_{role}_demo").set_value("—")
    a.run()
    from core import performance as _p
    assert _p.guess_wide_cols(wide) == [f"dpd_m{m}" for m in range(1, 9)]                    # bước 1 tự nhận cột performance ngang
    a.multiselect(key="perf_cols_demo").set_value(_p.guess_wide_cols(wide)).run()
    click(a, "Chốt dữ liệu")
    click(a, "Phân tích chất lượng")
    click(a, "Chốt làm sạch")
    assert not any(c.startswith("dpd_m") for c in a.session_state["feats"])                  # không thành biến mô hình
    a.radio(key="vsrc_kind").set_value(SRC_MAIN).run()
    a.radio(key="v_sub").set_value(B_WIDE).run()
    assert a.multiselect(key="v_wide").value == [f"dpd_m{m}" for m in range(1, 9)]
    click(a, "Chuẩn hoá & kiểm tra")
    assert "perf_long" in a.session_state
    click(a, "Vẽ vintage")
    assert a.session_state["vint"].drop(columns="size").notna().any().any()
