"""Chạy toàn bộ giao diện bằng Streamlit AppTest (không cần trình duyệt) trên dữ liệu demo."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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
    click(at, "Tính vintage")
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
