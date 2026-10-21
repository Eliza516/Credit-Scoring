"""Tiện ích dùng chung cho giao diện Streamlit (chỉ gọi core/, không chứa logic nghiệp vụ)."""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfgmod  # noqa: E402
from core.store import Store  # noqa: E402

# Khoá state theo bước; chạy lại một bước thì các bước sau bị xoá (nguyên tắc 3: quay lại sửa rồi chạy lại).
PIPELINE = [
    ("data", ["raw", "mapping", "feats_all"]),
    ("clean", ["clean_an", "clean_log", "feats", "df_clean"]),
    ("target", ["tgt", "dev", "vint", "target_ok"]),
    ("split", ["parts", "split_tab", "seg", "seg_res", "D", "y"]),
    ("binning", ["specs", "tabs", "tabs_test", "W_tr", "W_te", "W_oot", "bin_confirmed"]),
    ("selection", ["funnel", "final_vars", "sel_confirmed"]),
    ("model", ["model", "steps", "challengers", "resampled", "model_confirmed", "step_notes"]),
    ("eval", ["eval"]),
    ("score", ["points", "cutoff"]),
]
ORDER = [p[0] for p in PIPELINE]


def is_public() -> bool:
    v = os.environ.get("APP_MODE")
    if v is None:
        try:
            v = st.secrets.get("APP_MODE")
        except Exception:  # noqa: BLE001
            v = None
    return str(v or "").lower() == "public"


def init() -> None:
    ss = st.session_state
    if "sid" not in ss:
        ss.sid = uuid.uuid4().hex[:12]
        ss.store = Store(ss.sid)
        ss.cfg = cfgmod.load_default(prefer_real=not is_public())
        ss.cfg_default = cfgmod.clone(ss.cfg)
    ss.setdefault("bundles", {})


def cfg() -> dict:
    return st.session_state.cfg


def store() -> Store:
    return st.session_state.store


def limits() -> dict | None:
    return cfg().get("public_mode") if is_public() else None


def invalidate_from(step: str) -> None:
    """Xoá kết quả của `step` và mọi bước sau nó."""
    i = ORDER.index(step)
    for _, keys in PIPELINE[i:]:
        for k in keys:
            st.session_state.pop(k, None)
    st.session_state.pop("bin_edit", None)


def have(*keys: str) -> bool:
    return all(k in st.session_state for k in keys)


def gate(*keys: str, msg: str) -> bool:
    """Hiện cảnh báo và trả False nếu thiếu kết quả bước trước."""
    if have(*keys):
        return True
    st.info(msg)
    return False


def pct(x) -> str:
    return f"{x:.1%}"


def banner() -> None:
    if is_public() and cfg()["public_mode"].get("show_privacy_banner"):
        st.warning("⚠️ Bản dùng thử công khai: **đừng upload dữ liệu thật/nhạy cảm**. Kết quả lưu theo phiên làm việc trên máy chủ dùng thử; "
                   "hãy dùng dữ liệu demo hoặc dữ liệu đã ẩn danh.")
