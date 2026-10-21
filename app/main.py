"""Điểm vào Streamlit: 3 trang (Phân tích & xây mô hình · Quản trị ngưỡng · Hướng dẫn)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Credit Scoring", page_icon="📊", layout="wide")

from app import common  # noqa: E402

common.init()

pages = [
    st.Page("pages/1_analysis.py", title="Phân tích & xây mô hình", icon="📈", default=True),
    st.Page("pages/2_rules.py", title="Quy tắc & ngưỡng nghiệp vụ", icon="⚙️"),
    st.Page("pages/3_docs.py", title="Hướng dẫn & tài liệu", icon="📚"),
]
with st.sidebar:
    st.caption(f"Phiên: `{st.session_state.sid}`" + (" · chế độ dùng thử công khai" if common.is_public() else ""))
nav = st.navigation(pages)
common.banner()
nav.run()
