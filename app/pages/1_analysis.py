"""Trang phân tích & xây mô hình: đi hết pipeline, mỗi bước một tab (xem kết quả → chỉnh theo nghiệp vụ → chốt)."""
import streamlit as st

from app import common
from app.steps import s1_data, s2_clean, s3_target, s4_split, s5_binning, s6_selection, s7_model, s8_eval, s9_score

common.init()
st.title("📈 Phân tích & xây mô hình credit scoring")

LABELS = {"ingest": "Dữ liệu", "cleaning": "Làm sạch", "target": "Target & Vintage", "segmentation": "Chia mẫu & Phân tầng",
          "binning": "Binning", "selection": "Chọn biến", "modeling": "Huấn luyện", "evaluation": "Đánh giá", "scorecard": "Bảng điểm & xuất"}
ICON = {"done": "✅", "stale": "♻️ cần chạy lại", "config_changed": "⚙️ cấu hình đã đổi", "pending": "⬜"}
with st.expander("Tiến độ & phiên bản cấu hình của từng bước", expanded=False):
    for r in common.store().status(common.cfg()):
        st.write(f"{ICON[r['state']]} **{LABELS[r['step']]}**" + (f" · cấu hình `{r['config_hash']}`" if r["config_hash"] else ""))

tabs = st.tabs(["1 Dữ liệu", "2 Làm sạch", "3 Target & Vintage", "4 Chia mẫu & Phân tầng", "5 Binning", "6 Chọn biến", "7 Huấn luyện", "8 Đánh giá", "9 Bảng điểm & xuất"])
for tab, mod in zip(tabs, [s1_data, s2_clean, s3_target, s4_split, s5_binning, s6_selection, s7_model, s8_eval, s9_score]):
    with tab:
        mod.render()
