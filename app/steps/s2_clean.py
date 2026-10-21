import pandas as pd
import streamlit as st

from app import common
from core import cleaning


def render():
    ss, cfg = st.session_state, common.cfg()
    st.subheader("Bước 2 · Làm sạch dữ liệu")
    if not common.gate("mapping", msg="Hoàn thành bước 1 trước."):
        return
    cl = cfg["cleaning"]
    st.caption(f"Ngưỡng hiện tại: missing > {cl['max_missing_pct']:.0%} · missing xử lý `{cl['missing']['strategy']}` · outlier `{cl['outliers']['strategy']}` · "
               f"giá trị đặc biệt {cl['special_values'] or 'chưa khai báo'} (sửa ở trang *Quy tắc & ngưỡng*).")
    if st.button("Phân tích chất lượng dữ liệu", type="primary"):
        ss.clean_an = cleaning.analyze(ss.raw, ss.feats_all, cfg)
    if "clean_an" not in ss:
        return
    an = ss.clean_an
    n_drop, n_rev = int((an.suggestion == "drop").sum()), int((an.suggestion == "review").sum())
    st.write(f"**Gợi ý theo ngưỡng:** loại {n_drop} biến · xem lại {n_rev} biến · giữ {len(an) - n_drop - n_rev} biến.")
    view = an.copy()
    view["loại_bỏ"] = view["suggestion"] == "drop"
    edited = st.data_editor(
        view[["variable", "dtype", "missing_pct", "special_pct", "n_unique", "outlier_pct", "impossible_pct", "suggestion", "reason", "loại_bỏ"]],
        use_container_width=True, hide_index=True, disabled=[c for c in view.columns if c != "loại_bỏ"], key="clean_editor",
        column_config={"missing_pct": st.column_config.NumberColumn(format="%.2f%%", help="tỷ lệ"), "loại_bỏ": st.column_config.CheckboxColumn("Loại bỏ?")})
    iv = cleaning.impossible_values(ss.raw, cfg)
    if len(iv):
        st.write("Vi phạm khoảng hợp lệ:")
        st.dataframe(iv, use_container_width=True)
    drop_vars = edited.loc[edited["loại_bỏ"], "variable"].tolist()
    if st.button("✅ Chốt làm sạch"):
        common.invalidate_from("target")
        feats = [f for f in ss.feats_all if f not in drop_vars]
        df_clean, log = cleaning.apply(ss.raw, cfg, drop_vars, feats)
        ss.df_clean, ss.clean_log, ss.feats = df_clean, log, feats
        common.store().save("cleaning", {"log": log, "features": feats}, cfg)
        st.success(f"Đã chốt: loại {len(drop_vars)} biến, còn {len(feats)} biến.")
        if log["impossible_to_nan"]:
            st.info("Giá trị bất khả → missing: " + ", ".join(f"{k} ({v})" for k, v in log["impossible_to_nan"].items()))
