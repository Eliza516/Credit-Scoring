import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import common
from core import target, vintage


def render():
    ss, cfg = st.session_state, common.cfg()
    st.subheader("Bước 3 · Định nghĩa target & Vintage")
    if not common.gate("df_clean", msg="Hoàn thành bước 2 trước."):
        return
    t = cfg["target"]
    mapping = ss.mapping
    st.caption(f"Bad: DPD ≥ {t['bad']['dpd_threshold']} hoặc charge-off/write-off/phá sản · Good: DPD ≤ {t['good']['max_dpd']} · "
               f"Indeterminate: DPD {t['indeterminate']['dpd_range']} · performance window {t['performance_window_months']} tháng "
               f"(đổi ở trang *Quy tắc & ngưỡng*).")
    # ---- Vintage
    st.markdown("#### Vintage – chọn performance window & kiểm tra ổn định cohort")
    if all(mapping.get(k) for k in ("open_date", "first_bad_mob", "months_observed")):
        if st.button("Tính vintage"):
            tgt0 = target.build_target(ss.raw, mapping, cfg)
            ss.vint = vintage.vintage_table(ss.raw, mapping, cfg, tgt0["status"])
        if "vint" in ss:
            vt = ss.vint
            w = vintage.suggest_window(vt, cfg)
            st.info(w["message"] + ("" if w["suggested_mob"] is None else
                                      (" (nằm trong khoảng cho phép)" if w["within_range"] else f" ⚠️ nằm ngoài khoảng {cfg['target']['performance_window_range']}")))
            curves = vt.drop(columns="size")
            fig = go.Figure()
            for cohort, row in curves.iterrows():
                fig.add_scatter(x=list(curves.columns), y=row.values, mode="lines", name=cohort, line=dict(width=1), opacity=0.5)
            fig.add_scatter(x=list(w["curve"].index), y=w["curve"].values, mode="lines", name="Trung bình có trọng số", line=dict(width=4, color="black"))
            if w["suggested_mob"]:
                fig.add_vline(x=w["suggested_mob"], line_dash="dash", annotation_text="MOB gợi ý")
            fig.add_vline(x=t["performance_window_months"], line_dash="dot", line_color="red", annotation_text="window hiện tại")
            fig.update_layout(title="Bad rate tích lũy theo MOB (mỗi đường = cohort tháng mở)", xaxis_title="MOB", yaxis_title="bad rate tích lũy", yaxis_tickformat=".0%", height=420)
            st.plotly_chart(fig, use_container_width=True)
            an = vintage.cohort_anomalies(vt, cfg)
            flagged = an[an["flag"]]
            if len(flagged):
                st.warning(f"{len(flagged)} cohort lệch bất thường (|z| > {cfg['vintage']['cohort_anomaly']['zscore_threshold']}) tại MOB {an['at_mob'].iloc[0]}: "
                           + ", ".join(flagged["cohort"]))
            else:
                st.success("Không có cohort bất thường theo ngưỡng.")
            with st.expander("Bảng vintage & chi tiết cohort"):
                vt_show = vt.copy()
                vt_show.columns = [f"MOB {c}" if c != "size" else c for c in vt_show.columns]
                st.dataframe(vt_show.style.format("{:.2%}", subset=[c for c in vt_show.columns if c != "size"]), use_container_width=True)
                st.dataframe(an, use_container_width=True)
    else:
        st.caption("Cần ánh xạ open_date, first_bad_mob, months_observed để chạy vintage.")

    # ---- Target
    st.markdown("#### Dựng target")
    if st.button("Dựng target theo định nghĩa hiện tại", type="primary"):
        common.invalidate_from("split")
        ss.tgt = target.build_target(ss.df_clean, mapping, cfg)
        ss.dev = target.development_mask(ss.tgt, cfg)
    if "tgt" in ss:
        tgt = ss.tgt
        summ = target.summary(tgt, cfg)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(summ.style.format({"pct": "{:.1%}"}), use_container_width=True)
            st.metric("Bad rate mẫu phát triển", common.pct(target.bad_rate(tgt)))
        with c2:
            st.plotly_chart(px.pie(summ.reset_index(names="status"), names="status", values="count", hole=0.4, title="Phân bố trạng thái"), use_container_width=True)
        st.dataframe(target.exclusion_reasons(tgt), use_container_width=True)
        n_dev = int(ss.dev.sum())
        if n_dev == 0 or tgt["y"].nunique() < 2:
            st.error("Mẫu phát triển rỗng hoặc chỉ có một lớp – kiểm tra định nghĩa bad/good.")
            return
        if st.button("✅ Chốt target"):
            common.store().save("target", {"status_counts": summ, "window": cfg["target"]["performance_window_months"]}, cfg)
            ss.target_ok = True
            st.success("Đã chốt target. Sang bước 4.")
