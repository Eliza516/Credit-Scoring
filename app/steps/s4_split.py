import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import common, dash
from core import dashboard as Dd, segmentation, split


def render():
    ss, cfg = st.session_state, common.cfg()
    st.subheader("Bước 4 · Phân tầng (gợi ý từ dữ liệu) & chia Train/Test")
    if not common.gate("target_ok", msg="Chốt target ở bước 3 trước."):
        return
    sp, sg = cfg["split"], cfg["segmentation"]
    mapping = ss.mapping
    tgt, dev = ss.tgt, ss.dev
    D0 = ss.df_clean[dev]
    y0 = tgt.loc[dev, "y"]
    date_col = mapping.get("open_date")

    # ------------------------------------------------------------- phân tầng
    st.markdown("#### Phân tầng")
    if sg["mode"] == "none":
        st.caption("Chế độ phân tầng = none (đổi ở trang *Quy tắc & ngưỡng*).")
    else:
        st.caption(f"Gợi ý theo ngưỡng: mỗi tầng ≥ {sg['min_segment_size_pct']:.0%} mẫu & ≥ {sg['min_bads_per_segment']} bad, bad rate lệch ≥ {sg['min_bad_rate_ratio']}x, "
                   f"Gini tổng tăng ≥ {sg['suggest_if_gini_gain_pp']}pp, tối đa {sg['max_segments']} tầng.")
        if st.button("Phân tích gợi ý tầng"):
            try:
                tmp_parts = split.split_indices(D0, y0, cfg, date_col)
            except ValueError as e:
                st.error(str(e)); return
            with st.spinner("Đang phân tích…"):
                ss.seg_res = segmentation.suggest(D0, y0, ss.feats, cfg, tmp_parts["train"], tmp_parts["test"])
        if "seg_res" in ss:
            r = ss.seg_res
            st.write(f"Gini nền (một mô hình chung, logistic đơn giản): **{r['baseline_gini']:.1f}%**")
            st.dataframe(r["candidates"], use_container_width=True)
            if len(r["tree"]):
                with st.expander("Cây quyết định nông – các lá gợi ý"):
                    st.dataframe(r["tree"].style.format({"size_pct": "{:.1%}", "bad_rate": "{:.1%}"}), use_container_width=True)
                    st.code(r["tree"].attrs.get("tree_text", ""))
    cand = [c for c in (sg["candidate_features"] or []) if c in ss.feats]
    others = [c for c in ss.feats if c not in cand and D0[c].nunique() > 1]
    feats_seg = ["(không phân tầng – mô hình chung)"] + cand + others
    f = st.selectbox("Phân tầng theo biến nào? (quyết định của bạn) – chọn biến để xem dashboard bad rate theo nhóm", feats_seg, index=0, key="seg_feature",
                     format_func=lambda c: f"{c} — {ss.get('col_notes', {}).get(c, '')[:60]}" if ss.get("col_notes", {}).get(c) else c)
    seg = None
    Dseg, y = D0, y0
    if f == feats_seg[0]:
        with st.expander("📊 Dashboard: bad rate chung theo thời gian", expanded=True):
            if date_col and date_col in D0.columns:
                vol = Dd.volume_by_period(D0, date_col, y0, cfg["dashboard"]["time_freq"])
                st.plotly_chart(dash.time_chart(vol, "Khoản vay & bad rate theo kỳ (mô hình chung)"), use_container_width=True)
            st.caption("Chọn một biến ở trên để xem bad rate theo từng nhóm của biến đó và quyết định có tách tầng hay không.")
    else:
        with st.expander(f"📊 Dashboard bad rate theo nhóm của {f}", expanded=True):
            dash.feature_dashboard(D0, y0, ss.feats, cfg, "s4seg", date_col=date_col if date_col in D0.columns else None, notes=ss.get("col_notes", {}),
                                   thresholds={"min_size_pct": sg["min_segment_size_pct"], "min_bads": sg["min_bads_per_segment"]}, fixed_feature=f)
        grp = segmentation.assign_segments(D0, f)
        vals = sorted(grp.unique())
        pick = st.multiselect("Các giá trị thuộc tầng này", vals, default=vals[:1])
        if pick:
            m = grp.isin(pick)
            Dseg, y = D0[m], y0[m]
            seg = {"feature": f, "values": pick, "name": f"{f}={'|'.join(map(str, pick))}"}
            st.info(f"Tầng `{seg['name']}`: {len(Dseg):,} dòng, bad rate {y.mean():.1%}. Chạy lại các bước sau cho từng tầng; mỗi tầng xuất một bảng điểm riêng.")

    # --------------------------------------------------------------- chia mẫu
    st.markdown("#### Chia Train / Test")
    st.caption(f"Phương pháp `{sp['method']}` · train {sp['train_pct']:.0%} / test {sp['test_pct']:.0%} · OOT {'bật' if sp['oot']['enabled'] else 'tắt'} · "
               f"tối thiểu {sp['min_goods']:,} good & {sp['min_bads']:,} bad (không dùng K-fold).")
    if st.button("Chia mẫu", type="primary"):
        try:
            parts = split.split_indices(Dseg, y, cfg, date_col)
        except ValueError as e:
            st.error(str(e)); return
        common.invalidate_from("binning")
        ss.parts, ss.D, ss.y, ss.seg = parts, Dseg, y, seg
        ss.split_tab = split.check_sizes(y, parts, cfg)
    if "parts" in ss:
        tab = ss.split_tab
        st.dataframe(tab.style.format({"bad_rate": "{:.2%}"}), use_container_width=True)
        for w in tab["warning"]:
            if w:
                st.warning("Cảnh báo cỡ mẫu: " + w + " → cân nhắc đổi định nghĩa/window hoặc nới ngưỡng ở trang quy tắc.")
        with st.expander("📊 Dashboard chia mẫu: thời gian, bad rate, phân phối theo tập", expanded=True):
            sm = Dd.split_summary(ss.parts, ss.D, date_col, ss.y)
            st.dataframe(sm.style.format({"bad_rate": "{:.2%}"}), use_container_width=True, hide_index=True)
            st.plotly_chart(dash.bar_chart(sm["set"], sm["bad_rate"], "Bad rate theo tập (chênh lệch lớn = dữ liệu dịch chuyển theo thời gian)", fmt=".1%", height=300), use_container_width=True)
            num = [c for c in ss.feats if pd.api.types.is_numeric_dtype(ss.D[c]) and ss.D[c].nunique() > 8][:60]
            if num:
                v = st.selectbox("So sánh phân phối một biến giữa các tập (phát hiện dịch chuyển dữ liệu)", num, key="s4_shift")
                fig = go.Figure()
                lo, hi = ss.D[v].quantile([0.01, 0.99])
                for k, ix in ss.parts.items():
                    fig.add_histogram(x=ss.D.loc[ix, v].clip(lo, hi), name=k, histnorm="probability", opacity=0.55, nbinsx=30)
                fig.update_layout(barmode="overlay", title=f"Phân phối {v} theo tập", height=320)
                st.plotly_chart(fig, use_container_width=True)
        if date_col:
            d = ss.D[[date_col]].copy()
            d["tập"] = "-"
            for k, ix in ss.parts.items():
                d.loc[ix, "tập"] = k
            d["tháng"] = d[date_col].dt.to_period("M").astype(str)
            g = d.groupby(["tháng", "tập"]).size().reset_index(name="số dòng")
            st.plotly_chart(px.bar(g, x="tháng", y="số dòng", color="tập", title="Phân bố theo thời gian"), use_container_width=True)
        if st.button("✅ Chốt chia mẫu"):
            common.store().save("segmentation", {"segment": ss.seg, "split": ss.split_tab}, cfg)
            st.success("Đã chốt. Sang bước 5 (Binning).")
