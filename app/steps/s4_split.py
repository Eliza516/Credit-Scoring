import plotly.express as px
import streamlit as st

from app import common
from core import segmentation, split


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
    feats_seg = ["(không phân tầng – mô hình chung)"] + list(ss.feats)
    f = st.selectbox("Xây mô hình cho tầng theo biến (quyết định của bạn)", feats_seg, index=0)
    seg = None
    D, y = D0, y0
    if f != feats_seg[0]:
        grp = segmentation.assign_segments(D0, f)
        vals = sorted(grp.unique())
        pick = st.multiselect("Các giá trị thuộc tầng này", vals, default=vals[:1])
        if pick:
            m = grp.isin(pick)
            D, y = D0[m], y0[m]
            seg = {"feature": f, "values": pick, "name": f"{f}={'|'.join(map(str, pick))}"}
            st.info(f"Tầng `{seg['name']}`: {len(D):,} dòng, bad rate {y.mean():.1%}. Chạy lại các bước sau cho từng tầng; mỗi tầng xuất một bảng điểm riêng.")

    # --------------------------------------------------------------- chia mẫu
    st.markdown("#### Chia Train / Test")
    st.caption(f"Phương pháp `{sp['method']}` · train {sp['train_pct']:.0%} / test {sp['test_pct']:.0%} · OOT {'bật' if sp['oot']['enabled'] else 'tắt'} · "
               f"tối thiểu {sp['min_goods']:,} good & {sp['min_bads']:,} bad (không dùng K-fold).")
    if st.button("Chia mẫu", type="primary"):
        try:
            parts = split.split_indices(D, y, cfg, date_col)
        except ValueError as e:
            st.error(str(e)); return
        common.invalidate_from("binning")
        ss.parts, ss.D, ss.y, ss.seg = parts, D, y, seg
        ss.split_tab = split.check_sizes(y, parts, cfg)
    if "parts" in ss:
        tab = ss.split_tab
        st.dataframe(tab.style.format({"bad_rate": "{:.2%}"}), use_container_width=True)
        for w in tab["warning"]:
            if w:
                st.warning("Cảnh báo cỡ mẫu: " + w + " → cân nhắc đổi định nghĩa/window hoặc nới ngưỡng ở trang quy tắc.")
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
