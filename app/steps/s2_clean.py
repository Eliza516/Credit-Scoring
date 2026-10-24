import pandas as pd
import streamlit as st

from app import common, dash
from core import cleaning, dashboard as D, target


def _dashboard(an, feats, cfg):
    ss = st.session_state
    raw, mapping = ss.raw, ss.mapping
    cl = cfg["cleaning"]
    notes = ss.get("col_notes", {})
    counts = an["suggestion"].value_counts().reindex(["keep", "review", "drop"]).fillna(0).astype(int)
    dash.metrics_row([("Giữ", str(counts["keep"])), ("Xem lại", str(counts["review"])), ("Gợi ý loại", str(counts["drop"])), ("Tổng biến", str(len(an)))])
    c1, c2 = st.columns([3, 2])
    with c1:
        mt = D.missing_table(raw, feats)
        st.plotly_chart(dash.missing_bar(mt, cl["max_missing_pct"]), use_container_width=True)
    with c2:
        reasons = an[an["suggestion"] == "drop"]["reason"].str.replace(r"\d+(\.\d+)?%?", "#", regex=True).value_counts().head(8)
        if len(reasons):
            st.plotly_chart(dash.bar_chart(reasons.index.tolist(), reasons.to_numpy(), "Lý do gợi ý loại (số biến)", horizontal=True, height=360), use_container_width=True)
        else:
            st.success("Không có biến nào bị gợi ý loại.")
    st.markdown("**Phân phối của một biến**")
    var = st.selectbox("Chọn biến", feats, key="s2_var", format_func=lambda c: f"{c} — {notes[c][:60]}" if notes.get(c) else c)
    s = raw[var]
    y = target.provisional_y(raw, mapping, cfg)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        dist = D.distribution(s, p_lo=cl["outliers"]["cap_floor_percentiles"][0], p_hi=cl["outliers"]["cap_floor_percentiles"][1])
        a, b = st.columns([3, 2])
        with a:
            if y is not None and y.notna().any():
                h = D.numeric_by_class(raw, y, var)
                if len(h):
                    st.plotly_chart(dash.hist_by_class(h, f"{var}: good vs bad (cắt 0,5%–99,5%)"), use_container_width=True)
            else:
                st.plotly_chart(dash.hist_simple(dist["hist"], f"Phân phối {var}"), use_container_width=True)
        with b:
            st.dataframe(pd.DataFrame({"phân vị": list(dist["quantiles"]), "giá trị": list(dist["quantiles"].values())}), hide_index=True, use_container_width=True)
            st.metric("Ngoại lai theo IQR", f"{dist['outlier_iqr_pct']:.2%}" if dist["outlier_iqr_pct"] == dist["outlier_iqr_pct"] else "—")
            st.metric(f"Ngoài P{cl['outliers']['cap_floor_percentiles'][0]}–P{cl['outliers']['cap_floor_percentiles'][1]}",
                      f"{dist['outlier_pct_pct']:.2%}" if dist["outlier_pct_pct"] == dist["outlier_pct_pct"] else "—")
            sp = cl.get("special_values") or []
            if sp:
                st.metric("Giá trị đặc biệt", f"{s.isin(sp).mean():.2%}")
    else:
        vc = D.value_counts_table(s, top=cfg["dashboard"]["max_categories"] + 5)
        st.plotly_chart(dash.bar_chart(vc["value"], vc["share"], f"Giá trị của {var} (tỷ trọng)", fmt=".0%"), use_container_width=True)
        st.caption(f"{s.nunique():,} giá trị khác nhau; thiếu {s.isna().mean():.1%}.")


def render():
    ss, cfg = st.session_state, common.cfg()
    st.subheader("Bước 2 · Làm sạch dữ liệu")
    if not common.gate("mapping", msg="Hoàn thành bước 1 trước."):
        return
    cl = cfg["cleaning"]
    st.caption(f"Ngưỡng hiện tại: missing > {cl['max_missing_pct']:.0%} · biến chữ > {cl['max_unique_categorical']} giá trị · missing xử lý `{cl['missing']['strategy']}` · "
               f"outlier `{cl['outliers']['strategy']}` · giá trị đặc biệt {cl['special_values'] or 'chưa khai báo'} (sửa ở trang *Quy tắc & ngưỡng*).")
    forced = ss.get("forced_drop", {})
    if forced:
        st.info(f"Hồ sơ dữ liệu yêu cầu loại {len([c for c in forced if c in ss.feats_all])} cột (chỉ có sau khi vay / không rõ thời điểm / định danh). Bạn vẫn có thể tick lại nếu cần.")
    if st.button("Phân tích chất lượng dữ liệu", type="primary"):
        ss.clean_an = cleaning.analyze(ss.raw, ss.feats_all, cfg, forced)
    if "clean_an" not in ss:
        return
    an = ss.clean_an
    n_drop, n_rev = int((an.suggestion == "drop").sum()), int((an.suggestion == "review").sum())
    st.write(f"**Gợi ý theo ngưỡng:** loại {n_drop} biến · xem lại {n_rev} biến · giữ {len(an) - n_drop - n_rev} biến.")
    with st.expander("📊 Dashboard chất lượng dữ liệu", expanded=True):
        _dashboard(an, list(an["variable"]), cfg)
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
        # cột còn cần ở các bước sau (vai trò, ngày/kỳ hạn dùng cho lọc đáo hạn, gợi ý vintage, cột performance) giữ lại trong bảng dù không làm biến
        mt = cfg["target"]["status"].get("maturity") or {}
        hint = (ss.get("profile") or {}).get("vintage_hint") or {}
        keep_cols = {c for c in ss.mapping.values() if c} | {mt.get("term_col"), mt.get("asof_col"), hint.get("first_bad_date_col"), hint.get("asof_col")} | set(ss.get("perf_cols", []))
        df_clean, log = cleaning.apply(ss.raw, cfg, [c for c in drop_vars if c not in keep_cols], feats)
        ss.df_clean, ss.clean_log, ss.feats = df_clean, log, feats
        common.store().save("cleaning", {"log": log, "features": feats}, cfg)
        st.success(f"Đã chốt: loại {len(drop_vars)} biến, còn {len(feats)} biến.")
        if log["impossible_to_nan"]:
            st.info("Giá trị bất khả → missing: " + ", ".join(f"{k} ({v})" for k, v in log["impossible_to_nan"].items()))
