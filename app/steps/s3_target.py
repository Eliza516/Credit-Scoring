import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import common
from core import demo, ingest, performance, target, vintage

SRC_NONE = "Không dùng vintage (tự nhập performance window ở trang quy tắc)"
SRC_MAIN = "Chọn trường có sẵn trong file chính"
SRC_UP = "Upload file performance (tuỳ chọn)"
B_SUM = "Cột tổng hợp (MOB/ngày bad đầu tiên + tháng quan sát)"
B_WIDE = "Nhiều cột DPD theo tháng (dạng ngang, vd DPD_M1…DPD_M24)"
NONE_COL = "—"


def _pick(label, cols, default, key, allow_none=False):
    opts = ([NONE_COL] if allow_none else []) + list(cols)
    idx = opts.index(default) if default in opts else 0
    v = st.selectbox(label, opts, index=idx, key=key)
    return None if v == NONE_COL else v


def _perf_limits():
    lim = common.limits()
    if lim:
        lim = dict(lim)
        lim["max_rows"] = lim.get("max_rows_performance", lim["max_rows"])
    return lim


def _open_dates(df, id_col, open_col):
    if df[id_col].duplicated().any():
        raise performance.PerformanceError(f"ID '{id_col}' ở file chính bị trùng – không ghép được performance.")
    return pd.Series(pd.to_datetime(df[open_col]).to_numpy(), index=df[id_col].to_numpy())


def _source_ui(df, mapping, cfg):
    """Vẽ lựa chọn nguồn vintage; trả về (nguồn, tham số) – chưa tính toán."""
    ss = st.session_state
    src = st.radio("Nguồn dữ liệu cho vintage", [SRC_NONE, SRC_MAIN, SRC_UP], key="vsrc_kind")
    cols = list(df.columns)
    P = {"kind": src}
    if src == SRC_NONE:
        return P
    P["id"] = _pick("Cột ID khoản vay (file chính)", cols, mapping.get("id"), "v_id", allow_none=(src == SRC_MAIN))
    P["open"] = _pick("Cột ngày mở (file chính)", cols, mapping.get("open_date"), "v_open")
    if src == SRC_MAIN:
        P["sub"] = st.radio("Kiểu trường có sẵn", [B_SUM, B_WIDE], horizontal=True, key="v_sub")
        if P["sub"] == B_SUM:
            c1, c2 = st.columns(2)
            with c1:
                P["fb_kind"] = st.radio("Cột bad đầu tiên là", ["MOB (số tháng)", "Ngày bad đầu tiên"], key="v_fbk")
                P["fb"] = _pick("Cột MOB/ngày bad đầu tiên (rỗng = chưa bad)", cols, mapping.get("first_bad_mob"), "v_fb")
            with c2:
                P["ob_kind"] = st.radio("Quan sát đến khi nào", ["Cột số tháng đã quan sát", "Cột ngày quan sát", "Một ngày cắt dữ liệu chung"], key="v_obk")
                if P["ob_kind"] == "Một ngày cắt dữ liệu chung":
                    mx = pd.to_datetime(df[P["open"]], errors="coerce").max() if P["open"] else pd.Timestamp.today()
                    P["asof"] = st.date_input("Ngày cắt dữ liệu", value=(mx + pd.DateOffset(years=1)).date(), key="v_asof")
                else:
                    P["ob"] = _pick("Cột số tháng/ngày quan sát", cols, mapping.get("months_observed"), "v_ob")
            st.caption("Ngưỡng bad của các cột này đã cố định theo cách bạn tạo ra chúng (không đổi được trên UI).")
        else:
            P["wide_cols"] = st.multiselect("Các cột DPD theo tháng (tên có hậu tố số = MOB)", cols, default=[c for c in (ss.get("perf_cols") or performance.guess_wide_cols(df)) if c in cols], key="v_wide")
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            up = st.file_uploader("File performance (CSV/Excel) – dạng dọc (khoản vay × tháng) hoặc ngang (DPD_M1…)", type=["csv", "xlsx", "xls"], key="perf_up")
        with c2:
            st.write(""); st.write("")
            if st.button("Dùng performance demo", disabled=ss.get("loaded_key") != "demo", help="Chỉ dùng được khi đã nạp dữ liệu demo ở bước 1"):
                try:
                    ss.perf_raw = ingest.check_limits(demo.make_demo_performance(ss.raw), _perf_limits())
                    ss.perf_name = "demo"
                except ingest.IngestError as e:
                    st.error(str(e))
        if up is not None and ss.get("perf_name") != f"{up.name}:{up.size}":
            try:
                ss.perf_raw = ingest.read_file(up, up.name, _perf_limits(), up.size)
                ss.perf_name = f"{up.name}:{up.size}"
            except ingest.IngestError as e:
                st.error(str(e))
        if "perf_raw" not in ss:
            st.info("Hãy upload file performance, hoặc chọn nguồn khác.")
            return P
        pr = ss.perf_raw
        st.caption(f"Đã nạp {len(pr):,} dòng × {pr.shape[1]} cột ({ss.get('perf_name')}).")
        with st.expander("Xem mẫu file performance"):
            st.dataframe(pr.head(10), use_container_width=True)
        layout = performance.detect_layout(pr)
        P["layout"] = st.radio("Dạng file", ["long", "wide"], index=0 if layout == "long" else 1, horizontal=True,
                               format_func=lambda x: "Dọc (khoản vay × tháng)" if x == "long" else "Ngang (nhiều cột DPD)", key="p_layout")
        pcols = list(pr.columns)
        g = performance.guess_columns(pr)
        P["pid"] = _pick("Cột ID khoản vay (file performance)", pcols, g["id"] or mapping.get("id"), "p_id")
        if P["layout"] == "wide":
            P["wide_cols"] = st.multiselect("Các cột DPD theo tháng", pcols, default=performance.guess_wide_cols(pr), key="p_wide")
        else:
            P["pkind"] = st.radio("Cột kỳ quan sát là", ["Tháng quan sát (ngày / YYYYMM)", "MOB (số tháng kể từ khi mở)"], horizontal=True, key="p_kind")
            P["period"] = _pick("Cột kỳ quan sát", pcols, g["period"], "p_period")
            P["dpd"] = _pick("Cột DPD", pcols, g["dpd"], "p_dpd")
            P["bal"] = _pick("Cột dư nợ (tuỳ chọn, để vintage theo dư nợ)", pcols, g["balance"], "p_bal", allow_none=True)
    return P


def _drop_from_features(cols):
    """Cột dùng để suy ra target/vintage không được là biến mô hình (tránh rò rỉ)."""
    ss = st.session_state
    gone = [c for c in cols if c in ss.get("feats", [])]
    if gone:
        ss.feats = [f for f in ss.feats if f not in gone]
        ss.feats_all = [f for f in ss.feats_all if f not in gone]
        st.session_state["vint_dropped"] = sorted(set(ss.get("vint_dropped", []) + gone))


def _prepare(df, P):
    """Chuẩn hoá nguồn → ss.perf_long (panel) hoặc ss.loans_v (cột tổng hợp). Ném PerformanceError nếu thiếu cấu hình."""
    ss = st.session_state
    if P["kind"] == SRC_MAIN and P["sub"] == B_SUM:
        if not P["open"] or not P.get("fb"):
            raise performance.PerformanceError("Cần chọn cột ngày mở và cột bad đầu tiên.")
        obs_kind = {"Cột số tháng đã quan sát": "mob", "Cột ngày quan sát": "date", "Một ngày cắt dữ liệu chung": "asof"}[P["ob_kind"]]
        if obs_kind != "asof" and not P.get("ob"):
            raise performance.PerformanceError("Cần chọn cột quan sát.")
        _drop_from_features([c for c in (P["fb"], P.get("ob")) if c])
        ss.loans_v = performance.loans_from_main_summary(df, P["open"], P["fb"], "date" if P["fb_kind"].startswith("Ngày") else "mob",
                                                         P.get("ob"), obs_kind, P.get("asof"))
        ss.vsrc = {"kind": "summary"}
        ss.pop("perf_long", None)
        return
    if not P.get("id") or not P.get("open"):
        raise performance.PerformanceError("Cần chọn cột ID và cột ngày mở của file chính để ghép performance.")
    if P["kind"] == SRC_MAIN:
        long = performance.wide_to_long(df, P["id"], P["wide_cols"])
        _drop_from_features(P["wide_cols"])
        ss.perf_quality = None
    else:
        pr = ss.perf_raw
        if P["layout"] == "wide":
            long = performance.wide_to_long(pr, P["pid"], P["wide_cols"])
        else:
            od = _open_dates(df, P["id"], P["open"]) if P["pkind"].startswith("Tháng") else None
            long = performance.normalize_long(pr, P["pid"], P["period"], "date" if P["pkind"].startswith("Tháng") else "mob", P["dpd"], od, P["bal"])
        ss.perf_quality = performance.quality_report(long, df[P["id"]])
    ss.perf_long = long
    ss.vsrc = {"kind": "panel", "id": P["id"], "open": P["open"]}
    ss.pop("loans_v", None)


def _plot_vintage(vt, w, t):
    curves = vt.drop(columns="size")
    fig = go.Figure()
    for cohort, row in curves.iterrows():
        fig.add_scatter(x=list(curves.columns), y=row.values, mode="lines", name=cohort, line=dict(width=1), opacity=0.5)
    fig.add_scatter(x=list(w["curve"].index), y=w["curve"].values, mode="lines", name="Tổng hợp (cùng tập cohort)", line=dict(width=4, color="black"))
    if w["suggested_mob"]:
        fig.add_vline(x=w["suggested_mob"], line_dash="dash", annotation_text="MOB gợi ý")
    fig.add_vline(x=t["performance_window_months"], line_dash="dot", line_color="red", annotation_text="window hiện tại")
    fig.update_layout(title="Bad rate tích lũy (ever-bad) theo MOB – mỗi đường = cohort tháng mở", xaxis_title="MOB", yaxis_title="bad rate tích lũy",
                      yaxis_tickformat=".0%", height=420)
    heat = px.imshow(curves.to_numpy(dtype=float), x=[str(c) for c in curves.columns], y=list(curves.index), aspect="auto",
                     color_continuous_scale="YlOrRd", labels=dict(x="MOB", y="cohort", color="bad rate"), title="Ma trận vintage (cohort × MOB)")
    heat.update_layout(height=max(320, 18 * len(curves) + 120), coloraxis_colorbar=dict(tickformat=".0%"))
    return fig, heat


def render():
    ss, cfg = st.session_state, common.cfg()
    st.subheader("Bước 3 · Định nghĩa target & Vintage")
    if not common.gate("df_clean", msg="Hoàn thành bước 2 trước."):
        return
    t, v = cfg["target"], cfg["vintage"]
    mapping = ss.mapping
    df = ss.df_clean
    st.caption(f"Bad: DPD ≥ {t['bad']['dpd_threshold']} hoặc charge-off/write-off/phá sản · Good: DPD ≤ {t['good']['max_dpd']} · "
               f"Indeterminate: DPD {t['indeterminate']['dpd_range']} · performance window {t['performance_window_months']} tháng "
               f"(đổi ở trang *Quy tắc & ngưỡng*).")

    # ------------------------------------------------------------- Vintage
    st.markdown("#### Vintage – chọn performance window & kiểm tra ổn định cohort (tuỳ chọn)")
    P = _source_ui(df, mapping, cfg)
    panel_src = P["kind"] != SRC_NONE and not (P["kind"] == SRC_MAIN and P.get("sub") == B_SUM)
    if P["kind"] != SRC_NONE:
        ready = P["kind"] == SRC_MAIN or "perf_raw" in ss
        if ready and st.button("Chuẩn hoá & kiểm tra dữ liệu", key="btn_prepare"):
            try:
                _prepare(df, P)
                ss.pop("vint", None)
            except (performance.PerformanceError, KeyError, ValueError) as e:
                st.error(f"Không chuẩn hoá được: {e}")
        if ss.get("vint_dropped"):
            st.info("Đã loại khỏi biến ứng viên (vì dùng để dựng target/vintage, tránh rò rỉ): " + ", ".join(ss["vint_dropped"]))
        if ss.get("perf_quality") is not None and panel_src and "perf_long" in ss:
            with st.expander("Kiểm tra chất lượng dữ liệu performance", expanded=True):
                st.dataframe(ss.perf_quality, use_container_width=True, hide_index=True)
        have_data = ("perf_long" in ss) if panel_src else ("loans_v" in ss)
        if have_data:
            thr = v_thr = t["bad"]["dpd_threshold"]
            measure = "count"
            c1, c2 = st.columns(2)
            if panel_src:
                opts = sorted(set(v.get("dpd_thresholds", []) + [t["bad"]["dpd_threshold"]]))
                with c1:
                    thr = st.selectbox("Ngưỡng DPD tính là bad trong vintage", opts, index=opts.index(t["bad"]["dpd_threshold"]), key="v_thr")
                if "balance" in ss.perf_long.columns:
                    with c2:
                        measure = st.radio("Tính theo", ["count", "balance"], index=0 if v.get("measure", "count") == "count" else 1, horizontal=True,
                                           format_func=lambda x: "số khoản" if x == "count" else "dư nợ", key="v_measure")
            if st.button("Vẽ vintage", type="primary", key="btn_vintage"):
                try:
                    if panel_src:
                        loans = performance.loans_from_panel(ss.perf_long, df, ss.vsrc["id"], ss.vsrc["open"], thr)
                    else:
                        loans = ss.loans_v
                    excl = target.exclusion_flags(df, mapping, cfg)
                    ss.vint = vintage.vintage_from_loans(loans, cfg, excl, weight=(measure == "balance"))
                    ss.vint_info = {"thr": thr if panel_src else None, "measure": measure, "loans": int(len(loans) - excl.sum())}
                except (performance.PerformanceError, KeyError, ValueError) as e:
                    st.error(f"Không vẽ được vintage: {e}")
        if "vint" in ss:
            vt, info = ss.vint, ss.get("vint_info", {})
            if vt.drop(columns="size").notna().sum().sum() == 0:
                st.warning("Chưa có ô nào đủ cỡ mẫu (tối thiểu %s khoản/ô) – kiểm tra cách ghép ID/MOB hoặc hạ `vintage.min_loans_per_cell`." % v.get("min_loans_per_cell", 1))
            else:
                w = vintage.suggest_window(vt, cfg)
                st.info(w["message"] + ("" if w["suggested_mob"] is None else
                                        (" (nằm trong khoảng cho phép)" if w["within_range"] else f" ⚠️ nằm ngoài khoảng {t['performance_window_range']}")))
                st.caption(f"{info.get('loans', '?'):,} khoản vay · ngưỡng DPD {info.get('thr') or 'theo cột đã chọn'} · "
                           f"tính theo {'dư nợ' if info.get('measure') == 'balance' else 'số khoản'} · loại fraud/đã mất/nhân viên/chính sách; "
                           "giữ cả khoản chưa đủ chín (đóng góp cho các MOB đã quan sát).")
                fig, heat = _plot_vintage(vt, w, t)
                st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(heat, use_container_width=True)
                an = vintage.cohort_anomalies(vt, cfg)
                flagged = an[an["flag"]]
                if len(flagged):
                    st.warning(f"{len(flagged)} cohort lệch bất thường (|z| > {v['cohort_anomaly']['zscore_threshold']}) tại MOB {an['at_mob'].iloc[0]}: " + ", ".join(flagged["cohort"]))
                else:
                    st.success("Không có cohort bất thường theo ngưỡng.")
                with st.expander("Bảng vintage & chi tiết cohort"):
                    vt_show = vt.copy()
                    vt_show.columns = [f"MOB {c}" if c != "size" else c for c in vt_show.columns]
                    st.dataframe(vt_show.style.format("{:.2%}", subset=[c for c in vt_show.columns if c != "size"]), use_container_width=True)
                    st.dataframe(an, use_container_width=True)
                    st.download_button("Tải bảng vintage (CSV)", vt_show.to_csv().encode("utf-8-sig"), "vintage.csv", "text/csv")

    # -------------------------------------------------------------- Target
    st.markdown("#### Dựng target")
    use_src = False
    src_ready = (("perf_long" in ss) if panel_src else ("loans_v" in ss)) if P["kind"] != SRC_NONE else False
    if src_ready:
        use_src = st.checkbox("Dùng dữ liệu nguồn vintage ở trên để điền cột cho target (ghi đè ánh xạ ở bước 1)",
                              value=not (mapping.get("dpd_max") and mapping.get("first_bad_mob") and mapping.get("months_observed")), key="use_vsrc")
        if use_src:
            st.caption("Sẽ tính dpd_max, MOB bad đầu tiên (theo ngưỡng bad trong cấu hình target), số tháng quan sát và số kỳ thanh toán từ nguồn này. "
                       "Khoản vay không có performance bị loại “chưa đủ quan sát”." if panel_src else
                       "Sẽ dùng MOB bad đầu tiên và số tháng quan sát từ các cột đã chọn; DPD tối đa vẫn lấy theo ánh xạ ở bước 1.")
    if st.button("Dựng target theo định nghĩa hiện tại", type="primary"):
        df_t, map_t = df, mapping
        try:
            if use_src:
                if panel_src:
                    loans_t = performance.loans_from_panel(ss.perf_long, df, ss.vsrc["id"], ss.vsrc["open"], t["bad"]["dpd_threshold"])
                    fields = ["dpd_max", "first_bad_mob", "months_observed", "n_payments"]
                else:
                    loans_t, fields = ss.loans_v, ["first_bad_mob", "months_observed"]
                df_t, map_t = performance.apply_loans(df, mapping, loans_t, fields)
            errs = performance.target_source_errors(map_t)
            if errs:
                st.error(errs[0])
            else:
                common.invalidate_from("split")
                ss.tgt = target.build_target(df_t, map_t, cfg)
                ss.dev = target.development_mask(ss.tgt, cfg)
        except (performance.PerformanceError, KeyError, ValueError) as e:
            st.error(f"Không dựng được target: {e}")
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
