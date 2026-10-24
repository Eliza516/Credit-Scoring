import os
import time

import pandas as pd
import streamlit as st

from app import common, dash
from core import config as cfgmod, dashboard as D, demo, ingest, performance, profiles, target

ROLE_LABELS = {
    "id": "ID", "open_date": "Ngày mở / T0 (ngày phát hành)", "status_col": "Cột trạng thái dạng chữ (vd loan_status)",
    "target": "Cột target có sẵn (1=bad)", "dpd_max": "DPD tối đa",
    "first_bad_mob": "MOB lần đầu bad", "months_observed": "Số tháng đã quan sát (MOB)", "n_payments": "Số kỳ thanh toán",
    "charge_off_flag": "Cờ charge-off", "write_off_flag": "Cờ write-off", "bankruptcy_flag": "Cờ phá sản", "fraud_flag": "Cờ fraud",
    "deceased_flag": "Cờ đã mất", "staff_vip_flag": "Cờ nhân viên/VIP", "policy_flag": "Cờ tài khoản chính sách",
}
ROLES = {"main": "Dữ liệu chính", "performance": "Performance (DPD theo tháng)", "ignore": "Bỏ qua"}


def _choose_profile():
    ss = st.session_state
    profs = profiles.list_profiles()
    names = list(profs)
    cur = ss.get("profile_name", "generic")
    sel = st.selectbox("Hồ sơ dữ liệu", names, index=names.index(cur) if cur in names else 0, format_func=lambda n: profs[n], key="profile_sel",
                       help="Hồ sơ điền sẵn ánh xạ cột, định dạng ngày, định nghĩa nhãn từ cột trạng thái, cột rò rỉ… cho từng bộ dữ liệu.")
    if sel != ss.get("profile_applied"):
        prof = profiles.load_profile(sel)
        ss.profile_name = sel
        ss.profile_cfg = prof
        ss.cfg = profiles.apply_to_config(ss.cfg_default if ss.get("profile_applied") else ss.cfg, prof)
        ss.profile_applied = sel
        common.store().save_config_version(ss.cfg, f"hồ sơ {sel}")
    prof = ss.get("profile_cfg") or profiles.load_profile(sel)
    if prof.get("description"):
        st.caption(prof["description"])
    return prof


def _load_options(prof, cfg):
    lim = common.limits()
    load = prof.get("load") or {}
    with st.expander("Tuỳ chọn nạp file lớn", expanded=bool(load)):
        c1, c2, c3 = st.columns(3)
        default_rows = int(load.get("default_sample_rows", cfg["ingest"]["default_sample_rows"]))
        with c1:
            max_rows = st.number_input("Cỡ mẫu tối đa (0 = lấy hết)", min_value=0, value=default_rows, step=50_000, key=f"ld_rows_{prof.get('name')}",
                                       help="Lấy mẫu ngẫu nhiên đều từ các dòng thoả bộ lọc; giúp giao diện mượt với file hàng triệu dòng.")
        with c2:
            use_filter = st.checkbox("Áp bộ lọc của hồ sơ khi đọc (chỉ giữ khoản vay có nhãn, từ năm tối thiểu)", value=bool(load), key=f"ld_filter_{prof.get('name')}", disabled=not load)
        with c3:
            use_skip = st.checkbox("Bỏ cột văn bản nặng theo hồ sơ", value=bool(load.get("skip_columns")), key=f"ld_skip_{prof.get('name')}", disabled=not load.get("skip_columns"))
    if lim:
        max_rows = min(max_rows or lim["max_rows"], lim["max_rows"])
    return int(max_rows), use_filter, use_skip


def _read_all(sources, prof, cfg, opts):
    """sources: list[(tên, file-like|đường dẫn)] → list[dict(name, df, info)]"""
    max_rows, use_filter, use_skip = opts
    lim = common.limits()
    keep = profiles.row_filter(prof, cfg) if use_filter else None
    skip = (prof.get("load") or {}).get("skip_columns") if use_skip else None
    out = []
    bar = st.progress(0.0, text="Đang đọc…")
    for i, (name, src) in enumerate(sources):
        lname = name.lower()
        t0 = time.time()
        opened = open(src, "rb") if isinstance(src, str) else None      # đường dẫn → mở file; UploadedFile dùng trực tiếp
        src = opened or src
        if lname.endswith((".xlsx", ".xls")):
            df = ingest.read_file(src, name, lim)
            info = {"rows_read": len(df), "rows_kept": len(df), "sampled": False}
        else:
            def prog(r, a, _n=name):
                bar.progress((i + min(r / 2_500_000, 0.98)) / len(sources), text=f"{_n}: đã đọc {r:,} dòng, giữ {a:,}")
            df, info = ingest.read_csv_chunked(src, name, skip_columns=skip, chunk_rows=cfg["ingest"]["chunk_rows"], keep=keep, max_rows=max_rows,
                                               seed=cfg.get("seed", 42), progress=prog)
            df = ingest.check_limits(df, lim)
        if opened:
            opened.close()
        info["seconds"] = round(time.time() - t0, 1)
        out.append({"name": name, "df": df, "info": info})
    bar.empty()
    return out


def _roles_ui(files, cfg):
    """Gom file theo cấu trúc, cho gán vai trò. Trả (df chính, df performance|None) hoặc None nếu chưa đủ."""
    cols = [list(f["df"].columns) for f in files]
    groups = ingest.group_by_schema(cols, cfg["ingest"]["group_min_overlap"])
    rows = []
    sizes = [sum(len(files[i]["df"]) for i in g) for g in groups]
    largest = max(range(len(groups)), key=lambda k: sizes[k])
    st.markdown("**File đã nạp & nhóm theo cấu trúc**")
    sel = {}
    for k, g in enumerate(groups):
        frames = [files[i]["df"] for i in g]
        sample = frames[0]
        sug = ingest.suggest_group_role(sample, k == largest)
        c1, c2, c3 = st.columns([4, 2, 2])
        c1.write(f"**Nhóm {k + 1}** · {len(g)} file: " + ", ".join(files[i]["name"] for i in g))
        c2.write(f"{sizes[k]:,} dòng · {len(sample.columns)} cột")
        sel[k] = c3.selectbox("Vai trò", list(ROLES), index=list(ROLES).index(sug), format_func=ROLES.get, key=f"role_{k}_{'|'.join(map(str, g))}", label_visibility="collapsed")
    mains = [k for k, r in sel.items() if r == "main"]
    if len(mains) != 1:
        st.warning("Cần đúng 1 nhóm file làm *Dữ liệu chính*.")
        return None
    names = lambda ks: [files[i]["name"] for i in groups[ks]]  # noqa: E731
    main = ingest.combine_frames([files[i]["df"] for i in groups[mains[0]]], names(mains[0]))
    perf = [k for k, r in sel.items() if r == "performance"]
    perf_df = ingest.combine_frames([files[i]["df"] for k in perf for i in groups[k]], [files[i]["name"] for k in perf for i in groups[k]]) if perf else None
    return main, perf_df


def _dashboard(raw, mapping, prof, cfg, file_rows=None):
    """Dashboard tổng quan dữ liệu vừa nạp (trước khi chốt)."""
    notes = profiles.column_notes(prof)
    date_col, id_col, st_col = mapping.get("open_date"), mapping.get("id"), mapping.get("status_col")
    view = raw
    fm = cfg["columns"].get("date_formats", {})
    if date_col and date_col in raw.columns and not pd.api.types.is_datetime64_any_dtype(raw[date_col]):
        view = raw.copy(deep=False)
        view[date_col] = pd.to_datetime(raw[date_col], format=fm.get(date_col), errors="coerce")
    ov = D.overview(raw, id_col)
    dash.metrics_row([("Số dòng", f"{ov['rows']:,}"), ("Số cột", str(ov["cols"])), ("Bộ nhớ", f"{ov['memory_mb']:.0f} MB"),
                      ("Ô thiếu", f"{ov['missing_cells_pct']:.1%}"), ("ID trùng", f"{ov.get('duplicate_ids', '—')}")])
    y = target.provisional_y(view, mapping, cfg)
    c1, c2 = st.columns(2)
    with c1:
        k = ov["kinds"]
        st.plotly_chart(dash.bar_chart(list(k), list(k.values()), "Số cột theo kiểu dữ liệu"), use_container_width=True)
    with c2:
        if file_rows and len(file_rows) > 1:
            st.plotly_chart(dash.bar_chart([n for n, _ in file_rows], [r for _, r in file_rows], "Số dòng theo file (trước khi gộp)", height=340), use_container_width=True)
        elif st_col and st_col in raw.columns:
            vc = D.value_counts_table(raw[st_col])
            st.plotly_chart(dash.bar_chart(vc["value"], vc["count"], f"Phân bố {st_col}", horizontal=True), use_container_width=True)
        else:
            mt = D.missing_table(raw).head(15)
            st.plotly_chart(dash.missing_bar(mt, top=15), use_container_width=True)
    if date_col and date_col in view.columns and pd.api.types.is_datetime64_any_dtype(view[date_col]):
        vol = D.volume_by_period(view, date_col, y, cfg["dashboard"]["time_freq"])
        st.plotly_chart(dash.time_chart(vol, f"Số khoản vay & bad rate theo kỳ ({date_col})"), use_container_width=True)
    if st_col and file_rows and len(file_rows) > 1 and st_col in raw.columns:
        vc = D.value_counts_table(raw[st_col])
        st.plotly_chart(dash.bar_chart(vc["value"], vc["count"], f"Phân bố {st_col}", horizontal=True), use_container_width=True)
    if y is not None and y.notna().any():
        st.markdown("**Bad rate theo biến** (tạm tính theo cách khai báo nhãn hiện tại, chưa lọc đáo hạn/loại trừ)")
        excluded = {mapping.get(k) for k in mapping}
        feats = [c for c in raw.columns if c not in excluded and not c.startswith("_") and raw[c].nunique() > 1]
        dash.feature_dashboard(view, y, feats, cfg, "s1", date_col=date_col if date_col in view.columns else None, notes=notes)


def render():
    ss, cfg = st.session_state, common.cfg()
    st.subheader("Bước 1 · Nạp dữ liệu (nhiều file) & ánh xạ cột")
    prof = _choose_profile()
    cfg = common.cfg()
    lim = common.limits()
    if lim:
        st.caption(f"Giới hạn bản dùng thử: ≤ {lim['max_upload_mb']}MB/file · ≤ {lim['max_rows']:,} dòng · ≤ {lim['max_columns']} cột.")
    opts = _load_options(prof, cfg)

    tab_up, tab_path, tab_demo = st.tabs(["Upload file (chọn nhiều file một lúc)", "Đường dẫn trên máy chủ (file lớn)", "Dữ liệu demo"])
    sources = None
    with tab_up:
        ups = st.file_uploader("CSV / CSV.GZ / Excel – có thể chọn nhiều file", type=["csv", "gz", "xlsx", "xls"], accept_multiple_files=True, key="uploader")
        if ups and st.button("Nạp các file đã chọn", type="primary", key="btn_load_up"):
            bad = [u.name for u in ups if lim and u.size > lim["max_upload_mb"] * 1024 * 1024]
            if bad:
                st.error("File vượt giới hạn dung lượng: " + ", ".join(bad))
            else:
                sources = [(u.name, u) for u in ups]
    with tab_path:
        if common.is_public():
            st.info("Bản dùng thử công khai không đọc file từ máy chủ.")
        else:
            txt = st.text_area("Mỗi dòng một đường dẫn file (vd /data/accepted_2007_to_2018Q4.csv.gz)", key="paths", height=90)
            if st.button("Nạp các file theo đường dẫn", key="btn_load_path") and txt.strip():
                paths = [p.strip() for p in txt.splitlines() if p.strip()]
                miss = [p for p in paths if not os.path.isfile(p)]
                if miss:
                    st.error("Không thấy file: " + ", ".join(miss))
                else:
                    sources = [(os.path.basename(p), p) for p in paths]
    with tab_demo:
        c1, c2 = st.columns(2)
        if c1.button("Dùng dữ liệu demo (tổng hợp, kiểu DPD)", key="btn_demo"):
            try:
                df = ingest.check_limits(demo.make_demo(), lim)
                ss.pending = {"files": [{"name": "demo", "df": df, "info": {"rows_read": len(df), "rows_kept": len(df), "sampled": False}}], "demo": True}
                ss.pop("loaded_files", None)
            except ingest.IngestError as e:
                st.error(str(e))
        if c2.button("Dùng dữ liệu demo kiểu Lending Club (mô phỏng, cần chọn hồ sơ Lending Club)", key="btn_demo_lc"):
            try:
                df = ingest.check_limits(demo.make_lending_club_like(), lim)
                ss.pending = {"files": [{"name": "lending_club_like", "df": df, "info": {"rows_read": len(df), "rows_kept": len(df), "sampled": False}}], "demo": False}
                ss.pop("loaded_files", None)
            except ingest.IngestError as e:
                st.error(str(e))
    if sources:
        try:
            files = _read_all(sources, prof, cfg, opts)
            ss.pending = {"files": files, "demo": False}
        except (ingest.IngestError, ValueError, OSError) as e:
            st.error(f"Không nạp được: {e}")

    pend = ss.get("pending")
    if not pend:
        if "raw" not in ss:
            st.info("Hãy upload một hoặc nhiều file, nhập đường dẫn, hoặc dùng dữ liệu demo.")
            return
    else:
        files = pend["files"]
        with st.expander("Báo cáo nạp file", expanded=len(files) > 1):
            st.dataframe(pd.DataFrame([{"file": f["name"], "dòng đọc": f["info"]["rows_read"], "dòng giữ lại": f["info"]["rows_kept"],
                                         "lấy mẫu": f["info"]["sampled"], "giây": f["info"].get("seconds", 0), "cột": f["df"].shape[1]} for f in files]),
                         use_container_width=True, hide_index=True)
        res = _roles_ui(files, cfg)
        if res is None:
            return
        main, perf_df = res
        if st.button("Dùng dữ liệu này", type="primary", key="btn_use"):
            common.invalidate_from("data")
            ss.raw = main
            ss.pop("perf_raw", None)
            if perf_df is not None:
                ss.perf_raw, ss.perf_name = perf_df, "file performance đã nạp"
            ss.loaded_key = "demo" if pend["demo"] else f"files:{'|'.join(f['name'] for f in files)}"
            ss.loaded_files = [{"name": f["name"], "rows": len(f["df"])} for f in files]
            m = demo.DEMO_MAPPING if pend["demo"] else profiles.mapping_for(main.columns, prof)
            ss.mapping_draft = dict(m)
            ss.pop("pending", None)
            common.store().log(f"nạp dữ liệu {ss.loaded_key}: {main.shape}")
            st.rerun()
        if "raw" not in ss:
            return

    raw = ss.raw
    key_sfx = ss.get("loaded_key", "x")
    st.success(f"Đã nạp {len(raw):,} dòng × {raw.shape[1]} cột" + (f" từ {len(ss.loaded_files)} file." if ss.get("loaded_files") else "."))
    with st.expander("Xem mẫu dữ liệu & hồ sơ cột"):
        st.dataframe(raw.head(20), use_container_width=True)
        st.dataframe(ingest.profile(raw), use_container_width=True)

    st.markdown("**Ánh xạ vai trò cột**")
    draft = ss.get("mapping_draft") or {k: None for k in ROLE_LABELS}
    cols = ["—"] + list(raw.columns)
    mapping = {}
    grid = st.columns(3)
    for i, (k, lab) in enumerate(ROLE_LABELS.items()):
        cur = draft.get(k)
        with grid[i % 3]:
            sel = st.selectbox(lab, cols, index=cols.index(cur) if cur in cols else 0, key=f"map_{k}_{key_sfx}")
        mapping[k] = None if sel == "—" else sel
    perf_cols = st.multiselect("Cột performance theo tháng nằm ngay trong file chính (vd DPD_M1…DPD_M24) – dùng cho vintage, KHÔNG làm biến mô hình",
                               list(raw.columns), default=performance.guess_wide_cols(raw), key=f"perf_cols_{key_sfx}")
    errs = ingest.validate_mapping(raw, mapping)
    for e in errs:
        st.error(e)
    if mapping.get("status_col"):
        st_cfg = cfg["target"]["status"]
        vals = sorted(raw[mapping["status_col"]].dropna().astype(str).unique())[:60]
        c1, c2 = st.columns(2)
        with c1:
            bad = st.multiselect("Giá trị trạng thái = BAD", vals, default=[v for v in st_cfg["bad"] if v in vals], key=f"st_bad_{key_sfx}")
        with c2:
            good = st.multiselect("Giá trị trạng thái = GOOD", vals, default=[v for v in st_cfg["good"] if v in vals], key=f"st_good_{key_sfx}")
        st.caption("Trạng thái còn lại (vd Current, Late…) là *chưa xác định* và bị loại khỏi mẫu phát triển. Quyết định nghiệp vụ do bạn chốt.")
        if (bad, good) != (st_cfg["bad"], st_cfg["good"]):
            cfgmod.set_(cfg, "target.status.bad", bad)
            cfgmod.set_(cfg, "target.status.good", good)
    else:
        st.caption("dpd_max, first_bad_mob, months_observed, n_payments là tuỳ chọn ở bước này: có thể để trống rồi nạp file performance ở bước 3 để tự tính.")

    with st.expander("📊 Dashboard dữ liệu", expanded=True):
        _dashboard(raw, {**mapping}, prof, cfg, [(f["name"], f["rows"]) for f in ss.get("loaded_files", [])])

    if st.button("✅ Chốt dữ liệu & ánh xạ", type="primary", disabled=bool(errs)):
        common.invalidate_from("clean")
        fm = cfg["columns"].get("date_formats", {})
        df2 = ingest.apply_mapping(raw, mapping, fm)
        df2, made = profiles.apply_derived(df2, prof)
        ss.raw = df2
        ss.derived_cols = made
        ss.mapping = mapping
        ss.profile = prof
        ss.forced_drop = profiles.excluded_features(prof)
        ss.col_notes = profiles.column_notes(prof)
        extra = ([mapping["first_bad_mob"]] if mapping.get("first_bad_mob") else []) + list(perf_cols)
        ss.perf_cols = list(perf_cols)
        ss.feats_all = ingest.feature_columns(df2, mapping, extra)
        common.store().save("ingest", {"mapping": mapping, "shape": df2.shape, "features": ss.feats_all, "profile": prof.get("name"), "derived": made}, cfg)
        st.success(f"Chốt xong: {len(ss.feats_all)} biến ứng viên" + (f" (đã tạo {len(made)} cột suy ra: {', '.join(made)})" if made else "") + ". Sang bước 2.")
    if "mapping" in ss:
        st.caption(f"Đã chốt: {len(ss.feats_all)} biến ứng viên.")
