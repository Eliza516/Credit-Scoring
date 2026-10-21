import pandas as pd
import streamlit as st

from app import common
from core import demo, ingest

ROLE_LABELS = {
    "id": "ID", "open_date": "Ngày mở / T0", "target": "Cột target có sẵn (1=bad)", "dpd_max": "DPD tối đa",
    "first_bad_mob": "MOB lần đầu bad", "months_observed": "Số tháng đã quan sát (MOB)", "n_payments": "Số kỳ thanh toán",
    "charge_off_flag": "Cờ charge-off", "write_off_flag": "Cờ write-off", "bankruptcy_flag": "Cờ phá sản", "fraud_flag": "Cờ fraud",
    "deceased_flag": "Cờ đã mất", "staff_vip_flag": "Cờ nhân viên/VIP", "policy_flag": "Cờ tài khoản chính sách",
}


def render():
    ss, cfg = st.session_state, common.cfg()
    st.subheader("Bước 1 · Nạp dữ liệu & ánh xạ cột")
    lim = common.limits()
    if lim:
        st.caption(f"Giới hạn bản dùng thử: ≤ {lim['max_upload_mb']}MB · ≤ {lim['max_rows']:,} dòng · ≤ {lim['max_columns']} cột.")
    c1, c2 = st.columns([3, 1])
    with c1:
        up = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
    with c2:
        st.write("")
        st.write("")
        use_demo = st.button("Dùng dữ liệu demo (tổng hợp)")
    df = None
    if up is not None:
        key = f"{up.name}:{up.size}"
        if ss.get("loaded_key") != key:
            try:
                df = ingest.read_file(up, up.name, lim, up.size)
                ss.loaded_key = key
                ss.demo_mapping = None
            except ingest.IngestError as e:
                st.error(str(e))
    elif use_demo:
        try:
            df = ingest.check_limits(demo.make_demo(), lim)
            ss.loaded_key = "demo"
            ss.demo_mapping = dict(demo.DEMO_MAPPING)
        except ingest.IngestError as e:
            st.error(str(e))
    if df is not None:
        common.invalidate_from("data")
        ss.raw = df
        ss.mapping_draft = ss.demo_mapping or {k: cfg["columns"].get(k) for k in ROLE_LABELS}
        common.store().log(f"nạp dữ liệu {ss.loaded_key}: {df.shape}")
    if "raw" not in ss:
        st.info("Hãy upload file hoặc dùng dữ liệu demo.")
        return
    raw = ss.raw
    st.success(f"Đã nạp {len(raw):,} dòng × {raw.shape[1]} cột.")
    st.dataframe(raw.head(20), use_container_width=True)
    with st.expander("Hồ sơ cột (kiểu, missing, số giá trị)"):
        st.dataframe(ingest.profile(raw), use_container_width=True)

    st.markdown("**Ánh xạ vai trò cột** (cột không có vai trò sẽ là biến ứng viên)")
    draft = ss.get("mapping_draft") or {k: None for k in ROLE_LABELS}
    cols = ["—"] + list(raw.columns)
    mapping = {}
    grid = st.columns(3)
    for i, (k, lab) in enumerate(ROLE_LABELS.items()):
        cur = draft.get(k)
        with grid[i % 3]:
            sel = st.selectbox(lab, cols, index=cols.index(cur) if cur in cols else 0, key=f"map_{k}_{ss.get('loaded_key')}")
        mapping[k] = None if sel == "—" else sel
    errs = ingest.validate_mapping(raw, mapping)
    for e in errs:
        st.error(e)
    if mapping.get("first_bad_mob") is None or mapping.get("open_date") is None or mapping.get("months_observed") is None:
        st.caption("Thiếu open_date / first_bad_mob / months_observed → không dựng được Vintage (các bước khác vẫn chạy).")
    if st.button("✅ Chốt dữ liệu & ánh xạ", type="primary", disabled=bool(errs)):
        common.invalidate_from("clean")
        df2 = ingest.apply_mapping(raw, mapping)
        ss.raw = df2
        ss.mapping = mapping
        extra = [mapping["first_bad_mob"]] if mapping.get("first_bad_mob") else []
        ss.feats_all = ingest.feature_columns(df2, mapping, extra)
        common.store().save("ingest", {"mapping": mapping, "shape": df2.shape, "features": ss.feats_all}, cfg)
        st.success(f"Chốt xong: {len(ss.feats_all)} biến ứng viên. Sang bước 2.")
    if "mapping" in ss:
        st.caption(f"Đã chốt: {len(ss.feats_all)} biến ứng viên.")
