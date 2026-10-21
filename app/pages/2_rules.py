"""Dashboard quy tắc & ngưỡng nghiệp vụ: chỉnh tương tác, xem trước tác động, lưu/tải/so sánh phiên bản."""
import streamlit as st

from app import common
from core import binning, config as cfgmod, evaluation, scorecard

common.init()
ss = st.session_state
ss.setdefault("cfg_ver", 0)

ENUMS = {
    "split.method": ["time", "random"],
    "split.imbalance.method": [None] + cfgmod.IMBALANCE_METHODS,
    "split.imbalance.apply_to": ["train_only"],
    "segmentation.mode": ["suggest", "none", "manual"],
    "cleaning.missing.strategy": ["own_bin", "impute", "drop_records"],
    "cleaning.missing.impute_numeric": ["median", "mean"],
    "cleaning.missing.impute_categorical": ["mode"],
    "cleaning.outliers.strategy": ["coarse_binning", "cap_floor", "none"],
    "binning.monotonic_trend": ["auto", "ascending", "descending", "none"],
    "selection.correlation.keep_higher": ["iv"],
    "modeling.primary": ["logistic_woe"],
    "modeling.logistic.penalty": ["none", "l2"],
    "modeling.stepwise.direction": ["forward", "backward", "both"],
    "modeling.stepwise.criterion": ["pvalue", "aic", "bic"],
    "modeling.stepwise.expected_sign_on_woe": ["negative", "positive"],
    "modeling.reject_inference.method": ["augmentation"],
    "cutoff.strategy": ["target_bad_rate", "target_approval_rate", "profit"],
    "evaluation.ml_metrics.decision_threshold.mode": ["by_cutoff", "max_f1", "fixed"],
    "vintage.cohort_granularity": ["month"],
    "scorecard.higher_score_means": ["lower_risk"],
    "cutoff.zones": None,
}
TITLES = {
    "columns": "1. Ánh xạ cột", "target": "2. Định nghĩa target", "vintage": "3. Vintage", "split": "4. Chia mẫu (Train/Test) & mất cân bằng",
    "segmentation": "5. Phân tầng", "cleaning": "6. Làm sạch", "binning": "7. Binning", "selection": "8. Chọn biến",
    "modeling": "9. Mô hình & stepwise", "evaluation": "10. Đánh giá", "scorecard": "11. Bảng điểm", "cutoff": "12. Cutoff", "public_mode": "13. Chế độ public",
}
SKIP = {"version"}


def parse_free(txt: str):
    t = txt.strip()
    if t == "" or t.lower() in ("null", "none"):
        return None
    for conv in (int, float):
        try:
            return conv(t)
        except ValueError:
            pass
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    return t


def parse_list(txt: str):
    items = [parse_free(x) for x in txt.split(",") if x.strip() != ""]
    return items


def render(node, path: str):
    """Vẽ widget cho node cấu hình, trả về giá trị mới."""
    key = f"{ss.cfg_ver}::{path}"
    label = path.split(".")[-1]
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            p = f"{path}.{k}" if path else k
            if isinstance(v, dict):
                with st.expander(k, expanded=True):
                    out[k] = render(v, p)
            else:
                out[k] = render(v, p)
        return out
    if path in ENUMS and ENUMS[path] is not None:
        opts = ENUMS[path]
        idx = opts.index(node) if node in opts else 0
        fmt = (lambda x: "— (bạn phải chọn)" if x is None else str(x))
        return st.selectbox(label, opts, index=idx, key=key, format_func=fmt)
    if isinstance(node, bool):
        return st.checkbox(label, value=node, key=key)
    if isinstance(node, int):
        return int(st.number_input(label, value=node, step=1, key=key))
    if isinstance(node, float):
        return float(st.number_input(label, value=node, step=0.01, format="%.4f", key=key))
    if isinstance(node, list):
        txt = st.text_input(label + " (cách nhau dấu phẩy)", value=", ".join(map(str, node)), key=key)
        return parse_list(txt)
    if node is None:
        txt = st.text_input(label + " (trống = chưa chốt)", value="", key=key)
        return parse_free(txt)
    return st.text_input(label, value=str(node), key=key)


st.title("⚙️ Quy tắc & ngưỡng nghiệp vụ")
st.caption("Mọi ngưỡng ở đây chỉ **kích hoạt gợi ý** ở trang phân tích; bạn vẫn duyệt từng gợi ý. Chỉnh xong bấm **Áp dụng**.")

cur = ss.cfg
errs = cfgmod.validate(cur)
for e in errs:
    st.error(e)

c1, c2, c3, c4 = st.columns(4)
with c1:
    up = st.file_uploader("Tải business.yaml", type=["yaml", "yml"], label_visibility="collapsed")
    if up is not None and st.button("Dùng file đã tải"):
        try:
            new = cfgmod.loads(up.getvalue().decode("utf-8"))
            ss.cfg = new
            ss.cfg_ver += 1
            common.store().save_config_version(new, "tải lên")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"File không hợp lệ: {e}")
with c2:
    if st.button("Đặt lại mặc định (example)"):
        ss.cfg = cfgmod.clone(ss.cfg_default)
        ss.cfg_ver += 1
        st.rerun()
with c3:
    st.download_button("Tải cấu hình hiện tại (.yaml)", cfgmod.dumps(cur), "business.yaml", "text/yaml")
with c4:
    if not common.is_public() and st.button("Ghi ra config/business.yaml"):
        cfgmod.save(cur, cfgmod.REAL_PATH)
        st.success("Đã ghi config/business.yaml (file này nằm trong .gitignore).")

st.divider()
draft = {"version": cur.get("version", 1)}
groups = [k for k, v in cur.items() if isinstance(v, dict)]
scalars = [k for k, v in cur.items() if not isinstance(v, dict) and k not in SKIP]
tabs = st.tabs(["0. Chung"] + [TITLES.get(k, k) for k in groups])
with tabs[0]:
    for k in scalars:
        draft[k] = render(cur[k], k)
for tab, key in zip(tabs[1:], groups):
    with tab:
        draft[key] = render(cur[key], key)
draft = {k: draft[k] for k in cur if k in draft}

# ----------------------------------------------------------- xem trước tác động
st.divider()
st.subheader("Xem trước tác động của ngưỡng đang chỉnh")
pv = []
if common.have("raw", "feats_all"):
    miss = ss.raw[ss.feats_all].isna().mean()
    n_old = int((miss > cur["cleaning"]["max_missing_pct"]).sum())
    n_new = int((miss > draft["cleaning"]["max_missing_pct"]).sum())
    pv.append(("Biến bị gợi ý loại vì missing", f"{n_old} → {n_new}", f"ngưỡng {cur['cleaning']['max_missing_pct']:.0%} → {draft['cleaning']['max_missing_pct']:.0%}"))
if common.have("tabs"):
    ivs = {v: binning.total_iv(t) for v, t in ss.tabs.items()}
    def n_band(c, band):
        from core import selection
        return sum(selection.iv_band(v, c) == band for v in ivs.values())
    pv.append(("Biến IV < ngưỡng 'useless'", f"{n_band(cur, 'useless')} → {n_band(draft, 'useless')}", ""))
    pv.append(("Biến IV nghi ngờ (quá cao)", f"{n_band(cur, 'suspicious')} → {n_band(draft, 'suspicious')}", ""))
    small = lambda c: sum(int(((t.kind == 'normal') & (t.count_pct < c['binning']['min_bin_pct'])).sum()) for t in ss.tabs.values())
    pv.append(("Bin nhỏ hơn min_bin_pct", f"{small(cur)} → {small(draft)}", ""))
if common.have("eval"):
    ev = ss.eval["metrics"]
    for name, m in ev.items():
        a = evaluation.rate(m["gini"], cur["evaluation"]["gini"], "gini")
        b = evaluation.rate(m["gini"], draft["evaluation"]["gini"], "gini")
        pv.append((f"Xếp hạng Gini ({name}={m['gini']*100:.1f}%)", f"{a} → {b}", ""))
if common.have("cutoff") and "table_cut" in ss.cutoff:
    from copy import deepcopy
    d2 = deepcopy(draft)
    r = scorecard.suggest_cutoff(ss.cutoff["table_cut"], d2, ss.cutoff.get("y"), ss.cutoff.get("s"))
    pv.append(("Cutoff gợi ý theo chiến lược", r["message"], ""))
if pv:
    for label, val, note in pv:
        st.metric(label, val, note, delta_color="off")
else:
    st.info("Nạp dữ liệu và chạy các bước ở trang phân tích để thấy tác động trực tiếp của ngưỡng.")

derrs = cfgmod.validate(draft)
for e in derrs:
    st.error(e)
note = st.text_input("Ghi chú phiên bản (tuỳ chọn)")
if st.button("✅ Áp dụng cấu hình", type="primary", disabled=bool(derrs)):
    changes = cfgmod.diff(cur, draft)
    ss.cfg = draft
    h = common.store().save_config_version(draft, note or f"{len(changes)} thay đổi")
    common.store().log(f"config {h}: " + "; ".join(f"{p}: {a}→{b}" for p, a, b in changes[:20]))
    st.success(f"Đã áp dụng ({len(changes)} thay đổi, phiên bản {h}). Các bước đã chạy với cấu hình cũ sẽ được đánh dấu cần chạy lại.")
    st.rerun()

with st.expander("Thay đổi chưa áp dụng"):
    ch = cfgmod.diff(cur, draft)
    if ch:
        st.dataframe([{"tham số": p, "hiện tại": str(a), "mới": str(b)} for p, a, b in ch], use_container_width=True)
    else:
        st.write("Không có.")
with st.expander("Lịch sử phiên bản cấu hình"):
    hist = common.store().config_history()
    if not hist:
        st.write("Chưa có phiên bản nào được lưu.")
    for p in reversed(hist):
        with st.expander(p.name):
            old = cfgmod.loads(p.read_text(encoding="utf-8"))
            ch = cfgmod.diff(old, cur)
            st.write(f"Khác với cấu hình hiện tại: {len(ch)} tham số")
            if ch:
                st.dataframe([{"tham số": a, "phiên bản này": str(b), "hiện tại": str(c)} for a, b, c in ch], use_container_width=True)
            if st.button("Khôi phục phiên bản này", key=f"restore_{p.name}"):
                ss.cfg = old
                ss.cfg_ver += 1
                st.rerun()
