import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import common, dash
from core import dashboard as D
from core import selection


def render():
    ss, cfg = st.session_state, common.cfg()
    st.subheader("Bước 6 · Chọn biến (phễu theo ngưỡng + tuân thủ)")
    if not common.gate("bin_confirmed", msg="Chốt binning ở bước 5 trước."):
        return
    s = cfg["selection"]
    st.caption(f"Phễu: IV < {s['iv']['useless_below']} loại · IV > {s['iv']['suspicious_above']} nghi ngờ · |r| thô > {s['correlation']['raw_abs_threshold']} · "
               f"|r| sau binning > {s['correlation']['binned_abs_threshold']} · VIF > {s['vif']['threshold']} · đảo chiều bad rate train/test · "
               f"tối đa {s['final_variable_count'][1]} biến. Thuộc tính được bảo vệ luôn bị loại.")
    if st.button("Chạy phễu chọn biến", type="primary"):
        common.invalidate_from("selection")
        tr = ss.parts["train"]
        ss.funnel = selection.funnel(ss.D.loc[tr], ss.W_tr, ss.tabs, ss.tabs_test, cfg)
    if "funnel" not in ss:
        return
    f = ss.funnel
    stages = ["compliance", "must_drop", "iv", "stability", "corr_raw", "corr_binned", "vif", "count"]
    names = {"compliance": "Tuân thủ", "must_drop": "Bắt buộc loại", "iv": "IV", "stability": "Ổn định train/test", "corr_raw": "Tương quan thô",
             "corr_binned": "Tương quan sau binning", "vif": "VIF", "count": "Giới hạn số biến"}
    cnt, left = [], len(f)
    xs, ys = ["Ban đầu"], [left]
    for st_ in stages:
        n = int((f["stage"] == st_).sum())
        left -= n
        xs.append(names[st_]); ys.append(left)
    st.plotly_chart(go.Figure(go.Funnel(y=xs, x=ys, textinfo="value")).update_layout(height=320, title="Số biến còn lại sau mỗi bộ lọc"), use_container_width=True)

    with st.expander("📊 Dashboard chọn biến: IV theo quyết định, tương quan, VIF", expanded=True):
        top = f.sort_values("iv", ascending=False).head(30).iloc[::-1]
        col = {"keep": "#2a9d8f", "review": "#f2cc8f", "drop": "#bbbbbb"}
        st.plotly_chart(dash.bar_chart(top["variable"], top["iv"], "IV của 30 biến đầu (xanh = giữ · vàng = xem lại · xám = loại)", horizontal=True,
                                       height=max(320, 18 * len(top) + 100), color=[col[x] for x in top["suggestion"]]), use_container_width=True)
        kept = f[f["suggestion"] == "keep"]["variable"].tolist()
        a, b = st.columns(2)
        with a:
            if len(kept) >= 2:
                st.plotly_chart(dash.heatmap(D.corr_matrix(ss.W_tr, kept[:40], "pearson"), "Tương quan WoE giữa các biến được giữ"), use_container_width=True)
            else:
                st.info("Cần ≥ 2 biến được giữ để vẽ ma trận tương quan.")
        with b:
            v = f.dropna(subset=["vif"]).sort_values("vif", ascending=False).head(20).iloc[::-1]
            if len(v):
                fig = dash.bar_chart(v["variable"], v["vif"], "VIF (biến còn lại sau phễu)", horizontal=True, height=max(320, 18 * len(v) + 100))
                fig.add_vline(x=cfg["selection"]["vif"]["threshold"], line_dash="dash", annotation_text="ngưỡng")
                st.plotly_chart(fig, use_container_width=True)
    view = f.copy()
    view["dùng"] = view["suggestion"] == "keep"
    view = view[["variable", "dùng", "iv", "iv_band", "trend", "iv_test", "vif", "suggestion", "stage", "reason", "flags"]]
    edited = st.data_editor(view, use_container_width=True, hide_index=True, key="sel_editor",
                            disabled=[c for c in view.columns if c != "dùng"],
                            column_config={"dùng": st.column_config.CheckboxColumn("Dùng?"), "iv": st.column_config.NumberColumn(format="%.3f"),
                                           "iv_test": st.column_config.NumberColumn(format="%.3f"), "vif": st.column_config.NumberColumn(format="%.1f")})
    chosen = edited.loc[edited["dùng"], "variable"].tolist()
    comp = selection.compliance_flags(chosen, cfg)
    prot = comp[comp["protected"]]["variable"].tolist()
    prox = comp[comp["proxy_warning"]]["variable"].tolist()
    if prot:
        st.error("Thuộc tính được bảo vệ không được dùng: " + ", ".join(prot))
    ack = True
    if prox:
        st.warning("Biến proxy cần rà soát tuân thủ (đồng thuận, phân biệt đối xử gián tiếp, giải thích được): " + ", ".join(prox))
        ack = st.checkbox("Tôi đã rà soát tuân thủ cho các biến proxy trên và chấp nhận dùng")
    lo, hi = s["final_variable_count"]
    if chosen and not (lo <= len(chosen) <= hi):
        st.warning(f"Đang chọn {len(chosen)} biến, ngoài khoảng khuyến nghị {lo}–{hi}.")
    with st.expander("Checklist 'Safe to model?' (6 câu hỏi)"):
        st.markdown("1. Đồng thuận & quyền riêng tư · 2. Thuộc tính được bảo vệ · 3. Phân biệt đối xử gián tiếp (proxy) · 4. Nhân quả & danh tiếng · "
                    "5. Độ tin cậy nguồn / khó bị gaming · 6. Giải thích được. Biến phải trả lời 'Có' đủ cả 6 mới đưa vào mô hình.")
    if st.button("✅ Chốt danh sách biến", disabled=bool(prot) or not chosen or not ack):
        common.invalidate_from("model")
        ss.final_vars = chosen
        ss.sel_confirmed = True
        common.store().save("selection", {"funnel": f, "final_vars": chosen}, cfg)
        st.success(f"Đã chốt {len(chosen)} biến ứng viên cho bước huấn luyện.")
