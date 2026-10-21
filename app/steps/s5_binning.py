import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app import common
from core import binning


def _recompute(var):
    ss = st.session_state
    tr, te = ss.parts["train"], ss.parts["test"]
    spec = ss.specs[var]
    ss.tabs[var] = binning.make_table(binning.prepare(ss.D.loc[tr, var], spec), ss.y.loc[tr], spec)
    ss.tabs_test[var] = binning.make_table(binning.prepare(ss.D.loc[te, var], spec), ss.y.loc[te], spec)


def _plot(t, t_te, title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=t["bin"], y=t["count_pct"], name="% mẫu (train)", opacity=0.55, secondary_y=False)
    fig.add_scatter(x=t["bin"], y=t["bad_rate"], name="bad rate train", mode="lines+markers", secondary_y=True)
    if t_te is not None:
        fig.add_scatter(x=t_te["bin"], y=t_te["bad_rate"], name="bad rate test", mode="lines+markers", line=dict(dash="dot"), secondary_y=True)
    fig.update_layout(title=title, height=360, yaxis_tickformat=".0%")
    fig.update_yaxes(tickformat=".0%", secondary_y=True)
    return fig


def render():
    ss, cfg = st.session_state, common.cfg()
    st.subheader("Bước 5 · Binning WoE (xem & chỉnh tay từng biến)")
    if not common.gate("parts", msg="Chốt chia mẫu ở bước 4 trước."):
        return
    b = cfg["binning"]
    st.caption(f"Ngưỡng: ~{b['initial_fine_bins']} fine bin → ≤ {b['max_coarse_bins']} coarse bin · mỗi bin ≥ {b['min_bin_pct']:.0%} mẫu & ≥ {b['min_bin_bads']} bad · "
               f"đơn điệu `{b['monotonic_trend']}` · missing bin riêng. WoE = ln(%Good/%Bad). Binning học trên **Train**.")
    if st.button("Binning tự động tất cả biến", type="primary"):
        common.invalidate_from("binning")
        tr = ss.parts["train"]
        bar = st.progress(0.0)
        specs = binning.fit_all(ss.D.loc[tr], ss.y.loc[tr], ss.feats, cfg, progress=lambda i, n, c: bar.progress(i / n, text=c))
        ss.specs = specs
        ss.tabs, ss.tabs_test = {}, {}
        for v in specs:
            _recompute(v)
        bar.empty()
    if "specs" not in ss:
        return
    summ = binning.iv_summary(ss.tabs)
    sug_counts = {v: len(binning.suggest(t, cfg, v)) for v, t in ss.tabs.items()}
    summ["số gợi ý"] = summ["variable"].map(sug_counts)
    from core import selection
    summ["iv_band"] = summ["iv"].map(lambda x: selection.iv_band(x, cfg))
    st.dataframe(summ.style.format({"iv": "{:.3f}"}), use_container_width=True, height=240)

    var = st.selectbox("Chọn biến để xem / chỉnh", summ["variable"].tolist(), key="bin_var")
    spec, t = ss.specs[var], ss.tabs[var]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(_plot(t, ss.tabs_test.get(var), f"{var} · IV={binning.total_iv(t):.3f} · xu hướng {binning.trend(t)}"), use_container_width=True)
    with c2:
        st.markdown("**Gợi ý theo ngưỡng**")
        sg = binning.suggest(t, cfg, var)
        if sg:
            for m in sg:
                st.warning(m)
        else:
            st.success("Không có gợi ý – bin thoả ngưỡng.")
    show = t.copy()
    te = ss.tabs_test[var].set_index("bin")
    show["test_count_pct"] = show["bin"].map(te["count_pct"])
    show["test_bad_rate"] = show["bin"].map(te["bad_rate"])
    st.dataframe(show[["bin", "kind", "count", "count_pct", "bad", "good", "bad_rate", "woe", "iv", "test_count_pct", "test_bad_rate"]].style.format(
        {"count_pct": "{:.1%}", "bad_rate": "{:.2%}", "woe": "{:.3f}", "iv": "{:.4f}", "test_count_pct": "{:.1%}", "test_bad_rate": "{:.2%}"}), use_container_width=True)

    st.markdown("**Chỉnh tay**")
    e1, e2, e3 = st.columns(3)
    new = None
    with e1:
        if spec["kind"] == "numeric":
            txt = st.text_input("Điểm cắt (cách nhau dấu phẩy)", value=", ".join(f"{s:.6g}" for s in spec["splits"]), key=f"cuts_{var}")
            if st.button("Áp dụng điểm cắt", key=f"ap_{var}"):
                try:
                    new = binning.set_splits(spec, [float(x) for x in txt.split(",") if x.strip()])
                except ValueError:
                    st.error("Điểm cắt phải là số.")
        else:
            gl = binning.group_labels(spec["groups"])
            a = st.selectbox("Nhóm A", range(len(gl)), format_func=lambda i: gl[i], key=f"ga_{var}")
            bb = st.selectbox("Nhóm B", range(len(gl)), index=min(1, len(gl) - 1), format_func=lambda i: gl[i], key=f"gb_{var}")
            if st.button("Gộp A và B", key=f"gm_{var}"):
                try:
                    new = binning.merge_bins(spec, a, bb)
                except IndexError as e:
                    st.error(str(e))
    with e2:
        if spec["kind"] == "numeric":
            nb = [r for r in t["bin"] if r not in ("Special", "Missing")]
            i = st.selectbox("Gộp bin này với bin liền sau", range(max(len(nb) - 1, 0)), format_func=lambda i: nb[i], key=f"mg_{var}") if len(nb) > 1 else None
            if i is not None and st.button("Gộp", key=f"mgb_{var}"):
                new = binning.merge_bins(spec, i)
        else:
            gi = st.selectbox("Tách khỏi nhóm", range(len(spec["groups"])), format_func=lambda i: binning.group_labels(spec["groups"])[i], key=f"sg_{var}")
            cats = st.multiselect("Category chuyển sang nhóm mới", spec["groups"][gi], key=f"sc_{var}")
            if st.button("Tách", key=f"sgb_{var}"):
                try:
                    new = binning.split_group(spec, gi, cats)
                except ValueError as e:
                    st.error(str(e))
    with e3:
        if spec["kind"] == "numeric":
            v = st.number_input("Thêm điểm cắt tại giá trị", value=float(np.nanmedian(ss.D[var])) if pd.api.types.is_numeric_dtype(ss.D[var]) else 0.0, key=f"ad_{var}")
            if st.button("Tách bin", key=f"adb_{var}"):
                new = binning.split_numeric(spec, v)
        mono = st.selectbox("Ràng buộc đơn điệu", ["auto", "ascending", "descending", "none"], key=f"mono_{var}")
        if st.button("Binning lại tự động biến này", key=f"re_{var}"):
            tr = ss.parts["train"]
            sp = cfg["cleaning"].get("special_values") or []
            new = binning.fit_variable(ss.D.loc[tr, var], ss.y.loc[tr], cfg, sp if spec["kind"] == "numeric" else [], mono)
    if new is not None:
        ss.specs[var] = new
        _recompute(var)
        common.invalidate_from("selection")
        st.rerun()

    st.divider()
    if st.button("✅ Chốt binning"):
        common.invalidate_from("selection")
        specs = {c: binning.attach_woe(s, ss.tabs[c]) for c, s in ss.specs.items()}
        ss.specs = specs
        tr, te = ss.parts["train"], ss.parts["test"]
        ss.W_tr = binning.transform(ss.D.loc[tr], specs)
        ss.W_te = binning.transform(ss.D.loc[te], specs)
        if "oot" in ss.parts:
            ss.W_oot = binning.transform(ss.D.loc[ss.parts["oot"]], specs)
        ss.bin_confirmed = True
        common.store().save("binning", {"specs": specs, "tabs": ss.tabs}, cfg)
        st.success("Đã chốt binning. Sang bước 6.")
