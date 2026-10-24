"""Thành phần dashboard dùng chung cho các bước (vẽ bằng plotly; số liệu tính ở core/dashboard.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import dashboard as D

COLORS = {"bar": "#9db8d8", "line": "#d1495b", "good": "#2a9d8f", "bad": "#d1495b", "grey": "#bbbbbb"}


def metrics_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)


def badrate_chart(tab: pd.DataFrame, title: str, show_ci: bool = True) -> go.Figure:
    """Cột = tỷ trọng mỗi nhóm; đường = bad rate (+ khoảng tin cậy); nét đứt = bad rate trung bình. Nhóm nhỏ tô xám."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = [COLORS["grey"] if s else COLORS["bar"] for s in tab["small"]]
    fig.add_bar(x=tab["group"], y=tab["share"], name="% khoản vay", marker_color=colors, secondary_y=False,
                hovertemplate="%{x}<br>%{y:.1%} khoản vay<extra></extra>")
    err = dict(type="data", symmetric=False, array=(tab["ci_high"] - tab["bad_rate"]).to_numpy(), arrayminus=(tab["bad_rate"] - tab["ci_low"]).to_numpy()) if show_ci else None
    fig.add_scatter(x=tab["group"], y=tab["bad_rate"], mode="lines+markers", name="bad rate", line=dict(color=COLORS["line"]), error_y=err, secondary_y=True,
                    hovertemplate="%{x}<br>bad rate %{y:.2%}<extra></extra>")
    base = tab.attrs.get("base_rate")
    if base is not None and not np.isnan(base):
        fig.add_hline(y=base, line_dash="dash", line_color="#555", annotation_text=f"trung bình {base:.1%}", secondary_y=True)
    fig.update_layout(title=title, height=380, legend=dict(orientation="h", y=1.12), margin=dict(t=70))
    fig.update_yaxes(tickformat=".0%", secondary_y=False, title="% khoản vay")
    fig.update_yaxes(tickformat=".0%", secondary_y=True, title="bad rate", rangemode="tozero")
    return fig


def time_chart(vol: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=vol["period"], y=vol["n"], name="số khoản vay", marker_color=COLORS["bar"], secondary_y=False)
    if "bad_rate" in vol.columns:
        fig.add_scatter(x=vol["period"], y=vol["bad_rate"], mode="lines+markers", name="bad rate", line=dict(color=COLORS["line"]), secondary_y=True)
        fig.update_yaxes(tickformat=".0%", secondary_y=True, title="bad rate", rangemode="tozero")
    fig.update_layout(title=title, height=340, legend=dict(orientation="h", y=1.15), margin=dict(t=70))
    fig.update_yaxes(title="số khoản vay", secondary_y=False)
    return fig


def lines_by_group(t: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for g, sub in t.groupby("group"):
        fig.add_scatter(x=sub["period"], y=sub["bad_rate"], mode="lines+markers", name=str(g), customdata=sub["n"],
                        hovertemplate="%{x|%Y}<br>bad rate %{y:.2%} (n=%{customdata})<extra>" + str(g) + "</extra>")
    fig.update_layout(title=title, height=340, yaxis_tickformat=".0%", yaxis_rangemode="tozero", legend=dict(orientation="h", y=-0.2))
    return fig


def missing_bar(mt: pd.DataFrame, threshold: float | None = None, top: int = 30) -> go.Figure:
    d = mt.head(top).iloc[::-1]
    colors = [COLORS["bad"] if (threshold is not None and v > threshold) else COLORS["bar"] for v in d["missing_pct"]]
    fig = go.Figure(go.Bar(x=d["missing_pct"], y=d["column"], orientation="h", marker_color=colors,
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"ngưỡng {threshold:.0%}")
    fig.update_layout(title=f"Tỷ lệ thiếu (top {min(top, len(mt))} biến)", height=max(300, 18 * len(d) + 100), xaxis_tickformat=".0%", margin=dict(l=160))
    return fig


def hist_by_class(h: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for cls in ("good", "bad"):
        d = h[h["class"] == cls]
        fig.add_bar(x=(d["left"] + d["right"]) / 2, y=d["density"], name=cls, marker_color=COLORS[cls], opacity=0.6)
    fig.update_layout(barmode="overlay", title=title, height=320, yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.12), margin=dict(t=70))
    return fig


def hist_simple(h: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=(h["left"] + h["right"]) / 2, y=h["count"], marker_color=COLORS["bar"]))
    fig.update_layout(title=title, height=300)
    return fig


def bar_chart(x, y, title: str, horizontal: bool = False, fmt: str | None = None, color=None, height: int = 340) -> go.Figure:
    fig = go.Figure(go.Bar(x=y if horizontal else x, y=x if horizontal else y, orientation="h" if horizontal else "v",
                           marker_color=color or COLORS["bar"]))
    fig.update_layout(title=title, height=height)
    if fmt:
        (fig.update_xaxes if horizontal else fig.update_yaxes)(tickformat=fmt)
    return fig


def heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1, colorscale="RdBu_r",
                               hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title=title, height=max(380, 18 * len(corr) + 160))
    return fig


# --------------------------------------------------- dashboard chọn biến → bad rate
def feature_dashboard(df: pd.DataFrame, y: pd.Series, features: list[str], cfg: dict, key: str, date_col: str | None = None,
                      notes: dict | None = None, default: str | None = None, thresholds: dict | None = None,
                      label: str = "Chọn biến để xem bad rate theo nhóm", fixed_feature: str | None = None) -> str | None:
    """Chọn một biến → hiện bảng + biểu đồ bad rate theo nhóm, ổn định theo thời gian và phân phối good/bad.

    thresholds: {'min_size_pct': ..., 'min_bads': ...} để đánh dấu nhóm không đạt (dùng cho phân tầng).
    Trả về tên biến đã chọn.
    """
    dcfg = cfg["dashboard"]
    if not features:
        st.info("Chưa có biến để hiển thị.")
        return None
    notes = notes or {}
    idx = features.index(default) if default in features else 0
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        if fixed_feature:
            feat = fixed_feature
            st.markdown(f"**Biến đang xem: `{feat}`**")
        else:
            feat = st.selectbox(label, features, index=idx, key=f"{key}_feat", format_func=lambda c: f"{c} — {notes[c][:60]}" if notes.get(c) else c)
    with c2:
        nb = st.slider("Số nhóm (biến số)", 3, 20, dcfg["n_bins"], key=f"{key}_bins")
    with c3:
        freq = st.selectbox("Kỳ thời gian", ["Y", "Q", "M"], index=["Y", "Q", "M"].index(dcfg.get("time_freq", "Y")) if dcfg.get("time_freq") in ("Y", "Q", "M") else 0,
                            format_func={"Y": "Năm", "Q": "Quý", "M": "Tháng"}.get, key=f"{key}_freq")
    if notes.get(feat):
        st.caption(f"**{feat}**: {notes[feat]}")
    yy = y.reindex(df.index)
    if yy.notna().sum() == 0:
        st.info("Chưa có nhãn để tính bad rate (cần cột target/trạng thái hoặc đã dựng target).")
        return feat
    tab = D.badrate_by_feature(df, yy, feat, nb, dcfg["max_categories"], dcfg["min_group_count"])
    if thresholds:
        tab["đạt_cỡ"] = (tab["share"] >= thresholds.get("min_size_pct", 0)) & (tab["bads"] >= thresholds.get("min_bads", 0))
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(badrate_chart(tab, f"Bad rate theo nhóm của {feat}"), use_container_width=True)
    with right:
        show = tab.rename(columns={"group": "nhóm", "n": "số khoản", "share": "tỷ trọng", "bads": "bad", "bad_rate": "bad rate", "ci_low": "CI thấp", "ci_high": "CI cao", "small": "quá nhỏ"})
        st.dataframe(show.style.format({"tỷ trọng": "{:.1%}", "bad rate": "{:.2%}", "lift": "{:.2f}", "CI thấp": "{:.1%}", "CI cao": "{:.1%}", "bad": "{:.0f}"}),
                     use_container_width=True, height=330, hide_index=True)
        if thresholds:
            ok = int(tab["đạt_cỡ"].sum())
            st.caption(f"{ok}/{len(tab)} nhóm đạt cỡ tối thiểu (≥ {thresholds.get('min_size_pct', 0):.0%} mẫu và ≥ {thresholds.get('min_bads', 0)} bad).")
        ratio = tab.loc[~tab["small"], "bad_rate"]
        if len(ratio) >= 2 and ratio.min() > 0:
            st.metric("Chênh lệch bad rate giữa nhóm cao nhất/thấp nhất", f"{ratio.max() / ratio.min():.2f}×")
    if date_col and date_col in df.columns:
        t = D.badrate_by_feature_time(df, yy, feat, date_col, freq, min(nb, 6), min(dcfg["max_categories"], 6))
        if len(t):
            st.plotly_chart(lines_by_group(t, f"Bad rate theo thời gian, từng nhóm của {feat}"), use_container_width=True)
    if pd.api.types.is_numeric_dtype(df[feat]) and df[feat].nunique() > nb:
        h = D.numeric_by_class(df, yy, feat)
        if len(h):
            st.plotly_chart(hist_by_class(h, f"Phân phối {feat}: good vs bad"), use_container_width=True)
    return feat
