import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import common
from core import binning, evaluation, export, ingest, scorecard


def render():
    ss, cfg = st.session_state, common.cfg()
    st.subheader("Bước 9 · Bảng điểm, cutoff & xuất kết quả")
    if not common.gate("eval", msg="Chạy đánh giá ở bước 8 trước."):
        return
    sc = cfg["scorecard"]
    f, off = scorecard.scaling(cfg)
    st.caption(f"Base score {sc['base_score']} tại odds good:bad = {sc['base_odds_good_to_bad']}:1 · PDO {sc['pdo']} · làm tròn {sc['rounding']} · khoảng điểm {sc['score_range']} "
               f"(factor {f:.2f}, offset {off:.2f}). Điểm cao = rủi ro thấp.")
    model, tr, te = ss.model, ss.parts["train"], ss.parts["test"]
    if st.button("Tạo bảng điểm", type="primary"):
        pts = scorecard.build_table(model, ss.specs, ss.tabs, cfg)
        ss.points = pts
        s_tr = scorecard.score(ss.D.loc[tr], model, ss.specs, pts, cfg)
        s_te = scorecard.score(ss.D.loc[te], model, ss.specs, pts, cfg)
        y_te = ss.y.loc[te]
        ct = scorecard.cutoff_table(y_te, s_te)
        sug = scorecard.suggest_cutoff(ct, cfg, y_te.reset_index(drop=True), s_te.reset_index(drop=True))
        ss.cutoff = {"table_cut": ct, "y": y_te.reset_index(drop=True), "s": s_te.reset_index(drop=True), "s_tr": s_tr, "s_te": s_te, "suggest": sug, "cutoff": None}
    if "points" not in ss:
        return
    pts, C = ss.points, ss.cutoff
    with st.expander("Bảng điểm (điểm từng bin)", expanded=True):
        st.dataframe(pts.style.format({"count_pct": "{:.1%}", "bad_rate": "{:.2%}", "woe": "{:.3f}", "coef": "{:.3f}", "points": "{:.0f}"}), use_container_width=True)
    y_te = ss.y.loc[te]
    h = go.Figure()
    h.add_histogram(x=C["s_te"][y_te.values == 0], name="good", opacity=0.6, nbinsx=40)
    h.add_histogram(x=C["s_te"][y_te.values == 1], name="bad", opacity=0.6, nbinsx=40)
    h.update_layout(barmode="overlay", title="Phân phối điểm (test)", height=330)
    st.plotly_chart(h, use_container_width=True)
    tt = evaluation.tranche_table(y_te.values, C["s_te"].values, cfg["cutoff"]["risk_tranches"])
    st.markdown("**Tranche rủi ro A→… (A = rủi ro cao nhất) trên Test**")
    st.dataframe(tt.style.format({"bad_rate": "{:.2%}", "cum_bad_pct": "{:.1%}", "cum_good_pct": "{:.1%}", "ks_gap": "{:.3f}", "score_min": "{:.0f}", "score_max": "{:.0f}"}), use_container_width=True)
    if cfg["evaluation"]["rank_ordering"]["require_monotonic_bad_rate"]:
        (st.success if tt.attrs["monotonic"] else st.warning)("Bad rate giảm đơn điệu qua các tranche." if tt.attrs["monotonic"] else "Bad rate KHÔNG đơn điệu qua các tranche – xem lại binning/biến.")

    st.markdown("#### Cutoff & vùng quyết định (quyết định chính sách của bạn)")
    ct = C["table_cut"]
    fig = go.Figure()
    fig.add_scatter(x=ct["cutoff"], y=ct["approval_rate"], name="tỷ lệ duyệt")
    fig.add_scatter(x=ct["cutoff"], y=ct["bad_rate_approved"], name="bad rate nhóm duyệt", yaxis="y2")
    fig.update_layout(title="Đánh đổi theo cutoff (test)", height=330, yaxis=dict(tickformat=".0%"), yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)
    st.info(C["suggest"]["message"] + f"  (chiến lược: `{cfg['cutoff']['strategy']}`)")
    default = C["suggest"]["cutoff"] if C["suggest"]["cutoff"] is not None else float(C["s"].median())
    cut = st.number_input("Cutoff được chọn (điểm ≥ cutoff được duyệt)", value=float(default), step=1.0)
    lo, hi = scorecard.decision_zones(C["s"], cut, cfg)
    st.write(f"Vùng: **Reject** < {lo:.0f} · **Manual review** [{lo:.0f}, {hi:.0f}) · **Approve** ≥ {hi:.0f} (vùng xét tay ≈ {cfg['cutoff']['manual_review_band_width_pct']}% dân số).")
    z = scorecard.apply_zones(C["s"], lo, hi)
    st.dataframe(scorecard.zone_summary(C["y"], z).style.format({"pct": "{:.1%}", "bad_rate": "{:.2%}"}), use_container_width=True)
    if st.button("✅ Chốt cutoff & lưu bảng điểm"):
        C.update(cutoff=cut, reject_below=lo, approve_from=hi)
        name = ss.seg["name"] if ss.get("seg") else "all"
        bundle = export.build_bundle(cfg, model, ss.specs, pts, {"cutoff": cut, "reject_below": lo, "approve_from": hi}, ss.get("clean_log"), ss.get("seg"))
        ss.bundles[name] = bundle
        common.store().save("scorecard", {"points": pts, "cutoff": {"cutoff": cut, "lo": lo, "hi": hi}}, cfg)
        common.store().log(f"chốt scorecard tầng {name}, cutoff {cut}")
        st.success(f"Đã lưu bảng điểm cho tầng `{name}`.")

    # ------------------------------------------------------------------ xuất
    st.markdown("#### Xuất kết quả")
    if ss.bundles:
        st.caption("Các tầng đã lưu: " + ", ".join(ss.bundles))
    E = ss.eval
    sheets = {"BangDiem": pts, "HeSo": model["table"].reset_index(names="variable"), "ChiSo": E["assessment"], "SoSanhMoHinh": E["compare"].reset_index(names="model"),
              "Tranche": tt, "Cutoff": ct, "CSI": E["csi"], "Binning": pd.concat([t.assign(variable=v) for v, t in ss.tabs.items() if v in model["variables"]])}
    if ss.get("funnel") is not None:
        sheets["ChonBien"] = ss.funnel
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("📥 Excel báo cáo", export.to_excel(sheets, cfg), "scorecard_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button("📥 PDF tóm tắt", export.to_pdf("Báo cáo Credit Scoring", [("Chỉ số", E["assessment"]), ("Bảng điểm", pts), ("Tranche", tt)],
                                                        E["assessment"].attrs.get("warnings", [])), "scorecard_report.pdf", "application/pdf")
    if ss.bundles:
        bundle = next(iter(ss.bundles.values())) if len(ss.bundles) == 1 else None
        if bundle:
            c3.download_button("📥 Mô hình (.pkl)", export.bundle_to_bytes(bundle), "scorecard_model.pkl")
    c4.download_button("📥 Cấu hình đã dùng", export.config_json(cfg), "config_used.json")

    st.markdown("#### Chấm điểm file mới")
    if not ss.bundles:
        st.caption("Chốt cutoff & lưu bảng điểm để chấm điểm file mới.")
        return
    name = st.selectbox("Dùng bảng điểm của tầng", list(ss.bundles))
    up = st.file_uploader("File cần chấm điểm (CSV/Excel)", type=["csv", "xlsx"], key="score_up")
    if up is not None:
        try:
            new = ingest.read_file(up, up.name, common.limits(), up.size)
            out = export.score_new_data(ss.bundles[name], new)
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button("📥 Tải kết quả chấm điểm", out.to_csv(index=False).encode("utf-8-sig"), "scored.csv", "text/csv")
        except (ingest.IngestError, ValueError) as e:
            st.error(str(e))
