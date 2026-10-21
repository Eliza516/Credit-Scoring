import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from app import common
from core import binning, evaluation, modeling


def _sets():
    ss = st.session_state
    out = {"train": (ss.W_tr, ss.y.loc[ss.parts["train"]]), "test": (ss.W_te, ss.y.loc[ss.parts["test"]])}
    if "W_oot" in ss:
        out["oot"] = (ss.W_oot, ss.y.loc[ss.parts["oot"]])
    return out


def render():
    ss, cfg = st.session_state, common.cfg()
    st.subheader("Bước 8 · Đánh giá (chỉ số tín dụng + ML, train vs test)")
    if not common.gate("model_confirmed", msg="Chốt mô hình ở bước 7 trước."):
        return
    ev, ml = cfg["evaluation"], cfg["evaluation"]["ml_metrics"]
    st.caption(f"Ngưỡng: Gini xuất sắc > {ev['gini']['excellent_above']}% · tốt ≥ {ev['gini']['good_min']}% · chấp nhận ≥ {ev['gini']['acceptable_min']}% · "
               f"KS mạnh > {ev['ks']['strong_above']} · AUC tốt ≥ {ev['auc']['good_min']} · PR-AUC ≥ {ml['thresholds']['min_pr_auc_lift_over_base_rate']}× bad rate · "
               f"lift top decile ≥ {ml['lift']['top_decile_min']} · chênh AUC train–test ≤ {ml['overfit']['max_auc_gap_train_test']}. "
               "Ngưỡng phân loại tạm dùng max-F1 (cutoff điểm chốt ở bước 9).")
    if st.button("Chạy đánh giá", type="primary"):
        model = ss.model
        sets = _sets()
        risk = {k: modeling.predict_logit(model, W, with_prior=False) for k, (W, y) in sets.items()}
        metrics = {k: evaluation.metrics_row(sets[k][1].to_numpy(), risk[k], cfg) for k in sets}
        assess = evaluation.assessment(metrics, cfg)
        vars_ = model["variables"]
        sel = ss.model["variables"]
        rows = {"logistic": metrics["test"]}
        ytest = sets["test"][1].to_numpy()
        ch_risk = {}
        for k, mdl in ss.get("challengers", {}).items():
            p = mdl.predict_proba(sets["test"][0][vars_])[:, 1]
            ch_risk[k] = p
            rows[k] = evaluation.metrics_row(ytest, p, cfg)
        cmp_ = evaluation.compare_models(rows, cfg)
        warns = evaluation.ml_vs_primary_warnings(rows, "logistic", cfg) if len(rows) > 1 else []
        tr_ix, te_ix = ss.parts["train"], ss.parts["test"]
        psi_val = evaluation.psi(risk["train"], risk["test"])
        csi = evaluation.csi_table(
            pd.DataFrame({v: binning.assign(binning.prepare(ss.D.loc[tr_ix, v], ss.specs[v]), ss.specs[v]) for v in vars_}),
            pd.DataFrame({v: binning.assign(binning.prepare(ss.D.loc[te_ix, v], ss.specs[v]), ss.specs[v]) for v in vars_}))
        shap = modeling.shap_importance(ss.challengers["lightgbm"], sets["test"][0][vars_]) if "lightgbm" in ss.get("challengers", {}) else None
        ss.eval = {"metrics": metrics, "assessment": assess, "risk": risk, "compare": cmp_, "ml_warnings": warns, "psi": psi_val, "csi": csi,
                   "shap": shap, "ch_risk": ch_risk, "ytest": ytest, "ytrain": sets["train"][1].to_numpy()}
        common.invalidate_from("score")
        common.store().save("evaluation", {"assessment": assess, "compare": cmp_, "psi": psi_val}, cfg)
    if "eval" not in ss:
        return
    E = ss.eval
    a = E["assessment"]
    st.dataframe(a.style.format({"auc": "{:.3f}", "gini": "{:.3f}", "ks": "{:.3f}", "pr_auc": "{:.3f}", "lift_top_decile": "{:.2f}"}), use_container_width=True)
    for w in a.attrs.get("warnings", []):
        st.warning(w)
    if not a.attrs.get("warnings"):
        st.success("Không có cảnh báo vượt ngưỡng (overfit, PR-AUC, lift).")

    c1, c2 = st.columns(2)
    fig = go.Figure()
    for k in E["risk"]:
        y = E["ytrain"] if k == "train" else (E["ytest"] if k == "test" else ss.y.loc[ss.parts["oot"]].to_numpy())
        fpr, tpr, _ = roc_curve(y, E["risk"][k])
        fig.add_scatter(x=fpr, y=tpr, name=f"{k} (AUC {E['metrics'][k]['auc']:.3f})")
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="grey"), name="ngẫu nhiên")
    fig.update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR", height=360)
    c1.plotly_chart(fig, use_container_width=True)
    d = pd.DataFrame({"y": E["ytest"], "s": E["risk"]["test"]}).sort_values("s")
    cb = (d["y"] == 1).cumsum() / (d["y"] == 1).sum(); cg = (d["y"] == 0).cumsum() / (d["y"] == 0).sum()
    fk = go.Figure()
    fk.add_scatter(y=cb.values, name="% bad tích lũy"); fk.add_scatter(y=cg.values, name="% good tích lũy")
    fk.update_layout(title=f"KS (test) = {E['metrics']['test']['ks']:.3f}", xaxis_title="thứ hạng điểm rủi ro", height=360)
    c2.plotly_chart(fk, use_container_width=True)

    lt = evaluation.lift_table(E["ytest"], E["risk"]["test"])
    st.markdown("**Lift theo decile (test, decile 1 = rủi ro cao nhất)**")
    st.dataframe(lt.style.format({"bad_rate": "{:.1%}", "lift": "{:.2f}", "cum_bad_capture": "{:.1%}"}), use_container_width=True)

    st.markdown("**Chỉ số theo ngưỡng phân loại (test)**")
    m = E["metrics"]["test"]
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Precision", f"{m['precision']:.2f}"); t2.metric("Recall", f"{m['recall']:.2f}"); t3.metric("F1", f"{m['f1']:.2f}")
    t4.metric("Specificity", f"{m['specificity']:.2f}"); t5.metric("MCC", f"{m['mcc']:.2f}")
    st.dataframe(pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], index=["thực tế good", "thực tế bad"], columns=["dự đoán good", "dự đoán bad"]), use_container_width=False)

    st.markdown("**So sánh mô hình (test)**")
    st.dataframe(E["compare"].style.format("{:.3f}", subset=[c for c in E["compare"].columns if c != "rank"]), use_container_width=True)
    for w in E["ml_warnings"]:
        st.warning(w)
    if len(E["compare"]) == 1:
        st.caption("Huấn luyện mô hình đối chứng ở bước 7c để so sánh.")
    if E["shap"] is not None:
        st.bar_chart(E["shap"].rename("mean |SHAP| (LightGBM)"))

    st.markdown("**Ổn định: PSI điểm & CSI biến (train → test)**")
    g, r = ev["psi"]["green_below"], ev["psi"]["red_above"]
    st.write(f"PSI điểm = **{E['psi']:.3f}** → {evaluation.stability_flag(E['psi'], g, r)}")
    csi = E["csi"].copy()
    csi["cờ"] = csi["csi"].map(lambda v: evaluation.stability_flag(v, ev["csi"]["green_below"], ev["csi"]["red_above"]))
    st.dataframe(csi.style.format({"csi": "{:.3f}"}), use_container_width=True)
