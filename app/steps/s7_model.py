import pandas as pd
import streamlit as st

from app import common
from core import config as cfgmod, imbalance, modeling

LABEL = {"none": "Không can thiệp", "oversample_random": "Oversample ngẫu nhiên (nhân bản bad)", "oversample_smote": "Oversample SMOTE",
         "undersample_random": "Undersample ngẫu nhiên (bớt good)", "sample_weight": "Trọng số mẫu"}


def training_data():
    """(X, y, w, prior_offset) theo lựa chọn mất cân bằng của người dùng, chỉ trên Train."""
    ss, cfg = st.session_state, common.cfg()
    im = cfg["split"]["imbalance"]
    tr = ss.parts["train"]
    X, y = ss.W_tr[ss.final_vars], ss.y.loc[tr]
    Xr, yr, w = imbalance.resample(X, y, im["method"], im.get("target_bad_rate"), cfg.get("seed", 42))
    off = modeling.prior_offset_for(y, yr, w) if im.get("prior_correction") and im["method"] != "none" else 0.0
    return Xr, yr, w, off


def render():
    ss, cfg = st.session_state, common.cfg()
    st.subheader("Bước 7 · Huấn luyện: mất cân bằng → stepwise → logistic scorecard → mô hình đối chứng")
    if not common.gate("sel_confirmed", msg="Chốt danh sách biến ở bước 6 trước."):
        return
    tr, te = ss.parts["train"], ss.parts["test"]
    y_tr = ss.y.loc[tr]
    im = cfg["split"]["imbalance"]

    # ---------------------------------------------- quyết định mất cân bằng (bắt buộc)
    st.markdown("#### 7a · Xử lý mất cân bằng — *quyết định của bạn, không có mặc định*")
    st.write(f"Train: {len(y_tr):,} dòng · bad rate **{y_tr.mean():.1%}** ({int(y_tr.sum()):,} bad / {int((1 - y_tr).sum()):,} good). Chỉ áp dụng cho Train; Test giữ phân bố thật.")
    methods = im["allowed_methods"]
    cur = im["method"] if im["method"] in methods else None
    choice = st.radio("Phương án", methods, index=methods.index(cur) if cur else None, format_func=lambda m: LABEL[m], key="imb_choice")
    rate = st.slider("Bad rate mục tiêu sau xử lý (không áp dụng cho 'không can thiệp')", 0.10, 0.60, float(im["target_bad_rate"] or 0.30), 0.05, key="imb_rate",
                     disabled=choice in (None, "none"))
    prior_choice = None
    if choice not in (None, "none"):
        cur_p = im["prior_correction"]
        prior_choice = st.radio("Hiệu chỉnh prior để PD về bad rate thật?", ["Có", "Không"], index=None if cur_p is None else (0 if cur_p else 1), horizontal=True, key="imb_prior")
    ready = choice is not None and (choice == "none" or prior_choice is not None)
    if st.button("So sánh các phương án (Gini/KS/PR-AUC trên Test)"):
        with st.spinner("Đang so sánh…"):
            ss.imb_cmp = modeling.compare_imbalance(ss.W_tr, y_tr, ss.W_te, ss.y.loc[te], ss.final_vars, cfg, None, rate)
    if "imb_cmp" in ss:
        st.dataframe(ss.imb_cmp.style.format({"train_bad_rate": "{:.1%}", "gini_test": "{:.3f}", "ks_test": "{:.3f}", "pr_auc_test": "{:.3f}", "recall": "{:.2f}", "precision": "{:.2f}"}), use_container_width=True)
    if not ready:
        st.warning("Chọn phương án xử lý mất cân bằng (và hiệu chỉnh prior nếu có can thiệp) để tiếp tục. Hệ thống không tự chọn thay bạn.")
        return
    if st.button("Lưu lựa chọn mất cân bằng"):
        cfgmod.set_(cfg, "split.imbalance.method", choice)
        cfgmod.set_(cfg, "split.imbalance.target_bad_rate", None if choice == "none" else rate)
        cfgmod.set_(cfg, "split.imbalance.prior_correction", None if choice == "none" else prior_choice == "Có")
        common.store().save_config_version(cfg, f"mất cân bằng: {choice}")
        common.invalidate_from("model")
        st.rerun()
    if im["method"] != choice:
        st.info("Bấm **Lưu lựa chọn** để áp dụng phương án vừa chọn.")
        return

    # -------------------------------------------------------------------- stepwise
    st.markdown("#### 7b · Stepwise")
    sw = cfg["modeling"]["stepwise"]
    st.caption(f"Hướng `{sw['direction']}` · tiêu chí `{sw['criterion']}` · vào p<{sw['p_enter']} / ra p>{sw['p_remove']} · tối đa {sw['max_variables']} biến · "
               f"dấu hệ số WoE phải {sw['expected_sign_on_woe']} {'(bắt buộc)' if sw['require_expected_sign'] else ''}. "
               "Lưu ý: oversample nhân bản dòng làm p-value lạc quan.")
    if st.button("Chạy stepwise", type="primary"):
        common.invalidate_from("model")
        X, y, w, off = training_data()
        with st.spinner("Stepwise…"):
            res = modeling.stepwise(X, y, cfg, ss.final_vars, cfg["selection"].get("must_keep"), w)
        ss.steps, ss.step_notes = res["steps"], res["notes"]
        ss.sw_selected = res["selected"]
    if "steps" in ss:
        st.dataframe(ss.steps, use_container_width=True)
        for n in ss.step_notes:
            st.warning(n)
    default = ss.get("sw_selected", ss.final_vars)
    final = st.multiselect("Biến đưa vào scorecard (sửa tay được)", ss.final_vars, default=[v for v in default if v in ss.final_vars])
    if st.button("Huấn luyện logistic scorecard", disabled=not final):
        X, y, w, off = training_data()
        model = modeling.fit_logistic(X, y, final, cfg, w, off)
        ss.model = model
        ss.resampled = {"rows": len(y), "bad_rate": float(y.mean()), "offset": off}
        ss.pop("challengers", None)
        common.invalidate_from("eval")
    if "model" not in ss:
        return
    m = ss.model
    st.markdown("**Hệ số mô hình** (logit(P(bad)) = b0 + Σ b·WoE)")
    st.dataframe(m["table"].style.format({"coef": "{:.4f}", "std_err": "{:.4f}", "z": "{:.2f}", "p_value": "{:.2e}"}), use_container_width=True)
    st.caption(f"Intercept {m['intercept']:.4f} · prior offset {m['prior_offset']:.4f} · huấn luyện trên {ss.resampled['rows']:,} dòng, bad rate {ss.resampled['bad_rate']:.1%}.")
    if not m["table"]["sign_ok"].all():
        st.error("Có hệ số sai dấu: " + ", ".join(m["table"].index[~m["table"]["sign_ok"]]) + " → loại hoặc xem lại binning (đa cộng tuyến/ngược xu hướng).")
    if (m["table"]["p_value"] > sw["p_remove"]).any():
        st.warning("Có biến p-value > ngưỡng: " + ", ".join(m["table"].index[m["table"]["p_value"] > sw["p_remove"]]))

    # -------------------------------------------------------- mô hình đối chứng
    st.markdown("#### 7c · Mô hình đối chứng (Decision Tree · Random Forest · LightGBM)")
    on = [k for k, v in cfg["modeling"]["challengers"].items() if isinstance(v, dict) and v.get("enabled")]
    st.caption("Bật/tắt ở trang *Quy tắc & ngưỡng*. Đang bật: " + (", ".join(on) or "không"))
    if on and st.button("Huấn luyện mô hình đối chứng"):
        X, y, w, _ = training_data()
        with st.spinner("Đang huấn luyện…"):
            ss.challengers = modeling.fit_challengers(X, y, m["variables"], cfg, w)
    if st.button("✅ Chốt mô hình"):
        common.invalidate_from("eval")
        ss.model_confirmed = True
        common.store().save("modeling", {"model": ss.model, "steps": ss.get("steps"), "imbalance": cfg["split"]["imbalance"]}, cfg)
        st.success("Đã chốt mô hình. Sang bước 8 (Đánh giá).")
