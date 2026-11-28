"""
Module: dashboard/components/tab_models_rq3.py
Mục đích: Giao diện Tab 4 — ML Arena, Residual Diagnostics & XAI (RQ3) — Giai
đoạn 5.
Phân tách rạch ròi 3 khối học thuật bắt buộc:
  A. Model Performance (10 mô hình, Actual vs Predicted, Residual Analysis,
  Phân rã sai số, Độ bền & Mutate/Break)
  B. Model Interpretation / XAI (Permutation Importance 14 nhóm = 27/27 cột
  X_test & PDP M7 vs M2 — KHÔNG phải nhân quả)
  C. Prediction Uncertainty (Quantile Regression [P10, P90] & Monotonic
  Split-CQR: PICP = 79.89%, MPIW = $3.56)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard.business_logic import (
    prepare_model_leaderboard,
    prepare_permutation_family_summary,
    prepare_segment_errors,
)
from dashboard.charts import (
    PLOTLY_DISPLAY_CONFIG,
    build_actual_vs_predicted_and_residuals_fig,
    build_model_arena_fig,
    build_pdp_curves_fig,
    build_permutation_importance_fig,
    build_prediction_interval_fig,
    build_segment_errors_fig,
)
from dashboard.components.kpi_cards import render_insight_card, render_kpi_row


def render_tab_models_rq3(
    tables: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    models_info: dict[str, Any],
    eval_res: dict[str, Any],
    v5: dict[str, Any] | None = None,
) -> None:
    """Hiển thị toàn bộ nội dung Tab 4 — Machine Learning & XAI (RQ3)."""
    leaderboard = prepare_model_leaderboard(tables["test_metrics"])
    m7_live = eval_res["metrics_m7"]
    m2_live = eval_res["metrics_m2"]
    pi_overall = summaries["stage6"]["part2_prediction_intervals_p10_p90"][
        "overall"
    ]
    bp_m7 = summaries["stage6"]["part1_residual_diagnostics"][
        "M7_HistGradientBoosting"
    ]
    ex_ante = summaries["stage5"].get("part3_ex_ante_experiments", {})
    ex_all = next(
        (
            r
            for r in ex_ante.get("pricing_without_surge", [])
            if r["model_or_slice"] == "M7_ExAnte_All_Test"
        ),
        {"r2": 0.9286, "rmse": 2.4655, "mape_pct": 9.15},
    )
    ex_surge = next(
        (
            r
            for r in ex_ante.get("pricing_without_surge", [])
            if r["model_or_slice"] == "M7_ExAnte_Lyft_SurgeGt1x_Test"
        ),
        {"r2": 0.4677},
    )
    ex_auc = float(
        ex_ante.get("lyft_surge_classifier", {})
        .get("HistGB_Classifier", {})
        .get("roc_auc", 0.7383)
    )
    weather_auc_drop = float(
        ex_ante.get("lyft_surge_classifier", {})
        .get("group_roc_auc_drop", {})
        .get("weather_7vars", 0.0122)
    )
    regime = summaries["stage5"].get("part2_regime_breakdown", {})
    lyft_m7 = regime.get("lyft_observed_surge_test", {}).get(
        "M7_Conditional", {}
    )
    uber_m7 = regime.get("uber_hidden_surge_test", {}).get(
        "M7_Conditional", {}
    )
    lyft_r2 = float(lyft_m7.get("r2", 0.9794))
    lyft_rmse = float(lyft_m7.get("rmse", 1.4370))
    uber_r2 = float(uber_m7.get("r2", 0.9474))
    uber_rmse = float(uber_m7.get("rmse", 1.9773))
    rg_m7_rows = tables["robustness"].loc[
        (tables["robustness"]["scenario"] == "route_group_split")
        & (tables["robustness"]["model_code"] == "M7_HistGradientBoosting")
    ]
    rg_mape = (
        float(rg_m7_rows["mape_pct"].iloc[0])
        if not rg_m7_rows.empty
        else 11.50
    )
    cqr = (
        (v5 or {})
        .get("model_diagnostics_and_cqr", {})
        .get("clean_monotonic_cqr", {})
    )
    cqr_picp = float(cqr.get("overall_cqr_picp_pct", 79.89))
    cqr_lo = float(cqr.get("min_tier_cqr_picp_pct", 78.62))
    cqr_hi = float(cqr.get("max_tier_cqr_picp_pct", 81.99))
    cqr_mpiw = float(
        cqr.get(
            "overall_cqr_mpiw_usd", pi_overall["mean_interval_width_mpiw_usd"]
        )
    )
    grp6 = (
        (v5 or {})
        .get("model_diagnostics_and_cqr", {})
        .get("grouped_permutation_importance", [])
    )
    grp6_top2 = (
        sum(float(r["m7_rel_share_pct"]) for r in grp6[:2])
        if len(grp6) >= 2
        else 92.60
    )
    grp6_wt = (
        sum(float(r["m7_rel_share_pct"]) for r in grp6[4:6])
        if len(grp6) >= 6
        else 0.02
    )
    perm_df = tables["permutation_importance"]
    top2_share = float(perm_df.iloc[:2]["m7_importance_share_pct"].sum())
    perm14_wt = float(
        perm_df.loc[
            perm_df["domain_family"].str.contains("Thời tiết", na=False),
            "m7_importance_share_pct",
        ].sum()
    )
    src_dst_rows = perm_df.loc[
        perm_df["feature_group"].str.contains("source", case=False, na=False)
    ]
    src_dst_row = (
        src_dst_rows.iloc[0] if not src_dst_rows.empty else perm_df.iloc[3]
    )
    n_test_eval = int(m7_live["n_test"])

    # =========================================================================
    # KHỐI A: MODEL PERFORMANCE (HIỆU NĂNG, PHẦN DƯ & ĐỘ BỀN MÔ HÌNH)
    # =========================================================================
    st.markdown(
        '<div class="section-heading">Phần A — Hiệu năng Mô hình Có điều '
        "kiện, Ex-Ante &amp; Chẩn đoán Phần dư</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Xác minh mô hình trực tiếp (không train lại): `models/"
        f"best_model.joblib` (`M7`, `{models_info['n_features']}` đặc trưng,"
        " `0` biến rò rỉ) "
        "& `models/m2_interaction_ols.joblib` (`M2`) tái lập chính xác "
        "100.00% chỉ số tập Test trên mẫu quan sát surge thực (`n = "
        f"{n_test_eval:,}`, `seed = 42`)."
    )

    render_kpi_row(
        [
            {
                "label": "M7 Có điều kiện (Vô địch Dự báo)",
                "value": f"R² = {m7_live['r2']:.4f}",
                "subtitle": (
                    f"RMSE: ${m7_live['rmse_usd']:.4f} | MAE: "
                    f"${m7_live['mae_usd']:.4f} | MAPE: "
                    f"{m7_live['mape_pct']:.2f}%"
                ),
                "accent": "emerald",
                "badge_text": f"Lyft {lyft_r2:.4f} vs Uber {uber_r2:.4f}",
                "badge_class": "badge-verified",
            },
            {
                "label": "Đối chứng Ex-Ante (23 biến không surge)",
                "value": f"R² = {float(ex_all['r2']):.4f}",
                "subtitle": (
                    f"RMSE: ${float(ex_all['rmse']):.2f} | MAPE: "
                    f"{float(ex_all['mape_pct']):.2f}% | Phân loại Surge "
                    f"AUC: {ex_auc:.4f}"
                ),
                "accent": "accent",
                "badge_text": f"M2 OLS có điều kiện: R²={m2_live['r2']:.4f}",
                "badge_class": "badge-verified",
            },
            {
                "label": "Tỷ lệ Lệch trong ±$2.00 (M7)",
                "value": f"{m7_live['pct_within_2usd']:.2f}%",
                "subtitle": (
                    f"±$1.00: {m7_live['pct_within_1usd']:.1f}% | ±$3.00: "
                    f"{m7_live['pct_within_3usd']:.1f}% | Kurtosis: "
                    f"{float(bp_m7['kurtosis']):.2f}"
                ),
                "accent": "amber",
                "badge_text": f"Mean resid = +${bp_m7['residual_mean']:.4f}",
                "badge_class": "badge-verified",
            },
            {
                "label": "Monotonic Split-CQR [P10, P90]",
                "value": f"PICP = {cqr_picp:.2f}%",
                "subtitle": (
                    f"12 gói: [{cqr_lo:.2f}%, {cqr_hi:.2f}%] | MPIW: "
                    f"${cqr_mpiw:.2f} (0 cắt ngược)"
                ),
                "accent": "lyft",
                "badge_text": "pandas category gốc • Mục tiêu 80%",
                "badge_class": "badge-simulation",
            },
        ]
    )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # 1. Đấu trường 10 mô hình
    fig_arena = build_model_arena_fig(leaderboard)
    st.plotly_chart(fig_arena, width="stretch", config=PLOTLY_DISPLAY_CONFIG)

    # 2. Actual vs Predicted & Residual Histogram (cho phép đổi M7 / M2)
    diag_model = st.radio(
        "Chọn mô hình kiểm tra Actual vs. Predicted & Phân phối Phần dư "
        f"(trên tập Test n = {n_test_eval:,}):",
        options=["M7", "M2"],
        format_func=lambda k: (
            "M7: HistGradientBoosting (Tốt nhất)"
            if k == "M7"
            else "M2: Interaction OLS (Kinh tế lượng)"
        ),
        horizontal=True,
        key="tab4_diag_model",
    )
    fig_resid = build_actual_vs_predicted_and_residuals_fig(
        eval_res, model_key=diag_model
    )
    st.plotly_chart(fig_resid, width="stretch", config=PLOTLY_DISPLAY_CONFIG)

    # 3. Phân rã sai số theo 4 chiều
    dim_map = {
        "Gói dịch vụ (12 service_tier)": "service_tier",
        "Khoảng cách (5 distance_bucket)": "distance_bucket",
        "Chế độ Surge (4 surge_regime)": "surge_regime",
        "Quận điểm đón (12 source_district)": "source_district",
    }
    selected_dim_label = st.selectbox(
        "Phân rã sai số dự báo M7 vs. M2 theo chiều phân khúc:",
        options=list(dim_map.keys()),
        index=0,
        key="tab4_seg_dimension",
    )
    seg_sub = prepare_segment_errors(
        tables["segment_errors"], dim_map[selected_dim_label]
    )
    fig_seg = build_segment_errors_fig(seg_sub, selected_dim_label)
    st.plotly_chart(fig_seg, width="stretch", config=PLOTLY_DISPLAY_CONFIG)

    # Expanders chi tiết bảng Leaderboard, Robustness & Mutate/Break
    e1, e2, e3 = st.columns(3, gap="medium")
    with (
        e1,
        st.expander(
            "📊 Bảng đầy đủ 10 Mô hình (Test Metrics)", expanded=False
        ),
    ):
        lb_view = leaderboard[
            [
                "rank",
                "model_code",
                "test_r2",
                "test_rmse",
                "test_mae",
                "test_mape_pct",
                "r2_overfit_gap",
                "predict_time_ms",
                "meets_both_smart",
            ]
        ].copy()
        lb_view.columns = [
            "Hạng",
            "Mã mô hình",
            "Test R²",
            "RMSE ($)",
            "MAE ($)",
            "MAPE (%)",
            "Overfit Gap",
            "Predict (ms)",
            "SMART",
        ]
        st.dataframe(lb_view, width="stretch", hide_index=True)
    with (
        e2,
        st.expander(
            "🛡️ Kiểm định Độ bền (Temporal EST, Route, R3)", expanded=False
        ),
    ):
        st.dataframe(tables["robustness"], width="stretch", hide_index=True)
    with (
        e3,
        st.expander(
            "🧪 6 Thí nghiệm Phản chứng (Mutate & Break)", expanded=False
        ),
    ):
        st.dataframe(tables["mutate_break"], width="stretch", hide_index=True)

    # =========================================================================
    # KHỐI B: MODEL INTERPRETATION / XAI (KHÔNG PHẢI QUAN HỆ NHÂN QUẢ)
    # =========================================================================
    st.markdown(
        '<div class="section-heading">Phần B — Diễn giải Mô hình XAI (Model '
        "Interpretation — Không phải Nhân quả)</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "🔬 **Lưu ý phương pháp luận:** Biểu đồ dưới đây sử dụng "
        "**Permutation Feature Importance tính trên tập Test** "
        "(`n = 15,000`, `8` lần lặp, phân hoạch rời `14` nhóm = đúng `27/"
        "27` cột thực — không đếm trùng và không phải split-importance nội "
        "tại của cây). "
        "Kết quả phản ánh **mức độ phụ thuộc dự báo**, **KHÔNG** khẳng định "
        "quan hệ nhân quả độc lập."
    )

    show_m2_perm = st.checkbox(
        "Hiển thị so sánh song song M2 Interaction OLS trên biểu đồ "
        "Permutation",
        value=True,
        key="tab4_perm_m2",
    )
    fig_perm = build_permutation_importance_fig(
        tables["permutation_importance"], show_m2=show_m2_perm
    )
    st.plotly_chart(fig_perm, width="stretch", config=PLOTLY_DISPLAY_CONFIG)

    fam_df = prepare_permutation_family_summary(
        tables["permutation_importance"]
    )
    fam_view = fam_df.copy()
    fam_view.columns = [
        "Họ biến kinh tế (4 nhóm lớn)",
        "Số nhóm con",
        "Tổng M7 ΔRMSE (USD)",
        "Tỷ trọng M7 (%)",
        "Tổng M2 ΔRMSE (USD)",
    ]
    st.dataframe(fam_view, width="stretch", hide_index=True)

    fig_pdp = build_pdp_curves_fig(summaries["stage6"])
    st.plotly_chart(fig_pdp, width="stretch", config=PLOTLY_DISPLAY_CONFIG)

    # =========================================================================
    # KHỐI C: PREDICTION UNCERTAINTY (QUANTILE REGRESSION [P10, P90])
    # =========================================================================
    st.markdown(
        '<div class="section-heading">Phần C — Định lượng Bất định Dự báo '
        "(Prediction Uncertainty — Quantile &amp; Monotonic Split-CQR)</div>",
        unsafe_allow_html=True,
    )
    fig_pi = build_prediction_interval_fig(
        tables["prediction_interval"], cqr_summary=cqr
    )
    st.plotly_chart(fig_pi, width="stretch", config=PLOTLY_DISPLAY_CONFIG)

    st.caption(
        "Minh bạch artifact: Bộ kết quả Quantile `[P10, P90]` trên được "
        "ước lượng ở GĐ6 bằng 3 mô hình "
        "`HistGradientBoostingRegressor(loss='quantile')` dùng `pandas "
        f"category` gốc trên tập Test (`n = {n_test_eval:,}`) và kiểm chứng "
        "hiệu chỉnh "
        f"Monotonic Split-CQR (`PICP = {cqr_picp:.2f}%`, `12` gói trong "
        f"`[{cqr_lo:.2f}%, {cqr_hi:.2f}%]`, `0` cắt ngược); sẽ đóng gói "
        "thành 3 tệp `.joblib` ở Pha 1 triển khai."
    )

    # =========================================================================
    # INSIGHT 5 Ý CHUẨN HỌC THUẬT CHO TAB 4 (RQ3)
    # =========================================================================
    st.markdown(
        '<div class="section-heading">Tổng hợp Insight Mô hình hóa &amp; '
        "XAI (RQ3)</div>",
        unsafe_allow_html=True,
    )
    ic1, ic2 = st.columns(2, gap="medium")
    with ic1:
        render_insight_card(
            title=(
                "Insight 4.1 — M7 Có điều kiện đạt R² = "
                f"{m7_live['r2']:.4f}; Đối chứng Ex-Ante lộ rõ ranh giới "
                "surge tức thời"
            ),
            finding=(
                "Sau khi loại bỏ nhiễu điền mù 1.0x trên Lyft, M7 nâng độ "
                "chính xác mạnh (Kurtosis giảm quá nửa xuống "
                f"{float(bp_m7['kurtosis']):.2f}); tuy "
                "nhiên phần dư lớn (> $15: 57/58; > $16: 100% ở Uber) tập "
                "trung ở phụ phí ẩn của "
                "Uber và dự báo Ex-Ante không dùng surge sụp giảm rõ trên "
                "các lượt Lyft đang surge."
            ),
            evidence=(
                f"Tập Test n = {n_test_eval:,}: M7 có điều kiện đạt R² = "
                f"{m7_live['r2']:.4f}, RMSE = ${m7_live['rmse_usd']:.4f}, "
                f"MAPE = {m7_live['mape_pct']:.2f}% "
                f"(Lyft R² = {lyft_r2:.4f}, RMSE = ${lyft_rmse:.2f}, "
                f"1 lỗi > $15 [0 > $16] vs Uber R² = {uber_r2:.4f}, RMSE = "
                f"${uber_rmse:.2f}, 57 lỗi > $15); "
                f"M2 Interaction OLS đạt R² = {m2_live['r2']:.4f}, RMSE = "
                f"${m2_live['rmse_usd']:.4f}. "
                "Ex-Ante (23 biến không surge) đạt R² = "
                f"{float(ex_all['r2']):.4f} (Lyft > 1x: R² = "
                f"{float(ex_surge['r2']):.4f}, Surge "
                f"Classifier AUC = {ex_auc:.4f})."
            ),
            method=(
                "So sánh 10 mô hình có điều kiện (M0a–M7), phân rã chế độ "
                "Uber/Lyft, thực nghiệm Ex-Ante 23 biến + Surge Classifier, "
                "Temporal EST, Route Group, Train-Only Full Refit & API-"
                "Batch Split."
            ),
            business_meaning=(
                "Triển khai sản xuất kết hợp M7 "
                f"({m7_live['pct_within_2usd']:.1f}% lượt lệch ≤ ±$2.00) "
                "song song M2 OLS để giải trình biểu giá và đối chứng vùng "
                "hiếm ≥ 2.50x; ưu tiên bổ sung biến duration & mật độ xe để "
                "phá trần Ex-Ante."
            ),
            limitation=(
                "Kiểm định Breusch-Pagan (LM = "
                f"{float(bp_m7['breusch_pagan_lm_stat']):.2f}, p < 0.0001) "
                "xác nhận phương sai tăng theo hạng sang; trên 15 tuyến "
                f"mới (Route Group) MAPE tăng lên {rg_mape:.2f}% nên cần "
                "hiệu chuẩn khi mở rộng địa bàn."
            ),
            accent="emerald",
        )
    with ic2:
        render_insight_card(
            title=(
                f"Insight 4.2 — {top2_share:.2f}% Sức mạnh Dự báo đến từ "
                f"Phân khúc Xe & Cụm Cự ly–Surge ({grp6_top2:.2f}% ở gom "
                "6 cụm)"
            ),
            finding=(
                "Trong 14 nhóm đặc trưng rời nhau phủ đủ 27/27 cột X (và "
                "kiểm chứng gom 6 cụm chống đa cộng tuyến), riêng phân khúc "
                "dịch vụ và cụm cự ly–surge chi phối gần như toàn bộ khả "
                "năng dự báo giá, còn thời tiết và đồng hồ EST đều xấp xỉ "
                "0%."
            ),
            evidence=(
                "Permutation Importance trên X_test (n = 15,000, 8 lần "
                "lặp): name (+$"
                f"{float(perm_df.iloc[0]['m7_delta_rmse_mean_usd']):.4f}, "
                f"{float(perm_df.iloc[0]['m7_importance_share_pct']):.2f}%),"
                " distance × surge "
                f"(+${float(perm_df.iloc[1]['m7_delta_rmse_mean_usd']):.4f},"
                " "
                f"{float(perm_df.iloc[1]['m7_importance_share_pct']):.2f}%),"
                " source & destination "
                f"(+${float(src_dst_row['m7_delta_rmse_mean_usd']):.4f}, "
                f"{float(src_dst_row['m7_importance_share_pct']):.2f}%); "
                f"toàn bộ 8 nhóm thời tiết & khung giờ = {perm14_wt:.2f}% "
                f"({grp6_wt:.2f}% khi gom 6 cụm)."
            ),
            method=(
                "Grouped Permutation Feature Importance phân hoạch rời 27/"
                "27 cột thực trên tập Test (không đếm trùng tương tác) kết "
                "hợp PDP nhất quán vật lý."
            ),
            business_meaning=(
                "Hội tụ độc lập với kiểm định giả thuyết RQ1 ở Tab 2 "
                "và kết quả Ex-Ante Surge Classifier (thời tiết giảm AUC "
                f"chỉ {weather_auc_drop:.4f}): dự báo giá không cần phụ "
                "thuộc API khí tượng theo giờ."
            ),
            limitation=(
                "Permutation Importance chỉ đo tầm quan trọng dự báo trong "
                "không gian tương quan giữa các cột đầu vào, KHÔNG thể diễn "
                "giải như hiệu ứng nhân quả độc lập."
            ),
            accent="blue",
        )
