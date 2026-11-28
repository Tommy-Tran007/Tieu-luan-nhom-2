"""
Module: dashboard/components/tab_pricing_rq1.py
Mục đích: Giao diện Tab 2 — Pricing, Weather & Time (RQ1 & Giả thuyết H1–H3) —
Giai đoạn 4.
- Mọi biểu đồ thời tiết khóa cứng weather_matched == 1 (CRIT-01, tối đa n =
  179,280; riêng mưa/khô sạch loại sentinel là n = 152,043).
- Mọi biểu đồ giờ/thứ/buổi khóa cứng is_timestamp_original == 1 (CRIT-02, tối
  đa n = 208,034, Boston EST).
- Hiển thị đồng thời p-value kép (Tham số + Phi tham số), Effect Size và Ý
  nghĩa thực tiễn.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard.business_logic import (
    FilterResult,
    compute_dayofweek_est_profile,
    compute_hourly_est_profile,
    compute_surge_distribution,
    compute_weather_rain_comparison,
    prepare_hypothesis_table,
)
from dashboard.charts import (
    PLOTLY_DISPLAY_CONFIG,
    build_dayofweek_est_fig,
    build_hourly_est_fig,
    build_surge_levels_fig,
    build_weather_rain_audit_fig,
)
from dashboard.components.kpi_cards import render_insight_card, render_kpi_row


def render_tab_pricing_rq1(
    filter_res: FilterResult,
    tables: dict[str, pd.DataFrame],
    v5: dict[str, Any] | None = None,
) -> None:
    """Hiển thị toàn bộ nội dung Tab 2 — Pricing, Weather & Time (RQ1)."""
    hyp_df = tables["hypothesis_tests"]
    h1a = hyp_df.loc[hyp_df["hypothesis_id"] == "H1a_All_Rides"].iloc[0]
    h2a = hyp_df.loc[hyp_df["hypothesis_id"] == "H2a_Surge_All"].iloc[0]
    h2c = hyp_df.loc[hyp_df["hypothesis_id"] == "H2c_Price_All"].iloc[0]
    h3a = hyp_df.loc[hyp_df["hypothesis_id"] == "H3a_Surge_x_CabType"].iloc[0]
    h3d = hyp_df.loc[
        hyp_df["hypothesis_id"] == "H3d_Surge_x_RushHour_Lyft"
    ].iloc[0]
    h4c = hyp_df.loc[
        hyp_df["hypothesis_id"] == "H4c_Wald_Joint_Weather_Multivariate"
    ].iloc[0]

    st_v5 = (v5 or {}).get("statistical_rigor_extensions", {})
    h3b_p_hr = float(
        st_v5.get("h3b_lyft_surge_rain_cluster", {}).get(
            "p_hour_cluster", 0.0793
        )
    )
    hv_v5 = st_v5.get("heavy_rain_vs_dry", {})
    wcr_p = float(
        hv_v5.get("exact_wcr_loc_hr_B1999", {}).get("p_wild", 0.8205)
    )
    n_heavy_hrs = int(hv_v5.get("n_heavy_citywide_hours", 8))
    n_h1a = int(h1a["n_group1"]) + int(h1a["n_group2"])
    n_h2a = int(h2a["n_group1"]) + int(h2a["n_group2"])

    if filter_res.is_empty:
        st.warning(
            "⚠️ **Bộ lọc hiện tại trả về 0 quan sát.** "
            f"{filter_res.empty_reason}"
        )

    weather_df, weather_meta = compute_weather_rain_comparison(filter_res.df)
    hourly_df, hourly_meta = compute_hourly_est_profile(filter_res.df)
    dow_df = compute_dayofweek_est_profile(filter_res.df)
    surge_dist_df = compute_surge_distribution(filter_res.df)

    # =========================================================================
    # 1. KPI TÓM TẮT KIỂM ĐỊNH RQ1 (P-VALUE + EFFECT SIZE)
    # =========================================================================
    st.markdown(
        '<div class="section-heading">1. Tóm tắt Kiểm chứng RQ1: Thời tiết, '
        "Khung giờ EST &amp; Surge</div>",
        unsafe_allow_html=True,
    )

    render_kpi_row(
        [
            {
                "label": "H1a: Mưa vs. Không mưa (Sạch Sentinel)",
                "value": f"d = {float(h1a['effect_size_value']):+.5f}",
                "subtitle": (
                    f"Welch p = {float(h1a['parametric_pvalue']):.4f} | "
                    f"TOST ±$0.10 p < 0.0001 | n = {n_h1a:,}"
                ),
                "accent": "accent",
                "badge_text": f"Exact WCR mưa lớn p = {wcr_p:.4f}",
                "badge_class": "badge-verified",
            },
            {
                "label": "H4c: Đa biến 6 yếu tố Thời tiết",
                "value": f"ΔR² = {float(h4c['effect_size_value']):.7f}",
                "subtitle": (
                    f"Joint Wald F = {float(h4c['parametric_stat']):.4f} (p "
                    f"= {float(h4c['parametric_pvalue']):.4f}) trên n = "
                    f"{int(h4c['n_group1']):,}"
                ),
                "accent": "emerald",
                "badge_text": "≈ 0.000% phương sai giá",
                "badge_class": "badge-verified",
            },
            {
                "label": "H2a: Surge theo 4 Buổi EST",
                "value": f"η² = {float(h2a['effect_size_value']):.6f}",
                "subtitle": (
                    f"ANOVA p = {float(h2a['parametric_pvalue']):.4f} | "
                    f"Kruskal p = {float(h2a['nonparametric_pvalue']):.4f} |"
                    f" n = {n_h2a:,}"
                ),
                "accent": "amber",
                "badge_text": "Chưa bác bỏ H0 • Ổn định 24h",
                "badge_class": "badge-verified",
            },
            {
                "label": "H3a: Surge × Hãng xe (Lyft vs Uber)",
                "value": f"V = {float(h3a['effect_size_value']):.4f}",
                "subtitle": (
                    f"χ² = {float(h3a['parametric_stat']):,.2f} (p < "
                    "0.0001) | Lyft "
                    f"{float(h3a['mean_group1_or_rate']):.2f}% vs Uber 0.00%"
                ),
                "accent": "lyft",
                "badge_text": f"H3b cụm giờ p = {h3b_p_hr:.4f} (NS)",
                "badge_class": "badge-lyft",
            },
        ]
    )

    # =========================================================================
    # 2. BIỂU ĐỒ THỜI TIẾT & NHỊP 24 GIỜ BOSTON EST
    # =========================================================================
    st.markdown(
        '<div class="section-heading">2. Kiểm toán Tác động Thời tiết (Sạch '
        "lỗi cảm biến mưa) &amp; Nhịp 24 Giờ Boston EST</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Nguyên tắc bảo toàn nguồn gốc: Biểu đồ thời tiết chỉ dùng bản ghi "
        "khớp trạm khí tượng thực không lỗi cảm biến mưa "
        "(152,043 dòng; tách 91,618 dòng không khớp và 27,237 dòng ô "
        "sentinel); biểu đồ thời gian chỉ dùng dấu thời gian gốc EST "
        "(208,034 dòng; 0 dòng dùng `ffill`); tỷ lệ Surge tính trên mẫu "
        "quan sát thực (`92,384` lượt Lyft)."
    )

    fig_weather = build_weather_rain_audit_fig(weather_df, weather_meta)
    st.plotly_chart(fig_weather, width="stretch", config=PLOTLY_DISPLAY_CONFIG)

    fig_hourly = build_hourly_est_fig(hourly_df, hourly_meta)
    st.plotly_chart(fig_hourly, width="stretch", config=PLOTLY_DISPLAY_CONFIG)

    # =========================================================================
    # 3. PHÂN TÍCH THEO THỨ TRONG TUẦN (EST) & PHÂN BỐ 7 BẬC SURGE LYFT
    # =========================================================================
    st.markdown(
        '<div class="section-heading">3. Biến động theo Thứ trong tuần '
        "(EST) &amp; Phân bố 7 Bậc Surge của Lyft</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        fig_dow = build_dayofweek_est_fig(dow_df)
        st.plotly_chart(fig_dow, width="stretch", config=PLOTLY_DISPLAY_CONFIG)
    with c2:
        fig_surge = build_surge_levels_fig(surge_dist_df)
        st.plotly_chart(
            fig_surge, width="stretch", config=PLOTLY_DISPLAY_CONFIG
        )

    # =========================================================================
    # 4. BẢNG 13 KIỂM ĐỊNH THỐNG KÊ KÉP (P-VALUE + EFFECT SIZE + Ý NGHĨA THỰC
    # TIỄN)
    # =========================================================================
    st.markdown(
        '<div class="section-heading">4. Bảng Kết quả Kiểm định Giả thuyết '
        "Thống kê Kép (H1–H4)</div>",
        unsafe_allow_html=True,
    )
    rq_choice = st.radio(
        "Lọc bảng kiểm định theo câu hỏi nghiên cứu:",
        options=["Tất cả", "RQ1", "RQ2"],
        horizontal=True,
        key="tab2_hyp_filter",
    )
    hyp_view = prepare_hypothesis_table(hyp_df, rq_filter=rq_choice)[
        [
            "hypothesis_id",
            "research_question",
            "scope",
            "parametric_test",
            "parametric_pvalue",
            "nonparametric_test",
            "nonparametric_pvalue",
            "effect_size_name",
            "effect_size_value",
            "statistical_decision",
            "practical_significance",
            "economic_conclusion",
        ]
    ].copy()
    h3b_mask = hyp_view["hypothesis_id"] == "H3b_Surge_x_Rain_Lyft"
    if h3b_mask.any():
        hyp_view.loc[h3b_mask, "statistical_decision"] = (
            f"Chưa bác bỏ H0 sau gom cụm/Holm (cụm giờ p={h3b_p_hr:.4f})"
        )
    hyp_view.columns = [
        "Mã giả thuyết",
        "RQ",
        "Phạm vi mẫu kiểm định",
        "Kiểm định tham số",
        "p-value (Tham số)",
        "Kiểm định phi tham số",
        "p-value (Phi tham số)",
        "Độ lớn hiệu ứng",
        "Giá trị Effect Size",
        "Quyết định H0",
        "Ý nghĩa thực tiễn",
        "Kết luận kinh tế",
    ]
    st.dataframe(hyp_view, width="stretch", hide_index=True)

    # =========================================================================
    # 5. INSIGHT 5 PHẦN CHUẨN HỌC THUẬT (KHÔNG NHẦM TƯƠNG QUAN THÀNH NHÂN QUẢ)
    # =========================================================================
    st.markdown(
        '<div class="section-heading">5. Diễn giải Học thuật &amp; Hàm ý '
        "Kinh doanh (RQ1)</div>",
        unsafe_allow_html=True,
    )
    ic1, ic2 = st.columns(2, gap="medium")
    with ic1:
        render_insight_card(
            title=(
                "Insight 2.1 — Bác bỏ ngộ nhận 'Trời mưa tự động làm tăng "
                "giá báo API một cách cơ học'"
            ),
            finding=(
                "Trên dữ liệu khí tượng sạch lỗi cảm biến mưa (tách riêng "
                "27,237 dòng ô sentinel), mưa nhẹ–vừa không tạo mức tăng "
                "giá báo API hay xác suất Surge Lyft có ý nghĩa thống kê "
                "sau gom cụm."
            ),
            evidence=(
                f"Tập khí tượng sạch (n = {n_h1a:,}): Giá TB Mưa = "
                f"${float(h1a['mean_group1_or_rate']):.4f} (n = "
                f"{int(h1a['n_group1']):,}) "
                f"vs Tạnh = ${float(h1a['mean_group2_or_ref']):.4f} (n = "
                f"{int(h1a['n_group2']):,}), Welch p = "
                f"{float(h1a['parametric_pvalue']):.4f}, "
                "TOST ±$0.10 p < 0.0001; Exact WCR mưa lớn p = "
                f"{wcr_p:.4f}. Joint Wald 6 biến thời tiết H4c (n = "
                f"{int(h4c['n_group1']):,}): "
                f"F = {float(h4c['parametric_stat']):.4f}, p = "
                f"{float(h4c['parametric_pvalue']):.4f}, ΔR² = "
                f"{float(h4c['effect_size_value']):.7f}. "
                "H3b Lyft Surge×Mưa khi gom cụm theo giờ không còn ý nghĩa "
                f"(p = {h3b_p_hr:.4f}, trượt Holm & BH-FDR)."
            ),
            method=(
                "Kiểm định kép Welch + Mann-Whitney (H1a–c), sai số chuẩn "
                "cụm hai cấp (2,795 quận×giờ & 329 giờ Boston), Exact WCR "
                "(B = 1,999), TOST ±$0.10 và Joint Wald HC3 hạng đầy đủ "
                "(H4c)."
            ),
            business_meaning=(
                "Không nên đưa dữ liệu thời tiết gộp theo giờ làm đầu vào "
                "chính cho quy tắc định giá cơ học; ngân sách kỹ thuật nên "
                "tập trung vào trạng thái cung–cầu vi mô cục bộ theo phút."
            ),
            limitation=(
                "Bản tin khí tượng tổng hợp theo giờ và chỉ phủ 24 ngày "
                "cuối thu ôn hòa "
                f"(max rain 0.55 in/h, chỉ {n_heavy_hrs} giờ mưa lớn ≥ 0.10 "
                "in/h toàn Boston, "
                "chưa có bão tuyết mùa đông). Kết quả phản ánh tương quan "
                "quan sát, không thay "
                "thế thực nghiệm nhân quả."
            ),
            accent="blue",
        )
    with ic2:
        render_insight_card(
            title=(
                "Insight 2.2 — Khung giờ EST chi phối lưu lượng nhưng không "
                "tạo biểu giá cứng theo giờ"
            ),
            finding=(
                "Sau quy đổi chuẩn múi giờ America/New_York (EST) và giữ "
                "nguyên 62,864 dòng khuyết timestamp (0 ffill), lưu lượng "
                "khớp nhịp đô thị (đáy 03:00, đỉnh 19:00 EST) nhưng giá TB "
                "và tỷ lệ Surge Lyft rất phẳng xuyên suốt 24h."
            ),
            evidence=(
                "Trên mẫu TS gốc EST: ANOVA giá theo 4 buổi (H2c) cho F = "
                f"{float(h2c['parametric_stat']):.4f}, p = "
                f"{float(h2c['parametric_pvalue']):.4f}, "
                f"η² = {float(h2c['effect_size_value']):.6f}; tỷ lệ Surge "
                "Lyft quan sát thực đạt "
                f"{float(h3a['mean_group1_or_rate']):.2f}% "
                "(6,280 / 92,384 lượt) và hầu như không đổi giữa giờ cao "
                "điểm vs ngoài cao điểm "
                f"({float(h3d['mean_group1_or_rate']):.2f}% vs "
                f"{float(h3d['mean_group2_or_ref']):.2f}%, p = "
                f"{float(h3d['parametric_pvalue']):.4f})."
            ),
            method=(
                "One-way ANOVA + Kruskal-Wallis H (H2a–c), Chi-Square + "
                "Cramér's V (H3c–d) khóa is_timestamp_original == 1 và mẫu "
                "surge quan sát thực."
            ),
            business_meaning=(
                "Surge của Lyft tập trung theo điểm nóng địa bàn (Back Bay "
                "11.17%, Fenway 10.57%, Northeastern University 10.10%, "
                "Theatre District 9.58%, Boston University 9.57% — gấp 7–8 "
                "lần North End 1.33%) hơn là lịch giờ cố định."
            ),
            limitation=(
                "Bộ dữ liệu có 6 ngày khuyết thu thập gốc trong tháng 12/"
                "2018 (05–08/12, 10–11/12) và 5 khung giờ đêm khuyết, đồng "
                "thời thiếu biến thời lượng chạy thực tế (duration) và số "
                "tài xế rảnh."
            ),
            accent="amber",
        )
