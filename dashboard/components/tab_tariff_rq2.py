"""
Module: dashboard/components/tab_tariff_rq2.py
Mục đích: Giao diện Tab 3 — Competitive Pricing & Crossing-Point Explorer (RQ2
& H4a/H4b) — Giai đoạn 4.
- Cưỡng chế tuyệt đối HIGH-02: chỉ đối sánh đúng 5 cặp đối xứng Uber vs Lyft,
  tách riêng 2 gói đơn lẻ (WAV & Lux),
  và khóa miền phân tích trong [0.02, 7.86] dặm (tuyệt đối không ngoại suy điểm
  cắt 37.61 dặm cho UberPool vs Shared).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard.business_logic import (
    FilterResult,
    compute_all_pair_comparisons,
    compute_binned_empirical_tariff,
)
from dashboard.charts import (
    PLOTLY_DISPLAY_CONFIG,
    build_tariff_crossing_fig,
    build_tariff_strategy_scatter_fig,
)
from dashboard.components.kpi_cards import render_insight_card, render_kpi_row
from dashboard.config import (
    EMPIRICAL_DISTANCE_MAX,
    EMPIRICAL_DISTANCE_MIN,
    STANDALONE_TIERS,
)


def render_tab_tariff_rq2(
    filter_res: FilterResult,
    master_full: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    v5: dict[str, Any] | None = None,
) -> None:
    """Hiển thị toàn bộ nội dung Tab 3 — Competitive Pricing (RQ2)."""
    tariffs_df = tables["tier_tariffs"]
    hyp_df = tables["hypothesis_tests"]
    h4a = hyp_df.loc[
        hyp_df["hypothesis_id"] == "H4a_Wald_UberX_vs_Lyft_Slope"
    ].iloc[0]

    # Tính đối sánh 5 cặp đối xứng (tỷ lệ chuyến ngắn hơn d* tính trên dữ liệu
    # lọc hiện tại hoặc toàn mẫu nếu lọc rỗng)
    obs_for_share = master_full if filter_res.is_empty else filter_res.df
    pairs_df = compute_all_pair_comparisons(tariffs_df, obs_for_share)

    std_pair = pairs_df.loc[pairs_df["pair_id"] == "standard"].iloc[0]
    xl_pair = pairs_df.loc[pairs_df["pair_id"] == "xl"].iloc[0]
    bl_pair = pairs_df.loc[pairs_df["pair_id"] == "exec_black"].iloc[0]
    bs_pair = pairs_df.loc[pairs_df["pair_id"] == "lux_suv"].iloc[0]
    pool_pair = pairs_df.loc[pairs_df["pair_id"] == "pool_shared"].iloc[0]

    v5_cross = (v5 or {}).get("crossing_distances_5_pairs", [])
    pool_math_d = (
        float(v5_cross[0]["spec_a_train80"]["hc3"]["d_star"])
        if len(v5_cross) >= 1
        else 54.29
    )
    ux_ci = (
        v5_cross[1]["spec_a_train80"]["cluster"]["ci95"]
        if len(v5_cross) >= 2
        else [2.96, 3.46]
    )
    ux_sb = (
        float(v5_cross[1]["spec_b_train80_nonsurge"]["hc3"]["d_star"])
        if len(v5_cross) >= 2
        else 5.17
    )
    xl_sb = (
        float(v5_cross[2]["spec_b_train80_nonsurge"]["hc3"]["d_star"])
        if len(v5_cross) >= 3
        else 8.32
    )
    bl_sb = (
        float(v5_cross[3]["spec_b_train80_nonsurge"]["hc3"]["d_star"])
        if len(v5_cross) >= 4
        else -0.42
    )
    bs_sb = (
        float(v5_cross[4]["spec_b_train80_nonsurge"]["hc3"]["d_star"])
        if len(v5_cross) >= 5
        else 0.86
    )

    # =========================================================================
    # 1. KPI CẠNH TRANH BIỂU GIÁ TRỌNG TÂM
    # =========================================================================
    st.markdown(
        '<div class="section-heading">1. Điểm nhấn Chiến lược Biểu giá Cắt '
        "nhau (Uber vs. Lyft — RQ2)</div>",
        unsafe_allow_html=True,
    )

    render_kpi_row(
        [
            {
                "label": "Điểm cắt Phổ thông (UberX vs. Lyft)",
                "value": (
                    f"d* = {float(std_pair['crossing_distance_miles']):.2f} "
                    "dặm"
                ),
                "subtitle": (
                    f"CI cụm: [{ux_ci[0]:.2f}, {ux_ci[1]:.2f}] mi | Đặc tả "
                    f"B thuần 1.0x: {ux_sb:.2f} mi (Wald Z = "
                    f"{float(h4a['parametric_stat']):.2f})"
                ),
                "accent": "accent",
                "badge_text": (
                    "{:.1f}% chuyến ngắn hơn d*".format(
                        float(std_pair["pct_trips_shorter_than_dstar"] or 83.0)
                    )
                ),
                "badge_class": "badge-verified",
            },
            {
                "label": "Gia đình 6 chỗ (UberXL vs. Lyft XL)",
                "value": (
                    f"A: {float(xl_pair['crossing_distance_miles']):.2f} | "
                    "B: > 7.86 mi"
                ),
                "subtitle": (
                    f"Đặc tả B thuần 1.0x dịch ra {xl_sb:.2f} mi > 7.86 mi "
                    "→ Lyft XL rẻ hơn toàn miền khi 1.0x"
                ),
                "accent": "lyft",
                "badge_text": "Đảo chiều quan trọng khi 1.0x",
                "badge_class": "badge-lyft",
            },
            {
                "label": "Đi ghép (UberPool vs. Shared)",
                "value": "Không cắt nội miền",
                "subtitle": (
                    f"Miền Boston [{EMPIRICAL_DISTANCE_MIN:.2f}, "
                    f"{EMPIRICAL_DISTANCE_MAX:.2f}] dặm: Shared mở cửa thấp "
                    "hơn $"
                    f"{float(pool_pair['base_diff_uber_minus_lyft']):.2f}"
                ),
                "accent": "emerald",
                "badge_text": "Khóa nội miền • Không ngoại suy",
                "badge_class": "badge-verified",
            },
            {
                "label": "Hạng sang (Black & Black SUV)",
                "value": f"Uber lợi thế > {bs_sb:.2f} mi",
                "subtitle": (
                    "Black cắt "
                    f"{float(bl_pair['crossing_distance_miles']):.2f} mi (B:"
                    f" {bl_sb:.2f} mi, Uber rẻ toàn miền) | SUV cắt "
                    f"{float(bs_pair['crossing_distance_miles']):.2f} / "
                    f"{bs_sb:.2f} mi"
                ),
                "accent": "uber",
                "badge_text": "Ưu thế đơn giá/dặm Uber",
                "badge_class": "badge-uber",
            },
        ]
    )

    # =========================================================================
    # 2. BỘ KHÁM PHÁ ĐIỂM CẮT BIỂU GIÁ (CROSSING-POINT EXPLORER)
    # =========================================================================
    st.markdown(
        '<div class="section-heading">2. Khám phá Đường Biểu giá &amp; Điểm '
        "cắt Cạnh tranh (5 Cặp Đối xứng)</div>",
        unsafe_allow_html=True,
    )

    pair_labels = pairs_df["segment_label"].tolist()
    selected_label = st.selectbox(
        "Chọn phân khúc cạnh tranh đối xứng giữa Uber và Lyft (Đặc tả A "
        "trên mẫu quan sát surge thực):",
        options=pair_labels,
        index=1,  # Mặc định: Phổ thông 4 chỗ (UberX vs Lyft)
        key="tab3_selected_pair",
    )
    pair_info = (
        pairs_df.loc[pairs_df["segment_label"] == selected_label]
        .iloc[0]
        .to_dict()
    )

    if not pair_info["crossing_in_domain"]:
        st.info(
            f"🛡️ **Lưu ý khóa miền nội suy ({pair_info['uber_tier']} vs. "
            f"{pair_info['lyft_tier']}):** "
            "Khoảng cách tối đa quan sát được tại Boston là "
            f"**{EMPIRICAL_DISTANCE_MAX:.2f} dặm**. "
            f"Do {pair_info['lyft_tier']} có giá mở cửa thấp hơn tới "
            f"**${pair_info['base_diff_uber_minus_lyft']:.2f}** "
            "trong khi đơn giá/dặm chỉ cao hơn "
            f"**${pair_info['slope_diff_lyft_minus_uber']:.2f}/dặm**, "
            "hai đường biểu giá **KHÔNG cắt nhau trong miền quan sát** — "
            "tuyệt đối không ngoại suy giao điểm ngoài phạm vi dữ liệu."
        )

    binned_df = compute_binned_empirical_tariff(
        obs_for_share, pair_info["uber_tier"], pair_info["lyft_tier"]
    )
    fig_cross = build_tariff_crossing_fig(
        tariffs_df, pair_info, binned_df=binned_df
    )
    st.plotly_chart(fig_cross, width="stretch", config=PLOTLY_DISPLAY_CONFIG)

    # =========================================================================
    # 3. BẢN ĐỒ CHIẾN LƯỢC GIÁ MỞ CỬA (β0) VS ĐƠN GIÁ/DẶM (β1) CHO 12 GÓI
    # =========================================================================
    st.markdown(
        '<div class="section-heading">3. Bản đồ Chiến lược Định giá 12 Gói '
        "Dịch vụ (β₀ vs. β₁)</div>",
        unsafe_allow_html=True,
    )
    fig_strat = build_tariff_strategy_scatter_fig(tariffs_df)
    st.plotly_chart(fig_strat, width="stretch", config=PLOTLY_DISPLAY_CONFIG)

    # =========================================================================
    # 4. BẢNG ĐỐI SÁNH 5 CẶP ĐỐI XỨNG & 2 GÓI ĐỘC LẬP (WAV, LUX)
    # =========================================================================
    st.markdown(
        '<div class="section-heading">4. Bảng Tham số Biểu giá HC3: 5 Cặp '
        "Đối xứng &amp; 2 Gói Độc lập</div>",
        unsafe_allow_html=True,
    )
    col_t1, col_t2 = st.columns([1.6, 1.0], gap="medium")
    with col_t1:
        st.markdown(
            "**Bảng 3A — 5 Cặp Cạnh tranh Đối xứng Trực tiếp "
            "(Uber vs. Lyft — Đặc tả A: d × surge; xem Đặc tả B thuần 1.0x "
            "ở trên)**"
        )
        p_view = pairs_df[
            [
                "short_label",
                "uber_tier",
                "uber_base_b0",
                "uber_slope_b1",
                "lyft_tier",
                "lyft_base_b0",
                "lyft_slope_b1",
                "crossing_distance_miles",
                "cheaper_at_1_5_mi",
                "cheaper_at_4_5_mi",
            ]
        ].copy()
        p_view["crossing_distance_miles"] = p_view[
            "crossing_distance_miles"
        ].apply(
            lambda v: (
                f"{v:.2f} dặm" if pd.notna(v) else "Không cắt (≤ 7.86 mi)"
            )
        )
        p_view.columns = [
            "Phân khúc",
            "Uber",
            "β₀ Uber ($)",
            "β₁ Uber ($/mi)",
            "Lyft",
            "β₀ Lyft ($)",
            "β₁ Lyft ($/mi)",
            "Điểm cắt d*",
            "Rẻ hơn ở 1.5 mi",
            "Rẻ hơn ở 4.5 mi",
        ]
        st.dataframe(p_view, width="stretch", hide_index=True)

    with col_t2:
        st.markdown(
            "**Bảng 3B — 2 Gói Dịch vụ Độc lập (Tách riêng đối chứng)**"
        )
        standalone_names = [s["tier"] for s in STANDALONE_TIERS]
        s_view = tariffs_df.loc[
            tariffs_df["service_tier"].isin(standalone_names)
        ][
            [
                "cab_type",
                "service_tier",
                "base_fare_b0_usd",
                "per_mile_rate_b1_usd",
                "expected_fare_2_5mi_usd",
                "r2_within_tier",
            ]
        ].copy()
        s_view.columns = [
            "Hãng",
            "Gói đơn lẻ",
            "β₀ ($)",
            "β₁ ($/mi)",
            "Giá 2.5 mi ($)",
            "R² nội gói",
        ]
        st.dataframe(s_view, width="stretch", hide_index=True)
        st.caption(
            "**WAV** (xe hỗ trợ xe lăn của Uber) và **Lux** (sedan cao cấp "
            "trung gian của Lyft) không có đối thủ tương đương 1-1 "
            "ở hãng còn lại nên được tách bảng riêng để tránh ghép cặp khập "
            "khiễng."
        )

    # =========================================================================
    # 5. INSIGHT 5 PHẦN CHUẨN HỌC THUẬT (RQ2)
    # =========================================================================
    st.markdown(
        '<div class="section-heading">5. Insight Cạnh tranh Biểu giá &amp; '
        "Hàm ý Kinh doanh (RQ2)</div>",
        unsafe_allow_html=True,
    )
    i1, i2 = st.columns(2, gap="medium")
    with i1:
        render_insight_card(
            title=(
                "Insight 3.1 — Đối sánh Song đặc tả A/B: UberX–Lyft cắt bền "
                "vững; Lyft XL rẻ hơn toàn miền khi 1.0x"
            ),
            finding=(
                "Trên 5/5 cặp đối thủ đối xứng, Lyft luôn đặt giá mở cửa "
                "(β₀) thấp hơn Uber nhưng đơn giá mỗi dặm (β₁) cao hơn. Tuy "
                "nhiên, khi kiểm soát thuần trạng thái không surge (Đặc tả "
                "B) trên mẫu surge quan sát thực, cặp gia đình 6 chỗ đảo "
                "chiều quan trọng: Lyft XL rẻ hơn UberXL trên toàn miền "
                "Boston."
            ),
            evidence=(
                "Phổ thông (UberX vs. Lyft): Đặc tả A cắt tại d* = "
                f"{float(std_pair['crossing_distance_miles']):.2f} mi (CI "
                f"cụm [{ux_ci[0]:.2f}, {ux_ci[1]:.2f}], "
                f"Wald Z = {float(h4a['parametric_stat']):.2f}, p < 0.0001) "
                f"và Đặc tả B thuần 1.0x tại {ux_sb:.2f} mi. "
                "Gia đình 6 chỗ (UberXL vs. Lyft XL): Đặc tả A cắt tại "
                f"{float(xl_pair['crossing_distance_miles']):.2f} mi nhưng "
                f"Đặc tả B thuần 1.0x dịch ra {xl_sb:.2f} mi > 7.86 mi. "
                "Black vs. Lux Black cắt "
                f"{float(bl_pair['crossing_distance_miles']):.2f} mi (B: "
                f"{bl_sb:.2f} mi → Uber Black rẻ hơn toàn miền); SUV hạng "
                f"sang cắt {float(bs_pair['crossing_distance_miles']):.2f} /"
                f" {bs_sb:.2f} mi."
            ),
            method=(
                "Hồi quy tuyến tính OLS sai số chuẩn HC3 kết hợp Cluster-"
                "Robust SE (Delta-method 95% CI) song hành hai đặc tả trên "
                "X_train (n = 186,288)."
            ),
            business_meaning=(
                "Khi không surge, hành khách nên chọn Lyft cho chặng ngắn "
                "phổ thông (< "
                f"{float(std_pair['crossing_distance_miles']):.2f}–"
                f"{ux_sb:.2f} mi) và chọn Lyft XL trên mọi tuyến "
                "nội đô Boston ở hạng 6 chỗ; chọn UberX cho chặng dài và "
                "chọn Uber Black / Black SUV cho phân khúc doanh nhân."
            ),
            limitation=(
                "Điểm cắt chỉ đúng khi Lyft ở trạng thái không surge (1.0x) "
                "và hệ số β₁ chịu thiên lệch bỏ sót biến thời lượng di "
                "chuyển thực (duration). Khi Lyft bật surge ≥ 1.25x, ưu thế "
                "lập tức đảo chiều về Uber."
            ),
            accent="lyft",
        )
    with i2:
        render_insight_card(
            title=(
                "Insight 3.2 — Phân khúc Đi ghép (Shared vs. UberPool) & Độ "
                "cong Giá sàn < 1 dặm"
            ),
            finding=(
                "Ở phân khúc tiết kiệm nhất, Lyft Shared rẻ hơn UberPool "
                "trên toàn bộ miền quãng đường Boston [0.02, 7.86] dặm; "
                "đồng thời cả hai gói đi ghép có R² nội phân khúc thấp nhất "
                "(~0.42–0.43) do phụ thuộc xác suất ghép khách và ngưỡng "
                "giá sàn tối thiểu."
            ),
            evidence=(
                f"Lyft Shared (β₀ = ${float(pool_pair['lyft_base_b0']):.2f},"
                f" β₁ = ${float(pool_pair['lyft_slope_b1']):.2f}/mi) rẻ hơn "
                f"UberPool (β₀ = ${float(pool_pair['uber_base_b0']):.2f}, "
                f"β₁ = ${float(pool_pair['uber_slope_b1']):.2f}/mi) trên "
                "toàn miền quan sát "
                f"(giao điểm toán học {pool_math_d:.2f} mi nằm ngoài miền "
                "7.86 mi). Các "
                "chấm giá TB thực tế ở cự ly < 1 dặm nằm phẳng hơn đường "
                "thẳng OLS do giá sàn tối thiểu."
            ),
            method=(
                "Đối chiếu đường biểu giá HC3 với giá trung bình thực "
                "nghiệm theo nhóm cự ly (0.5 dặm/nhóm) trên miền [0.02, "
                "7.86] dặm."
            ),
            business_meaning=(
                "Hành khách tối ưu chi phí nội đô không có nhu cầu gấp nên "
                "ưu tiên Lyft Shared khi không có surge; đồng thời đội kỹ "
                "thuật cần dùng mô hình phi tuyến (M7 HistGBR) để nắm bắt "
                "chính xác ngưỡng giá sàn ở chặng < 1 dặm."
            ),
            limitation=(
                "Không có dữ liệu số hành khách ghép chung thực tế "
                "(matching load) hay thời gian đi vòng đón thêm khách — "
                "những yếu tố làm giảm R² nội gói của Shared/UberPool."
            ),
            accent="emerald",
        )
