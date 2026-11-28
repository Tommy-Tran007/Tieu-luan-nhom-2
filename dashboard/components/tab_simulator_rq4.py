"""
Module: dashboard/components/tab_simulator_rq4.py
Mục đích: Giao diện Tab 5 — Live API Fare Predictor & Surge Cap Policy
Simulator (RQ4) — Giai đoạn 6.

Tuân thủ nghiêm ngặt các chốt chặn học thuật:
- Phần A (Live Fare Predictor):
  * Gọi trực tiếp models/best_model.joblib (M7) &
    models/m2_interaction_ols.joblib (M2) với 27 đặc trưng chuẩn.
  * Kiểm tra 72 tuyến thực tế Boston và miền [0.02, 7.86] dặm; kích hoạt
    EXTRAPOLATION WARNING nếu vượt miền.
  * Ghi rõ kết quả là Dự báo Giá Báo API (API Fare Quote); dải ±MPIW/2 gắn nhãn
    ASSUMPTION (cấp phân khúc).
- Phần B (Surge Cap Policy Simulator — HIGH-03):
  * Gắn nhãn nổi bật SCENARIO / SIMULATION.
  * Sử dụng nhất quán thuật ngữ 'Tổng giá trị báo giá Lyft (API Quote Proxy)' —
    KHÔNG gọi là doanh thu thực.
  * Công bố đầy đủ 3 giả định & giới hạn kinh tế học (API Quote phân tầng, Cung
    tài xế tĩnh, Bão hòa cây 2.00x).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard.business_logic import (
    extract_surge_cap_scenario,
    prepare_surge_cap_display_table,
)
from dashboard.charts import (
    PLOTLY_DISPLAY_CONFIG,
    build_live_fare_comparison_fig,
    build_surge_cap_m7_vs_m2_fig,
    build_surge_cap_tradeoff_fig,
)
from dashboard.components.kpi_cards import render_insight_card, render_kpi_row
from dashboard.config import (
    DAY_NAME_VI,
    EMPIRICAL_DISTANCE_MAX,
    EMPIRICAL_DISTANCE_MIN,
    VALID_BOSTON_DISTRICTS,
)
from dashboard.model_loader import predict_fares_all_tiers


def render_tab_simulator_rq4(
    tables: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    models_info: dict[str, Any],
    route_catalog: pd.DataFrame,
    weather_ref: dict[str, float],
    v5: dict[str, Any] | None = None,
) -> None:
    """
    Hiển thị toàn bộ nội dung Tab 5 — Live Fare Predictor & Surge Cap Policy
    Simulator.
    """
    # =========================================================================
    # PHẦN A: LIVE API FARE PREDICTOR (M7 HISTGBR + M2 OLS ĐỐI CHỨNG)
    # =========================================================================
    st.markdown(
        '<div class="section-heading">Phần A — Công cụ Dự báo Giá Báo API '
        "Trực tiếp (Live Fare Predictor: M7 &amp; M2)</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Chạy suy luận trực tiếp trên mô hình đã đóng gói (`models/"
        "best_model.joblib` & `m2_interaction_ols.joblib` — 27 đặc trưng "
        "chuẩn). "
        "Kết quả là **Giá Báo API dự kiến (API Fare Quote)**, không bao gồm "
        "tiền tip hoặc tắc đường thời gian thực."
    )

    allow_extrap = st.toggle(
        "🧪 Bật chế độ nhập tự do để kiểm thử cảnh báo ngoại suy "
        "(Extrapolation Demo)",
        value=False,
        key="tab5_allow_extrap",
        help=(
            "Khi tắt (mặc định), chỉ hiển thị 6 điểm đến kết nối thực tế "
            "của quận đón và cự ly trong miền Boston [0.02, 7.86] dặm."
        ),
    )

    c_in1, c_in2, c_in3, c_in4 = st.columns([1.2, 1.2, 1.1, 1.0], gap="medium")
    with c_in1:
        source = st.selectbox(
            "Điểm đón (`source`)",
            options=list(VALID_BOSTON_DISTRICTS),
            index=0,  # Back Bay
            key="tab5_source",
        )

    valid_dests = sorted(
        route_catalog.loc[route_catalog["source"] == source, "destination"]
        .unique()
        .tolist()
    )
    dest_options = (
        [d for d in VALID_BOSTON_DISTRICTS if d != source]
        if allow_extrap
        else valid_dests
    )
    default_dest_idx = (
        dest_options.index("South Station")
        if "South Station" in dest_options
        else 0
    )

    with c_in2:
        destination = st.selectbox(
            "Điểm đến (`destination`)",
            options=dest_options,
            index=default_dest_idx,
            key="tab5_dest",
        )

    # Tra trung vị quãng đường thực tế của tuyến đang chọn (nếu tuyến tồn tại
    # trong 72 tuyến Boston)
    rmatch = route_catalog.loc[
        (route_catalog["source"] == source)
        & (route_catalog["destination"] == destination)
    ]
    suggested_dist = (
        float(rmatch.iloc[0]["median_distance_miles"])
        if not rmatch.empty
        else 2.20
    )

    with c_in3:
        if allow_extrap:
            distance = st.number_input(
                "Quãng đường (`distance`, dặm)",
                min_value=0.01,
                max_value=25.0,
                value=suggested_dist,
                step=0.25,
                key="tab5_dist_free",
                help=(
                    "Miền quan sát thực tế tại Boston là "
                    f"[{EMPIRICAL_DISTANCE_MIN:.2f}, "
                    f"{EMPIRICAL_DISTANCE_MAX:.2f}] dặm. Nhập ngoài miền để "
                    "kiểm thử EXTRAPOLATION WARNING."
                ),
            )
        else:
            distance = st.slider(
                f"Quãng đường (dặm • TV tuyến: {suggested_dist:.2f})",
                min_value=float(EMPIRICAL_DISTANCE_MIN),
                max_value=float(EMPIRICAL_DISTANCE_MAX),
                value=min(
                    max(suggested_dist, EMPIRICAL_DISTANCE_MIN),
                    EMPIRICAL_DISTANCE_MAX,
                ),
                step=0.05,
                key="tab5_dist_slider",
            )

    with c_in4:
        surge_options = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0] + (
            [3.5] if allow_extrap else []
        )
        lyft_surge = st.select_slider(
            "Hệ số Surge Lyft (`x`)",
            options=surge_options,
            value=1.0,
            key="tab5_lyft_surge",
            help="Uber luôn cố định surge = 1.0x trong bộ dữ liệu Boston.",
        )

    c_ctx1, c_ctx2, c_ctx3, c_ctx4 = st.columns(
        [1.0, 1.1, 1.0, 1.2], gap="medium"
    )
    with c_ctx1:
        hour_est = st.slider(
            "Giờ khởi hành (Boston EST)",
            min_value=0,
            max_value=23,
            value=8,
            key="tab5_hour",
        )
    with c_ctx2:
        dow = st.selectbox(
            "Thứ trong tuần (EST)",
            options=list(DAY_NAME_VI.keys()),
            format_func=lambda k: DAY_NAME_VI[k],
            index=2,
            key="tab5_dow",
        )
    with c_ctx3:
        is_raining = st.checkbox(
            "Trời đang mưa (`is_raining = 1`)", value=False, key="tab5_rain"
        )
    with c_ctx4:
        temp_f = st.slider(
            "Nhiệt độ không khí (°F)",
            min_value=20.0,
            max_value=56.0,
            value=40.0,
            step=1.0,
            key="tab5_temp",
        )

    # Chạy suy luận trực tiếp M7 + M2
    pred_res = predict_fares_all_tiers(
        models=models_info,
        pi_df=tables["prediction_interval"],
        route_catalog=route_catalog,
        weather_ref=weather_ref,
        source=source,
        destination=destination,
        distance=float(distance),
        lyft_surge=float(lyft_surge),
        hour=int(hour_est),
        day_of_week=int(dow),
        is_raining=bool(is_raining),
        temp_f=float(temp_f),
    )
    chk = pred_res["input_check"]

    if not chk.is_valid:
        for err in chk.errors:
            st.error(f"❌ {err}")
    else:
        if chk.is_extrapolation:
            for warn in chk.warnings:
                st.error(f"🚨 **{warn}**")
        elif chk.warnings:
            for warn in chk.warnings:
                st.warning(f"⚠️ {warn}")

        pairs_df = pred_res["pairs_df"]
        std_row = pairs_df.iloc[1]  # Phổ thông: UberX vs Lyft
        pool_row = pairs_df.iloc[0]  # Đi ghép: UberPool vs Shared

        render_kpi_row(
            [
                {
                    "label": "Phổ thông — UberX (M7 Dự báo)",
                    "value": f"${std_row['Giá báo Uber (M7, USD)']:.2f}",
                    "subtitle": (
                        "Đối chứng M2 OLS: "
                        f"${std_row['Giá báo Uber (M2 OLS, USD)']:.2f} | "
                        "Surge Uber = 1.00x"
                    ),
                    "accent": "uber",
                    "badge_text": "API Fare Quote",
                    "badge_class": "badge-uber",
                },
                {
                    "label": f"Phổ thông — Lyft (Surge {lyft_surge:.2f}x)",
                    "value": f"${std_row['Giá báo Lyft (M7, USD)']:.2f}",
                    "subtitle": (
                        "Đối chứng M2 OLS: "
                        f"${std_row['Giá báo Lyft (M2 OLS, USD)']:.2f}"
                    ),
                    "accent": "lyft",
                    "badge_text": "API Fare Quote",
                    "badge_class": "badge-lyft",
                },
                {
                    "label": "Gợi ý Phổ thông Rẻ hơn (M7)",
                    "value": str(std_row["Hãng rẻ hơn (theo M7)"]),
                    "subtitle": (
                        f"Chênh lệch: ${std_row['Chênh lệch (USD)']:.2f} "
                        f"(~{std_row['Tiết kiệm (%)']:.1f}%)"
                    ),
                    "accent": "emerald",
                    "badge_text": "EXTRAPOLATION!"
                    if chk.is_extrapolation
                    else "VERIFIED • Trong miền",
                    "badge_class": "badge-assumption"
                    if chk.is_extrapolation
                    else "badge-verified",
                },
                {
                    "label": "Đi ghép Rẻ hơn (Pool vs. Shared)",
                    "value": str(pool_row["Hãng rẻ hơn (theo M7)"]),
                    "subtitle": (
                        f"UberPool ${pool_row['Giá báo Uber (M7, USD)']:.2f}"
                        " vs Shared $"
                        f"{pool_row['Giá báo Lyft (M7, USD)']:.2f}"
                    ),
                    "accent": "accent",
                    "badge_text": f"Chênh ${pool_row['Chênh lệch (USD)']:.2f}",
                    "badge_class": "badge-simulation",
                },
            ]
        )

        st.markdown(
            "<div style='height:0.45rem;'></div>", unsafe_allow_html=True
        )
        fig_live = build_live_fare_comparison_fig(
            pairs_df,
            route_label=f"{source} → {destination}",
            distance=float(distance),
            lyft_surge=float(lyft_surge),
        )
        st.plotly_chart(
            fig_live, width="stretch", config=PLOTLY_DISPLAY_CONFIG
        )

        tb1, tb2 = st.columns([1.65, 1.0], gap="medium")
        with tb1:
            st.markdown(
                "**Bảng 5A — Đối chiếu Dự báo Giá Báo API trên 5 Phân khúc "
                "Đối xứng (M7 vs. M2 OLS)**"
            )
            st.dataframe(pairs_df, width="stretch", hide_index=True)
        with tb2:
            st.markdown(
                "**Bảng 5B — 2 Gói Độc lập (`WAV` & `Lux`) + Dải tham chiếu**"
            )
            st.dataframe(
                pred_res["standalone_df"], width="stretch", hide_index=True
            )
            st.caption(
                "**Lưu ý `ASSUMPTION`:** Dải tham chiếu `±MPIW/2` được lấy "
                "từ bề rộng trung bình cấp phân khúc "
                "trong `stage6_prediction_interval.csv` (bộ 3 mô hình "
                "Quantile Monotonic Split-CQR sẽ được đóng gói ở Pha 1 "
                "triển khai)."
            )

    # =========================================================================
    # PHẦN B: SURGE CAP POLICY SIMULATOR (RQ4 — Q_realized = min(Q_D, Q_S))
    # =========================================================================
    st.divider()
    st.markdown(
        '<div class="section-heading">Phần B — Mô phỏng Chính sách Trần '
        "Surge Cap dưới Ràng buộc Cung (RQ4 — SCENARIO)</div>",
        unsafe_allow_html=True,
    )
    sim_df = tables["surge_cap_simulation"]
    sim_meta = summaries["stage6"]["part6_surge_cap_policy_simulation"][
        "metadata"
    ]
    n_lyft_obs = int(sim_meta.get("n_total_lyft_quotes", 92_384))
    base_proxy = float(
        sim_meta.get("total_lyft_baseline_quote_proxy_usd", 1_603_696.20)
    )

    st.warning(
        "📌 **NHÃN MÔ PHỎNG HỌC THUẬT (`SCENARIO / SIMULATION` — Phân phối "
        "bên ngắn `Q_realized = min(Q_D, Q_S)`):** "
        "Toàn bộ kết quả dưới đây là **mô phỏng phản thực tế có điều "
        f"kiện** trên `{n_lyft_obs:,}` lượt báo giá Lyft quan sát surge "
        "thực "
        f"(Tổng cơ sở = **${base_proxy:,.2f} API Quote Proxy**, trong đó `6,"
        "280` lượt có `surge > 1.0x`, tách riêng 38,037 lượt khuyết "
        "surge). "
        "**KHÔNG** đồng nhất `API Quote Proxy` với doanh thu kế toán; khi "
        "cung nhạy giá (`ε_S = 0.30` hoặc `0.60`; `Q_S = clip(1 + ε_S·Δp/p)`),"
        " cầu dư `Σ max(0, Q_D − Q_realized)` trở thành lượt chưa được phục "
        "vụ và biến động Proxy giảm đơn điệu chặt."
    )

    sc_col1, sc_col2 = st.columns([1.15, 1.35], gap="medium")
    with sc_col1:
        cap_options = [3.0, 2.0, 1.75, 1.5, 1.25, 1.0]
        cap_labels = {
            3.0: "3.00x — Hiện trạng (Không áp trần)",
            2.0: "2.00x — Pha 1 Bảo thủ (Đạt τ ≤ 0.10% mọi cung ★)",
            1.75: "1.75x — Pha 2 Triggered Switchback (τ = 0.30%–0.50%)",
            1.5: "1.50x — Kịch bản C (Bảo vệ giờ cao điểm)",
            1.25: "1.25x — Kịch bản D (Trần siết chặt mạnh)",
            1.0: "1.00x — Kịch bản E (Xóa bỏ Surge — Đối chứng)",
        }
        selected_cap = st.select_slider(
            "1. Chọn ngưỡng trần Surge Cap mô phỏng (Lyft):",
            options=cap_options,
            value=2.0,
            format_func=lambda v: cap_labels[v],
            key="tab5_surge_cap",
        )
    with sc_col2:
        eps_options = {
            "static_000": "Cầu tĩnh (ε = 0.00 — Cung không đổi)",
            "eps_050": "Cận trên dư cung Cohen et al., 2016 (ε = -0.50)",
            "eps_080": "Cận trên dư cung cạnh tranh cao (ε = -0.80)",
        }
        elasticity_mode = st.radio(
            "2. Chế độ cầu khi cung không đổi (khi bật cung nhạy giá "
            "ε_S ∈ {0.30, 0.60}, Q_S tự động chặn cầu dư):",
            options=list(eps_options.keys()),
            format_func=lambda k: eps_options[k],
            index=0,
            key="tab5_elasticity",
        )

    supply_drop_pct = st.radio(
        "3. Ràng buộc phía cung tài xế nhạy giá tại đỉnh Surge — Nguyên tắc "
        "bên ngắn Q_realized = min(Q_D, Q_S) (khớp Bảng 5.1 Tiểu luận):",
        options=[0, 5, 10],
        format_func=lambda v: (
            "ε_S = 0.00 — Cung tài xế không đổi (Cơ sở / Cận trên)"
            if v == 0
            else (
                "ε_S = 0.30 — Cung nhạy giá vừa "
                "(Q_S = clip(1 + 0.30·Δp/p, 0.50, 1.0))"
                if v == 5
                else (
                    "ε_S = 0.60 — Cung nhạy giá mạnh "
                    "(Q_S = clip(1 + 0.60·Δp/p, 0.40, 1.0))"
                )
            )
        ),
        horizontal=True,
        index=0,
        key="tab5_supply_drop",
    )

    scen = extract_surge_cap_scenario(
        sim_df,
        surge_cap=float(selected_cap),
        elasticity_mode=elasticity_mode,
        supply_drop_pct=int(supply_drop_pct),
    )

    if scen["tree_saturation_warning"]:
        st.info(f"🔎 **{scen['tree_saturation_warning']}**")

    proxy_sub = (
        f"Dư cầu chưa phục vụ: {scen['unfulfilled_quotes_est']:.1f} lượt | "
        f"{scen['elasticity_label']}"
        if supply_drop_pct > 0
        else (
            f"Giữ chân cận trên: +{scen['retained_quotes_est']:.1f} lượt | "
            f"{scen['elasticity_label']}"
        )
    )

    render_kpi_row(
        [
            {
                "label": "Lượt Báo giá Chịu tác động",
                "value": f"{scen['affected_quotes_count']:,}",
                "subtitle": (
                    f"{scen['affected_surge_quotes_pct']:.2f}% lượt Surge | "
                    f"{scen['affected_lyft_quotes_pct']:.2f}% Lyft thực"
                ),
                "accent": "accent",
                "badge_text": "SCENARIO • Lyft Observed",
                "badge_class": "badge-simulation",
            },
            {
                "label": "Tiết kiệm TB/Lượt Bị sốc (M7)",
                "value": f"${scen['mean_saving_m7_usd']:.2f}",
                "subtitle": (
                    f"Giảm {scen['mean_saving_m7_pct']:.2f}% | Đối chứng M2 "
                    f"OLS: ${scen['mean_saving_m2_usd']:.2f}/lượt"
                ),
                "accent": "emerald",
                "badge_text": (
                    f"Gốc ${scen['mean_orig_quote_usd']:.2f} → "
                    f"${scen['mean_capped_quote_usd']:.2f}"
                ),
                "badge_class": "badge-simulation",
            },
            {
                "label": "Tổng Tiết kiệm Báo giá (M7)",
                "value": f"${scen['total_savings_m7_usd']:,.2f}",
                "subtitle": (
                    "Đối chứng M2 OLS: "
                    f"${scen['total_savings_m2_usd']:,.2f} (trên "
                    f"{n_lyft_obs:,} lượt thực)"
                ),
                "accent": "lyft",
                "badge_text": "SIMULATION",
                "badge_class": "badge-simulation",
            },
            {
                "label": "Biến động Lyft API Quote Proxy",
                "value": f"{scen['quote_proxy_change_pct']:+.3f}%",
                "subtitle": proxy_sub,
                "accent": "amber",
                "badge_text": "KHÔNG phải doanh thu thực",
                "badge_class": "badge-assumption",
            },
        ]
    )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="medium")
    with p1:
        fig_tradeoff = build_surge_cap_tradeoff_fig(
            sim_df,
            selected_cap=float(selected_cap),
            elasticity_mode=elasticity_mode,
            supply_drop_pct=int(supply_drop_pct),
        )
        st.plotly_chart(
            fig_tradeoff, width="stretch", config=PLOTLY_DISPLAY_CONFIG
        )
    with p2:
        fig_cross_m = build_surge_cap_m7_vs_m2_fig(sim_df)
        st.plotly_chart(
            fig_cross_m, width="stretch", config=PLOTLY_DISPLAY_CONFIG
        )

    with st.expander(
        "📋 Xem Bảng đầy đủ 6 Kịch bản Mô phỏng Surge Cap (Gồm 2 cột ràng "
        "buộc cung Q = min(Q_D, Q_S))",
        expanded=False,
    ):
        st.dataframe(
            prepare_surge_cap_display_table(sim_df),
            width="stretch",
            hide_index=True,
        )

    # Hiển thị công khai 3 giả định & giới hạn kinh tế học từ
    # stage6_summary.json
    assumptions = sim_meta.get("economic_assumptions_and_limitations", [])
    if assumptions:
        st.markdown(
            "**⚖️ 3 Giả định & Giới hạn Kinh tế học Bắt buộc của Mô phỏng "
            "(`ASSUMPTION / LIMITATION`):**"
        )
        for item in assumptions:
            st.markdown(f"- {item}")

    c200 = sim_df.loc[sim_df["surge_cap"] == 2.00].iloc[0]
    c175 = sim_df.loc[sim_df["surge_cap"] == 1.75].iloc[0]
    sw_top5 = (
        (v5 or {})
        .get("policy_rq4_supply_power_and_io", {})
        .get("switchback_power_analysis", {})
        .get("scopes", {})
        .get("top5_hotspot_districts", {})
        .get("cap_1.75x", {})
    )
    sw_mde = {float(m["mde_pp"]): m for m in sw_top5.get("mde_scenarios", [])}
    sw_days_3pp = float(
        sw_mde.get(3.0, {}).get("days_needed_at_prod_traffic_25x", 38.5)
    )
    sw_days_2pp = float(
        sw_mde.get(2.0, {}).get("days_needed_at_prod_traffic_25x", 86.6)
    )
    sw_deff_25x = float(sw_top5.get("deff_25x", 2.648))
    v5_cross = (v5 or {}).get("crossing_distances_5_pairs", [])
    ux_sa = (
        float(v5_cross[1]["spec_a_train80"]["hc3"]["d_star"])
        if len(v5_cross) >= 2
        else 3.21
    )
    ux_sb = (
        float(v5_cross[1]["spec_b_train80_nonsurge"]["hc3"]["d_star"])
        if len(v5_cross) >= 2
        else 5.17
    )
    st.markdown(
        '<div class="section-heading">Hàm ý Quản trị hai nấc &amp; '
        "Triggered Switchback (RQ4)</div>",
        unsafe_allow_html=True,
    )
    r1, r2 = st.columns(2, gap="medium")
    with r1:
        render_insight_card(
            title=(
                "Hàm ý 5.1 — Lộ trình 2 nấc: Pha 1 Trần bảo thủ 2.00x (τ ≤ "
                "0.10%), Pha 2 Triggered Switchback 1.75x"
            ),
            finding=(
                "Dưới nguyên tắc phân phối bên ngắn Q_realized = min(Q_D, "
                "Q_S) trên 92,384 lượt Lyft quan sát thực, CHỈ duy nhất "
                "trần 2.00x đáp ứng ngưỡng bảo thủ τ ≤ 0.10% ở mọi kịch bản "
                "cung; trần 1.75x phù hợp làm ứng viên thử nghiệm Pha 2 khi "
                "chấp nhận biên rủi ro mở rộng τ = 0.30%–0.50%."
            ),
            evidence=(
                "Trần 2.00x (53 lượt, tiết kiệm M7 "
                "$"
                f"{float(c200['mean_rider_saving_per_affected_trip_usd']):.2f}"
                f" / -{float(c200['mean_rider_saving_pct_affected']):.2f}%, "
                f"M2 ${float(c200['m2_ols_mean_saving_affected_usd']):.2f}):"
                " "
                "Proxy tĩnh "
                f"{float(c200['static_lyft_rev_change_pct']):+.3f}%, cung "
                "ε_S=0.30 "
                f"{float(c200['supply_constrained_rev_change_drop5_pct']):+.3f}"
                "%, cung ε_S=0.60 "
                f"{float(c200['supply_constrained_rev_change_drop10_pct']):+.3f}"
                "%. "
                "Trần 1.75x (704 lượt, tiết kiệm "
                "$"
                f"{float(c175['mean_rider_saving_per_affected_trip_usd']):.2f}"
                f" / -{float(c175['mean_rider_saving_pct_affected']):.2f}%):"
                " "
                "Proxy tĩnh "
                f"{float(c175['static_lyft_rev_change_pct']):+.3f}%, cung "
                "ε_S=0.30 "
                f"{float(c175['supply_constrained_rev_change_drop5_pct']):+.3f}"
                "%, cung ε_S=0.60 "
                f"{float(c175['supply_constrained_rev_change_drop10_pct']):+.3f}"
                "%."
            ),
            method=(
                "Mô phỏng phản thực tế đồng thời M7 & M2 dưới Q_realized = "
                "min(Q_D, Q_S) kết hợp Triggered Switchback Power Analysis "
                "(không dùng ITT vì bị pha loãng xuống 0.0152 pp)."
            ),
            business_meaning=(
                "Áp trần 2.00x ở Pha 1 theo dõi an toàn ETA, sau đó chạy "
                "Triggered Switchback A/B "
                "cho trần 1.75x tại 5 quận điểm nóng (Back Bay, Fenway, "
                "Northeastern University, "
                "Theatre District, Boston University — cần "
                f"{sw_days_3pp:.1f}–{sw_days_2pp:.1f} ngày "
                f"ở lưu lượng sản xuất 25× với DEFF_25× = {sw_deff_25x:.3f} "
                "cho MDE 3.0–2.0 pp)."
            ),
            limitation=(
                "KHÔNG phải ước lượng nhân quả thực địa: mô phỏng phụ thuộc "
                "giả định mức mất cung tài xế và dùng API Quote Proxy thay "
                "vì dữ liệu chuyến hoàn thành thực tế. Dừng sớm ngay nếu "
                "ETA tăng > 15% hoặc tỷ lệ hủy thiếu xe tăng > 2 pp."
            ),
            accent="amber",
            tag_text="SCENARIO / SIMULATION",
            tag_class="badge-simulation",
        )
    with r2:
        render_insight_card(
            title=(
                "Hàm ý 5.2 — Quy tắc Lựa chọn Nền tảng theo Cự ly & Trạng "
                "thái Surge cho Hành khách"
            ),
            finding=(
                "Kết hợp Tab 3 (song đặc tả A/B) và Tab 5A (dự báo trực "
                "tiếp M7/M2), chiến lược tiết kiệm chi phí hiệu quả nhất "
                "phụ thuộc vào cự ly chuyến đi, phân khúc xe và trạng thái "
                "Surge của Lyft."
            ),
            evidence=(
                "Khi không surge (1.0x): chọn Lyft Shared cho đi ghép và "
                "Lyft XL cho hạng 6 chỗ trên toàn miền Boston; chọn Lyft "
                f"cho chặng phổ thông ngắn (< {ux_sa:.2f}–{ux_sb:.2f} mi) và "
                "UberX cho "
                "chặng dài; chọn Uber Black/Black SUV ở hạng sang. Ngay khi "
                "Lyft bật surge ≥ 1.25x, giá báo Lyft thường vượt UberX "
                "ngay trên chặng ngắn."
            ),
            method=(
                "Suy luận đồng thời 12 gói bằng M7 HistGBR & M2 OLS trên 27 "
                "đặc trưng hợp lệ trong miền Boston [0.02, 7.86] dặm."
            ),
            business_meaning=(
                "Cơ sở xây dựng công cụ so sánh giá báo thông minh (Fare "
                "Aggregator) cho người dùng cuối mà chỉ cần đầu vào lộ "
                "trình + hệ số surge hiện hành."
            ),
            limitation=(
                "Chưa phản ánh biến động phụ phí ẩn tức thời của Uber "
                "(thiếu biến duration), chưa tính mã khuyến mãi cá nhân và "
                "không suy rộng ra ngoài 72 tuyến nội đô Boston."
            ),
            accent="emerald",
            tag_text="VERIFIED (IN-DOMAIN)",
            tag_class="badge-verified",
        )
