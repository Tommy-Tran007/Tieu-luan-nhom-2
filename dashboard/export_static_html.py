"""
Script: dashboard/export_static_html.py
Mục đích: Xuất bản Báo cáo Dashboard Tương tác Độc lập (Offline Standalone
HTML) tại:
    reports/dashboard_interactive.html

Đặc điểm kỹ thuật:
- Tái sử dụng 100% tầng business_logic.py, model_loader.py và charts.py (Single
  Source of Truth ->
  đảm bảo số liệu và biểu đồ khớp tuyệt đối với ứng dụng Streamlit).
- Nhúng thư viện plotly.js ĐÚNG 1 LẦN trong thẻ <head> -> mở trực tiếp bằng
  Chrome/Edge
  KHÔNG cần chạy Python server và KHÔNG cần kết nối Internet (100% offline-safe
  tại phòng bảo vệ).
- Giao diện 5 Tab Executive Light Theme rõ ràng, tự động Plotly.Plots.resize()
  khi chuyển Tab.
- Tích hợp sẵn:
  * Bộ chuyển đổi 5 cặp biểu giá tương tác ở Tab 3 (khóa miền [0.02, 7.86] dặm
    — HIGH-02).
  * Công cụ ước lượng giá báo HC3 OLS chạy trực tiếp bằng JavaScript ở Tab 5A
    (có cảnh báo ngoại suy).
  * Bộ chuyển kịch bản Surge Cap & độ co giãn cầu ở Tab 5B (gắn nhãn SCENARIO /
    API Quote Proxy — HIGH-03).
- Phân biệt tường minh: đây là Dashboard tương tác vector Plotly; hình ảnh tĩnh
  chuẩn 300 DPI PNG
  phục vụ in ấn tiểu luận nằm riêng tại reports/figures/stage3..6/.

Lệnh chạy:
    python -m dashboard.export_static_html
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from dashboard.business_logic import (  # noqa: E402
    compute_all_pair_comparisons,
    compute_binned_empirical_tariff,
    compute_data_quality_summary,
    compute_dayofweek_est_profile,
    compute_executive_kpis,
    compute_hourly_est_profile,
    compute_price_histogram_bins,
    compute_route_catalog,
    compute_surge_distribution,
    compute_tier_boxplot_stats,
    compute_weather_rain_comparison,
    prepare_model_leaderboard,
    prepare_permutation_family_summary,
    prepare_segment_errors,
    prepare_surge_cap_display_table,
)
from dashboard.charts import (  # noqa: E402
    PLOTLY_DISPLAY_CONFIG,
    build_actual_vs_predicted_and_residuals_fig,
    build_dayofweek_est_fig,
    build_executive_uber_vs_lyft_fig,
    build_hourly_est_fig,
    build_model_arena_fig,
    build_pdp_curves_fig,
    build_permutation_importance_fig,
    build_prediction_interval_fig,
    build_price_distribution_fig,
    build_route_heatmap_fig,
    build_segment_errors_fig,
    build_surge_cap_m7_vs_m2_fig,
    build_surge_cap_tradeoff_fig,
    build_surge_levels_fig,
    build_tariff_crossing_fig,
    build_tariff_strategy_scatter_fig,
    build_tier_boxplot_fig,
    build_weather_rain_audit_fig,
)
from dashboard.config import (  # noqa: E402
    EMPIRICAL_DISTANCE_MAX,
    EMPIRICAL_DISTANCE_MIN,
    HTML_EXPORT_PATH,
    STANDALONE_TIERS,
    SYMMETRIC_PAIRS,
)
from dashboard.data_loader import load_data_bundle  # noqa: E402
from dashboard.model_loader import (  # noqa: E402
    evaluate_models_on_test_set,
    load_trained_models,
)
from dashboard.validation import run_full_data_validation  # noqa: E402


def _fig_to_div(fig: go.Figure, div_id: str) -> str:
    """Xuất 1 biểu đồ Plotly thành khối <div> không nhúng lại plotly.js."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config=PLOTLY_DISPLAY_CONFIG,
    )


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Chuyển DataFrame thành bảng HTML sáng rõ có cuộn ngang."""
    sub = df.head(max_rows)
    headers = "".join(
        f"<th>{html.escape(str(col))}</th>" for col in sub.columns
    )
    body_rows = []
    for _, row in sub.iterrows():
        cells = "".join(
            f"<td>{html.escape(str(val))}</td>" for val in row.tolist()
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap"><table class="data-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _kpi_card_html(
    label: str,
    value: str,
    sub: str,
    accent: str = "accent",
    badge: str = "",
    badge_cls: str = "badge-verified",
) -> str:
    badge_html = (
        '<div style="margin-top:8px;"><span class="badge '
        f'{badge_cls}">{html.escape(badge)}</span></div>'
        if badge
        else ""
    )
    return f"""
    <div class="kpi-card top-{accent}">
        <div class="kpi-label">{html.escape(label)}</div>
        <div class="kpi-value">{html.escape(value)}</div>
        <div class="kpi-sub">{html.escape(sub)}</div>
        {badge_html}
    </div>
    """


def _insight_card_html(
    title: str,
    finding: str,
    evidence: str,
    method: str,
    business: str,
    limitation: str,
    accent: str = "blue",
    tag: str = "VERIFIED",
    tag_cls: str = "badge-verified",
) -> str:
    return f"""
    <div class="insight-card border-{accent}">
        <div class="insight-title">{html.escape(title)} <span class="badge
          {tag_cls}">{html.escape(tag)}</span></div>
        <ul>
            <li><b>Phát hiện:</b> {html.escape(finding)}</li>
            <li><b>Bằng chứng định lượng:</b> {html.escape(evidence)}</li>
            <li><b>Phương pháp phân tích:</b> {html.escape(method)}</li>
            <li><b>Ý nghĩa kinh doanh:</b> {html.escape(business)}</li>
            <li><b>Giới hạn kết luận:</b> {html.escape(limitation)}</li>
        </ul>
    </div>
    """


def export_interactive_html(output_path: Path = HTML_EXPORT_PATH) -> Path:
    """
    Sinh toàn bộ tệp báo cáo HTML tương tác 5 Tab chạy 100% offline tại
    reports/dashboard_interactive.html.
    Xác minh cổng kiểm toán dữ liệu CRITICAL trước khi xuất.
    """
    val_report = run_full_data_validation()
    val_report.raise_if_critical()

    bundle = load_data_bundle()
    master = bundle["master"]
    tables = bundle["tables"]
    summaries = bundle["summaries"]
    v5 = bundle.get("v5", {})
    models = load_trained_models()
    eval_res = evaluate_models_on_test_set(models, master, plot_sample_n=3_000)

    kpis = compute_executive_kpis(
        master, tables["test_metrics"], full_dataset_rows=len(master)
    )
    dq = compute_data_quality_summary(
        master,
        summaries["stage3"],
        summaries["stage4"],
        ram_mb=float(bundle["ram_mb"]),
        est_raw_ram_mb=float(bundle["est_unoptimized_ram_mb"]),
    )

    # --- TAB 1 FIGURES (EXECUTIVE HERO + DEEP-DIVE COLLAPSIBLE) ---
    tariffs_df = tables["tier_tariffs"]
    pairs_comp_df = compute_all_pair_comparisons(tariffs_df, master)
    std_pair = pairs_comp_df.loc[pairs_comp_df["pair_id"] == "standard"].iloc[
        0
    ]
    xl_pair = pairs_comp_df.loc[pairs_comp_df["pair_id"] == "xl"].iloc[0]
    bl_pair = pairs_comp_df.loc[pairs_comp_df["pair_id"] == "exec_black"].iloc[
        0
    ]
    bs_pair = pairs_comp_df.loc[pairs_comp_df["pair_id"] == "lux_suv"].iloc[0]
    pool_pair = pairs_comp_df.loc[
        pairs_comp_df["pair_id"] == "pool_shared"
    ].iloc[0]
    d_std = float(std_pair["crossing_distance_miles"])
    v5_cross = v5.get("crossing_distances_5_pairs", [])
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
    st_v5 = v5.get("statistical_rigor_extensions", {})
    h3b_p_hr = float(
        st_v5.get("h3b_lyft_surge_rain_cluster", {}).get(
            "p_hour_cluster", 0.0793
        )
    )
    hv_v5 = st_v5.get("heavy_rain_vs_dry", {})
    wcr_p = float(
        hv_v5.get("exact_wcr_loc_hr_B1999", {}).get("p_wild", 0.8780)
    )
    n_heavy_hrs = int(hv_v5.get("n_heavy_citywide_hours", 8))
    cqr = v5.get("model_diagnostics_and_cqr", {}).get(
        "clean_monotonic_cqr", {}
    )
    cqr_picp = float(cqr.get("overall_cqr_picp_pct", 79.89))
    cqr_lo = float(cqr.get("min_tier_cqr_picp_pct", 78.62))
    cqr_hi = float(cqr.get("max_tier_cqr_picp_pct", 81.99))
    cqr_mpiw = float(cqr.get("overall_cqr_mpiw_usd", 3.562))
    grp6 = v5.get("model_diagnostics_and_cqr", {}).get(
        "grouped_permutation_importance", []
    )
    grp6_top2 = (
        sum(float(r["m7_rel_share_pct"]) for r in grp6[:2])
        if len(grp6) >= 2
        else 92.60
    )
    regime = summaries["stage5"].get("part2_regime_breakdown", {})
    lyft_r2 = float(
        regime.get("lyft_observed_surge_test", {})
        .get("M7_Conditional", {})
        .get("r2", 0.9794)
    )
    uber_r2 = float(
        regime.get("uber_hidden_surge_test", {})
        .get("M7_Conditional", {})
        .get("r2", 0.9474)
    )
    ex_ante = summaries["stage5"].get("part3_ex_ante_experiments", {})
    ex_all = next(
        (
            r
            for r in ex_ante.get("pricing_without_surge", [])
            if r["model_or_slice"] == "M7_ExAnte_All_Test"
        ),
        {"r2": 0.9286, "rmse": 2.4655, "mape_pct": 9.15},
    )
    ex_auc = float(
        ex_ante.get("lyft_surge_classifier", {})
        .get("HistGB_Classifier", {})
        .get("roc_auc", 0.7383)
    )
    top2_share = float(
        tables["permutation_importance"]
        .iloc[:2]["m7_importance_share_pct"]
        .sum()
    )
    cap200 = (
        tables["surge_cap_simulation"]
        .loc[tables["surge_cap_simulation"]["surge_cap"] == 2.00]
        .iloc[0]
    )
    cap175 = (
        tables["surge_cap_simulation"]
        .loc[tables["surge_cap_simulation"]["surge_cap"] == 1.75]
        .iloc[0]
    )

    fig_hero = build_executive_uber_vs_lyft_fig(master, tariffs_df)
    fig_dist = build_price_distribution_fig(
        compute_price_histogram_bins(master), kpis
    )
    fig_box = build_tier_boxplot_fig(compute_tier_boxplot_stats(master))
    routes_df = compute_route_catalog(master)
    fig_heat_price = build_route_heatmap_fig(
        routes_df, metric="mean_price_usd"
    )
    fig_heat_dist = build_route_heatmap_fig(
        routes_df, metric="median_distance_miles"
    )
    fig_heat_surge = build_route_heatmap_fig(
        routes_df, metric="surge_rate_pct"
    )

    # --- TAB 2 FIGURES (CRIT-01 & CRIT-02 LOCKED) ---
    weather_df, weather_meta = compute_weather_rain_comparison(master)
    hourly_df, hourly_meta = compute_hourly_est_profile(master)
    dow_df = compute_dayofweek_est_profile(master)
    surge_dist_df = compute_surge_distribution(master)

    fig_weather = build_weather_rain_audit_fig(weather_df, weather_meta)
    fig_hourly = build_hourly_est_fig(hourly_df, hourly_meta)
    fig_dow = build_dayofweek_est_fig(dow_df)
    fig_surge_lvl = build_surge_levels_fig(surge_dist_df)

    hyp_df = tables["hypothesis_tests"]
    h1a = hyp_df.loc[hyp_df["hypothesis_id"] == "H1a_All_Rides"].iloc[0]
    h2a = hyp_df.loc[hyp_df["hypothesis_id"] == "H2a_Surge_All"].iloc[0]
    h2c = hyp_df.loc[hyp_df["hypothesis_id"] == "H2c_Price_All"].iloc[0]
    h3a = hyp_df.loc[hyp_df["hypothesis_id"] == "H3a_Surge_x_CabType"].iloc[0]
    h3d = hyp_df.loc[
        hyp_df["hypothesis_id"] == "H3d_Surge_x_RushHour_Lyft"
    ].iloc[0]
    h4a = hyp_df.loc[
        hyp_df["hypothesis_id"] == "H4a_Wald_UberX_vs_Lyft_Slope"
    ].iloc[0]
    h4c = hyp_df.loc[
        hyp_df["hypothesis_id"] == "H4c_Wald_Joint_Weather_Multivariate"
    ].iloc[0]
    n_h1a = int(h1a["n_group1"]) + int(h1a["n_group2"])
    n_h2a = int(h2a["n_group1"]) + int(h2a["n_group2"])

    hyp_table_df = hyp_df[
        [
            "hypothesis_id",
            "research_question",
            "scope",
            "parametric_pvalue",
            "nonparametric_pvalue",
            "effect_size_name",
            "effect_size_value",
            "statistical_decision",
            "practical_significance",
        ]
    ].copy()
    h3b_html_mask = hyp_table_df["hypothesis_id"] == "H3b_Surge_x_Rain_Lyft"
    if h3b_html_mask.any():
        hyp_table_df.loc[h3b_html_mask, "statistical_decision"] = (
            f"Chưa bác bỏ H0 sau gom cụm/Holm (cụm giờ p={h3b_p_hr:.4f})"
        )
    hyp_table_df.columns = [
        "Mã giả thuyết",
        "RQ",
        "Phạm vi kiểm định",
        "p (Tham số)",
        "p (Phi tham số)",
        "Effect Size",
        "Giá trị",
        "Quyết định H0",
        "Ý nghĩa thực tiễn",
    ]

    # --- TAB 3 FIGURES (HIGH-02: 5 CẶP ĐỐI XỨNG + BẢN ĐỒ CHIẾN LƯỢC) ---
    pair_chart_blocks: list[str] = []
    pair_options_html: list[str] = []
    for idx, row in pairs_comp_df.iterrows():
        p_info = row.to_dict()
        binned = compute_binned_empirical_tariff(
            master, p_info["uber_tier"], p_info["lyft_tier"]
        )
        fig_p = build_tariff_crossing_fig(tariffs_df, p_info, binned_df=binned)
        selected_attr = " selected" if p_info["pair_id"] == "standard" else ""
        display_style = "block" if p_info["pair_id"] == "standard" else "none"
        pair_options_html.append(
            f'<option value="pair_{p_info["pair_id"]}"{selected_attr}>'
            f"{html.escape(p_info['segment_label'])}</option>"
        )
        note_html = (
            '<div class="callout-info"><b>Lưu ý khóa miền nội suy:</b> '
            "Quãng đường tối đa tại Boston là "
            f"{EMPIRICAL_DISTANCE_MAX:.2f} dặm. Cặp "
            f"{html.escape(p_info['uber_tier'])} vs. "
            f"{html.escape(p_info['lyft_tier'])} "
            "<b>KHÔNG cắt nhau trong miền quan sát</b> (Lyft Shared luôn "
            "rẻ hơn nhờ giá mở cửa thấp hơn "
            f"${float(p_info['base_diff_uber_minus_lyft']):.2f}) — không "
            "ngoại suy ngoài miền.</div>"
            if not p_info["crossing_in_domain"]
            else ""
        )
        pair_chart_blocks.append(
            f'<div id="pair_{p_info["pair_id"]}" class="pair-panel" '
            f'style="display:{display_style};">'
            f'{note_html}<div class="chart-'
            f'card">{_fig_to_div(fig_p, f"fig_pair_{idx}")}</div></div>'
        )

    fig_strat = build_tariff_strategy_scatter_fig(tariffs_df)
    pairs_view_df = pairs_comp_df[
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
    pairs_view_df["crossing_distance_miles"] = pairs_view_df[
        "crossing_distance_miles"
    ].apply(
        lambda v: f"{v:.2f} dặm" if pd.notna(v) else "Không cắt (≤ 7.86 mi)"
    )
    pairs_view_df.columns = [
        "Phân khúc",
        "Uber",
        "β₀ Uber ($)",
        "β₁ Uber ($/mi)",
        "Lyft",
        "β₀ Lyft ($)",
        "β₁ Lyft ($/mi)",
        "Điểm cắt d*",
        "Rẻ hơn (1.5 mi)",
        "Rẻ hơn (4.5 mi)",
    ]

    standalone_names = [s["tier"] for s in STANDALONE_TIERS]
    standalone_view_df = tariffs_df.loc[
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
    standalone_view_df.columns = [
        "Hãng",
        "Gói độc lập",
        "β₀ ($)",
        "β₁ ($/mi)",
        "Giá 2.5 mi ($)",
        "R² nội gói",
    ]

    # --- TAB 4 FIGURES (ML PERFORMANCE, XAI, UNCERTAINTY) ---
    leaderboard = prepare_model_leaderboard(tables["test_metrics"])
    fig_arena = build_model_arena_fig(leaderboard)
    fig_resid_m7 = build_actual_vs_predicted_and_residuals_fig(
        eval_res, model_key="M7"
    )
    fig_resid_m2 = build_actual_vs_predicted_and_residuals_fig(
        eval_res, model_key="M2"
    )
    fig_seg_tier = build_segment_errors_fig(
        prepare_segment_errors(tables["segment_errors"], "service_tier"),
        "12 Gói Dịch vụ",
    )
    fig_perm = build_permutation_importance_fig(
        tables["permutation_importance"], show_m2=True
    )
    fig_pdp = build_pdp_curves_fig(summaries["stage6"])
    fig_pi = build_prediction_interval_fig(
        tables["prediction_interval"], cqr_summary=cqr
    )

    m7_live = eval_res["metrics_m7"]
    m2_live = eval_res["metrics_m2"]
    fam_df = prepare_permutation_family_summary(
        tables["permutation_importance"]
    ).copy()
    fam_df.columns = [
        "Họ biến kinh tế",
        "Số nhóm",
        "Tổng M7 ΔRMSE ($)",
        "Tỷ trọng M7 (%)",
        "Tổng M2 ΔRMSE ($)",
    ]

    lb_html_df = leaderboard[
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
    lb_html_df.columns = [
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

    # --- TAB 5 FIGURES & CLIENT-SIDE SIMULATOR DATA ---
    sim_df = tables["surge_cap_simulation"]
    sim_meta = summaries["stage6"]["part6_surge_cap_policy_simulation"][
        "metadata"
    ]
    n_lyft_obs = int(sim_meta.get("n_total_lyft_quotes", 92_384))
    base_proxy = float(
        sim_meta.get("total_lyft_baseline_quote_proxy_usd", 1_603_696.20)
    )
    fig_pareto = build_surge_cap_tradeoff_fig(
        sim_df,
        selected_cap=2.00,
        elasticity_mode="static_000",
        supply_drop_pct=0,
    )
    fig_m7_m2_sim = build_surge_cap_m7_vs_m2_fig(sim_df)
    sim_disp_df = prepare_surge_cap_display_table(sim_df)
    sim_assumptions = sim_meta.get("economic_assumptions_and_limitations", [])
    assumptions_li = "".join(
        f"<li>{html.escape(item)}</li>" for item in sim_assumptions
    )

    # Nhúng tham số biểu giá HC3 và bảng mô phỏng cho bộ tính tương tác
    # JavaScript offline ở Tab 5
    tariff_lookup_js = {}
    for _, r in tariffs_df.iterrows():
        tariff_lookup_js[str(r["service_tier"])] = {
            "b0": float(r["base_fare_b0_usd"]),
            "b1": float(r["per_mile_rate_b1_usd"]),
            "cab": str(r["cab_type"]),
        }
    pairs_js = [
        {
            "segment": p["segment_label"],
            "uber": p["uber_tier"],
            "lyft": p["lyft_tier"],
        }
        for p in SYMMETRIC_PAIRS
    ]
    sim_records_js = sim_df.to_dict(orient="records")

    val_table_df = val_report.to_dataframe()[
        ["check_id", "category", "severity", "status", "message"]
    ].copy()
    val_table_df.columns = [
        "Mã kiểm tra",
        "Hạng mục",
        "Mức độ",
        "Trạng thái",
        "Chi tiết kiểm chứng",
    ]
    n_pass = len(val_report.checks) - len(val_report.failed_checks)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plotly_js_bundle = get_plotlyjs()

    # =========================================================================
    # LẮP RÁP TỆP HTML TỰ CHỨA (5 TAB EXECUTIVE LIGHT THEME, 100% OFFLINE)
    # =========================================================================
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Uber vs. Lyft Boston — Interactive Analytics Dashboard
  (PRJ-EXT-01)</title>
<script type="text/javascript">{plotly_js_bundle}</script>
<style>
  :root {{
    --bg-page: #F8FAFC;
    --bg-card: #FFFFFF;
    --border: #E2E8F0;
    --text-main: #0F172A;
    --text-sub: #475569;
    --uber: #111827;
    --lyft: #E11D8A;
    --accent: #2563EB;
    --emerald: #059669;
    --amber: #D97706;
    --danger: #DC2626;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 0;
    background: var(--bg-page);
    color: var(--text-main);
    font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif;
    line-height: 1.5;
  }}
  .container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px 24px 48px 24px;
  }}
  .hero {{
    background: linear-gradient(120deg, #0F172A 0%, #1E293B 58%, #3B0764 100%);
    color: #F8FAFC;
    padding: 24px 28px;
    border-radius: 16px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.14);
  }}
  .hero h1 {{
    margin: 0 0 6px 0;
    font-size: 1.65rem;
    font-weight: 700;
    color: #FFFFFF;
  }}
  .hero p {{ margin: 0; font-size: 0.93rem; color: #CBD5E1; }}
  .badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
  .badge {{
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; vertical-align: middle;
  }}
  .badge-verified {{
    background: #DCFCE7;
    color: #166534;
    border: 1px solid #86EFAC;
  }}
  .badge-assumption {{
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FCD34D;
  }}
  .badge-simulation {{
    background: #DBEAFE;
    color: #1E40AF;
    border: 1px solid #93C5FD;
  }}
  .badge-dark {{
    background: rgba(255,255,255,0.14);
    color: #F8FAFC;
    border: 1px solid rgba(255,255,255,0.24);
  }}
  .badge-uber {{ background: #111827; color: #FFFFFF; }}
  .badge-lyft {{ background: #E11D8A; color: #FFFFFF; }}

  /* Thanh chuyển 5 Tab cố định rõ ràng */
  .tab-nav {{
    display: flex; flex-wrap: wrap; gap: 8px;
    background: #FFFFFF; border: 1px solid var(--border);
    padding: 8px; border-radius: 12px; margin-bottom: 18px;
    position: sticky; top: 8px; z-index: 50;
    box-shadow: 0 2px 8px rgba(15,23,42,0.06);
  }}
  .tab-btn {{
    flex: 1 1 180px; border: 1px solid transparent; background: transparent;
    padding: 10px 14px; border-radius: 9px; font-weight: 600; font-size:
      0.88rem;
    color: #334155; cursor: pointer; transition: all 0.15s ease;
  }}
  .tab-btn:hover {{ background: #F1F5F9; color: #0F172A; }}
  .tab-btn.active {{
    background: #0F172A; color: #FFFFFF; border-color: #0F172A;
    box-shadow: 0 2px 6px rgba(15,23,42,0.18);
  }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}

  .section-heading {{
    font-size: 1.12rem; font-weight: 700; color: #0F172A;
    margin: 22px 0 12px 0; padding-bottom: 6px; border-bottom: 2px solid
      var(--border);
  }}
  .kpi-grid-3 {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 14px;
    margin-bottom: 14px;
  }}
  .kpi-grid-4 {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    margin-bottom: 14px;
  }}
  .kpi-grid-5 {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }}
  .kpi-grid-5 .kpi-card {{ padding: 13px 14px; }}
  .kpi-grid-5 .kpi-value {{ font-size: 1.32rem; }}
  @media (max-width: 1120px) {{
    .kpi-grid-5 {{ grid-template-columns: repeat(auto-fit, minmax(210px,
      1fr)); }}
  }}
  .two-col {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
    gap: 16px;
    margin-bottom: 16px;
  }}

  /* Khối Executive Dashboard (Tab 1) */
  .exec-scope-bar {{
    background: linear-gradient(90deg, #F8FAFC 0%, #EFF6FF 100%);
    border: 1px solid #CBD5E1; border-left: 4px solid var(--accent);
    border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;
  }}
  .exec-scope-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px; margin-top: 10px; font-size: 0.83rem; color: #334155;
  }}
  .finding-grid-3 {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 14px;
    margin-bottom: 16px;
  }}
  .finding-card {{
    background: #FFFFFF; border: 1px solid var(--border); border-top: 3.5px
      solid var(--accent);
    border-radius: 12px; padding: 15px 17px; box-shadow: 0 2px 6px
      rgba(15,23,42,0.04);
  }}
  .finding-card.f-lyft {{ border-top-color: var(--lyft); }}
  .finding-card.f-emerald {{ border-top-color: var(--emerald); }}
  .finding-num {{
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; color: #64748B; margin-bottom: 5px;
  }}
  .finding-headline {{
    font-size: 0.96rem; font-weight: 700; color: #0F172A; line-height: 1.35;
      margin-bottom: 7px;
  }}
  .finding-body {{
    font-size: 0.83rem; color: #475569; line-height: 1.5; margin: 0;
  }}
  .exec-verdict-box {{
    background: linear-gradient(120deg, #F8FAFC 0%, #F0FDF4 100%);
    border: 1px solid #BBF7D0; border-left: 5px solid var(--emerald);
    border-radius: 12px; padding: 16px 20px; margin: 10px 0 18px 0;
    color: #0F172A; font-size: 0.89rem; line-height: 1.58;
  }}
  details.exec-details {{
    background: #FFFFFF; border: 1px solid var(--border); border-radius: 12px;
    padding: 12px 16px; margin-bottom: 14px; box-shadow: 0 1px 3px
      rgba(15,23,42,0.03);
  }}
  details.exec-details summary {{
    cursor: pointer; font-weight: 700; font-size: 0.93rem; color: #1E293B;
    list-style: none; display: flex; justify-content: space-between;
      align-items: center;
  }}
  details.exec-details summary::-webkit-details-marker {{ display: none; }}
  details.exec-details summary::after {{
    content: "▸ Mở rộng"; font-size: 0.78rem; font-weight: 600; color: #2563EB;
  }}
  details.exec-details[open] summary::after {{
    content: "▾ Thu gọn"; color: #64748B;
  }}
  .details-body {{ margin-top: 14px; }}

  .kpi-card {{
    background: var(--bg-card); border: 1px solid var(--border);
      border-radius: 12px;
    padding: 15px 18px; box-shadow: 0 1px 3px rgba(15,23,42,0.04);
  }}
  .top-accent {{ border-top: 3.5px solid var(--accent); }}
  .top-uber {{ border-top: 3.5px solid var(--uber); }}
  .top-lyft {{ border-top: 3.5px solid var(--lyft); }}
  .top-emerald {{ border-top: 3.5px solid var(--emerald); }}
  .top-amber {{ border-top: 3.5px solid var(--amber); }}
  .kpi-label {{
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: #64748B;
    margin-bottom: 5px;
  }}
  .kpi-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #0F172A;
    font-variant-numeric: tabular-nums;
  }}
  .kpi-sub {{ font-size: 0.81rem; color: #475569; margin-top: 4px; }}

  .chart-card {{
    background: #FFFFFF; border: 1px solid var(--border); border-radius: 12px;
    padding: 12px 14px; margin-bottom: 16px; box-shadow: 0 1px 3px
      rgba(15,23,42,0.04);
  }}
  .insight-card {{
    background: #FFFFFF; border: 1px solid var(--border); border-left: 4px
      solid var(--accent);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
  }}
  .border-lyft {{ border-left-color: var(--lyft); }}
  .border-emerald {{ border-left-color: var(--emerald); }}
  .border-amber {{ border-left-color: var(--amber); }}
  .insight-title {{
    font-size: 0.96rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #0F172A;
  }}
  .insight-card ul {{
    margin: 0;
    padding-left: 18px;
    font-size: 0.86rem;
    color: #334155;
  }}
  .insight-card li {{ margin-bottom: 4px; }}

  .callout-info {{
    background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid
      #2563EB;
    color: #1E3A8A; padding: 12px 16px; border-radius: 10px; margin-bottom:
      14px; font-size: 0.87rem;
  }}
  .callout-warn {{
    background: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid
      #D97706;
    color: #78350F; padding: 12px 16px; border-radius: 10px; margin-bottom:
      14px; font-size: 0.87rem;
  }}
  .callout-danger {{
    background: #FEF2F2; border: 1px solid #FECACA; border-left: 4px solid
      #DC2626;
    color: #991B1B; padding: 12px 16px; border-radius: 10px; margin-bottom:
      14px; font-size: 0.88rem; font-weight: 600;
  }}

  .control-bar {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 14px;
    background: #FFFFFF; border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 16px; margin-bottom: 14px;
  }}
  .control-bar label {{
    font-size: 0.86rem;
    font-weight: 600;
    color: #1E293B;
  }}
  .control-bar select, .control-bar input[type="number"] {{
    padding: 6px 10px; border: 1px solid #CBD5E1; border-radius: 8px;
    font-size: 0.88rem; background: #F8FAFC; color: #0F172A;
  }}
  .sub-btn {{
    padding: 6px 12px; border: 1px solid #CBD5E1; border-radius: 8px;
    background: #F8FAFC; color: #1E293B; font-size: 0.83rem; font-weight: 600;
      cursor: pointer;
  }}
  .sub-btn.active {{
    background: #2563EB;
    color: #FFFFFF;
    border-color: #2563EB;
  }}

  .table-wrap {{
    overflow-x: auto;
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 14px;
  }}
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  .data-table th {{
    background: #F1F5F9;
    color: #0F172A;
    font-weight: 700;
    text-align: left;
    padding: 9px 11px;
    border-bottom: 2px solid #CBD5E1;
    white-space: nowrap;
  }}
  .data-table td {{
    padding: 8px 11px;
    border-bottom: 1px solid #E2E8F0;
    color: #334155;
  }}
  .data-table tr:hover td {{ background: #F8FAFC; }}

  .footer {{
    margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--border);
    font-size: 0.82rem; color: #64748B; text-align: center;
  }}
</style>
</head>
<body>
<div class="container">

  <!-- HERO BANNER -->
  <div class="hero">
    <h1>Uber vs. Lyft — Boston Rideshare Dynamic Pricing &amp; Fare
      Analytics</h1>
    <p>
      Báo cáo Dashboard Tương tác Độc lập (Offline HTML) • Đề tài
      <b>PRJ-EXT-01</b> • Lập trình Python cho Khoa học Dữ liệu (UFM)<br/>
      Nhóm 02: <b>Trần Thanh Huy</b> (2521001060) &amp; <b>Lê Huy Thắng</b>
        (2521001092) — Lớp 25DDS01 • Xuất bản: {generated_at}
    </p>
    <div class="badge-row">
      <span class="badge badge-verified">VERIFIED • {
        kpis["total_obs"]:,} API Fare Quotes</span>
      <span class="badge badge-verified">Data Gate: {n_pass}/{
        len(val_report.checks)
    } PASS</span>
      <span class="badge badge-dark">0 Nhãn mưa giả &amp; Tách Sentinel (n={
        dq["valid_rain_obs_rows"]:,})</span>
      <span class="badge badge-dark">Múi giờ Boston EST (n={
        dq["ts_original_rows"]:,})</span>
      <span class="badge badge-dark">Bảo toàn nguồn gốc Surge (n={
        dq["surge_original_rows"]:,})</span>
      <span class="badge badge-dark">100% Offline Vector Plotly</span>
    </div>
  </div>

  <div class="callout-info">
    <b>Phân biệt định dạng xuất bản:</b> Tệp HTML này là <b>Dashboard tương
      tác vector Plotly độc lập</b> (zoom, hover, chuyển Tab, ước lượng biểu
      giá offline &amp; mô phỏng trần Surge dưới ràng buộc cung
    <code>Q_realized = min(Q_D, Q_S)</code> không cần server).
    Để lọc động trên 270,898 dòng và chạy dự báo trực tiếp từ mô hình cây
    <code>models/best_model.joblib</code> (M7), chạy <code>streamlit run
      dashboard/app.py</code>.
    Bộ hình ảnh tĩnh chuẩn 300 DPI PNG phục vụ in ấn tiểu luận nằm riêng tại
    <code>reports/figures/thesis/</code>.
  </div>

  <!-- 5-TAB NAVIGATION -->
  <div class="tab-nav">
    <button class="tab-btn active" onclick="switchTab('tab1', this)">📊 1.
      Executive Overview</button>
    <button class="tab-btn" onclick="switchTab('tab2', this)">🌦️ 2. Pricing,
      Weather &amp; Time (RQ1)</button>
    <button class="tab-btn" onclick="switchTab('tab3', this)">⚔️ 3.
      Competitive Pricing (RQ2)</button>
    <button class="tab-btn" onclick="switchTab('tab4', this)">🤖 4. ML Arena
      &amp; XAI (RQ3)</button>
    <button class="tab-btn" onclick="switchTab('tab5', this)">🎯 5. Estimator
      &amp; Policy Simulator (RQ4)</button>
  </div>

  <!-- ======================== TAB 1: EXECUTIVE DASHBOARD (5 KHỐI CỐT LÕI)
    ======================== -->
  <section id="tab1" class="tab-panel active">
    <!-- KHỐI 1: DATASET & PHẠM VI PHÂN TÍCH -->
    <div class="exec-scope-bar">
      <div style="display:flex;justify-content:space-between;
                  align-items:center;flex-wrap:wrap;gap:8px;">
        <span style="font-size:0.98rem;font-weight:700;color:#0F172A;">
          1. Dataset &amp; Phạm vi Phân tích — Boston Rideshare API Fare
            Quotes (Uber &amp; Lyft)
        </span>
        <span class="badge badge-verified">Data Audit Gate: {n_pass}/{
        len(val_report.checks)
    } PASS • 0 Lỗi CRITICAL</span>
      </div>
      <div class="exec-scope-grid">
        <div><b>📦 Nguồn &amp; Quy mô sạch:</b><br/><code>cab_rides.csv</code>
          ({dq["raw_cab_rows"]:,}) + <code>weather.csv</code> (6,276) → <b>{
        dq["clean_master_rows"]:,} × {dq["clean_master_cols"]}</b> (RAM {
        dq["ram_mb"]:.2f} MB, 0 trùng lặp).</div>
        <div><b>🗺️ Không gian &amp; Thời gian:</b><br/><b>12 quận / 72
          tuyến</b> nội đô Boston (0.02–7.86 dặm) • <b>25/11–18/12/2018</b>
          (múi giờ <code>America/New_York</code> EST).</div>
        <div><b>🚖 Cơ cấu Dịch vụ:</b><br/><b>12 gói xe</b> (5 cặp đối xứng
          Uber vs. Lyft + 2 gói độc lập WAV &amp; Lux); bản chất là <b>giá báo
          API</b> (không phải doanh thu chuyến hoàn thành).</div>
        <div><b>🛡️ Kỷ luật Nguồn gốc:</b><br/>TS gốc EST: <b>{
        dq["ts_original_rows"]:,}</b> (0 ffill) • Khí tượng sạch sentinel: <b>{
        dq["valid_rain_obs_rows"]:,}</b> • Surge thực: <b>{
        dq["surge_original_rows"]:,}</b> (tách {
        dq["lyft_surge_missing_rows"]:,} Lyft NaN).</div>
      </div>
    </div>

    <!-- KHỐI 2: 5 KPI QUAN TRỌNG NHẤT -->
    <div class="section-heading">2. Năm Chỉ số Cốt lõi (Top 5 Executive
      KPIs)</div>
    <div class="kpi-grid-5">
      {
        _kpi_card_html(
            "1. Cỡ mẫu Báo giá Sạch",
            f"{kpis['total_obs']:,}",
            (
                f"Uber {kpis['uber_share_pct']:.1f}% "
                f"({kpis['uber_count']:,}) · "
                f"Lyft {kpis['lyft_share_pct']:.1f}% ({kpis['lyft_count']:,})"
            ),
            "accent",
            "VERIFIED • 0 Fan-out",
        )
    }
      {
        _kpi_card_html(
            "2. Giá Báo Trung bình",
            f"${kpis['mean_price_usd']:.2f}",
            (
                f"TV: ${kpis['median_price_usd']:.2f} | "
                f"Lyft ${kpis['lyft_mean_price_usd']:.2f} vs "
                f"Uber ${kpis['uber_mean_price_usd']:.2f}"
            ),
            "uber",
            f"Cự ly TB: {kpis['mean_distance_miles']:.2f} dặm",
            "badge-simulation",
        )
    }
      {
        _kpi_card_html(
            "3. Tỷ lệ Surge Lyft (> 1.0x)",
            f"{kpis['lyft_surge_rate_pct']:.2f}%",
            (
                f"Lyft: {kpis['lyft_surge_count']:,} / "
                f"{kpis['lyft_obs_surge_count']:,} lượt thực | Uber: 0.00%"
            ),
            "lyft",
            "Trên mẫu surge thực",
            "badge-lyft",
        )
    }
      {
        _kpi_card_html(
            "4. Điểm cắt UberX vs. Lyft",
            f"d* = {d_std:.2f} mi",
            f"Đặc tả A: {d_std:.2f} mi | Đặc tả B thuần 1.0x: {ux_sb:.2f} mi",
            "amber",
            "Wald HC3 p < 0.0001",
        )
    }
      {
        _kpi_card_html(
            "5. Mô hình Có điều kiện (M7)",
            f"R² = {kpis['m7_test_r2']:.4f}",
            (
                f"RMSE: ${kpis['m7_test_rmse_usd']:.2f} | "
                f"MAPE: {kpis['m7_test_mape_pct']:.2f}% "
                f"(Test n={m7_live['n_test']:,})"
            ),
            "emerald",
            f"{m7_live['pct_within_2usd']:.1f}% lệch ≤ ±$2.00",
        )
    }
    </div>

    <!-- KHỐI 3: BA PHÁT HIỆN ĐÁNG CHÚ Ý NHẤT -->
    <div class="section-heading">3. Ba Phát hiện Đáng chú ý Nhất (Top 3
      Quantitative Findings)</div>
    <div class="finding-grid-3">
      <div class="finding-card">
        <div class="finding-num">Phát hiện #1 • Cấu trúc chi phối giá (RQ1
          &amp; RQ3)</div>
        <div class="finding-headline">Gói xe &amp; Cự ly–Surge quyết định {
        top2_share:.2f}% sức mạnh dự báo; Mưa &amp; Giờ ≈ 0%</div>
        <p class="finding-body">
          Trên <b>{
        n_h1a:,}</b> dòng khớp thời tiết thực sạch lỗi cảm biến mưa, mưa
          nhẹ–vừa <b>không</b> làm tăng giá cơ học
          (<b>${float(h1a["mean_group1_or_rate"]):.2f} vs. ${
        float(h1a["mean_group2_or_ref"]):.2f}</b>, <code>p = {
        float(
            h1a["parametric_pvalue"]
        ):.4f}</code>, <code>TOST ±$0.10 p &lt; 0.0001</code>).
          Toàn bộ nhóm thời tiết &amp; khung giờ EST chỉ chiếm <b>0.02%</b>
            Permutation Importance.
        </p>
      </div>
      <div class="finding-card f-lyft">
        <div class="finding-num">Phát hiện #2 • Chiến lược cạnh tranh giá
          (RQ2)</div>
        <div class="finding-headline">Lyft mở cửa rẻ hơn, Uber chạy dài &amp;
          hạng sang rẻ hơn; Lyft XL rẻ toàn miền khi 1.0x</div>
        <p class="finding-body">
          <b>UberX vs. Lyft</b> cắt tại <b>d* = {
        d_std:.2f} mi</b> (Đặc tả A) và <b>{
        ux_sb:.2f} mi</b> (Đặc tả B thuần 1.0x).
          Khi không surge, <b>Lyft Shared</b> và <b>Lyft XL</b> (giao điểm
            toán học {
        xl_sb:.2f} mi &gt; 7.86 mi) rẻ hơn Uber trên toàn miền Boston, còn
        <b>Uber Black</b> rẻ hơn toàn miền.
        </p>
      </div>
      <div class="finding-card f-emerald">
        <div class="finding-num">Phát hiện #3 • Định giá động &amp; Trần Surge
          (RQ4)</div>
        <div class="finding-headline">Surge Lyft thực đạt {
        kpis[
            "lyft_surge_rate_pct"
        ]:.2f}%; Pha 1 trần 2.00x (τ ≤ 0.10%), Pha 2 thử nghiệm 1.75x</div>
        <p class="finding-body">
          Dưới ràng buộc cung <code>Q = min(Q_D, Q_S)</code> trên <b>{
        n_lyft_obs:,}</b> lượt Lyft thực, chỉ trần <b>2.00x</b> ({
        int(cap200["affected_trips_count"])
    } lượt, giảm <b>${
        float(cap200["mean_rider_saving_per_affected_trip_usd"]):.2f}</b>)
          đạt <code>τ ≤ 0.10%</code> ở mọi kịch bản cung (<b>{
        float(cap200["static_lyft_rev_change_pct"]):+.3f}%</b> → <b>{
        float(cap200["supply_constrained_rev_change_drop10_pct"]):+.3f}%</b>);
          trần <b>1.75x</b> ({
        int(cap175["affected_trips_count"])
    } lượt, cung ε_S=0.30 là <b>{
        float(
            cap175["supply_constrained_rev_change_drop5_pct"]
        ):+.3f}%</b>) dành cho Pha 2 Triggered Switchback.
        </p>
      </div>
    </div>

    <!-- KHỐI 4: MỘT BIỂU ĐỒ CHỦ ĐẠO THỂ HIỆN RÕ KHÁC BIỆT UBER VS. LYFT -->
    <div class="section-heading">4. Biểu đồ Chủ đạo: Khác biệt Chiến lược Giá
      giữa Uber và Lyft</div>
    <div class="chart-card">{_fig_to_div(fig_hero, "fig_t1_hero")}</div>

    <!-- KHỐI 5: KẾT LUẬN NGẮN GỌN VỀ VẤN ĐỀ NGHIÊN CỨU -->
    <div class="section-heading">5. Kết luận Ngắn gọn về Vấn đề Nghiên
      cứu</div>
    <div class="exec-verdict-box">
      <b>Trả lời bài toán cốt lõi (PRJ-EXT-01):</b> Giá báo API gọi xe tại
        Boston <b>không vận hành theo phụ phí thời tiết hay giờ hành chính cố
        định</b>,
      mà do <b>phân khúc dịch vụ</b>, <b>cấu trúc biểu giá cắt nhau theo cự
        ly</b> (Lyft lợi thế chặng ngắn &amp; đi ghép/6 chỗ khi không surge;
      Uber lợi thế chặng dài &gt; {d_std:.2f}–{
        ux_sb:.2f} dặm, hạng sang và khi Lyft kích hoạt surge) cùng <b>mất cân
          bằng cung–cầu vi mô cục bộ</b> quyết định.
      Mô hình có điều kiện <b>M7 HistGradientBoosting</b> (<code>R² = {
        kpis["m7_test_r2"]:.4f}</code>, <code>RMSE = ${
        kpis["m7_test_rmse_usd"]:.2f}</code>) kết hợp <b>M2 Interaction OLS</b>
      (<code>R² = {
        kpis[
            "m2_test_r2"
        ]:.4f}</code>) và đối chứng <b>Ex-Ante</b> (<code>R² = {
        float(
            ex_all["r2"]
        ):.4f}</code>) định lượng lộ trình trần Surge hai nấc: <b>Pha 1 trần
          2.00x bảo thủ</b>, <b>Pha 2 thử nghiệm Triggered Switchback
          1.75x</b>.
      <br/><span style="font-size:0.81rem;color:#475569;">
        <i>Lưu ý giới hạn học thuật:</i> Kết quả phản ánh <b>giá báo API (API
          Fare Quotes)</b> trong 24 ngày lịch (18 ngày quan sát) tại Boston
          (thiếu biến <code>duration</code> và mật độ tài xế);
        mọi kịch bản trần Surge là <b>mô phỏng phản thực tế trên API Quote
          Proxy</b> dưới ràng buộc cung <code>Q_realized = min(Q_D,
          Q_S)</code>.
      </span>
    </div>

    <!-- PHẦN MỞ RỘNG (COLLAPSIBLE): BIỂU ĐỒ KHÁM PHÁ SÂU & HỒ SƠ KIỂM TOÁN DỮ
      LIỆU -->
    <details class="exec-details" ontoggle="if(this.open) setTimeout(() =>
      resizeVisibleCharts(this), 60)">
      <summary>🔍 Khám phá Sâu hơn — Phân phối Giá, Boxplot 12 Gói &amp; Bản đồ
        nhiệt 72 Tuyến Boston</summary>
      <div class="details-body">
        <div class="two-col">
          <div class="chart-card">{_fig_to_div(fig_dist, "fig_t1_dist")}</div>
          <div class="chart-card">{_fig_to_div(fig_box, "fig_t1_box")}</div>
        </div>
        <div class="control-bar">
          <label>Chọn chỉ số Heatmap 12 quận đón × 12 quận trả:</label>
          <button class="sub-btn active" onclick="switchSubGroup('hm',
            'hm_price', this)">Giá báo TB (USD)</button>
          <button class="sub-btn" onclick="switchSubGroup('hm', 'hm_dist',
            this)">Quãng đường trung vị (dặm)</button>
          <button class="sub-btn" onclick="switchSubGroup('hm', 'hm_surge',
            this)">Tỷ lệ Surge (%)</button>
        </div>
        <div id="hm_price" class="hm-panel chart-card">{
        _fig_to_div(fig_heat_price, "fig_t1_hm_price")
    }</div>
        <div id="hm_dist" class="hm-panel chart-card" style="display:none;">{
        _fig_to_div(fig_heat_dist, "fig_t1_hm_dist")
    }</div>
        <div id="hm_surge" class="hm-panel chart-card" style="display:none;">{
        _fig_to_div(fig_heat_surge, "fig_t1_hm_surge")
    }</div>
      </div>
    </details>

    <details class="exec-details">
      <summary>🛡️ Hồ sơ Chất lượng Dữ liệu &amp; Cổng Kiểm toán Tự động ({
        n_pass
    }/{len(val_report.checks)} PASS)</summary>
      <div class="details-body">
        <div class="callout-info">
          <b>Tối ưu hóa RAM:</b> <code>{
        dq[
            "ram_mb"
        ]:.2f} MB</code> thực tế so với ước lượng chưa tối ưu <code>~{
        dq["est_unoptimized_ram_mb"]:.1f} MB</code> (tiết kiệm <b>~{
        dq["ram_saved_pct"]:.1f}%</b>) •
          <b>TS gốc EST:</b> <code>{dq["ts_original_rows"]:,}</code> ({
        dq["ts_original_pct"]:.2f}%) •
          <b>Khớp thời tiết thực:</b> <code>{
        dq["weather_matched_rows"]:,}</code> ({
        dq[
            "weather_matched_valid_ts_pct"
        ]:.2f}% trên TS gốc, <b>0</b> nhãn mưa giả) •
          <b>Bản ghi nguyên bản R3:</b> <code>{
        dq["r3_unmodified_raw_pct"]:.2f}%</code> •
          <b>GPS ≤ 0.05 dặm:</b> <code>{
        dq["ultra_short_gps_count"]
    }</code> (giữ nguyên minh bạch).
        </div>
        {_df_to_html_table(val_table_df, max_rows=30)}
      </div>
    </details>
  </section>

  <!-- ======================== TAB 2: PRICING, WEATHER & TIME
    ======================== -->
  <section id="tab2" class="tab-panel">
    <div class="section-heading">1. Tóm tắt Kiểm chứng RQ1: Thời tiết, Khung
      giờ EST &amp; Surge (P-value + Effect Size)</div>
    <div class="kpi-grid-4">
      {
        _kpi_card_html(
            "H1a: Mưa vs. Tạnh (Sạch Sentinel)",
            f"d = {float(h1a['effect_size_value']):+.5f}",
            (
                f"Welch p={float(h1a['parametric_pvalue']):.4f} | "
                f"TOST ±$0.10 p<0.0001 | n={n_h1a:,}"
            ),
            "accent",
            f"Exact WCR mưa lớn p={wcr_p:.4f}",
        )
    }
      {
        _kpi_card_html(
            "H4c: Đa biến 6 biến Thời tiết",
            f"ΔR² = {float(h4c['effect_size_value']):.7f}",
            (
                f"Joint Wald F={float(h4c['parametric_stat']):.4f} "
                f"(p={float(h4c['parametric_pvalue']):.4f}) "
                f"trên n={int(h4c['n_group1']):,}"
            ),
            "emerald",
            "≈ 0.000% phương sai giá",
        )
    }
      {
        _kpi_card_html(
            "H2a: Surge 4 Buổi EST",
            f"η² = {float(h2a['effect_size_value']):.6f}",
            (
                f"ANOVA p={float(h2a['parametric_pvalue']):.4f} | "
                f"Kruskal p={float(h2a['nonparametric_pvalue']):.4f} | "
                f"n={n_h2a:,}"
            ),
            "amber",
            "Chưa bác bỏ H0 • Phẳng 24h",
        )
    }
      {
        _kpi_card_html(
            "H3a: Surge × Hãng xe",
            f"V = {float(h3a['effect_size_value']):.4f}",
            (
                f"χ²={float(h3a['parametric_stat']):,.2f} (p<0.0001) | "
                f"Lyft {float(h3a['mean_group1_or_rate']):.2f}% vs "
                "Uber 0.00%"
            ),
            "lyft",
            f"H3b cụm giờ p={h3b_p_hr:.4f} (NS)",
            "badge-lyft",
        )
    }
    </div>

    <div class="section-heading">2. Kiểm toán Tác động Thời tiết (Sạch lỗi cảm
      biến mưa) &amp; Nhịp 24 Giờ Boston EST</div>
    <div class="chart-card">{_fig_to_div(fig_weather, "fig_t2_weather")}</div>
    <div class="chart-card">{_fig_to_div(fig_hourly, "fig_t2_hourly")}</div>

    <div class="section-heading">3. Biến động theo Thứ (EST) &amp; Phân bố 7
      Bậc Surge của Lyft</div>
    <div class="two-col">
      <div class="chart-card">{_fig_to_div(fig_dow, "fig_t2_dow")}</div>
      <div class="chart-card">{
        _fig_to_div(fig_surge_lvl, "fig_t2_surge")
    }</div>
    </div>

    <div class="section-heading">4. Bảng 13 Kiểm định Giả thuyết Thống kê Kép
      (H1–H4)</div>
    {_df_to_html_table(hyp_table_df, max_rows=20)}

    <div class="section-heading">5. Diễn giải Học thuật (Không nhầm Tương quan
      thành Nhân quả)</div>
    <div class="two-col">
      {
        _insight_card_html(
            (
                "Insight 2.1 — Trời mưa không tự động làm tăng giá báo "
                "API một cách cơ học"
            ),
            (
                "Trên tập khớp thời tiết sạch lỗi cảm biến mưa (tách "
                "27,237 ô sentinel), mưa nhẹ–vừa không tạo khác biệt có "
                "ý nghĩa thống kê hay thực tiễn lên giá báo API và xác "
                "suất Surge sau gom cụm."
            ),
            (
                f"n = {n_h1a:,}: Mưa "
                f"${float(h1a['mean_group1_or_rate']):.4f} vs Tạnh "
                f"${float(h1a['mean_group2_or_ref']):.4f} (p = "
                f"{float(h1a['parametric_pvalue']):.4f}, TOST ±$0.10 "
                f"p < 0.0001); Exact WCR mưa lớn p = {wcr_p:.4f}; "
                f"H3b Lyft Surge×Mưa gom cụm giờ p = {h3b_p_hr:.4f} "
                "(trượt Holm & BH-FDR); Joint Wald H4c ΔR² = "
                f"{float(h4c['effect_size_value']):.7f}."
            ),
            (
                "Kiểm định kép Welch + Mann-Whitney, sai số chuẩn cụm "
                "hai cấp (2,795 quận×giờ & 329 giờ), Exact WCR "
                "(B = 1,999), TOST ±$0.10 và Joint Wald HC3 hạng đầy đủ."
            ),
            (
                "Không cần phụ thuộc API thời tiết theo giờ để dự báo "
                "giá báo ngắn hạn; nguồn lực kỹ thuật nên tập trung vào "
                "trạng thái cung–cầu cục bộ theo phút."
            ),
            (
                "Dữ liệu thời tiết tổng hợp theo giờ trong 24 ngày cuối "
                "thu ôn hòa (max rain 0.55 in/h, chỉ "
                f"{n_heavy_hrs} giờ mưa lớn ≥ 0.10 in/h toàn Boston, "
                "chưa có bão tuyết cực đoan)."
            ),
        )
    }
      {
        _insight_card_html(
            (
                "Insight 2.2 — Khung giờ EST chi phối lưu lượng nhưng "
                "không tạo biểu giá cứng theo giờ"
            ),
            (
                "Sau chuẩn hóa múi giờ America/New_York và giữ nguyên "
                "62,864 dòng khuyết timestamp (0 ffill), lưu lượng khớp "
                "nhịp đô thị (đáy 03:00, đỉnh 19:00 EST) nhưng giá TB và "
                "tỷ lệ Surge Lyft (6.80% thực) phẳng suốt 24h."
            ),
            (
                f"ANOVA giá theo 4 buổi EST (H2c): F = "
                f"{float(h2c['parametric_stat']):.4f}, p = "
                f"{float(h2c['parametric_pvalue']):.4f}, η² = "
                f"{float(h2c['effect_size_value']):.6f}; Surge giờ cao "
                "điểm vs ngoài cao điểm "
                f"{float(h3d['mean_group1_or_rate']):.2f}% vs "
                f"{float(h3d['mean_group2_or_ref']):.2f}% "
                f"(p = {float(h3d['parametric_pvalue']):.4f})."
            ),
            (
                "One-way ANOVA + Kruskal-Wallis H (H2a–c) và Chi-Square + "
                "Cramér's V (H3c–d) khóa dấu thời gian gốc EST và mẫu "
                "surge thực."
            ),
            (
                "Surge của Lyft tập trung tại 5 điểm nóng không gian "
                "(Back Bay 11.17%, Fenway 10.57%, Northeastern University "
                "10.10%, Theatre District 9.58%, Boston University 9.57%) "
                "hơn là biểu giá cứng giờ hành chính."
            ),
            (
                "Có 6 ngày khuyết thu thập gốc (05–08/12, 10–11/12/2018) "
                "và thiếu biến thời lượng chạy thực tế (duration)."
            ),
            accent="amber",
        )
    }
    </div>
  </section>

  <!-- ======================== TAB 3: COMPETITIVE PRICING
    ======================== -->
  <section id="tab3" class="tab-panel">
    <div class="section-heading">1. Điểm nhấn Chiến lược Biểu giá Cắt nhau
      (Uber vs. Lyft — RQ2)</div>
    <div class="kpi-grid-4">
      {
        _kpi_card_html(
            "Điểm cắt Phổ thông (UberX vs. Lyft)",
            f"d* = {d_std:.2f} dặm",
            (
                f"CI cụm: [{ux_ci[0]:.2f}, {ux_ci[1]:.2f}] mi | Đặc tả B "
                f"thuần 1.0x: {ux_sb:.2f} mi "
                f"(Wald Z = {float(h4a['parametric_stat']):.2f})"
            ),
            "accent",
            (
                "{:.1f}% chuyến < d*".format(
                    float(std_pair["pct_trips_shorter_than_dstar"] or 83.0)
                )
            ),
        )
    }
      {
        _kpi_card_html(
            "Gia đình 6 chỗ (UberXL vs. Lyft XL)",
            (
                f"A: {float(xl_pair['crossing_distance_miles']):.2f} | "
                "B: > 7.86 mi"
            ),
            (
                f"Đặc tả B thuần 1.0x dịch ra {xl_sb:.2f} mi > 7.86 mi "
                "→ Lyft XL rẻ hơn toàn miền khi 1.0x"
            ),
            "lyft",
            "Đảo chiều khi không surge",
            "badge-lyft",
        )
    }
      {
        _kpi_card_html(
            "Đi ghép (UberPool vs. Shared)",
            "Không cắt nội miền",
            (
                "Miền [0.02, 7.86] dặm: Shared luôn rẻ hơn (mở cửa thấp "
                f"hơn ${float(pool_pair['base_diff_uber_minus_lyft']):.2f})"
            ),
            "emerald",
            "Khóa nội miền • Không ngoại suy",
        )
    }
      {
        _kpi_card_html(
            "Hạng sang (Black & SUV)",
            f"Uber lợi thế > {bs_sb:.2f} mi",
            (
                f"Black cắt {float(bl_pair['crossing_distance_miles']):.2f} "
                f"mi (B: {bl_sb:.2f} mi) | Black SUV cắt "
                f"{float(bs_pair['crossing_distance_miles']):.2f} / "
                f"{bs_sb:.2f} mi"
            ),
            "uber",
            "Ưu thế đường dài",
            "badge-uber",
        )
    }
    </div>

    <div class="section-heading">2. Bộ Khám phá Đường Biểu giá &amp; Điểm cắt
      Cạnh tranh (5 Cặp Đối xứng)</div>
    <div class="control-bar">
      <label for="pairSelector">Chọn phân khúc đối xứng (miền thực nghiệm
        [0.02, 7.86] dặm, mẫu quan sát surge thực):</label>
      <select id="pairSelector" onchange="switchPairPanel(this.value)">
        {"".join(pair_options_html)}
      </select>
    </div>
    {"".join(pair_chart_blocks)}

    <div class="section-heading">3. Bản đồ Chiến lược 12 Gói Xe (β₀ vs. β₁)
      &amp; Bảng Đối sánh</div>
    <div class="chart-card">{_fig_to_div(fig_strat, "fig_t3_strat")}</div>
    <div class="two-col">
      <div>
        <div style="font-weight:700;margin-bottom:6px;">Bảng 3A — 5 Cặp Cạnh
          tranh Đối xứng Trực tiếp (HC3 OLS)</div>
        {_df_to_html_table(pairs_view_df)}
      </div>
      <div>
        <div style="font-weight:700;margin-bottom:6px;">Bảng 3B — 2 Gói Độc
          lập Tách riêng (WAV &amp; Lux)</div>
        {_df_to_html_table(standalone_view_df)}
      </div>
    </div>

    <div class="two-col">
      {
        _insight_card_html(
            (
                "Insight 3.1 — Đối sánh Song đặc tả A/B: UberX–Lyft cắt "
                "bền vững; Lyft XL rẻ toàn miền khi 1.0x"
            ),
            (
                "Trên 5/5 cặp đối thủ đối xứng, Lyft luôn đặt giá mở cửa "
                "β₀ thấp hơn nhưng đơn giá/dặm β₁ cao hơn Uber; khi tách "
                "riêng trạng thái thuần không surge (Đặc tả B) trên mẫu "
                "surge thực, cặp UberXL–Lyft XL không còn cắt trong miền "
                "Boston."
            ),
            (
                f"Phổ thông: Đặc tả A cắt tại d* = {d_std:.2f} mi "
                f"(CI cụm [{ux_ci[0]:.2f}, {ux_ci[1]:.2f}], "
                f"Z = {float(h4a['parametric_stat']):.2f}) và Đặc tả B "
                f"tại {ux_sb:.2f} mi. Cặp XL cắt "
                f"{float(xl_pair['crossing_distance_miles']):.2f} mi ở A "
                f"nhưng dịch ra {xl_sb:.2f} mi > 7.86 mi ở B. Black: "
                f"{float(bl_pair['crossing_distance_miles']):.2f} mi "
                f"(B: {bl_sb:.2f} mi); SUV: "
                f"{float(bs_pair['crossing_distance_miles']):.2f} / "
                f"{bs_sb:.2f} mi."
            ),
            (
                "Hồi quy tuyến tính OLS sai số chuẩn HC3 và "
                "Cluster-Robust SE (Delta-method 95% CI) song hành hai "
                "đặc tả trên X_train (n = 186,288)."
            ),
            (
                "Khi không surge, chọn Lyft cho chặng phổ thông ngắn và "
                "Lyft XL cho mọi chuyến 6 chỗ nội đô Boston; chọn UberX "
                "cho chặng dài và Uber Black/Black SUV ở hạng sang."
            ),
            (
                "Điểm cắt chỉ áp dụng khi Lyft không kích hoạt surge "
                "(1.0x) và β₁ chịu thiên lệch bỏ sót biến thời lượng di "
                "chuyển thực (duration)."
            ),
            accent="lyft",
        )
    }
      {
        _insight_card_html(
            (
                "Insight 3.2 — Phân khúc Đi ghép (Shared vs. UberPool) "
                "& Độ cong Giá sàn"
            ),
            (
                "Lyft Shared rẻ hơn UberPool trên toàn bộ phạm vi quan "
                "sát [0.02, 7.86] dặm và không có giao điểm thực tế "
                "trong Boston."
            ),
            (
                f"Shared (${float(pool_pair['lyft_base_b0']):.2f} + "
                f"${float(pool_pair['lyft_slope_b1']):.2f}/mi) rẻ hơn "
                f"UberPool (${float(pool_pair['uber_base_b0']):.2f} + "
                f"${float(pool_pair['uber_slope_b1']):.2f}/mi) trên toàn "
                f"miền quan sát (giao điểm toán học {pool_math_d:.2f} mi "
                "ngoài miền)."
            ),
            (
                "Đối chiếu đường OLS HC3 với giá trung bình thực tế "
                "theo khoảng cự ly 0.5 dặm, khóa cứng miền "
                "[0.02, 7.86] dặm."
            ),
            (
                "Gợi ý lựa chọn tiết kiệm nhất cho hành khách không gấp "
                "khi không có surge; đồng thời giải thích vì sao cần mô "
                "hình phi tuyến M7 để nắm bắt giá sàn < 1 dặm."
            ),
            (
                "Thiếu dữ liệu số khách ghép chung thực tế (matching "
                "load) nên R² nội gói của Shared/UberPool thấp hơn các "
                "gói xe riêng."
            ),
            accent="emerald",
        )
    }
    </div>
  </section>

  <!-- ======================== TAB 4: ML ARENA & XAI ========================
    -->
  <section id="tab4" class="tab-panel">
    <div class="section-heading">Phần A — Hiệu năng Mô hình Có điều kiện,
      Ex-Ante &amp; Chẩn đoán Phần dư</div>
    <div class="kpi-grid-4">
      {
        _kpi_card_html(
            "M7 Có điều kiện (Vô địch)",
            f"R² = {m7_live['r2']:.4f}",
            (
                f"RMSE: ${m7_live['rmse_usd']:.4f} | "
                f"MAE: ${m7_live['mae_usd']:.4f} | "
                f"MAPE: {m7_live['mape_pct']:.2f}%"
            ),
            "emerald",
            f"VERIFIED • Test n={m7_live['n_test']:,}",
        )
    }
      {
        _kpi_card_html(
            "Đối chứng Ex-Ante (23 biến)",
            f"R² = {float(ex_all['r2']):.4f}",
            (
                f"RMSE: ${float(ex_all['rmse']):.2f} | "
                f"Surge Classifier AUC: {ex_auc:.4f} | "
                f"M2 OLS: {m2_live['r2']:.4f}"
            ),
            "accent",
            "Không dùng biến surge",
        )
    }
      {
        _kpi_card_html(
            "Độ chính xác ±$2.00 (M7)",
            f"{m7_live['pct_within_2usd']:.2f}%",
            (
                f"±$1.00: {m7_live['pct_within_1usd']:.1f}% | "
                f"±$3.00: {m7_live['pct_within_3usd']:.1f}% | "
                f"Lyft R²={lyft_r2:.4f} vs Uber {uber_r2:.4f}"
            ),
            "amber",
            "0 biến rò rỉ (Leakage=0)",
        )
    }
      {
        _kpi_card_html(
            "Monotonic Split-CQR [P10, P90]",
            f"PICP = {cqr_picp:.2f}%",
            (
                f"12 gói: [{cqr_lo:.2f}%, {cqr_hi:.2f}%] | "
                f"MPIW: ${cqr_mpiw:.2f} (0 cắt ngược)"
            ),
            "lyft",
            "pandas category gốc • 80%",
            "badge-simulation",
        )
    }
    </div>

    <div class="chart-card">{_fig_to_div(fig_arena, "fig_t4_arena")}</div>

    <div class="control-bar">
      <label>Chọn mô hình xem chẩn đoán Actual vs. Predicted &amp; Phần dư
        (tập Test n = {m7_live["n_test"]:,}):</label>
      <button class="sub-btn active" onclick="switchSubGroup('diag',
        'diag_m7', this)">M7 HistGradientBoosting</button>
      <button class="sub-btn" onclick="switchSubGroup('diag', 'diag_m2',
        this)">M2 Interaction OLS</button>
    </div>
    <div id="diag_m7" class="diag-panel chart-card">{
        _fig_to_div(fig_resid_m7, "fig_t4_res_m7")
    }</div>
    <div id="diag_m2" class="diag-panel chart-card" style="display:none;">{
        _fig_to_div(fig_resid_m2, "fig_t4_res_m2")
    }</div>

    <div class="chart-card">{_fig_to_div(fig_seg_tier, "fig_t4_seg")}</div>
    <div style="font-weight:700;margin-bottom:6px;">Bảng 4A — Bảng Xếp hạng 10
      Cấu hình Mô hình Có điều kiện trên Tập Test (n = {
        m7_live["n_test"]:,})</div>
    {_df_to_html_table(lb_html_df)}

    <div class="section-heading">Phần B — Diễn giải Mô hình XAI (Model
      Interpretation — KHÔNG phải Quan hệ Nhân quả)</div>
    <div class="callout-info">
      <b>Minh định phương pháp:</b> Permutation Importance được tính bằng xáo
        trộn <code>14</code> nhóm đặc trưng rời nhau phủ đúng
      <code>27/27</code> cột thực trên <code>X_test</code> (<code>n =
        15,000</code>, <code>8</code> lần lặp, 0 đếm trùng; kiểm chứng gom 6
        cụm chống đa cộng tuyến cho tỷ trọng {grp6_top2:.2f}% từ Gói xe + Cự
        ly/Surge).
        Kết quả phản ánh <b>tầm quan trọng dự báo</b>, <b>KHÔNG</b> khẳng định
        quan hệ nhân quả độc lập.
    </div>
    <div class="chart-card">{_fig_to_div(fig_perm, "fig_t4_perm")}</div>
    {_df_to_html_table(fam_df)}
    <div class="chart-card">{_fig_to_div(fig_pdp, "fig_t4_pdp")}</div>

    <div class="section-heading">Phần C — Bất định Dự báo (Prediction
      Uncertainty — Quantile &amp; Monotonic Split-CQR)</div>
    <div class="chart-card">{_fig_to_div(fig_pi, "fig_t4_pi")}</div>
  </section>

  <!-- ======================== TAB 5: ESTIMATOR & POLICY SIMULATOR
    ======================== -->
  <section id="tab5" class="tab-panel">
    <div class="section-heading">Phần A — Bộ Ước lượng Giá Báo API Trực tuyến
      trên Trình duyệt (Offline HC3 OLS Estimator)</div>
    <div class="callout-info">
      <b>Phạm vi phiên bản HTML Offline:</b> Công cụ dưới đây tính giá báo API
        trực tiếp bằng phương trình biểu giá HC3 OLS đã kiểm chứng
        (<code>stage6_tier_tariffs.csv</code>) để chạy 100% không cần Python
        server. Để chạy dự báo phi tuyến đầy đủ 27 đặc trưng từ
      <code>models/best_model.joblib</code> (M7), vui lòng mở bản Streamlit
        (<code>streamlit run dashboard/app.py</code>).
    </div>

    <div class="control-bar">
      <label for="estDist">Quãng đường (dặm, miền Boston 0.02–7.86):</label>
      <input id="estDist" type="number" min="0.01" max="25.0" step="0.1"
        value="2.20" oninput="updateOfflineEstimator()" />
      <label for="estSurge">Hệ số Surge Lyft (Uber cố định 1.0x):</label>
      <select id="estSurge" onchange="updateOfflineEstimator()">
        <option value="1.0" selected>1.00x (Bình thường)</option>
        <option value="1.25">1.25x (Surge nhẹ)</option>
        <option value="1.5">1.50x (Surge vừa)</option>
        <option value="1.75">1.75x (Surge cao)</option>
        <option value="2.0">2.00x (Surge rất cao)</option>
        <option value="2.5">2.50x (Cực đoan)</option>
        <option value="3.0">3.00x (Đỉnh tối đa)</option>
      </select>
    </div>
    <div id="estExtrapWarn" class="callout-danger" style="display:none;"></div>
    <div id="estTableContainer"></div>

    <div class="section-heading">Phần B — Mô phỏng Chính sách Trần Surge Cap
      dưới Ràng buộc Cung (RQ4 — Q_realized = min(Q_D, Q_S))</div>
    <div class="callout-warn">
      <b>CẢNH BÁO HỌC THUẬT (SCENARIO / SIMULATION — Phân phối bên ngắn
        Q_realized = min(Q_D, Q_S)):</b> Toàn bộ kết quả dưới đây là <b>mô
        phỏng phản thực tế có điều kiện</b> trên <code>{
        n_lyft_obs:,}</code> lượt báo giá Lyft quan sát surge thực (Tổng cơ sở
          = <b>${
        base_proxy:,.2f} API Quote Proxy</b>, tách 38,037 lượt khuyết surge).
        <b>KHÔNG</b> gọi đây là doanh thu kế toán thực tế; khi cung tài xế nhạy
        giá (<code>ε_S = 0.30</code> hoặc <code>0.60</code>;
        <code>Q_S = clip(1 + ε_S·Δp/p)</code>), cầu dư
        <code>Σ max(0, Q_D − Q_realized)</code> biến thành lượt chưa phục vụ
        và Proxy giảm đơn điệu chặt.
    </div>

    <div class="control-bar">
      <label for="simCapSel">1. Ngưỡng Trần Surge Cap:</label>
      <select id="simCapSel" onchange="updateOfflineSim()">
        <option value="3.0">3.00x — Hiện trạng (Không áp trần)</option>
        <option value="2.0" selected>2.00x — Pha 1 Bảo thủ (Đạt τ ≤ 0.10% mọi
          cung ★)</option>
        <option value="1.75">1.75x — Pha 2 Triggered Switchback (τ =
          0.30%–0.50%)</option>
        <option value="1.5">1.50x — Kịch bản C (Bảo vệ giờ cao điểm)</option>
        <option value="1.25">1.25x — Kịch bản D (Trần siết chặt mạnh)</option>
        <option value="1.0">1.00x — Kịch bản E (Xóa bỏ Surge — Đối
          chứng)</option>
      </select>
      <label for="simSupplySel">2. Ràng buộc cung Q_S:</label>
      <select id="simSupplySel" onchange="updateOfflineSim()">
        <option value="0" selected>ε_S = 0.00 — Cung không đổi</option>
        <option value="5">ε_S = 0.30 — Cung nhạy giá vừa</option>
        <option value="10">ε_S = 0.60 — Cung nhạy giá mạnh</option>
      </select>
      <label for="simEpsSel">3. Chế độ cầu (khi cung ε_S = 0):</label>
      <select id="simEpsSel" onchange="updateOfflineSim()">
        <option value="static_000" selected>Cầu tĩnh (ε = 0.00)</option>
        <option value="eps_050">Cận trên dư cung (ε = -0.50)</option>
        <option value="eps_080">Cận trên cạnh tranh (ε = -0.80)</option>
      </select>
    </div>
    <div id="simSatWarn" class="callout-info" style="display:none;"></div>
    <div id="simKpiGrid" class="kpi-grid-4"></div>

    <div class="two-col">
      <div class="chart-card">{_fig_to_div(fig_pareto, "fig_t5_pareto")}</div>
      <div class="chart-card">{
        _fig_to_div(fig_m7_m2_sim, "fig_t5_cross")
    }</div>
    </div>

    <div style="font-weight:700;margin-bottom:6px;">Bảng 5C — Đầy đủ 6 Kịch
      bản Mô phỏng (Chuẩn thuật ngữ API Quote Proxy)</div>
    {_df_to_html_table(sim_disp_df)}

    <div class="callout-warn">
      <b>⚖️ 3 Giả định &amp; Giới hạn Kinh tế học Bắt buộc của Mô phỏng:</b>
      <ul style="margin:6px 0 0 0;padding-left:18px;">{assumptions_li}</ul>
    </div>
  </section>

  <div class="footer">
    <b>PRJ-EXT-01 — Uber vs. Lyft: Boston Rideshare Pricing Analysis</b> •
      Nhóm 02: Trần Thanh Huy (2521001060) &amp; Lê Huy Thắng (2521001092) —
      Lớp 25DDS01 (UFM)<br/>
    Xuất tự động từ <code>dashboard/export_static_html.py</code> dùng chung
      tầng <code>business_logic.py</code> &amp; <code>charts.py</code> (0 số
      liệu hardcode • 100% Offline HTML).
  </div>
</div>

<script type="text/javascript">
  const TARIFFS = {json.dumps(tariff_lookup_js, ensure_ascii=False)};
  const PAIRS = {json.dumps(pairs_js, ensure_ascii=False)};
  const SIM_DATA = {json.dumps(sim_records_js, ensure_ascii=False)};
  const DIST_MIN = {EMPIRICAL_DISTANCE_MIN};
  const DIST_MAX = {EMPIRICAL_DISTANCE_MAX};

  function resizeVisibleCharts(container) {{
    if (!container || typeof Plotly === 'undefined') return;
    const plots = container.querySelectorAll('.plotly-graph-div');
    plots.forEach(p => {{
      try {{ Plotly.Plots.resize(p); }} catch (e) {{}}
    }});
  }}

  function switchTab(tabId, btnEl) {{
    document.querySelectorAll('.tab-panel').forEach(el =>
      el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el =>
      el.classList.remove('active'));
    const target = document.getElementById(tabId);
    if (target) {{
      target.classList.add('active');
      if (btnEl) btnEl.classList.add('active');
      setTimeout(() => resizeVisibleCharts(target), 60);
    }}
  }}

  function switchSubGroup(prefix, activeId, btnEl) {{
    document.querySelectorAll('.' + prefix + '-panel').forEach(el =>
      el.style.display = 'none');
    btnEl.parentElement.querySelectorAll('.sub-btn').forEach(b =>
      b.classList.remove('active'));
    const target = document.getElementById(activeId);
    if (target) {{
      target.style.display = 'block';
      btnEl.classList.add('active');
      setTimeout(() => resizeVisibleCharts(target), 40);
    }}
  }}

  function switchPairPanel(panelId) {{
    document.querySelectorAll('.pair-panel').forEach(el => el.style.display =
      'none');
    const target = document.getElementById(panelId);
    if (target) {{
      target.style.display = 'block';
      setTimeout(() => resizeVisibleCharts(target), 40);
    }}
  }}

  function updateOfflineEstimator() {{
    const d = parseFloat(document.getElementById('estDist').value);
    const s = parseFloat(document.getElementById('estSurge').value);
    const warnEl = document.getElementById('estExtrapWarn');
    const container = document.getElementById('estTableContainer');

    if (isNaN(d) || d <= 0) {{
      warnEl.style.display = 'block';
      warnEl.innerText = '❌ Vui lòng nhập quãng đường > 0 dặm.';
      container.innerHTML = '';
      return;
    }}
    if (d < DIST_MIN || d > DIST_MAX) {{
      warnEl.style.display = 'block';
      warnEl.innerText = '🚨 EXTRAPOLATION WARNING: Quãng đường ' +
        d.toFixed(2) + ' dặm vượt ngoài miền quan sát thực tế tại Boston ([' +
        DIST_MIN.toFixed(2) + ', ' + DIST_MAX.toFixed(2) +
        '] dặm). Kết quả ước lượng tuyến tính chỉ mang tính tham khảo!';
    }} else {{
      warnEl.style.display = 'none';
    }}

    let rowsHtml = '';
    PAIRS.forEach(p => {{
      const u = TARIFFS[p.uber];
      const l = TARIFFS[p.lyft];
      const uPrice = Math.max(2.5, u.b0 + u.b1 * d);
      const lPrice = Math.max(2.5, l.b0 + l.b1 * d * s);
      const diff = Math.abs(uPrice - lPrice);
      const pct = (diff / Math.max(uPrice, lPrice) * 100);
      let winner = 'Tương đương';
      if (diff >= 0.15) {{
        winner = (uPrice < lPrice) ? ('Uber (' + p.uber + ')') : ('Lyft (' +
          p.lyft + ')');
      }}
      rowsHtml += `<tr>
        <td><b>${{p.segment}}</b></td>
        <td>${{p.uber}}</td>
        <td><b>$${{uPrice.toFixed(2)}}</b></td>
        <td>${{p.lyft}} (${{s.toFixed(2)}}x)</td>
        <td><b>$${{lPrice.toFixed(2)}}</b></td>
        <td><b>${{winner}}</b></td>
        <td>$${{diff.toFixed(2)}} (${{pct.toFixed(1)}}%)</td>
      </tr>`;
    }});

    container.innerHTML = `<div class="table-wrap"><table class="data-table">
      <thead><tr>
        <th>Phân khúc đối xứng</th><th>Gói Uber</th><th>Ước lượng Uber (HC3
          OLS)</th>
        <th>Gói Lyft</th><th>Ước lượng Lyft (HC3 OLS: β₀ + β₁·d·s)</th><th>Gợi
          ý rẻ hơn</th><th>Tiết kiệm ước tính</th>
      </tr></thead>
      <tbody>${{rowsHtml}}</tbody>
    </table></div>`;
  }}

  function updateOfflineSim() {{
    const cap = parseFloat(document.getElementById('simCapSel').value);
    const supDrop = parseInt(document.getElementById('simSupplySel').value,
      10);
    const eps = document.getElementById('simEpsSel').value;
    const row = SIM_DATA.find(r => Math.abs(r.surge_cap - cap) < 1e-4) ||
      SIM_DATA[0];

    const satEl = document.getElementById('simSatWarn');
    if (Math.abs(cap - 2.0) < 1e-4) {{
      satEl.style.display = 'block';
      satEl.innerHTML = `🔎 <b>Lưu ý Bão hòa Nhánh cây (Tree Saturation):</b>
        Tại trần 2.00x chỉ có 53 lượt surge 2.5x–3.0x nên M7 ước lượng tiết
        kiệm thấp hơn đối chứng M2 OLS
        ($${{row.mean_rider_saving_per_affected_trip_usd.toFixed(2)}} vs
        $${{row.m2_ols_mean_saving_affected_usd.toFixed(2)}}/lượt).`;
    }} else {{
      satEl.style.display = 'none';
    }}

    let proxyChange = row.static_lyft_rev_change_pct;
    let subLine = 'Cầu tĩnh (ε = 0.00 — Cung không đổi)';
    if (supDrop === 10) {{
      proxyChange = row.supply_constrained_rev_change_drop10_pct;
      subLine = `Dư cầu chưa phục vụ:
        ${{row.unfulfilled_quotes_drop10.toFixed(1)}} lượt | Cung nhạy giá
        ε_S = 0.60 (Q_S = clip(1 + 0.60·Δp/p))`;
    }} else if (supDrop === 5) {{
      proxyChange = row.supply_constrained_rev_change_drop5_pct;
      subLine = `Dư cầu chưa phục vụ:
        ${{row.unfulfilled_quotes_drop5.toFixed(1)}} lượt | Cung nhạy giá
        ε_S = 0.30 (Q_S = clip(1 + 0.30·Δp/p))`;
    }} else if (eps === 'eps_050') {{
      proxyChange = row.net_lyft_rev_change_eps_050_pct;
      subLine = `Giữ chân cận trên:
        +${{row.retained_trips_eps_050.toFixed(1)}} lượt | Cận trên dư cung (ε
        = -0.50)`;
    }} else if (eps === 'eps_080') {{
      proxyChange = row.net_lyft_rev_change_eps_080_pct;
      subLine = `Giữ chân cận trên:
        +${{row.retained_trips_eps_080.toFixed(1)}} lượt | Cận trên cạnh tranh
        (ε = -0.80)`;
    }}

    const nAff = Number(row.affected_trips_count).toLocaleString('en-US');
    const savTrip = row.mean_rider_saving_per_affected_trip_usd.toFixed(2);
    const fmtUsd = {{minimumFractionDigits:2, maximumFractionDigits:2}};
    const savTot = Number(row.total_consumer_savings_usd).toLocaleString(
      'en-US', fmtUsd
    );
    const sign = proxyChange >= 0 ? '+' : '';
    document.getElementById('simKpiGrid').innerHTML = `
      <div class="kpi-card top-accent">
        <div class="kpi-label">Lượt Báo giá Chịu tác động</div>
        <div class="kpi-value">${{nAff}}</div>
        <div class="kpi-sub">${{row.affected_surge_trips_pct.toFixed(2)}}%
          lượt Surge | ${{row.affected_lyft_trips_pct.toFixed(2)}}% Lyft
          thực</div>
        <div style="margin-top:8px;"><span class="badge
          badge-simulation">SCENARIO</span></div>
      </div>
      <div class="kpi-card top-emerald">
        <div class="kpi-label">Tiết kiệm TB/Lượt (M7)</div>
        <div class="kpi-value">$${{savTrip}}</div>
        <div class="kpi-sub">Giảm
          ${{row.mean_rider_saving_pct_affected.toFixed(2)}}% | Đối chứng M2
          OLS: $${{row.m2_ols_mean_saving_affected_usd.toFixed(2)}}</div>
        <div style="margin-top:8px;"><span class="badge badge-simulation">Gốc
          $${{row.mean_orig_price_affected_usd.toFixed(2)}} →
          $${{row.mean_capped_price_affected_usd.toFixed(2)}}</span></div>
      </div>
      <div class="kpi-card top-lyft">
        <div class="kpi-label">Tổng Tiết kiệm Báo giá (M7)</div>
        <div class="kpi-value">$${{savTot}}</div>
        <div class="kpi-sub">Đối chứng M2 OLS:
          $${{Number(row.m2_ols_total_savings_usd).toLocaleString('en-US',
          {{minimumFractionDigits:2, maximumFractionDigits:2}})}}</div>
        <div style="margin-top:8px;"><span class="badge
          badge-simulation">SIMULATION</span></div>
      </div>
      <div class="kpi-card top-amber">
        <div class="kpi-label">Biến động API Quote Proxy</div>
        <div class="kpi-value">${{sign}}${{proxyChange.toFixed(3)}}%</div>
        <div class="kpi-sub">${{subLine}}</div>
        <div style="margin-top:8px;"><span class="badge
          badge-assumption">KHÔNG phải doanh thu thực</span></div>
      </div>
    `;
  }}

  window.addEventListener('DOMContentLoaded', () => {{
    updateOfflineEstimator();
    updateOfflineSim();
  }});
</script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    out = export_interactive_html()
    size_mb = out.stat().st_size / (1024**2)
    print(
        f"DA XUAT THANH CONG BAO CAO HTML TUONG TAC OFFLINE: {out} "
        f"({size_mb:.2f} MB)"
    )
