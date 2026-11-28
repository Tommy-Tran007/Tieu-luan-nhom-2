"""
Script: dashboard/app.py
Mục đích: Điểm vào chính (Presentation Orchestrator) cho Web Dashboard
Streamlit + Plotly
của đề tài PRJ-EXT-01 — Uber vs. Lyft: Boston Rideshare Pricing Analysis.

Kiến trúc tách lớp nghiêm ngặt:
- Data Layer      : dashboard/data_loader.py & dashboard/validation.py (cache
  bằng @st.cache_data)
- Business Logic  : dashboard/business_logic.py (không đặt công thức tính KPI
  trong app.py)
- Model Layer     : dashboard/model_loader.py (cache bằng @st.cache_resource)
- Visualization   : dashboard/charts.py (dùng chung với export_static_html.py)
- UI Components   : dashboard/components/{styles, sidebar, kpi_cards, tab_*.py}

Lệnh chạy chuẩn (từ thư mục gốc dự án 2521001060_Nhom02_TieuLuan/):
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Bảo đảm thư mục gốc dự án nằm trong sys.path dù chạy từ bất kỳ thư mục làm
# việc nào
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from dashboard.business_logic import (  # noqa: E402
    FilterParams,
    FilterResult,
    apply_global_filters,
    compute_data_quality_summary,
    compute_executive_kpis,
    compute_route_catalog,
    compute_weather_reference_medians,
)
from dashboard.components.sidebar import render_sidebar  # noqa: E402
from dashboard.components.styles import inject_custom_css  # noqa: E402
from dashboard.components.tab_models_rq3 import render_tab_models_rq3  # noqa: E402
from dashboard.components.tab_overview import render_tab_overview  # noqa: E402
from dashboard.components.tab_pricing_rq1 import render_tab_pricing_rq1  # noqa: E402
from dashboard.components.tab_simulator_rq4 import render_tab_simulator_rq4  # noqa: E402
from dashboard.components.tab_tariff_rq2 import render_tab_tariff_rq2  # noqa: E402
from dashboard.data_loader import DataLoadError, load_data_bundle  # noqa: E402
from dashboard.model_loader import (  # noqa: E402
    ModelLoadError,
    evaluate_models_on_test_set,
    load_trained_models,
)
from dashboard.validation import (  # noqa: E402
    DataValidationError,
    ValidationReport,
    validate_master_dataset,
    validate_result_tables,
)


@st.cache_resource(
    show_spinner="Đang nạp mô hình M7 & M2 đã huấn luyện (không train lại)..."
)
def get_cached_models_and_test_eval(
    _master_df: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Nạp M7 & M2 từ models/*.joblib và đánh giá xác minh sống trên 46,573 dòng
    tập Test 1 lần duy nhất.
    """
    models = load_trained_models()
    eval_res = evaluate_models_on_test_set(
        models, _master_df, plot_sample_n=3_500
    )
    return models, eval_res


@st.cache_resource(
    show_spinner="Đang nạp bộ dữ liệu sạch Parquet & 14 bảng kiểm toán..."
)
def get_cached_data_and_validation() -> tuple[
    dict[str, Any],
    ValidationReport,
    dict[str, Any],
    dict[str, Any],
    Any,
    dict[str, float],
]:
    """
    Nạp, kiểm toán và tính sẵn bộ chỉ số toàn tập (read-only singleton) 1 lần
    duy nhất.
    Dùng @st.cache_resource thay vì @st.cache_data để không tốn chi phí nhân
    bản/pickle 41 MB RAM trên mỗi thao tác lọc.
    """
    bundle = load_data_bundle()
    master_rep = validate_master_dataset(bundle["master"])
    tables_rep = validate_result_tables(bundle["tables"])
    combined = ValidationReport(
        checks=master_rep.checks + tables_rep.checks,
        metrics=master_rep.metrics,
    )
    dq_summary = compute_data_quality_summary(
        bundle["master"],
        bundle["summaries"]["stage3"],
        bundle["summaries"]["stage4"],
        ram_mb=float(bundle["ram_mb"]),
        est_raw_ram_mb=float(bundle["est_unoptimized_ram_mb"]),
    )
    full_kpis = compute_executive_kpis(
        bundle["master"],
        bundle["tables"]["test_metrics"],
        full_dataset_rows=len(bundle["master"]),
    )
    route_catalog = compute_route_catalog(bundle["master"])
    weather_ref = compute_weather_reference_medians(bundle["master"])
    return bundle, combined, dq_summary, full_kpis, route_catalog, weather_ref


@st.cache_data(max_entries=16, show_spinner=False)
def get_cached_filter_result(
    _master_df: Any, params: FilterParams
) -> FilterResult:
    """
    Cache kết quả lọc theo bộ tham số Sidebar (FilterParams bất biến/hashable)
    để chuyển Tab/widget cực nhanh.
    """
    return apply_global_filters(_master_df, params)


def render_hero_header(
    total_rows: int, n_passed: int, n_total_checks: int
) -> None:
    """Hiển thị phần đầu trang chuẩn học thuật & quản trị."""
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">
                Uber vs. Lyft — Boston Rideshare Dynamic Pricing &amp;
                Fare Analytics
            </div>
            <p class="hero-subtitle">
                Đề tài <b>PRJ-EXT-01</b> • Học phần Lập trình Python cho
                Khoa học Dữ liệu (UFM) •
                Nhóm 02: <b>Trần Thanh Huy</b> (2521001060) &amp;
                <b>Lê Huy Thắng</b> (2521001092) — Lớp 25DDS01
            </p>
            <div class="hero-badges">
                <span class="badge-pill badge-verified">
                    VERIFIED • {total_rows:,} API Fare Quotes
                </span>
                <span class="badge-pill badge-verified">
                    Data Gate: {n_passed}/{n_total_checks} PASS
                </span>
                <span class="badge-pill badge-dark">
                    0 Nhãn mưa giả &amp; Tách Sentinel
                </span>
                <span class="badge-pill badge-dark">Múi giờ Boston EST</span>
                <span class="badge-pill badge-dark">
                    Bảo toàn nguồn gốc Surge
                </span>
                <span class="badge-pill badge-dark">
                    Interactive Vector Plotly
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title=(
            "Uber vs. Lyft Boston — Rideshare Pricing Dashboard (PRJ-EXT-01)"
        ),
        page_icon="🚖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    # 1. Nạp dữ liệu, mô hình & chạy cổng kiểm toán tự động
    try:
        (
            bundle,
            val_report,
            dq_summary,
            full_kpis,
            route_catalog,
            weather_ref,
        ) = get_cached_data_and_validation()
        val_report.raise_if_critical()
        models_info, eval_res = get_cached_models_and_test_eval(
            bundle["master"]
        )
    except (DataLoadError, DataValidationError, ModelLoadError) as exc:
        st.error(
            f"❌ **Lỗi Nạp / Kiểm toán Dữ liệu hoặc Mô hình CRITICAL:** {exc}"
        )
        st.stop()

    master_df = bundle["master"]
    tables = bundle["tables"]
    summaries = bundle["summaries"]
    v5_data = bundle.get("v5", {})
    ram_mb = float(bundle["ram_mb"])

    # 2. Thanh lọc toàn cục bên trái (Sidebar)
    filter_params = render_sidebar(total_rows=len(master_df), ram_mb=ram_mb)
    filter_res = get_cached_filter_result(master_df, filter_params)

    # 3. Hero Banner
    n_passed = len(val_report.checks) - len(val_report.failed_checks)
    render_hero_header(
        total_rows=len(master_df),
        n_passed=n_passed,
        n_total_checks=len(val_report.checks),
    )

    # 4. Hệ thống 5 Tab theo lộ trình nghiên cứu RQ1–RQ4
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 1. Executive Overview",
            "🌦️ 2. Pricing, Weather & Time (RQ1)",
            "⚔️ 3. Competitive Pricing (RQ2)",
            "🤖 4. ML Arena & XAI (RQ3)",
            "🎯 5. Predictor & Policy Simulator (RQ4)",
        ]
    )

    with tab1:
        render_tab_overview(
            filter_res=filter_res,
            full_kpis=full_kpis,
            dq=dq_summary,
            tables=tables,
            validation_report=val_report,
            v5=v5_data,
            summaries=summaries,
        )

    with tab2:
        render_tab_pricing_rq1(
            filter_res=filter_res,
            tables=tables,
            v5=v5_data,
        )

    with tab3:
        render_tab_tariff_rq2(
            filter_res=filter_res,
            master_full=master_df,
            tables=tables,
            v5=v5_data,
        )

    with tab4:
        render_tab_models_rq3(
            tables=tables,
            summaries=summaries,
            models_info=models_info,
            eval_res=eval_res,
            v5=v5_data,
        )

    with tab5:
        render_tab_simulator_rq4(
            tables=tables,
            summaries=summaries,
            models_info=models_info,
            route_catalog=route_catalog,
            weather_ref=weather_ref,
            v5=v5_data,
        )


if __name__ == "__main__":
    main()
