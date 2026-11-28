"""
UI & Chart Integration Tests cho Giai đoạn 3 (charts.py + dashboard/app.py qua
Streamlit AppTest).
Kiểm tra:
1. 3 hàm dựng biểu đồ Tab 1 (Histogram, Boxplot 12 gói, Heatmap 72 tuyến) hoạt
động với cả dữ liệu đầy đủ và dữ liệu rỗng (0 dòng).
2. Chạy sống toàn bộ dashboard/app.py bằng Streamlit AppTest không phát sinh
bất kỳ ngoại lệ nào.
3. Kiểm thử tương tác bộ lọc Sidebar: bật surge_only + chọn riêng Uber -> giao
diện hiển thị cảnh báo thân thiện, 0 crash.
4. Kiểm thử đổi chỉ số Heatmap trên Tab 1.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from dashboard.business_logic import (
    compute_executive_kpis,
    compute_price_histogram_bins,
    compute_route_catalog,
    compute_tier_boxplot_stats,
)
from dashboard.charts import (
    build_executive_uber_vs_lyft_fig,
    build_price_distribution_fig,
    build_route_heatmap_fig,
    build_tier_boxplot_fig,
)
from dashboard.config import DASHBOARD_DIR
from dashboard.data_loader import load_data_bundle


def test_tab1_plotly_charts_build_for_full_and_empty_data() -> None:
    """
    Kiểm tra dựng biểu đồ chủ đạo Executive + 3 biểu đồ khám phá Tab 1 trên dữ
    liệu thật lẫn rỗng.
    """
    bundle = load_data_bundle()
    master = bundle["master"]
    test_metrics = bundle["tables"]["test_metrics"]
    tariffs = bundle["tables"]["tier_tariffs"]

    fig_hero = build_executive_uber_vs_lyft_fig(master, tariffs)
    assert (
        len(fig_hero.data) >= 4
        # Bar Uber + Bar Lyft + 2 vùng chênh lệch + đường Lyft-Uber + điểm cắt
        # d*
    )

    kpis = compute_executive_kpis(master, test_metrics)
    fig_dist = build_price_distribution_fig(
        compute_price_histogram_bins(master), kpis
    )
    assert len(fig_dist.data) == 2  # Uber & Lyft

    fig_box = build_tier_boxplot_fig(compute_tier_boxplot_stats(master))
    assert len(fig_box.data) == 4  # 2 Box traces + 2 Mean marker traces

    routes = compute_route_catalog(master)
    for metric in (
        "mean_price_usd",
        "median_distance_miles",
        "surge_rate_pct",
    ):
        fig_heat = build_route_heatmap_fig(routes, metric=metric)
        assert len(fig_heat.data) == 1

    # Dữ liệu rỗng không được ném lỗi
    empty_df = master.iloc[0:0]
    empty_kpis = compute_executive_kpis(empty_df, test_metrics)
    assert build_executive_uber_vs_lyft_fig(empty_df, tariffs) is not None
    assert (
        build_price_distribution_fig(
            compute_price_histogram_bins(empty_df), empty_kpis
        )
        is not None
    )
    assert (
        build_tier_boxplot_fig(compute_tier_boxplot_stats(empty_df))
        is not None
    )
    assert build_route_heatmap_fig(compute_route_catalog(empty_df)) is not None


def test_streamlit_app_renders_cleanly_and_handles_empty_filter() -> None:
    """
    Chạy sống dashboard/app.py qua AppTest và thử nghiệm tương tác Sidebar +
    Radio Tab 1.
    """
    app_path = str(DASHBOARD_DIR / "app.py")
    at = AppTest.from_file(app_path, default_timeout=60)
    at.run()

    assert not at.exception, (
        f"Streamlit app phát sinh lỗi runtime: {at.exception}"
    )
    assert len(at.tabs) == 5
    # Đảm bảo không có st.error nào trên màn hình mặc định
    assert len(at.error) == 0

    # Đổi chỉ số Heatmap trên Tab 1 sang Tỷ lệ Surge
    at.radio(key="tab1_heatmap_metric").set_value(
        "Tỷ lệ kích hoạt Surge (%)"
    ).run()
    assert not at.exception

    # Thử tổ hợp lọc rỗng: chỉ chọn Uber + bật Chỉ xem Surge > 1.0x
    at.sidebar.multiselect(key="flt_cabs").set_value(["Uber"]).run()
    assert not at.exception
    at.sidebar.toggle(key="flt_surge_only").set_value(True).run()
    assert not at.exception
    assert len(at.warning) >= 1, (
        "Khi lọc ra 0 dòng phải hiển thị cảnh báo thân thiện."
    )
