"""
UI & Chart Integration Tests cho Giai đoạn 4 (Tab 2 RQ1 & Tab 3 RQ2).
Kiểm tra:
1. Toàn bộ 6 hàm dựng biểu đồ Plotly của Tab 2 & Tab 3 chạy sạch trên cả dữ
liệu đầy đủ và dữ liệu rỗng.
2. Kiểm tra sống dashboard/app.py qua AppTest khi đổi bộ lọc giả thuyết (Tab 2)
và chọn cặp cạnh tranh
   UberPool vs Shared (Tab 3 -> hiển thị lưu ý HIGH-02 không ngoại suy, 0 ngoại
   lệ).
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from dashboard.business_logic import (
    compute_all_pair_comparisons,
    compute_binned_empirical_tariff,
    compute_dayofweek_est_profile,
    compute_hourly_est_profile,
    compute_surge_distribution,
    compute_weather_rain_comparison,
)
from dashboard.charts import (
    build_dayofweek_est_fig,
    build_hourly_est_fig,
    build_surge_levels_fig,
    build_tariff_crossing_fig,
    build_tariff_strategy_scatter_fig,
    build_weather_rain_audit_fig,
)
from dashboard.config import DASHBOARD_DIR
from dashboard.data_loader import load_data_bundle


def test_stage4_charts_build_for_full_and_empty_data() -> None:
    """
    Kiểm tra 6 hàm biểu đồ Plotly Tab 2 & Tab 3 trên dữ liệu đầy đủ và dữ liệu
    rỗng.
    """
    bundle = load_data_bundle()
    master = bundle["master"]
    tariffs = bundle["tables"]["tier_tariffs"]

    w_df, w_meta = compute_weather_rain_comparison(master)
    assert len(build_weather_rain_audit_fig(w_df, w_meta).data) >= 3

    h_df, h_meta = compute_hourly_est_profile(master)
    assert len(build_hourly_est_fig(h_df, h_meta).data) == 3

    dow_df = compute_dayofweek_est_profile(master)
    assert len(build_dayofweek_est_fig(dow_df).data) == 2

    surge_df = compute_surge_distribution(master)
    assert len(build_surge_levels_fig(surge_df).data) == 2

    pairs = compute_all_pair_comparisons(tariffs, master)
    for _, row in pairs.iterrows():
        pair_info = row.to_dict()
        binned = compute_binned_empirical_tariff(
            master, pair_info["uber_tier"], pair_info["lyft_tier"]
        )
        fig_c = build_tariff_crossing_fig(tariffs, pair_info, binned_df=binned)
        assert len(fig_c.data) >= 2

    fig_s = build_tariff_strategy_scatter_fig(tariffs)
    assert len(fig_s.data) == 7  # 5 đường nối cặp + 2 nhóm điểm Uber/Lyft

    # Kiểm tra an toàn với dữ liệu rỗng
    empty = master.iloc[0:0]
    ew_df, ew_meta = compute_weather_rain_comparison(empty)
    eh_df, eh_meta = compute_hourly_est_profile(empty)
    assert build_weather_rain_audit_fig(ew_df, ew_meta) is not None
    assert build_hourly_est_fig(eh_df, eh_meta) is not None
    assert (
        build_dayofweek_est_fig(compute_dayofweek_est_profile(empty))
        is not None
    )
    assert (
        build_surge_levels_fig(compute_surge_distribution(empty)) is not None
    )


def test_streamlit_stage4_tabs_interactive_and_high02_notice() -> None:
    """
    Chạy sống AppTest tương tác trên Tab 2 (RQ1) và Tab 3 (RQ2 — kiểm chứng
    HIGH-02).
    """
    at = AppTest.from_file(str(DASHBOARD_DIR / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.error) == 0

    # Đổi bộ lọc kiểm định giả thuyết trên Tab 2 sang RQ1 rồi RQ2
    at.radio(key="tab2_hyp_filter").set_value("RQ1").run()
    assert not at.exception
    at.radio(key="tab2_hyp_filter").set_value("RQ2").run()
    assert not at.exception

    # Chuyển cặp biểu giá trên Tab 3 sang '1. Đi ghép Tiết kiệm (Pool /
    # Shared)' -> xác nhận thông báo khóa miền không ngoại suy
    at.selectbox(key="tab3_selected_pair").set_value(
        "1. Đi ghép Tiết kiệm (Pool / Shared)"
    ).run()
    assert not at.exception
    info_texts = " ".join(box.value for box in at.info)
    assert (
        "không ngoại suy" in info_texts.lower()
        and "KHÔNG cắt nhau" in info_texts
    )
