"""
UI & Chart Integration Tests cho Giai đoạn 6 (Tab 5 — Live Fare Predictor &
Surge Cap Simulator).
Kiểm tra:
1. 3 hàm biểu đồ Plotly Tab 5 dựng sạch trên dữ liệu & mô hình thật.
2. Tương tác sống AppTest trên Tab 5:
   - Dự báo trong miền (lyft_surge = 1.75x) -> 0 lỗi runtime.
   - Bật chế độ nhập tự do + nhập distance = 12.5 dặm -> kích hoạt rõ
     EXTRAPOLATION WARNING.
   - Chuyển Surge Cap sang 2.00x -> hiển thị cảnh báo Tree Saturation; đổi
     elasticity sang eps_080.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from dashboard.business_logic import (
    compute_route_catalog,
    compute_weather_reference_medians,
)
from dashboard.charts import (
    build_live_fare_comparison_fig,
    build_surge_cap_m7_vs_m2_fig,
    build_surge_cap_tradeoff_fig,
)
from dashboard.config import DASHBOARD_DIR
from dashboard.data_loader import load_data_bundle
from dashboard.model_loader import load_trained_models, predict_fares_all_tiers


def test_stage6_predictor_and_simulator_charts_build_cleanly() -> None:
    """Kiểm tra 3 hàm biểu đồ Plotly Tab 5."""
    bundle = load_data_bundle()
    models = load_trained_models()
    routes = compute_route_catalog(bundle["master"])
    w_ref = compute_weather_reference_medians(bundle["master"])

    pred = predict_fares_all_tiers(
        models=models,
        pi_df=bundle["tables"]["prediction_interval"],
        route_catalog=routes,
        weather_ref=w_ref,
        source="Back Bay",
        destination="South Station",
        distance=1.51,
        lyft_surge=1.5,
    )
    fig_live = build_live_fare_comparison_fig(
        pred["pairs_df"], "Back Bay → South Station", 1.51, 1.5
    )
    assert len(fig_live.data) == 2

    sim_df = bundle["tables"]["surge_cap_simulation"]
    for eps in ("static_000", "eps_050", "eps_080"):
        assert (
            len(
                build_surge_cap_tradeoff_fig(
                    sim_df,
                    selected_cap=2.00,
                    elasticity_mode=eps,
                    supply_drop_pct=0,
                ).data
            )
            == 1
        )
    for s_drop in (5, 10):
        assert (
            len(
                build_surge_cap_tradeoff_fig(
                    sim_df,
                    selected_cap=2.00,
                    elasticity_mode="static_000",
                    supply_drop_pct=s_drop,
                ).data
            )
            == 1
        )
    assert len(build_surge_cap_m7_vs_m2_fig(sim_df).data) == 2


def test_streamlit_stage6_tab5_predictor_extrapolation_and_simulator() -> None:
    """
    Chạy sống AppTest kiểm tra Tab 5A (Live Predictor + Extrapolation Warning)
    & Tab 5B (Surge Cap Simulator).
    """
    at = AppTest.from_file(str(DASHBOARD_DIR / "app.py"), default_timeout=90)
    at.run()
    assert not at.exception
    assert len(at.error) == 0

    # 1. Đổi Lyft Surge sang 1.75x ở chế độ trong miền
    at.select_slider(key="tab5_lyft_surge").set_value(1.75).run()
    assert not at.exception
    assert len(at.error) == 0

    # 2. Bật chế độ kiểm thử ngoại suy & nhập quãng đường 12.5 dặm (> 7.86 dặm)
    # -> phải hiện EXTRAPOLATION WARNING
    at.toggle(key="tab5_allow_extrap").set_value(True).run()
    assert not at.exception
    at.number_input(key="tab5_dist_free").set_value(12.5).run()
    assert not at.exception
    err_texts = " ".join(e.value for e in at.error)
    assert "EXTRAPOLATION WARNING" in err_texts

    # 3. Chuyển trần Surge Cap sang 2.00x (phải hiện cảnh báo Tree Saturation)
    # & kiểm thử ràng buộc cung nhạy giá ε_S = 0.30 / 0.60
    at.select_slider(key="tab5_surge_cap").set_value(2.0).run()
    assert not at.exception
    info_texts = " ".join(box.value for box in at.info)
    assert "Tree Saturation" in info_texts

    at.radio(key="tab5_supply_drop").set_value(5).run()
    assert not at.exception
    at.radio(key="tab5_supply_drop").set_value(10).run()
    assert not at.exception
    at.radio(key="tab5_elasticity").set_value("eps_080").run()
    assert not at.exception
