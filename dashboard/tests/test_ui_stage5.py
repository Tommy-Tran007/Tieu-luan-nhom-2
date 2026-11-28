"""
UI & Chart Integration Tests cho Giai đoạn 5 (Tab 4 — Machine Learning,
Residual Diagnostics & XAI).
Kiểm tra:
1. Toàn bộ 6 hàm dựng biểu đồ Tab 4 chạy sạch và đúng số lượng trace/nhóm.
2. Chạy sống dashboard/app.py qua AppTest tương tác trên Tab 4: đổi M7 <-> M2 ở
chẩn đoán phần dư,
   đổi chiều phân rã sai số (service_tier -> surge_regime), tắt/bật M2 trên
   Permutation Importance.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from dashboard.business_logic import (
    prepare_model_leaderboard,
    prepare_segment_errors,
)
from dashboard.charts import (
    build_actual_vs_predicted_and_residuals_fig,
    build_model_arena_fig,
    build_pdp_curves_fig,
    build_permutation_importance_fig,
    build_prediction_interval_fig,
    build_segment_errors_fig,
)
from dashboard.config import DASHBOARD_DIR
from dashboard.data_loader import load_data_bundle
from dashboard.model_loader import (
    evaluate_models_on_test_set,
    load_trained_models,
)


def test_stage5_ml_and_xai_charts_build_cleanly() -> None:
    """Kiểm tra dựng đủ 6 biểu đồ Plotly Tab 4 từ dữ liệu & mô hình thật."""
    bundle = load_data_bundle()
    models = load_trained_models()
    eval_res = evaluate_models_on_test_set(
        models, bundle["master"], plot_sample_n=1_500
    )

    lb = prepare_model_leaderboard(bundle["tables"]["test_metrics"])
    assert len(build_model_arena_fig(lb).data) == 2

    for mk in ("M7", "M2"):
        fig_r = build_actual_vs_predicted_and_residuals_fig(
            eval_res, model_key=mk
        )
        assert len(fig_r.data) == 4  # Uber + Lyft + y=x + Residual histogram

    for dim in (
        "service_tier",
        "distance_bucket",
        "surge_regime",
        "source_district",
    ):
        seg = prepare_segment_errors(bundle["tables"]["segment_errors"], dim)
        assert len(build_segment_errors_fig(seg, dim).data) == 3

    fig_perm_both = build_permutation_importance_fig(
        bundle["tables"]["permutation_importance"], show_m2=True
    )
    assert len(fig_perm_both.data) == 2
    fig_perm_m7 = build_permutation_importance_fig(
        bundle["tables"]["permutation_importance"], show_m2=False
    )
    assert len(fig_perm_m7.data) == 1

    assert len(build_pdp_curves_fig(bundle["summaries"]["stage6"]).data) == 4
    assert (
        len(
            build_prediction_interval_fig(
                bundle["tables"]["prediction_interval"]
            ).data
        )
        == 3
    )


def test_streamlit_stage5_tab4_interactions() -> None:
    """
    Chạy sống AppTest kiểm tra tương tác trên Tab 4 (M7/M2 radio, chiều phân rã
    sai số, checkbox M2 XAI).
    """
    at = AppTest.from_file(str(DASHBOARD_DIR / "app.py"), default_timeout=90)
    at.run()
    assert not at.exception
    assert len(at.error) == 0

    at.radio(key="tab4_diag_model").set_value("M2").run()
    assert not at.exception

    at.selectbox(key="tab4_seg_dimension").set_value(
        "Chế độ Surge (4 surge_regime)"
    ).run()
    assert not at.exception

    at.checkbox(key="tab4_perm_m2").set_value(False).run()
    assert not at.exception
