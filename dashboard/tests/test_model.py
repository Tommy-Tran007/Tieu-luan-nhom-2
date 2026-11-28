"""
Unit & Reproducibility Tests cho dashboard/model_loader.py (Giai đoạn 2).
Kiểm tra:
1. Nạp an toàn M7 và M2 từ bất kỳ thư mục làm việc nào (khắc phục RSK-06
sys.path).
2. Xác nhận 27 đặc trưng đúng thứ tự và 0 biến rò rỉ trong FORBIDDEN_IN_X.
3. Kiểm tra suy luận trực tiếp 12 gói xe (tính nhất quán kinh tế: < 3.21 dặm
Đặc tả A / 5.17 dặm Đặc tả B Lyft rẻ hơn UberX ở 1.0x,
   khi Lyft surge = 1.75x thì UberX rẻ hơn).
4. Kiểm tra bắt buộc cảnh báo EXTRAPOLATION WARNING khi nhập distance > 7.86
dặm hoặc tuyến ngoài 72 tuyến Boston.
5. Kiểm chứng đánh giá sống M7 & M2 trên 46,573 dòng tập Test
(is_surge_original == 1) khớp chính xác stage5_test_metrics.csv.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.business_logic import (
    compute_route_catalog,
    compute_weather_reference_medians,
)
from dashboard.data_loader import load_data_bundle
from dashboard.model_loader import (
    EXPECTED_MODEL_FEATURES,
    ModelLoadError,
    evaluate_models_on_test_set,
    load_trained_models,
    predict_fares_all_tiers,
    validate_prediction_inputs,
)


@pytest.fixture(scope="module")
def bundle() -> dict:
    return load_data_bundle()


@pytest.fixture(scope="module")
def models() -> dict:
    return load_trained_models()


def test_load_models_works_from_different_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Đảm bảo nạp best_model.joblib và m2_interaction_ols.joblib thành công dù
    chạy từ thư mục khác.
    """
    monkeypatch.chdir(tmp_path)
    loaded = load_trained_models()
    assert loaded["n_features"] == 27
    assert loaded["feature_names"] == EXPECTED_MODEL_FEATURES
    assert loaded["has_quantile_joblib"] is False


def test_missing_model_file_raises_model_load_error(tmp_path: Path) -> None:
    """Kiểm thử âm tính: thiếu file .joblib phải ném ModelLoadError rõ ràng."""
    with pytest.raises(ModelLoadError, match="Không tìm thấy tệp mô hình"):
        load_trained_models(m7_path=tmp_path / "missing_m7.joblib")


def test_live_prediction_in_domain_economic_consistency(
    bundle: dict, models: dict
) -> None:
    """
    Kiểm tra suy luận trực tiếp 12 gói trên tuyến thật Back Bay -> South
    Station (1.51 dặm).
    """
    master = bundle["master"]
    pi_df = bundle["tables"]["prediction_interval"]
    routes = compute_route_catalog(master)
    w_ref = compute_weather_reference_medians(master)

    # 1. Ở cự ly ngắn 1.51 dặm (< d* = 3.21 / 5.17 dặm) & không surge: Lyft
    # phải rẻ hơn UberX
    res_no_surge = predict_fares_all_tiers(
        models=models,
        pi_df=pi_df,
        route_catalog=routes,
        weather_ref=w_ref,
        source="Back Bay",
        destination="South Station",
        distance=1.51,
        lyft_surge=1.0,
        hour=8,
        day_of_week=2,
        is_raining=False,
        temp_f=40.0,
    )
    assert res_no_surge["input_check"].is_valid
    assert not res_no_surge["input_check"].is_extrapolation
    assert len(res_no_surge["all_tiers_df"]) == 12
    assert len(res_no_surge["pairs_df"]) == 5
    assert len(res_no_surge["standalone_df"]) == 2

    std_no_surge = res_no_surge["pairs_df"].iloc[1]
    assert (
        std_no_surge["Giá báo Lyft (M7, USD)"]
        < std_no_surge["Giá báo Uber (M7, USD)"]
    )

    # 2. Khi Lyft bật surge 1.75x trên cùng lộ trình: giá báo Lyft phải tăng
    # mạnh và UberX trở nên rẻ hơn
    res_surge = predict_fares_all_tiers(
        models=models,
        pi_df=pi_df,
        route_catalog=routes,
        weather_ref=w_ref,
        source="Back Bay",
        destination="South Station",
        distance=1.51,
        lyft_surge=1.75,
        hour=8,
        day_of_week=2,
        is_raining=False,
        temp_f=40.0,
    )
    std_surge = res_surge["pairs_df"].iloc[1]
    assert (
        std_surge["Giá báo Lyft (M7, USD)"]
        > std_no_surge["Giá báo Lyft (M7, USD)"] + 3.0
    )
    assert "Uber" in std_surge["Hãng rẻ hơn (theo M7)"]


def test_extrapolation_warnings_and_hard_input_errors(bundle: dict) -> None:
    """
    Kiểm tra kích hoạt cảnh báo ngoại suy khi distance > 7.86 dặm hoặc tuyến
    ngoài 72 tuyến Boston.
    """
    routes = compute_route_catalog(bundle["master"])

    # Ngoại suy quãng đường 15 dặm
    chk_far = validate_prediction_inputs(
        "Back Bay",
        "South Station",
        distance=15.0,
        lyft_surge=1.0,
        hour=9,
        day_of_week=1,
        route_catalog=routes,
    )
    assert chk_far.is_valid
    assert chk_far.is_extrapolation
    assert any("EXTRAPOLATION WARNING" in w for w in chk_far.warnings)

    # Tuyến không tồn tại trong đồ thị 72 tuyến (Back Bay -> Beacon Hill không
    # kết nối trực tiếp trong dữ liệu)
    chk_route = validate_prediction_inputs(
        "Back Bay",
        "Beacon Hill",
        distance=1.5,
        lyft_surge=1.0,
        hour=9,
        day_of_week=1,
        route_catalog=routes,
    )
    assert chk_route.is_valid
    assert not chk_route.route_observed_in_boston
    assert chk_route.is_extrapolation

    # Lỗi cứng: trùng điểm đón/trả và cự ly âm
    chk_bad = validate_prediction_inputs(
        "Back Bay",
        "Back Bay",
        distance=-1.0,
        lyft_surge=0.5,
        hour=25,
        day_of_week=9,
        route_catalog=routes,
    )
    assert not chk_bad.is_valid
    assert len(chk_bad.errors) >= 4


def test_live_test_set_evaluation_reproduces_stage5_metrics_exact(
    bundle: dict, models: dict
) -> None:
    """
    Xác nhận đánh giá sống M7 & M2 trên 46,573 dòng tập Test
    (is_surge_original==1) khớp chính xác stage5_test_metrics.csv.
    """
    eval_res = evaluate_models_on_test_set(
        models, bundle["master"], plot_sample_n=2_000
    )
    m7 = eval_res["metrics_m7"]
    m2 = eval_res["metrics_m2"]

    assert m7["n_test"] == 46_573
    assert m7["r2"] == 0.9626
    assert m7["rmse_usd"] == 1.7830
    assert m7["mae_usd"] == 1.1338
    assert m7["mape_pct"] == 8.19

    assert m2["r2"] == 0.9539
    assert m2["rmse_usd"] == 1.9802
    assert m2["mae_usd"] == 1.2888
    assert m2["mape_pct"] == 9.33

    assert len(eval_res["plot_sample_df"]) == 2_000
    assert not eval_res["resid_hist_df"].empty
