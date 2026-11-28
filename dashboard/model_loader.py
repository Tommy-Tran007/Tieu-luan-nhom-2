"""
Module: dashboard/model_loader.py
Mục đích: Tầng Nạp Mô hình & Suy luận An toàn (ML Model & Inference Layer) cho
Dashboard PRJ-EXT-01.
- Tự động gắn src/ vào sys.path bằng pathlib để giải mã (unpickle) an toàn cả
  M7 và M2 từ mọi thư mục làm việc.
- Xác minh mô hình đã lưu (0 biến rò rỉ trong FORBIDDEN_IN_X, đủ 27 đặc trưng
  đúng thứ tự).
- Không huấn luyện lại (retrain) mô hình; sử dụng nguyên bản best_model.joblib
  và m2_interaction_ols.joblib.
- Kiểm tra chặt chẽ miền giá trị đầu vào và phát cảnh báo EXTRAPOLATION WARNING
  nếu vượt miền [0.02, 7.86] dặm
  hoặc chọn cặp tuyến chưa từng xuất hiện trong 72 tuyến Boston.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from dashboard.config import (
    ALL_TIERS_ORDERED,
    EMPIRICAL_DISTANCE_MAX,
    EMPIRICAL_DISTANCE_MIN,
    MODEL_M2_PATH,
    MODEL_M7_PATH,
    SRC_DIR,
    STANDALONE_TIERS,
    SYMMETRIC_PAIRS,
    VALID_BOSTON_DISTRICTS,
)

# Đảm bảo src/ nằm trong sys.path trước khi import config/features và unpickle
# M2 (chứa EconomicInteractionPreprocessor)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    BASE_CAT_FEATURES,
    BASE_NUM_FEATURES,
    ENGINEERED_NUM_FEATURES,
    FORBIDDEN_IN_X,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)
from features import engineer_features_stage4  # noqa: E402

EXPECTED_MODEL_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys(
        BASE_NUM_FEATURES + BASE_CAT_FEATURES + ENGINEERED_NUM_FEATURES
    )
)


class ModelLoadError(RuntimeError):
    """
    Ngoại lệ khi thiếu tệp mô hình .joblib hoặc mô hình vi phạm hợp đồng đặc
    trưng / chống rò rỉ.
    """


@dataclass
class InputValidationResult:
    """Kết quả kiểm tra tham số đầu vào dự báo của người dùng tại Tab 5."""

    is_valid: bool
    is_extrapolation: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    route_observed_in_boston: bool = True
    route_median_distance: float | None = None


def load_trained_models(
    m7_path: Path = MODEL_M7_PATH,
    m2_path: Path = MODEL_M2_PATH,
) -> dict[str, Any]:
    """
    Nạp và kiểm định an toàn 2 mô hình đã huấn luyện (M7 HistGBR & M2
    Interaction OLS).
    Xác nhận đủ 27 đặc trưng đúng thứ tự và 0 biến rò rỉ (FORBIDDEN_IN_X).
    """
    for path, label in (
        (m7_path, "M7 HistGradientBoosting"),
        (m2_path, "M2 Interaction OLS"),
    ):
        if not path.exists() or path.stat().st_size == 0:
            raise ModelLoadError(
                f"Không tìm thấy tệp mô hình {label} tại {path}. "
                "Vui lòng chạy 'python main.py --stage model' trước khi "
                "khởi chạy Dashboard."
            )

    try:
        model_m7 = joblib.load(m7_path)
        model_m2 = joblib.load(m2_path)
    except Exception as exc:
        raise ModelLoadError(
            f"Lỗi khi giải mã tệp mô hình .joblib: {exc}"
        ) from exc

    # Kiểm tra hợp đồng đặc trưng & chống rò rỉ trên M7
    m7_features = tuple(getattr(model_m7, "feature_names_in_", ()))
    if len(m7_features) != len(EXPECTED_MODEL_FEATURES):
        raise ModelLoadError(
            f"Mô hình M7 mong đợi {len(m7_features)} đặc trưng, không khớp "
            f"chuẩn {len(EXPECTED_MODEL_FEATURES)} đặc trưng."
        )
    if set(m7_features) != set(EXPECTED_MODEL_FEATURES):
        raise ModelLoadError(
            "Danh sách đặc trưng M7 sai lệch: "
            f"thiếu={set(EXPECTED_MODEL_FEATURES) - set(m7_features)}, "
            f"dư={set(m7_features) - set(EXPECTED_MODEL_FEATURES)}."
        )
    leak_overlap = set(m7_features) & FORBIDDEN_IN_X
    if leak_overlap:
        raise ModelLoadError(
            "CẢNH BÁO RÒ RỈ: Mô hình M7 chứa biến cấm trong FORBIDDEN_IN_X:"
            f" {leak_overlap}"
        )

    return {
        "M7": model_m7,
        "M2": model_m2,
        "feature_names": m7_features,
        "n_features": len(m7_features),
        # Minh bạch GAP-MODEL-01: không có file .joblib cho Quantile P10/P90
        "has_quantile_joblib": False,
    }


def hour_to_time_of_day(hour: int) -> str:
    """Quy đổi giờ Boston EST (0–23) sang đúng 4 buổi như pipeline GĐ3."""
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 22:
        return "Evening"
    return "Night"


def validate_prediction_inputs(
    source: str,
    destination: str,
    distance: float,
    lyft_surge: float,
    hour: int,
    day_of_week: int,
    route_catalog: pd.DataFrame | None = None,
) -> InputValidationResult:
    """
    Kiểm tra tính hợp lệ và nguy cơ ngoại suy (Extrapolation) cho dữ liệu đầu
    vào Tab 5.
    - Chặn lỗi cứng nếu: distance <= 0, surge < 1.0, source == destination,
      giờ/thứ ngoài miền.
    - Bật cờ is_extrapolation = True nếu: distance ngoài [0.02, 7.86] dặm,
      surge > 3.0x,
      hoặc tuyến (source -> destination) chưa từng xuất hiện trong 72 tuyến
      thực tế của Boston.
    """
    errors: list[str] = []
    warnings: list[str] = []
    is_extrapolation = False

    if source not in VALID_BOSTON_DISTRICTS:
        errors.append(
            f"Điểm đón '{source}' không nằm trong 12 quận Boston đã huấn "
            "luyện."
        )
    if destination not in VALID_BOSTON_DISTRICTS:
        errors.append(
            f"Điểm đến '{destination}' không nằm trong 12 quận Boston đã "
            "huấn luyện."
        )
    if source == destination:
        errors.append("Điểm đón và điểm đến không được trùng nhau.")
    if not np.isfinite(distance) or distance <= 0.0:
        errors.append("Khoảng cách phải lớn hơn 0 dặm.")
    if not np.isfinite(lyft_surge) or lyft_surge < 1.0:
        errors.append("Hệ số surge_multiplier phải >= 1.0x.")
    if not (0 <= int(hour) <= 23):
        errors.append("Giờ khởi hành (Boston EST) phải nằm trong [0, 23].")
    if not (0 <= int(day_of_week) <= 6):
        errors.append(
            "Thứ trong tuần phải nằm trong [0..6] (0=Thứ 2, 6=Chủ nhật)."
        )

    if np.isfinite(distance) and distance > 0.0:
        if distance < EMPIRICAL_DISTANCE_MIN:
            is_extrapolation = True
            warnings.append(
                f"EXTRAPOLATION WARNING: Quãng đường {distance:.2f} dặm "
                "thấp hơn ngưỡng cực tiểu quan sát "
                f"({EMPIRICAL_DISTANCE_MIN:.2f} dặm). Kết quả chỉ mang tính "
                "tham khảo."
            )
        elif distance > EMPIRICAL_DISTANCE_MAX:
            is_extrapolation = True
            warnings.append(
                f"EXTRAPOLATION WARNING: Quãng đường {distance:.2f} dặm "
                "vượt quá miền huấn luyện Boston "
                f"([{EMPIRICAL_DISTANCE_MIN:.2f}, "
                f"{EMPIRICAL_DISTANCE_MAX:.2f}] dặm). Mô hình cây M7 sẽ bão "
                "hòa ở biên; "
                f"cần thận trọng và đối chiếu với M2 OLS."
            )

    if np.isfinite(lyft_surge) and lyft_surge > 3.0:
        is_extrapolation = True
        warnings.append(
            f"EXTRAPOLATION WARNING: Hệ số surge {lyft_surge:.2f}x vượt mức "
            "tối đa quan sát (3.00x) trong dữ liệu Boston."
        )
    elif np.isfinite(lyft_surge) and lyft_surge >= 2.5:
        warnings.append(
            "Lưu ý vùng dữ liệu hiếm: hệ số surge >= 2.50x chỉ chiếm 53 "
            "lượt trong dữ liệu huấn luyện Lyft; "
            "mô hình cây M7 có thể bị bão hòa nhẹ — nên xem song song dự "
            "báo đối chứng của M2 OLS."
        )

    route_observed = True
    route_median_dist: float | None = None
    if route_catalog is not None and not route_catalog.empty and not errors:
        rmatch = route_catalog.loc[
            (route_catalog["source"].astype(str) == source)
            & (route_catalog["destination"].astype(str) == destination)
        ]
        if rmatch.empty:
            route_observed = False
            is_extrapolation = True
            warnings.append(
                f"EXTRAPOLATION WARNING: Tuyến '{source} -> {destination}' "
                "không nằm trong 72 tuyến liên quận "
                "thực tế của tập huấn luyện Boston (cấu trúc đồ thị lưỡng "
                "phân 6x6)."
            )
        else:
            route_median_dist = float(rmatch.iloc[0]["median_distance_miles"])

    return InputValidationResult(
        is_valid=(len(errors) == 0),
        is_extrapolation=is_extrapolation,
        errors=errors,
        warnings=warnings,
        route_observed_in_boston=route_observed,
        route_median_distance=route_median_dist,
    )


def build_feature_matrix_12_tiers(
    source: str,
    destination: str,
    distance: float,
    lyft_surge: float,
    hour: int,
    day_of_week: int,
    is_raining: bool,
    temp_f: float,
    weather_ref: dict[str, float],
    feature_order: tuple[str, ...] = EXPECTED_MODEL_FEATURES,
) -> pd.DataFrame:
    """
    Dựng ma trận đầu vào đúng 12 dòng x 27 đặc trưng theo đúng thứ tự của mô
    hình M7
    (và tương thích tuyệt đối với EconomicInteractionPreprocessor của M2).
    """
    is_weekend = 1 if int(day_of_week) in (5, 6) else 0
    is_rush_hour = (
        1
        if (int(hour) in (7, 8, 9, 16, 17, 18, 19) and is_weekend == 0)
        else 0
    )
    time_of_day = hour_to_time_of_day(int(hour))

    rain_val = float(weather_ref["rain_wet_median"]) if is_raining else 0.0
    clouds_val = min(
        1.0, float(weather_ref["clouds"]) + (0.20 if is_raining else 0.0)
    )
    humidity_val = min(
        0.99, float(weather_ref["humidity"]) + (0.12 if is_raining else 0.0)
    )

    hour_rad = 2.0 * np.pi * float(hour) / 24.0
    dow_rad = 2.0 * np.pi * float(day_of_week) / 7.0

    rows: list[dict[str, Any]] = []
    for cab_type, tier_name in ALL_TIERS_ORDERED:
        is_lyft = 1 if cab_type == "Lyft" else 0
        # Trong dữ liệu Boston, chỉ Lyft có hệ số nhân surge > 1.0 trên API;
        # Uber cố định 1.0x
        surge_applied = float(lyft_surge) if is_lyft == 1 else 1.0
        lyft_excess = (surge_applied - 1.0) * float(is_lyft)

        rows.append(
            {
                "distance": float(distance),
                "surge_multiplier": surge_applied,
                "temp": float(temp_f),
                "clouds": clouds_val,
                "pressure": float(weather_ref["pressure"]),
                "rain": rain_val,
                "humidity": humidity_val,
                "wind": float(weather_ref["wind"]),
                "hour": float(hour),
                "day_of_week": float(day_of_week),
                "is_weekend": float(is_weekend),
                "is_rush_hour": float(is_rush_hour),
                "is_raining": 1.0 if is_raining else 0.0,
                "cab_type": cab_type,
                "name": tier_name,
                "source": source,
                "destination": destination,
                "time_of_day": time_of_day,
                "distance_x_surge": float(distance) * surge_applied,
                "log_distance": float(np.log1p(distance)),
                "is_lyft": np.int8(is_lyft),
                "lyft_surge_excess": float(lyft_excess),
                "distance_x_lyft_surge_excess": float(distance)
                * float(lyft_excess),
                "hour_sin": float(np.sin(hour_rad)),
                "hour_cos": float(np.cos(hour_rad)),
                "dow_sin": float(np.sin(dow_rad)),
                "dow_cos": float(np.cos(dow_rad)),
            }
        )

    X_pred = pd.DataFrame(rows)[list(feature_order)]
    for c in BASE_CAT_FEATURES:
        X_pred[c] = X_pred[c].astype("category")
    return X_pred


def predict_fares_all_tiers(
    models: dict[str, Any],
    pi_df: pd.DataFrame,
    route_catalog: pd.DataFrame,
    weather_ref: dict[str, float],
    source: str,
    destination: str,
    distance: float,
    lyft_surge: float = 1.0,
    hour: int = 8,
    day_of_week: int = 2,
    is_raining: bool = False,
    temp_f: float = 40.0,
) -> dict[str, Any]:
    """
    Chạy suy luận trực tiếp bằng M7 (HistGBR) và M2 (Interaction OLS) cho cả 12
    gói dịch vụ.
    Trả về đối tượng chứa:
      - input_check (cờ hợp lệ & cảnh báo ngoại suy)
      - all_tiers_df (12 gói kèm dải tham chiếu cấp phân khúc gắn nhãn
        ASSUMPTION)
      - pairs_df (so sánh 5 cặp đối xứng Uber vs Lyft)
      - standalone_df (2 gói đơn lẻ WAV & Lux)
    """
    check = validate_prediction_inputs(
        source=source,
        destination=destination,
        distance=distance,
        lyft_surge=lyft_surge,
        hour=hour,
        day_of_week=day_of_week,
        route_catalog=route_catalog,
    )
    if not check.is_valid:
        return {
            "input_check": check,
            "all_tiers_df": pd.DataFrame(),
            "pairs_df": pd.DataFrame(),
            "standalone_df": pd.DataFrame(),
        }

    X_pred = build_feature_matrix_12_tiers(
        source=source,
        destination=destination,
        distance=distance,
        lyft_surge=lyft_surge,
        hour=hour,
        day_of_week=day_of_week,
        is_raining=is_raining,
        temp_f=temp_f,
        weather_ref=weather_ref,
        feature_order=models["feature_names"],
    )

    m7_preds = np.clip(models["M7"].predict(X_pred), 2.5, None)
    m2_preds = np.clip(models["M2"].predict(X_pred), 2.5, None)
    pi_map = pi_df.set_index("service_tier")[
        ["mpiw_width_usd", "picp_coverage_pct"]
    ].to_dict("index")

    tier_rows: list[dict[str, Any]] = []
    for idx, (cab_type, tier_name) in enumerate(ALL_TIERS_ORDERED):
        p_m7 = float(m7_preds[idx])
        p_m2 = float(m2_preds[idx])
        half_mpiw = (
            float(pi_map.get(tier_name, {}).get("mpiw_width_usd", 3.49)) / 2.0
        )
        picp = float(pi_map.get(tier_name, {}).get("picp_coverage_pct", 78.8))
        tier_rows.append(
            {
                "cab_type": cab_type,
                "service_tier": tier_name,
                "surge_applied": float(lyft_surge)
                if cab_type == "Lyft"
                else 1.0,
                "pred_m7_usd": round(p_m7, 2),
                "pred_m2_usd": round(p_m2, 2),
                "model_gap_m7_minus_m2_usd": round(p_m7 - p_m2, 2),
                "ref_band_low_usd": round(max(2.5, p_m7 - half_mpiw), 2),
                "ref_band_high_usd": round(p_m7 + half_mpiw, 2),
                "ref_tier_picp_pct": round(picp, 1),
                "band_type": (
                    "ASSUMPTION: ±MPIW/2 tham chiếu cấp phân khúc (không "
                    "phải Quantile cá thể)"
                ),
            }
        )

    all_tiers_df = pd.DataFrame(tier_rows)
    lookup = all_tiers_df.set_index("service_tier").to_dict("index")

    pair_rows: list[dict[str, Any]] = []
    for pair in SYMMETRIC_PAIRS:
        uber_info = lookup[pair["uber_tier"]]
        lyft_info = lookup[pair["lyft_tier"]]
        u_p, l_p = uber_info["pred_m7_usd"], lyft_info["pred_m7_usd"]
        diff = abs(u_p - l_p)
        higher = max(u_p, l_p)
        save_pct = (diff / higher * 100.0) if higher > 0 else 0.0

        if diff < 0.15:
            winner = "Tương đương"
        elif u_p < l_p:
            winner = f"Uber ({pair['uber_tier']})"
        else:
            winner = f"Lyft ({pair['lyft_tier']})"

        pair_rows.append(
            {
                "Phân khúc": pair["segment_label"],
                "Gói Uber": pair["uber_tier"],
                "Giá báo Uber (M7, USD)": u_p,
                "Giá báo Uber (M2 OLS, USD)": uber_info["pred_m2_usd"],
                "Gói Lyft": pair["lyft_tier"],
                "Giá báo Lyft (M7, USD)": l_p,
                "Giá báo Lyft (M2 OLS, USD)": lyft_info["pred_m2_usd"],
                "Hãng rẻ hơn (theo M7)": winner,
                "Chênh lệch (USD)": round(diff, 2),
                "Tiết kiệm (%)": round(save_pct, 1),
            }
        )

    standalone_rows = [
        {
            "Phân khúc độc lập": item["segment_label"],
            "Hãng": item["cab_type"],
            "Gói dịch vụ": item["tier"],
            "Giá báo (M7, USD)": lookup[item["tier"]]["pred_m7_usd"],
            "Giá báo (M2 OLS, USD)": lookup[item["tier"]]["pred_m2_usd"],
            "Dải tham chiếu (ASSUMPTION, USD)": (
                f"${lookup[item['tier']]['ref_band_low_usd']:.2f} – "
                f"${lookup[item['tier']]['ref_band_high_usd']:.2f}"
            ),
        }
        for item in STANDALONE_TIERS
    ]

    return {
        "input_check": check,
        "all_tiers_df": all_tiers_df,
        "pairs_df": pd.DataFrame(pair_rows),
        "standalone_df": pd.DataFrame(standalone_rows),
    }


def evaluate_models_on_test_set(
    models: dict[str, Any],
    master_df: pd.DataFrame,
    plot_sample_n: int = 3_500,
) -> dict[str, Any]:
    """
    Tái lập đúng tập Test chuẩn v5 trên mẫu quan sát surge thực
    (is_surge_original == 1,
    n_obs = 232,861, test_size=0.20, random_state=42 -> n_test = 46,573) để:
    1. Xác minh trực tiếp các chỉ số R2, RMSE, MAE, MAPE của M7 và M2 khớp báo
    cáo GĐ5.
    2. Trả về mẫu phân tầng nhẹ (plot_sample_n điểm, seed=42) cho biểu đồ
    Actual vs Predicted
       và Residual Diagnostics trên Tab 4 mà không làm nặng trình duyệt hay
       file HTML.
    """
    df_feat = engineer_features_stage4(master_df)
    if "is_surge_original" in df_feat.columns:
        df_valid = df_feat.loc[df_feat["is_surge_original"] == 1].reset_index(
            drop=True
        )
    else:
        df_valid = df_feat.loc[
            df_feat["surge_multiplier"].notna()
        ].reset_index(drop=True)
    idx = np.arange(len(df_valid))
    _, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_test = df_valid[list(models["feature_names"])].iloc[test_idx].copy()
    for c in BASE_CAT_FEATURES:
        X_test[c] = X_test[c].astype("category")
    y_test = df_valid[TARGET_COL].iloc[test_idx].astype(np.float64).values

    pred_m7 = models["M7"].predict(X_test).astype(np.float64)
    pred_m2 = models["M2"].predict(X_test).astype(np.float64)
    resid_m7 = y_test - pred_m7
    resid_m2 = y_test - pred_m2

    def _metrics(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
        abs_err = np.abs(y_true - y_hat)
        return {
            "n_test": int(len(y_true)),
            "r2": round(float(r2_score(y_true, y_hat)), 4),
            "rmse_usd": round(
                float(root_mean_squared_error(y_true, y_hat)), 4
            ),
            "mae_usd": round(float(mean_absolute_error(y_true, y_hat)), 4),
            "mape_pct": round(
                float(mean_absolute_percentage_error(y_true, y_hat) * 100.0), 2
            ),
            "pct_within_1usd": round(
                float((abs_err <= 1.0).mean() * 100.0), 2
            ),
            "pct_within_2usd": round(
                float((abs_err <= 2.0).mean() * 100.0), 2
            ),
            "pct_within_3usd": round(
                float((abs_err <= 3.0).mean() * 100.0), 2
            ),
        }

    # Trích mẫu đại diện cố định seed=42 để vẽ scatter nhẹ mượt
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(
        len(y_test), size=min(plot_sample_n, len(y_test)), replace=False
    )
    plot_df = pd.DataFrame(
        {
            "actual_usd": np.round(y_test[sample_idx], 2),
            "pred_m7_usd": np.round(pred_m7[sample_idx], 2),
            "pred_m2_usd": np.round(pred_m2[sample_idx], 2),
            "resid_m7_usd": np.round(resid_m7[sample_idx], 2),
            "resid_m2_usd": np.round(resid_m2[sample_idx], 2),
            "cab_type": X_test["cab_type"].iloc[sample_idx].astype(str).values,
            "service_tier": X_test["name"].iloc[sample_idx].astype(str).values,
            "distance": np.round(
                X_test["distance"].iloc[sample_idx].astype(np.float64).values,
                2,
            ),
            "surge_multiplier": np.round(
                X_test["surge_multiplier"]
                .iloc[sample_idx]
                .astype(np.float64)
                .values,
                2,
            ),
        }
    )

    # Tính histogram phần dư trên toàn bộ 46,573 dòng tập Test trong [-8, +8]
    # (không clip đỉnh giả)
    edges = np.linspace(-8.0, 8.0, 65)
    valid_r7 = resid_m7[(resid_m7 >= -8.0) & (resid_m7 <= 8.0)]
    valid_r2 = resid_m2[(resid_m2 >= -8.0) & (resid_m2 <= 8.0)]
    c7, _ = np.histogram(valid_r7, bins=edges)
    c2, _ = np.histogram(valid_r2, bins=edges)
    resid_hist_df = pd.DataFrame(
        {
            "bin_center_usd": np.round((edges[:-1] + edges[1:]) / 2.0, 3),
            "m7_count": c7.astype(int),
            "m2_count": c2.astype(int),
        }
    )

    return {
        "metrics_m7": _metrics(y_test, pred_m7),
        "metrics_m2": _metrics(y_test, pred_m2),
        "plot_sample_df": plot_df,
        "resid_hist_df": resid_hist_df,
        "resid_within_8usd_pct_m7": round(
            float(len(valid_r7) / len(resid_m7) * 100.0), 2
        ),
    }
