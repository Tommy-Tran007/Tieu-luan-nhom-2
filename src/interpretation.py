"""
Module: interpretation.py
Mục đích: Diễn giải mô hình đa phương pháp (Multi-method Model Interpretation &
XAI): biểu giá 12 gói xe & 5 cặp đối xứng, Permutation Feature Importance trên
M7/M2, và đường cong PDP/ICE nhất quán vật lý.

Tác giả: Trần Thanh Huy (MSSV: 2521001060) & Lê Huy Thắng (MSSV: 2521001092) —
Nhóm 02 — Lớp 25DDS01
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from config import RANDOM_STATE, TABLES_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_TIER_SEGMENT_MAP: dict[str, str] = {
    "Shared": "1. Đi ghép Tiết kiệm (Pool/Shared)",
    "UberPool": "1. Đi ghép Tiết kiệm (Pool/Shared)",
    "Lyft": "2. Phổ thông 4 chỗ (Standard)",
    "UberX": "2. Phổ thông 4 chỗ (Standard)",
    "WAV": "3. Hỗ trợ Xe lăn (Accessible)",
    "Lyft XL": "4. Gia đình 6 chỗ (XL SUV/Minivan)",
    "UberXL": "4. Gia đình 6 chỗ (XL SUV/Minivan)",
    "Lux": "5. Cao cấp Chuyên nghiệp (Premium Sedan)",
    "Black": "6. Hạng sang Doanh nhân (Executive Black)",
    "Lux Black": "6. Hạng sang Doanh nhân (Executive Black)",
    "Black SUV": "7. Hạng sang SUV Cỡ lớn (Luxury SUV)",
    "Lux Black XL": "7. Hạng sang SUV Cỡ lớn (Luxury SUV)",
}

_PAIR_SPECS: list[tuple[str, str, str]] = [
    ("Đi ghép (Pool/Shared)", "UberPool", "Shared"),
    ("Phổ thông 4 chỗ (Standard)", "UberX", "Lyft"),
    ("Gia đình 6 chỗ (XL)", "UberXL", "Lyft XL"),
    ("Hạng sang Sedan (Executive)", "Black", "Lux Black"),
    ("Hạng sang SUV (Luxury SUV)", "Black SUV", "Lux Black XL"),
]

_PERM_FEATURE_GROUPS: list[tuple[str, str, list[str]]] = [
    ("name (Phân khúc dịch vụ)", "1. Dịch vụ & Hãng xe", ["name"]),
    (
        "cab_type & is_lyft (Hãng xe Uber/Lyft)",
        "1. Dịch vụ & Hãng xe",
        ["cab_type", "is_lyft"],
    ),
    (
        "distance & log_distance (Quãng đường gốc)",
        "2. Khoảng cách & Không gian",
        ["distance", "log_distance"],
    ),
    (
        "source & destination (Điểm đón & trả)",
        "2. Khoảng cách & Không gian",
        ["source", "destination"],
    ),
    (
        "distance × surge (Tương tác cự ly–surge)",
        "3. Định giá Động & Tương tác",
        ["distance_x_surge", "distance_x_lyft_surge_excess"],
    ),
    (
        "surge_multiplier & lyft_excess (Hệ số surge)",
        "3. Định giá Động & Tương tác",
        ["surge_multiplier", "lyft_surge_excess"],
    ),
    ("temp (Nhiệt độ)", "4. Thời tiết & Khung giờ", ["temp"]),
    (
        "rain & is_raining (Lượng mưa & cờ mưa)",
        "4. Thời tiết & Khung giờ",
        ["rain", "is_raining"],
    ),
    ("humidity (Độ ẩm)", "4. Thời tiết & Khung giờ", ["humidity"]),
    ("wind (Tốc độ gió)", "4. Thời tiết & Khung giờ", ["wind"]),
    (
        "pressure (Áp suất khí quyển)",
        "4. Thời tiết & Khung giờ",
        ["pressure"],
    ),
    ("clouds (Độ che phủ mây)", "4. Thời tiết & Khung giờ", ["clouds"]),
    (
        "hour & time_of_day (Khung giờ EST)",
        "4. Thời tiết & Khung giờ",
        ["hour", "hour_sin", "hour_cos", "time_of_day", "is_rush_hour"],
    ),
    (
        "day_of_week & is_weekend (Thứ EST)",
        "4. Thời tiết & Khung giờ",
        ["day_of_week", "dow_sin", "dow_cos", "is_weekend"],
    ),
]


def _compare_symmetric_pair(
    seg_label: str,
    u_tier: str,
    l_tier: str,
    u_row: pd.Series,
    l_row: pd.Series,
    max_dist: float = 7.86,
) -> dict[str, Any]:
    """So sánh điểm cắt biểu giá HC3 của 1 cặp đối xứng Uber vs Lyft."""
    u_b0, u_b1 = (
        float(u_row["base_fare_b0_usd"]),
        float(u_row["per_mile_rate_b1_usd"]),
    )
    l_b0, l_b1 = (
        float(l_row["base_fare_b0_usd"]),
        float(l_row["per_mile_rate_b1_usd"]),
    )
    base_diff = u_b0 - l_b0
    slope_diff = l_b1 - u_b1
    d_cross = base_diff / slope_diff if abs(slope_diff) > 1e-6 else np.nan
    in_domain = bool(np.isfinite(d_cross) and (0.0 < d_cross <= max_dist))
    if in_domain:
        cross_status = f"Cắt trong miền tại {d_cross:.2f} dặm"
    elif np.isfinite(d_cross) and d_cross > max_dist:
        cross_status = (
            f"Không cắt trong miền [0.02, {max_dist:.2f}] dặm "
            f"({l_tier} luôn rẻ hơn)"
        )
    else:
        cross_status = "Không cắt trong miền quan sát"
    return {
        "segment_label": seg_label,
        "uber_tier": u_tier,
        "lyft_tier": l_tier,
        "uber_base_b0": round(u_b0, 2),
        "lyft_base_b0": round(l_b0, 2),
        "base_diff_uber_minus_lyft": round(base_diff, 2),
        "uber_slope_b1": round(u_b1, 2),
        "lyft_slope_b1": round(l_b1, 2),
        "slope_diff_lyft_minus_uber": round(slope_diff, 2),
        "crossing_distance_miles": round(float(d_cross), 2)
        if in_domain
        else None,
        "crossing_in_domain": in_domain,
        "crossing_status": cross_status,
        "cheaper_at_1_5_miles": l_tier
        if (l_b0 + 1.5 * l_b1) < (u_b0 + 1.5 * u_b1)
        else u_tier,
        "cheaper_at_4_5_miles": l_tier
        if (l_b0 + 4.5 * l_b1) < (u_b0 + 4.5 * u_b1)
        else u_tier,
    }


def build_tier_tariffs_and_pairs_table(
    ols_coef_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Mở rộng bảng hệ số OLS (HC3) 12 gói xe & đối sánh 5 cặp đối xứng."""
    df = ols_coef_df.copy()
    for col, d_mi in (
        ("expected_fare_1mi_usd", 1.0),
        ("expected_fare_2_5mi_usd", 2.5),
        ("expected_fare_5mi_usd", 5.0),
    ):
        df[col] = (
            df["base_fare_b0_usd"] + df["per_mile_rate_b1_usd"] * d_mi
        ).round(2)
    df["market_segment"] = df["service_tier"].map(_TIER_SEGMENT_MAP)

    idx_map = {row["service_tier"]: row for _, row in df.iterrows()}
    pair_comparisons = [
        _compare_symmetric_pair(
            seg, u_t, l_t, idx_map[u_t], idx_map[l_t], max_dist=7.86
        )
        for seg, u_t, l_t in _PAIR_SPECS
    ]
    out_path = TABLES_DIR / "stage6_tier_tariffs.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Đã lưu bảng biểu giá 12 gói xe Giai đoạn 6 tại: {out_path}")
    return df, pair_comparisons


def _validate_perm_groups(columns: pd.Index) -> None:
    """Xác minh phân hoạch rời 100% đúng 27 cột thực trên X_test."""
    all_cols = [c for _, _, cols in _PERM_FEATURE_GROUPS for c in cols]
    if len(all_cols) != len(set(all_cols)):
        raise ValueError(
            "LỖI HIGH-01: Phát hiện cột bị đếm trùng trong feature_groups "
            "của Permutation Importance!"
        )
    missing = set(columns) - set(all_cols)
    ghost = set(all_cols) - set(columns)
    if missing or ghost:
        raise ValueError(
            "LỖI HIGH-01: feature_groups không khớp X_test! "
            f"Thiếu={missing}, Ảo={ghost}"
        )


def _eval_perm_group(
    feat_label: str,
    family_name: str,
    cols_to_perm: list[str],
    X_sub: pd.DataFrame,
    y_sub: np.ndarray,
    model_m7: Any,
    model_m2: Any,
    base_metrics: tuple[float, float, float, float],
    n_repeats: int,
    rng: np.random.RandomState,
) -> dict[str, Any]:
    """Đánh giá mức tăng RMSE & giảm R2 khi hoán vị 1 nhóm đặc trưng."""
    base_rmse_m7, base_r2_m7, base_rmse_m2, base_r2_m2 = base_metrics
    m7_rmse_inc, m7_r2_drop = [], []
    m2_rmse_inc, m2_r2_drop = [], []
    for _rep in range(n_repeats):
        perm_order = rng.permutation(len(X_sub))
        X_perm = X_sub.copy()
        for c in cols_to_perm:
            if c in X_perm.columns:
                X_perm[c] = X_sub[c].iloc[perm_order].values
                if str(X_sub[c].dtype) == "category":
                    X_perm[c] = X_perm[c].astype("category")
        p7 = model_m7.predict(X_perm)
        m7_rmse_inc.append(
            float(np.sqrt(mean_squared_error(y_sub, p7))) - base_rmse_m7
        )
        m7_r2_drop.append(base_r2_m7 - float(r2_score(y_sub, p7)))
        p2 = model_m2.predict(X_perm)
        m2_rmse_inc.append(
            float(np.sqrt(mean_squared_error(y_sub, p2))) - base_rmse_m2
        )
        m2_r2_drop.append(base_r2_m2 - float(r2_score(y_sub, p2)))
    return {
        "feature_group": feat_label,
        "domain_family": family_name,
        "permuted_columns": ", ".join(cols_to_perm),
        "m7_delta_rmse_mean_usd": round(float(np.mean(m7_rmse_inc)), 4),
        "m7_delta_rmse_std_usd": round(float(np.std(m7_rmse_inc, ddof=1)), 4),
        "m7_delta_r2_mean": round(float(np.mean(m7_r2_drop)), 5),
        "m7_delta_r2_std": round(float(np.std(m7_r2_drop, ddof=1)), 5),
        "m2_delta_rmse_mean_usd": round(float(np.mean(m2_rmse_inc)), 4),
        "m2_delta_rmse_std_usd": round(float(np.std(m2_rmse_inc, ddof=1)), 4),
        "m2_delta_r2_mean": round(float(np.mean(m2_r2_drop)), 5),
        "m2_delta_r2_std": round(float(np.std(m2_r2_drop, ddof=1)), 5),
    }


def run_permutation_importance_analysis(
    model_m7: Any,
    model_m2: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_repeats: int = 8,
    sample_size: int = 15000,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Tính Permutation Importance phân hoạch rời 14 nhóm = 27/27 cột X."""
    logger.info(
        "Đang tính toán Permutation Importance trên tập Test cho M7 và M2..."
    )
    rng = np.random.RandomState(random_state)
    idx = rng.choice(
        len(X_test), size=min(sample_size, len(X_test)), replace=False
    )
    X_sub = X_test.iloc[idx].copy().reset_index(drop=True)
    prep_m2 = getattr(model_m2, "named_steps", {}).get("prep")
    imp_tr = getattr(prep_m2, "route_dist_imputer_", None)
    if imp_tr is not None:
        X_sub = imp_tr.transform(X_sub)
    y_sub = np.asarray(y_test, dtype=np.float64)[idx]
    _validate_perm_groups(X_sub.columns)

    p7_0, p2_0 = model_m7.predict(X_sub), model_m2.predict(X_sub)
    base_metrics = (
        float(np.sqrt(mean_squared_error(y_sub, p7_0))),
        float(r2_score(y_sub, p7_0)),
        float(np.sqrt(mean_squared_error(y_sub, p2_0))),
        float(r2_score(y_sub, p2_0)),
    )
    rows = [
        _eval_perm_group(
            lbl,
            fam,
            cols,
            X_sub,
            y_sub,
            model_m7,
            model_m2,
            base_metrics,
            n_repeats,
            rng,
        )
        for lbl, fam, cols in _PERM_FEATURE_GROUPS
    ]
    perm_df = (
        pd.DataFrame(rows)
        .sort_values("m7_delta_rmse_mean_usd", ascending=False)
        .reset_index(drop=True)
    )
    tot_pos = float(perm_df["m7_delta_rmse_mean_usd"].clip(lower=0.0).sum())
    perm_df["m7_importance_share_pct"] = (
        perm_df["m7_delta_rmse_mean_usd"].clip(lower=0.0)
        / max(tot_pos, 1e-9)
        * 100.0
    ).round(2)
    family_df = (
        perm_df.groupby("domain_family", observed=False)
        .agg(
            n_feature_groups=("feature_group", "count"),
            m7_sum_delta_rmse_usd=(
                "m7_delta_rmse_mean_usd",
                lambda s: round(float(np.clip(s, 0, None).sum()), 4),
            ),
            m7_share_pct=(
                "m7_importance_share_pct",
                lambda s: round(float(s.sum()), 2),
            ),
            m2_sum_delta_rmse_usd=(
                "m2_delta_rmse_mean_usd",
                lambda s: round(float(np.clip(s, 0, None).sum()), 4),
            ),
        )
        .reset_index()
        .sort_values("m7_share_pct", ascending=False)
    )
    out_path = TABLES_DIR / "stage6_permutation_importance.csv"
    perm_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Đã lưu bảng Permutation Importance tại: {out_path}")
    return perm_df, family_df.to_dict(orient="records")


def _compute_distance_pdp_ice(
    model_m7: Any,
    model_m2: Any,
    X_test: pd.DataFrame,
    rng: np.random.RandomState,
) -> dict[str, Any]:
    """Tính PDP & ICE nhất quán vật lý theo quãng đường và 4 gói tiêu biểu."""
    sub_idx = rng.choice(
        len(X_test), size=min(1500, len(X_test)), replace=False
    )
    X_base = X_test.iloc[sub_idx].copy().reset_index(drop=True)
    ice_idx = rng.choice(len(X_base), size=min(40, len(X_base)), replace=False)

    d_grid = np.linspace(0.3, 5.8, 35)
    pdp_m7, pdp_m2 = [], []
    ice_m7 = np.zeros((len(ice_idx), len(d_grid)), dtype=np.float64)
    for j, d_val in enumerate(d_grid):
        X_tmp = X_base.copy()
        X_tmp["distance"] = float(d_val)
        X_tmp["log_distance"] = float(np.log1p(d_val))
        X_tmp["distance_x_surge"] = float(d_val) * X_tmp[
            "surge_multiplier"
        ].astype(np.float64)
        X_tmp["distance_x_lyft_surge_excess"] = float(d_val) * X_tmp[
            "lyft_surge_excess"
        ].astype(np.float64)
        p7, p2 = model_m7.predict(X_tmp), model_m2.predict(X_tmp)
        pdp_m7.append(float(np.mean(p7)))
        pdp_m2.append(float(np.mean(p2)))
        ice_m7[:, j] = p7[ice_idx]

    tier_curves: dict[str, list[float]] = {}
    for tier_name in ("Shared", "UberX", "Lyft", "Black SUV"):
        X_tier = (
            X_test.loc[X_test["name"].astype(str) == tier_name]
            .iloc[:500]
            .copy()
        )
        vals = []
        for d_val in d_grid:
            X_tmp = X_tier.copy()
            X_tmp["distance"] = float(d_val)
            X_tmp["log_distance"] = float(np.log1p(d_val))
            X_tmp["surge_multiplier"] = 1.0
            X_tmp["lyft_surge_excess"] = 0.0
            X_tmp["distance_x_surge"] = float(d_val)
            X_tmp["distance_x_lyft_surge_excess"] = 0.0
            vals.append(round(float(np.mean(model_m7.predict(X_tmp))), 3))
        tier_curves[tier_name] = vals
    return {
        "distance_grid": [round(float(x), 3) for x in d_grid],
        "pdp_distance_m7": [round(float(x), 3) for x in pdp_m7],
        "pdp_distance_m2": [round(float(x), 3) for x in pdp_m2],
        "ice_distance_m7": ice_m7.tolist(),
        "tier_pdp_distance_curves": tier_curves,
    }


def _compute_surge_and_2d_pdp(
    model_m7: Any,
    model_m2: Any,
    X_test: pd.DataFrame,
    rng: np.random.RandomState,
) -> dict[str, Any]:
    """Tính PDP/ICE hệ số surge Lyft và lưới 2D distance × surge."""
    X_lyft_all = X_test.loc[X_test["cab_type"].astype(str) == "Lyft"]
    sub_idx = rng.choice(
        len(X_lyft_all), size=min(1200, len(X_lyft_all)), replace=False
    )
    X_base = X_lyft_all.iloc[sub_idx].copy().reset_index(drop=True)
    ice_idx = rng.choice(len(X_base), size=min(40, len(X_base)), replace=False)

    s_grid = np.array([1.0, 1.125, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
    pdp_m7, pdp_m2 = [], []
    ice_m7 = np.zeros((len(ice_idx), len(s_grid)), dtype=np.float64)
    for j, s_val in enumerate(s_grid):
        X_tmp = X_base.copy()
        X_tmp["surge_multiplier"] = float(s_val)
        X_tmp["lyft_surge_excess"] = float(s_val - 1.0)
        dist_f = X_tmp["distance"].astype(np.float64)
        X_tmp["distance_x_surge"] = dist_f * float(s_val)
        X_tmp["distance_x_lyft_surge_excess"] = dist_f * float(s_val - 1.0)
        p7, p2 = model_m7.predict(X_tmp), model_m2.predict(X_tmp)
        pdp_m7.append(float(np.mean(p7)))
        pdp_m2.append(float(np.mean(p2)))
        ice_m7[:, j] = p7[ice_idx]

    lyft_std = (
        X_lyft_all.loc[X_lyft_all["name"].astype(str) == "Lyft"]
        .iloc[:400]
        .copy()
    )
    d_2d = np.linspace(0.5, 5.5, 18)
    s_2d = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
    z_2d = np.zeros((len(s_2d), len(d_2d)), dtype=np.float64)
    for r_i, s_val in enumerate(s_2d):
        for c_i, d_val in enumerate(d_2d):
            X_tmp = lyft_std.copy()
            X_tmp["distance"] = float(d_val)
            X_tmp["log_distance"] = float(np.log1p(d_val))
            X_tmp["surge_multiplier"] = float(s_val)
            X_tmp["lyft_surge_excess"] = float(s_val - 1.0)
            X_tmp["distance_x_surge"] = float(d_val * s_val)
            X_tmp["distance_x_lyft_surge_excess"] = float(
                d_val * (s_val - 1.0)
            )
            z_2d[r_i, c_i] = float(np.mean(model_m7.predict(X_tmp)))
    return {
        "surge_grid": [round(float(x), 3) for x in s_grid],
        "pdp_surge_lyft_m7": [round(float(x), 3) for x in pdp_m7],
        "pdp_surge_lyft_m2": [round(float(x), 3) for x in pdp_m2],
        "ice_surge_lyft_m7": ice_m7.tolist(),
        "grid_2d_distance": [round(float(x), 3) for x in d_2d],
        "grid_2d_surge": [round(float(x), 3) for x in s_2d],
        "grid_2d_lyft_fare": z_2d.tolist(),
    }


def compute_pdp_and_ice_data(
    model_m7: Any,
    model_m2: Any,
    X_test: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Tính PDP & ICE nhất quán vật lý cho distance, surge và lưới 2D."""
    logger.info(
        "Đang tính toán đường cong PDP & ICE nhất quán vật lý cho distance "
        "và surge_multiplier..."
    )
    rng = np.random.RandomState(random_state)
    res = _compute_distance_pdp_ice(model_m7, model_m2, X_test, rng)
    res.update(_compute_surge_and_2d_pdp(model_m7, model_m2, X_test, rng))
    return res
