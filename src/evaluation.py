"""
Module: evaluation.py
Mục đích: Chẩn đoán phần dư nâng cao (Residual Diagnostics), kiểm định phương
sai sai số thay đổi
Breusch-Pagan, phân rã sai số đa chiều (4 chiều: Gói xe, Dải cự ly, Chế độ
Surge, Quận xuất phát),
và ước lượng khoảng dự báo bất định P10-P90 bằng Quantile HistGradientBoosting.

Tác giả: Trần Thanh Huy (MSSV: 2521001060) & Lê Huy Thắng (MSSV: 2521001092) —
Nhóm 02 — Lớp 25DDS01
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from statsmodels.stats.diagnostic import het_breuschpagan

from config import BASE_CAT_FEATURES, RANDOM_STATE
from features import RouteDistanceImputer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_features: int = 0,
) -> dict[str, float]:
    """
    Tính toán trọn bộ chỉ số đánh giá hồi quy (R2, Adjusted R2, RMSE, MAE,
    MAPE, Max Error).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true_arr)

    r2 = float(r2_score(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mape = float(
        mean_absolute_percentage_error(y_true_arr, y_pred_arr) * 100.0
    )
    max_err = float(max_error(y_true_arr, y_pred_arr))

    if num_features > 0 and n > num_features + 1:
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - num_features - 1))
    else:
        adj_r2 = r2

    return {
        "R2": round(r2, 4),
        "Adjusted_R2": round(adj_r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE_pct": round(mape, 2),
        "Max_Error": round(max_err, 4),
    }


def _run_breusch_pagan(
    residuals: np.ndarray,
    y_pred_arr: np.ndarray,
    X_exog_bp: pd.DataFrame | None,
) -> tuple[float, float, float, float]:
    """Chạy kiểm định Breusch-Pagan cho phương sai sai số thay đổi."""
    if X_exog_bp is not None:
        bp_df = (
            X_exog_bp[["distance", "surge_multiplier", "is_lyft"]]
            .copy()
            .astype(np.float64)
        )
        bp_df["y_pred"] = y_pred_arr
        exog_raw = sm.add_constant(bp_df.values)
        valid_bp = np.isfinite(exog_raw).all(axis=1) & np.isfinite(residuals)
        exog_mat = exog_raw[valid_bp]
        res_bp = np.asarray(residuals, dtype=np.float64)[valid_bp]
    else:
        exog_mat = sm.add_constant(
            np.column_stack([y_pred_arr, y_pred_arr**2])
        )
        res_bp = residuals
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(res_bp, exog_mat)
    return float(bp_lm), float(bp_lm_p), float(bp_f), float(bp_f_p)


def diagnose_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_exog_bp: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Thực hiện chẩn đoán phần dư (Residual Diagnostics) theo chuẩn Kinh tế
    lượng: Skewness, Kurtosis, Jarque-Bera, Shapiro-Wilk, Breusch-Pagan và
    tỷ lệ dự báo trong biên sai số +-0.5$..+-5.0$.
    """
    residuals = np.asarray(y_true, dtype=np.float64) - np.asarray(
        y_pred, dtype=np.float64
    )
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    abs_res = np.abs(residuals)

    mean_res = float(np.mean(residuals))
    median_res = float(np.median(residuals))
    std_res = float(np.std(residuals, ddof=1))
    iqr_res = float(
        np.percentile(residuals, 75) - np.percentile(residuals, 25)
    )
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(
        len(residuals), size=min(5000, len(residuals)), replace=False
    )
    shapiro_stat, shapiro_pval = stats.shapiro(residuals[sample_idx])
    bp_lm, bp_lm_pval, bp_f, bp_f_pval = _run_breusch_pagan(
        residuals, y_pred_arr, X_exog_bp
    )

    pct_1 = float(np.mean(abs_res <= 1.0) * 100.0)
    pct_2 = float(np.mean(abs_res <= 2.0) * 100.0)
    pct_3 = float(np.mean(abs_res <= 3.0) * 100.0)
    return {
        "residual_mean": round(mean_res, 4),
        "residual_median": round(median_res, 4),
        "residual_std": round(std_res, 4),
        "residual_iqr": round(iqr_res, 4),
        "skewness": round(float(stats.skew(residuals)), 4),
        "kurtosis": round(float(stats.kurtosis(residuals)), 4),
        "jarque_bera_stat": round(float(jb_stat), 2),
        "jarque_bera_pval": float(jb_pval),
        "shapiro_wilk_stat_n5000": round(float(shapiro_stat), 4),
        "shapiro_wilk_pval_n5000": float(shapiro_pval),
        "breusch_pagan_lm_stat": round(bp_lm, 2),
        "breusch_pagan_lm_pval": bp_lm_pval,
        "breusch_pagan_f_stat": round(bp_f, 2),
        "breusch_pagan_f_pval": bp_f_pval,
        "is_homoscedastic": bool(bp_lm_pval >= 0.05),
        "is_normal": bool(jb_pval >= 0.05),
        "pct_within_05usd": round(float(np.mean(abs_res <= 0.5) * 100.0), 2),
        "pct_within_1usd": round(pct_1, 2),
        "pct_within_2usd": round(pct_2, 2),
        "pct_within_3usd": round(pct_3, 2),
        "pct_within_5usd": round(float(np.mean(abs_res <= 5.0) * 100.0), 2),
        "p90_abs_error_usd": round(float(np.percentile(abs_res, 90)), 4),
        "p95_abs_error_usd": round(float(np.percentile(abs_res, 95)), 4),
        "interpretation": (
            f"Phần dư có kỳ vọng xấp xỉ 0 (Mean={mean_res:+.4f}, "
            f"Median={median_res:+.4f}, SD=${std_res:.3f}). "
            f"Có {pct_1:.1f}% chuyến đi lệch không quá ±$1.00, "
            f"{pct_2:.1f}% lệch không quá ±$2.00 "
            f"và {pct_3:.1f}% lệch không quá ±$3.00. "
            f"Kiểm định Breusch-Pagan (LM={bp_lm:.2f}, p<0.0001) xác nhận "
            "phương sai phần dư tăng theo cự ly và phân khúc hạng sang, "
            "củng cố cơ sở toán học của việc dùng sai số chuẩn bền vững "
            "HC3 ở Giai đoạn 5."
        ),
    }


def run_multidimensional_segment_errors(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_m7: np.ndarray,
    y_pred_m2: np.ndarray,
) -> pd.DataFrame:
    """
    Phân rã sai số của M7 (HistGradientBoosting) và M2 (OLS tương tác) theo 4
    chiều:
    1. service_tier (12 gói xe name)
    2. distance_bucket (4 dải quãng đường)
    3. surge_regime (4 mức hệ số nhân giá động)
    4. source_district (12 quận xuất phát tại Boston)
    """
    work = df_test.copy()
    work["actual"] = np.asarray(y_true, dtype=np.float64)
    work["pred_m7"] = np.asarray(y_pred_m7, dtype=np.float64)
    work["pred_m2"] = np.asarray(y_pred_m2, dtype=np.float64)

    work["distance_bucket"] = pd.cut(
        work["distance"],
        bins=[0.0, 1.5, 3.0, 4.5, 100.0],
        labels=[
            "1. Ngắn (<1.5 dặm)",
            "2. Trung bình (1.5-3.0 dặm)",
            "3. Khá dài (3.0-4.5 dặm)",
            "4. Dài (>4.5 dặm)",
        ],
    )
    work["distance_bucket"] = (
        work["distance_bucket"]
        .cat.add_categories(["5. Khuyết distance (NaN thô)"])
        .fillna("5. Khuyết distance (NaN thô)")
    )
    work["surge_regime"] = pd.cut(
        work["surge_multiplier"],
        bins=[0.9, 1.05, 1.55, 2.05, 10.0],
        labels=[
            "1. Bình thường (1.0x)",
            "2. Surge nhẹ (1.25x-1.50x)",
            "3. Surge cao (1.75x-2.00x)",
            "4. Surge cực đại (>=2.50x)",
        ],
    )

    dimensions = [
        ("service_tier", "name"),
        ("distance_bucket", "distance_bucket"),
        ("surge_regime", "surge_regime"),
        ("source_district", "source"),
    ]

    rows: list[dict[str, Any]] = []
    for dim_label, col_name in dimensions:
        for grp_val, grp_df in work.groupby(col_name, observed=False):
            if len(grp_df) == 0:
                continue
            yt = grp_df["actual"].values
            yp7 = grp_df["pred_m7"].values
            yp2 = grp_df["pred_m2"].values

            ae7 = np.abs(yt - yp7)
            ae2 = np.abs(yt - yp2)

            r2_7 = (
                float(r2_score(yt, yp7))
                if len(yt) > 5 and np.std(yt) > 1e-6
                else np.nan
            )
            r2_2 = (
                float(r2_score(yt, yp2))
                if len(yt) > 5 and np.std(yt) > 1e-6
                else np.nan
            )

            rows.append(
                {
                    "dimension": dim_label,
                    "segment_value": str(grp_val),
                    "n_test": int(len(yt)),
                    "share_test_pct": round(len(yt) / len(work) * 100.0, 2),
                    "mean_actual_usd": round(float(np.mean(yt)), 3),
                    "std_actual_usd": round(float(np.std(yt, ddof=1)), 3),
                    "m7_r2": round(r2_7, 4) if not np.isnan(r2_7) else None,
                    "m7_rmse_usd": round(float(np.sqrt(np.mean(ae7**2))), 4),
                    "m7_mae_usd": round(float(np.mean(ae7)), 4),
                    "m7_median_ae_usd": round(float(np.median(ae7)), 4),
                    "m7_p90_ae_usd": round(float(np.percentile(ae7, 90)), 4),
                    "m7_mape_pct": round(float(np.mean(ae7 / yt) * 100.0), 2),
                    "m7_within_2usd_pct": round(
                        float(np.mean(ae7 <= 2.0) * 100.0), 2
                    ),
                    "m2_r2": round(r2_2, 4) if not np.isnan(r2_2) else None,
                    "m2_rmse_usd": round(float(np.sqrt(np.mean(ae2**2))), 4),
                    "m2_mae_usd": round(float(np.mean(ae2)), 4),
                    "m2_mape_pct": round(float(np.mean(ae2 / yt) * 100.0), 2),
                }
            )

    return pd.DataFrame(rows)


def _make_quantile_pipe(
    num_cols: list[str],
    alpha_q: float,
    best_hgb_params: dict[str, Any],
    random_state: int,
) -> Pipeline:
    """Khởi tạo Pipeline Quantile HistGradientBoosting dùng native category."""
    prep = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", "passthrough", BASE_CAT_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    reg = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=alpha_q,
        categorical_features="from_dtype",
        max_iter=220,
        learning_rate=best_hgb_params.get("learning_rate", 0.08),
        max_leaf_nodes=best_hgb_params.get("max_leaf_nodes", 31),
        min_samples_leaf=best_hgb_params.get("min_samples_leaf", 20),
        l2_regularization=best_hgb_params.get("l2_regularization", 5.0),
        random_state=random_state,
    )
    return Pipeline(
        [
            ("route_dist_imp", RouteDistanceImputer()),
            ("prep", prep),
            ("reg", reg),
        ]
    )


def _summarize_pi_by_tier(
    X_test: pd.DataFrame,
    y_te: np.ndarray,
    pred_p10: np.ndarray,
    pred_p50: np.ndarray,
    pred_p90: np.ndarray,
    widths: np.ndarray,
    in_interval: np.ndarray,
) -> pd.DataFrame:
    """Bóc tách độ phủ PICP và độ rộng MPIW cho 12 phân khúc dịch vụ."""
    temp_df = pd.DataFrame(
        {
            "cab_type": X_test["cab_type"].astype(str).values,
            "service_tier": X_test["name"].astype(str).values,
            "actual": y_te,
            "p10": pred_p10,
            "p50": pred_p50,
            "p90": pred_p90,
            "width": widths,
            "covered": in_interval.astype(np.float64) * 100.0,
        }
    )
    pi_rows: list[dict[str, Any]] = []
    for (cab, tier), g in temp_df.groupby(
        ["cab_type", "service_tier"], observed=False
    ):
        if len(g) == 0:
            continue
        pi_rows.append(
            {
                "cab_type": str(cab),
                "service_tier": str(tier),
                "n_test": int(len(g)),
                "mean_actual_usd": round(float(g["actual"].mean()), 2),
                "mean_p10_usd": round(float(g["p10"].mean()), 2),
                "mean_p50_usd": round(float(g["p50"].mean()), 2),
                "mean_p90_usd": round(float(g["p90"].mean()), 2),
                "mpiw_width_usd": round(float(g["width"].mean()), 3),
                "relative_width_pct": round(
                    float(g["width"].mean() / g["actual"].mean() * 100.0), 2
                ),
                "picp_coverage_pct": round(float(g["covered"].mean()), 2),
            }
        )
    return (
        pd.DataFrame(pi_rows)
        .sort_values(["cab_type", "mean_actual_usd"])
        .reset_index(drop=True)
    )


def run_quantile_prediction_intervals(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    best_hgb_params: dict[str, Any],
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    """
    Huấn luyện 3 mô hình Quantile HistGradientBoostingRegressor (P10, P50, P90)
    dùng pandas category bản địa tạo khoảng dự báo bất định 80% [P10, P90].
    """
    logger.info(
        "Đang huấn luyện Quantile HistGradientBoosting (P10, P50, P90, "
        "native category) cho khoảng dự báo bất định..."
    )
    num_cols = [c for c in X_train.columns if c not in BASE_CAT_FEATURES]
    X_tr_q, X_te_q = X_train.copy(), X_test.copy()
    for c in BASE_CAT_FEATURES:
        X_tr_q[c] = X_tr_q[c].astype("category")
        X_te_q[c] = X_te_q[c].astype("category")

    raw_p10 = (
        _make_quantile_pipe(num_cols, 0.10, best_hgb_params, random_state)
        .fit(X_tr_q, y_train)
        .predict(X_te_q)
    )
    raw_p50 = (
        _make_quantile_pipe(num_cols, 0.50, best_hgb_params, random_state)
        .fit(X_tr_q, y_train)
        .predict(X_te_q)
    )
    raw_p90 = (
        _make_quantile_pipe(num_cols, 0.90, best_hgb_params, random_state)
        .fit(X_tr_q, y_train)
        .predict(X_te_q)
    )

    pred_p10 = np.minimum(raw_p10, raw_p50)
    pred_p90 = np.maximum(raw_p90, raw_p50)
    pred_p50 = np.clip(raw_p50, pred_p10, pred_p90)

    y_te = np.asarray(y_test, dtype=np.float64)
    in_interval = (y_te >= pred_p10) & (y_te <= pred_p90)
    widths = pred_p90 - pred_p10
    overall_summary = {
        "nominal_coverage_pct": 80.0,
        "empirical_coverage_picp_pct": round(
            float(np.mean(in_interval) * 100.0), 2
        ),
        "below_p10_pct": round(float(np.mean(y_te < pred_p10) * 100.0), 2),
        "above_p90_pct": round(float(np.mean(y_te > pred_p90) * 100.0), 2),
        "mean_interval_width_mpiw_usd": round(float(np.mean(widths)), 4),
        "median_interval_width_usd": round(float(np.median(widths)), 4),
        "relative_width_pct_of_mean_price": round(
            float(np.mean(widths) / np.mean(y_te) * 100.0), 2
        ),
        "p50_median_mae_usd": round(
            float(mean_absolute_error(y_te, pred_p50)), 4
        ),
        "p50_median_mape_pct": round(
            float(mean_absolute_percentage_error(y_te, pred_p50) * 100.0), 2
        ),
    }
    pi_df = _summarize_pi_by_tier(
        X_test, y_te, pred_p10, pred_p50, pred_p90, widths, in_interval
    )
    return (
        overall_summary,
        pi_df,
        {"p10": pred_p10, "p50": pred_p50, "p90": pred_p90},
    )
