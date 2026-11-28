"""
Module: src/thesis_v5_verification.py
Mục đích: Hợp nhất toàn bộ hệ thống kiểm chứng thống kê, kinh tế lượng, học
máy,
công bằng không gian và mô phỏng chính sách (v5) vào MỘT nguồn sự thật duy
nhất.

Chỉ-đọc (READ-ONLY) trên data/raw/, data/processed/, models/, reports/tables/
và
xuất duy nhất tệp chuẩn:
    reports/results/thesis_v5_verification.json

Các khối kiểm chứng hợp nhất (6 nhóm):
1. Phễu làm sạch (693,071 -> 270,898), kiểm toán thiên lệch chọn mẫu &
CRIT-03/H4.
2. Thống kê bền vững: Cluster-Robust OLS (quận×giờ & giờ toàn Boston), TOST
(±$0.10/±$0.25),
   phân tầng cường độ mưa, Exact WCR Wild Cluster Bootstrap (B=1,999, chiếu
   M_Z, đủ hệ số
   bậc tự do (N-1)/(N-k)), kiểm định cụm H3b (Lyft Surge x Mưa), độ nhạy
   sentinel mưa,
   Holm-Bonferroni & BH-FDR (13 giả thuyết), và F-test cụm cho H2c.
3. RQ2 Biểu giá cắt nhau: 95% CI phương pháp Delta (HC3 & Cluster) cho cả 5 cặp
trên
   Đặc tả A (X_train 80%), Đặc tả B (X_train 80% & toàn mẫu không surge), và
   kiểm tra
   cước sàn Black >= 1.0 dặm.
4. Chẩn đoán mô hình & CQR: Paired 5-Fold M7 vs M6, M0b vs M7 theo chế độ
Surge,
   Monotonic Split-CQR duy nhất dùng pandas category gốc (khớp
   M7/evaluation.py),
   Grouped XAI 6 cụm = 27/27 cột, độ nhạy dữ liệu khuyết, API-Batch
   GroupShuffleSplit
   (trên 232,861 dòng quan sát surge thực), và Strict Train-Only Full Refit
   (tuân CRIT-03).
5. Công bằng không gian 12 quận Boston (M7 bias/MAE/MAPE) & xếp hạng Surge Lyft
12 quận
   trên mẫu quan sát thực (92,384 báo giá, 6.80%).
6. RQ4 Chính sách Trần Surge có ràng buộc cung Q_realized = min(Q_D, Q_S) (C2),
đối chứng
   số học & Out-of-Sample Test-only, bảng tối ưu theo ngưỡng tau, Power
   Analysis Switchback
   phân biệt Triggered vs ITT (H3), và Benchmark I/O công bằng CSV vs Parquet
   cùng bảng (M1).

Lệnh chạy:
    python src/thesis_v5_verification.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (str(ROOT_DIR), str(ROOT_DIR / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from clean_raw_data import (  # noqa: E402
    PRODUCT_MAP,
    clean_unit_string,
    recover_scaled_series,
    standardize_location,
)
from config import (  # noqa: E402
    BASE_CAT_FEATURES,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DIR,
    RESULTS_DIR,
    TABLES_DIR,
)
from features import (  # noqa: E402
    EconomicInteractionPreprocessor,
    build_stage4_preprocessors,
    create_data_splits,
    engineer_features_stage4,
    get_feature_columns_for_modeling,
)
from models import GroupMedianBaselineRegressor  # noqa: E402

OUT_JSON = RESULTS_DIR / "thesis_v5_verification.json"


# =============================================================================
# HELPER FUNCTIONS (SINGLE SOURCE — ZERO DUPLICATION)
# =============================================================================
def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(y_true)),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mape_pct": round(
            float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0), 2
        ),
    }


def _tost_test(
    est: float, se: float, delta: float, df_dof: int
) -> dict[str, Any]:
    t1 = (est - (-delta)) / se
    t2 = (est - delta) / se
    p1 = float(1.0 - stats.t.cdf(t1, df=df_dof))
    p2 = float(stats.t.cdf(t2, df=df_dof))
    p_tost = max(p1, p2)
    return {
        "delta_usd": delta,
        "t1": round(t1, 3),
        "t2": round(t2, 3),
        "p_tost": round(p_tost, 6),
        "equivalent_at_5pct": bool(p_tost < 0.05),
    }


def _run_exact_wcr_bootstrap(
    y_vec: np.ndarray,
    x_vec: np.ndarray,
    z_mat: np.ndarray,
    cl_series: pd.Series,
    b_reps: int = 1999,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Exact Restricted Wild Cluster Bootstrap (Cameron, Gelbach, & Miller, 2008):
    - Chiếu trực giao M_Z bằng phân rã QR thu gọn.
    - Áp dụng đầy đủ hệ số hiệu chỉnh bậc tự do hữu hạn: (G/(G-1)) *
      ((N-1)/(N-k)).
    - Phép chiếu lại trực giao M_Z y* trong mọi vòng Rademacher.
    """
    q_mat, _ = np.linalg.qr(z_mat, mode="reduced")
    y_tilde = y_vec - q_mat @ (q_mat.T @ y_vec)
    x_tilde = x_vec - q_mat @ (q_mat.T @ x_vec)
    denom = float(np.dot(x_tilde, x_tilde))
    beta_obs = float(np.dot(x_tilde, y_tilde) / denom)
    resid_obs = y_tilde - beta_obs * x_tilde

    cl_codes, cl_uniques = pd.factorize(cl_series)
    n_cl = len(cl_uniques)
    n_obs, k_full = len(y_vec), q_mat.shape[1] + 1
    dof_mult = (n_cl / (n_cl - 1.0)) * ((n_obs - 1.0) / (n_obs - k_full))

    def _t_cl(e_vec: np.ndarray, b_val: float) -> float:
        cl_s = np.bincount(cl_codes, weights=x_tilde * e_vec, minlength=n_cl)
        se = np.sqrt(np.sum(cl_s**2) * dof_mult) / denom
        return float(b_val / se) if se > 0 else 0.0

    t_obs = _t_cl(resid_obs, beta_obs)
    rng = np.random.default_rng(seed)
    t_boot = np.empty(b_reps, dtype=np.float64)
    for bi in range(b_reps):
        w_cl = rng.choice([-1.0, 1.0], size=n_cl)
        y_star = y_tilde * w_cl[cl_codes]
        y_star_m = y_star - q_mat @ (q_mat.T @ y_star)
        b_star = float(np.dot(x_tilde, y_star_m) / denom)
        e_star = y_star_m - b_star * x_tilde
        t_boot[bi] = _t_cl(e_star, b_star)

    p_w = float((np.sum(np.abs(t_boot) >= abs(t_obs)) + 1) / (b_reps + 1))
    mc_se = float(np.sqrt(p_w * (1.0 - p_w) / b_reps))
    return {
        "n_obs": int(n_obs),
        "n_total_clusters": int(n_cl),
        "beta_obs_usd": round(beta_obs, 4),
        "t_obs": round(t_obs, 4),
        "B": int(b_reps),
        "exact_MZ_reprojection": True,
        "full_dof_correction": True,
        "p_wild": round(p_w, 4),
        "mc_se": round(mc_se, 4),
    }


def _get_tuned_hgb_params() -> dict[str, Any]:
    """Đọc siêu tham số M7 từ best_model.joblib (fallback nếu chưa fit)."""
    defaults: dict[str, Any] = {
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
    }
    best_path = MODELS_DIR / "best_model.joblib"
    if best_path.exists():
        try:
            loaded = joblib.load(best_path)
            reg = loaded.named_steps.get("reg")
            if reg is not None:
                params = reg.get_params()
                return {k: params[k] for k in defaults if k in params}
        except Exception:
            pass
    return defaults


def _make_native_quantile_pipe(alpha_q: float) -> Pipeline:
    """
    Bộ dự báo phân vị HistGBR dùng pandas category gốc — đồng nhất 100% với M7
    & evaluation.py.
    """
    prep = build_stage4_preprocessors()["m7_hgb_native"]
    hgb_params = _get_tuned_hgb_params()
    reg = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=alpha_q,
        categorical_features="from_dtype",
        max_iter=220,
        **hgb_params,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", prep), ("reg", reg)])


# =============================================================================
# 1. PHỄU LÀM SẠCH, THIÊN LỆCH CHỌN MẪU & CRIT-03 / H4 PROVENANCE
# =============================================================================
def compute_cleaning_waterfall_and_selection_bias(
    master: pd.DataFrame,
) -> dict[str, Any]:
    """
    Truy vết chính xác phễu 693,071 -> 270,898 (tuân thủ CRIT-03: không
    fillna(1.0) mù trên Lyft).
    """
    raw = pd.read_csv(RAW_DIR / "cab_rides.csv", low_memory=False)
    n0 = len(raw)
    df = raw.copy()

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["cab_type_rec"] = df["product_id"].map(
        lambda pid: PRODUCT_MAP.get(pid, ("Uber", "Unknown"))[0]
    )
    df["name_rec"] = df["product_id"].map(
        lambda pid: PRODUCT_MAP.get(pid, ("Uber", "Unknown"))[1]
    )
    df["source_std"] = standardize_location(df["source"])
    df["dest_std"] = standardize_location(df["destination"])

    for col in ["distance", "price", "surge_multiplier"]:
        df[col + "_num"] = df[col].apply(clean_unit_string)

    is_taxi = df["name_rec"] == "Taxi"
    n_taxi = int(is_taxi.sum())
    df_nt = df.loc[~is_taxi].copy()
    n_after_taxi = len(df_nt)
    price_raw_before_fix = df_nt["price_num"].copy()

    df_nt.loc[
        df_nt["price_num"].isin([-1, -1.0, -100, 1000, 9999, 10000, 100000]),
        "price_num",
    ] = np.nan
    df_nt.loc[
        df_nt["distance_num"].isin([-1, -1.0, 1000, 9999, 10000, 100000]),
        "distance_num",
    ] = np.nan
    df_nt.loc[
        df_nt["surge_multiplier_num"].isin([-1, -1.0, 9999, 10000, 100000]),
        "surge_multiplier_num",
    ] = np.nan

    for col in ("distance_num", "price_num", "surge_multiplier_num"):
        df_nt[col] = df_nt[col].abs()

    # [CRIT-03 FIX] Giữ nguyên NaN cho Lyft khuyết/sentinel surge; Uber API
    # luôn cố định 1.0x
    df_nt.loc[
        df_nt["surge_multiplier_num"] >= 1000, "surge_multiplier_num"
    ] /= 1000.0
    df_nt.loc[
        df_nt["surge_multiplier_num"] >= 100, "surge_multiplier_num"
    ] /= 100.0
    df_nt.loc[df_nt["surge_multiplier_num"] >= 10, "surge_multiplier_num"] /= (
        10.0
    )
    df_nt.loc[
        (df_nt["surge_multiplier_num"] < 1.0)
        | (df_nt["surge_multiplier_num"] > 3.0),
        "surge_multiplier_num",
    ] = np.nan
    df_nt.loc[df_nt["cab_type_rec"] == "Uber", "surge_multiplier_num"] = 1.0
    df_nt["is_surge_original"] = (
        df_nt["surge_multiplier_num"].notna().astype(np.int8)
    )

    route_ref = pd.Series(2.15, index=df_nt.index, dtype=np.float64)
    df_nt["distance_num"], _ = recover_scaled_series(
        df_nt["distance_num"], route_ref, 0.02, 8.0, preserve_decimals=2
    )

    # Khớp đúng clean_raw_data.py v5: học giá tham chiếu trên is_surge_original
    # == 1, dùng surge_for_ref tạm cục bộ
    normal_price = df_nt["price_num"].where(
        (df_nt["price_num"] >= 2.5)
        & (df_nt["price_num"] <= 65.0)
        & (df_nt["is_surge_original"] == 1)
    )
    base_p_obs = normal_price / df_nt["surge_multiplier_num"]
    base_p_route = base_p_obs.groupby(
        [df_nt["name_rec"], df_nt["source_std"], df_nt["dest_std"]]
    ).transform("median")
    base_p_tier = base_p_obs.groupby(df_nt["name_rec"]).transform("median")
    surge_for_ref = df_nt["surge_multiplier_num"].fillna(1.0)
    price_ref = base_p_route.fillna(base_p_tier).fillna(13.5) * surge_for_ref
    df_nt["price_num"], _ = recover_scaled_series(
        df_nt["price_num"], price_ref, 2.5, 95.0, preserve_decimals=2
    )

    valid_all = (
        df_nt["price_num"].notna()
        & df_nt["source_std"].notna()
        & df_nt["dest_std"].notna()
    )
    scaled_mask = (
        valid_all
        & price_raw_before_fix.notna()
        & (price_raw_before_fix > 0)
        & (
            (price_raw_before_fix / df_nt["price_num"])
            .round()
            .isin([10, 100, 1000])
        )
    )
    ex_df = df_nt.loc[
        scaled_mask,
        ["cab_type_rec", "name_rec", "source_std", "dest_std", "price_num"],
    ].copy()
    ex_df["raw_price"] = price_raw_before_fix.loc[scaled_mask]
    ex_df["ref_price"] = price_ref.loc[scaled_mask]
    ex_df["factor"] = (
        (ex_df["raw_price"] / ex_df["price_num"]).round().astype(int)
    )
    examples = []
    for fac in [10, 100, 1000]:
        for _, r in ex_df.loc[ex_df["factor"] == fac].head(2).iterrows():
            examples.append(
                {
                    "tier": str(r["name_rec"]),
                    "route": f"{r['source_std']} → {r['dest_std']}",
                    "raw_price_usd": round(float(r["raw_price"]), 2),
                    "ref_median_usd": round(float(r["ref_price"]), 2),
                    "adjusted_price_usd": round(float(r["price_num"]), 2),
                    "factor": int(r["factor"]),
                }
            )

    has_price = df_nt["price_num"].notna()
    n_drop_price = int((~has_price).sum())
    df_p = df_nt.loc[has_price]
    has_src = df_p["source_std"].notna()
    n_drop_src = int((~has_src).sum())
    df_ps = df_p.loc[has_src]
    has_dst = df_ps["dest_std"].notna()
    n_drop_dst = int((~has_dst).sum())
    df_final = df_ps.loc[has_dst]

    assert len(df_final) == 270_898, f"Lệch phễu: {len(df_final)}"
    assert n_taxi + n_drop_price + n_drop_src + n_drop_dst == n0 - 270_898

    kept_nt = df_nt.loc[valid_all]
    dropped_nt = df_nt.loc[~valid_all]

    # Tính tỷ lệ surge Lyft trên đúng nhóm có surge quan sát thực
    # (is_surge_original == 1)
    lyft_k = kept_nt.loc[
        (kept_nt["cab_type_rec"] == "Lyft")
        & (kept_nt["is_surge_original"] == 1)
    ]
    lyft_d = dropped_nt.loc[
        (dropped_nt["cab_type_rec"] == "Lyft")
        & (dropped_nt["is_surge_original"] == 1)
    ]

    return {
        "raw_cab_size_mib": round(
            (RAW_DIR / "cab_rides.csv").stat().st_size / (1024**2), 2
        ),
        "waterfall": [
            {
                "step": "0. Dữ liệu gốc (cab_rides.csv)",
                "in_rows": n0,
                "dropped": 0,
                "out_rows": n0,
                "share_raw_pct": 100.0,
            },
            {
                "step": "1. Khôi phục nhãn & chuẩn hóa chuỗi/lính canh/quy mô",
                "in_rows": n0,
                "dropped": 0,
                "out_rows": n0,
                "share_raw_pct": 100.0,
            },
            {
                "step": "2. Tách riêng Uber Taxi (MNAR — khuyết giá 100%)",
                "in_rows": n0,
                "dropped": n_taxi,
                "out_rows": n_after_taxi,
                "share_raw_pct": round(n_after_taxi / n0 * 100, 2),
            },
            {
                "step": (
                    "3. Loại bản ghi khuyết/mã lính canh biến mục tiêu price"
                ),
                "in_rows": n_after_taxi,
                "dropped": n_drop_price,
                "out_rows": len(df_p),
                "share_raw_pct": round(len(df_p) / n0 * 100, 2),
            },
            {
                "step": "4. Loại bản ghi khuyết điểm đón (source)",
                "in_rows": len(df_p),
                "dropped": n_drop_src,
                "out_rows": len(df_ps),
                "share_raw_pct": round(len(df_ps) / n0 * 100, 2),
            },
            {
                "step": "5. Loại bản ghi khuyết điểm trả (destination)",
                "in_rows": len(df_ps),
                "dropped": n_drop_dst,
                "out_rows": len(df_final),
                "share_raw_pct": round(len(df_final) / n0 * 100, 2),
            },
            {
                "step": "6. Ghép nối thời tiết Left Join (0 fan-out)",
                "in_rows": len(df_final),
                "dropped": 0,
                "out_rows": len(df_final),
                "share_raw_pct": round(len(df_final) / n0 * 100, 2),
            },
        ],
        "total_dropped": n0 - len(df_final),
        "master_provenance_breakdown": {
            "total_master_rows": int(len(master)),
            "is_surge_original_1_total": int(
                (master["is_surge_original"] == 1).sum()
            ),
            "uber_api_fixed_1x": int(
                (master["cab_type"].astype(str) == "Uber").sum()
            ),
            "lyft_total": int(
                (master["cab_type"].astype(str) == "Lyft").sum()
            ),
            "lyft_surge_observed": int(
                (
                    (master["cab_type"].astype(str) == "Lyft")
                    & (master["is_surge_original"] == 1)
                ).sum()
            ),
            "lyft_surge_missing_nan": int(
                (
                    (master["cab_type"].astype(str) == "Lyft")
                    & (master["is_surge_original"] == 0)
                ).sum()
            ),
            "lyft_surge_missing_pct": round(
                float(
                    (
                        (master["cab_type"].astype(str) == "Lyft")
                        & (master["is_surge_original"] == 0)
                    ).sum()
                )
                / float((master["cab_type"].astype(str) == "Lyft").sum())
                * 100.0,
                2,
            ),
            "is_timestamp_original_1": int(
                (master["is_timestamp_original"] == 1).sum()
            ),
            "weather_matched_1": int((master["weather_matched"] == 1).sum()),
            "weather_valid_rain_obs": int(
                (
                    (master["weather_matched"] == 1)
                    & (master["is_raining"].notna())
                ).sum()
            ),
            "weather_rain_sentinel_nan": int(
                (
                    (master["weather_matched"] == 1)
                    & (master["is_raining"].isna())
                ).sum()
            ),
            "weather_true_rain_rows": int(
                (
                    (master["weather_matched"] == 1)
                    & (master["is_raining"] == 1)
                ).sum()
            ),
        },
        "selection_bias_audit": {
            "kept_n": int(len(kept_nt)),
            "dropped_non_taxi_n": int(len(dropped_nt)),
            "uber_share_kept_pct": round(
                float((kept_nt["cab_type_rec"] == "Uber").mean() * 100), 2
            ),
            "uber_share_dropped_pct": round(
                float((dropped_nt["cab_type_rec"] == "Uber").mean() * 100), 2
            ),
            "lyft_surge_observed_kept_pct": round(
                float((lyft_k["surge_multiplier_num"] > 1.0).mean() * 100), 2
            ),
            "lyft_surge_observed_dropped_pct": round(
                float((lyft_d["surge_multiplier_num"] > 1.0).mean() * 100), 2
            ),
            "mean_dist_kept_mi": round(
                float(kept_nt["distance_num"].mean()), 3
            ),
            "mean_dist_dropped_mi": round(
                float(dropped_nt["distance_num"].mean()), 3
            ),
            "mean_price_kept_usd": round(
                float(kept_nt["price_num"].mean()), 2
            ),
            "mean_price_dropped_available_usd": round(
                float(dropped_nt["price_num"].dropna().mean()), 2
            ),
        },
        "scale_recovery_examples": examples[:6],
    }


# =============================================================================
# 2. THỐNG KÊ BỀN VỮNG: CLUSTER SE, TOST, EXACT WCR, H3b CLUSTER, SENTINEL, FDR
# =============================================================================
def compute_statistical_rigor_and_wcr(master: pd.DataFrame) -> dict[str, Any]:
    """
    Toàn bộ kiểm định suy luận bền vững RQ1 & đa kiểm định H1-H4 trên dữ liệu
    sạch v5.
    """
    # Tập thời tiết khớp thực & đã loại ô all-sentinel mưa (is_raining.notna(),
    # n = 152,043)
    wm = master.loc[
        (master["weather_matched"] == 1) & (master["is_raining"].notna())
    ].copy()
    wm["hour_key"] = wm["datetime"].dt.floor("h")
    wm["hour_str"] = wm["hour_key"].astype(str)
    wm["loc_hr_cl"] = wm["source"].astype(str) + "|" + wm["hour_str"]
    wm["route_fe"] = (
        wm["source"].astype(str) + "->" + wm["destination"].astype(str)
    )
    wm["name_str"] = wm["name"].astype(str)

    n_loc_hr = int(wm["loc_hr_cl"].nunique())
    n_hour = int(wm["hour_str"].nunique())

    # 2A. Unadjusted Rain vs Dry trên toàn bộ n = 152,043 (cả loc_hr và
    # citywide hour)
    m_unadj_loc = smf.ols("price ~ is_raining", data=wm).fit(
        cov_type="cluster", cov_kwds={"groups": wm["loc_hr_cl"]}
    )
    m_unadj_hr = smf.ols("price ~ is_raining", data=wm).fit(
        cov_type="cluster", cov_kwds={"groups": wm["hour_str"]}
    )

    # 2B. Controlled OLS KHÔNG có surge (lọc distance hợp lệ) & CÓ surge
    # (wm_obs_dist: is_surge_original == 1 & distance.notna())
    wm_dist = wm.loc[wm["distance"].notna()].copy()
    n_dist_loc = int(wm_dist["loc_hr_cl"].nunique())
    n_dist_hr = int(wm_dist["hour_str"].nunique())
    m_ctrl_nosurge_hr = smf.ols(
        "price ~ is_raining + C(name_str) + distance + C(route_fe)",
        data=wm_dist,
    ).fit(cov_type="cluster", cov_kwds={"groups": wm_dist["hour_str"]})

    wm_obs = wm.loc[wm["is_surge_original"] == 1].copy()
    wm_obs_dist = wm_obs.loc[wm_obs["distance"].notna()].copy()
    n_obs_loc = int(wm_obs_dist["loc_hr_cl"].nunique())
    n_obs_hr = int(wm_obs_dist["hour_str"].nunique())
    m_ctrl_loc = smf.ols(
        "price ~ is_raining + C(name_str) + distance + surge_multiplier + "
        "C(route_fe)",
        data=wm_obs_dist,
    ).fit(cov_type="cluster", cov_kwds={"groups": wm_obs_dist["loc_hr_cl"]})
    m_ctrl_hr = smf.ols(
        "price ~ is_raining + C(name_str) + distance + surge_multiplier + "
        "C(route_fe)",
        data=wm_obs_dist,
    ).fit(cov_type="cluster", cov_kwds={"groups": wm_obs_dist["hour_str"]})

    # 2C. Kiểm định cấp cụm tổng hợp
    cl_agg = wm.groupby("loc_hr_cl", observed=True).agg(
        mean_price=("price", "mean"),
        is_raining=("is_raining", "first"),
        n_obs=("price", "size"),
    )
    r_cl = cl_agg.loc[cl_agg["is_raining"] == 1, "mean_price"]
    d_cl = cl_agg.loc[cl_agg["is_raining"] == 0, "mean_price"]
    t_cl_agg, p_cl_agg = stats.ttest_ind(r_cl, d_cl, equal_var=False)

    # 2D. Phân tầng cường độ mưa (tỷ lệ surge Lyft chỉ tính trên
    # is_surge_original == 1)
    wm["rain_tier"] = pd.cut(
        wm["rain"],
        bins=[-0.001, 0.0, 0.05, 0.10, 1.0],
        labels=[
            "Khong_mua_0",
            "Mua_nhe_0_0.05",
            "Mua_vua_0.05_0.10",
            "Mua_lon_ge_0.10",
        ],
    )
    rain_strata = []
    for lbl, grp in wm.groupby("rain_tier", observed=True):
        lyft_obs_sub = grp.loc[
            (grp["cab_type"].astype(str) == "Lyft")
            & (grp["is_surge_original"] == 1)
        ]
        rain_strata.append(
            {
                "stratum": str(lbl),
                "n_obs": int(len(grp)),
                "n_clusters": int(grp["loc_hr_cl"].nunique()),
                "mean_price_usd": round(float(grp["price"].mean()), 4),
                "median_price_usd": round(float(grp["price"].median()), 2),
                "n_lyft_surge_obs": int(len(lyft_obs_sub)),
                "lyft_surge_rate_pct": round(
                    float(
                        (lyft_obs_sub["surge_multiplier"] > 1.0).mean() * 100.0
                    ),
                    2,
                )
                if len(lyft_obs_sub)
                else 0.0,
            }
        )

    # 2E. Mưa lớn (>= 0.10 in/h) vs Không mưa: Cluster OLS + Exact WCR (B =
    # 1,999, full DOF)
    wm_hv = wm_obs.loc[
        wm_obs["distance"].notna()
        & ((wm_obs["rain"] == 0.0) | (wm_obs["rain"] >= 0.10))
    ].copy()
    wm_hv["is_heavy"] = (wm_hv["rain"] >= 0.10).astype(np.float64)
    m_heavy_loc = smf.ols(
        "price ~ is_heavy + C(name_str) + distance + surge_multiplier + "
        "C(route_fe)",
        data=wm_hv,
    ).fit(cov_type="cluster", cov_kwds={"groups": wm_hv["loc_hr_cl"]})
    m_heavy_hr = smf.ols(
        "price ~ is_heavy + C(name_str) + distance + surge_multiplier + "
        "C(route_fe)",
        data=wm_hv,
    ).fit(cov_type="cluster", cov_kwds={"groups": wm_hv["hour_str"]})

    z_df = pd.get_dummies(
        wm_hv[["name_str", "route_fe"]], drop_first=True, dtype=np.float64
    )
    z_df["distance"] = wm_hv["distance"].astype(np.float64).values
    z_df["surge_multiplier"] = (
        wm_hv["surge_multiplier"].astype(np.float64).values
    )
    z_mat = np.column_stack(
        [np.ones(len(wm_hv), dtype=np.float64), z_df.values]
    )
    y_hv = wm_hv["price"].to_numpy(dtype=np.float64)
    x_hv = wm_hv["is_heavy"].to_numpy(dtype=np.float64)

    wcr_loc = _run_exact_wcr_bootstrap(
        y_hv, x_hv, z_mat, wm_hv["loc_hr_cl"], b_reps=1999
    )
    wcr_hr = _run_exact_wcr_bootstrap(
        y_hv, x_hv, z_mat, wm_hv["hour_str"], b_reps=1999
    )

    # 2F. Kiểm định gom cụm cho H3b (Lyft Surge Probability vs Rain) — giải mã
    # p_iid = 0.0173!
    lyft_w = wm_obs.loc[wm_obs["cab_type"].astype(str) == "Lyft"].copy()
    lyft_w["has_surge"] = (lyft_w["surge_multiplier"] > 1.0).astype(float)
    m_h3b_iid = smf.ols("has_surge ~ is_raining", data=lyft_w).fit()
    m_h3b_loc = smf.ols("has_surge ~ is_raining", data=lyft_w).fit(
        cov_type="cluster", cov_kwds={"groups": lyft_w["loc_hr_cl"]}
    )
    m_h3b_hr = smf.ols("has_surge ~ is_raining", data=lyft_w).fit(
        cov_type="cluster", cov_kwds={"groups": lyft_w["hour_str"]}
    )
    h3b_cluster_res = {
        "n_lyft_obs": int(len(lyft_w)),
        "n_loc_hr_clusters": int(lyft_w["loc_hr_cl"].nunique()),
        "n_hour_clusters": int(lyft_w["hour_str"].nunique()),
        "surge_rate_rain_pct": round(
            float(
                lyft_w.loc[lyft_w["is_raining"] == 1, "has_surge"].mean() * 100
            ),
            2,
        ),
        "surge_rate_dry_pct": round(
            float(
                lyft_w.loc[lyft_w["is_raining"] == 0, "has_surge"].mean() * 100
            ),
            2,
        ),
        "diff_pp": round(float(m_h3b_iid.params["is_raining"] * 100.0), 3),
        "p_naive_iid": round(float(m_h3b_iid.pvalues["is_raining"]), 4),
        "se_loc_hr_pp": round(float(m_h3b_loc.bse["is_raining"] * 100.0), 3),
        "p_loc_hr_cluster": round(float(m_h3b_loc.pvalues["is_raining"]), 4),
        "se_hour_pp": round(float(m_h3b_hr.bse["is_raining"] * 100.0), 3),
        "p_hour_cluster": round(float(m_h3b_hr.pvalues["is_raining"]), 4),
    }

    # 2G. Độ nhạy loại trừ triệt để mọi ô (quận × giờ) từng chứa dù chỉ 1 dòng
    # sentinel mưa trong weather.csv
    rw = pd.read_csv(RAW_DIR / "weather.csv", low_memory=False)
    rw["loc_std"] = standardize_location(rw["location"])
    rw["ts_num"] = rw["time_stamp"].apply(clean_unit_string)
    rw["dt_hr"] = (
        pd.to_datetime(rw["ts_num"], unit="s", utc=True, errors="coerce")
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.floor("h")
    )
    s_str = rw["rain"].astype(str).str.strip().str.lower()
    str_set = {"?", "-", "unknown", "chưa rõ", "nan", "null", "none"}
    is_str_sent = rw["rain"].notna() & (
        s_str.isin(str_set) | s_str.str.startswith("nan ")
    )
    r_num = rw["rain"].apply(clean_unit_string)
    is_num_sent = r_num.isin([-1, -1.0, -100, 1000, 9999, 10000, 100000])
    sent_mask = is_str_sent | is_num_sent
    any_sent_cells = set(
        zip(
            rw.loc[sent_mask, "loc_std"].dropna().astype(str),
            rw.loc[sent_mask, "dt_hr"].dropna(),
            strict=False,
        )
    )
    keys_obs = list(
        zip(wm_obs["source"].astype(str), wm_obs["hour_key"], strict=False)
    )
    wm_obs_clean_any = wm_obs.loc[
        [k not in any_sent_cells for k in keys_obs]
    ].copy()
    wm_obs_clean_any = wm_obs_clean_any.loc[
        wm_obs_clean_any["distance"].notna()
    ].copy()
    m_any_loc = smf.ols(
        "price ~ is_raining + C(name_str) + distance + surge_multiplier + "
        "C(route_fe)",
        data=wm_obs_clean_any,
    ).fit(
        cov_type="cluster", cov_kwds={"groups": wm_obs_clean_any["loc_hr_cl"]}
    )
    m_any_hr = smf.ols(
        "price ~ is_raining + C(name_str) + distance + surge_multiplier + "
        "C(route_fe)",
        data=wm_obs_clean_any,
    ).fit(
        cov_type="cluster", cov_kwds={"groups": wm_obs_clean_any["hour_str"]}
    )
    n_any_loc = int(wm_obs_clean_any["loc_hr_cl"].nunique())
    n_any_hr = int(wm_obs_clean_any["hour_str"].nunique())

    # 2H. Hiệu chỉnh đa kiểm định Holm-Bonferroni & BH-FDR trên 13 giả thuyết
    # H1-H4
    hyp = pd.read_csv(TABLES_DIR / "stage5_hypothesis_tests.csv")
    raw_p = hyp["parametric_pvalue"].to_numpy(dtype=float)
    _, p_holm, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
    _, q_bh, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")
    multitest_rows = [
        {
            "hypothesis_id": str(r["hypothesis_id"]),
            "raw_p": float(raw_p[i]),
            "p_holm": float(p_holm[i]),
            "q_bh": float(q_bh[i]),
            "reject_holm_05": bool(p_holm[i] < 0.05),
            "reject_bh_05": bool(q_bh[i] < 0.05),
        }
        for i, r in hyp.iterrows()
    ]

    # 2I. Cluster F-test cho H2c (4 buổi EST trên n = 208,034 TS gốc)
    ts_df = master.loc[master["is_timestamp_original"] == 1].copy()
    ts_df["hour_cluster"] = (
        ts_df["source"].astype(str)
        + "|"
        + ts_df["datetime"].dt.floor("h").astype(str)
    )
    ts_df["tod_str"] = ts_df["time_of_day"].astype(str)
    m_tod = smf.ols("price ~ C(tod_str)", data=ts_df).fit(
        cov_type="cluster", cov_kwds={"groups": ts_df["hour_cluster"]}
    )
    wald_tod = m_tod.f_test(np.eye(len(m_tod.params))[1:])
    tod_means = (
        ts_df.groupby("time_of_day", observed=True)["price"].mean().to_dict()
    )

    def _pack_ctrl(
        m_fit: Any,
        df_dof: int,
        n_loc_cl: int | None = None,
        n_hr_cl: int | None = None,
    ) -> dict[str, Any]:
        b = float(m_fit.params["is_raining"])
        se = float(m_fit.bse["is_raining"])
        ci = m_fit.conf_int().loc["is_raining"]
        res_d: dict[str, Any] = {
            "n_obs": int(m_fit.nobs),
            "beta_usd": round(b, 4),
            "se": round(se, 4),
            "ci95": [round(float(ci.iloc[0]), 4), round(float(ci.iloc[1]), 4)],
            "p_value": round(float(m_fit.pvalues["is_raining"]), 4),
            "tost_010": _tost_test(b, se, 0.10, df_dof),
            "tost_025": _tost_test(b, se, 0.25, df_dof),
        }
        if n_loc_cl is not None:
            res_d["n_loc_hr_clusters"] = int(n_loc_cl)
        if n_hr_cl is not None:
            res_d["n_hour_clusters"] = int(n_hr_cl)
        return res_d

    return {
        "n_valid_rain_obs": int(len(wm)),
        "n_loc_hr_clusters_all": n_loc_hr,
        "n_citywide_hour_clusters_all": n_hour,
        "n_valid_rain_dist_obs": int(len(wm_dist)),
        "n_loc_hr_clusters_dist": n_dist_loc,
        "n_citywide_hour_clusters_dist": n_dist_hr,
        "n_valid_rain_and_surge_obs": int(len(wm_obs)),
        "n_valid_rain_and_surge_dist_obs": int(len(wm_obs_dist)),
        "n_loc_hr_clusters_obs": n_obs_loc,
        "n_citywide_hour_clusters_obs": n_obs_hr,
        "weather_unadjusted_loc_hr": {
            "n_obs": int(m_unadj_loc.nobs),
            "diff_usd": round(float(m_unadj_loc.params["is_raining"]), 4),
            "se_cluster": round(float(m_unadj_loc.bse["is_raining"]), 4),
            "ci95_cluster": [
                round(float(m_unadj_loc.conf_int().loc["is_raining", 0]), 4),
                round(float(m_unadj_loc.conf_int().loc["is_raining", 1]), 4),
            ],
            "p_cluster": round(float(m_unadj_loc.pvalues["is_raining"]), 4),
            "tost_025": _tost_test(
                float(m_unadj_loc.params["is_raining"]),
                float(m_unadj_loc.bse["is_raining"]),
                0.25,
                n_loc_hr - 1,
            ),
        },
        "weather_unadjusted_citywide_hour": {
            "n_obs": int(m_unadj_hr.nobs),
            "diff_usd": round(float(m_unadj_hr.params["is_raining"]), 4),
            "se_cluster": round(float(m_unadj_hr.bse["is_raining"]), 4),
            "p_cluster": round(float(m_unadj_hr.pvalues["is_raining"]), 4),
        },
        "weather_controlled_loc_hr": _pack_ctrl(
            m_ctrl_loc, n_obs_loc - 1, n_loc_cl=n_obs_loc, n_hr_cl=n_obs_hr
        ),
        "weather_controlled_citywide_hour": _pack_ctrl(
            m_ctrl_hr, n_obs_hr - 1, n_loc_cl=n_obs_loc, n_hr_cl=n_obs_hr
        ),
        "weather_controlled_nosurge_full152k_hour": _pack_ctrl(
            m_ctrl_nosurge_hr,
            n_dist_hr - 1,
            n_loc_cl=n_dist_loc,
            n_hr_cl=n_dist_hr,
        ),
        "cluster_level_aggregated_test": {
            "n_rain_clusters": int(len(r_cl)),
            "n_dry_clusters": int(len(d_cl)),
            "mean_rain_cluster_usd": round(float(r_cl.mean()), 4),
            "mean_dry_cluster_usd": round(float(d_cl.mean()), 4),
            "welch_t": round(float(t_cl_agg), 4),
            "welch_p": round(float(p_cl_agg), 4),
        },
        "rain_intensity_strata": rain_strata,
        "heavy_rain_vs_dry": {
            "n_heavy_obs": int((wm_hv["is_heavy"] == 1).sum()),
            "n_heavy_loc_hr_clusters": int(
                wm_hv.loc[wm_hv["is_heavy"] == 1, "loc_hr_cl"].nunique()
            ),
            "n_heavy_citywide_hours": int(
                wm_hv.loc[wm_hv["is_heavy"] == 1, "hour_str"].nunique()
            ),
            "ctrl_beta_usd": round(float(m_heavy_loc.params["is_heavy"]), 4),
            "ctrl_se_loc_hr": round(float(m_heavy_loc.bse["is_heavy"]), 4),
            "ctrl_p_loc_hr": round(float(m_heavy_loc.pvalues["is_heavy"]), 4),
            "ctrl_se_hour": round(float(m_heavy_hr.bse["is_heavy"]), 4),
            "ctrl_p_hour": round(float(m_heavy_hr.pvalues["is_heavy"]), 4),
            "exact_wcr_loc_hr_B1999": wcr_loc,
            "exact_wcr_citywide_hour_B1999": wcr_hr,
        },
        "h3b_lyft_surge_rain_cluster": h3b_cluster_res,
        "weather_sentinel_strict_sensitivity": {
            "weather_raw_total_sentinels": int(sent_mask.sum()),
            "n_obs_excl_any_sentinel_cell": int(len(wm_obs_clean_any)),
            "n_loc_hr_clusters": n_any_loc,
            "n_hour_clusters": n_any_hr,
            "beta_usd": round(float(m_any_loc.params["is_raining"]), 4),
            "se_loc_hr": round(float(m_any_loc.bse["is_raining"]), 4),
            "p_loc_hr": round(float(m_any_loc.pvalues["is_raining"]), 4),
            "se_hour": round(float(m_any_hr.bse["is_raining"]), 4),
            "p_hour": round(float(m_any_hr.pvalues["is_raining"]), 4),
            "p_tost_010_hour": _tost_test(
                float(m_any_hr.params["is_raining"]),
                float(m_any_hr.bse["is_raining"]),
                0.10,
                n_any_hr - 1,
            )["p_tost"],
        },
        "time_of_day_cluster_f_test": {
            "n_obs": int(len(ts_df)),
            "n_clusters": int(ts_df["hour_cluster"].nunique()),
            "f_stat_cluster": round(
                float(np.asarray(wald_tod.fvalue).item()), 4
            ),
            "p_cluster": round(float(np.asarray(wald_tod.pvalue).item()), 4),
            "max_min_gap_usd": round(
                float(max(tod_means.values()) - min(tod_means.values())), 4
            ),
        },
        "multiple_testing_13": multitest_rows,
    }


# =============================================================================
# 3. RQ2: ĐIỂM CẮT BIỂU GIÁ 5 CẶP — ĐẶC TẢ A & ĐẶC TẢ B (X_TRAIN 80% + FULL)
# =============================================================================
def compute_crossing_distances_spec_a_and_b(
    master: pd.DataFrame, df_feat: pd.DataFrame, splits: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Tính điểm cắt d* kèm 95% CI phương pháp Delta (HC3 & Cluster-Robust) cho cả
    5 cặp:
    - Spec A (X_train 80%, is_surge_original == 1): Price ~ 1 + (distance *
      surge)
    - Spec B Train-80% (parity): Price ~ 1 + distance trên X_train & surge ==
      1.0
    - Spec B Full Observed Non-Surge: Price ~ 1 + distance trên toàn mẫu surge
      == 1.0 (is_surge_original == 1)
    """
    tr_idx, _ = splits["random_80_20"]
    df_tr = df_feat.iloc[tr_idx].copy()
    df_tr["cl_key"] = np.where(
        df_tr["is_timestamp_original"] == 1,
        df_tr["source"].astype(str)
        + "|"
        + df_tr["datetime"].dt.floor("h").astype(str),
        df_tr["source"].astype(str)
        + "|NO_TS_"
        + df_tr["destination"].astype(str),
    )
    df_tr_ns = df_tr.loc[df_tr["surge_multiplier"] == 1.0].copy()

    ns_full = master.loc[
        (master["is_surge_original"] == 1)
        & (master["surge_multiplier"] == 1.0)
    ].copy()
    ns_full["cl_key"] = np.where(
        ns_full["is_timestamp_original"] == 1,
        ns_full["source"].astype(str)
        + "|"
        + ns_full["datetime"].dt.floor("h").astype(str),
        ns_full["source"].astype(str)
        + "|NO_TS_"
        + ns_full["destination"].astype(str),
    )

    pairs = [
        ("1. Đi ghép", "UberPool", "Shared"),
        ("2. Phổ thông 4 chỗ", "UberX", "Lyft"),
        ("3. Gia đình 6 chỗ", "UberXL", "Lyft XL"),
        ("4. Hạng sang Sedan", "Black", "Lux Black"),
        ("5. Hạng sang SUV", "Black SUV", "Lux Black XL"),
    ]

    def _fit_delta(
        df_in: pd.DataFrame, x_col: str, cov_mode: str
    ) -> dict[str, Any]:
        sub_f = df_in.dropna(subset=[x_col, "price"]).copy()
        sub_f["x_var"] = sub_f[x_col].astype(float)
        sub_f["x_lyft"] = sub_f["x_var"] * sub_f["is_lyft_tier"]
        mod = smf.ols("price ~ x_var + is_lyft_tier + x_lyft", data=sub_f)
        fit = (
            mod.fit(cov_type="HC3")
            if cov_mode == "HC3"
            else mod.fit(
                cov_type="cluster", cov_kwds={"groups": sub_f["cl_key"]}
            )
        )
        b0_u = float(fit.params["Intercept"])
        b1_u = float(fit.params["x_var"])
        b0_l = b0_u + float(fit.params["is_lyft_tier"])
        b1_l = b1_u + float(fit.params["x_lyft"])
        a_val = -float(fit.params["is_lyft_tier"])
        b_val = float(fit.params["x_lyft"])
        d_star = a_val / b_val if abs(b_val) > 1e-9 else np.nan
        cov = fit.cov_params()
        var_a = float(cov.loc["is_lyft_tier", "is_lyft_tier"])
        var_b = float(cov.loc["x_lyft", "x_lyft"])
        cov_ab = -float(cov.loc["is_lyft_tier", "x_lyft"])
        var_d = (
            (1.0 / b_val) ** 2 * var_a
            + (a_val / (b_val**2)) ** 2 * var_b
            - 2.0 * (a_val / (b_val**3)) * cov_ab
        )
        se_d = float(np.sqrt(max(var_d, 0.0)))
        return {
            "n": int(len(sub_f)),
            "b0_uber": round(b0_u, 4),
            "b1_uber": round(b1_u, 4),
            "b0_lyft": round(b0_l, 4),
            "b1_lyft": round(b1_l, 4),
            "d_star": round(d_star, 3),
            "se": round(se_d, 3),
            "ci95": [
                round(d_star - 1.96 * se_d, 2),
                round(d_star + 1.96 * se_d, 2),
            ],
        }

    out = []
    for seg, ut, lt in pairs:
        sub_a = df_tr.loc[df_tr["name"].isin([ut, lt])].copy()
        sub_a["is_lyft_tier"] = (sub_a["name"] == lt).astype(int)
        a_hc3 = _fit_delta(sub_a, "distance_x_surge", "HC3")
        a_cl = _fit_delta(sub_a, "distance_x_surge", "cluster")
        a_in = bool(0.02 <= a_hc3["d_star"] <= 7.86)
        sub_a_d = sub_a["distance"].dropna()
        tr_d_all = df_tr["distance"].dropna()
        a_share = (
            round(float((sub_a_d < a_hc3["d_star"]).mean() * 100.0), 1)
            if a_in
            else None
        )
        a_longer = (
            round(float((sub_a_d > a_hc3["d_star"]).mean() * 100.0), 1)
            if a_in
            else None
        )
        a_all_shorter = (
            round(float((tr_d_all < a_hc3["d_star"]).mean() * 100.0), 1)
            if a_in
            else None
        )
        a_all_longer = (
            round(float((tr_d_all > a_hc3["d_star"]).mean() * 100.0), 1)
            if a_in
            else None
        )

        sub_b80 = df_tr_ns.loc[df_tr_ns["name"].isin([ut, lt])].copy()
        sub_b80["is_lyft_tier"] = (sub_b80["name"] == lt).astype(int)
        b80_hc3 = _fit_delta(sub_b80, "distance", "HC3")
        b80_cl = _fit_delta(sub_b80, "distance", "cluster")
        b80_in = bool(0.02 <= b80_hc3["d_star"] <= 7.86)
        sub_b80_d = sub_b80["distance"].dropna()
        tr_ns_d_all = df_tr_ns["distance"].dropna()
        b80_share = (
            round(
                float((sub_b80_d < b80_hc3["d_star"]).mean() * 100.0),
                1,
            )
            if b80_in
            else None
        )
        b80_longer = (
            round(
                float((sub_b80_d > b80_hc3["d_star"]).mean() * 100.0),
                1,
            )
            if b80_in
            else None
        )
        b80_all_shorter = (
            round(
                float((tr_ns_d_all < b80_hc3["d_star"]).mean() * 100.0),
                1,
            )
            if b80_in
            else None
        )
        b80_all_longer = (
            round(
                float((tr_ns_d_all > b80_hc3["d_star"]).mean() * 100.0),
                1,
            )
            if b80_in
            else None
        )

        sub_bf = ns_full.loc[ns_full["name"].isin([ut, lt])].copy()
        sub_bf["is_lyft_tier"] = (sub_bf["name"] == lt).astype(int)
        bf_hc3 = _fit_delta(sub_bf, "distance", "HC3")
        bf_cl = _fit_delta(sub_bf, "distance", "cluster")
        bf_in = bool(0.02 <= bf_hc3["d_star"] <= 7.86)
        sub_bf_d = sub_bf["distance"].dropna()
        bf_share = (
            round(
                float((sub_bf_d < bf_hc3["d_star"]).mean() * 100.0),
                1,
            )
            if bf_in
            else None
        )
        bf_longer = (
            round(
                float((sub_bf_d > bf_hc3["d_star"]).mean() * 100.0),
                1,
            )
            if bf_in
            else None
        )

        # Kiểm tra độ nhạy ngoài vùng giá sàn (distance >= 1.0 dặm) cho đủ cả 5
        # cặp đối xứng
        sub_af = sub_bf.loc[sub_bf["distance"] >= 1.0].copy()
        af_hc3 = _fit_delta(sub_af, "distance", "HC3")
        af_cl = _fit_delta(sub_af, "distance", "cluster")
        above_floor = {
            "n_ge_1mi": int(len(sub_af)),
            "b0_uber_ge_1mi": af_hc3["b0_uber"],
            "b1_uber_ge_1mi": af_hc3["b1_uber"],
            "b0_lyft_ge_1mi": af_hc3["b0_lyft"],
            "b1_lyft_ge_1mi": af_hc3["b1_lyft"],
            "d_star_ge_1mi": af_hc3["d_star"],
            "ci95_ge_1mi": af_hc3["ci95"],
            "ci95_cluster_ge_1mi": af_cl["ci95"],
            "in_domain_ge_1mi": bool(1.0 <= af_hc3["d_star"] <= 7.86),
        }

        out.append(
            {
                "segment": seg,
                "uber_tier": ut,
                "lyft_tier": lt,
                "spec_a_train80": {
                    "in_domain": a_in,
                    "hc3": a_hc3,
                    "cluster": a_cl,
                    "share_shorter_pct": a_share,
                    "share_longer_pct": a_longer,
                    "all_train_share_shorter_pct": a_all_shorter,
                    "all_train_share_longer_pct": a_all_longer,
                },
                "spec_b_train80_nonsurge": {
                    "in_domain": b80_in,
                    "hc3": b80_hc3,
                    "cluster": b80_cl,
                    "share_shorter_pct": b80_share,
                    "share_longer_pct": b80_longer,
                    "all_train_share_shorter_pct": b80_all_shorter,
                    "all_train_share_longer_pct": b80_all_longer,
                },
                "spec_b_full_nonsurge": {
                    "in_domain": bf_in,
                    "hc3": bf_hc3,
                    "cluster": bf_cl,
                    "share_shorter_pct": bf_share,
                    "share_longer_pct": bf_longer,
                },
                "above_floor_check": above_floor,
            }
        )
    return out


# =============================================================================
# 4. CHẨN ĐOÁN MÔ HÌNH, MONOTONIC SPLIT-CQR (NATIVE CAT), GROUPED XAI & LEAKAGE
# =============================================================================
def compute_model_diagnostics_cqr_and_xai(
    master: pd.DataFrame,
    df_feat: pd.DataFrame,
    valid_cols: list[str],
    splits: dict[str, Any],
) -> dict[str, Any]:
    """
    Chạy hợp nhất M7 vs M6 paired, M0b vs M7, Monotonic Split-CQR (native cat),
    XAI, API-Batch & Strict Train-Only.
    """
    tr_idx, te_idx = splits["random_80_20"]
    X_tr = df_feat.iloc[tr_idx][valid_cols].copy()
    y_tr = df_feat.iloc[tr_idx]["price"].to_numpy(dtype=float)
    X_te = df_feat.iloc[te_idx][valid_cols].copy()
    y_te = df_feat.iloc[te_idx]["price"].to_numpy(dtype=float)

    for xd in (X_tr, X_te):
        for c in BASE_CAT_FEATURES:
            xd[c] = xd[c].astype("category")

    m7 = joblib.load(MODELS_DIR / "best_model.joblib")
    m2 = joblib.load(MODELS_DIR / "m2_interaction_ols.joblib")
    p7 = m7.predict(X_te)
    p2 = m2.predict(X_te)

    # 4A. Paired 5-Fold M7 vs M6
    s5 = json.loads(
        (RESULTS_DIR / "stage5_summary.json").read_text(encoding="utf-8")
    )
    cv_rows = {r["model_code"]: r for r in s5["part2_cv_results"]}
    f7 = np.asarray(
        cv_rows["M7_HistGradientBoosting"]["fold_rmses"], dtype=float
    )
    f6 = np.asarray(cv_rows["M6_Random_Forest"]["fold_rmses"], dtype=float)
    diffs = f7 - f6
    t_stat, p_paired = stats.ttest_rel(f7, f6)
    rng = np.random.default_rng(RANDOM_STATE)
    boot_means = [
        float(np.mean(rng.choice(diffs, size=len(diffs), replace=True)))
        for _ in range(2000)
    ]

    # 4B. M0b vs M7 by Surge Regime
    m0b = GroupMedianBaselineRegressor(
        group_cols=("name", "source", "destination")
    ).fit(X_tr, y_tr)
    p0b = m0b.predict(X_te)
    surge_mask = X_te["surge_multiplier"].to_numpy(dtype=float) > 1.0

    # 4C. Monotonic Split-CQR (85/15) dùng native pandas category + kiểm toán
    # biên GĐ6 (100% Train)
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_tr,
        y_tr,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=X_tr["name"].astype(str),
    )
    m10 = _make_native_quantile_pipe(0.10).fit(X_fit, y_fit)
    m50 = _make_native_quantile_pipe(0.50).fit(X_fit, y_fit)
    m90 = _make_native_quantile_pipe(0.90).fit(X_fit, y_fit)

    cal_lo = np.minimum(m10.predict(X_cal), m50.predict(X_cal))
    cal_hi = np.maximum(m90.predict(X_cal), m50.predict(X_cal))
    te_lo = np.minimum(m10.predict(X_te), m50.predict(X_te))
    te_hi = np.maximum(m90.predict(X_te), m50.predict(X_te))

    cal_scores = np.maximum(cal_lo - y_cal, y_cal - cal_hi)
    te_scores_raw = np.maximum(te_lo - y_te, y_te - te_hi)
    cal_tiers = X_cal["name"].astype(str).to_numpy()
    te_tiers = X_te["name"].astype(str).to_numpy()
    te_cabs = X_te["cab_type"].astype(str).to_numpy()

    alpha = 0.20
    q_map: dict[str, float] = {}
    for tier in sorted(np.unique(cal_tiers)):
        s_t = cal_scores[cal_tiers == tier]
        n_t = len(s_t)
        level = min(1.0, np.ceil((1.0 - alpha) * (n_t + 1)) / n_t)
        q_map[tier] = float(np.quantile(s_t, level, method="higher"))

    adj = np.array([q_map[t] for t in te_tiers], dtype=float)
    raw_cqr_lo = np.clip(te_lo - adj, 2.5, None)
    raw_cqr_hi = te_hi + adj
    mid = 0.5 * (raw_cqr_lo + raw_cqr_hi)
    cqr_lo = np.where(raw_cqr_lo <= raw_cqr_hi, raw_cqr_lo, mid)
    cqr_hi = np.where(raw_cqr_lo <= raw_cqr_hi, raw_cqr_hi, mid)

    cov_raw_strict = (y_te >= te_lo) & (y_te <= te_hi)
    cov_raw_tol = te_scores_raw <= 1e-6
    cov_cqr = (y_te >= cqr_lo) & (y_te <= cqr_hi)

    cqr_by_tier = []
    for tier in sorted(np.unique(te_tiers)):
        m = te_tiers == tier
        cqr_by_tier.append(
            {
                "cab_type": str(te_cabs[m][0]),
                "service_tier": tier,
                "n_cal": int((cal_tiers == tier).sum()),
                "n_test": int(m.sum()),
                "mean_actual_usd": round(float(y_te[m].mean()), 2),
                "base85_raw_strict_picp_pct": round(
                    float(cov_raw_strict[m].mean() * 100.0), 2
                ),
                "base85_raw_tol1e6_picp_pct": round(
                    float(cov_raw_tol[m].mean() * 100.0), 2
                ),
                "base85_raw_mpiw_usd": round(
                    float((te_hi[m] - te_lo[m]).mean()), 3
                ),
                "q_hat_usd_6dec": round(q_map[tier], 6),
                "cqr_picp_pct": round(float(cov_cqr[m].mean() * 100.0), 2),
                "cqr_mpiw_usd": round(
                    float((cqr_hi[m] - cqr_lo[m]).mean()), 3
                ),
                "delta_mpiw_usd": round(
                    float(
                        (cqr_hi[m] - cqr_lo[m]).mean()
                        - (te_hi[m] - te_lo[m]).mean()
                    ),
                    4,
                ),
            }
        )

    # Kiểm toán biên 10^-8 USD trên mô hình GĐ6 (100% Train, native category)
    s10 = _make_native_quantile_pipe(0.10).fit(X_tr, y_tr)
    s50 = _make_native_quantile_pipe(0.50).fit(X_tr, y_tr)
    s90 = _make_native_quantile_pipe(0.90).fit(X_tr, y_tr)
    s_lo = np.minimum(s10.predict(X_te), s50.predict(X_te))
    s_hi = np.maximum(s90.predict(X_te), s50.predict(X_te))
    s_scores = np.maximum(s_lo - y_te, y_te - s_hi)
    s_strict = (y_te >= s_lo) & (y_te <= s_hi)
    s_tol = s_scores <= 1e-6
    m_lux, m_suv = te_tiers == "Lux Black XL", te_tiers == "Black SUV"

    cqr_block = {
        "categorical_encoding": "native_pandas_category_from_dtype",
        "n_fit": int(len(X_fit)),
        "n_cal": int(len(X_cal)),
        "n_test": int(len(X_te)),
        "overall_base85_raw_strict_picp_pct": round(
            float(cov_raw_strict.mean() * 100.0), 2
        ),
        "overall_base85_raw_tol1e6_picp_pct": round(
            float(cov_raw_tol.mean() * 100.0), 2
        ),
        "overall_base85_raw_mpiw_usd": round(float((te_hi - te_lo).mean()), 3),
        "overall_cqr_picp_pct": round(float(cov_cqr.mean() * 100.0), 2),
        "overall_cqr_mpiw_usd": round(float((cqr_hi - cqr_lo).mean()), 3),
        "min_tier_cqr_picp_pct": min(r["cqr_picp_pct"] for r in cqr_by_tier),
        "max_tier_cqr_picp_pct": max(r["cqr_picp_pct"] for r in cqr_by_tier),
        "pre_repair_crossings": int((raw_cqr_lo > raw_cqr_hi).sum()),
        "post_repair_crossings": int((cqr_lo > cqr_hi).sum()),
        "by_tier": cqr_by_tier,
        "stage6_100pct_boundary_audit": {
            "overall_strict_pct": round(float(s_strict.mean() * 100.0), 2),
            "overall_tol1e6_pct": round(float(s_tol.mean() * 100.0), 2),
            "lux_black_xl_strict_pct": round(
                float(s_strict[m_lux].mean() * 100.0), 2
            ),
            "lux_black_xl_tol1e6_pct": round(
                float(s_tol[m_lux].mean() * 100.0), 2
            ),
            "black_suv_strict_pct": round(
                float(s_strict[m_suv].mean() * 100.0), 2
            ),
            "black_suv_tol1e6_pct": round(
                float(s_tol[m_suv].mean() * 100.0), 2
            ),
        },
    }

    # 4D. Grouped Permutation Importance (6 cụm ngữ nghĩa = 27/27 cột) &
    # Missingness Sensitivity
    wm_test = master.iloc[te_idx]["weather_matched"].to_numpy()
    rain_ok_test = master.iloc[te_idx]["is_raining"].notna().to_numpy()
    ts_test = master.iloc[te_idx]["is_timestamp_original"].to_numpy()
    miss_sens = {
        "M7_full_test": _reg_metrics(y_te, p7),
        "M7_complete_weather_valid_rain": _reg_metrics(
            y_te[(wm_test == 1) & rain_ok_test],
            p7[(wm_test == 1) & rain_ok_test],
        ),
        "M7_unmatched_weather_0": _reg_metrics(
            y_te[wm_test == 0], p7[wm_test == 0]
        ),
        "M2_full_test": _reg_metrics(y_te, p2),
        "M2_complete_weather_valid_rain": _reg_metrics(
            y_te[(wm_test == 1) & rain_ok_test],
            p2[(wm_test == 1) & rain_ok_test],
        ),
        "M2_unmatched_weather_0": _reg_metrics(
            y_te[wm_test == 0], p2[wm_test == 0]
        ),
        "M7_original_ts_1": _reg_metrics(y_te[ts_test == 1], p7[ts_test == 1]),
        "M7_missing_ts_0": _reg_metrics(y_te[ts_test == 0], p7[ts_test == 0]),
    }

    sample_n = 15_000
    sub_pos = rng.choice(len(X_te), size=sample_n, replace=False)
    Xs = X_te.iloc[sub_pos].reset_index(drop=True)
    ys = y_te[sub_pos]
    base_rmse_m7 = float(np.sqrt(mean_squared_error(ys, m7.predict(Xs))))
    base_rmse_m2 = float(np.sqrt(mean_squared_error(ys, m2.predict(Xs))))

    semantic_groups = [
        ("1. Phân khúc dịch vụ (name)", ["name"]),
        (
            "2. Cụm Cự ly & Surge gộp (chống đa cộng tuyến)",
            [
                "distance",
                "log_distance",
                "surge_multiplier",
                "lyft_surge_excess",
                "distance_x_surge",
                "distance_x_lyft_surge_excess",
            ],
        ),
        (
            "3. Tuyến không gian (source, destination)",
            ["source", "destination"],
        ),
        ("4. Hãng xe (cab_type, is_lyft)", ["cab_type", "is_lyft"]),
        (
            "5. Toàn bộ 7 biến Thời tiết",
            [
                "temp",
                "rain",
                "humidity",
                "wind",
                "clouds",
                "pressure",
                "is_raining",
            ],
        ),
        (
            "6. Toàn bộ 9 biến Thời gian Boston EST",
            [
                "hour",
                "day_of_week",
                "is_weekend",
                "is_rush_hour",
                "time_of_day",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
            ],
        ),
    ]
    flat_cols = [c for _, cols in semantic_groups for c in cols]
    assert len(flat_cols) == 27 and set(flat_cols) == set(valid_cols), (
        "Phân hoạch 6 cụm phải phủ đúng 27/27 cột!"
    )

    grouped_rows = []
    for gname, cols in semantic_groups:
        d7_l, d2_l = [], []
        for _ in range(8):
            perm_idx = rng.permutation(sample_n)
            Xp = Xs.copy()
            for c in cols:
                Xp[c] = Xs[c].iloc[perm_idx].to_numpy()
            d7_l.append(
                float(np.sqrt(mean_squared_error(ys, m7.predict(Xp))))
                - base_rmse_m7
            )
            d2_l.append(
                float(np.sqrt(mean_squared_error(ys, m2.predict(Xp))))
                - base_rmse_m2
            )
        grouped_rows.append(
            {
                "group": gname,
                "n_cols": len(cols),
                "cols": ", ".join(cols),
                "m7_delta_rmse": round(float(np.mean(d7_l)), 4),
                "m7_delta_std": round(float(np.std(d7_l)), 4),
                "m2_delta_rmse": round(float(np.mean(d2_l)), 4),
                "m2_delta_std": round(float(np.std(d2_l)), 4),
            }
        )
    tot7 = sum(max(r["m7_delta_rmse"], 0.0) for r in grouped_rows)
    tot2 = sum(max(r["m2_delta_rmse"], 0.0) for r in grouped_rows)
    for r in grouped_rows:
        r["m7_rel_share_pct"] = round(
            max(r["m7_delta_rmse"], 0.0) / tot7 * 100.0, 2
        )
        r["m2_rel_share_pct"] = round(
            max(r["m2_delta_rmse"], 0.0) / tot2 * 100.0, 2
        )

    # Complete-case weather-only XAI check
    cc_mask = (wm_test == 1) & rain_ok_test
    X_cc = X_te.loc[cc_mask].reset_index(drop=True)
    y_cc = y_te[cc_mask]
    cc_sub = rng.choice(len(X_cc), size=min(12_000, len(X_cc)), replace=False)
    Xcc_s, ycc_s = X_cc.iloc[cc_sub].reset_index(drop=True), y_cc[cc_sub]
    base_cc_m7 = float(np.sqrt(mean_squared_error(ycc_s, m7.predict(Xcc_s))))
    w_cc_d7 = []
    for _ in range(5):
        pi = rng.permutation(len(Xcc_s))
        Xp = Xcc_s.copy()
        for c in [
            "temp",
            "rain",
            "humidity",
            "wind",
            "clouds",
            "pressure",
            "is_raining",
        ]:
            Xp[c] = Xcc_s[c].iloc[pi].to_numpy()
        w_cc_d7.append(
            float(np.sqrt(mean_squared_error(ycc_s, m7.predict(Xp))))
            - base_cc_m7
        )

    # 4E. API-Batch GroupShuffleSplit trên tập quan sát surge thực (232,861
    # dòng)
    obs_idx = np.flatnonzero(master["is_surge_original"].to_numpy() == 1)
    master_obs = master.iloc[obs_idx]
    feat_obs = df_feat.iloc[obs_idx]
    ts_key_obs = np.where(
        master_obs["is_timestamp_original"] == 1,
        master_obs["time_stamp"].astype(str),
        "NO_TS_" + master_obs["id"].astype(str),
    )
    batch_key_obs = (
        master_obs["source"].astype(str)
        + "|"
        + master_obs["destination"].astype(str)
        + "|"
        + master_obs["cab_type"].astype(str)
        + "|"
        + pd.Series(ts_key_obs, index=master_obs.index)
    )
    # Tỷ lệ trùng phiên batch giữa tr_idx và te_idx (cả hai đều nằm trong
    # obs_idx)
    full_ts_key = np.where(
        master["is_timestamp_original"] == 1,
        master["time_stamp"].astype(str),
        "NO_TS_" + master["id"].astype(str),
    )
    full_batch_key = (
        master["source"].astype(str)
        + "|"
        + master["destination"].astype(str)
        + "|"
        + master["cab_type"].astype(str)
        + "|"
        + pd.Series(full_ts_key, index=master.index)
    )
    tr_batches = set(full_batch_key.iloc[tr_idx])
    overlap_pct = round(
        float(full_batch_key.iloc[te_idx].isin(tr_batches).mean() * 100.0), 2
    )

    gss = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE
    )
    g_tr, g_te = next(gss.split(feat_obs, groups=batch_key_obs))
    Xg_tr = feat_obs.iloc[g_tr][valid_cols].copy()
    Xg_te = feat_obs.iloc[g_te][valid_cols].copy()
    for xd in (Xg_tr, Xg_te):
        for c in BASE_CAT_FEATURES:
            xd[c] = xd[c].astype("category")
    yg_tr = feat_obs.iloc[g_tr]["price"].to_numpy(dtype=np.float64)
    yg_te = feat_obs.iloc[g_te]["price"].to_numpy(dtype=np.float64)

    preps = build_stage4_preprocessors()
    hgb_params = _get_tuned_hgb_params()
    m7_g = Pipeline(
        [
            ("prep", preps["m7_hgb_native"]),
            (
                "reg",
                HistGradientBoostingRegressor(
                    categorical_features="from_dtype",
                    max_iter=350,
                    early_stopping=True,
                    **hgb_params,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ).fit(Xg_tr, yg_tr)
    m2_g = Pipeline(
        [
            ("prep", EconomicInteractionPreprocessor(scale_continuous=True)),
            ("reg", LinearRegression()),
        ]
    ).fit(Xg_tr, yg_tr)

    api_batch = {
        "n_train": int(len(g_tr)),
        "n_test": int(len(g_te)),
        "sibling_overlap_random_pct": overlap_pct,
        "m7": _reg_metrics(yg_te, m7_g.predict(Xg_te)),
        "m2": _reg_metrics(yg_te, m2_g.predict(Xg_te)),
    }

    # 4F. Strict Train-Only Full Refit (random_80_20 & route_group_split,
    # tuân thủ CRIT-03: không fillna(1.0) mù trên Lyft)
    obs_all_ids = set(master.iloc[tr_idx]["id"].astype(str).str.strip()) | set(
        master.iloc[te_idx]["id"].astype(str).str.strip()
    )
    raw_c = pd.read_csv(RAW_DIR / "cab_rides.csv", low_memory=False)
    raw_c["id"] = raw_c["id"].astype(str).str.strip()
    raw_c = raw_c.loc[raw_c["id"].isin(obs_all_ids)].copy()
    raw_c["product_id"] = raw_c["product_id"].astype(str).str.strip()
    raw_c["cab_type"] = raw_c["product_id"].map(
        lambda pid: PRODUCT_MAP.get(pid, ("Uber", "Unknown"))[0]
    )
    raw_c["name"] = raw_c["product_id"].map(
        lambda pid: PRODUCT_MAP.get(pid, ("Uber", "Unknown"))[1]
    )
    raw_c["source"] = standardize_location(raw_c["source"])
    raw_c["destination"] = standardize_location(raw_c["destination"])
    for col in ("distance", "price", "surge_multiplier"):
        raw_c[col] = raw_c[col].apply(clean_unit_string)
    raw_c.loc[
        raw_c["price"].isin([-1, -1.0, -100, 1000, 9999, 10000, 100000]),
        "price",
    ] = np.nan
    raw_c.loc[
        raw_c["distance"].isin([-1, -1.0, 1000, 9999, 10000, 100000]),
        "distance",
    ] = np.nan
    raw_c.loc[
        raw_c["surge_multiplier"].isin([-1, -1.0, 9999, 10000, 100000]),
        "surge_multiplier",
    ] = np.nan
    for col in ("distance", "price", "surge_multiplier"):
        raw_c[col] = raw_c[col].abs()
    raw_c.loc[raw_c["surge_multiplier"] >= 1000, "surge_multiplier"] /= 1000.0
    raw_c.loc[raw_c["surge_multiplier"] >= 100, "surge_multiplier"] /= 100.0
    raw_c.loc[raw_c["surge_multiplier"] >= 10, "surge_multiplier"] /= 10.0
    raw_c.loc[
        (raw_c["surge_multiplier"] < 1.0) | (raw_c["surge_multiplier"] > 3.0),
        "surge_multiplier",
    ] = np.nan
    raw_c.loc[raw_c["cab_type"] == "Uber", "surge_multiplier"] = 1.0

    flags = pd.read_parquet(PROCESSED_DIR / "stage4_audit_flags.parquet")

    def _run_strict_train_only_split(
        sp_tr_idx: np.ndarray, sp_te_idx: np.ndarray
    ) -> dict[str, Any]:
        tr_ids = set(master.iloc[sp_tr_idx]["id"].astype(str).str.strip())
        te_ids = set(master.iloc[sp_te_idx]["id"].astype(str).str.strip())
        df_tr_r = raw_c.loc[raw_c["id"].isin(tr_ids)].copy()
        df_te_r = raw_c.loc[raw_c["id"].isin(te_ids)].copy()

        tr_nd = df_tr_r["distance"].where(
            (df_tr_r["distance"] >= 0.02) & (df_tr_r["distance"] <= 6.0)
        )
        tr_rref_map = (
            df_tr_r.assign(_nd=tr_nd)
            .groupby(["source", "destination"], observed=True)["_nd"]
            .median()
            .to_dict()
        )
        tr_dist_med = float(tr_nd.median())

        def _map_route(sub_d: pd.DataFrame, mp: dict, fb: float) -> pd.Series:
            return pd.Series(
                [
                    mp.get(k, fb)
                    for k in zip(
                        sub_d["source"], sub_d["destination"], strict=False
                    )
                ],
                index=sub_d.index,
                dtype=float,
            )

        df_tr_r["distance"], _ = recover_scaled_series(
            df_tr_r["distance"],
            _map_route(df_tr_r, tr_rref_map, tr_dist_med),
            0.02,
            8.0,
            preserve_decimals=2,
        )
        df_te_r["distance"], _ = recover_scaled_series(
            df_te_r["distance"],
            _map_route(df_te_r, tr_rref_map, tr_dist_med),
            0.02,
            8.0,
            preserve_decimals=2,
        )

        tr_np = df_tr_r["price"].where(
            (df_tr_r["price"] >= 2.5) & (df_tr_r["price"] <= 65.0)
        )
        tr_bp = tr_np / df_tr_r["surge_multiplier"]
        tr_bp_route = (
            df_tr_r.assign(_bp=tr_bp)
            .groupby(["name", "source", "destination"], observed=True)["_bp"]
            .median()
            .to_dict()
        )
        tr_bp_tier = (
            df_tr_r.assign(_bp=tr_bp)
            .groupby("name", observed=True)["_bp"]
            .median()
            .to_dict()
        )
        tr_p_med = float(tr_bp.median())

        def _map_pref(sub_d: pd.DataFrame) -> pd.Series:
            vals = []
            for nm, sc, ds, sg in zip(
                sub_d["name"],
                sub_d["source"],
                sub_d["destination"],
                sub_d["surge_multiplier"],
                strict=False,
            ):
                bp = tr_bp_route.get((nm, sc, ds), np.nan)
                if pd.isna(bp):
                    bp = tr_bp_tier.get(nm, tr_p_med)
                if pd.isna(bp):
                    bp = tr_p_med
                vals.append(float(bp) * float(sg))
            return pd.Series(vals, index=sub_d.index, dtype=float)

        df_tr_r["price"], tr_p_fix = recover_scaled_series(
            df_tr_r["price"],
            _map_pref(df_tr_r),
            2.5,
            95.0,
            preserve_decimals=2,
        )
        df_te_r["price"], te_p_fix = recover_scaled_series(
            df_te_r["price"],
            _map_pref(df_te_r),
            2.5,
            95.0,
            preserve_decimals=2,
        )

        tr_imp_map = (
            df_tr_r.groupby(["source", "destination"], observed=True)[
                "distance"
            ]
            .median()
            .to_dict()
        )
        tr_imp_fb = float(df_tr_r["distance"].median())
        df_tr_r["distance"] = df_tr_r["distance"].fillna(
            _map_route(df_tr_r, tr_imp_map, tr_imp_fb)
        )
        df_te_r["distance"] = df_te_r["distance"].fillna(
            _map_route(df_te_r, tr_imp_map, tr_imp_fb)
        )

        orig_tr = master.iloc[sp_tr_idx].set_index("id")
        orig_te = master.iloc[sp_te_idx].set_index("id")
        new_tr = df_tr_r.set_index("id").reindex(orig_tr.index)
        new_te = df_te_r.set_index("id").reindex(orig_te.index)

        tr_p_diff = int(
            (
                np.abs(
                    new_tr["price"].fillna(-999)
                    - orig_tr["price"].astype(float)
                )
                > 1e-4
            ).sum()
        )
        te_p_diff = int(
            (
                np.abs(
                    new_te["price"].fillna(-999)
                    - orig_te["price"].astype(float)
                )
                > 1e-4
            ).sum()
        )
        tr_d_obs = orig_tr["distance"].notna()
        te_d_obs = orig_te["distance"].notna()
        tr_d_diff = int(
            (
                np.abs(
                    new_tr.loc[tr_d_obs, "distance"].fillna(-999)
                    - orig_tr.loc[tr_d_obs, "distance"].astype(float)
                )
                > 1e-4
            ).sum()
        )
        te_d_diff = int(
            (
                np.abs(
                    new_te.loc[te_d_obs, "distance"].fillna(-999)
                    - orig_te.loc[te_d_obs, "distance"].astype(float)
                )
                > 1e-4
            ).sum()
        )
        te_d_max = float(
            np.nanmax(
                np.abs(
                    new_te.loc[te_d_obs, "distance"].values
                    - orig_te.loc[te_d_obs, "distance"].astype(float).values
                )
            )
        )

        X_tr_cl = df_feat.iloc[sp_tr_idx][valid_cols].copy()
        X_te_cl = df_feat.iloc[sp_te_idx][valid_cols].copy()
        X_tr_cl["distance"] = new_tr["distance"].values.astype(np.float32)
        X_te_cl["distance"] = new_te["distance"].values.astype(np.float32)
        for xd in (X_tr_cl, X_te_cl):
            xd["log_distance"] = np.log1p(xd["distance"]).astype(np.float32)
            xd["distance_x_surge"] = (
                xd["distance"] * xd["surge_multiplier"]
            ).astype(np.float32)
            xd["distance_x_lyft_surge_excess"] = (
                xd["distance"] * xd["lyft_surge_excess"]
            ).astype(np.float32)

        y_tr_cl = new_tr["price"].values.astype(np.float64)
        y_te_cl = new_te["price"].values.astype(np.float64)
        tr_ok, te_ok = np.isfinite(y_tr_cl), np.isfinite(y_te_cl)

        hgb_params = _get_tuned_hgb_params()
        m7_st = Pipeline(
            [
                ("prep", preps["m7_hgb_native"]),
                (
                    "reg",
                    HistGradientBoostingRegressor(
                        categorical_features="from_dtype",
                        max_iter=350,
                        early_stopping=True,
                        **hgb_params,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ).fit(X_tr_cl.iloc[tr_ok], y_tr_cl[tr_ok])
        m2_st = Pipeline(
            [
                (
                    "prep",
                    EconomicInteractionPreprocessor(scale_continuous=True),
                ),
                ("reg", LinearRegression()),
            ]
        ).fit(X_tr_cl.iloc[tr_ok], y_tr_cl[tr_ok])

        unmod_m = (
            flags["is_strictly_unmodified_raw"].to_numpy()[sp_te_idx] & te_ok
        )
        return {
            "n_train": int(len(orig_tr)),
            "n_test": int(len(orig_te)),
            "train_price_scale_fixed_count": tr_p_fix,
            "test_price_scale_fixed_count": te_p_fix,
            "train_price_rows_differing_vs_global": tr_p_diff,
            "test_price_rows_differing_vs_global": te_p_diff,
            "train_dist_rows_differing_vs_global": tr_d_diff,
            "test_dist_rows_differing_vs_global": te_d_diff,
            "test_dist_max_abs_diff_mi": round(te_d_max, 5),
            "m7_strict_train_only": _reg_metrics(
                y_te_cl[te_ok], m7_st.predict(X_te_cl.iloc[te_ok])
            ),
            "m2_strict_train_only": _reg_metrics(
                y_te_cl[te_ok], m2_st.predict(X_te_cl.iloc[te_ok])
            ),
            "m7_strict_unmodified_raw": _reg_metrics(
                y_te_cl[unmod_m], m7_st.predict(X_te_cl.iloc[unmod_m])
            ),
        }

    strict_cleaning = _run_strict_train_only_split(tr_idx, te_idx)
    rg_tr, rg_te = splits["route_group_split"]
    strict_cleaning["route_group_split"] = _run_strict_train_only_split(
        rg_tr, rg_te
    )

    # 4G. Độ nhạy ngưỡng tỷ số log10 trên price (3.0 / 3.55 / 4.0)
    raw_p = pd.read_csv(
        RAW_DIR / "cab_rides.csv", usecols=["id", "price"], low_memory=False
    )
    p_num = raw_p["price"].apply(clean_unit_string)
    p_num = p_num.where(
        ~p_num.isin([-1, -1.0, -100, 1000, 9999, 10000, 100000]), np.nan
    ).abs()
    raw_map = pd.Series(
        p_num.values, index=raw_p["id"].astype(str).str.strip()
    ).to_dict()
    master_raw_p = (
        master["id"].astype(str).str.strip().map(raw_map).to_numpy(dtype=float)
    )
    base_ref = (
        master.groupby(["name", "source", "destination"], observed=True)[
            "price"
        ]
        .transform("median")
        .to_numpy(dtype=float)
    )
    ratios = np.where(
        (base_ref > 0) & np.isfinite(master_raw_p),
        master_raw_p / base_ref,
        1.0,
    )
    thresh_sens = [
        {
            "ratio_threshold": thr,
            "n_price_flagged_in_master": int(
                np.sum((ratios > thr) | (master_raw_p > 95.0))
            ),
            "pct_of_master": round(
                float(np.sum((ratios > thr) | (master_raw_p > 95.0)))
                / len(master)
                * 100.0,
                2,
            ),
        }
        for thr in (3.0, 3.55, 4.0)
    ]

    return {
        "m7_vs_m6_paired_5fold": {
            "m7_folds_rmse": f7.tolist(),
            "m6_folds_rmse": f6.tolist(),
            "mean_diff_m7_minus_m6_usd": round(float(np.mean(diffs)), 4),
            "sd_diff_usd": round(float(np.std(diffs, ddof=1)), 4),
            "m7_better_folds_count": int(np.sum(diffs < 0)),
            "paired_t_stat": round(float(t_stat), 4),
            "paired_p_value": round(float(p_paired), 4),
            "bootstrap_95ci_diff_usd": [
                round(float(np.percentile(boot_means, 2.5)), 4),
                round(float(np.percentile(boot_means, 97.5)), 4),
            ],
        },
        "m0b_vs_m7_by_surge_regime": {
            "non_surge_m0b": _reg_metrics(y_te[~surge_mask], p0b[~surge_mask]),
            "non_surge_m7": _reg_metrics(y_te[~surge_mask], p7[~surge_mask]),
            "surge_gt1_m0b": _reg_metrics(y_te[surge_mask], p0b[surge_mask]),
            "surge_gt1_m7": _reg_metrics(y_te[surge_mask], p7[surge_mask]),
        },
        "clean_monotonic_cqr": cqr_block,
        "missingness_sensitivity": miss_sens,
        "grouped_permutation_importance": grouped_rows,
        "complete_case_weather_delta_rmse_m7": round(
            float(np.mean(w_cc_d7)), 5
        ),
        "api_batch_group_split": api_batch,
        "strict_train_only_cleaning": strict_cleaning,
        "log10_ratio_threshold_sensitivity": thresh_sens,
    }


# =============================================================================
# 5. CÔNG BẰNG KHÔNG GIAN 12 QUẬN & XẾP HẠNG SURGE LYFT (CRIT-03 CLEAN)
# =============================================================================
def compute_spatial_fairness_and_surge_eda(
    master: pd.DataFrame,
    df_feat: pd.DataFrame,
    valid_cols: list[str],
    splits: dict[str, Any],
) -> dict[str, Any]:
    """
    Kiểm toán công bằng 12 quận Boston (M7) & xếp hạng Surge Lyft trên 92,384
    báo giá thực.
    """
    _, te_idx = splits["random_80_20"]
    X_te = df_feat.iloc[te_idx][valid_cols].copy()
    for c in BASE_CAT_FEATURES:
        X_te[c] = X_te[c].astype("category")
    y_te = df_feat.iloc[te_idx]["price"].to_numpy(dtype=float)
    m7 = joblib.load(MODELS_DIR / "best_model.joblib")
    p7 = m7.predict(X_te)

    eval_df = pd.DataFrame(
        {
            "source": X_te["source"].astype(str).values,
            "actual": y_te,
            "resid": y_te - p7,
            "abs_err": np.abs(y_te - p7),
            "pct_err": np.abs((y_te - p7) / y_te) * 100.0,
        }
    )
    district_fairness = [
        {
            "district": str(d_name),
            "n_test": int(len(g)),
            "mean_price_usd": round(float(g["actual"].mean()), 2),
            "mean_signed_residual_usd": round(float(g["resid"].mean()), 4),
            "mae_usd": round(float(g["abs_err"].mean()), 2),
            "mape_pct": round(float(g["pct_err"].mean()), 2),
        }
        for d_name, g in eval_df.groupby("source")
    ]

    # [CRIT-03 FIX] Chỉ tính tỷ lệ Surge Lyft trên tập quan sát thực
    # (is_surge_original == 1, n = 92,384)
    lyft_obs = master.loc[
        (master["cab_type"].astype(str) == "Lyft")
        & (master["is_surge_original"] == 1)
    ].copy()
    lyft_obs["has_surge"] = (lyft_obs["surge_multiplier"] > 1.0).astype(int)
    dist_surge = (
        lyft_obs.groupby("source", observed=True)["has_surge"]
        .agg(
            n_lyft_obs="count",
            surge_count="sum",
            surge_rate_pct=lambda s: round(float(s.mean() * 100.0), 2),
        )
        .reset_index()
        .sort_values("surge_rate_pct", ascending=False)
    )
    dist_surge["rank"] = np.arange(1, len(dist_surge) + 1)

    lyft_ts = lyft_obs.loc[lyft_obs["is_timestamp_original"] == 1].copy()
    lyft_ts["hr_key"] = (
        lyft_ts["source"].astype(str)
        + "|"
        + lyft_ts["datetime"].dt.floor("h").astype(str)
    )
    s_per_hr = lyft_ts.groupby("hr_key")["has_surge"].sum()
    active_hrs = s_per_hr[s_per_hr > 0]

    return {
        "spatial_fairness_12_districts": district_fairness,
        "max_abs_district_bias_usd": round(
            max(abs(r["mean_signed_residual_usd"]) for r in district_fairness),
            4,
        ),
        "district_mape_range_pct": [
            min(r["mape_pct"] for r in district_fairness),
            max(r["mape_pct"] for r in district_fairness),
        ],
        "lyft_overall_observed_n": int(len(lyft_obs)),
        "lyft_overall_surge_count": int(lyft_obs["has_surge"].sum()),
        "lyft_overall_surge_rate_pct": round(
            float(lyft_obs["has_surge"].mean() * 100.0), 2
        ),
        "lyft_spatial_surge_all12": dist_surge.to_dict("records"),
        "surge_clustering_in_loc_hour": {
            "total_lyft_loc_hours": int(len(s_per_hr)),
            "active_surge_loc_hours": int(len(active_hrs)),
            "pct_loc_hours_with_surge": round(
                float(len(active_hrs) / len(s_per_hr) * 100.0), 2
            ),
            "mean_surge_quotes_per_active_loc_hour": round(
                float(active_hrs.mean()), 2
            ),
            "max_surge_quotes_in_single_loc_hour": int(active_hrs.max()),
        },
    }


# =============================================================================
# 6. RQ4: RÀNG BUỘC CUNG (C2), OOS, POWER ANALYSIS TRIGGERED (H3) & FAIR I/O
# (M1)
# =============================================================================
def compute_rq4_supply_power_and_io(
    master: pd.DataFrame,
    df_feat: pd.DataFrame,
    valid_cols: list[str],
    splits: dict[str, Any],
) -> dict[str, Any]:
    """
    Mô phỏng ràng buộc cung C2, đối chứng số học/OOS, Power Analysis Triggered
    H3 & Fair I/O M1.
    """
    sim = pd.read_csv(TABLES_DIR / "stage6_surge_cap_simulation.csv")

    # 6A. Kiểm tra đơn điệu nghiêm ngặt C2 trên mọi mức trần kích hoạt
    supply_table = []
    for _, r in sim.iterrows():
        cap = float(r["surge_cap"])
        n_aff = int(r["affected_trips_count"])
        static_pct = float(r["static_lyft_rev_change_pct"])
        eps50_pct = float(r["net_lyft_rev_change_eps_050_pct"])
        eps80_pct = float(r["net_lyft_rev_change_eps_080_pct"])
        drop5_pct = float(r["supply_constrained_rev_change_drop5_pct"])
        drop10_pct = float(r["supply_constrained_rev_change_drop10_pct"])
        if n_aff > 0:
            assert (
                drop10_pct < drop5_pct < static_pct < eps50_pct < eps80_pct
            ), f"C2 monotonicity violated at cap={cap}"
        supply_table.append(
            {
                "surge_cap": cap,
                "n_affected": n_aff,
                "affected_lyft_pct": float(r["affected_lyft_trips_pct"]),
                "mean_saving_usd": float(
                    r["mean_rider_saving_per_affected_trip_usd"]
                ),
                "mean_saving_pct": float(r["mean_rider_saving_pct_affected"]),
                "m2_ols_mean_saving_usd": float(
                    r["m2_ols_mean_saving_affected_usd"]
                ),
                "total_consumer_savings_usd": float(
                    r["total_consumer_savings_usd"]
                ),
                "proxy_static_pct": static_pct,
                "proxy_supply_drop5_pct": drop5_pct,
                "proxy_supply_drop10_pct": drop10_pct,
                "proxy_elastic_upper_eps050_pct": eps50_pct,
                "proxy_elastic_upper_eps080_pct": eps80_pct,
                "unfulfilled_quotes_drop5": float(
                    r["unfulfilled_quotes_drop5"]
                ),
                "unfulfilled_quotes_drop10": float(
                    r["unfulfilled_quotes_drop10"]
                ),
            }
        )

    # 6B. Bảng tối ưu chính sách có ràng buộc động (|Delta Proxy| <= tau) theo
    # từng giả định cung
    active_caps = [r for r in supply_table if r["n_affected"] > 0]
    constrained_matrix = []
    for tau in (0.05, 0.10, 0.30, 0.50):
        row_t: dict[str, Any] = {"max_proxy_drop_tau_pct": tau}
        for mode_lbl, col_k in [
            ("static_supply_0pct", "proxy_static_pct"),
            ("supply_drop_5pct", "proxy_supply_drop5_pct"),
            ("supply_drop_10pct", "proxy_supply_drop10_pct"),
            ("elastic_upper_eps050", "proxy_elastic_upper_eps050_pct"),
        ]:
            qual = [r for r in active_caps if abs(r[col_k]) <= tau + 1e-9]
            if qual:
                best = max(qual, key=lambda x: x["n_affected"])
                row_t[mode_lbl] = {
                    "selected_cap": f"{best['surge_cap']:.2f}x",
                    "protected_quotes": best["n_affected"],
                    "proxy_change_pct": best[col_k],
                }
            else:
                row_t[mode_lbl] = {
                    "selected_cap": "Không đạt (Giữ nguyên 3.00x)",
                    "protected_quotes": 0,
                    "proxy_change_pct": 0.0,
                }
        constrained_matrix.append(row_t)

    # 6C. Đối chứng số học không mô hình (1 - cap/surge) trên 92,384 báo giá
    # Lyft quan sát thực
    lyft_obs = master.loc[
        (master["cab_type"].astype(str) == "Lyft")
        & (master["is_surge_original"] == 1)
    ].copy()
    base_proxy = float(lyft_obs["price"].astype(np.float64).sum())
    arith_rows = []
    for cap in (3.0, 2.0, 1.75, 1.5, 1.25, 1.0):
        aff = lyft_obs.loc[lyft_obs["surge_multiplier"] > cap + 1e-6]
        if len(aff) == 0:
            arith_rows.append(
                {
                    "surge_cap": cap,
                    "n_affected": 0,
                    "arith_mean_saving_usd": 0.0,
                    "arith_saving_pct": 0.0,
                    "arith_static_proxy_change_pct": 0.0,
                    "arith_supply_drop5_pct": 0.0,
                    "arith_supply_drop10_pct": 0.0,
                }
            )
        else:
            aff_p = aff["price"].astype(np.float64)
            aff_s = aff["surge_multiplier"].astype(np.float64)
            capped_p = aff_p * (cap / aff_s)
            sav = aff_p - capped_p
            static_c = float(-sav.sum() / base_proxy * 100.0)
            drop5_c = float(
                (-sav.sum() - 0.05 * capped_p.sum()) / base_proxy * 100.0
            )
            drop10_c = float(
                (-sav.sum() - 0.10 * capped_p.sum()) / base_proxy * 100.0
            )
            arith_rows.append(
                {
                    "surge_cap": cap,
                    "n_affected": int(len(aff)),
                    "arith_mean_saving_usd": round(float(sav.mean()), 2),
                    "arith_saving_pct": round(
                        float(((1.0 - cap / aff_s) * 100.0).mean()), 2
                    ),
                    "arith_static_proxy_change_pct": round(static_c, 3),
                    "arith_supply_drop5_pct": round(drop5_c, 3),
                    "arith_supply_drop10_pct": round(drop10_c, 3),
                }
            )

    # 6D. Kiểm chứng ngoài mẫu Out-of-Sample Test-Only trên tập Test Lyft
    # (18,440 báo giá) cho 2.00x & 1.75x
    _, te_idx = splits["random_80_20"]
    df_te = df_feat.iloc[te_idx]
    lyft_te = df_te.loc[df_te["cab_type"].astype(str) == "Lyft"].copy()
    X_base = lyft_te[valid_cols].copy()
    for c in BASE_CAT_FEATURES:
        X_base[c] = X_base[c].astype("category")
    model_m7 = joblib.load(MODELS_DIR / "best_model.joblib")
    actual_te = lyft_te["price"].to_numpy(dtype=np.float64)
    orig_s_te = lyft_te["surge_multiplier"].to_numpy(dtype=np.float64)
    pred_base_te = np.clip(model_m7.predict(X_base), 2.5, None)
    tot_base_te = float(np.sum(actual_te))

    oos_caps = {}
    for cap_v in (2.00, 1.75):
        aff_m = orig_s_te > cap_v + 1e-6
        X_cap = X_base.copy()
        cs = np.minimum(orig_s_te, cap_v)
        d_f = X_cap["distance"].astype(np.float64)
        X_cap["surge_multiplier"] = cs
        X_cap["lyft_surge_excess"] = (cs - 1.0) * X_cap["is_lyft"].astype(
            np.float64
        )
        X_cap["distance_x_surge"] = d_f * cs
        X_cap["distance_x_lyft_surge_excess"] = (
            d_f * X_cap["lyft_surge_excess"]
        )
        ratio = np.clip(
            np.clip(model_m7.predict(X_cap), 2.5, None) / pred_base_te,
            0.25,
            1.0,
        )
        new_p = np.where(aff_m, actual_te * ratio, actual_te)
        sav = actual_te - new_p
        rel_dp = (new_p - actual_te) / actual_te
        q_dem = 1.0 + (-0.50) * rel_dp
        q_s_mod = np.where(aff_m, np.clip(1.0 + 0.30 * rel_dp, 0.50, 1.0), 1.0)
        q_s_sev = np.where(aff_m, np.clip(1.0 + 0.60 * rel_dp, 0.40, 1.0), 1.0)
        q_d5 = np.where(aff_m, np.minimum(q_dem, q_s_mod), 1.0)
        q_d10 = np.where(aff_m, np.minimum(q_dem, q_s_sev), 1.0)
        oos_caps[f"cap_{cap_v:.2f}x"] = {
            "n_lyft_test": int(len(lyft_te)),
            "n_affected": int(np.sum(aff_m)),
            "mean_saving_usd": round(float(np.mean(sav[aff_m])), 2),
            "mean_saving_pct": round(
                float(np.mean((sav[aff_m] / actual_te[aff_m]) * 100.0)), 2
            ),
            "static_proxy_change_pct": round(
                (float(np.sum(new_p)) - tot_base_te) / tot_base_te * 100.0, 3
            ),
            "supply_drop5_proxy_change_pct": round(
                (float(np.sum(new_p * q_d5)) - tot_base_te)
                / tot_base_te
                * 100.0,
                3,
            ),
            "supply_drop10_proxy_change_pct": round(
                (float(np.sum(new_p * q_d10)) - tot_base_te)
                / tot_base_te
                * 100.0,
                3,
            ),
            "elastic_upper_eps050_pct": round(
                (float(np.sum(new_p * q_dem)) - tot_base_te)
                / tot_base_te
                * 100.0,
                3,
            ),
        }

    # 6E. [H3 FIX] Switchback Power Analysis — Tính đúng DEFF_{k×} = 1 + (k *
    # m_0 - 1) * ICC
    # tại lưu lượng 1×, 25×, 50× cho cả 2 phạm vi: Toàn 12 quận vs. 5 quận điểm
    # nóng Surge
    lyft_ts = lyft_obs.loc[lyft_obs["is_timestamp_original"] == 1].copy()
    lyft_ts["loc_hr"] = (
        lyft_ts["source"].astype(str)
        + "|"
        + lyft_ts["datetime"].dt.floor("h").astype(str)
    )
    n_obs_days = float(
        (lyft_ts["datetime"].max() - lyft_ts["datetime"].min()).total_seconds()
        / 86400.0
    )
    n_active_days = int(lyft_ts["datetime"].dt.date.nunique())
    top5_hotspots = [
        "Back Bay",
        "Fenway",
        "Northeastern University",
        "Theatre District",
        "Boston University",
    ]
    lyft_ts_top5 = lyft_ts.loc[
        lyft_ts["source"].astype(str).isin(top5_hotspots)
    ].copy()

    z_a, z_b = 1.96, 0.8416
    p0 = 0.50
    icc = 0.05

    def _power_for_cap(
        sub_df: pd.DataFrame,
        cap_thresh: float,
        all_triggered_count: int,
        scope_label: str,
    ) -> dict[str, Any]:
        trig = sub_df["surge_multiplier"] > cap_thresh + 1e-6
        trig_sum = int(trig.sum())
        trig_rate = float(trig.mean())
        n_total_loc_hr = int(sub_df["loc_hr"].nunique())
        trig_per_blk = (
            sub_df.assign(_t=trig.astype(int)).groupby("loc_hr")["_t"].sum()
        )
        active_blks = trig_per_blk[trig_per_blk > 0]
        mean_all_blk = float(trig_per_blk.mean())
        m0_active = float(active_blks.mean()) if len(active_blks) else 1.0
        # Hệ số thiết kế DEFF_{k×} = 1 + max(k * m_0 - 1, 0) * ICC theo quy mô
        # cụm thực tế tại lưu lượng k×
        deff_1x = 1.0 + max(m0_active - 1.0, 0.0) * icc
        deff_25x = 1.0 + max(25.0 * m0_active - 1.0, 0.0) * icc
        deff_50x = 1.0 + max(50.0 * m0_active - 1.0, 0.0) * icc
        # Tốc độ tích lũy sự kiện kích hoạt mỗi ngày quan sát thực ở quy mô 1×
        trig_per_day_1x = float(trig_sum) / max(float(n_active_days), 1.0)

        mde_grid = []
        for mde_pp in (2.0, 3.0, 5.0):
            mde = mde_pp / 100.0
            n_indiv_arm = 2.0 * ((z_a + z_b) ** 2) * p0 * (1.0 - p0) / (mde**2)

            n_trig_arm_1x = int(np.ceil(n_indiv_arm * deff_1x))
            n_trig_total_1x = n_trig_arm_1x * 2
            days_1x = n_trig_total_1x / max(trig_per_day_1x, 1e-6)

            n_trig_arm_25x = int(np.ceil(n_indiv_arm * deff_25x))
            n_trig_total_25x = n_trig_arm_25x * 2
            days_25x = n_trig_total_25x / max(25.0 * trig_per_day_1x, 1e-6)
            blks_25x = int(np.ceil(n_trig_total_25x / (25.0 * m0_active)))

            n_trig_arm_50x = int(np.ceil(n_indiv_arm * deff_50x))
            n_trig_total_50x = n_trig_arm_50x * 2
            days_50x = n_trig_total_50x / max(50.0 * trig_per_day_1x, 1e-6)
            blks_50x = int(np.ceil(n_trig_total_50x / (50.0 * m0_active)))

            # ITT pha loãng trên toàn bộ yêu cầu Lyft (MDE_ITT = MDE *
            # trig_rate)
            mde_itt = mde * trig_rate
            n_itt_indiv_arm = (
                2.0
                * ((z_a + z_b) ** 2)
                * p0
                * (1.0 - p0)
                / max(mde_itt**2, 1e-12)
            )
            mde_grid.append(
                {
                    "mde_pp": mde_pp,
                    "triggered_quotes_per_arm": n_trig_arm_1x,
                    "triggered_quotes_both_arms": n_trig_total_1x,
                    "active_triggered_blocks_both_arms": int(
                        np.ceil(n_trig_total_1x / m0_active)
                    ),
                    "days_needed_at_api_scrape_rate_1x": round(days_1x, 1),
                    "triggered_quotes_per_arm_25x": n_trig_arm_25x,
                    "triggered_quotes_both_arms_25x": n_trig_total_25x,
                    "active_triggered_blocks_both_arms_25x": blks_25x,
                    "days_needed_at_prod_traffic_25x": round(days_25x, 1),
                    "triggered_quotes_per_arm_50x": n_trig_arm_50x,
                    "triggered_quotes_both_arms_50x": n_trig_total_50x,
                    "active_triggered_blocks_both_arms_50x": blks_50x,
                    "days_needed_at_prod_traffic_50x": round(days_50x, 1),
                    "itt_diluted_mde_pp": round(mde_itt * 100.0, 4),
                    "itt_unfeasible_requests_per_arm": int(
                        np.ceil(n_itt_indiv_arm)
                    ),
                }
            )

        return {
            "scope": scope_label,
            "cap_threshold": cap_thresh,
            "n_lyft_quotes_with_ts": int(len(sub_df)),
            "empirical_trigger_rate_pct": round(trig_rate * 100.0, 3),
            "observed_triggered_quotes_with_ts": trig_sum,
            "share_of_citywide_triggered_pct": round(
                trig_sum / max(all_triggered_count, 1) * 100.0, 2
            ),
            "total_lyft_loc_hr_blocks": n_total_loc_hr,
            "active_triggered_loc_hr_blocks": int(len(active_blks)),
            "mean_triggered_per_all_loc_hr": round(mean_all_blk, 4),
            "mean_triggered_per_active_loc_hr": round(m0_active, 2),
            "deff_on_active_blocks_icc005": round(deff_1x, 3),
            "deff_1x": round(deff_1x, 3),
            "deff_25x": round(deff_25x, 3),
            "deff_50x": round(deff_50x, 3),
            "m_active_25x": round(25.0 * m0_active, 2),
            "m_active_50x": round(50.0 * m0_active, 2),
            "observed_triggered_per_day_scrape": round(trig_per_day_1x, 2),
            "mde_scenarios": mde_grid,
        }

    all_trig_175 = int((lyft_ts["surge_multiplier"] > 1.75 + 1e-6).sum())
    all_trig_200 = int((lyft_ts["surge_multiplier"] > 2.00 + 1e-6).sum())

    pw_all_175 = _power_for_cap(
        lyft_ts, 1.75, all_trig_175, "all_12_districts"
    )
    pw_all_200 = _power_for_cap(
        lyft_ts, 2.00, all_trig_200, "all_12_districts"
    )
    pw_top5_175 = _power_for_cap(
        lyft_ts_top5, 1.75, all_trig_175, "top5_hotspot_districts"
    )
    pw_top5_200 = _power_for_cap(
        lyft_ts_top5, 2.00, all_trig_200, "top5_hotspot_districts"
    )

    power_analysis = {
        "methodology": (
            "Triggered / Exposed-Only Switchback Power Analysis (alpha=0.05,"
            " power=80%, p0=50%, ICC=0.05) "
            "với DEFF_{k×} = 1 + (k·m_0 - 1)·ICC phụ thuộc quy mô cụm thực "
            "tế tại lưu lượng 1×/25×/50×, "
            "so sánh phạm vi toàn 12 quận và 5 quận điểm nóng."
        ),
        "observed_span_days": n_active_days,
        "observed_active_days": n_active_days,
        "observed_calendar_span_days": round(n_obs_days, 2),
        "top5_hotspot_districts_list": top5_hotspots,
        "cap_1.75x": pw_all_175,
        "cap_2.00x": pw_all_200,
        "scopes": {
            "all_12_districts": {
                "cap_1.75x": pw_all_175,
                "cap_2.00x": pw_all_200,
            },
            "top5_hotspot_districts": {
                "cap_1.75x": pw_top5_175,
                "cap_2.00x": pw_top5_200,
            },
        },
    }

    # 6F. [M1 FIX] Benchmark I/O công bằng trên CÙNG BẢNG master_rideshare
    # (270,898 x 32): CSV vs Parquet (trung vị 3 lần)
    pq_path = PROCESSED_DIR / "master_rideshare.parquet"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        tmp_csv = Path(tf.name)
    try:
        master.to_csv(tmp_csv, index=False)
        csv_size_mb = tmp_csv.stat().st_size / (1024**2)
        pq_size_mb = pq_path.stat().st_size / (1024**2)

        csv_full_times, csv_chunk_times, pq_times = [], [], []
        csv_ram_mb, pq_ram_mb, csv_peak_chunk_mb = 0.0, 0.0, 0.0
        for _ in range(3):
            t0 = time.perf_counter()
            df_c = pd.read_csv(tmp_csv, low_memory=False)
            csv_full_times.append(time.perf_counter() - t0)
            csv_ram_mb = df_c.memory_usage(deep=True).sum() / (1024**2)
            del df_c

            t1 = time.perf_counter()
            pk = 0.0
            for ch in pd.read_csv(
                tmp_csv, chunksize=100_000, low_memory=False
            ):
                pk = max(pk, ch.memory_usage(deep=True).sum() / (1024**2))
            csv_chunk_times.append(time.perf_counter() - t1)
            csv_peak_chunk_mb = pk

            t2 = time.perf_counter()
            df_p = pd.read_parquet(pq_path)
            pq_times.append(time.perf_counter() - t2)
            pq_ram_mb = df_p.memory_usage(deep=True).sum() / (1024**2)
            del df_p
    finally:
        if tmp_csv.exists():
            tmp_csv.unlink()

    # Đo riêng streaming đọc tệp thô cab_rides.csv (693,071 dòng) với
    # chunksize=100,000
    t_raw0 = time.perf_counter()
    raw_rows = 0
    raw_peak_chunk_mb = 0.0
    for ch in pd.read_csv(
        RAW_DIR / "cab_rides.csv", chunksize=100_000, low_memory=False
    ):
        raw_rows += len(ch)
        raw_peak_chunk_mb = max(
            raw_peak_chunk_mb, ch.memory_usage(deep=True).sum() / (1024**2)
        )
    raw_chunk_sec = time.perf_counter() - t_raw0

    med_csv = float(np.median(csv_full_times))
    med_csv_ch = float(np.median(csv_chunk_times))
    med_pq = float(np.median(pq_times))

    io_bench = {
        "same_table_master_benchmark": {
            "n_rows": int(len(master)),
            "n_cols": int(master.shape[1]),
            "csv_disk_mb": round(csv_size_mb, 2),
            "parquet_disk_mb": round(pq_size_mb, 2),
            "disk_compression_ratio": round(
                csv_size_mb / max(pq_size_mb, 1e-4), 2
            ),
            "csv_full_median_sec": round(med_csv, 3),
            "csv_chunk100k_median_sec": round(med_csv_ch, 3),
            "parquet_median_sec": round(med_pq, 3),
            "parquet_vs_csv_full_speedup": round(
                med_csv / max(med_pq, 1e-4), 1
            ),
            "parquet_vs_csv_chunk_speedup": round(
                med_csv_ch / max(med_pq, 1e-4), 1
            ),
            "csv_full_ram_mb": round(csv_ram_mb, 2),
            "csv_peak_chunk_ram_mb": round(csv_peak_chunk_mb, 2),
            "parquet_ram_mb": round(pq_ram_mb, 2),
        },
        "raw_cab_streaming_benchmark": {
            "raw_rows_read": int(raw_rows),
            "chunksize": 100_000,
            "raw_chunk_time_sec": round(raw_chunk_sec, 3),
            "raw_peak_single_chunk_mb": round(raw_peak_chunk_mb, 2),
        },
    }

    return {
        "lyft_observed_baseline_quote_proxy_usd": round(base_proxy, 2),
        "lyft_observed_n_quotes": int(len(lyft_obs)),
        "surge_cap_supply_constrained_table": supply_table,
        "constrained_policy_matrix": constrained_matrix,
        "arithmetic_surge_benchmark": arith_rows,
        "oos_surge_cap_test_only": oos_caps,
        "switchback_power_analysis": power_analysis,
        "io_benchmark_fair_same_table": io_bench,
    }


def main() -> None:
    t_start = time.perf_counter()
    master = pd.read_parquet(PROCESSED_DIR / "master_rideshare.parquet")
    df_feat = engineer_features_stage4(master)
    valid_cols = get_feature_columns_for_modeling()
    splits, _ = create_data_splits(df_feat)

    print("1/6 Cleaning waterfall, selection bias & CRIT-03/H4 provenance...")
    sec1 = compute_cleaning_waterfall_and_selection_bias(master)

    print(
        "2/6 Cluster SE, TOST, Exact WCR (B=1,999, full DOF), H3b cluster, "
        "Sentinel & FDR..."
    )
    sec2 = compute_statistical_rigor_and_wcr(master)

    print(
        "3/6 RQ2 Crossing distances (Spec A & Spec B, HC3 + Cluster Delta "
        "95% CI)..."
    )
    sec3 = compute_crossing_distances_spec_a_and_b(master, df_feat, splits)

    print(
        "4/6 Model diagnostics, Monotonic Split-CQR (native cat), Grouped "
        "XAI, API-Batch & Strict Train-Only..."
    )
    sec4 = compute_model_diagnostics_cqr_and_xai(
        master, df_feat, valid_cols, splits
    )

    print(
        "5/6 Spatial fairness (12 districts) & Lyft spatial surge ranking "
        "(92,384 observed)..."
    )
    sec5 = compute_spatial_fairness_and_surge_eda(
        master, df_feat, valid_cols, splits
    )

    print(
        "6/6 RQ4 Supply-constrained C2, OOS, Triggered Switchback Power H3 "
        "& Fair I/O M1..."
    )
    sec6 = compute_rq4_supply_power_and_io(master, df_feat, valid_cols, splits)

    payload = {
        "schema_version": "v5.0-unified",
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "waterfall_and_selection_bias": sec1,
        "statistical_rigor_extensions": sec2,
        "crossing_distances_5_pairs": sec3,
        "model_diagnostics_and_cqr": sec4,
        "spatial_fairness_and_surge_eda": sec5,
        "policy_rq4_supply_power_and_io": sec6,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\nDA LUU NGUON SU THAT DUY NHAT V5: {OUT_JSON} "
        f"({payload['elapsed_sec']}s)"
    )


if __name__ == "__main__":
    main()
