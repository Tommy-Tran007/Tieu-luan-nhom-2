"""
Module: models.py
Mục đích: Định nghĩa đầy đủ hệ thống 10 cấu hình mô hình hồi quy (từ Mô hình
mốc M0a/M0b, Nhóm Tuyến tính M1-M4 kèm biến thể Log-Duan Smearing,
đến Nhóm Cây quyết định & Ensemble M5-M7) đóng gói trong Scikit-Learn Pipeline
chống rò rỉ dữ liệu (Zero Data Leakage).

Tác giả: Trần Thanh Huy (MSSV: 2521001060) & Lê Huy Thắng (MSSV: 2521001092) —
Nhóm 02 — Lớp 25DDS01
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from config import RANDOM_STATE, TUNING_SUBSAMPLE_SIZE
from features import (
    EconomicInteractionPreprocessor,
    build_stage4_preprocessors,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. CÁC ESTIMATOR TÙY CHỈNH CHUẨN SCIKIT-LEARN (M0a, M0b, M2-Log Duan
# Smearing)
# =============================================================================
class GroupMedianBaselineRegressor(BaseEstimator, RegressorMixin):
    """
    Mô hình mốc kinh nghiệm (Empirical Lookup Baseline):
    - M0a (group_cols=['name']): Dự báo bằng trung vị giá cước của từng gói
      dịch vụ trên tập Train.
    - M0b (group_cols=['name', 'source', 'destination']): Dự báo bằng bảng tra
      trung vị của
      tổ hợp (Gói xe x Điểm đón x Điểm đến) trên tập Train; nếu gặp tuyến mới
      chưa thấy ở tập Train
      thì tự động lùi về (fallback) trung vị của gói xe tương ứng, sau đó tới
      trung vị toàn cục.
    """

    def __init__(self, group_cols: tuple[str, ...] = ("name",)):
        self.group_cols = group_cols

    def fit(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> GroupMedianBaselineRegressor:
        df_tmp = X[list(self.group_cols)].copy()
        for c in self.group_cols:
            df_tmp[c] = df_tmp[c].astype(str)
        df_tmp["_target_y"] = np.asarray(y, dtype=np.float64)

        self.global_median_ = float(np.median(df_tmp["_target_y"].values))
        self.tier_map_ = (
            df_tmp.groupby("name", observed=True)["_target_y"]
            .median()
            .to_dict()
            if "name" in df_tmp.columns
            else {}
        )
        self.group_map_ = (
            df_tmp.groupby(list(self.group_cols), observed=True)["_target_y"]
            .median()
            .to_dict()
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        df_tmp = X[list(self.group_cols)].copy()
        for c in self.group_cols:
            df_tmp[c] = df_tmp[c].astype(str)

        if len(self.group_cols) == 1:
            col = self.group_cols[0]
            preds = (
                df_tmp[col]
                .map(self.group_map_)
                .fillna(self.global_median_)
                .values
            )
            return np.asarray(preds, dtype=np.float64)

        keys = list(
            zip(*(df_tmp[c].values for c in self.group_cols), strict=False)
        )
        tiers = (
            df_tmp["name"].values
            if "name" in df_tmp.columns
            else [None] * len(df_tmp)
        )
        out = np.empty(len(df_tmp), dtype=np.float64)
        for i, (k, t) in enumerate(zip(keys, tiers, strict=False)):
            val = self.group_map_.get(k, None)
            if val is None or np.isnan(val):
                val = self.tier_map_.get(t, self.global_median_)
            out[i] = val
        return out


class LogSmearingLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Mô hình Hồi quy Tuyến tính trên thang Logarit biến mục tiêu ln(price) kèm
    hệ số hiệu chỉnh
    Duan's (1983) Nonparametric Smearing Retransformation:
        ln(y) = X * beta + u
        y_hat = exp(X * beta_hat) * mean(exp(u_hat_train))
    Giúp khắc phục thiên lệch giảm (Downward Bias) do bất đẳng thức Jensen khi
    chuyển ngược từ ln(USD) về USD.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogSmearingLinearRegressor:
        y_arr = np.asarray(y, dtype=np.float64)
        log_y = np.log(np.clip(y_arr, 1e-3, None))
        self.ols_ = LinearRegression()
        self.ols_.fit(X, log_y)
        log_pred_tr = self.ols_.predict(X)
        residuals_log = log_y - log_pred_tr
        self.smearing_factor_ = float(np.mean(np.exp(residuals_log)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_pred = self.ols_.predict(X)
        # Giới hạn mũ an toàn tránh tràn số trên các tổ hợp ngoại suy cực đoan
        log_pred = np.clip(log_pred, 0.0, 5.5)
        return np.exp(log_pred) * self.smearing_factor_


# =============================================================================
# 2. TINH CHỈNH SIÊU THAM SỐ TRÊN TẬP TRAIN (HYPERPARAMETER TUNING — ZERO CV
# LEAKAGE)
# =============================================================================
def _tune_ridge_and_lasso(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_sub: pd.DataFrame,
    y_sub: np.ndarray,
    random_state: int,
) -> dict[str, Any]:
    """Tinh chỉnh Ridge (GridSearchCV) và Lasso (5-Fold CV warm_start)."""
    from sklearn.model_selection import GridSearchCV, KFold

    alphas_ridge = np.logspace(-3, 3, 13)
    ridge_pipe = Pipeline(
        [
            ("prep", EconomicInteractionPreprocessor(scale_continuous=True)),
            ("reg", Ridge(random_state=random_state)),
        ]
    )
    ridge_search = GridSearchCV(
        ridge_pipe,
        param_grid={"reg__alpha": alphas_ridge},
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=4,
    ).fit(X_sub, y_sub)
    best_r = float(ridge_search.best_params_["reg__alpha"])
    logger.info(f"  -> M3 (Ridge Pipeline CV): best_alpha = {best_r}")

    alphas_lasso = np.logspace(-0.5, -5.0, 25)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    rmses = np.zeros((5, len(alphas_lasso)), dtype=np.float64)
    for f_idx, (f_tr, f_va) in enumerate(kf.split(X_sub)):
        prep_f = EconomicInteractionPreprocessor(scale_continuous=True)
        Xf_tr = prep_f.fit_transform(X_sub.iloc[f_tr], y_sub[f_tr])
        Xf_va = prep_f.transform(X_sub.iloc[f_va])
        ws_mod = Lasso(
            max_iter=5000,
            tol=1e-4,
            precompute=True,
            warm_start=True,
            random_state=random_state,
        )
        for a_idx, a_val in enumerate(alphas_lasso):
            ws_mod.set_params(alpha=float(a_val)).fit(Xf_tr, y_sub[f_tr])
            rmses[f_idx, a_idx] = float(
                np.sqrt(mean_squared_error(y_sub[f_va], ws_mod.predict(Xf_va)))
            )
    best_l = float(alphas_lasso[int(np.argmin(rmses.mean(axis=0)))])
    prep_full = EconomicInteractionPreprocessor(scale_continuous=True)
    X_full = prep_full.fit_transform(X_train, y_train)
    names = prep_full.get_feature_names_out()
    lasso_full = Lasso(
        alpha=best_l,
        max_iter=5000,
        tol=1e-4,
        precompute=True,
        random_state=random_state,
    ).fit(X_full, y_train)
    z_mask = np.abs(lasso_full.coef_) < 1e-6
    w_cols = ("temp", "rain", "humidity", "wind", "clouds", "pressure")
    w_coefs = {
        wc: round(float(lasso_full.coef_[list(names).index(wc)]), 6)
        for wc in w_cols
    }
    w_zero = [wc for wc, val in w_coefs.items() if abs(val) < 1e-5]
    logger.info(
        f"  -> M4 (Lasso Pipeline CV): best_alpha = {best_l:.5f} | "
        f"Ép {int(np.sum(z_mask))}/{len(names)} đặc trưng về 0"
    )
    return {
        "M3_Ridge": {
            "best_alpha": best_r,
            "alphas_tested": [round(float(a), 5) for a in alphas_ridge],
        },
        "M4_Lasso": {
            "best_alpha": round(best_l, 6),
            "n_total_features": int(len(names)),
            "n_zeroed_features": int(np.sum(z_mask)),
            "zeroed_features": [
                str(f) for f, z in zip(names, z_mask, strict=False) if z
            ],
            "weather_coefficients_in_lasso": w_coefs,
            "weather_features_zeroed_by_lasso": w_zero,
        },
    }


def _run_randomized_pipe_search(
    prep_key: str,
    estimator: Any,
    param_dist: dict[str, list[Any]],
    n_iter: int,
    cv: int,
    X_sub: pd.DataFrame,
    y_sub: np.ndarray,
    random_state: int,
    label: str,
) -> dict[str, Any]:
    """Chạy RandomizedSearchCV bọc Pipeline chống rò rỉ cho 1 mô hình cây."""
    pipe = Pipeline(
        [
            ("prep", build_stage4_preprocessors()[prep_key]),
            ("reg", estimator),
        ]
    )
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=4,
    ).fit(X_sub, y_sub)
    best_params = {
        k.replace("reg__", ""): v for k, v in search.best_params_.items()
    }
    logger.info(f"  -> {label}: best_params = {best_params}")
    return {
        "best_params": best_params,
        "best_sub_cv_rmse": round(float(-search.best_score_), 4),
    }


def tune_hyperparameters_on_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
    subsample_size: int = TUNING_SUBSAMPLE_SIZE,
) -> dict[str, Any]:
    """Tinh chỉnh siêu tham số cho M3–M7 chỉ trên X_train (bọc Pipeline)."""
    logger.info(
        "Đang tinh chỉnh siêu tham số (Hyperparameter Tuning, bọc Pipeline "
        "chống rò rỉ CV)..."
    )
    rng = np.random.RandomState(random_state)
    if len(X_train) > subsample_size:
        idx = rng.choice(len(X_train), size=subsample_size, replace=False)
        X_sub, y_sub = X_train.iloc[idx].copy(), y_train[idx]
    else:
        X_sub, y_sub = X_train.copy(), y_train

    info = _tune_ridge_and_lasso(X_train, y_train, X_sub, y_sub, random_state)
    info["M5_DecisionTree"] = _run_randomized_pipe_search(
        "m5_m6_tree_ohe",
        DecisionTreeRegressor(random_state=random_state),
        {
            "reg__max_depth": [8, 10, 12, 15, 18, 22],
            "reg__min_samples_leaf": [5, 15, 30, 50, 100],
            "reg__min_samples_split": [10, 30, 60, 100],
        },
        12,
        5,
        X_sub,
        y_sub,
        random_state,
        "M5 (DecisionTree Pipeline CV)",
    )
    info["M6_RandomForest"] = _run_randomized_pipe_search(
        "m5_m6_tree_ohe",
        RandomForestRegressor(
            n_estimators=60,
            max_samples=0.45,
            random_state=random_state,
            n_jobs=1,
        ),
        {
            "reg__max_depth": [16, 20, 25, None],
            "reg__min_samples_leaf": [2, 5, 12],
            "reg__max_features": [0.5, 0.7, 1.0],
        },
        6,
        3,
        X_sub,
        y_sub,
        random_state,
        "M6 (RandomForest Pipeline CV)",
    )
    info["M7_HistGradientBoosting"] = _run_randomized_pipe_search(
        "m7_hgb_native",
        HistGradientBoostingRegressor(
            categorical_features="from_dtype",
            max_iter=300,
            early_stopping=True,
            random_state=random_state,
        ),
        {
            "reg__learning_rate": [0.05, 0.08, 0.10, 0.15],
            "reg__max_leaf_nodes": [31, 63, 127],
            "reg__min_samples_leaf": [20, 40, 80],
            "reg__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        },
        12,
        5,
        X_sub,
        y_sub,
        random_state,
        "M7 (HistGBR Pipeline CV)",
    )
    return info


# =============================================================================
# 3. KHỞI TẠO DANH MỤC ĐẦY ĐỦ 10 CẤU HÌNH MÔ HÌNH ĐÓNG GÓI TRONG PIPELINE
# =============================================================================
def _build_linear_models(
    preps: dict[str, Any],
    ridge_alpha: float,
    lasso_alpha: float,
    random_state: int,
) -> dict[str, Any]:
    """Khởi tạo 7 cấu hình đối chứng & tuyến tính (M0a–M4)."""
    return {
        "M0a_Tier_Median_Baseline": GroupMedianBaselineRegressor(
            group_cols=("name",)
        ),
        "M0b_Tier_Route_Lookup_Baseline": GroupMedianBaselineRegressor(
            group_cols=("name", "source", "destination")
        ),
        "M1_Basic_OLS": Pipeline(
            [("prep", preps["m1_basic_linear"]), ("reg", LinearRegression())]
        ),
        "M2_Interaction_OLS": Pipeline(
            [
                (
                    "prep",
                    EconomicInteractionPreprocessor(scale_continuous=True),
                ),
                ("reg", LinearRegression()),
            ]
        ),
        "M2_Log_Duan_Smearing_OLS": Pipeline(
            [
                (
                    "prep",
                    EconomicInteractionPreprocessor(scale_continuous=True),
                ),
                ("reg", LogSmearingLinearRegressor()),
            ]
        ),
        "M3_Ridge_Regression": Pipeline(
            [
                (
                    "prep",
                    EconomicInteractionPreprocessor(scale_continuous=True),
                ),
                ("reg", Ridge(alpha=ridge_alpha, random_state=random_state)),
            ]
        ),
        "M4_Lasso_Regression": Pipeline(
            [
                (
                    "prep",
                    EconomicInteractionPreprocessor(scale_continuous=True),
                ),
                (
                    "reg",
                    Lasso(
                        alpha=lasso_alpha,
                        max_iter=5000,
                        tol=1e-4,
                        precompute=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def build_model_registry(
    tuning_info: dict[str, Any], random_state: int = RANDOM_STATE
) -> dict[str, Any]:
    """Xây dựng từ điển 10 cấu hình mô hình đóng gói Pipeline (M0a -> M7)."""
    preps = build_stage4_preprocessors()
    registry = _build_linear_models(
        preps,
        float(tuning_info["M3_Ridge"]["best_alpha"]),
        float(tuning_info["M4_Lasso"]["best_alpha"]),
        random_state,
    )
    dt_p = tuning_info["M5_DecisionTree"]["best_params"]
    rf_p = tuning_info["M6_RandomForest"]["best_params"]
    hgb_p = tuning_info["M7_HistGradientBoosting"]["best_params"]
    registry["M5_Decision_Tree"] = Pipeline(
        [
            ("prep", preps["m5_m6_tree_ohe"]),
            ("reg", DecisionTreeRegressor(**dt_p, random_state=random_state)),
        ]
    )
    registry["M6_Random_Forest"] = Pipeline(
        [
            ("prep", preps["m5_m6_tree_ohe"]),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=100,
                    max_samples=0.50,
                    **rf_p,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    registry["M7_HistGradientBoosting"] = Pipeline(
        [
            ("prep", preps["m7_hgb_native"]),
            (
                "reg",
                HistGradientBoostingRegressor(
                    categorical_features="from_dtype",
                    max_iter=350,
                    early_stopping=True,
                    **hgb_p,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return registry
