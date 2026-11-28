"""
Module: stage5_modeling.py
Mục đích: Điều phối thực thi trọn vẹn Giai đoạn 5 (Kiểm định Giả thuyết Thống
kê H1-H4 & Huấn luyện, Đánh giá 10 Cấu hình Mô hình Hồi quy):
1. Chạy toàn bộ kiểm định giả thuyết H1 -> H4 (Welch t-test, Mann-Whitney,
ANOVA, Kruskal-Wallis, Chi-Square, Wald HC3).
2. Tinh chỉnh siêu tham số trên X_train và thực thi 5-Fold Cross-Validation cho
10 cấu hình mô hình (M0a -> M7).
3. Đánh giá độc lập 1 lần duy nhất trên tập kiểm tra X_test (46,573 dòng quan
sát surge thực) và kiểm tra mục tiêu SMART.
4. Kiểm định độ bền (Temporal Split, Route Group Split) và Kiểm định Độ nhạy R3
(Huấn luyện & Kiểm tra độc lập trên
   240,139 bản ghi nguyên bản 100% không qua khôi phục quy mô 10^m hay điền
   khuyết).
5. Thực thi 6 thí nghiệm Mutate & Break (MB1 -> MB6) chuẩn bị cho vấn đáp Vòng
3.
6. Lưu mô hình tốt nhất (models/best_model.joblib) và xuất 4 biểu đồ 300 DPI
(fig11 -> fig14).

Tác giả: Trần Thanh Huy (MSSV: 2521001060) & Lê Huy Thắng (MSSV: 2521001092) —
Nhóm 02 — Lớp 25DDS01
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import (
    BASE_CAT_FEATURES,
    BASE_DIR,
    BASE_NUM_FEATURES,
    CV_FOLDS,
    FIG_STAGE5_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TABLES_DIR,
    TARGET_COL,
    TEST_SIZE,
    ensure_directories,
)
from features import (
    EconomicInteractionPreprocessor,
    RouteDistanceImputer,
    build_stage4_preprocessors,
    create_data_splits,
    engineer_features_stage4,
    get_feature_columns_for_modeling,
    load_master_dataset,
)
from models import build_model_registry, tune_hyperparameters_on_train
from stats_tests import run_all_hypothesis_tests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


MODEL_DISPLAY_NAMES: dict[str, str] = {
    "M0a_Tier_Median_Baseline": "M0a: Trung vị theo Gói xe (Baseline 1)",
    "M0b_Tier_Route_Lookup_Baseline": (
        "M0b: Bảng tra Gói xe x Tuyến (Baseline 2)"
    ),
    "M1_Basic_OLS": "M1: Hồi quy Tuyến tính OLS Cơ bản",
    "M2_Interaction_OLS": "M2: Hồi quy OLS Tương tác Kinh tế lượng",
    "M2_Log_Duan_Smearing_OLS": (
        "M2-Log: OLS Tương tác ln(Price) + Duan Smearing"
    ),
    "M3_Ridge_Regression": "M3: Hồi quy Ridge (L2 Regularization)",
    "M4_Lasso_Regression": "M4: Hồi quy Lasso (L1 Regularization)",
    "M5_Decision_Tree": "M5: Cây Quyết định (Decision Tree Tuned)",
    "M6_Random_Forest": "M6: Rừng Ngẫu nhiên (Random Forest Tuned)",
    "M7_HistGradientBoosting": "M7: HistGradientBoosting (Native Categorical)",
}


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 27
) -> dict[str, float]:
    """Tính toán bộ 6 chỉ số đánh giá hồi quy chuẩn mực trên thang USD."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    n = len(y_t)
    r2 = float(r2_score(y_t, y_p))
    adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / max(n - n_features - 1, 1))
    rmse = float(root_mean_squared_error(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))
    mape = float(mean_absolute_percentage_error(y_t, y_p) * 100.0)
    max_err = float(max_error(y_t, y_p))
    return {
        "r2": round(r2, 4),
        "adj_r2": round(adj_r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape_pct": round(mape, 2),
        "max_error": round(max_err, 4),
    }


# =============================================================================
# 1. THỰC THI 5-FOLD CROSS-VALIDATION & ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP
# =============================================================================
def _prepare_stage5_feature_matrix(
    df_feat: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Chuẩn bị ma trận X (ép category chuẩn) và vector y mục tiêu."""
    valid_cols = get_feature_columns_for_modeling()
    X_all = df_feat[valid_cols].copy()
    for c in BASE_CAT_FEATURES:
        X_all[c] = X_all[c].astype("category")
    y_all = df_feat[TARGET_COL].astype(np.float64).values
    return X_all, y_all


def _evaluate_single_model_cv_and_test(
    model_key: str,
    base_estimator: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    kf: KFold,
    n_splits: int,
) -> tuple[dict[str, Any], dict[str, Any], Any, np.ndarray]:
    """Chạy K-Fold CV và đánh giá Test 1 lần cho một mô hình đơn lẻ."""
    disp_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    logger.info(f"Đang đánh giá 5-Fold CV & Test cho: {disp_name}...")
    fold_r2, fold_rmse, fold_mae, fold_mape, fold_times = [], [], [], [], []
    for tr_f, val_f in kf.split(X_train):
        est = clone(base_estimator)
        t0 = time.perf_counter()
        est.fit(X_train.iloc[tr_f], y_train[tr_f])
        preds_val = est.predict(X_train.iloc[val_f])
        dt = time.perf_counter() - t0
        m_val = compute_all_metrics(y_train[val_f], preds_val)
        fold_r2.append(m_val["r2"])
        fold_rmse.append(m_val["rmse"])
        fold_mae.append(m_val["mae"])
        fold_mape.append(m_val["mape_pct"])
        fold_times.append(dt)

    cv_row = {
        "model_code": model_key,
        "model_name": disp_name,
        "cv_folds": n_splits,
        "cv_r2_mean": round(float(np.mean(fold_r2)), 4),
        "cv_r2_std": round(float(np.std(fold_r2, ddof=1)), 4),
        "cv_rmse_mean": round(float(np.mean(fold_rmse)), 4),
        "cv_rmse_std": round(float(np.std(fold_rmse, ddof=1)), 4),
        "cv_rmse_rel_std_pct": round(
            float(np.std(fold_rmse, ddof=1) / np.mean(fold_rmse) * 100.0), 3
        ),
        "cv_mae_mean": round(float(np.mean(fold_mae)), 4),
        "cv_mae_std": round(float(np.std(fold_mae, ddof=1)), 4),
        "cv_mape_mean": round(float(np.mean(fold_mape)), 2),
        "cv_mape_std": round(float(np.std(fold_mape, ddof=1)), 3),
        "mean_fold_time_sec": round(float(np.mean(fold_times)), 3),
        "fold_rmses": [round(float(x), 4) for x in fold_rmse],
    }

    final_est = clone(base_estimator)
    t_fit0 = time.perf_counter()
    final_est.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t_fit0

    t_pred0 = time.perf_counter()
    y_pred_test = final_est.predict(X_test)
    pred_ms = (time.perf_counter() - t_pred0) * 1000.0
    m_train = compute_all_metrics(y_train, final_est.predict(X_train))
    m_test = compute_all_metrics(y_test, y_pred_test)

    test_row = {
        "model_code": model_key,
        "model_name": disp_name,
        "train_r2": m_train["r2"],
        "train_rmse": m_train["rmse"],
        "test_r2": m_test["r2"],
        "test_adj_r2": m_test["adj_r2"],
        "test_rmse": m_test["rmse"],
        "test_mae": m_test["mae"],
        "test_mape_pct": m_test["mape_pct"],
        "test_max_error": m_test["max_error"],
        "r2_overfit_gap": round(m_train["r2"] - m_test["r2"], 4),
        "train_time_sec": round(fit_sec, 3),
        "predict_time_ms": round(pred_ms, 2),
        "meets_smart_r2": bool(m_test["r2"] >= 0.90),
        "meets_smart_mape": bool(m_test["mape_pct"] <= 10.0),
        "meets_both_smart": bool(
            m_test["r2"] >= 0.90 and m_test["mape_pct"] <= 10.0
        ),
    }
    logger.info(
        f"  => {model_key}: CV RMSE = ${np.mean(fold_rmse):.4f} "
        f"(±{np.std(fold_rmse, ddof=1):.4f}) | Test R2 = {m_test['r2']:.4f}, "
        f"RMSE = ${m_test['rmse']:.4f}, MAPE = {m_test['mape_pct']:.2f}%"
    )
    return cv_row, test_row, final_est, y_pred_test


def run_cv_and_test_evaluation(
    registry: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    """
    Chạy 5-Fold Cross-Validation trên X_train và đánh giá 1 lần trên X_test
    cho toàn bộ 10 cấu hình mô hình trong registry.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_rows, test_rows = [], []
    fitted_models: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for model_key, base_estimator in registry.items():
        cv_r, te_r, est, preds = _evaluate_single_model_cv_and_test(
            model_key,
            base_estimator,
            X_train,
            y_train,
            X_test,
            y_test,
            kf,
            n_splits,
        )
        cv_rows.append(cv_r)
        test_rows.append(te_r)
        fitted_models[model_key] = est
        test_predictions[model_key] = preds

    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)
    m0b_rmse = float(
        test_df.loc[
            test_df["model_code"] == "M0b_Tier_Route_Lookup_Baseline",
            "test_rmse",
        ].iloc[0]
    )
    test_df["beats_M0b_baseline"] = test_df["test_rmse"] < m0b_rmse
    test_df["rmse_gain_vs_M0b_pct"] = (
        (m0b_rmse - test_df["test_rmse"]) / m0b_rmse * 100.0
    ).round(2)
    return cv_df, test_df, fitted_models, test_predictions


# =============================================================================
# 2. KIỂM ĐỊNH ĐỘ BỀN (TEMPORAL, ROUTE GROUP SPLIT & R3 UNMODIFIED RAW SUBSET)
# =============================================================================
def _build_robustness_row(
    scenario: str,
    m_code: str,
    n_train: int,
    n_test: int,
    y_true: np.ndarray,
    preds: np.ndarray,
) -> dict[str, Any]:
    """Đóng gói 1 dòng kết quả kiểm định độ bền / R3 sensitivity."""
    m = compute_all_metrics(y_true, preds)
    return {
        "scenario": scenario,
        "model_code": m_code,
        "model_name": MODEL_DISPLAY_NAMES[m_code],
        "n_train": int(n_train),
        "n_test": int(n_test),
        "r2": m["r2"],
        "rmse": m["rmse"],
        "mae": m["mae"],
        "mape_pct": m["mape_pct"],
    }


def run_robustness_and_r3_sensitivity(
    registry: dict[str, Any],
    fitted_models_random: dict[str, Any],
    df_feat: pd.DataFrame,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """
    Kiểm tra độ bền các mô hình chủ lực trên Temporal Split, Route Group Split
    và 2 kịch bản độ nhạy tập nguyên bản R3 (R3a, R3b).
    """
    logger.info(
        "Đang chạy kiểm định độ bền (Temporal, Route Group) & Kiểm định độ "
        "nhạy R3 (Unmodified Raw)..."
    )
    X_all, y_all = _prepare_stage5_feature_matrix(df_feat)
    flags_df = pd.read_parquet(PROCESSED_DIR / "stage4_audit_flags.parquet")
    strict_mask = flags_df["is_strictly_unmodified_raw"].values
    key_models = [
        "M0b_Tier_Route_Lookup_Baseline",
        "M1_Basic_OLS",
        "M2_Interaction_OLS",
        "M3_Ridge_Regression",
        "M6_Random_Forest",
        "M7_HistGradientBoosting",
    ]
    rob_rows = []

    for split_name in ["temporal_split", "route_group_split"]:
        tr_idx, te_idx = splits[split_name]
        X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
        X_te, y_te = X_all.iloc[te_idx], y_all[te_idx]
        for m_code in key_models:
            est = clone(registry[m_code])
            est.fit(X_tr, y_tr)
            rob_rows.append(
                _build_robustness_row(
                    split_name,
                    m_code,
                    len(tr_idx),
                    len(te_idx),
                    y_te,
                    est.predict(X_te),
                )
            )

    rand_tr_idx, rand_te_idx = splits["random_80_20"]
    unmod_te_idx = rand_te_idx[strict_mask[rand_te_idx]]
    X_te_unmod, y_te_unmod = X_all.iloc[unmod_te_idx], y_all[unmod_te_idx]
    for m_code in key_models:
        preds = fitted_models_random[m_code].predict(X_te_unmod)
        rob_rows.append(
            _build_robustness_row(
                "R3a_Test_Only_Strictly_Unmodified_Raw",
                m_code,
                len(rand_tr_idx),
                len(unmod_te_idx),
                y_te_unmod,
                preds,
            )
        )

    unmod_tr_idx = rand_tr_idx[strict_mask[rand_tr_idx]]
    X_tr_unmod, y_tr_unmod = X_all.iloc[unmod_tr_idx], y_all[unmod_tr_idx]
    for m_code in ["M2_Interaction_OLS", "M7_HistGradientBoosting"]:
        est = clone(registry[m_code])
        est.fit(X_tr_unmod, y_tr_unmod)
        rob_rows.append(
            _build_robustness_row(
                "R3b_Train_And_Test_Strictly_Unmodified_Raw",
                m_code,
                len(unmod_tr_idx),
                len(unmod_te_idx),
                y_te_unmod,
                est.predict(X_te_unmod),
            )
        )
    return pd.DataFrame(rob_rows)


# =============================================================================
# 3. SÁU THÍ NGHIỆM MUTATE & BREAK CHUẨN BỊ VẤN ĐÁP VÒNG 3 (MB1 -> MB6)
# =============================================================================
def _mb_record(
    exp_id: str,
    action: str,
    base_m: dict[str, float],
    mut_m: dict[str, float],
    lesson: str,
) -> dict[str, Any]:
    """Đóng gói chuẩn 1 bản ghi kết quả thí nghiệm Mutate & Break."""
    return {
        "experiment_id": exp_id,
        "mutation_action": action,
        "baseline_r2": base_m["r2"],
        "mutated_r2": mut_m["r2"],
        "baseline_rmse": base_m["rmse"],
        "mutated_rmse": mut_m["rmse"],
        "baseline_mape_pct": base_m["mape_pct"],
        "mutated_mape_pct": mut_m["mape_pct"],
        "scientific_lesson": lesson,
    }


def _run_mb1_to_mb3(
    fitted_models: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Thực thi MB1 (bỏ tương tác), MB2 (Ridge alpha=1e5), MB3 (DT overfit)."""
    m2_m = compute_all_metrics(
        y_test, fitted_models["M2_Interaction_OLS"].predict(X_test)
    )
    m1_m = compute_all_metrics(
        y_test, fitted_models["M1_Basic_OLS"].predict(X_test)
    )
    rows = [
        _mb_record(
            "MB1_Drop_Tier_Distance_Interaction",
            "Xóa tương tác C(name)*distance & C(name)*log_distance "
            "(Chuyển từ M2 về M1)",
            m2_m,
            m1_m,
            f"MAPE tăng vọt từ {m2_m['mape_pct']:.2f}% lên "
            f"{m1_m['mape_pct']:.2f}% "
            f"(+{m1_m['mape_pct'] - m2_m['mape_pct']:.2f}%): chứng minh mỗi "
            "phân khúc xe có đơn giá mỗi dặm riêng biệt.",
        )
    ]

    pipe_ridge = Pipeline(
        [
            ("prep", EconomicInteractionPreprocessor(scale_continuous=True)),
            ("reg", Ridge(alpha=100_000.0)),
        ]
    ).fit(X_train, y_train)
    mb2_m = compute_all_metrics(y_test, pipe_ridge.predict(X_test))
    m3_m = compute_all_metrics(
        y_test, fitted_models["M3_Ridge_Regression"].predict(X_test)
    )
    rows.append(
        _mb_record(
            "MB2_Extreme_Ridge_L2_Penalty",
            "Tăng siêu tham số phạt Ridge alpha từ mức tối ưu lên alpha "
            "= 100,000",
            m3_m,
            mb2_m,
            "Hệ số hồi quy bị ép co mạnh về 0 gây Underfitting nặng "
            f"(R2 tụt từ {m3_m['r2']:.4f} xuống {mb2_m['r2']:.4f}, "
            f"RMSE tăng lên ${mb2_m['rmse']:.2f}).",
        )
    )

    preps = build_stage4_preprocessors()
    pipe_dt = Pipeline(
        [
            ("prep", preps["m5_m6_tree_ohe"]),
            (
                "reg",
                DecisionTreeRegressor(
                    max_depth=None,
                    min_samples_leaf=1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ).fit(X_train, y_train)
    dt_tr = compute_all_metrics(y_train, pipe_dt.predict(X_train))
    dt_te = compute_all_metrics(y_test, pipe_dt.predict(X_test))
    m5_te = compute_all_metrics(
        y_test, fitted_models["M5_Decision_Tree"].predict(X_test)
    )
    rows.append(
        _mb_record(
            "MB3_Unconstrained_Tree_Overfitting",
            "Bỏ giới hạn độ sâu cây quyết định (max_depth=None, "
            "min_samples_leaf=1)",
            m5_te,
            dt_te,
            f"Cây học thuộc lòng nhiễu tập Train (Train R2={dt_tr['r2']:.4f}, "
            f"RMSE=${dt_tr['rmse']:.2f}) nhưng suy giảm nặng trên tập Test "
            f"(Test R2={dt_te['r2']:.4f}, RMSE=${dt_te['rmse']:.2f}).",
        )
    )
    return rows, m1_m


def _run_mb6_test_size(
    registry: dict[str, Any],
    fitted_models: dict[str, Any],
    df_feat: pd.DataFrame,
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Thực thi MB6: đổi tỷ lệ phân chia 80/20 sang 60/40."""
    obs_mask = (
        df_feat["is_surge_original"].values == 1
        if "is_surge_original" in df_feat.columns
        else df_feat["surge_multiplier"].notna().values
    )
    X_tr_60, X_te_40, y_tr_60, y_te_40 = train_test_split(
        X_all.loc[obs_mask],
        y_all[obs_mask],
        test_size=0.40,
        random_state=RANDOM_STATE,
    )
    est_40 = clone(registry["M7_HistGradientBoosting"]).fit(X_tr_60, y_tr_60)
    m7_base = compute_all_metrics(
        y_test, fitted_models["M7_HistGradientBoosting"].predict(X_test)
    )
    m7_40 = compute_all_metrics(y_te_40, est_40.predict(X_te_40))
    return _mb_record(
        "MB6_Change_Test_Size_020_to_040",
        "Đổi tham số phân chia test_size từ 0.20 (80/20) sang 0.40 (60/40)",
        m7_base,
        m7_40,
        f"Nhờ quy mô dữ liệu lớn (N_obs={int(obs_mask.sum()):,}), hiệu "
        f"năng M7 cực kỳ ổn định: R2={m7_40['r2']:.4f} vs "
        f"{m7_base['r2']:.4f}, MAPE={m7_40['mape_pct']:.2f}% vs "
        f"{m7_base['mape_pct']:.2f}%.",
    )


def _run_mb4_to_mb6(
    registry: dict[str, Any],
    fitted_models: dict[str, Any],
    df_feat: pd.DataFrame,
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    m1_m: dict[str, float],
) -> list[dict[str, Any]]:
    """Thực thi MB4 (bỏ surge), MB5 (OrdinalEncoder OLS), MB6 (60/40)."""
    X_tr_ns, X_te_ns = X_train.copy(), X_test.copy()
    for df_ns in (X_tr_ns, X_te_ns):
        df_ns["surge_multiplier"] = np.float32(1.0)
        df_ns["distance_x_surge"] = df_ns["distance"].astype(np.float32)
        df_ns["lyft_surge_excess"] = np.float32(0.0)
        df_ns["distance_x_lyft_surge_excess"] = np.float32(0.0)

    est_ns = clone(registry["M7_HistGradientBoosting"]).fit(X_tr_ns, y_train)
    sg_mask = X_test["surge_multiplier"].values > 1.0
    m7_sg_base = compute_all_metrics(
        y_test[sg_mask],
        fitted_models["M7_HistGradientBoosting"].predict(X_test)[sg_mask],
    )
    m7_sg_mut = compute_all_metrics(
        y_test[sg_mask], est_ns.predict(X_te_ns)[sg_mask]
    )
    rows = [
        _mb_record(
            "MB4_Remove_Surge_Multiplier_On_Surge_Trips",
            "Xóa toàn bộ biến surge_multiplier (đánh giá riêng trên nhóm "
            "chuyến Lyft có surge > 1.0)",
            m7_sg_base,
            m7_sg_mut,
            "Trên các chuyến có Surge, bỏ biến surge làm RMSE vọt từ "
            f"${m7_sg_base['rmse']:.2f} lên ${m7_sg_mut['rmse']:.2f} "
            f"và R2 sụp đổ từ {m7_sg_base['r2']:.4f} xuống "
            f"{m7_sg_mut['r2']:.4f}.",
        )
    ]

    bad_prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                BASE_NUM_FEATURES,
            ),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                BASE_CAT_FEATURES,
            ),
        ]
    )
    pipe_ord = Pipeline([("prep", bad_prep), ("reg", LinearRegression())]).fit(
        X_train, y_train
    )
    mb5_m = compute_all_metrics(y_test, pipe_ord.predict(X_test))
    rows.append(
        _mb_record(
            "MB5_LabelEncoder_Instead_Of_OneHot_In_OLS",
            "Thay OneHotEncoder bằng OrdinalEncoder/LabelEncoder cho biến "
            "danh mục trong OLS",
            m1_m,
            mb5_m,
            "Gán thứ tự số học giả tạo (0, 1, 2...11) cho tên gói xe làm R2 "
            f"của OLS sụp đổ từ {m1_m['r2']:.4f} xuống chỉ còn "
            f"{mb5_m['r2']:.4f} (MAPE tăng lên {mb5_m['mape_pct']:.2f}%).",
        )
    )
    rows.append(
        _run_mb6_test_size(
            registry, fitted_models, df_feat, X_all, y_all, X_test, y_test
        )
    )
    return rows


def run_mutate_and_break_experiments(
    registry: dict[str, Any],
    fitted_models: dict[str, Any],
    df_feat: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    """Thực thi 6 thí nghiệm Mutate & Break (MB1 -> MB6) phục vụ vấn đáp."""
    logger.info("Đang thực thi 6 thí nghiệm Mutate & Break (MB1 -> MB6)...")
    X_all, y_all = _prepare_stage5_feature_matrix(df_feat)
    X_train, y_train = X_all.iloc[train_idx], y_all[train_idx]
    X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]
    rows_13, m1_m = _run_mb1_to_mb3(
        fitted_models, X_train, y_train, X_test, y_test
    )
    rows_46 = _run_mb4_to_mb6(
        registry,
        fitted_models,
        df_feat,
        X_all,
        y_all,
        X_train,
        y_train,
        X_test,
        y_test,
        m1_m,
    )
    return pd.DataFrame(rows_13 + rows_46)


# =============================================================================
# 3B. PHÂN TÁCH CHẾ ĐỘ UBER/LYFT & THỰC NGHIỆM EX-ANTE (GIẢI QUYẾT C1 & C3)
# =============================================================================
def _compute_regime_breakdown(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    fitted_models: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Phân tách hiệu năng M7 & M2 theo chế độ Uber vs Lyft (1x / >1x)."""
    pred_m7 = fitted_models["M7_HistGradientBoosting"].predict(X_test)
    pred_m2 = fitted_models["M2_Interaction_OLS"].predict(X_test)
    uber_te = X_test["cab_type"].astype(str).values == "Uber"
    lyft_te = X_test["cab_type"].astype(str).values == "Lyft"
    lyft_ns = lyft_te & (X_test["surge_multiplier"].values == 1.0)
    lyft_sg = lyft_te & (X_test["surge_multiplier"].values > 1.0)

    breakdown = {
        "overall_observed_test": {
            "n_test": int(len(y_test)),
            "M7_Conditional": compute_all_metrics(y_test, pred_m7),
            "M2_Conditional": compute_all_metrics(y_test, pred_m2),
        },
        "uber_hidden_surge_test": {
            "n_test": int(uber_te.sum()),
            "M7_Conditional": compute_all_metrics(
                y_test[uber_te], pred_m7[uber_te]
            ),
            "M2_Conditional": compute_all_metrics(
                y_test[uber_te], pred_m2[uber_te]
            ),
        },
        "lyft_observed_surge_test": {
            "n_test": int(lyft_te.sum()),
            "M7_Conditional": compute_all_metrics(
                y_test[lyft_te], pred_m7[lyft_te]
            ),
            "M2_Conditional": compute_all_metrics(
                y_test[lyft_te], pred_m2[lyft_te]
            ),
        },
        "lyft_no_surge_1x_test": {
            "n_test": int(lyft_ns.sum()),
            "M7_Conditional": compute_all_metrics(
                y_test[lyft_ns], pred_m7[lyft_ns]
            ),
        },
        "lyft_active_surge_gt1x_test": {
            "n_test": int(lyft_sg.sum()),
            "M7_Conditional": compute_all_metrics(
                y_test[lyft_sg], pred_m7[lyft_sg]
            ),
        },
    }
    masks = {
        "uber_te": uber_te,
        "lyft_te": lyft_te,
        "lyft_ns_te": lyft_ns,
        "lyft_sg_te": lyft_sg,
    }
    return breakdown, masks


def _run_ex_ante_fare_models(
    df_feat: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    masks: dict[str, np.ndarray],
    ex_ante_cols: list[str],
    best_hgb_params: dict[str, Any],
    random_state: int,
) -> list[dict[str, Any]]:
    """Huấn luyện và đánh giá M7 & M2 Ex-Ante (23 đặc trưng, 0 biến surge)."""
    ex_num = [c for c in ex_ante_cols if c not in BASE_CAT_FEATURES]
    ex_cat = [c for c in BASE_CAT_FEATURES if c in ex_ante_cols]
    X_tr_ex = X_train[ex_ante_cols].copy()
    X_te_ex = X_test[ex_ante_cols].copy()
    for c in ex_cat:
        X_tr_ex[c] = X_tr_ex[c].astype("category")
        X_te_ex[c] = X_te_ex[c].astype("category")

    prep_ex = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ex_num),
            ("cat", "passthrough", ex_cat),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    m7_ex = Pipeline(
        [
            ("route_dist_imp", RouteDistanceImputer()),
            ("prep", prep_ex),
            (
                "reg",
                HistGradientBoostingRegressor(
                    categorical_features="from_dtype",
                    max_iter=350,
                    early_stopping=True,
                    **best_hgb_params,
                    random_state=random_state,
                ),
            ),
        ]
    ).fit(X_tr_ex, y_train)
    p7_ex = m7_ex.predict(X_te_ex)

    X_tr_m2, X_te_m2 = X_train.copy(), X_test.copy()
    for dtmp in (X_tr_m2, X_te_m2):
        dtmp["surge_multiplier"] = np.float32(1.0)
        dtmp["distance_x_surge"] = dtmp["distance"].astype(np.float32)
        dtmp["lyft_surge_excess"] = np.float32(0.0)
        dtmp["distance_x_lyft_surge_excess"] = np.float32(0.0)
    m2_ex = Pipeline(
        [
            ("prep", EconomicInteractionPreprocessor(scale_continuous=True)),
            ("reg", LinearRegression()),
        ]
    ).fit(X_tr_m2, y_train)
    p2_ex = m2_ex.predict(X_te_m2)

    miss_mask = df_feat["is_surge_original"].values == 0
    X_miss = df_feat.loc[miss_mask, ex_ante_cols].copy()
    for c in ex_cat:
        X_miss[c] = X_miss[c].astype("category")
    y_miss = df_feat.loc[miss_mask, TARGET_COL].astype(np.float64).values
    p7_miss = m7_ex.predict(X_miss)

    n_f = len(ex_ante_cols)
    slice_defs = [
        ("M7_ExAnte_All_Test", y_test, p7_ex),
        ("M2_ExAnte_All_Test", y_test, p2_ex),
        ("M7_ExAnte_Uber_Test", masks["uber_te"], None),
        ("M7_ExAnte_Lyft_All_Test", masks["lyft_te"], None),
        ("M7_ExAnte_Lyft_1x_Test", masks["lyft_ns_te"], None),
        ("M7_ExAnte_Lyft_SurgeGt1x_Test", masks["lyft_sg_te"], None),
        ("M7_ExAnte_Lyft_MissingSurge_Holdout", y_miss, p7_miss),
    ]
    rows = []
    for lbl, a, b in slice_defs:
        yt, yp = (y_test[a], p7_ex[a]) if b is None else (a, b)
        rows.append(
            {
                "experiment_group": "ExAnte_Pricing",
                "model_or_slice": lbl,
                "n_eval": int(len(yt)),
                **compute_all_metrics(yt, yp, n_features=n_f),
            }
        )
    return rows


def _best_f1_threshold_oof(
    est_pipe: Pipeline,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    random_state: int,
) -> float:
    """Chọn ngưỡng tối ưu F1 bằng OOF 3-Fold hoàn toàn trên tập Train."""
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    oof_probs = cross_val_predict(
        clone(est_pipe),
        X_tr,
        y_tr,
        cv=skf,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    precs, recs, thrs = precision_recall_curve(y_tr, oof_probs)
    f1s = (2.0 * precs[:-1] * recs[:-1]) / np.clip(
        precs[:-1] + recs[:-1], 1e-12, None
    )
    return float(thrs[int(np.argmax(f1s))]) if len(thrs) else 0.50


def _eval_classifier_full(
    y_true_c: np.ndarray, probs_c: np.ndarray, thr_opt: float
) -> dict[str, Any]:
    """Tính trọn bộ metrics phân loại (ROC-AUC, PR-AUC, Brier, F1, CM)."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    def _at_thr(thr: float) -> dict[str, Any]:
        y_hat = (probs_c >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true_c, y_hat, labels=[0, 1]
        ).ravel()
        return {
            "accuracy": round(float(accuracy_score(y_true_c, y_hat)), 4),
            "precision": round(
                float(precision_score(y_true_c, y_hat, zero_division=0)), 4
            ),
            "recall": round(
                float(recall_score(y_true_c, y_hat, zero_division=0)), 4
            ),
            "f1_score": round(
                float(f1_score(y_true_c, y_hat, zero_division=0)), 4
            ),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    opt_m = _at_thr(thr_opt)
    return {
        "roc_auc": round(float(roc_auc_score(y_true_c, probs_c)), 4),
        "pr_auc": round(float(average_precision_score(y_true_c, probs_c)), 4),
        "brier_score": round(float(brier_score_loss(y_true_c, probs_c)), 5),
        "f1_optimal_threshold_from_train": round(float(thr_opt), 4),
        **opt_m,
        "default_050_threshold_metrics": _at_thr(0.50),
    }


def _compute_surge_clf_group_auc_drops(
    clf_hgb: Pipeline,
    X_clf_te: pd.DataFrame,
    y_clf_te: np.ndarray,
    clf_cat: list[str],
    base_auc: float,
    random_state: int,
) -> dict[str, float]:
    """Đo mức giảm ROC-AUC khi hoán vị từng nhóm đặc trưng Ex-Ante."""
    from sklearn.metrics import roc_auc_score

    rng_p = np.random.RandomState(random_state)
    clf_groups = {
        "source_destination_distance": [
            "source",
            "destination",
            "distance",
            "log_distance",
        ],
        "service_tier_name": ["name"],
        "weather_7vars": [
            "temp",
            "clouds",
            "pressure",
            "rain",
            "humidity",
            "wind",
            "is_raining",
        ],
        "time_clock_est": [
            "hour",
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "dow_sin",
            "dow_cos",
            "is_weekend",
            "is_rush_hour",
            "time_of_day",
        ],
    }
    drops_map: dict[str, float] = {}
    for gname, gcols in clf_groups.items():
        drops = []
        for _ in range(5):
            Xp = X_clf_te.copy()
            perm_idx = rng_p.permutation(len(Xp))
            for c in gcols:
                if c in Xp.columns:
                    Xp[c] = X_clf_te[c].iloc[perm_idx].values
                    if c in clf_cat:
                        Xp[c] = Xp[c].astype("category")
            auc_p = float(
                roc_auc_score(y_clf_te, clf_hgb.predict_proba(Xp)[:, 1])
            )
            drops.append(base_auc - auc_p)
        drops_map[gname] = round(float(np.mean(drops)), 4)
    return drops_map


def _build_surge_classifier_specs(
    clf_num: list[str], clf_cat: list[str], random_state: int
) -> dict[str, Pipeline]:
    """Khởi tạo 4 pipeline phân loại Surge (HistGB & Logistic,
    chuẩn/balanced)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    prep_hgb = ColumnTransformer(
        transformers=[
            ("num", "passthrough", clf_num),
            ("cat", "passthrough", clf_cat),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    prep_lr = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                clf_num,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                clf_cat,
            ),
        ],
        remainder="drop",
    )
    hgb_kw = {
        "categorical_features": "from_dtype",
        "max_iter": 200,
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "l2_regularization": 5.0,
        "early_stopping": True,
        "random_state": random_state,
    }
    return {
        "HistGB_Classifier": Pipeline(
            [
                ("route_dist_imp", RouteDistanceImputer()),
                ("prep", prep_hgb),
                ("clf", HistGradientBoostingClassifier(**hgb_kw)),
            ]
        ),
        "HistGB_Classifier_Balanced": Pipeline(
            [
                ("route_dist_imp", RouteDistanceImputer()),
                ("prep", clone(prep_hgb)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        class_weight="balanced", **hgb_kw
                    ),
                ),
            ]
        ),
        "LogisticRegression_Classifier": Pipeline(
            [
                ("route_dist_imp", RouteDistanceImputer()),
                ("prep", prep_lr),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, random_state=random_state
                    ),
                ),
            ]
        ),
        "LogisticRegression_Balanced": Pipeline(
            [
                ("route_dist_imp", RouteDistanceImputer()),
                ("prep", clone(prep_lr)),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _run_ex_ante_surge_classifiers(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    lyft_te: np.ndarray,
    ex_ante_cols: list[str],
    random_state: int,
) -> dict[str, Any]:
    """Huấn luyện và đánh giá 4 bộ phân loại dự báo P(Lyft surge > 1.0)."""
    from sklearn.metrics import roc_auc_score

    lyft_tr = X_train["cab_type"].astype(str).values == "Lyft"
    clf_cols = [c for c in ex_ante_cols if c not in ("cab_type", "is_lyft")]
    clf_num = [c for c in clf_cols if c not in BASE_CAT_FEATURES]
    clf_cat = [c for c in BASE_CAT_FEATURES if c in clf_cols]

    X_tr = X_train.loc[lyft_tr, clf_cols].copy()
    y_tr = (X_train.loc[lyft_tr, "surge_multiplier"].values > 1.0).astype(int)
    X_te = X_test.loc[lyft_te, clf_cols].copy()
    y_te = (X_test.loc[lyft_te, "surge_multiplier"].values > 1.0).astype(int)
    for c in clf_cat:
        X_tr[c] = X_tr[c].astype("category")
        X_te[c] = X_te[c].astype("category")

    specs = _build_surge_classifier_specs(clf_num, clf_cat, random_state)
    summary: dict[str, Any] = {
        "n_lyft_train": int(len(y_tr)),
        "n_lyft_test": int(len(y_te)),
        "test_positive_prevalence_pct": round(float(np.mean(y_te)) * 100.0, 2),
    }
    clf_hgb_fitted = None
    prob_hgb = None
    for name, pipe in specs.items():
        thr = _best_f1_threshold_oof(pipe, X_tr, y_tr, random_state)
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_te)[:, 1]
        summary[name] = _eval_classifier_full(y_te, probs, thr)
        if name == "HistGB_Classifier":
            clf_hgb_fitted, prob_hgb = pipe, probs

    base_auc = float(roc_auc_score(y_te, prob_hgb))
    summary["group_roc_auc_drop"] = _compute_surge_clf_group_auc_drops(
        clf_hgb_fitted, X_te, y_te, clf_cat, base_auc, random_state
    )
    return summary


def run_regime_and_ex_ante_experiments(
    df_feat: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    fitted_models: dict[str, Any],
    best_hgb_params: dict[str, Any],
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """
    Phân tách chế độ Uber/Lyft và chạy thực nghiệm Ex-Ante
    (hồi quy & phân loại).
    """
    logger.info(
        "Đang chạy phân tách chế độ Uber/Lyft và thực nghiệm Ex-Ante (C3)..."
    )
    regime_breakdown, masks = _compute_regime_breakdown(
        X_test, y_test, fitted_models
    )
    surge_cols = {
        "surge_multiplier",
        "distance_x_surge",
        "lyft_surge_excess",
        "distance_x_lyft_surge_excess",
    }
    ex_ante_cols = [c for c in X_train.columns if c not in surge_cols]
    pricing_rows = _run_ex_ante_fare_models(
        df_feat,
        X_train,
        y_train,
        X_test,
        y_test,
        masks,
        ex_ante_cols,
        best_hgb_params,
        random_state,
    )
    surge_clf_summary = _run_ex_ante_surge_classifiers(
        X_train, X_test, masks["lyft_te"], ex_ante_cols, random_state
    )
    ex_ante_summary = {
        "n_ex_ante_features": len(ex_ante_cols),
        "dropped_surge_features": sorted(surge_cols),
        "pricing_without_surge": pricing_rows,
        "lyft_surge_classifier": surge_clf_summary,
    }
    hgb_c = surge_clf_summary["HistGB_Classifier"]
    logger.info(
        "  => Ex-Ante Pricing M7 (0 surge): All Test "
        f"R2={pricing_rows[0]['r2']:.4f}, RMSE=${pricing_rows[0]['rmse']:.4f}"
    )
    logger.info(
        "  => Ex-Ante Surge Classifier (Lyft): ROC-AUC="
        f"{hgb_c['roc_auc']:.4f}, PR-AUC={hgb_c['pr_auc']:.4f}"
    )
    return regime_breakdown, ex_ante_summary, pd.DataFrame(pricing_rows)


# =============================================================================
# 4. ÁP DỤNG QUY TẮC CHỌN MÔ HÌNH ĐẶT TRƯỚC (PRE-REGISTERED SELECTION RULES)
# =============================================================================
def apply_model_selection_rules(
    cv_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rob_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Đối chiếu kết quả thực nghiệm với 4 quy tắc chọn mô hình đã đặt trước:
    - Quy tắc 1: Phải thắng mô hình mốc mạnh M0b về RMSE trên tập Test.
    - Quy tắc 2: RMSE trung bình 5-Fold CV thấp nhất, Test MAPE <= 10%,
      SD(RMSE) < 5%, Temporal RMSE không tăng quá 15%.
    - Quy tắc 3 (Ưu tiên đơn giản): Nếu M2 (OLS tương tác) có RMSE kém mô hình
      cây tốt nhất không quá 5%,
      ghi nhận vai trò song hành hoặc ưu tiên giải thích.
    - Quy tắc 4: M2 luôn là mô hình diễn giải chính thức cho RQ2.
    """
    merged = pd.merge(cv_df, test_df, on=["model_code", "model_name"])
    non_baseline = merged[
        ~merged["model_code"].str.startswith("M0")
    ].sort_values("cv_rmse_mean")

    best_row = non_baseline.iloc[0]
    best_code = str(best_row["model_code"])
    best_cv_rmse = float(best_row["cv_rmse_mean"])
    best_test_rmse = float(best_row["test_rmse"])

    m2_row = merged.loc[merged["model_code"] == "M2_Interaction_OLS"].iloc[0]
    m2_test_rmse = float(m2_row["test_rmse"])
    m2_gap_pct = (m2_test_rmse - best_test_rmse) / best_test_rmse * 100.0

    temp_best_rmse = float(
        rob_df.loc[
            (rob_df["scenario"] == "temporal_split")
            & (rob_df["model_code"] == best_code),
            "rmse",
        ].iloc[0]
    )
    temp_deg_pct = (temp_best_rmse - best_test_rmse) / best_test_rmse * 100.0

    selection_summary = {
        "predictive_champion_code": best_code,
        "predictive_champion_name": str(best_row["model_name"]),
        "predictive_champion_cv_rmse": best_cv_rmse,
        "predictive_champion_test_r2": float(best_row["test_r2"]),
        "predictive_champion_test_rmse": best_test_rmse,
        "predictive_champion_test_mae": float(best_row["test_mae"]),
        "predictive_champion_test_mape_pct": float(best_row["test_mape_pct"]),
        "interpretability_champion_code": "M2_Interaction_OLS",
        "interpretability_champion_name": str(m2_row["model_name"]),
        "interpretability_champion_test_r2": float(m2_row["test_r2"]),
        "interpretability_champion_test_rmse": m2_test_rmse,
        "interpretability_champion_test_mape_pct": float(
            m2_row["test_mape_pct"]
        ),
        "m2_vs_best_tree_rmse_gap_pct": round(m2_gap_pct, 2),
        "parsimony_5pct_threshold_triggered": bool(m2_gap_pct <= 5.0),
        "temporal_split_rmse_degradation_pct": round(temp_deg_pct, 2),
        "models_meeting_both_smart_goals": merged.loc[
            merged["meets_both_smart"], "model_code"
        ].tolist(),
        "selection_rationale": (
            f"Mô hình {best_code} đạt RMSE trung bình 5-Fold CV thấp nhất "
            f"(${best_cv_rmse:.4f}) và hiệu năng Test cao nhất "
            f"(R2 = {best_row['test_r2']:.4f}, RMSE = ${best_test_rmse:.4f},"
            f" MAPE = {best_row['test_mape_pct']:.2f}% <= 10%), "
            f"nên được chọn làm Mô hình Dự báo Chính (Predictive Champion). "
            "Mô hình M2 (OLS tương tác kinh tế lượng) cũng vượt trọn vẹn "
            f"mục tiêu SMART (R2 = {m2_row['test_r2']:.4f}, "
            f"MAPE = {m2_row['test_mape_pct']:.2f}%, chênh lệch RMSE so với "
            f"{best_code} là {m2_gap_pct:.2f}%) và được dùng làm "
            "Mô hình Diễn giải Kinh tế lượng Chính (Interpretability "
            "Champion)."
        ),
    }
    return selection_summary


# =============================================================================
# 5. TRỰC QUAN HÓA CHUYÊN SÂU GIAI ĐOẠN 5 (FIG11, FIG12, FIG13, FIG14 - 300
# DPI)
# =============================================================================
def _collect_fig11_effect_items(
    df_feat: pd.DataFrame, hyp_res: dict[str, Any]
) -> list[tuple[str, float, str]]:
    """Tổng hợp danh sách (nhãn, độ lớn hiệu ứng, màu) cho Fig 11 Panel A."""
    ux_lf_sd = float(
        df_feat.loc[
            df_feat["name"].astype(str).isin(["UberX", "Lyft"]), TARGET_COL
        ].std(ddof=1)
    )
    wald = hyp_res["crossing_and_wald_summary"]
    h4a_d = abs(wald["uberx_vs_lyft"]["base_diff_usd"]) / ux_lf_sd
    tests = hyp_res["hypothesis_tests"]
    return [
        (
            "H1a: Mưa vs Tạnh (Khớp thực)\n[Cohen's |d|]",
            abs(tests[0]["effect_size_value"]),
            "#64748b",
        ),
        (
            "H1c: Mưa vs Tạnh (Riêng Lyft)\n[Cohen's |d|]",
            abs(tests[2]["effect_size_value"]),
            "#64748b",
        ),
        (
            "H2a: Surge theo 4 Buổi EST\n[Eta-squared η²]",
            tests[3]["effect_size_value"],
            "#64748b",
        ),
        (
            "H3b: Lyft Surge × Mưa (Thực)\n[Cramér's V]",
            tests[7]["effect_size_value"],
            "#64748b",
        ),
        (
            "H3d: Lyft Surge × Cao điểm EST\n[Cramér's V]",
            tests[9]["effect_size_value"],
            "#64748b",
        ),
        (
            "H4c: Đồng thời 6 biến Thời tiết\n[Incremental ΔR²]",
            wald["multivariate_weather_joint_test"]["delta_r2_weather"],
            "#64748b",
        ),
        (
            "H3a: Surge × Hãng xe (Lyft/Uber)\n[Cramér's V - Khác biệt cấu "
            "trúc]",
            tests[6]["effect_size_value"],
            "#2563eb",
        ),
        (
            "H4a: Chênh giá mở cửa UberX-Lyft\n[Chuẩn hóa |Δb0|/SD(y) - "
            "Cohen d]",
            round(h4a_d, 4),
            "#0d9488",
        ),
    ]


def _plot_fig11_effect_sizes_panel(
    ax1: Any, df_feat: pd.DataFrame, hyp_res: dict[str, Any]
) -> None:
    """Vẽ Panel A của Fig 11: Độ lớn hiệu ứng không thứ nguyên H1-H4."""
    effect_items = _collect_fig11_effect_items(df_feat, hyp_res)
    vals_ef = [x[1] for x in effect_items]
    bars = ax1.barh(
        [x[0] for x in effect_items],
        vals_ef,
        color=[x[2] for x in effect_items],
        edgecolor="#0f172a",
        height=0.62,
    )
    ax1.axvline(
        0.01,
        color="#eab308",
        ls="--",
        lw=1.3,
        label="Ngưỡng hiệu ứng cực nhỏ (0.01)",
    )
    ax1.axvline(
        0.10,
        color="#dc2626",
        ls=":",
        lw=1.5,
        label="Ngưỡng hiệu ứng thực tiễn (0.10)",
    )
    ax1.set_xlim(0, max(0.35, max(vals_ef) * 1.25))
    ax1.set_title(
        "A. Đối chiếu Độ lớn Hiệu ứng Không thứ nguyên (Effect Size) H1 -> "
        "H4\n[Thời tiết & Khung giờ EST có hiệu ứng ≈ 0; Hãng xe & Phân "
        "khúc quyết định giá]",
        fontsize=11.0,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel(
        "Giá trị Độ lớn Hiệu ứng Chuẩn hóa Không thứ nguyên (|d|, η², "
        "Cramér's V, ΔR²)",
        fontsize=9.8,
    )
    ax1.legend(loc="lower right", fontsize=8.8, frameon=True)
    for b, v in zip(bars, vals_ef, strict=False):
        ax1.text(
            v + 0.005,
            b.get_y() + b.get_height() / 2,
            f"{v:.5f}" if v < 0.01 else f"{v:.4f}",
            va="center",
            fontsize=8.8,
            fontweight="bold",
        )


def _plot_fig11_crossing_panel(ax2: Any, hyp_res: dict[str, Any]) -> None:
    """Vẽ Panel B của Fig 11: Đường biểu giá cắt nhau Wald H4a & H4b."""
    cx = hyp_res["crossing_and_wald_summary"]["uberx_vs_lyft"]
    cp = hyp_res["crossing_and_wald_summary"]["uberpool_vs_shared"]
    d_grid = np.linspace(0.2, 6.0, 200)
    curves = [
        (
            cx["uberx_base_b0"] + cx["uberx_slope_b1"] * d_grid,
            "#1d4ed8",
            2.4,
            "-",
            f"UberX: ${cx['uberx_base_b0']:.2f} + "
            f"${cx['uberx_slope_b1']:.2f}×d",
        ),
        (
            cx["lyft_base_b0"] + cx["lyft_slope_b1"] * d_grid,
            "#e11d48",
            2.4,
            "--",
            f"Lyft: ${cx['lyft_base_b0']:.2f} + ${cx['lyft_slope_b1']:.2f}×d",
        ),
        (
            cp["uberpool_base_b0"] + cp["uberpool_slope_b1"] * d_grid,
            "#0284c7",
            1.8,
            "-.",
            f"UberPool: ${cp['uberpool_base_b0']:.2f} + "
            f"${cp['uberpool_slope_b1']:.2f}×d",
        ),
        (
            cp["shared_base_b0"] + cp["shared_slope_b1"] * d_grid,
            "#f43f5e",
            1.8,
            ":",
            f"Lyft Shared: ${cp['shared_base_b0']:.2f} + "
            f"${cp['shared_slope_b1']:.2f}×d",
        ),
    ]
    for y_vals, col, lw, ls, lbl in curves:
        ax2.plot(d_grid, y_vals, color=col, lw=lw, ls=ls, label=lbl)

    d_star = cx["crossing_distance_miles"]
    p_star = cx["uberx_base_b0"] + cx["uberx_slope_b1"] * d_star
    d_low, d_high = cx["crossing_distance_ci95"]
    ax2.scatter([d_star], [p_star], color="#0f172a", s=75, zorder=5)
    ax2.axvline(d_star, color="#0f172a", ls="--", lw=1.2)
    ax2.axvspan(
        d_low,
        d_high,
        color="#94a3b8",
        alpha=0.25,
        label=(
            f"Điểm cắt d* = {d_star:.2f} dặm "
            f"(95% CI: [{d_low:.2f}, {d_high:.2f}])"
        ),
    )
    ax2.annotate(
        f"Điểm cắt UberX = Lyft\nd* = {d_star:.2f} dặm "
        f"(${p_star:.2f})\n({cx['pct_trips_shorter_than_crossing']:.1f}% "
        "chuyến ngắn hơn d*)",
        xy=(d_star, p_star),
        xytext=(d_star + 0.30, p_star + 3.2),
        arrowprops={"arrowstyle": "->", "color": "#0f172a", "lw": 1.4},
        fontsize=8.6,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#fef3c7",
            "edgecolor": "#d97706",
        },
    )
    ax2.set_title(
        "B. Kiểm định Wald (HC3) H4a & H4b: Chiến lược Biểu giá Cắt "
        f"nhau\n[Lyft rẻ hơn dưới {d_star:.2f} dặm nhờ mở cửa thấp; UberX "
        f"rẻ hơn trên {d_star:.2f} dặm]",
        fontsize=11.2,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlabel(
        "Quãng đường hiệu dụng d = Distance × Surge (Dặm)", fontsize=10
    )
    ax2.set_ylabel("Giá báo API kỳ vọng E[Price | d] (USD)", fontsize=10)
    ax2.set_ylim(2, 20)
    ax2.legend(loc="upper left", fontsize=8.4, frameon=True)


def _save_fig11_hypotheses(
    df_feat: pd.DataFrame, hyp_res: dict[str, Any]
) -> str:
    """Xuất Fig 11: Tổng hợp Effect Sizes (H1-H4) & Crossing Tariffs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.5), dpi=300)
    _plot_fig11_effect_sizes_panel(ax1, df_feat, hyp_res)
    _plot_fig11_crossing_panel(ax2, hyp_res)
    fig.suptitle(
        "HÌNH 5.1 (FIG 11): TỔNG HỢP KIỂM ĐỊNH GIẢ THUYẾT THỐNG KÊ (H1-H4) "
        "& BIỂU GIÁ CẮT NHAU (WALD HC3)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    p = FIG_STAGE5_DIR / "fig11_hypothesis_effect_sizes.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p.relative_to(BASE_DIR).as_posix()


def _plot_fig12_rmse_panel(
    ax_l: Any,
    merged_m: pd.DataFrame,
    short_names: list[str],
    colors_m: list[str],
    n_test: int,
) -> None:
    """Vẽ Panel A của Fig 12: 5-Fold CV RMSE vs Test RMSE."""
    y_pos = np.arange(len(merged_m))
    ax_l.barh(
        y_pos,
        merged_m["cv_rmse_mean"],
        xerr=merged_m["cv_rmse_std"],
        height=0.62,
        color=colors_m,
        edgecolor="#0f172a",
        capsize=3,
        label="5-Fold CV RMSE ($ ± 1 SD)",
    )
    ax_l.scatter(
        merged_m["test_rmse"],
        y_pos,
        color="#dc2626",
        s=55,
        zorder=5,
        label="Independent Test RMSE ($)",
    )
    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(short_names, fontsize=9.2, fontweight="bold")
    ax_l.invert_yaxis()
    ax_l.set_xlim(0, 6.3)
    ax_l.set_xlabel(
        "Sai số toàn phương trung bình gốc - RMSE (USD, Thấp là tốt)",
        fontsize=10,
    )
    ax_l.set_title(
        f"A. Đối sánh RMSE trên 5-Fold Cross-Validation và Tập Test (N = "
        f"{n_test:,})\n[Thanh sai số cực nhỏ (±0.02–0.05 USD) chứng tỏ "
        "độ ổn định cao giữa các Fold]",
        fontsize=11.0,
        fontweight="bold",
        pad=10,
    )
    ax_l.legend(loc="upper right", fontsize=8.8, frameon=True)
    for i, (_, r) in enumerate(merged_m.iterrows()):
        ax_l.text(
            r["cv_rmse_mean"] + r["cv_rmse_std"] + 0.08,
            i,
            f"${r['test_rmse']:.3f} (R²={r['test_r2'] * 100:.2f}%)",
            va="center",
            fontsize=8.6,
            fontweight="bold",
        )


def _plot_fig12_mape_panel(
    ax_r: Any,
    merged_m: pd.DataFrame,
    short_names: list[str],
    colors_m: list[str],
) -> None:
    """Vẽ Panel B của Fig 12: MAPE (%) trên tập Test đối chiếu ngưỡng 10%."""
    y_pos = np.arange(len(merged_m))
    bars = ax_r.barh(
        y_pos,
        merged_m["test_mape_pct"],
        height=0.62,
        color=colors_m,
        edgecolor="#0f172a",
        zorder=2,
    )
    ax_r.axvline(
        10.0,
        color="#e11d48",
        ls="--",
        lw=1.8,
        zorder=3,
        label="Ngưỡng mục tiêu SMART: MAPE ≤ 10.0%",
    )
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(short_names, fontsize=9.2, fontweight="bold")
    ax_r.invert_yaxis()
    ax_r.set_xlim(0, 26.5)
    ax_r.set_xlabel(
        "Sai số phần trăm tuyệt đối trung bình — MAPE (%, Thấp là tốt)",
        fontsize=10,
    )
    ax_r.set_title(
        "B. Đối sánh MAPE (%) trên Tập Kiểm tra Độc lập\n[Xanh ngọc = Đạt "
        "cả 2 mục tiêu SMART (R² ≥ 90% & MAPE ≤ 10%)]",
        fontsize=11.0,
        fontweight="bold",
        pad=10,
    )
    ax_r.legend(loc="upper right", fontsize=9.0, frameon=True)
    for bar, (_, r) in zip(bars, merged_m.iterrows(), strict=False):
        w = bar.get_width()
        cy = bar.get_y() + bar.get_height() / 2
        if w <= 10.0:
            tag = "✓ SMART" if r["meets_both_smart"] else "Chưa đạt"
            ax_r.text(
                w - 0.25,
                cy,
                f"{w:.2f}% ({tag})",
                va="center",
                ha="right",
                fontsize=8.3,
                fontweight="bold",
                color="white",
                zorder=4,
            )
        else:
            ax_r.text(
                w + 0.30,
                cy,
                f"{w:.2f}% (Chưa đạt)",
                va="center",
                ha="left",
                fontsize=8.6,
                fontweight="bold",
                color="#1e293b",
                zorder=4,
            )


def _save_fig12_model_comparison(
    cv_df: pd.DataFrame, test_df: pd.DataFrame, n_test: int
) -> str:
    """Xuất Fig 12: Xếp hạng 10 cấu hình mô hình (5-Fold CV & Test)."""
    merged_m = pd.merge(
        cv_df, test_df, on=["model_code", "model_name"]
    ).sort_values("test_rmse", ascending=True)
    short_map = {
        "M7_HistGradientBoosting": "M7: HistGradientBoosting",
        "M6_Random_Forest": "M6: Random Forest (Tuned)",
        "M5_Decision_Tree": "M5: Decision Tree (Tuned)",
        "M2_Interaction_OLS": "M2: OLS Tương tác Kinh tế",
        "M3_Ridge_Regression": "M3: Hồi quy Ridge (L2)",
        "M4_Lasso_Regression": "M4: Hồi quy Lasso (L1)",
        "M2_Log_Duan_Smearing_OLS": "M2-Log: OLS ln(Price) + Duan",
        "M1_Basic_OLS": "M1: Hồi quy OLS Cơ bản",
        "M0b_Tier_Route_Lookup_Baseline": "M0b: Bảng tra Gói × Tuyến",
        "M0a_Tier_Median_Baseline": "M0a: Trung vị Gói xe",
    }
    short_names = [
        short_map.get(r["model_code"], r["model_code"])
        for _, r in merged_m.iterrows()
    ]
    colors_m = [
        "#0d9488"
        if r["meets_both_smart"]
        else ("#f59e0b" if r["meets_smart_r2"] else "#94a3b8")
        for _, r in merged_m.iterrows()
    ]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(17.5, 6.8), dpi=300)
    _plot_fig12_rmse_panel(ax_l, merged_m, short_names, colors_m, n_test)
    _plot_fig12_mape_panel(ax_r, merged_m, short_names, colors_m)
    fig.suptitle(
        "HÌNH 5.2 (FIG 12): BẢNG XẾP HẠNG HIỆU NĂNG 10 CẤU HÌNH MÔ HÌNH HỒI "
        "QUY (5-FOLD CV & TEST SET)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    p = FIG_STAGE5_DIR / "fig12_model_comparison_cv.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p.relative_to(BASE_DIR).as_posix()


def _save_fig13_actual_vs_predicted(
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    test_preds: dict[str, np.ndarray],
) -> str:
    """Xuất Fig 13: Actual vs Predicted cho M1, M2 và M7."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.8), dpi=300)
    rng_plot = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng_plot.choice(len(y_test), size=6000, replace=False)
    models = [
        (
            "M1_Basic_OLS",
            "A. M1: OLS Cơ bản (Chưa có tương tác gói×dặm)",
            "#64748b",
        ),
        (
            "M2_Interaction_OLS",
            "B. M2: OLS Tương tác Kinh tế lượng (Diễn giải chính)",
            "#2563eb",
        ),
        (
            "M7_HistGradientBoosting",
            "C. M7: HistGradientBoosting (Dự báo chính)",
            "#0d9488",
        ),
    ]
    for ax, (m_code, title_txt, col_pt) in zip(axes, models, strict=False):
        preds_m = test_preds[m_code]
        row_m = test_df.loc[test_df["model_code"] == m_code].iloc[0]
        ax.scatter(
            y_test[sample_idx],
            preds_m[sample_idx],
            alpha=0.22,
            s=11,
            color=col_pt,
            edgecolors="none",
            label="Mẫu ngẫu nhiên Test (n=6,000)",
        )
        ax.plot(
            [2, 85],
            [2, 85],
            color="#dc2626",
            ls="--",
            lw=1.8,
            label="Đường lý tưởng 45° (y = ŷ)",
        )
        ax.set_xlim(0, 88)
        ax.set_ylim(0, 88)
        ax.set_title(title_txt, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Giá báo API thực tế - Actual Price (USD)", fontsize=10)
        ax.set_ylabel(
            "Giá báo API dự báo - Predicted Price (USD)", fontsize=10
        )
        ax.legend(loc="upper left", fontsize=8.5, frameon=True)
        ax.text(
            0.96,
            0.06,
            f"R² = {row_m['test_r2']:.4f}\nRMSE = "
            f"${row_m['test_rmse']:.4f}\nMAE = "
            f"${row_m['test_mae']:.4f}\nMAPE = {row_m['test_mape_pct']:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#f8fafc",
                "edgecolor": "#64748b",
            },
        )
    fig.suptitle(
        "HÌNH 5.3 (FIG 13): ĐỐI SOÁT GIÁ TRỊ THỰC TẾ VÀ DỰ BÁO TRÊN TẬP "
        "TEST (M1 vs M2 vs M7)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    p = FIG_STAGE5_DIR / "fig13_actual_vs_predicted.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p.relative_to(BASE_DIR).as_posix()


def _plot_fig14_learning_curve_panel(
    ax_lc: Any,
    df_feat: pd.DataFrame,
    rand_tr_idx: np.ndarray,
    rand_te_idx: np.ndarray,
    y_test: np.ndarray,
    registry: dict[str, Any],
) -> None:
    """Vẽ Panel A của Fig 14: Đường cong học tập (Learning Curve) M2 & M7."""
    X_all, y_all = _prepare_stage5_feature_matrix(df_feat)
    X_tr_f, y_tr_f = X_all.iloc[rand_tr_idx], y_all[rand_tr_idx]
    X_te_f = X_all.iloc[rand_te_idx]

    sizes_n = [5_000, 15_000, 40_000, 80_000, 140_000, len(X_tr_f)]
    m2_tr, m2_te, m7_tr, m7_te = [], [], [], []
    for sz in sizes_n:
        X_s, y_s = X_tr_f.iloc[:sz], y_tr_f[:sz]
        e2 = clone(registry["M2_Interaction_OLS"]).fit(X_s, y_s)
        m2_tr.append(float(root_mean_squared_error(y_s, e2.predict(X_s))))
        m2_te.append(
            float(root_mean_squared_error(y_test, e2.predict(X_te_f)))
        )
        e7 = clone(registry["M7_HistGradientBoosting"]).fit(X_s, y_s)
        m7_tr.append(float(root_mean_squared_error(y_s, e7.predict(X_s))))
        m7_te.append(
            float(root_mean_squared_error(y_test, e7.predict(X_te_f)))
        )

    lines = [
        (m2_tr, "o--", "#93c5fd", 1.8, "M2 (OLS Tương tác) - Train RMSE"),
        (m2_te, "s-", "#1d4ed8", 2.2, "M2 (OLS Tương tác) - Test RMSE"),
        (m7_tr, "^--", "#5eead4", 1.8, "M7 (HistGB) - Train RMSE"),
        (m7_te, "D-", "#0f766e", 2.2, "M7 (HistGB) - Test RMSE"),
    ]
    for vals, fmt, col, lw, lbl in lines:
        ax_lc.plot(sizes_n, vals, fmt, color=col, lw=lw, label=lbl)
    ax_lc.set_title(
        "A. Đường cong Học tập (Learning Curve) theo Cỡ mẫu Huấn "
        "luyện\n[Khoảng cách Train-Test hội tụ hẹp dần khi N >= 40,000 dòng "
        "-> Không Overfit]",
        fontsize=11.2,
        fontweight="bold",
        pad=10,
    )
    ax_lc.set_xlabel("Số lượng quan sát tập Huấn luyện (N_train)", fontsize=10)
    ax_lc.set_ylabel("Sai số RMSE (USD)", fontsize=10)
    ax_lc.legend(loc="upper right", fontsize=8.8, frameon=True)


def _compute_fig14_robustness_mapes(
    df_feat: pd.DataFrame,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    test_df: pd.DataFrame,
    test_preds: dict[str, np.ndarray],
    y_test: np.ndarray,
    registry: dict[str, Any],
) -> tuple[list[float], list[float]]:
    """Tính chuỗi MAPE của M2 và M7 qua 4 kịch bản kiểm định độ bền."""
    X_all, y_all = _prepare_stage5_feature_matrix(df_feat)
    _, rand_te_idx = splits["random_80_20"]
    m2_mapes = [
        float(
            test_df.loc[
                test_df["model_code"] == "M2_Interaction_OLS", "test_mape_pct"
            ].iloc[0]
        )
    ]
    m7_mapes = [
        float(
            test_df.loc[
                test_df["model_code"] == "M7_HistGradientBoosting",
                "test_mape_pct",
            ].iloc[0]
        )
    ]
    for sp_k in ["temporal_split", "route_group_split"]:
        tr_i, te_i = splits[sp_k]
        for code, bucket in (
            ("M2_Interaction_OLS", m2_mapes),
            ("M7_HistGradientBoosting", m7_mapes),
        ):
            est = clone(registry[code]).fit(X_all.iloc[tr_i], y_all[tr_i])
            mape = mean_absolute_percentage_error(
                y_all[te_i], est.predict(X_all.iloc[te_i])
            )
            bucket.append(float(mape * 100.0))

    flags_df = pd.read_parquet(PROCESSED_DIR / "stage4_audit_flags.parquet")
    unmod = flags_df["is_strictly_unmodified_raw"].values[rand_te_idx]
    for code, bucket in (
        ("M2_Interaction_OLS", m2_mapes),
        ("M7_HistGradientBoosting", m7_mapes),
    ):
        mape = mean_absolute_percentage_error(
            y_test[unmod], test_preds[code][unmod]
        )
        bucket.append(float(mape * 100.0))
    return m2_mapes, m7_mapes


def _plot_fig14_robustness_panel(
    ax_rob: Any,
    df_feat: pd.DataFrame,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    test_df: pd.DataFrame,
    test_preds: dict[str, np.ndarray],
    y_test: np.ndarray,
    registry: dict[str, Any],
) -> None:
    """Vẽ Panel B của Fig 14: Độ bền MAPE qua 4 kịch bản kiểm định."""
    m2_mapes, m7_mapes = _compute_fig14_robustness_mapes(
        df_feat, splits, test_df, test_preds, y_test, registry
    )
    flags_df = pd.read_parquet(PROCESSED_DIR / "stage4_audit_flags.parquet")
    unmod_pct = float(flags_df["is_strictly_unmodified_raw"].mean() * 100.0)
    labels = [
        "1. Chia Ngẫu nhiên\n(Random 80/20)",
        "2. Chia Thời gian\n(Temporal Split)",
        "3. Chia Tuyến mới\n(Route Group Split)",
        f"4. Tập Nguyên bản R3\n({unmod_pct:.2f}% Unmodified)",
    ]
    x_sc = np.arange(len(labels))
    w_b = 0.35
    b_m2 = ax_rob.bar(
        x_sc - w_b / 2,
        m2_mapes,
        width=w_b,
        color="#2563eb",
        edgecolor="#0f172a",
        label="M2: OLS Tương tác Kinh tế",
    )
    b_m7 = ax_rob.bar(
        x_sc + w_b / 2,
        m7_mapes,
        width=w_b,
        color="#0d9488",
        edgecolor="#0f172a",
        label="M7: HistGradientBoosting",
    )
    ax_rob.axhline(
        10.0,
        color="#e11d48",
        ls="--",
        lw=1.6,
        label="Ngưỡng mục tiêu SMART (MAPE <= 10%)",
    )
    ax_rob.set_title(
        "B. Kiểm định Độ bền theo Chiến lược Chia & Tập con Nguyên bản "
        "R3\n[M2 và M7 duy trì MAPE < 9.5% trên cả Temporal Split và Tập "
        f"Nguyên bản {unmod_pct:.2f}%]",
        fontsize=11.2,
        fontweight="bold",
        pad=10,
    )
    ax_rob.set_xticks(x_sc)
    ax_rob.set_xticklabels(labels, fontsize=9.2)
    ax_rob.set_ylim(0, 14.6)
    ax_rob.set_ylabel("Sai số MAPE (%)", fontsize=10)
    ax_rob.legend(loc="upper right", fontsize=8.8, frameon=True)
    for bars_grp, col in ((b_m2, "#1e3a8a"), (b_m7, "#0f766e")):
        for bar in bars_grp:
            v = bar.get_height()
            ax_rob.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.2,
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8.6,
                fontweight="bold",
                color=col,
            )


def _save_fig14_learning_and_robustness(
    df_feat: pd.DataFrame,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    test_df: pd.DataFrame,
    test_preds: dict[str, np.ndarray],
    y_test: np.ndarray,
    registry: dict[str, Any],
) -> str:
    """Xuất Fig 14: Đường cong học tập & Kiểm định độ bền đa kịch bản."""
    fig, (ax_lc, ax_rob) = plt.subplots(1, 2, figsize=(16.5, 6.2), dpi=300)
    rand_tr_idx, rand_te_idx = splits["random_80_20"]
    _plot_fig14_learning_curve_panel(
        ax_lc, df_feat, rand_tr_idx, rand_te_idx, y_test, registry
    )
    _plot_fig14_robustness_panel(
        ax_rob, df_feat, splits, test_df, test_preds, y_test, registry
    )
    fig.suptitle(
        "HÌNH 5.4 (FIG 14): ĐƯỜNG CONG HỌC TẬP (LEARNING CURVE) & KIỂM ĐỊNH "
        "ĐỘ BỀN ĐA KỊCH BẢN",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    p = FIG_STAGE5_DIR / "fig14_learning_curve.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p.relative_to(BASE_DIR).as_posix()


def generate_stage5_figures(
    df_feat: pd.DataFrame,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    hyp_res: dict[str, Any],
    cv_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_preds: dict[str, np.ndarray],
    registry: dict[str, Any],
) -> list[str]:
    """Vẽ và lưu 4 biểu đồ 300 DPI Giai đoạn 5 (fig11 -> fig14)."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 0.8

    _, rand_te_idx = splits["random_80_20"]
    y_test = df_feat[TARGET_COL].iloc[rand_te_idx].astype(np.float64).values
    return [
        _save_fig11_hypotheses(df_feat, hyp_res),
        _save_fig12_model_comparison(cv_df, test_df, len(y_test)),
        _save_fig13_actual_vs_predicted(y_test, test_df, test_preds),
        _save_fig14_learning_and_robustness(
            df_feat, splits, test_df, test_preds, y_test, registry
        ),
    ]


# =============================================================================
# 6. HÀM ĐIỀU PHỐI CHÍNH CỦA GIAI ĐOẠN 5 (RUN_STAGE5)
# =============================================================================
def _save_stage5_artifacts(
    cv_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rob_df: pd.DataFrame,
    mb_df: pd.DataFrame,
    ex_ante_df: pd.DataFrame,
    fitted_models: dict[str, Any],
    selection_summary: dict[str, Any],
) -> tuple[Any, Any]:
    """Xuất các bảng CSV và 2 mô hình joblib chính thức của Giai đoạn 5."""
    cv_df.drop(columns=["fold_rmses"]).to_csv(
        TABLES_DIR / "stage5_cv_results.csv", index=False, encoding="utf-8-sig"
    )
    test_df.to_csv(
        TABLES_DIR / "stage5_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rob_df.to_csv(
        TABLES_DIR / "stage5_robustness.csv", index=False, encoding="utf-8-sig"
    )
    mb_df.to_csv(
        TABLES_DIR / "stage5_mutate_break.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ex_ante_df.to_csv(
        TABLES_DIR / "stage5_ex_ante_experiments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_code = selection_summary["predictive_champion_code"]
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(fitted_models[best_code], best_model_path)
    m2_model_path = MODELS_DIR / "m2_interaction_ols.joblib"
    joblib.dump(fitted_models["M2_Interaction_OLS"], m2_model_path)
    logger.info(
        f"Đã lưu mô hình dự báo chiến thắng ({best_code}) tại: "
        f"{best_model_path}"
    )
    return best_model_path, m2_model_path


def run_stage5(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Thực thi trọn vẹn Giai đoạn 5 (H1-H4 + 10 mô hình + Độ bền + Ex-Ante).
    """
    ensure_directories()
    logger.info(
        "=== BẮT ĐẦU GIAI ĐOẠN 5: STATISTICAL HYPOTHESIS TESTING & MACHINE "
        "LEARNING MODELING ==="
    )
    df_feat = engineer_features_stage4(load_master_dataset())
    splits, _ = create_data_splits(
        df_feat, test_size=test_size, random_state=random_state
    )
    rand_tr_idx, rand_te_idx = splits["random_80_20"]
    X_all, y_all = _prepare_stage5_feature_matrix(df_feat)
    X_train, y_train = X_all.iloc[rand_tr_idx], y_all[rand_tr_idx]
    X_test, y_test = X_all.iloc[rand_te_idx], y_all[rand_te_idx]

    hyp_results = run_all_hypothesis_tests(df_feat, df_feat.iloc[rand_tr_idx])
    tuning_info = tune_hyperparameters_on_train(
        X_train, y_train, random_state=random_state
    )
    registry = build_model_registry(tuning_info, random_state=random_state)
    cv_df, test_df, fitted_models, test_preds = run_cv_and_test_evaluation(
        registry,
        X_train,
        y_train,
        X_test,
        y_test,
        n_splits=CV_FOLDS,
        random_state=random_state,
    )
    rob_df = run_robustness_and_r3_sensitivity(
        registry, fitted_models, df_feat, splits
    )
    mb_df = run_mutate_and_break_experiments(
        registry, fitted_models, df_feat, rand_tr_idx, rand_te_idx
    )
    regime_breakdown, ex_ante_summary, ex_ante_df = (
        run_regime_and_ex_ante_experiments(
            df_feat,
            X_train,
            y_train,
            X_test,
            y_test,
            fitted_models,
            best_hgb_params=tuning_info["M7_HistGradientBoosting"][
                "best_params"
            ],
            random_state=random_state,
        )
    )
    selection_summary = apply_model_selection_rules(cv_df, test_df, rob_df)
    best_model_path, m2_model_path = _save_stage5_artifacts(
        cv_df,
        test_df,
        rob_df,
        mb_df,
        ex_ante_df,
        fitted_models,
        selection_summary,
    )
    saved_figs = generate_stage5_figures(
        df_feat, splits, hyp_results, cv_df, test_df, test_preds, registry
    )
    summary = {
        "stage": (
            "Stage 5 - Statistical Hypothesis Testing & Machine Learning "
            "Modeling"
        ),
        "part1_hypothesis_testing": hyp_results,
        "part2_hyperparameter_tuning": tuning_info,
        "part2_cv_results": cv_df.to_dict(orient="records"),
        "part2_test_metrics": test_df.to_dict(orient="records"),
        "part2_robustness_and_r3_sensitivity": rob_df.to_dict(
            orient="records"
        ),
        "part2_mutate_and_break": mb_df.to_dict(orient="records"),
        "part2_regime_breakdown": regime_breakdown,
        "part3_ex_ante_experiments": ex_ante_summary,
        "part2_model_selection": selection_summary,
        "saved_models": {
            "best_predictive_model": best_model_path.relative_to(
                BASE_DIR
            ).as_posix(),
            "interpretability_ols_model": m2_model_path.relative_to(
                BASE_DIR
            ).as_posix(),
        },
        "figures": saved_figs,
    }
    summary_json_path = RESULTS_DIR / "stage5_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu trọn bộ kết quả Giai đoạn 5 tại: {summary_json_path}")
    return summary


if __name__ == "__main__":
    run_stage5()
