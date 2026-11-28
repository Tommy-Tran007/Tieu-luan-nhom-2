"""
Module: dashboard/config.py
Mục đích: Tập trung toàn bộ đường dẫn tương đối (pathlib), hợp đồng cấu trúc dữ
liệu (Schema Contracts),
hằng số miền giá trị thực nghiệm (Boston Uber/Lyft), bảng màu thương hiệu và
ánh xạ nhãn hiển thị chuẩn
học thuật (đặc biệt tuân thủ CRIT-01, CRIT-02, HIGH-01, HIGH-02, HIGH-03).

Nguyên tắc:
- Tuyệt đối không dùng đường dẫn tuyệt đối gắn theo thư mục cá nhân (Hardcoded
  Paths).
- Không chứa logic tính toán KPI hay mã giao diện Streamlit.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# 1. ĐƯỜNG DẪN TƯƠNG ĐỐI CHUẨN (PATHLIB)
# =============================================================================
DASHBOARD_DIR: Path = Path(__file__).resolve().parent
BASE_DIR: Path = DASHBOARD_DIR.parent
SRC_DIR: Path = BASE_DIR / "src"

DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = BASE_DIR / "reports"
TABLES_DIR: Path = REPORTS_DIR / "tables"
RESULTS_DIR: Path = REPORTS_DIR / "results"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

MODELS_DIR: Path = BASE_DIR / "models"

# Tệp dữ liệu đã xử lý (Data Layer chính)
MASTER_PARQUET_PATH: Path = PROCESSED_DIR / "master_rideshare.parquet"
CAB_CLEAN_PARQUET_PATH: Path = PROCESSED_DIR / "cab_rides_cleaned.parquet"
WEATHER_CLEAN_CSV_PATH: Path = PROCESSED_DIR / "weather_cleaned.csv"
AUDIT_FLAGS_PARQUET_PATH: Path = PROCESSED_DIR / "stage4_audit_flags.parquet"

# Tệp JSON kiểm toán & kết quả tổng hợp
STAGE3_AUDIT_JSON_PATH: Path = PROCESSED_DIR / "stage3_audit_summary.json"
STAGE4_SUMMARY_JSON_PATH: Path = RESULTS_DIR / "stage4_summary.json"
STAGE5_SUMMARY_JSON_PATH: Path = RESULTS_DIR / "stage5_summary.json"
STAGE6_SUMMARY_JSON_PATH: Path = RESULTS_DIR / "stage6_summary.json"
THESIS_V5_JSON_PATH: Path = RESULTS_DIR / "thesis_v5_verification.json"

# Mô hình học máy đã huấn luyện (read-only, không retrain)
MODEL_M7_PATH: Path = MODELS_DIR / "best_model.joblib"
MODEL_M2_PATH: Path = MODELS_DIR / "m2_interaction_ols.joblib"

# Tệp báo cáo HTML tương tác độc lập (xuất ở Giai đoạn 7)
HTML_EXPORT_PATH: Path = REPORTS_DIR / "dashboard_interactive.html"

# Ánh xạ 14 bảng CSV chuẩn trong reports/tables/
TABLE_PATHS: dict[str, Path] = {
    "data_dictionary": TABLES_DIR / "data_dictionary_complete.csv",
    "vif": TABLES_DIR / "stage4_vif.csv",
    "split_summary": TABLES_DIR / "stage4_split_summary.csv",
    "hypothesis_tests": TABLES_DIR / "stage5_hypothesis_tests.csv",
    "ols_coefficients": TABLES_DIR / "stage5_ols_coefficients.csv",
    "cv_results": TABLES_DIR / "stage5_cv_results.csv",
    "test_metrics": TABLES_DIR / "stage5_test_metrics.csv",
    "robustness": TABLES_DIR / "stage5_robustness.csv",
    "mutate_break": TABLES_DIR / "stage5_mutate_break.csv",
    "tier_tariffs": TABLES_DIR / "stage6_tier_tariffs.csv",
    "permutation_importance": TABLES_DIR / "stage6_permutation_importance.csv",
    "segment_errors": TABLES_DIR / "stage6_segment_errors.csv",
    "prediction_interval": TABLES_DIR / "stage6_prediction_interval.csv",
    "surge_cap_simulation": TABLES_DIR / "stage6_surge_cap_simulation.csv",
}

# =============================================================================
# 2. HỢP ĐỒNG CẤU TRÚC DỮ LIỆU (SCHEMA CONTRACTS CHO VALIDATION.PY)
# =============================================================================
GATEKEEPER_MIN_ROWS: int = 100_000
BOSTON_TIMEZONE: str = "America/New_York"

MASTER_REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "cab_type",
    "name",
    "product_id",
    "source",
    "destination",
    "distance",
    "surge_multiplier",
    "is_surge_original",
    "price",
    "time_stamp",
    "is_timestamp_original",
    "datetime",
    "date_hour",
    "location",
    "temp",
    "clouds",
    "pressure",
    "rain",
    "humidity",
    "wind",
    "weather_matched",
    "is_rain_sentinel",
    "hour",
    "day_of_week",
    "day_name",
    "is_weekend",
    "is_rush_hour",
    "time_of_day",
    "is_raining",
    "distance_x_surge",
    "price_per_mile",
)

# Các cột tuyệt đối KHÔNG được phép có giá trị thiếu (NaN)
# Lưu ý CRIT-03 & zero-leakage: surge_multiplier & distance_x_surge giữ NaN
# trên 38,037 dòng Lyft có is_surge_original == 0; distance giữ NaN trên
# 22,262 dòng thô để RouteDistanceImputer nội suy strictly-on-train trong
# Pipeline.
MASTER_NON_NULL_COLUMNS: tuple[str, ...] = (
    "id",
    "cab_type",
    "name",
    "product_id",
    "source",
    "destination",
    "is_surge_original",
    "price",
    "is_timestamp_original",
    "weather_matched",
    "is_rain_sentinel",
    "day_name",
    "time_of_day",
)

# Các cột thời gian phụ thuộc trực tiếp vào is_timestamp_original == 1
# (CRIT-02)
TIMESTAMP_DEPENDENT_COLS: tuple[str, ...] = (
    "time_stamp",
    "datetime",
    "date_hour",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
)

# Các cột khí tượng phụ thuộc trực tiếp vào weather_matched == 1 (CRIT-01)
WEATHER_DEPENDENT_COLS: tuple[str, ...] = (
    "location",
    "temp",
    "clouds",
    "pressure",
    "rain",
    "humidity",
    "wind",
    "is_raining",
)

# Hợp đồng cột bắt buộc cho các bảng CSV đầu vào của Dashboard
TABLE_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "data_dictionary": (
        "column_name",
        "source_stage",
        "unit",
        "dtype",
        "n_missing",
        "role",
        "decision",
    ),
    "vif": ("config", "feature", "vif", "assessment"),
    "split_summary": ("split_strategy", "n_train", "n_test"),
    "hypothesis_tests": (
        "hypothesis_id",
        "research_question",
        "scope",
        "parametric_pvalue",
        "nonparametric_pvalue",
        "effect_size_name",
        "effect_size_value",
        "statistical_decision",
        "practical_significance",
        "economic_conclusion",
    ),
    "ols_coefficients": (
        "cab_type",
        "service_tier",
        "base_fare_b0_usd",
        "per_mile_rate_b1_usd",
    ),
    "cv_results": ("model_code", "model_name", "cv_r2_mean", "cv_rmse_mean"),
    "test_metrics": (
        "model_code",
        "model_name",
        "test_r2",
        "test_rmse",
        "test_mae",
        "test_mape_pct",
        "predict_time_ms",
        "meets_both_smart",
    ),
    "robustness": (
        "scenario",
        "model_code",
        "n_train",
        "n_test",
        "r2",
        "rmse",
        "mape_pct",
    ),
    "mutate_break": (
        "experiment_id",
        "mutation_action",
        "baseline_r2",
        "mutated_r2",
    ),
    "tier_tariffs": (
        "cab_type",
        "service_tier",
        "base_fare_b0_usd",
        "per_mile_rate_b1_usd",
        "base_fare_ci95_low",
        "base_fare_ci95_high",
        "per_mile_ci95_low",
        "per_mile_ci95_high",
        "market_segment",
    ),
    "permutation_importance": (
        "feature_group",
        "domain_family",
        "permuted_columns",
        "m7_delta_rmse_mean_usd",
        "m7_delta_rmse_std_usd",
        "m2_delta_rmse_mean_usd",
        "m7_importance_share_pct",
    ),
    "segment_errors": (
        "dimension",
        "segment_value",
        "n_test",
        "m7_r2",
        "m7_rmse_usd",
        "m7_mae_usd",
        "m7_mape_pct",
        "m2_rmse_usd",
    ),
    "prediction_interval": (
        "cab_type",
        "service_tier",
        "n_test",
        "mean_actual_usd",
        "mean_p10_usd",
        "mean_p50_usd",
        "mean_p90_usd",
        "mpiw_width_usd",
        "picp_coverage_pct",
    ),
    "surge_cap_simulation": (
        "scenario_code",
        "surge_cap",
        "scenario_description",
        "affected_trips_count",
        "mean_rider_saving_per_affected_trip_usd",
        "total_consumer_savings_usd",
        "static_lyft_revenue_usd",
        "static_lyft_rev_change_pct",
        "supply_constrained_rev_change_drop5_pct",
        "supply_constrained_rev_change_drop10_pct",
        "unfulfilled_quotes_drop5",
        "unfulfilled_quotes_drop10",
        "net_lyft_rev_change_eps_050_pct",
        "net_lyft_rev_change_eps_080_pct",
        "m2_ols_mean_saving_affected_usd",
        "m2_ols_total_savings_usd",
    ),
}

# =============================================================================
# 3. HẰNG SỐ MIỀN GIÁ TRỊ THỰC NGHIỆM BOSTON (PHÒNG CHỐNG NGOẠI SUY HIGH-02)
# =============================================================================
EMPIRICAL_DISTANCE_MIN: float = 0.02
EMPIRICAL_DISTANCE_MAX: float = 7.86
ULTRA_SHORT_GPS_THRESHOLD: float = 0.05

EMPIRICAL_PRICE_MIN: float = 2.50
EMPIRICAL_PRICE_MAX: float = 92.00

VALID_SURGE_LEVELS: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
VALID_CAB_TYPES: tuple[str, ...] = ("Uber", "Lyft")
VALID_BOSTON_DISTRICTS: tuple[str, ...] = (
    "Back Bay",
    "Beacon Hill",
    "Boston University",
    "Fenway",
    "Financial District",
    "Haymarket Square",
    "North End",
    "North Station",
    "Northeastern University",
    "South Station",
    "Theatre District",
    "West End",
)
EXPECTED_ROUTE_COUNT: int = 72
MAX_VALID_RAIN_INCH_PER_HOUR: float = 0.85

# 5 cặp cạnh tranh đối xứng (thống nhất sau Final Audit HIGH-02)
SYMMETRIC_PAIRS: tuple[dict[str, str], ...] = (
    {
        "pair_id": "pool_shared",
        "segment_label": "1. Đi ghép Tiết kiệm (Pool / Shared)",
        "short_label": "Đi ghép (Pool/Shared)",
        "uber_tier": "UberPool",
        "lyft_tier": "Shared",
    },
    {
        "pair_id": "standard",
        "segment_label": "2. Phổ thông 4 chỗ (Standard)",
        "short_label": "Phổ thông (Standard)",
        "uber_tier": "UberX",
        "lyft_tier": "Lyft",
    },
    {
        "pair_id": "xl",
        "segment_label": "3. Gia đình 6 chỗ (XL SUV)",
        "short_label": "Gia đình 6 chỗ (XL)",
        "uber_tier": "UberXL",
        "lyft_tier": "Lyft XL",
    },
    {
        "pair_id": "exec_black",
        "segment_label": "4. Hạng sang Doanh nhân (Executive)",
        "short_label": "Doanh nhân (Black)",
        "uber_tier": "Black",
        "lyft_tier": "Lux Black",
    },
    {
        "pair_id": "lux_suv",
        "segment_label": "5. Hạng sang SUV Cỡ lớn (Luxury SUV)",
        "short_label": "SUV Hạng sang",
        "uber_tier": "Black SUV",
        "lyft_tier": "Lux Black XL",
    },
)

STANDALONE_TIERS: tuple[dict[str, str], ...] = (
    {
        "cab_type": "Uber",
        "tier": "WAV",
        "segment_label": "Hỗ trợ Xe lăn (Accessible — riêng Uber)",
    },
    {
        "cab_type": "Lyft",
        "tier": "Lux",
        "segment_label": "Cao cấp Sedan (Premium — riêng Lyft)",
    },
)

ALL_TIERS_ORDERED: tuple[tuple[str, str], ...] = (
    ("Lyft", "Shared"),
    ("Uber", "UberPool"),
    ("Lyft", "Lyft"),
    ("Uber", "UberX"),
    ("Uber", "WAV"),
    ("Lyft", "Lyft XL"),
    ("Uber", "UberXL"),
    ("Lyft", "Lux"),
    ("Uber", "Black"),
    ("Lyft", "Lux Black"),
    ("Uber", "Black SUV"),
    ("Lyft", "Lux Black XL"),
)

DAY_NAME_VI: dict[int, str] = {
    0: "Thứ 2 (Mon)",
    1: "Thứ 3 (Tue)",
    2: "Thứ 4 (Wed)",
    3: "Thứ 5 (Thu)",
    4: "Thứ 6 (Fri)",
    5: "Thứ 7 (Sat)",
    6: "Chủ nhật (Sun)",
}

TIME_OF_DAY_ORDER: tuple[str, ...] = (
    "Morning",
    "Afternoon",
    "Evening",
    "Night",
)
TIME_OF_DAY_VI: dict[str, str] = {
    "Morning": "Sáng (05:00–11:59 EST)",
    "Afternoon": "Chiều (12:00–16:59 EST)",
    "Evening": "Tối (17:00–21:59 EST)",
    "Night": "Đêm (22:00–04:59 EST)",
}

# =============================================================================
# 4. BẢNG MÀU THƯƠNG HIỆU & NHÃN CHUẨN HỌC THUẬT (HIGH-03 ENFORCEMENT)
# =============================================================================
BRAND_COLORS: dict[str, str] = {
    "Uber": "#111827",  # Obsidian Slate
    "Lyft": "#E11D8A",  # Lyft Signature Magenta
    "accent": "#2563EB",  # Royal Cobalt
    "success": "#059669",  # Emerald
    "warning": "#D97706",  # Amber
    "danger": "#DC2626",  # Crimson
    "muted": "#64748B",  # Cool Slate
    "bg_page": "#F8FAFC",  # Crisp Light Slate
    "bg_card": "#FFFFFF",  # Pure White Card
    "border": "#E2E8F0",  # Subtle Border
}

# Ánh xạ bắt buộc đổi tên cột CSV mô phỏng sang đúng thuật ngữ API Quote Proxy
# (HIGH-03)
SIMULATION_COLUMN_LABELS_VI: dict[str, str] = {
    "scenario_code": "Mã kịch bản",
    "surge_cap": "Trần Surge (x)",
    "scenario_description": "Mô tả kịch bản mô phỏng",
    "affected_trips_count": "Số lượt báo giá chịu tác động",
    "affected_lyft_trips_pct": "Tỷ lệ trên báo giá Lyft (%)",
    "affected_surge_trips_pct": "Tỷ lệ trên báo giá Surge (%)",
    "mean_orig_price_affected_usd": "Giá báo gốc TB (USD)",
    "mean_capped_price_affected_usd": "Giá báo sau trần TB (USD)",
    "mean_rider_saving_per_affected_trip_usd": "Tiết kiệm TB/lượt M7 (USD)",
    "mean_rider_saving_pct_affected": "Giảm giá TB/lượt M7 (%)",
    "total_consumer_savings_usd": "Tổng tiết kiệm báo giá M7 (USD)",
    "static_lyft_revenue_usd": (
        "Tổng giá trị báo giá Lyft tĩnh — Quote Proxy (USD)"
    ),
    "static_lyft_rev_change_pct": "Quote Proxy tĩnh ε=0.0 (%)",
    "supply_constrained_rev_change_drop5_pct": "Ràng buộc cung ε_S=0.30 (%)",
    "supply_constrained_rev_change_drop10_pct": "Ràng buộc cung ε_S=0.60 (%)",
    "unfulfilled_quotes_drop5": "Dư cầu chưa phục vụ (ε_S=0.30)",
    "unfulfilled_quotes_drop10": "Dư cầu chưa phục vụ (ε_S=0.60)",
    "static_market_rev_change_pct": "Quote Proxy toàn mẫu (%)",
    "retained_trips_eps_050": "Cận trên giữ chân (ε=-0.50)",
    "net_lyft_rev_change_eps_050_pct": "Cận trên dư cung ε=-0.50 (%)",
    "retained_trips_eps_080": "Cận trên giữ chân (ε=-0.80)",
    "net_lyft_rev_change_eps_080_pct": "Cận trên dư cung ε=-0.80 (%)",
    "m2_ols_mean_saving_affected_usd": "Đối chứng tiết kiệm TB M2 OLS (USD)",
    "m2_ols_total_savings_usd": "Đối chứng tổng tiết kiệm M2 OLS (USD)",
    "m2_ols_static_lyft_rev_change_pct": (
        "Đối chứng biến động Quote Proxy M2 OLS (%)"
    ),
}
