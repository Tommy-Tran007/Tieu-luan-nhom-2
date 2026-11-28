"""
Module: stage3_pipeline_and_audit.py
Mục đích: Thực thi toàn diện Giai đoạn 3 (Data Preparation, Exploratory Data
Analysis,
Data Visualization & Business Questions Resolution) trên bộ dữ liệu gốc.
Ghi nhận 100% số liệu thực nghiệm trước và sau xử lý vào tệp JSON phục vụ Báo
cáo Kỹ thuật.
"""

import json
import logging
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from clean_raw_data import (
    SENTINEL_STRINGS,
    clean_cab_rides,
    clean_weather,
    load_cab_rides_chunked,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "reports" / "figures" / "stage3"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_numeric_probe(series: pd.Series):
    """
    Hàm hỗ trợ kiểm toán sâu một cột số đang ở dạng chuỗi (không sửa dữ liệu
    gốc).
    """
    non_null = series.dropna().astype(str)
    stripped = non_null.str.strip()
    lower = stripped.str.lower()

    sentinel_str_mask = lower.isin(SENTINEL_STRINGS) | lower.str.startswith(
        "nan "
    )
    sentinel_str_count = int(sentinel_str_mask.sum())

    unit_mask = (~sentinel_str_mask) & stripped.str.contains(
        r"(?:USD|VND|đ|kg|%)", case=False, regex=True
    )
    unit_count = int(unit_mask.sum())

    comma_mask = (~sentinel_str_mask) & stripped.str.contains(",", regex=False)
    comma_count = int(comma_mask.sum())

    cleaned_str = (
        stripped[~sentinel_str_mask]
        .str.replace(
            r"\s*(USD|VND|đ|kg|%)\s*$", "", flags=re.IGNORECASE, regex=True
        )
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    nums = pd.to_numeric(cleaned_str, errors="coerce")

    return {
        "raw_null_count": int(series.isna().sum()),
        "sentinel_str_count": sentinel_str_count,
        "unit_suffix_count": unit_count,
        "comma_decimal_count": comma_count,
        "parsed_valid_numeric_count": int(nums.notna().sum()),
        "parsed_min": float(nums.min()) if nums.notna().any() else None,
        "parsed_max": float(nums.max()) if nums.notna().any() else None,
        "nums": nums,
    }


def _probe_column_anomalies(
    series: pd.Series,
    sentinels: list[float],
    neg_exclude: list[float],
    scale_thresh: float,
    scale_exclude: list[float],
    *,
    strict_ge: bool = False,
) -> dict:
    """Chạy probe số học và đếm sentinel/âm/phóng đại theo ngưỡng cột."""
    probe = parse_numeric_probe(series)
    nums = probe.pop("nums")
    scale_cond = (
        nums.abs() >= scale_thresh
        if strict_ge
        else (nums.abs() > scale_thresh)
    )
    probe.update(
        {
            "numeric_sentinel_count": int(nums.isin(sentinels).sum()),
            "negative_sign_inverted_count": int(
                ((nums < 0) & (~nums.isin(neg_exclude))).sum()
            ),
            "scale_inflated_count": int(
                (scale_cond & (~nums.isin(scale_exclude))).sum()
            ),
        }
    )
    return probe


def _audit_cab_raw(cab_raw: pd.DataFrame) -> dict:
    """Kiểm toán chất lượng dữ liệu thô bảng cab_rides."""
    info: dict = {
        "shape": {
            "rows": int(cab_raw.shape[0]),
            "cols": int(cab_raw.shape[1]),
        },
        "memory_mb": round(
            float(cab_raw.memory_usage(deep=True).sum() / (1024**2)), 2
        ),
        "dtypes": cab_raw.dtypes.astype(str).to_dict(),
        "raw_nulls": {k: int(v) for k, v in cab_raw.isna().sum().items()},
        "full_row_duplicates": int(cab_raw.duplicated().sum()),
        "id_duplicates_raw": int(cab_raw["id"].duplicated().sum()),
        "id_duplicates_stripped": int(
            cab_raw["id"].astype(str).str.strip().duplicated().sum()
        ),
    }
    cat_cols = [
        "id",
        "cab_type",
        "name",
        "product_id",
        "source",
        "destination",
    ]
    for col in cat_cols:
        s = cab_raw[col].dropna().astype(str)
        lower = s.str.strip().str.lower()
        info[f"cat_{col}"] = {
            "raw_unique": int(cab_raw[col].nunique(dropna=True)),
            "stripped_title_unique": int(
                s.str.strip()
                .str.title()
                .loc[~lower.isin(SENTINEL_STRINGS)]
                .nunique()
            ),
            "whitespace_padded_count": int((s != s.str.strip()).sum()),
            "sentinel_string_count": int(lower.isin(SENTINEL_STRINGS).sum()),
        }
    info["truncated_northeastern"] = {
        c: int(
            cab_raw[c]
            .dropna()
            .astype(str)
            .str.strip()
            .str.title()
            .eq("Northeastern Universi")
            .sum()
        )
        for c in ("source", "destination")
    }
    ts_probe = parse_numeric_probe(cab_raw["time_stamp"])
    ts_probe.pop("nums")
    info["numeric_probes"] = {
        "distance": _probe_column_anomalies(
            cab_raw["distance"],
            [-1, -1.0, 9999, 10000, 100000],
            [-1, -1.0],
            8.0,
            [9999, 10000, 100000],
        ),
        "price": _probe_column_anomalies(
            cab_raw["price"],
            [-1, -1.0, -100, 9999, 10000, 100000],
            [-1, -1.0, -100],
            95.0,
            [-100, 9999, 10000, 100000],
        ),
        "surge_multiplier": _probe_column_anomalies(
            cab_raw["surge_multiplier"],
            [-1, -1.0, 9999, 10000, 100000],
            [-1, -1.0],
            10.0,
            [9999, 10000, 100000],
            strict_ge=True,
        ),
        "time_stamp": ts_probe,
    }
    return info


def _audit_weather_raw(weather_raw: pd.DataFrame) -> dict:
    """Kiểm toán chất lượng dữ liệu thô bảng weather."""
    loc_s = weather_raw["location"].dropna().astype(str)
    loc_lower = loc_s.str.strip().str.lower()
    info: dict = {
        "shape": {
            "rows": int(weather_raw.shape[0]),
            "cols": int(weather_raw.shape[1]),
        },
        "memory_mb": round(
            float(weather_raw.memory_usage(deep=True).sum() / (1024**2)), 2
        ),
        "dtypes": weather_raw.dtypes.astype(str).to_dict(),
        "raw_nulls": {k: int(v) for k, v in weather_raw.isna().sum().items()},
        "full_row_duplicates": int(weather_raw.duplicated().sum()),
        "cat_location": {
            "raw_unique": int(weather_raw["location"].nunique(dropna=True)),
            "whitespace_padded_count": int((loc_s != loc_s.str.strip()).sum()),
            "sentinel_string_count": int(
                loc_lower.isin(SENTINEL_STRINGS).sum()
            ),
            "truncated_northeastern": int(
                loc_s.str.strip().str.title().eq("Northeastern Universi").sum()
            ),
        },
    }
    specs = {
        "temp": ([-1, 9999], [-1], 100.0, [9999]),
        "clouds": ([-1, 9999], [-1], 1.0, [9999]),
        "pressure": ([-1, 9999], [-1], 5000.0, [9999]),
        "rain": (
            [-1, -100, 1000, 9999, 10000, 100000],
            [-1, -100],
            1.0,
            [-100, 1000, 9999, 10000, 100000],
        ),
        "humidity": ([-1, 9999], [-1], 1.0, [9999]),
        "wind": ([-1, 9999], [-1], 25.0, [9999]),
    }
    probes = {
        col: _probe_column_anomalies(weather_raw[col], *args)
        for col, args in specs.items()
    }
    ts_probe = parse_numeric_probe(weather_raw["time_stamp"])
    ts_probe.pop("nums")
    ts_probe.update(
        {
            "numeric_sentinel_count": 0,
            "negative_sign_inverted_count": 0,
            "scale_inflated_count": 0,
        }
    )
    probes["time_stamp"] = ts_probe
    info["numeric_probes"] = probes
    return info


def step1_initial_audit(
    cab_raw: pd.DataFrame, weather_raw: pd.DataFrame
) -> dict:
    """Thực hiện kiểm toán ban đầu cho cab_rides, weather và độ khớp chéo."""
    logger.info("=== BƯỚC 1.1: KIỂM TRA DỮ LIỆU BAN ĐẦU (INITIAL AUDIT) ===")
    raw_cab_locs = set(cab_raw["source"].dropna().astype(str).unique())
    raw_wea_locs = set(weather_raw["location"].dropna().astype(str).unique())
    return {
        "cab_rides": _audit_cab_raw(cab_raw),
        "weather": _audit_weather_raw(weather_raw),
        "cross_table": {
            "raw_source_unique_count": len(raw_cab_locs),
            "raw_weather_loc_unique_count": len(raw_wea_locs),
            "raw_exact_overlap_count": len(
                raw_cab_locs.intersection(raw_wea_locs)
            ),
        },
    }


def step2_clean_datasets(cab_raw: pd.DataFrame, weather_raw: pd.DataFrame):
    """
    [M4 FIX] Ủy quyền trực tiếp quy trình làm sạch cho clean_raw_data.py
    (Single Source of Truth).
    """
    logger.info(
        "=== BƯỚC 1.2 & 1.3: LÀM SẠCH CAB_RIDES VÀ WEATHER (GỌI "
        "CLEAN_RAW_DATA.PY) ==="
    )
    df_clean, cab_log = clean_cab_rides(cab_raw, return_log=True)
    wdf, wea_log = clean_weather(weather_raw, return_log=True)
    cleaning_log = {"cab_rides": cab_log, "weather": wea_log}

    df_clean.to_parquet(
        PROCESSED_DIR / "cab_rides_cleaned.parquet", index=False
    )
    wdf.to_csv(PROCESSED_DIR / "weather_cleaned.csv", index=False)

    return df_clean, wdf, cleaning_log


def _merge_cab_and_weather(
    cab_clean: pd.DataFrame, weather_clean: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """
    Chuyển múi giờ Boston EST, gom weather theo (location, date_hour) và ghép.
    """
    cab_df = cab_clean.copy()
    wea_df = weather_clean.copy()

    cab_df["datetime"] = (
        pd.to_datetime(
            cab_df["time_stamp"].astype("float64"), unit="ms", utc=True
        )
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    cab_df["date_hour"] = cab_df["datetime"].dt.floor("h")

    wea_df["datetime"] = (
        pd.to_datetime(
            wea_df["time_stamp"].astype("float64"), unit="s", utc=True
        )
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    wea_df["date_hour"] = wea_df["datetime"].dt.floor("h")

    dup_loc_hour = int(
        wea_df.duplicated(subset=["location", "date_hour"]).sum()
    )
    unique_loc_hour = int(
        wea_df[["location", "date_hour"]].drop_duplicates().shape[0]
    )
    weather_hourly = wea_df.groupby(
        ["location", "date_hour"], as_index=False
    ).agg(
        {
            "temp": "mean",
            "clouds": "mean",
            "pressure": "mean",
            "rain": "mean",
            "humidity": "mean",
            "wind": "mean",
        }
    )

    rows_before_join = len(cab_df)
    master_df = pd.merge(
        cab_df,
        weather_hourly,
        left_on=["source", "date_hour"],
        right_on=["location", "date_hour"],
        how="left",
    )
    merge_meta = {
        "weather_rows_cleaned": int(len(wea_df)),
        "dup_loc_hour": dup_loc_hour,
        "unique_loc_hour": unique_loc_hour,
        "rows_before_join": rows_before_join,
    }
    return master_df, merge_meta


def _enrich_temporal_and_weather_flags(master_df: pd.DataFrame) -> None:
    """Tạo cờ weather_matched, đặc trưng thời gian Boston EST và phái sinh."""
    master_df["weather_matched"] = (
        (master_df["is_timestamp_original"] == 1)
        & master_df["location"].notna()
    ).astype(np.int8)

    ts_valid = master_df["is_timestamp_original"] == 1
    dt_col = master_df["datetime"].dt
    master_df["hour"] = np.where(ts_valid, dt_col.hour, np.nan).astype(
        np.float32
    )
    master_df["day_of_week"] = np.where(
        ts_valid, dt_col.day_of_week, np.nan
    ).astype(np.float32)
    master_df["day_name"] = np.where(ts_valid, dt_col.day_name(), "Unknown")
    master_df["is_weekend"] = np.where(
        ts_valid, (dt_col.day_of_week >= 5).astype(np.float32), np.nan
    ).astype(np.float32)
    is_rush = (dt_col.day_of_week < 5) & (
        ((dt_col.hour >= 7) & (dt_col.hour <= 9))
        | ((dt_col.hour >= 16) & (dt_col.hour <= 19))
    )
    master_df["is_rush_hour"] = np.where(
        ts_valid, is_rush.astype(np.float32), np.nan
    ).astype(np.float32)

    conditions = [
        ~ts_valid,
        (master_df["hour"] >= 5) & (master_df["hour"] < 12),
        (master_df["hour"] >= 12) & (master_df["hour"] < 17),
        (master_df["hour"] >= 17) & (master_df["hour"] < 22),
    ]
    master_df["time_of_day"] = np.select(
        conditions,
        ["Unknown", "Morning", "Afternoon", "Evening"],
        default="Night",
    )
    master_df["is_rain_sentinel"] = np.where(
        (master_df["weather_matched"] == 1) & master_df["rain"].isna(), 1, 0
    ).astype(np.int8)
    master_df["is_raining"] = np.where(
        (master_df["weather_matched"] == 1) & master_df["rain"].notna(),
        (master_df["rain"] > 0.0).astype(np.float32),
        np.nan,
    ).astype(np.float32)
    master_df["distance_x_surge"] = (
        master_df["distance"] * master_df["surge_multiplier"]
    )
    master_df["price_per_mile"] = np.where(
        master_df["distance"] > 0.05,
        master_df["price"] / master_df["distance"],
        np.nan,
    )


def _downcast_master_dataframe(
    master_df: pd.DataFrame,
) -> tuple[float, float]:
    """Tối ưu bộ nhớ bằng category + float32, trả về (MB trước, MB sau)."""
    mem_before = float(master_df.memory_usage(deep=True).sum() / (1024**2))
    cat_cols = [
        "cab_type",
        "name",
        "source",
        "destination",
        "product_id",
        "location",
        "day_name",
        "time_of_day",
    ]
    num_cols = [
        "distance",
        "price",
        "surge_multiplier",
        "temp",
        "clouds",
        "pressure",
        "rain",
        "humidity",
        "wind",
        "distance_x_surge",
        "price_per_mile",
    ]
    for col in cat_cols:
        master_df[col] = master_df[col].astype("category")
    for col in num_cols:
        master_df[col] = master_df[col].astype(np.float32)
    mem_after = float(master_df.memory_usage(deep=True).sum() / (1024**2))
    return mem_before, mem_after


def _compute_merge_audit_and_gaps(
    master_df: pd.DataFrame,
    merge_meta: dict,
    mem_before: float,
    mem_after: float,
) -> tuple[dict, dict]:
    """Tổng hợp log đối soát phép ghép nối và các khoảng trống dữ liệu."""
    rows_after = len(master_df)
    matched_rows = int(master_df["weather_matched"].sum())
    unmatched_rows = int(rows_after - matched_rows)
    ts_valid = master_df["is_timestamp_original"] == 1
    valid_ts_rows = int(ts_valid.sum())
    missing_ts_rows = int((~ts_valid).sum())
    unmatched_valid_ts = int(
        (ts_valid & (master_df["weather_matched"] == 0)).sum()
    )

    valid_dates = sorted(
        master_df.loc[ts_valid, "datetime"].dt.date.unique().tolist()
    )
    full_dates = pd.date_range(
        min(valid_dates), max(valid_dates), freq="D"
    ).date.tolist()
    missing_dates = [str(d) for d in full_dates if d not in valid_dates]

    merge_audit = {
        "timezone": "America/New_York (EST = UTC-5)",
        "weather_rows_cleaned": merge_meta["weather_rows_cleaned"],
        "weather_duplicate_loc_hour_before_agg": merge_meta["dup_loc_hour"],
        "weather_unique_loc_hour_after_agg": merge_meta["unique_loc_hour"],
        "cab_rows_before_join": merge_meta["rows_before_join"],
        "master_rows_after_join": rows_after,
        "fan_out_extra_rows": rows_after - merge_meta["rows_before_join"],
        "valid_original_timestamp_rows": valid_ts_rows,
        "valid_original_timestamp_pct": round(
            valid_ts_rows / rows_after * 100, 2
        ),
        "missing_timestamp_rows": missing_ts_rows,
        "missing_timestamp_pct": round(missing_ts_rows / rows_after * 100, 2),
        "direct_matched_weather_rows": matched_rows,
        "direct_matched_pct": round(matched_rows / rows_after * 100, 2),
        "direct_matched_among_valid_ts_pct": round(
            matched_rows / valid_ts_rows * 100, 2
        ),
        "unmatched_weather_among_valid_ts_rows": unmatched_valid_ts,
        "unmatched_weather_total_rows": unmatched_rows,
        "fallback_imputed_weather_rows": 0,
        "is_raining_on_unmatched_count": int(
            master_df.loc[master_df["weather_matched"] == 0, "is_raining"]
            .notna()
            .sum()
        ),
        "memory_mb_before_downcast": round(mem_before, 2),
        "memory_mb_after_downcast": round(mem_after, 2),
        "memory_saved_pct": round((1 - mem_after / mem_before) * 100, 2),
    }
    anomalies = {
        "short_gps_trips_le_0_05_miles": int(
            (master_df["distance"] <= 0.05).sum()
        ),
        "observed_local_dates_count": len(valid_dates),
        "date_range_est": [str(min(valid_dates)), str(max(valid_dates))],
        "missing_collection_dates_est": missing_dates,
        "zero_count_dow_hour_cells_est": [
            "Tuesday 21h-23h EST",
            "Wednesday 00h-01h EST",
        ],
    }
    return merge_audit, anomalies


def _compute_descriptive_stats(master_df: pd.DataFrame) -> tuple[dict, dict]:
    """Tính thống kê mô tả cho các cột số và cột phân loại."""
    num_cols = [
        "price",
        "distance",
        "surge_multiplier",
        "temp",
        "clouds",
        "pressure",
        "rain",
        "humidity",
        "wind",
    ]
    num_stats = {}
    for col in num_cols:
        s = master_df[col].dropna().astype(np.float64)
        q1, q2, q3 = s.quantile([0.25, 0.50, 0.75])
        num_stats[col] = {
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "median": round(float(q2), 4),
            "std": round(float(s.std()), 4),
            "variance": round(float(s.var()), 4),
            "min": round(float(s.min()), 4),
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(q3 - q1), 4),
            "max": round(float(s.max()), 4),
            "skewness": round(float(stats.skew(s)), 4),
            "kurtosis": round(float(stats.kurtosis(s)), 4),
        }

    cat_cols = ["cab_type", "name", "source", "destination", "time_of_day"]
    cat_stats = {}
    for col in cat_cols:
        vc = master_df[col].value_counts(dropna=False)
        pct = (vc / len(master_df) * 100).round(2)
        cat_stats[col] = {
            "unique_count": int(master_df[col].nunique()),
            "mode": str(vc.index[0]),
            "mode_freq": int(vc.iloc[0]),
            "mode_pct": float(pct.iloc[0]),
            "top_distribution": {
                str(k): {"freq": int(vc[k]), "pct": float(pct[k])}
                for k in vc.index
            },
        }
    return num_stats, cat_stats


def _compute_correlation_and_tier_stats(
    master_df: pd.DataFrame,
) -> tuple[dict, dict, dict]:
    """Tính ma trận tương quan, bảng chéo Surge và tóm tắt 12 phân khúc."""
    corr_cols = [
        "price",
        "distance",
        "surge_multiplier",
        "distance_x_surge",
        "temp",
        "rain",
        "humidity",
        "wind",
        "clouds",
        "pressure",
    ]
    pearson_corr = (
        master_df[corr_cols].corr(method="pearson").round(4).to_dict()
    )
    spearman_corr = (
        master_df[corr_cols].corr(method="spearman").round(4).to_dict()
    )
    correlations = {
        "pearson_with_price": pearson_corr["price"],
        "spearman_with_price": spearman_corr["price"],
        "pearson_with_surge": pearson_corr["surge_multiplier"],
    }

    obs_surge = master_df[master_df["is_surge_original"] == 1].copy()
    obs_surge["has_surge"] = obs_surge["surge_multiplier"] > 1.0
    ct = pd.crosstab(obs_surge["cab_type"], obs_surge["has_surge"])
    crosstab_surge = {
        "Uber": {
            "no_surge": int(ct.loc["Uber", False]),
            "has_surge": int(ct.loc["Uber", True])
            if True in ct.columns
            else 0,
        },
        "Lyft": {
            "no_surge": int(ct.loc["Lyft", False]),
            "has_surge": int(ct.loc["Lyft", True])
            if True in ct.columns
            else 0,
            "missing_surge_kept_nan": int(
                (
                    (master_df["cab_type"] == "Lyft")
                    & (master_df["is_surge_original"] == 0)
                ).sum()
            ),
        },
    }

    tier_grp = (
        master_df.groupby("name", observed=True)
        .agg(
            count=("price", "count"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            std_price=("price", "std"),
            mean_distance=("distance", "mean"),
            mean_surge=("surge_multiplier", "mean"),
        )
        .sort_values("mean_price", ascending=True)
    )
    tier_summary = {
        str(idx): {
            "count": int(row["count"]),
            "mean_price": round(float(row["mean_price"]), 2),
            "median_price": round(float(row["median_price"]), 2),
            "std_price": round(float(row["std_price"]), 2),
            "mean_distance": round(float(row["mean_distance"]), 2),
            "mean_surge": round(float(row["mean_surge"]), 3),
        }
        for idx, row in tier_grp.iterrows()
    }
    return correlations, crosstab_surge, tier_summary


def step3_merge_and_eda(
    cab_clean: pd.DataFrame, weather_clean: pd.DataFrame
) -> tuple:
    """
    Ghép nối đa bảng theo giờ Boston EST, tối ưu kiểu dữ liệu và chạy EDA.
    """
    logger.info(
        "=== BƯỚC 2.1 - 2.4: GHÉP NỐI ĐA BẢNG & PHÂN TÍCH KHÁM PHÁ (EDA) ==="
    )
    master_df, merge_meta = _merge_cab_and_weather(cab_clean, weather_clean)
    _enrich_temporal_and_weather_flags(master_df)
    mem_before, mem_after = _downcast_master_dataframe(master_df)
    master_df.to_parquet(
        PROCESSED_DIR / "master_rideshare.parquet", index=False
    )

    merge_audit, anomalies = _compute_merge_audit_and_gaps(
        master_df, merge_meta, mem_before, mem_after
    )
    num_stats, cat_stats = _compute_descriptive_stats(master_df)
    correlations, crosstab_surge, tier_summary = (
        _compute_correlation_and_tier_stats(master_df)
    )
    eda_res = {
        "merge_audit": merge_audit,
        "data_anomalies_and_gaps": anomalies,
        "numerical_stats": num_stats,
        "categorical_stats": cat_stats,
        "correlations": correlations,
        "crosstab_cab_surge": crosstab_surge,
        "service_tier_summary": tier_summary,
    }
    return master_df, eda_res


def _plot_fig01_market_overview(master_df: pd.DataFrame) -> None:
    """Xuất Hình 1 (Nhóm A - Tổng quan quy mô báo giá)."""
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.8]}
    )
    cab_counts = master_df["cab_type"].value_counts()
    bars0 = axes[0].bar(
        cab_counts.index.astype(str),
        cab_counts.values,
        color=["#1e3a8a", "#0f766e"],
        width=0.55,
        edgecolor="black",
    )
    axes[0].set_title("A. Quy mô Lượt báo giá theo Nền tảng (Uber vs. Lyft)")
    axes[0].set_xlabel("Thương hiệu nền tảng (cab_type)")
    axes[0].set_ylabel("Số lượng lượt báo giá (Quotes)")
    axes[0].set_ylim(0, cab_counts.max() * 1.18)
    for b in bars0:
        h = b.get_height()
        pct = h / len(master_df) * 100
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            h + 4000,
            f"{h:,.0f}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    tier_totals = master_df["name"].value_counts()
    tier_totals = tier_totals[tier_totals > 0].sort_values(ascending=True)
    tier_df = (
        master_df.groupby("name", observed=True)
        .agg(cnt=("id", "count"), cab=("cab_type", "first"))
        .loc[tier_totals.index]
    )
    bar_colors = [
        "#1e3a8a" if c == "Uber" else "#0f766e" for c in tier_df["cab"]
    ]
    bars1 = axes[1].barh(
        tier_df.index.astype(str),
        tier_df["cnt"],
        color=bar_colors,
        edgecolor="black",
        height=0.72,
    )
    axes[1].set_title(
        "B. Phân bố Số lượng Lượt báo giá theo 12 Phân khúc Dịch vụ"
    )
    axes[1].set_xlabel("Số lượng lượt báo giá (Quotes)")
    axes[1].set_ylabel("Tên gói dịch vụ (name)")
    axes[1].set_xlim(0, tier_df["cnt"].max() * 1.18)
    for b in bars1:
        w = b.get_width()
        axes[1].text(
            w + 600,
            b.get_y() + b.get_height() / 2,
            f"{w:,.0f}",
            va="center",
            fontsize=9,
        )
    plt.suptitle(
        "NHÓM A (TỔNG QUAN): QUY MÔ BÁO GIÁ API THEO THƯƠNG HIỆU VÀ PHÂN "
        f"KHÚC DỊCH VỤ (N = {len(master_df):,})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "fig01_groupA_market_overview.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_fig02_distributions(master_df: pd.DataFrame) -> None:
    """Xuất Hình 2 (Nhóm B - Phân phối & Boxplot của price và distance)."""
    fig, axes = plt.subplots(
        2, 2, figsize=(15, 10), gridspec_kw={"height_ratios": [2.2, 1]}
    )
    specs = [
        (
            0,
            "price",
            "#1e3a8a",
            "#93c5fd",
            f"Mean = ${master_df['price'].mean():.2f}",
            f"Median = ${master_df['price'].median():.2f}",
            "A. Phân phối Mật độ Giá báo API (price)",
            "C. Biểu đồ Hộp (Boxplot) Kiểm tra Độ lệch & Ngoại lai Giá báo",
            "Giá báo API (USD)",
        ),
        (
            1,
            "distance",
            "#0f766e",
            "#5eead4",
            f"Mean = {master_df['distance'].mean():.2f} dặm",
            f"Median = {master_df['distance'].median():.2f} dặm",
            "B. Phân phối Mật độ Quãng đường Di chuyển (distance)",
            "D. Boxplot Kiểm tra Độ lệch & Ngoại lai Quãng đường",
            "Quãng đường hành trình (Dặm - Miles)",
        ),
    ]
    for c_idx, col, h_col, b_col, m_lbl, med_lbl, t_top, t_bot, x_lbl in specs:
        sns.histplot(
            master_df[col],
            bins=50,
            kde=True,
            ax=axes[0, c_idx],
            color=h_col,
            stat="density",
        )
        axes[0, c_idx].axvline(
            master_df[col].mean(), color="#dc2626", ls="--", lw=2, label=m_lbl
        )
        axes[0, c_idx].axvline(
            master_df[col].median(),
            color="#f59e0b",
            ls="-",
            lw=2,
            label=med_lbl,
        )
        axes[0, c_idx].set_title(t_top)
        axes[0, c_idx].set_xlabel(x_lbl)
        axes[0, c_idx].set_ylabel("Mật độ xác suất (Density)")
        axes[0, c_idx].legend()

        sns.boxplot(
            x=master_df[col], ax=axes[1, c_idx], color=b_col, fliersize=2
        )
        axes[1, c_idx].set_title(t_bot)
        axes[1, c_idx].set_xlabel(x_lbl)

    plt.suptitle(
        "NHÓM B (PHÂN PHỐI): HÌNH DÁNG PHÂN PHỐI, ĐỘ LỆCH PHẢI VÀ NGOẠI LAI "
        "CỦA GIÁ BÁO & QUÃNG ĐƯỜNG",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "fig02_groupB_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_fig03_temporal_trends(master_df: pd.DataFrame) -> None:
    """Xuất Hình 3 (Nhóm C - Xu hướng lưu lượng & giá theo 24h Boston EST)."""
    ts_valid = master_df[master_df["is_timestamp_original"] == 1].copy()
    ts_valid["hour_int"] = ts_valid["hour"].astype(int)
    hourly = (
        ts_valid.groupby(["hour_int", "cab_type"], observed=True)
        .agg(ride_count=("id", "count"), mean_price=("price", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    sns.lineplot(
        data=hourly,
        x="hour_int",
        y="ride_count",
        hue="cab_type",
        marker="o",
        linewidth=2.5,
        palette=["#0f766e", "#1e3a8a"],
        ax=axes[0],
    )
    axes[0].axvspan(
        7, 9, color="#fef08a", alpha=0.35, label="Cao điểm sáng (7h–9h EST)"
    )
    axes[0].axvspan(
        16,
        19,
        color="#fed7aa",
        alpha=0.35,
        label="Cao điểm chiều (16h–19h EST)",
    )
    axes[0].set_title(
        "A. Lưu lượng Báo giá theo 24 Giờ Boston EST (n = "
        f"{len(ts_valid):,} TS gốc)"
    )
    axes[0].set_xlabel("Giờ địa phương Boston — EST / UTC-5 (0h–23h)")
    axes[0].set_ylabel("Số lượt báo giá (Quotes)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_ylim(0, hourly["ride_count"].max() * 1.25)
    axes[0].legend(
        title="Chú thích",
        loc="lower center",
        ncol=2,
        fontsize=8.8,
        frameon=True,
    )

    sns.lineplot(
        data=hourly,
        x="hour_int",
        y="mean_price",
        hue="cab_type",
        marker="s",
        linewidth=2.5,
        palette=["#0f766e", "#1e3a8a"],
        ax=axes[1],
    )
    axes[1].axvspan(7, 9, color="#fef08a", alpha=0.35)
    axes[1].axvspan(16, 19, color="#fed7aa", alpha=0.35)
    axes[1].set_title(
        "B. Giá báo API Trung bình theo 24 Giờ Boston EST (n = "
        f"{len(ts_valid):,} TS gốc)"
    )
    axes[1].set_xlabel("Giờ địa phương Boston — EST / UTC-5 (0h–23h)")
    axes[1].set_ylabel("Giá báo trung bình (USD)")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].set_ylim(12.0, 21.0)
    axes[1].legend(
        title="Nền tảng",
        loc="upper center",
        ncol=2,
        fontsize=9.2,
        frameon=True,
    )
    plt.suptitle(
        "NHÓM C (XU HƯỚNG): BIẾN ĐỘNG LƯU LƯỢNG VÀ GIÁ BÁO API THEO 24 GIỜ "
        "BOSTON (EST)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "fig03_groupC_temporal_trends.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_fig04_relationships(master_df: pd.DataFrame) -> None:
    """Xuất Hình 4 (Nhóm D - Ma trận tương quan & độ dốc theo 5 phân khúc)."""
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    corr_vars = [
        "price",
        "distance",
        "surge_multiplier",
        "distance_x_surge",
        "temp",
        "rain",
        "humidity",
        "wind",
        "clouds",
    ]
    sns.heatmap(
        master_df[corr_vars].corr(method="pearson"),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.3,
        vmax=1.0,
        cbar_kws={"label": "Hệ số tương quan Pearson (r)"},
        ax=axes[0],
    )
    axes[0].set_title(
        "D1. Ma trận Tương quan Tuyến tính Pearson giữa Giá, Lộ trình & "
        "Thời tiết"
    )

    sample_df = master_df.sample(n=6000, random_state=42)
    tiers = ["Shared", "UberX", "UberXL", "Black", "Lux Black XL"]
    sub_sample = sample_df[sample_df["name"].isin(tiers)].copy()
    sub_sample["name"] = sub_sample["name"].cat.remove_unused_categories()
    palette = ["#10b981", "#3b82f6", "#f59e0b", "#6366f1", "#ef4444"]
    for tier, col_hex in zip(tiers, palette, strict=False):
        t_df = sub_sample[
            (sub_sample["name"] == tier) & sub_sample["distance"].notna()
        ]
        axes[1].scatter(
            t_df["distance"],
            t_df["price"],
            alpha=0.35,
            s=16,
            color=col_hex,
            label=tier,
        )
        if len(t_df) > 10:
            m, b = np.polyfit(t_df["distance"], t_df["price"], 1)
            x_vals = np.linspace(0.2, 6.0, 50)
            axes[1].plot(x_vals, m * x_vals + b, color=col_hex, linewidth=2.2)

    axes[1].set_title(
        "D2. Độ dốc Giá báo API theo Quãng đường giữa 5 Phân khúc Đại diện"
    )
    axes[1].set_xlabel("Quãng đường di chuyển (Dặm - Miles)")
    axes[1].set_ylabel("Giá báo API (USD)")
    axes[1].set_xlim(0, 6.5)
    axes[1].set_ylim(0, 80)
    axes[1].legend(title="Phân khúc (Đường OLS)")
    plt.suptitle(
        "NHÓM D (MỐI QUAN HỆ): TƯƠNG QUAN ĐA BIẾN VÀ ĐỘ CO GIÃN GIÁ THEO "
        "QUÃNG ĐƯỜNG",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "fig04_groupD_relationships.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_fig05_service_comparison(master_df: pd.DataFrame) -> None:
    """Xuất Hình 5 (Nhóm E - Boxplot so sánh 12 phân khúc dịch vụ)."""
    plot_df = master_df.copy()
    plot_df["name"] = plot_df["name"].cat.remove_unused_categories()
    order_tiers = (
        plot_df.groupby("name", observed=True)["price"]
        .median()
        .sort_values()
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=plot_df,
        x="price",
        y="name",
        order=order_tiers,
        hue="cab_type",
        dodge=False,
        palette={"Uber": "#1e3a8a", "Lyft": "#0f766e"},
        fliersize=1.5,
        ax=ax,
    )
    ax.set_title(
        "NHÓM E (SO SÁNH): PHÂN BỐ GIÁ BÁO API GIỮA 12 PHÂN KHÚC DỊCH VỤ "
        "CỦA UBER VÀ LYFT (SẮP XẾP THEO TRUNG VỊ)"
    )
    ax.set_xlabel("Giá báo API (USD)")
    ax.set_ylabel("Phân khúc dịch vụ (name)")
    ax.set_xlim(0, 75)
    ax.legend(title="Thương hiệu", loc="lower right")
    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "fig05_groupE_service_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_fig06_surge_weather(master_df: pd.DataFrame) -> None:
    """Xuất Hình 6 (Nhóm F - Heatmap Lyft Surge & so sánh giá Mưa/Tạnh)."""
    fig, axes = plt.subplots(
        1, 2, figsize=(16.5, 6.4), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    lyft_ts = master_df[
        (master_df["cab_type"] == "Lyft")
        & (master_df["is_timestamp_original"] == 1)
        & (master_df["is_surge_original"] == 1)
    ].copy()
    lyft_ts["hour_int"] = lyft_ts["hour"].astype(int)
    piv_surge = lyft_ts.pivot_table(
        index="day_name",
        columns="hour_int",
        values="surge_multiplier",
        aggfunc="mean",
        observed=True,
    ).reindex(day_order)
    sns.heatmap(
        piv_surge,
        cmap="YlOrRd",
        annot=False,
        cbar_kws={"label": "Hệ số Surge trung bình (Lyft)"},
        ax=axes[0],
    )
    axes[0].set_facecolor("#e2e8f0")
    axes[0].set_title(
        "A. Bản đồ Nhiệt Lyft Surge theo Thứ × Giờ Boston EST (n = "
        f"{len(lyft_ts):,} surge thực)\n[Ô xám (T3 21h–23h & T4 0h–1h "
        "EST) = Khung giờ khuyết thu thập dữ liệu gốc]"
    )
    axes[0].set_xlabel("Giờ địa phương Boston — EST / UTC-5 (0h–23h)")
    axes[0].set_ylabel("Ngày trong tuần (Boston EST)")

    wea_valid = master_df[
        (master_df["weather_matched"] == 1) & (master_df["is_raining"].notna())
    ].copy()
    rain_grp = (
        wea_valid.groupby(["cab_type", "is_raining"], observed=True)["price"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    rain_grp["rain_label"] = rain_grp["is_raining"].map(
        {0.0: "Không mưa (Dry)", 1.0: "Có mưa (Rainy)"}
    )
    sns.barplot(
        data=rain_grp,
        x="cab_type",
        y="mean",
        hue="rain_label",
        palette=["#94a3b8", "#0284c7"],
        edgecolor="black",
        ax=axes[1],
    )
    axes[1].set_title(
        "B. Giá báo API Trung bình Mưa vs. Không Mưa\n[Tập khớp thời tiết "
        f"thực không sentinel mưa: n = {len(wea_valid):,}]"
    )
    axes[1].set_xlabel("Thương hiệu nền tảng (cab_type)")
    axes[1].set_ylabel("Giá báo trung bình (USD)")
    axes[1].set_ylim(0, 22)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="$%.2f", fontweight="bold", padding=4)
    axes[1].legend(title="Trạng thái thời tiết thực")
    plt.suptitle(
        "NHÓM F (PHÂN TÍCH NGHIỆP VỤ): CƠ CHẾ ĐỊNH GIÁ ĐỘNG THEO GIỜ BOSTON "
        "EST VÀ THỜI TIẾT KHỚP THỰC",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "fig06_groupF_surge_weather_business.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def step4_generate_visualizations(master_df: pd.DataFrame) -> None:
    """Tạo bộ 6 biểu đồ EDA Giai đoạn 3 (Nhóm A đến F) tại 300 DPI."""
    logger.info(
        "=== BƯỚC 3.1: TẠO BỘ BIỂU ĐỒ TRỰC QUAN HÓA CHUYÊN SÂU (NHÓM A -> "
        "F) ==="
    )
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11

    _plot_fig01_market_overview(master_df)
    _plot_fig02_distributions(master_df)
    _plot_fig03_temporal_trends(master_df)
    _plot_fig04_relationships(master_df)
    _plot_fig05_service_comparison(master_df)
    _plot_fig06_surge_weather(master_df)


def _anova_surge_by_tod(
    df: pd.DataFrame, periods: list[str]
) -> tuple[float, float, float]:
    """Tính One-way ANOVA F, p-value và Eta-squared cho surge theo 4 buổi."""
    groups = [
        df.loc[df["time_of_day"].astype(str) == p, "surge_multiplier"]
        .astype(np.float64)
        .values
        for p in periods
    ]
    f_stat, p_val = stats.f_oneway(*groups)
    vals = df["surge_multiplier"].astype(np.float64)
    grand_mean = float(vals.mean())
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = float(((vals - grand_mean) ** 2).sum())
    return float(f_stat), float(p_val), float(ss_between / ss_total)


def _analyze_rq1_weather_and_time(master_df: pd.DataFrame) -> dict:
    """Phân tích RQ1: Kiểm định Mưa/Tạnh (Welch) & Surge theo 4 buổi EST."""
    wea_df = master_df[
        (master_df["weather_matched"] == 1) & (master_df["is_raining"].notna())
    ]
    rain_p = wea_df.loc[wea_df["is_raining"] == 1.0, "price"].astype(
        np.float64
    )
    dry_p = wea_df.loc[wea_df["is_raining"] == 0.0, "price"].astype(np.float64)

    t_stat, p_val_t = stats.ttest_ind(rain_p, dry_p, equal_var=False)
    nx, ny = len(rain_p), len(dry_p)
    pooled_sd = np.sqrt(
        ((nx - 1) * rain_p.var(ddof=1) + (ny - 1) * dry_p.var(ddof=1))
        / (nx + ny - 2)
    )
    cohens_d = (rain_p.mean() - dry_p.mean()) / pooled_sd

    ts_df = master_df[
        (master_df["is_timestamp_original"] == 1)
        & (master_df["is_surge_original"] == 1)
    ]
    lyft_only = ts_df[ts_df["cab_type"] == "Lyft"]
    periods = ["Morning", "Afternoon", "Evening", "Night"]
    f_all, p_all, eta_all = _anova_surge_by_tod(ts_df, periods)
    f_lyft, p_lyft, eta_lyft = _anova_surge_by_tod(lyft_only, periods)

    tod_means = {}
    for p in periods:
        sub_all = ts_df[ts_df["time_of_day"].astype(str) == p]
        sub_lyft = lyft_only[lyft_only["time_of_day"].astype(str) == p]
        tod_means[p] = {
            "count": int(len(sub_all)),
            "mean_price": round(float(sub_all["price"].mean()), 3),
            "mean_surge_all": round(
                float(sub_all["surge_multiplier"].mean()), 4
            ),
            "mean_surge_lyft": round(
                float(sub_lyft["surge_multiplier"].mean()), 4
            ),
        }

    return {
        "weather_matched_n": int(len(wea_df)),
        "valid_timestamp_est_n": int(len(ts_df)),
        "rain_count": int(nx),
        "dry_count": int(ny),
        "rain_mean_price": round(float(rain_p.mean()), 4),
        "dry_mean_price": round(float(dry_p.mean()), 4),
        "rain_median_price": round(float(rain_p.median()), 4),
        "dry_median_price": round(float(dry_p.median()), 4),
        "diff_usd": round(float(rain_p.mean() - dry_p.mean()), 4),
        "diff_pct": round(
            float((rain_p.mean() - dry_p.mean()) / dry_p.mean() * 100), 3
        ),
        "welch_t_stat": round(float(t_stat), 4),
        "welch_p_value": float(p_val_t),
        "cohens_d": round(float(cohens_d), 5),
        "anova_all_f_stat": round(f_all, 4),
        "anova_all_p_value": p_all,
        "anova_all_eta_squared": round(eta_all, 6),
        "anova_lyft_f_stat": round(f_lyft, 4),
        "anova_lyft_p_value": p_lyft,
        "anova_lyft_eta_squared": round(eta_lyft, 6),
        "time_of_day_means": tod_means,
    }


def _analyze_rq2_tier_pricing(
    master_df: pd.DataFrame,
) -> tuple[dict, list[dict], dict]:
    """Phân tích RQ2: Ước lượng OLS từng gói và đối đầu 5 cặp Uber vs Lyft."""
    tier_ols = {}
    obs_master = master_df[
        (master_df["is_surge_original"] == 1) & master_df["distance"].notna()
    ]
    for tier_name, grp in obs_master.groupby("name", observed=True):
        if len(grp) < 10:
            continue
        x_raw = grp["distance"].astype(np.float64).values
        x_eff = grp["distance_x_surge"].astype(np.float64).values
        y_val = grp["price"].astype(np.float64).values
        slope_raw, intercept_raw, r_raw, _, _ = stats.linregress(x_raw, y_val)
        slope_eff, intercept_eff, r_eff, _, _ = stats.linregress(x_eff, y_val)
        tier_ols[str(tier_name)] = {
            "cab_type": str(grp["cab_type"].iloc[0]),
            "count": int(len(grp)),
            "base_fare_unadjusted": round(float(intercept_raw), 2),
            "rate_per_mile_unadjusted": round(float(slope_raw), 2),
            "r2_unadjusted": round(float(r_raw**2), 4),
            "base_fare_surge_adj": round(float(intercept_eff), 2),
            "rate_per_mile_surge_adj": round(float(slope_eff), 2),
            "r2_surge_adj": round(float(r_eff**2), 4),
            "mean_price": round(float(y_val.mean()), 2),
            "median_price": round(float(np.median(y_val)), 2),
        }

    head_to_head_pairs = [
        ("Tiết kiệm đi ghép (Pool/Shared)", "UberPool", "Shared"),
        ("Phổ thông 4 chỗ (Standard)", "UberX", "Lyft"),
        ("Phổ thông 6 chỗ (XL)", "UberXL", "Lyft XL"),
        ("Hạng sang Sedan (Luxury Black)", "Black", "Lux Black"),
        ("Hạng sang SUV (Luxury SUV)", "Black SUV", "Lux Black XL"),
    ]
    pair_comparisons = []
    for seg_label, u_name, l_name in head_to_head_pairs:
        u_info, l_info = tier_ols[u_name], tier_ols[l_name]
        pair_comparisons.append(
            {
                "segment": seg_label,
                "uber_tier": u_name,
                "lyft_tier": l_name,
                "uber_mean_price": u_info["mean_price"],
                "lyft_mean_price": l_info["mean_price"],
                "uber_base_fare": u_info["base_fare_surge_adj"],
                "lyft_base_fare": l_info["base_fare_surge_adj"],
                "uber_rate_per_mile": u_info["rate_per_mile_surge_adj"],
                "lyft_rate_per_mile": l_info["rate_per_mile_surge_adj"],
                "price_diff_lyft_minus_uber": round(
                    l_info["mean_price"] - u_info["mean_price"], 2
                ),
            }
        )
    standalone = {"WAV": tier_ols["WAV"], "Lux": tier_ols["Lux"]}
    return tier_ols, pair_comparisons, standalone


def _analyze_rq3_baseline_models(
    master_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, int]:
    """Chạy 3 mô hình nhanh GĐ3 (OLS, Ridge, HistGB) và trích xuất Beta*."""
    num_features = [
        "distance",
        "surge_multiplier",
        "distance_x_surge",
        "temp",
        "clouds",
        "pressure",
        "rain",
        "humidity",
        "wind",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_rush_hour",
    ]
    cat_features = ["cab_type", "name", "source", "destination", "time_of_day"]
    model_df = master_df[master_df["is_surge_original"] == 1]
    X = model_df[num_features + cat_features].copy()
    for c in cat_features:
        X[c] = X[c].cat.remove_unused_categories()
    y = model_df["price"].astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                cat_features,
            ),
        ]
    )
    models = {
        "OLS_LinearRegression": LinearRegression(),
        "Ridge_L2_alpha1": Ridge(alpha=1.0),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, max_depth=8, random_state=42
        ),
    }

    model_metrics = {}
    ols_pipe = None
    for m_name, estimator in models.items():
        pipe = Pipeline([("prep", preprocessor), ("reg", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        model_metrics[m_name] = {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "R2": round(float(r2_score(y_test, preds)), 4),
            "RMSE": round(float(root_mean_squared_error(y_test, preds)), 4),
            "MAE": round(float(mean_absolute_error(y_test, preds)), 4),
            "MAPE_pct": round(
                float(mean_absolute_percentage_error(y_test, preds) * 100), 2
            ),
        }
        if m_name == "OLS_LinearRegression":
            ols_pipe = pipe

    X_tr_mat = ols_pipe.named_steps["prep"].transform(X_train)
    feat_names = ols_pipe.named_steps["prep"].get_feature_names_out()
    raw_coefs = ols_pipe.named_steps["reg"].coef_
    std_betas = raw_coefs * np.std(X_tr_mat, axis=0)
    coef_df = (
        pd.DataFrame(
            {
                "feature": feat_names,
                "coef": std_betas,
                "raw_coef": raw_coefs,
                "abs_coef": np.abs(std_betas),
            }
        )
        .sort_values("abs_coef", ascending=False)
        .head(12)
    )
    return model_metrics, coef_df, len(X_test)


def _plot_fig07_business_models(
    model_metrics: dict, coef_df: pd.DataFrame, n_test: int
) -> None:
    """Xuất Hình 7: Hiệu năng 3 mô hình GĐ3 và Top 10 Beta*."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    m_keys = [
        ("OLS Linear", "OLS_LinearRegression"),
        ("Ridge (L2)", "Ridge_L2_alpha1"),
        ("HistGradientBoosting", "HistGradientBoosting"),
    ]
    r2_vals = [model_metrics[k]["R2"] * 100 for _, k in m_keys]
    mape_vals = [model_metrics[k]["MAPE_pct"] for _, k in m_keys]

    x_pos = np.arange(len(m_keys))
    w = 0.35
    b1 = axes[0].bar(
        x_pos - w / 2, r2_vals, width=w, color="#1e3a8a", label="R² Score (%)"
    )
    b2 = axes[0].bar(
        x_pos + w / 2,
        mape_vals,
        width=w,
        color="#0f766e",
        label="MAPE Error (%)",
    )
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([lbl for lbl, _ in m_keys])
    axes[0].set_ylim(0, 122)
    axes[0].set_title(
        f"RQ3-A. Hiệu năng Dự báo Giá báo API trên Tập Test (N = {n_test:,})"
    )
    axes[0].set_ylabel("Tỷ lệ phần trăm (%)")
    axes[0].bar_label(b1, fmt="%.2f%%", fontweight="bold", padding=3)
    axes[0].bar_label(b2, fmt="%.2f%%", fontweight="bold", padding=3)
    axes[0].legend(loc="upper center", ncol=2, frameon=True)

    top10 = coef_df.head(10).sort_values("abs_coef", ascending=True)
    c_bars = ["#1e3a8a" if c > 0 else "#dc2626" for c in top10["coef"]]
    axes[1].barh(
        top10["feature"].str.replace("cat__", "").str.replace("num__", ""),
        top10["coef"],
        color=c_bars,
        edgecolor="black",
    )
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title(
        "RQ3-B. Top 10 Nhân tố Chi phối Giá báo API (Standardized Beta đồng "
        "nhất)"
    )
    axes[1].set_xlabel(
        "Hệ số chuẩn hóa Beta* (USD thay đổi trên +1 SD của biến)"
    )
    plt.suptitle(
        "PHẦN 4 (BÀI TOÁN KINH DOANH): ĐÁNH GIÁ MÔ HÌNH DỰ BÁO VÀ NHÂN TỐ "
        "CHI PHỐI GIÁ BÁO API",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "fig07_business_questions_modeling.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def step5_business_questions_analysis(master_df: pd.DataFrame) -> dict:
    """Phân tích & kiểm định 3 câu hỏi kinh doanh RQ1–RQ3 và xuất Fig 07."""
    logger.info("=== BƯỚC 4.1: PHÂN TÍCH & KIỂM ĐỊNH 3 CÂU HỎI KINH DOANH ===")
    rq1 = _analyze_rq1_weather_and_time(master_df)
    tier_ols, pair_comparisons, standalone = _analyze_rq2_tier_pricing(
        master_df
    )
    model_metrics, coef_df, n_test = _analyze_rq3_baseline_models(master_df)
    _plot_fig07_business_models(model_metrics, coef_df, n_test)
    return {
        "RQ1_weather_and_time": rq1,
        "RQ2_pricing_structure_by_tier": tier_ols,
        "RQ2_head_to_head_pairs": pair_comparisons,
        "RQ2_standalone_tiers": standalone,
        "RQ3_model_evaluation": model_metrics,
        "RQ3_top_drivers_standardized_ols": [
            {
                "feature": str(r["feature"]),
                "coef": round(float(r["coef"]), 4),
                "raw_coef": round(float(r["raw_coef"]), 4),
            }
            for _, r in coef_df.iterrows()
        ],
    }


def main():
    cab_raw = load_cab_rides_chunked(chunksize=100_000)
    weather_raw = pd.read_csv(RAW_DIR / "weather.csv", low_memory=False)

    initial_audit = step1_initial_audit(cab_raw, weather_raw)
    cab_clean, weather_clean, cleaning_log = step2_clean_datasets(
        cab_raw, weather_raw
    )
    master_df, eda_res = step3_merge_and_eda(cab_clean, weather_clean)
    step4_generate_visualizations(master_df)
    bq_res = step5_business_questions_analysis(master_df)

    full_log = {
        "initial_audit": initial_audit,
        "cleaning_log": cleaning_log,
        "eda_results": eda_res,
        "business_questions": bq_res,
    }
    out_json = PROCESSED_DIR / "stage3_audit_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_log, f, ensure_ascii=False, indent=2)
    logger.info(
        "=== ĐÃ HOÀN TẤT CHẠY TOÀN BỘ PIPELINE GIAI ĐOẠN 3 VÀ LƯU LOG TẠI: "
        f"{out_json} ==="
    )


if __name__ == "__main__":
    main()
