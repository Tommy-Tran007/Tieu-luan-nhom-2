"""
Script: clean_raw_data.py
Mục đích: Thực hiện quy trình làm sạch dữ liệu thô THỰC TẾ (Real-world Data
Wrangling)
trên bộ dữ liệu gốc data/raw/cab_rides.csv và data/raw/weather.csv.
Tuân thủ nguyên tắc Khoa học Dữ liệu (v5 — khắc phục triệt để CRIT-01, CRIT-02,
CRIT-03 & H4):
- Giữ nguyên 100% dữ liệu gốc trong data/raw/, xuất dữ liệu sạch ra
  data/processed/.
- Không sử dụng bất kỳ tệp đáp án bên ngoài nào để điền dữ liệu bị xóa.
- Khôi phục tất định 1-1: product_id -> (cab_type, name).
- Xử lý 7 tầng bẫy nhiễu: đơn vị chuỗi, dấu phẩy thập phân, khoảng trắng/lỗi gõ
  tắt địa danh,
  mã lính canh (sentinels), đảo dấu âm, và phóng đại quy mô bậc 10 (10x, 100x,
  1000x) qua log10(val/ref).
- [CRIT-02] Không ffill/bfill time_stamp; tạo cờ is_timestamp_original và giữ
  NA.
- [CRIT-03] Không điền mù surge_multiplier = 1.0 trên các dòng Lyft bị
  khuyết/sentinel;
  tạo cờ is_surge_original (1 = quan sát hợp lệ / Uber API cố định 1.0x, 0 =
  Lyft khuyết surge -> giữ NaN).
- [H4] Tách cờ is_rain_sentinel khỏi ô khô tự nhiên Dark Sky (không ép sentinel
  mưa về 0.0),
  khôi phục quy mô bậc 10 khí tượng bằng log10 theo trung vị cùng giờ và điền
  khuyết khí tượng
  bằng trung vị cùng giờ toàn Boston (thay vì trung bình 24 ngày).
Tác giả: Trần Thanh Huy (MSSV: 2521001060) & Lê Huy Thắng (MSSV: 2521001092)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PRODUCT_MAP = {
    "55c66225-fbe7-4fd5-9072-eab1ece5e23e": ("Uber", "UberX"),
    "6c84fd89-3f11-4782-9b50-97c468b19529": ("Uber", "Black"),
    "6d318bcc-22a3-4af6-bddd-b409bfce1546": ("Uber", "Black SUV"),
    "6f72dfc5-27f1-42e8-84db-ccc7a75f6969": ("Uber", "UberXL"),
    "8cf7e821-f0d3-49c6-8eba-e679c0ebcf6a": ("Uber", "Taxi"),
    "997acbb5-e102-41e1-b155-9df7de0a73f2": ("Uber", "UberPool"),
    "9a0e7b09-b92b-4c41-9779-2ad22b4d779d": ("Uber", "WAV"),
    "lyft": ("Lyft", "Lyft"),
    "lyft_line": ("Lyft", "Shared"),
    "lyft_lux": ("Lyft", "Lux Black"),
    "lyft_luxsuv": ("Lyft", "Lux Black XL"),
    "lyft_plus": ("Lyft", "Lyft XL"),
    "lyft_premier": ("Lyft", "Lux"),
}

SENTINEL_STRINGS = {"?", "-", "unknown", "chưa rõ", "nan", "null", "none"}
WEATHER_NUM_SENTINELS = (-1, -1.0, -100, 9999, 10000, 100000)
RAIN_NUM_SENTINELS = (-1, -1.0, -100, 1000, 9999, 10000, 100000)


def clean_unit_string(val: Any) -> float:
    """
    Bóc tách đơn vị tiền tệ/khối lượng, sửa dấu thập phân và trả về số thực.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in SENTINEL_STRINGS or s.lower().startswith("nan "):
        return np.nan
    s = re.sub(r"\s*(USD|VND|đ|kg|%)\s*$", "", s, flags=re.IGNORECASE).strip()
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def standardize_location(loc_series: pd.Series) -> pd.Series:
    """
    Chuẩn hóa địa danh: loại bỏ khoảng trắng, sửa lỗi gõ tắt và Title Case.
    """
    s = loc_series.astype(str).str.strip().str.title()
    s = s.replace(
        {
            "Northeastern Universi": "Northeastern University",
            "?": np.nan,
            "-": np.nan,
            "Unknown": np.nan,
            "Chưa Rõ": np.nan,
            "Nan": np.nan,
            "Null": np.nan,
            "None": np.nan,
        }
    )
    return s


def recover_scaled_series(
    series: pd.Series,
    ref_median: pd.Series,
    min_valid: float,
    max_valid: float,
    preserve_decimals: int | None = None,
) -> tuple[pd.Series, int]:
    """
    Khôi phục các giá trị bị nhân phóng đại 10^m (m in {1, 2, 3}) bằng công
    thức logarit cơ số 10:
    m = clip(round(log10(val / ref_median)), 1, 3) khi val > 3.55 * ref_median
    hoặc val > max_valid. Nếu val <= max_valid và preserve_decimals được bật,
    chỉ chia 10^m khi giá trị sau chia bảo toàn đúng số chữ số thập phân gốc.
    Trả về: (Series đã khôi phục, số lượng bản ghi đã được chia tỷ lệ).
    """
    out = series.copy()
    valid_mask = (
        out.notna() & (out > 0) & ref_median.notna() & (ref_median > 0)
    )
    ratio = out[valid_mask] / ref_median[valid_mask]
    # Ngưỡng 10^0.55 ≈ 3.55 bảo vệ toàn bộ giao dịch thực tế (kể cả surge
    # 3.0x), chỉ tác động lên các bội số 10x, 100x, 1000x.
    candidate_mask = (ratio > 3.55) | (out[valid_mask] > max_valid)
    idx_to_fix = ratio[candidate_mask].index
    if len(idx_to_fix) == 0:
        return out, 0

    exponents = np.clip(
        np.round(np.log10(out.loc[idx_to_fix] / ref_median.loc[idx_to_fix])),
        1,
        3,
    )
    candidates = out.loc[idx_to_fix] / (10.0**exponents)
    if preserve_decimals is not None:
        dec_scale = 10.0**preserve_decimals
        preserves = (
            np.abs(candidates * dec_scale - np.round(candidates * dec_scale))
            < 1e-6
        )
        valid_fix = (out.loc[idx_to_fix] > max_valid) | preserves
        idx_to_fix = idx_to_fix[valid_fix]
        candidates = candidates.loc[idx_to_fix]
    if len(idx_to_fix) == 0:
        return out, 0

    out.loc[idx_to_fix] = candidates
    out.loc[(out < min_valid) | (out > max_valid)] = np.nan
    return out, int(len(idx_to_fix))


def recover_weather_metric_scaled(
    series: pd.Series,
    dt_hour: pd.Series,
    min_valid: float,
    max_valid: float,
) -> tuple[pd.Series, int]:
    """
    [H4 FIX] Khôi phục các giá trị khí tượng vượt ngưỡng vật lý max_valid (bị
    nhân 10x/100x/1000x)
    bằng cách chọn bậc m in {1, 2, 3} vừa rơi vào miền vật lý [min_valid,
    max_valid] vừa gần nhất
    với trung vị cùng khung giờ toàn Boston (thay cho quy tắc chia ngưỡng cứng
    dễ sai lệch 10x).
    """
    out = series.copy()
    normal = out.where((out >= min_valid) & (out <= max_valid))
    pos_normal = normal.where(normal > 0)
    global_ref = (
        float(pos_normal.median())
        if pos_normal.notna().any()
        else (min_valid + max_valid) / 2.0
    )
    hour_ref = normal.groupby(dt_hour).transform("median")
    hour_ref = hour_ref.where(hour_ref > 1e-4, global_ref).fillna(global_ref)

    # Kích hoạt khi vượt ngưỡng vật lý max_valid HOẶC cao gấp > 5.0 lần trung
    # vị cùng giờ Boston
    # (bắt trọn cả các ô bị nhân 10x nhưng vẫn nằm dưới max_valid, ví dụ wind
    # 2.23 -> 22.3 mph).
    candidate_mask = out.notna() & (
        (out > max_valid) | ((hour_ref > 0.04) & ((out / hour_ref) > 5.0))
    )
    idx_to_fix = out[candidate_mask].index
    if len(idx_to_fix) == 0:
        out.loc[(out < min_valid) | (out > max_valid)] = np.nan
        return out, 0

    fixed_count = 0
    for idx in idx_to_fix:
        val = float(out.loc[idx])
        ref = float(hour_ref.loc[idx])
        candidates = [val / (10.0**e) for e in (1, 2, 3)]
        valid_cands = [c for c in candidates if min_valid <= c <= max_valid]
        if valid_cands:
            best = min(
                valid_cands,
                key=lambda c: abs(np.log10(max(c, 1e-4) / max(ref, 1e-4))),
            )
            out.loc[idx] = best
            fixed_count += 1
        else:
            out.loc[idx] = np.nan

    out.loc[(out < min_valid) | (out > max_valid)] = np.nan
    return out, fixed_count


def recover_weather_rain_scaled(
    rain_series: pd.Series,
    time_stamp_series: pd.Series,
) -> tuple[pd.Series, int]:
    """
    [MED-02 FIX] Khôi phục các giá trị rain > 1.0 inch/h (bị nhân 10x, 100x,
    1000x)
    bằng log10(val / ref_same_hour_median) CHỈ trên các ô rain > 1.0 để tuyệt
    đối không làm biến dạng
    các bản tin mưa thật hợp lệ trong miền [0, 0.85] inch/h.
    """
    out = rain_series.copy()
    dt_hour = (
        pd.to_datetime(time_stamp_series, unit="s", utc=True, errors="coerce")
        .dt.tz_convert("America/New_York")
        .dt.floor("h")
    )
    normal_rain = out.where((out > 0.0) & (out <= 1.0))
    global_rain_med = (
        float(normal_rain.median()) if normal_rain.notna().any() else 0.035
    )
    hour_rain_ref = (
        normal_rain.groupby(dt_hour)
        .transform("median")
        .fillna(global_rain_med)
    )

    idx_to_fix = out[out.notna() & (out > 1.0)].index
    if len(idx_to_fix) == 0:
        return out, 0

    exponents = np.clip(
        np.round(
            np.log10(out.loc[idx_to_fix] / hour_rain_ref.loc[idx_to_fix])
        ),
        1,
        3,
    )
    out.loc[idx_to_fix] = out.loc[idx_to_fix] / (10.0**exponents)
    out.loc[out > 1.0] = np.nan
    return out, int(len(idx_to_fix))


def load_cab_rides_chunked(
    chunksize: int = 100_000,
    pre_filter_valid: bool = False,
) -> pd.DataFrame:
    """
    Đọc dữ liệu lớn cab_rides.csv theo phân khối chunksize để kiểm soát đỉnh
    RAM.
    - Nén ngay từng khối (ép product_id sang category) trước khi gộp.
    - Khi pre_filter_valid=True (chế độ ETL tiết kiệm RAM thực sự), lọc sớm
      từng khối
      các dòng Uber Taxi và khuyết price/source/destination trước khi nối (~41%
      dòng bị loại ngay).
    """
    chunks: list[pd.DataFrame] = []
    taxi_pid = "8cf7e821-f0d3-49c6-8eba-e679c0ebcf6a"
    for chunk in pd.read_csv(
        RAW_DIR / "cab_rides.csv",
        chunksize=chunksize,
        low_memory=False,
    ):
        if pre_filter_valid:
            pid_clean = chunk["product_id"].astype(str).str.strip()
            keep = (
                (pid_clean != taxi_pid)
                & chunk["price"].notna()
                & chunk["source"].notna()
                & chunk["destination"].notna()
            )
            chunk = chunk.loc[keep].copy()
        chunk["product_id"] = (
            chunk["product_id"].astype(str).str.strip().astype("category")
        )
        chunks.append(chunk)
    df_out = pd.concat(chunks, ignore_index=True)
    del chunks
    return df_out


def _clean_cab_ids_and_sentinels(
    df: pd.DataFrame, log: dict[str, Any]
) -> tuple[pd.DataFrame, int]:
    """Chuẩn hóa khóa chính, product_id, địa bàn, đơn vị và mã lính canh."""
    df["id"] = df["id"].astype(str).str.strip()
    df["product_id"] = df["product_id"].astype(str).str.strip()
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["id"]).copy()
    log["duplicate_ids_dropped"] = int(before_dedup - len(df))
    log["valid_product_id_count"] = int(
        df["product_id"].isin(PRODUCT_MAP.keys()).sum()
    )

    valid_names = {v[1] for v in PRODUCT_MAP.values()}
    log["restored_cab_type_records"] = int(
        (
            df["cab_type"].isna()
            | ~df["cab_type"].astype(str).isin(["Uber", "Lyft"])
        ).sum()
    )
    log["restored_name_records"] = int(
        (df["name"].isna() | ~df["name"].astype(str).isin(valid_names)).sum()
    )
    df["cab_type"] = df["product_id"].map(
        lambda pid: PRODUCT_MAP.get(pid, ("Uber", "Unknown"))[0]
    )
    df["name"] = df["product_id"].map(
        lambda pid: PRODUCT_MAP.get(pid, ("Uber", "Unknown"))[1]
    )
    df["source"] = standardize_location(df["source"])
    df["destination"] = standardize_location(df["destination"])
    log["missing_source_after_std"] = int(df["source"].isna().sum())
    log["missing_destination_after_std"] = int(df["destination"].isna().sum())

    for col in ("distance", "price", "surge_multiplier", "time_stamp"):
        df[col] = df[col].apply(clean_unit_string)

    taxi_rows = int((df["name"] == "Taxi").sum())
    log["taxi_mnar_total_rows"] = taxi_rows
    log["taxi_sentinel_injected_rows"] = int(
        (df.loc[df["name"] == "Taxi", "price"].notna()).sum()
    )
    df.loc[df["name"] == "Taxi", "price"] = np.nan

    sent_specs = (
        ("price", [-1, -1.0, -100, 1000, 9999, 10000, 100000]),
        ("distance", [-1, -1.0, 1000, 9999, 10000, 100000]),
        ("surge_multiplier", [-1, -1.0, 9999, 10000, 100000]),
    )
    for col, vals in sent_specs:
        mask = df[col].isin(vals)
        key = "surge" if col == "surge_multiplier" else col
        log[f"{key}_numeric_sentinels_to_nan"] = int(mask.sum())
        df.loc[mask, col] = np.nan
        log[f"{key}_neg_fixed"] = int((df[col] < 0).sum())
        df[col] = df[col].abs()
    return df, taxi_rows


def _recover_cab_scales_and_filter(
    df: pd.DataFrame,
    taxi_rows: int,
    initial_rows: int,
    log: dict[str, Any],
) -> pd.DataFrame:
    """Khôi phục quy mô bậc 10 cho surge, distance, price và lọc dòng sạch."""
    log["surge_scale_fixed"] = int((df["surge_multiplier"] >= 10).sum())
    df.loc[df["surge_multiplier"] >= 1000, "surge_multiplier"] /= 1000.0
    df.loc[df["surge_multiplier"] >= 100, "surge_multiplier"] /= 100.0
    df.loc[df["surge_multiplier"] >= 10, "surge_multiplier"] /= 10.0
    df.loc[
        (df["surge_multiplier"] < 1.0) | (df["surge_multiplier"] > 3.0),
        "surge_multiplier",
    ] = np.nan
    df.loc[df["cab_type"] == "Uber", "surge_multiplier"] = 1.0
    df["is_surge_original"] = df["surge_multiplier"].notna().astype(np.int8)

    # Dùng mốc tham chiếu cố định (2.15 mi) thay vì median toàn cục theo tuyến
    # để khôi phục thập phân mà không rò rỉ thông tin phân phối giữa train/test
    route_ref = pd.Series(2.15, index=df.index, dtype=np.float64)
    df["distance"], log["distance_scale_fixed"] = recover_scaled_series(
        df["distance"],
        route_ref,
        min_valid=0.02,
        max_valid=8.0,
        preserve_decimals=2,
    )

    norm_p = df["price"].where(
        (df["is_surge_original"] == 1)
        & (df["price"] >= 2.5)
        & (df["price"] <= 65.0)
    )
    base_route = (
        (norm_p / df["surge_multiplier"])
        .groupby([df["name"], df["source"], df["destination"]])
        .transform("median")
    )
    base_tier = (
        (norm_p / df["surge_multiplier"])
        .groupby(df["name"])
        .transform("median")
    )
    price_ref = base_route.fillna(base_tier).fillna(13.5) * df[
        "surge_multiplier"
    ].fillna(1.0)
    df["price"], log["price_scale_fixed"] = recover_scaled_series(
        df["price"],
        price_ref,
        min_valid=2.5,
        max_valid=95.0,
        preserve_decimals=2,
    )

    miss_p = int(df["price"].isna().sum())
    log["missing_price_total"] = miss_p
    log["missing_price_uber_taxi_mnar"] = taxi_rows
    log["missing_price_other_corrupted"] = miss_p - taxi_rows
    valid = (
        df["price"].notna() & df["source"].notna() & df["destination"].notna()
    )
    df_clean = df[valid].copy()
    log["rows_after_target_route_filter"] = int(len(df_clean))
    log["rows_removed_total"] = int(initial_rows - len(df_clean))
    logger.info(
        "Số dòng cab_rides hợp lệ sau khi lọc missing price/route: "
        f"{len(df_clean):,} / {initial_rows:,}"
    )
    return df_clean


def _finalize_cab_audit_and_imputation(
    df_clean: pd.DataFrame, log: dict[str, Any]
) -> pd.DataFrame:
    """Điền khuyết distance theo trung vị tuyến, bảo toàn TS & log IQR."""
    lyft_mask = df_clean["cab_type"] == "Lyft"
    lyft_obs = int((lyft_mask & (df_clean["is_surge_original"] == 1)).sum())
    lyft_miss = int((lyft_mask & (df_clean["is_surge_original"] == 0)).sum())
    log["surge_filled_1_0"] = 0
    log["surge_original_valid_total"] = int(
        (df_clean["is_surge_original"] == 1).sum()
    )
    log["lyft_surge_original_count"] = lyft_obs
    log["lyft_surge_missing_kept_nan_count"] = lyft_miss
    logger.info(
        "[CRIT-03] Kiểm toán surge_multiplier Lyft: Quan sát thực = "
        f"{lyft_obs:,}, Khuyết/Sentinel = {lyft_miss:,} "
        f"({lyft_miss / max(int(lyft_mask.sum()), 1) * 100:.2f}%)."
    )

    # Giữ nguyên NaN của `distance` ở Stage 2 để `RouteDistanceImputer` trong
    # `sklearn.pipeline.Pipeline` nội suy strictly-on-train sau khi chia tập.
    n_missing_distance = int(df_clean["distance"].isna().sum())
    log["distance_missing_kept_nan_for_pipeline"] = n_missing_distance
    log["distance_imputed_by_route_median"] = 0

    df_clean["is_timestamp_original"] = (
        df_clean["time_stamp"].notna().astype(np.int8)
    )
    df_clean["time_stamp"] = df_clean["time_stamp"].round().astype("Int64")
    ts_orig = int(df_clean["is_timestamp_original"].sum())
    ts_miss = int((df_clean["is_timestamp_original"] == 0).sum())
    log["timestamp_original_valid_count"] = ts_orig
    log["timestamp_missing_not_imputed_count"] = ts_miss
    log["timestamp_imputed_count"] = 0
    logger.info(
        f"[CRIT-02] Kiểm toán time_stamp: Gốc hợp lệ = {ts_orig:,} "
        f"({ts_orig / len(df_clean) * 100:.2f}%), "
        f"Thiếu (giữ NA) = {ts_miss:,}."
    )

    for col in ("price", "distance", "surge_multiplier"):
        s_v = df_clean[col].dropna()
        q1, q3 = float(s_v.quantile(0.25)), float(s_v.quantile(0.75))
        iqr = q3 - q1
        lo, hi = max(0.0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
        cnt = int(((s_v < lo) | (s_v > hi)).sum())
        log[f"remaining_iqr_outliers_{col}"] = {
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_fence": round(lo, 4),
            "upper_fence": round(hi, 4),
            "count": cnt,
            "pct": round(cnt / max(len(s_v), 1) * 100, 2),
        }
    out_cols = df_clean[
        [
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
        ]
    ].copy()
    for cat_c in ("cab_type", "name", "product_id", "source", "destination"):
        out_cols[cat_c] = out_cols[cat_c].astype("category")
    for f32_c in ("distance", "surge_multiplier", "price"):
        out_cols[f32_c] = out_cols[f32_c].astype(np.float32)
    return out_cols


def clean_cab_rides(
    cab_raw: pd.DataFrame | None = None,
    chunksize: int = 100_000,
    return_log: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Làm sạch 7 tầng nhiễu cho cab_rides.csv, bảo toàn nguồn gốc TS/surge."""
    logger.info("=== BẮT ĐẦU LÀM SẠCH CAB_RIDES.CSV (CHUNKED STREAMING) ===")
    df = (
        load_cab_rides_chunked(chunksize=chunksize, pre_filter_valid=False)
        if cab_raw is None
        else cab_raw.copy()
    )
    initial_rows = len(df)
    log: dict[str, Any] = {"initial_rows_loaded": int(initial_rows)}
    df, taxi_rows = _clean_cab_ids_and_sentinels(df, log)
    df_clean = _recover_cab_scales_and_filter(df, taxi_rows, initial_rows, log)
    out_df = _finalize_cab_audit_and_imputation(df_clean, log)
    return (out_df, log) if return_log else out_df


def _clean_weather_metrics_and_rain(
    df: pd.DataFrame, log: dict[str, Any]
) -> pd.DataFrame:
    """Làm sạch sentinel, dấu âm, quy mô bậc 10 cho 5 biến khí tượng & mưa."""
    df["location"] = standardize_location(df["location"])
    raw_rain_str = df["rain"].astype(str).str.strip().str.lower()
    rain_str_sent = df["rain"].notna() & (
        raw_rain_str.isin(SENTINEL_STRINGS)
        | raw_rain_str.str.startswith("nan ")
    )
    for col in (
        "temp",
        "clouds",
        "pressure",
        "rain",
        "humidity",
        "wind",
        "time_stamp",
    ):
        df[col] = df[col].apply(clean_unit_string)

    rain_num_sent = df["rain"].isin(RAIN_NUM_SENTINELS)
    rain_sent = rain_str_sent | rain_num_sent
    df["is_rain_sentinel"] = rain_sent.astype(np.int8)
    log["rain_str_sentinels"] = int(rain_str_sent.sum())
    log["rain_num_sentinels"] = int(rain_num_sent.sum())
    log["rain_sentinels_to_nan"] = int(rain_sent.sum())

    dt_hour = (
        pd.to_datetime(df["time_stamp"], unit="s", utc=True, errors="coerce")
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.floor("h")
    )
    bounds = {
        "temp": (10.0, 80.0),
        "clouds": (0.0, 1.0),
        "pressure": (970.0, 1045.0),
        "humidity": (0.10, 1.0),
        "wind": (0.0, 25.0),
    }
    for col, (lo, hi) in bounds.items():
        sent_m = df[col].isin(WEATHER_NUM_SENTINELS)
        log[f"{col}_sentinels_to_nan"] = int(sent_m.sum())
        df.loc[sent_m, col] = np.nan
        log[f"{col}_neg_fixed"] = int((df[col] < 0).sum())
        df[col] = df[col].abs()
        df[col], scale_cnt = recover_weather_metric_scaled(
            df[col], dt_hour, lo, hi
        )
        log[f"{col}_scale_fixed"] = int(scale_cnt)

    df.loc[rain_sent, "rain"] = np.nan
    df["rain"] = df["rain"].abs()
    df["rain"], rain_sc = recover_weather_rain_scaled(
        df["rain"], df["time_stamp"]
    )
    log["rain_scale_fixed"] = int(rain_sc)
    nat_dry = (df["is_rain_sentinel"] == 0) & df["rain"].isna()
    log["rain_natural_dry_filled_zero_count"] = int(nat_dry.sum())
    log["rain_sentinel_kept_nan_count"] = int(
        (df["is_rain_sentinel"] == 1).sum()
    )
    df.loc[nat_dry, "rain"] = 0.0
    return df


def clean_weather(
    weather_raw: pd.DataFrame | None = None,
    return_log: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Làm sạch weather.csv (log10 + trung vị cùng giờ + tách rain sentinel).
    """
    logger.info(
        "=== BẮT ĐẦU LÀM SẠCH WEATHER.CSV (H4: LOG10 + SAME-HOUR MEDIAN + "
        "RAIN SENTINEL) ==="
    )
    if weather_raw is None:
        weather_raw = pd.read_csv(RAW_DIR / "weather.csv", low_memory=False)
    df = weather_raw.copy()
    init_rows = len(df)
    log: dict[str, Any] = {"initial_rows": int(init_rows)}
    df = _clean_weather_metrics_and_rain(df, log)

    before_dedup = len(df)
    df = df.drop_duplicates()
    log["exact_duplicates_dropped"] = int(before_dedup - len(df))
    before_key = len(df)
    df = df.dropna(subset=["location", "time_stamp"]).copy()
    log["missing_key_rows_dropped"] = int(before_key - len(df))
    df["time_stamp"] = df["time_stamp"].astype(np.int64)

    dt_clean = (
        pd.to_datetime(df["time_stamp"].astype("float64"), unit="s", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    hr_key, day_key = dt_clean.dt.floor("h"), dt_clean.dt.date
    for col in ("temp", "clouds", "pressure", "humidity", "wind"):
        na_0 = int(df[col].isna().sum())
        df[col] = df[col].fillna(df[col].groupby(hr_key).transform("median"))
        na_1 = int(df[col].isna().sum())
        if na_1 > 0:
            df[col] = df[col].fillna(
                df[col].groupby(day_key).transform("median")
            )
        if df[col].isna().any():
            df[col] = df.groupby("location", observed=False)[col].transform(
                lambda s: s.fillna(s.median())
            )
        log[f"{col}_imputed_same_hour_median"] = int(na_0 - na_1)
        log[f"{col}_imputed_fallback_day_or_station"] = int(na_1)

    log["rows_after_cleaning"] = int(len(df))
    log["total_rows_removed"] = int(init_rows - len(df))
    logger.info(
        f"Hoàn thành làm sạch weather: {len(df):,} dòng "
        "(rain sentinels giữ NaN: "
        f"{int((df['is_rain_sentinel'] == 1).sum()):,})."
    )
    return (df, log) if return_log else df


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_cab = clean_cab_rides()
    clean_wea = clean_weather()

    clean_cab.to_parquet(
        PROCESSED_DIR / "cab_rides_cleaned.parquet", index=False
    )
    clean_wea.to_csv(PROCESSED_DIR / "weather_cleaned.csv", index=False)
    logger.info(
        "Đã xuất thành công cab_rides_cleaned.parquet và weather_cleaned.csv."
    )
