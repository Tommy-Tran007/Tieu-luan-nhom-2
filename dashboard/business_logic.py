"""
Module: dashboard/business_logic.py
Mục đích: Tầng Logic Nghiệp vụ & Tính toán Chỉ số (Business Logic Layer) cho
Dashboard PRJ-EXT-01.
- Tập trung toàn bộ công thức tính KPI, bộ lọc động an toàn (xử lý trường hợp 0
  dòng không crash),
  tổng hợp dữ liệu cho 5 Tab (Executive, RQ1, RQ2, RQ3, RQ4).
- Cưỡng chế tuyệt đối các quy tắc kiểm toán:
  * CRIT-01: Mọi phép tính thời tiết tự động khóa weather_matched == 1.
  * CRIT-02: Mọi phép tính theo giờ/thứ/buổi tự động khóa is_timestamp_original
    == 1 (Boston EST).
  * HIGH-02: Tính điểm cắt biểu giá chỉ công nhận trong miền thực nghiệm [0.02,
    7.86] dặm.
  * HIGH-03: Mọi chỉ số mô phỏng Surge Cap gắn nhãn tường minh API Quote Proxy
    & SCENARIO.
- Tách bạch hoàn toàn khỏi Streamlit UI; dùng chung cho cả Streamlit App và
  trình xuất HTML độc lập.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dashboard.config import (
    ALL_TIERS_ORDERED,
    DAY_NAME_VI,
    EMPIRICAL_DISTANCE_MAX,
    EMPIRICAL_DISTANCE_MIN,
    SIMULATION_COLUMN_LABELS_VI,
    SYMMETRIC_PAIRS,
    TIME_OF_DAY_ORDER,
    TIME_OF_DAY_VI,
    ULTRA_SHORT_GPS_THRESHOLD,
)


# =============================================================================
# 1. BỘ LỌC TOÀN CỤC AN TOÀN (GLOBAL FILTER ENGINE)
# =============================================================================
@dataclass(frozen=True)
class FilterParams:
    """
    Tham số bộ lọc toàn cục từ Sidebar (hoặc mặc định toàn mẫu cho HTML tĩnh).
    """

    cab_types: tuple[str, ...] = ("Uber", "Lyft")
    service_tiers: tuple[str, ...] = tuple(
        tier for _, tier in ALL_TIERS_ORDERED
    )
    sources: tuple[str, ...] | None = None  # None = toàn bộ 12 quận Boston
    distance_min: float = EMPIRICAL_DISTANCE_MIN
    distance_max: float = EMPIRICAL_DISTANCE_MAX
    surge_min: float = 1.0
    surge_max: float = 3.0
    surge_only: bool = False


@dataclass
class FilterResult:
    """
    Kết quả lọc kèm chẩn đoán trường hợp rỗng (0 dòng) để giao diện xử lý thân
    thiện.
    """

    df: pd.DataFrame
    total_before: int
    total_after: int
    share_pct: float
    is_empty: bool
    empty_reason: str | None = None


def _observed_surge_rate_pct(s: pd.Series) -> float:
    """
    Tính tỷ lệ kích hoạt Surge (>1.0x) trên riêng mẫu có surge quan sát thực
    (bỏ NaN).
    """
    valid = s.dropna()
    if valid.empty:
        return 0.0
    return round(float((valid > 1.0).mean() * 100.0), 2)


def apply_global_filters(
    df: pd.DataFrame, params: FilterParams
) -> FilterResult:
    """
    Áp dụng bộ lọc toàn cục trên bản sao mặt nạ boolean (không thay đổi df
    gốc).
    - Giữ nguyên 38,037 dòng Lyft khuyết surge (NaN) khi bộ lọc surge ở trạng
      thái toàn miền [1.0, 3.0] & không bật surge_only.
    - Phát hiện nguyên nhân kinh tế khi kết quả lọc rỗng (ví dụ: chọn Uber +
      Surge > 1.0).
    """
    n_total = len(df)
    mask = pd.Series(True, index=df.index)

    if params.cab_types is not None:
        mask &= df["cab_type"].astype(str).isin(params.cab_types)
    if params.service_tiers is not None:
        mask &= df["name"].astype(str).isin(params.service_tiers)
    if params.sources is not None:
        mask &= df["source"].astype(str).isin(params.sources)

    in_dist_range = (df["distance"] >= float(params.distance_min) - 1e-6) & (
        df["distance"] <= float(params.distance_max) + 1e-6
    )
    include_nan_dist = (float(params.distance_min) <= 0.02 + 1e-6) and (
        float(params.distance_max) >= 7.86 - 1e-6
    )
    mask &= (
        (in_dist_range | df["distance"].isna())
        if include_nan_dist
        else in_dist_range
    )
    in_surge_range = (
        df["surge_multiplier"] >= float(params.surge_min) - 1e-6
    ) & (df["surge_multiplier"] <= float(params.surge_max) + 1e-6)
    include_nan_surge = (
        (not params.surge_only)
        and (float(params.surge_min) <= 1.0 + 1e-6)
        and (float(params.surge_max) >= 3.0 - 1e-6)
    )
    mask &= (
        (in_surge_range | df["surge_multiplier"].isna())
        if include_nan_surge
        else in_surge_range
    )
    if params.surge_only:
        mask &= df["surge_multiplier"].notna() & (df["surge_multiplier"] > 1.0)

    filtered = df.loc[mask]
    n_after = len(filtered)
    is_empty = n_after == 0
    empty_reason: str | None = None

    if is_empty:
        selected_cabs = set(params.cab_types or ())
        if selected_cabs == {"Uber"} and (
            params.surge_only or params.surge_min > 1.0
        ):
            empty_reason = (
                "Không có bản ghi nào khớp bộ lọc: trong bộ dữ liệu Boston, "
                "nền tảng Uber luôn "
                "ghi nhận surge_multiplier = 1.0 (0/140,477 lượt có surge > "
                "1.0). Vui lòng chọn thêm Lyft hoặc tắt lọc Surge > 1.0."
            )
        elif not params.cab_types or not params.service_tiers:
            empty_reason = (
                "Bạn đang bỏ chọn toàn bộ hãng xe hoặc gói dịch vụ. Hãy "
                "chọn ít nhất 1 mục để hiển thị dữ liệu."
            )
        else:
            empty_reason = (
                "Tổ hợp điều kiện lọc hiện tại trả về 0 quan sát. "
                "Vui lòng mở rộng phạm vi khoảng cách, hệ số surge hoặc "
                "quận đón."
            )

    return FilterResult(
        df=filtered,
        total_before=n_total,
        total_after=n_after,
        share_pct=round(n_after / max(n_total, 1) * 100.0, 2),
        is_empty=is_empty,
        empty_reason=empty_reason,
    )


# =============================================================================
# 2. TAB 1 — EXECUTIVE OVERVIEW & DATA QUALITY KPIs
# =============================================================================
def compute_executive_kpis(
    df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    full_dataset_rows: int = 270_898,
) -> dict[str, Any]:
    """
    Tính động toàn bộ KPI tổng quan cho Tab 1 từ dữ liệu lọc hiện hành và bảng
    test_metrics.
    An toàn tuyệt đối với DataFrame rỗng (trả về None cho đại lượng không xác
    định).
    """
    n_obs = len(df)
    m7_row = test_metrics_df.loc[
        test_metrics_df["model_code"] == "M7_HistGradientBoosting"
    ].iloc[0]
    m2_row = test_metrics_df.loc[
        test_metrics_df["model_code"] == "M2_Interaction_OLS"
    ].iloc[0]

    base_model_kpis = {
        "m7_test_r2": round(float(m7_row["test_r2"]), 4),
        "m7_test_rmse_usd": round(float(m7_row["test_rmse"]), 4),
        "m7_test_mae_usd": round(float(m7_row["test_mae"]), 4),
        "m7_test_mape_pct": round(float(m7_row["test_mape_pct"]), 2),
        "m2_test_r2": round(float(m2_row["test_r2"]), 4),
        "m2_test_rmse_usd": round(float(m2_row["test_rmse"]), 4),
        "m2_test_mae_usd": round(float(m2_row["test_mae"]), 4),
        "m2_test_mape_pct": round(float(m2_row["test_mape_pct"]), 2),
    }

    if n_obs == 0:
        return {
            "is_empty": True,
            "total_obs": 0,
            "share_of_full_pct": 0.0,
            "uber_count": 0,
            "uber_share_pct": 0.0,
            "lyft_count": 0,
            "lyft_share_pct": 0.0,
            "mean_price_usd": None,
            "median_price_usd": None,
            "iqr_price_usd": None,
            "uber_mean_price_usd": None,
            "uber_median_price_usd": None,
            "lyft_mean_price_usd": None,
            "lyft_median_price_usd": None,
            "price_gap_lyft_minus_uber_usd": None,
            "lyft_surge_count": 0,
            "lyft_obs_surge_count": 0,
            "lyft_surge_missing_count": 0,
            "lyft_surge_rate_pct": None,
            "uber_surge_count": 0,
            "uber_surge_rate_pct": None,
            "mean_distance_miles": None,
            "median_distance_miles": None,
            **base_model_kpis,
        }

    uber_df = df.loc[df["cab_type"].astype(str) == "Uber"]
    lyft_df = df.loc[df["cab_type"].astype(str) == "Lyft"]
    n_uber = len(uber_df)
    n_lyft = len(lyft_df)

    price_s = df["price"].astype(np.float64)
    q1_p, med_p, q3_p = (
        float(price_s.quantile(0.25)),
        float(price_s.median()),
        float(price_s.quantile(0.75)),
    )

    uber_mean = (
        float(uber_df["price"].astype(np.float64).mean())
        if n_uber > 0
        else None
    )
    uber_med = (
        float(uber_df["price"].astype(np.float64).median())
        if n_uber > 0
        else None
    )
    lyft_mean = (
        float(lyft_df["price"].astype(np.float64).mean())
        if n_lyft > 0
        else None
    )
    lyft_med = (
        float(lyft_df["price"].astype(np.float64).median())
        if n_lyft > 0
        else None
    )
    gap_lyft_uber = (
        (lyft_mean - uber_mean)
        if (lyft_mean is not None and uber_mean is not None)
        else None
    )

    lyft_valid_surge = (
        lyft_df["surge_multiplier"].dropna()
        if n_lyft > 0
        else pd.Series(dtype="float64")
    )
    n_lyft_obs = int(len(lyft_valid_surge))
    lyft_surge_cnt = (
        int((lyft_valid_surge > 1.0).sum()) if n_lyft_obs > 0 else 0
    )
    uber_surge_cnt = (
        int((uber_df["surge_multiplier"].dropna() > 1.0).sum())
        if n_uber > 0
        else 0
    )

    return {
        "is_empty": False,
        "total_obs": int(n_obs),
        "share_of_full_pct": round(
            n_obs / max(full_dataset_rows, 1) * 100.0, 2
        ),
        "uber_count": int(n_uber),
        "uber_share_pct": round(n_uber / n_obs * 100.0, 2),
        "lyft_count": int(n_lyft),
        "lyft_share_pct": round(n_lyft / n_obs * 100.0, 2),
        "mean_price_usd": round(float(price_s.mean()), 2),
        "median_price_usd": round(med_p, 2),
        "iqr_price_usd": round(q3_p - q1_p, 2),
        "uber_mean_price_usd": round(uber_mean, 2)
        if uber_mean is not None
        else None,
        "uber_median_price_usd": round(uber_med, 2)
        if uber_med is not None
        else None,
        "lyft_mean_price_usd": round(lyft_mean, 2)
        if lyft_mean is not None
        else None,
        "lyft_median_price_usd": round(lyft_med, 2)
        if lyft_med is not None
        else None,
        "price_gap_lyft_minus_uber_usd": round(gap_lyft_uber, 2)
        if gap_lyft_uber is not None
        else None,
        "lyft_surge_count": lyft_surge_cnt,
        "lyft_obs_surge_count": n_lyft_obs,
        "lyft_surge_missing_count": int(n_lyft - n_lyft_obs),
        "lyft_surge_rate_pct": round(lyft_surge_cnt / n_lyft_obs * 100.0, 2)
        if n_lyft_obs > 0
        else None,
        "uber_surge_count": uber_surge_cnt,
        "uber_surge_rate_pct": round(uber_surge_cnt / n_uber * 100.0, 2)
        if n_uber > 0
        else None,
        "mean_distance_miles": round(
            float(df["distance"].astype(np.float64).mean()), 2
        ),
        "median_distance_miles": round(
            float(df["distance"].astype(np.float64).median()), 2
        ),
        **base_model_kpis,
    }


def compute_tier_boxplot_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính chính xác thống kê phân vị (q1, median, q3, lowerfence, upperfence,
    mean, count)
    trên 100% quan sát của từng gói dịch vụ để vẽ Boxplot Plotly siêu nhẹ mà
    không mất độ chính xác.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "cab_type",
                "service_tier",
                "count",
                "mean_usd",
                "q1_usd",
                "median_usd",
                "q3_usd",
                "lowerfence_usd",
                "upperfence_usd",
                "min_usd",
                "max_usd",
            ]
        )

    rows: list[dict[str, Any]] = []
    for (cab, tier), grp in df.groupby(["cab_type", "name"], observed=True):
        s = grp["price"].astype(np.float64)
        q1 = float(s.quantile(0.25))
        med = float(s.median())
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        vmin = float(s.min())
        vmax = float(s.max())
        lf = max(vmin, q1 - 1.5 * iqr)
        uf = min(vmax, q3 + 1.5 * iqr)
        rows.append(
            {
                "cab_type": str(cab),
                "service_tier": str(tier),
                "count": int(len(s)),
                "mean_usd": round(float(s.mean()), 2),
                "q1_usd": round(q1, 2),
                "median_usd": round(med, 2),
                "q3_usd": round(q3, 2),
                "lowerfence_usd": round(lf, 2),
                "upperfence_usd": round(uf, 2),
                "min_usd": round(vmin, 2),
                "max_usd": round(vmax, 2),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["median_usd", "mean_usd"], ascending=[True, True])
        .reset_index(drop=True)
    )


def compute_price_histogram_bins(
    df: pd.DataFrame, bin_width: float = 2.5, max_price: float = 65.0
) -> pd.DataFrame:
    """
    Tính phân phối tần suất giá cước theo hãng xe bằng các cột đều nhau (USD).
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "cab_type",
                "bin_center_usd",
                "bin_label",
                "count",
                "share_pct",
            ]
        )

    edges = np.arange(2.5, max_price + bin_width, bin_width)
    rows: list[dict[str, Any]] = []
    for cab, grp in df.groupby("cab_type", observed=True):
        prices = (
            grp["price"].astype(np.float64).clip(upper=max_price - 1e-6).values
        )
        counts, _ = np.histogram(prices, bins=edges)
        total = max(int(counts.sum()), 1)
        for i, cnt in enumerate(counts):
            lo, hi = float(edges[i]), float(edges[i + 1])
            rows.append(
                {
                    "cab_type": str(cab),
                    "bin_center_usd": round((lo + hi) / 2.0, 2),
                    "bin_label": f"${lo:.1f}–${hi:.1f}",
                    "count": int(cnt),
                    "share_pct": round(int(cnt) / total * 100.0, 2),
                }
            )
    return pd.DataFrame(rows)


def compute_route_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tổng hợp 72 tuyến liên quận thực tế tại Boston (số lượt, giá TB/trung vị,
    cự ly trung vị, tỷ lệ Surge).
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "destination",
                "trips",
                "mean_price_usd",
                "median_price_usd",
                "median_distance_miles",
                "min_distance_miles",
                "max_distance_miles",
                "surge_rate_pct",
            ]
        )

    routes = (
        df.groupby(["source", "destination"], observed=True)
        .agg(
            trips=("price", "size"),
            mean_price_usd=(
                "price",
                lambda s: round(float(s.astype(np.float64).mean()), 2),
            ),
            median_price_usd=(
                "price",
                lambda s: round(float(s.astype(np.float64).median()), 2),
            ),
            median_distance_miles=(
                "distance",
                lambda s: round(float(s.astype(np.float64).median()), 2),
            ),
            min_distance_miles=(
                "distance",
                lambda s: round(float(s.astype(np.float64).min()), 2),
            ),
            max_distance_miles=(
                "distance",
                lambda s: round(float(s.astype(np.float64).max()), 2),
            ),
            surge_rate_pct=("surge_multiplier", _observed_surge_rate_pct),
        )
        .reset_index()
    )
    routes["source"] = routes["source"].astype(str)
    routes["destination"] = routes["destination"].astype(str)
    return routes


def compute_data_quality_summary(
    master_df: pd.DataFrame,
    stage3_summary: dict[str, Any],
    stage4_summary: dict[str, Any],
    ram_mb: float,
    est_raw_ram_mb: float,
) -> dict[str, Any]:
    """Tổng hợp hồ sơ Chất lượng Dữ liệu & Kiểm toán minh bạch cho Tab 1."""
    n_rows = len(master_df)
    n_ts1 = int((master_df["is_timestamp_original"] == 1).sum())
    n_ts0 = int((master_df["is_timestamp_original"] == 0).sum())
    n_wm1 = int((master_df["weather_matched"] == 1).sum())
    n_wm0 = int((master_df["weather_matched"] == 0).sum())
    n_rain_sent = (
        int((master_df["is_rain_sentinel"] == 1).sum())
        if "is_rain_sentinel" in master_df.columns
        else 0
    )
    n_valid_rain = int(
        ((master_df["weather_matched"] == 1) & master_df["rain"].notna()).sum()
    )
    n_surge_orig = (
        int((master_df["is_surge_original"] == 1).sum())
        if "is_surge_original" in master_df.columns
        else int(master_df["surge_multiplier"].notna().sum())
    )
    n_lyft_surge_miss = int(
        (
            (master_df["cab_type"].astype(str) == "Lyft")
            & master_df["surge_multiplier"].isna()
        ).sum()
    )
    n_ultra_short = int(
        (master_df["distance"] <= ULTRA_SHORT_GPS_THRESHOLD).sum()
    )
    false_rain = int(
        master_df.loc[master_df["weather_matched"] == 0, "is_raining"]
        .notna()
        .sum()
    )

    raw_cab_rows = int(
        stage3_summary.get("initial_audit", {})
        .get("cab_rides", {})
        .get("shape", {})
        .get("rows", 693_071)
    )
    r3_pct = float(
        stage4_summary.get("r3_scale_recovery_audit", {}).get(
            "strictly_unmodified_both_pct", 88.85
        )
    )

    return {
        "raw_cab_rows": raw_cab_rows,
        "clean_master_rows": n_rows,
        "clean_master_cols": int(master_df.shape[1]),
        "ram_mb": round(ram_mb, 2),
        "est_unoptimized_ram_mb": round(est_raw_ram_mb, 2),
        "ram_saved_pct": round(
            (1.0 - ram_mb / max(est_raw_ram_mb, 1e-6)) * 100.0, 1
        ),
        "ts_original_rows": n_ts1,
        "ts_original_pct": round(n_ts1 / max(n_rows, 1) * 100.0, 2),
        "ts_missing_rows": n_ts0,
        "ts_missing_pct": round(n_ts0 / max(n_rows, 1) * 100.0, 2),
        "weather_matched_rows": n_wm1,
        "weather_matched_all_pct": round(n_wm1 / max(n_rows, 1) * 100.0, 2),
        "weather_matched_valid_ts_pct": round(
            n_wm1 / max(n_ts1, 1) * 100.0, 2
        ),
        "weather_unmatched_rows": n_wm0,
        "rain_sentinel_rows": n_rain_sent,
        "valid_rain_obs_rows": n_valid_rain,
        "surge_original_rows": n_surge_orig,
        "lyft_surge_missing_rows": n_lyft_surge_miss,
        "false_rain_labels_count": false_rain,
        "ultra_short_gps_count": n_ultra_short,
        "r3_unmodified_raw_pct": round(r3_pct, 2),
        "duplicate_rows": int(master_df.duplicated().sum()),
    }


# =============================================================================
# 3. TAB 2 — RQ1: THỜI TIẾT (CRIT-01), THỜI GIAN BOSTON EST (CRIT-02) & KIỂM
# ĐỊNH
# =============================================================================
def compute_weather_rain_comparison(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    So sánh giá báo API và tỷ lệ Surge khi Mưa vs Không mưa.
    Khóa weather_matched == 1 và loại bỏ ô lỗi cảm biến mưa (rain.notna(), tách
    riêng is_rain_sentinel == 1).
    """
    wm = df.loc[df["weather_matched"] == 1]
    n_matched = len(wm)
    valid_rain = wm.loc[wm["rain"].notna()]
    n_valid = len(valid_rain)
    n_sent = int(n_matched - n_valid)
    if n_valid == 0:
        empty = pd.DataFrame(
            columns=[
                "scope",
                "weather_state",
                "n_obs",
                "mean_price_usd",
                "median_price_usd",
                "surge_rate_pct",
            ]
        )
        return empty, {
            "n_matched": int(n_matched),
            "n_valid_rain_obs": 0,
            "n_rain_sentinel": n_sent,
            "n_rain": 0,
            "n_dry": 0,
            "all_rain_minus_dry_usd": None,
        }

    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("Toàn thị trường", valid_rain)]
    for cab in ("Uber", "Lyft"):
        sub = valid_rain.loc[valid_rain["cab_type"].astype(str) == cab]
        if not sub.empty:
            scopes.append((cab, sub))

    for scope_label, sub_df in scopes:
        for flag_val, state_label in (
            (0.0, "Không mưa (Dry)"),
            (1.0, "Có mưa (Rain)"),
        ):
            grp = sub_df.loc[sub_df["is_raining"] == flag_val]
            if grp.empty:
                continue
            p = grp["price"].astype(np.float64)
            rows.append(
                {
                    "scope": scope_label,
                    "weather_state": state_label,
                    "n_obs": int(len(grp)),
                    "mean_price_usd": round(float(p.mean()), 4),
                    "median_price_usd": round(float(p.median()), 2),
                    "surge_rate_pct": _observed_surge_rate_pct(
                        grp["surge_multiplier"]
                    ),
                }
            )

    res_df = pd.DataFrame(rows)
    rain_all = valid_rain.loc[valid_rain["is_raining"] == 1.0, "price"].astype(
        np.float64
    )
    dry_all = valid_rain.loc[valid_rain["is_raining"] == 0.0, "price"].astype(
        np.float64
    )
    diff_usd = (
        round(float(rain_all.mean() - dry_all.mean()), 4)
        if (len(rain_all) > 0 and len(dry_all) > 0)
        else None
    )

    meta = {
        "n_matched": int(n_matched),
        "n_valid_rain_obs": int(n_valid),
        "n_rain_sentinel": n_sent,
        "n_rain": int(len(rain_all)),
        "n_dry": int(len(dry_all)),
        "rain_share_pct": round(len(rain_all) / n_valid * 100.0, 2),
        "all_rain_mean_usd": round(float(rain_all.mean()), 4)
        if len(rain_all) > 0
        else None,
        "all_dry_mean_usd": round(float(dry_all.mean()), 4)
        if len(dry_all) > 0
        else None,
        "all_rain_minus_dry_usd": diff_usd,
    }
    return res_df, meta


def compute_hourly_est_profile(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Tính hồ sơ 24 giờ Boston EST (0h–23h) theo hãng xe.
    CƯỠNG CHẾ CRIT-02: Bắt buộc lọc is_timestamp_original == 1 trước mọi phép
    tính.
    """
    ts = df.loc[df["is_timestamp_original"] == 1]
    n_valid_ts = len(ts)
    if n_valid_ts == 0:
        empty = pd.DataFrame(
            columns=[
                "hour",
                "cab_type",
                "trips",
                "mean_price_usd",
                "median_price_usd",
                "surge_rate_pct",
            ]
        )
        return empty, {
            "n_valid_ts": 0,
            "peak_hour_est": None,
            "trough_hour_est": None,
        }

    hourly = (
        ts.groupby(["hour", "cab_type"], observed=True)
        .agg(
            trips=("price", "size"),
            mean_price_usd=(
                "price",
                lambda s: round(float(s.astype(np.float64).mean()), 2),
            ),
            median_price_usd=(
                "price",
                lambda s: round(float(s.astype(np.float64).median()), 2),
            ),
            surge_rate_pct=("surge_multiplier", _observed_surge_rate_pct),
        )
        .reset_index()
    )
    hourly["hour"] = hourly["hour"].astype(int)
    hourly["cab_type"] = hourly["cab_type"].astype(str)
    hourly = hourly.sort_values(["hour", "cab_type"]).reset_index(drop=True)

    vol_by_hour = ts.groupby("hour").size()
    meta = {
        "n_valid_ts": int(n_valid_ts),
        "peak_hour_est": int(vol_by_hour.idxmax()),
        "peak_hour_trips": int(vol_by_hour.max()),
        "trough_hour_est": int(vol_by_hour.idxmin()),
        "trough_hour_trips": int(vol_by_hour.min()),
    }
    return hourly, meta


def compute_dayofweek_est_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính giá cước, lưu lượng và tỷ lệ surge theo 7 thứ trong tuần (Boston EST).
    CƯỠNG CHẾ CRIT-02: Bắt buộc lọc is_timestamp_original == 1.
    """
    ts = df.loc[df["is_timestamp_original"] == 1]
    if ts.empty:
        return pd.DataFrame(
            columns=[
                "day_of_week",
                "day_label_vi",
                "cab_type",
                "trips",
                "mean_price_usd",
                "median_price_usd",
                "surge_rate_pct",
            ]
        )

    dow = (
        ts.groupby(["day_of_week", "cab_type"], observed=True)
        .agg(
            trips=("price", "size"),
            mean_price_usd=(
                "price",
                lambda s: round(float(s.astype(np.float64).mean()), 2),
            ),
            median_price_usd=(
                "price",
                lambda s: round(float(s.astype(np.float64).median()), 2),
            ),
            surge_rate_pct=("surge_multiplier", _observed_surge_rate_pct),
        )
        .reset_index()
    )
    dow["day_of_week"] = dow["day_of_week"].astype(int)
    dow["day_label_vi"] = dow["day_of_week"].map(DAY_NAME_VI)
    dow["cab_type"] = dow["cab_type"].astype(str)
    return dow.sort_values(["day_of_week", "cab_type"]).reset_index(drop=True)


def compute_time_of_day_est_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính thống kê theo 4 buổi Boston EST (Morning/Afternoon/Evening/Night).
    CƯỠNG CHẾ CRIT-02: Bắt buộc lọc is_timestamp_original == 1 (loại
    'Unknown').
    """
    ts = df.loc[
        (df["is_timestamp_original"] == 1)
        & (df["time_of_day"].astype(str).isin(TIME_OF_DAY_ORDER))
    ]
    if ts.empty:
        return pd.DataFrame(
            columns=[
                "time_of_day",
                "time_label_vi",
                "cab_type",
                "trips",
                "mean_price_usd",
                "surge_rate_pct",
            ]
        )

    tod = (
        ts.groupby(["time_of_day", "cab_type"], observed=True)
        .agg(
            trips=("price", "size"),
            mean_price_usd=(
                "price",
                lambda s: round(float(s.astype(np.float64).mean()), 2),
            ),
            surge_rate_pct=("surge_multiplier", _observed_surge_rate_pct),
        )
        .reset_index()
    )
    tod["time_of_day"] = pd.Categorical(
        tod["time_of_day"].astype(str),
        categories=list(TIME_OF_DAY_ORDER),
        ordered=True,
    )
    tod["time_label_vi"] = tod["time_of_day"].astype(str).map(TIME_OF_DAY_VI)
    tod["cab_type"] = tod["cab_type"].astype(str)
    return tod.sort_values(["time_of_day", "cab_type"]).reset_index(drop=True)


def compute_surge_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thống kê phân bố 7 bậc surge_multiplier trên riêng mẫu Lyft quan sát surge
    thực (không gộp NaN).
    """
    lyft = df.loc[
        (df["cab_type"].astype(str) == "Lyft") & df["surge_multiplier"].notna()
    ]
    if lyft.empty:
        return pd.DataFrame(
            columns=[
                "surge_multiplier",
                "trips",
                "share_of_lyft_pct",
                "share_of_surge_pct",
                "mean_price_usd",
                "median_price_usd",
            ]
        )

    n_lyft = len(lyft)
    n_surge = int((lyft["surge_multiplier"] > 1.0).sum())
    dist = (
        lyft.groupby("surge_multiplier", observed=True)
        .agg(
            trips=("price", "size"),
            mean_price_usd=(
                "price",
                lambda s: round(float(s.astype(np.float64).mean()), 2),
            ),
            median_price_usd=(
                "price",
                lambda s: round(float(s.astype(np.float64).median()), 2),
            ),
        )
        .reset_index()
        .sort_values("surge_multiplier")
        .reset_index(drop=True)
    )
    dist["surge_multiplier"] = dist["surge_multiplier"].astype(float).round(2)
    dist["share_of_lyft_pct"] = (dist["trips"] / n_lyft * 100.0).round(3)
    dist["share_of_surge_pct"] = np.where(
        dist["surge_multiplier"] > 1.0,
        (dist["trips"] / max(n_surge, 1) * 100.0).round(2),
        0.0,
    )
    return dist


def prepare_hypothesis_table(
    hyp_df: pd.DataFrame, rq_filter: str | None = None
) -> pd.DataFrame:
    """
    Chuẩn hóa bảng kiểm định giả thuyết H1–H4 giữ đủ p-value kép, Effect Size
    và ý nghĩa thực tiễn.
    """
    out = hyp_df.copy()
    if rq_filter and rq_filter != "Tất cả":
        out = out.loc[out["research_question"].astype(str) == rq_filter].copy()
    return out.reset_index(drop=True)


# =============================================================================
# 4. TAB 3 — RQ2: CẠNH TRANH BIỂU GIÁ & ĐIỂM CẮT NỘI MIỀN (HIGH-02)
# =============================================================================
def compute_crossing_point_for_pair(
    tariffs_df: pd.DataFrame,
    pair_spec: dict[str, str],
    df_obs: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Tính điểm cắt biểu giá d* = (b0_Uber - b0_Lyft) / (b1_Lyft - b1_Uber) từ
    tham số OLS HC3.
    CƯỠNG CHẾ HIGH-02: Chỉ công nhận điểm cắt nếu nằm trong miền quan sát thực
    [0.02, 7.86] dặm;
    tuyệt đối KHÔNG ngoại suy ngoài miền (ví dụ loại bỏ 37.61 dặm của UberPool
    vs Shared).
    """
    u_row = tariffs_df.loc[
        tariffs_df["service_tier"] == pair_spec["uber_tier"]
    ].iloc[0]
    l_row = tariffs_df.loc[
        tariffs_df["service_tier"] == pair_spec["lyft_tier"]
    ].iloc[0]

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
    raw_cross = (base_diff / slope_diff) if abs(slope_diff) > 1e-6 else np.nan
    in_domain = bool(
        np.isfinite(raw_cross)
        and (EMPIRICAL_DISTANCE_MIN <= raw_cross <= EMPIRICAL_DISTANCE_MAX)
    )

    crossing_dist = round(float(raw_cross), 2) if in_domain else None
    crossing_price = (
        round(float(u_b0 + u_b1 * raw_cross), 2) if in_domain else None
    )

    if in_domain:
        status_vi = (
            f"Cắt trong miền Boston tại d* = {crossing_dist:.2f} dặm "
            f"(~${crossing_price:.2f})"
        )
    else:
        cheaper_all = (
            pair_spec["lyft_tier"]
            if (l_b0 + l_b1 * 2.0) < (u_b0 + u_b1 * 2.0)
            else pair_spec["uber_tier"]
        )
        status_vi = (
            f"Không cắt trong miền quan sát [{EMPIRICAL_DISTANCE_MIN:.2f}, "
            f"{EMPIRICAL_DISTANCE_MAX:.2f}] dặm "
            f"({cheaper_all} luôn rẻ hơn — không ngoại suy ngoài miền)"
        )

    pct_shorter_than_dstar: float | None = None
    if in_domain and df_obs is not None and not df_obs.empty:
        pair_obs = df_obs.loc[
            df_obs["name"]
            .astype(str)
            .isin([pair_spec["uber_tier"], pair_spec["lyft_tier"]])
        ]
        pair_dist = pair_obs["distance"].dropna()
        if not pair_dist.empty:
            pct_shorter_than_dstar = round(
                float((pair_dist < raw_cross).mean() * 100.0), 1
            )

    return {
        "pair_id": pair_spec["pair_id"],
        "segment_label": pair_spec["segment_label"],
        "short_label": pair_spec["short_label"],
        "uber_tier": pair_spec["uber_tier"],
        "lyft_tier": pair_spec["lyft_tier"],
        "uber_base_b0": round(u_b0, 2),
        "lyft_base_b0": round(l_b0, 2),
        "base_diff_uber_minus_lyft": round(base_diff, 2),
        "uber_slope_b1": round(u_b1, 2),
        "lyft_slope_b1": round(l_b1, 2),
        "slope_diff_lyft_minus_uber": round(slope_diff, 2),
        "crossing_distance_miles": crossing_dist,
        "crossing_price_usd": crossing_price,
        "crossing_in_domain": in_domain,
        "crossing_status_vi": status_vi,
        "pct_trips_shorter_than_dstar": pct_shorter_than_dstar,
        "cheaper_at_1_5_mi": pair_spec["lyft_tier"]
        if (l_b0 + 1.5 * l_b1) < (u_b0 + 1.5 * u_b1)
        else pair_spec["uber_tier"],
        "cheaper_at_4_5_mi": pair_spec["lyft_tier"]
        if (l_b0 + 4.5 * l_b1) < (u_b0 + 4.5 * u_b1)
        else pair_spec["uber_tier"],
    }


def compute_all_pair_comparisons(
    tariffs_df: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Trả về bảng đối sánh đầy đủ cho đúng 5 cặp cạnh tranh đối xứng Uber vs.
    Lyft.
    """
    return pd.DataFrame(
        [
            compute_crossing_point_for_pair(tariffs_df, p, df_obs)
            for p in SYMMETRIC_PAIRS
        ]
    )


def compute_binned_empirical_tariff(
    df: pd.DataFrame,
    uber_tier: str,
    lyft_tier: str,
    bin_width: float = 0.5,
) -> pd.DataFrame:
    """
    Tính giá trung bình thực tế theo từng khoảng cự ly (chỉ trên chuyến không
    surge = 1.0x)
    để đối chiếu trực tiếp với đường biểu giá tuyến tính OLS HC3.
    """
    sub = df.loc[
        df["name"].astype(str).isin([uber_tier, lyft_tier])
        & (df["surge_multiplier"] == 1.0)
    ].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "service_tier",
                "cab_type",
                "dist_bin_center",
                "mean_price_usd",
                "n_obs",
            ]
        )

    sub = sub.loc[sub["distance"].notna()].copy()
    sub["dist_bin_center"] = (
        np.floor(sub["distance"].astype(np.float64) / bin_width) + 0.5
    ) * bin_width
    binned = (
        sub.groupby(["cab_type", "name", "dist_bin_center"], observed=True)
        .agg(
            mean_price_usd=(
                "price",
                lambda s: round(float(s.astype(np.float64).mean()), 2),
            ),
            n_obs=("price", "size"),
        )
        .reset_index()
        .rename(columns={"name": "service_tier"})
    )
    binned = binned.loc[binned["n_obs"] >= 15].reset_index(drop=True)
    binned["cab_type"] = binned["cab_type"].astype(str)
    binned["service_tier"] = binned["service_tier"].astype(str)
    return binned


# =============================================================================
# 5. TAB 4 — RQ3: ML ARENA, PERMUTATION IMPORTANCE & PREDICTION INTERVALS
# =============================================================================
def prepare_model_leaderboard(test_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Xếp hạng 10 mô hình (M0a–M7) theo Test RMSE tăng dần kèm cờ đạt chuẩn
    SMART.
    """
    df = (
        test_metrics_df.copy()
        .sort_values("test_rmse", ascending=True)
        .reset_index(drop=True)
    )
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def prepare_permutation_family_summary(perm_df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp 14 nhóm Permutation Importance về 4 họ biến kinh tế."""
    fam = (
        perm_df.groupby("domain_family", observed=False)
        .agg(
            n_groups=("feature_group", "count"),
            m7_delta_rmse_sum_usd=(
                "m7_delta_rmse_mean_usd",
                lambda s: round(float(s.clip(lower=0.0).sum()), 4),
            ),
            m7_share_pct=(
                "m7_importance_share_pct",
                lambda s: round(float(s.sum()), 2),
            ),
            m2_delta_rmse_sum_usd=(
                "m2_delta_rmse_mean_usd",
                lambda s: round(float(s.clip(lower=0.0).sum()), 4),
            ),
        )
        .reset_index()
        .sort_values("m7_share_pct", ascending=False)
        .reset_index(drop=True)
    )
    return fam


def prepare_segment_errors(
    seg_df: pd.DataFrame, dimension: str
) -> pd.DataFrame:
    """
    Lọc bảng sai số phân khúc theo 1 trong 4 chiều (service_tier,
    distance_bucket, surge_regime, source_district).
    """
    sub = seg_df.loc[seg_df["dimension"] == dimension].copy()
    if dimension in ("service_tier", "source_district"):
        sub = sub.sort_values("m7_rmse_usd", ascending=True)
    return sub.reset_index(drop=True)


# =============================================================================
# 6. TAB 5 — RQ4: MÔ PHỎNG TRẦN SURGE CAP (CƯỠNG CHẾ HIGH-03) & THAM CHIẾU KHÍ
# TƯỢNG
# =============================================================================
def compute_weather_reference_medians(df: pd.DataFrame) -> dict[str, float]:
    """
    Trích xuất giá trị trung vị khí tượng làm tham chiếu đầu vào cho Live Fare
    Predictor.
    CƯỠNG CHẾ CRIT-01: Chỉ tính trên tập weather_matched == 1.
    """
    wm = df.loc[df["weather_matched"] == 1]
    if wm.empty:
        return {
            "temp": 39.47,
            "clouds": 0.6925,
            "pressure": 1008.535,
            "humidity": 0.75,
            "wind": 6.8297,
            "rain_wet_median": 0.011,
        }
    rain_wet = wm.loc[wm["is_raining"] == 1.0, "rain"].dropna()
    return {
        "temp": round(float(wm["temp"].median()), 4),
        "clouds": round(float(wm["clouds"].median()), 4),
        "pressure": round(float(wm["pressure"].median()), 4),
        "humidity": round(float(wm["humidity"].median()), 4),
        "wind": round(float(wm["wind"].median()), 4),
        "rain_wet_median": round(float(rain_wet.median()), 4)
        if not rain_wet.empty
        else 0.011,
    }


def prepare_surge_cap_display_table(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Đổi toàn bộ tên cột CSV mô phỏng sang tiếng Việt chuẩn thuật ngữ API Quote
    Proxy.
    Hiển thị cả 2 cột ràng buộc cung nhạy giá Q_realized = min(Q_D, Q_S)
    (ε_S = 0.30 và ε_S = 0.60) lẫn cận trên co giãn cầu.
    """
    selected_cols = [
        "surge_cap",
        "scenario_description",
        "affected_trips_count",
        "affected_surge_trips_pct",
        "mean_orig_price_affected_usd",
        "mean_capped_price_affected_usd",
        "mean_rider_saving_per_affected_trip_usd",
        "mean_rider_saving_pct_affected",
        "total_consumer_savings_usd",
        "static_lyft_rev_change_pct",
        "supply_constrained_rev_change_drop5_pct",
        "supply_constrained_rev_change_drop10_pct",
        "net_lyft_rev_change_eps_050_pct",
        "net_lyft_rev_change_eps_080_pct",
        "m2_ols_mean_saving_affected_usd",
        "m2_ols_total_savings_usd",
    ]
    display_df = (
        sim_df[selected_cols]
        .copy()
        .rename(columns=SIMULATION_COLUMN_LABELS_VI)
    )
    return display_df


def extract_surge_cap_scenario(
    sim_df: pd.DataFrame,
    surge_cap: float,
    elasticity_mode: str = "static_000",
    supply_drop_pct: int = 0,
) -> dict[str, Any]:
    """
    Trích xuất chỉ số mô phỏng cho một kịch bản trần Surge Cap dưới nguyên tắc
    bên ngắn
    Q_realized = min(Q_D, Q_S) (đảm bảo đơn điệu chặt: drop10 < drop5 < static
    <= upper_bound).
    - supply_drop_pct ∈ {0, 5, 10} ánh xạ sang độ nhạy cung ε_S ∈ {0.00, 0.30,
      0.60}. Khi > 0, Q_S = clip(1 + ε_S · Δp/p, sàn, 1.0) chặn cầu dư.
    - Khi supply_drop_pct == 0, elasticity_mode ∈ {'static_000', 'eps_050',
      'eps_080'}.
    """
    matches = sim_df.loc[
        np.isclose(
            sim_df["surge_cap"].astype(float), float(surge_cap), atol=1e-4
        )
    ]
    if matches.empty:
        raise ValueError(
            f"Không tìm thấy kịch bản trần surge_cap={surge_cap} trong "
            "stage6_surge_cap_simulation.csv."
        )
    row = matches.iloc[0]

    static_pct = float(row["static_lyft_rev_change_pct"])
    drop5_pct = float(row["supply_constrained_rev_change_drop5_pct"])
    drop10_pct = float(row["supply_constrained_rev_change_drop10_pct"])
    unful5 = float(row["unfulfilled_quotes_drop5"])
    unful10 = float(row["unfulfilled_quotes_drop10"])

    if int(supply_drop_pct) == 10:
        proxy_change_pct = drop10_pct
        retained_quotes = 0.0
        unfulfilled_quotes = unful10
        elasticity_label = (
            "Cung nhạy giá mạnh ε_S = 0.60 (Q_realized = min(Q_D, Q_S))"
        )
    elif int(supply_drop_pct) == 5:
        proxy_change_pct = drop5_pct
        retained_quotes = 0.0
        unfulfilled_quotes = unful5
        elasticity_label = (
            "Cung nhạy giá vừa ε_S = 0.30 (Q_realized = min(Q_D, Q_S))"
        )
    elif elasticity_mode == "static_000":
        proxy_change_pct = static_pct
        retained_quotes = 0.0
        unfulfilled_quotes = 0.0
        elasticity_label = "Cầu tĩnh (ε = 0.00 — Cung không đổi)"
    elif elasticity_mode == "eps_080":
        proxy_change_pct = float(row["net_lyft_rev_change_eps_080_pct"])
        retained_quotes = float(row["retained_trips_eps_080"])
        unfulfilled_quotes = 0.0
        elasticity_label = "Cận trên dư cung cạnh tranh (ε = -0.80)"
    else:
        proxy_change_pct = float(row["net_lyft_rev_change_eps_050_pct"])
        retained_quotes = float(row["retained_trips_eps_050"])
        unfulfilled_quotes = 0.0
        elasticity_label = "Cận trên dư cung Cohen et al., 2016 (ε = -0.50)"

    m7_sav = float(row["mean_rider_saving_per_affected_trip_usd"])
    m2_sav = float(row["m2_ols_mean_saving_affected_usd"])
    is_tree_saturation_zone = bool(
        np.isclose(float(surge_cap), 2.00, atol=1e-4)
    )
    return {
        "is_simulation": True,
        "scenario_code": str(row["scenario_code"]),
        "surge_cap": float(row["surge_cap"]),
        "scenario_description": str(row["scenario_description"]),
        "affected_quotes_count": int(row["affected_trips_count"]),
        "affected_lyft_quotes_pct": float(row["affected_lyft_trips_pct"]),
        "affected_surge_quotes_pct": float(row["affected_surge_trips_pct"]),
        "mean_orig_quote_usd": float(row["mean_orig_price_affected_usd"]),
        "mean_capped_quote_usd": float(row["mean_capped_price_affected_usd"]),
        "mean_saving_m7_usd": m7_sav,
        "mean_saving_m7_pct": float(row["mean_rider_saving_pct_affected"]),
        "total_savings_m7_usd": float(row["total_consumer_savings_usd"]),
        "mean_saving_m2_usd": m2_sav,
        "total_savings_m2_usd": float(row["m2_ols_total_savings_usd"]),
        "quote_proxy_change_pct": proxy_change_pct,
        "static_proxy_change_pct": static_pct,
        "supply_drop5_proxy_change_pct": drop5_pct,
        "supply_drop10_proxy_change_pct": drop10_pct,
        "supply_drop_pct": int(supply_drop_pct),
        "retained_quotes_est": retained_quotes,
        "unfulfilled_quotes_est": unfulfilled_quotes,
        "elasticity_label": elasticity_label,
        "tree_saturation_warning": (
            "Lưu ý bão hòa nhánh cây (Tree Saturation): tại trần 2.00x chỉ "
            "có 53 lượt surge 2.5x–3.0x, "
            "nên M7 ước lượng tiết kiệm thấp hơn đối chứng tuyến tính M2 "
            f"OLS (${m7_sav:.2f} vs ${m2_sav:.2f}/lượt)."
            if is_tree_saturation_zone
            else None
        ),
    }
